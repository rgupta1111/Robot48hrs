"""Find the object, CENTRE it, PLAN a straight approach, show the plan, EXECUTE it as one smooth motion.

    1. centre   small camera-guided steps until the object sits in the middle of the image (the camera response J is
                learned once by three small probe moves, so any camera mounting works)
    2. plan     measure the distance, then build a straight line along the camera axis to the stop distance:
                  fast until --slow-below-cm (5 cm), easing to a crawl, gripper opens at --open-at-cm (10 cm),
                  stop at --stop-distance-mm (10 mm = 1 cm).  Distances are gripper-to-object: camera estimate minus
                  --tip-offset-mm (camera-to-fingertip distance - set it, or distances are camera-to-object!)
                The line is solved in JOINT space from the arm's own Viam kinematics, each waypoint seeded from the
                last, so joint movement is minimal (checked against the live Viam pose; falls back to Cartesian moves).
                The plan is checked against the table / walls (machine frame system, or the built-in lab world) with
                --margin-mm clearance and --obstacle boxes, and printed in the chat / on the page before it runs.
    3. execute  one MoveThroughJointPositions trajectory per leg (fast, slow) at the planned speed; the gripper opens
                when the arm's real Viam pose passes the planned point. After the fast leg the object is re-measured
                and the plan is rebuilt if reality differs from the plan.
    --planner   uses Viam's collision-planning motion service for the centring moves too.

    scripts/run.sh src/servo_object.py --viam-address <machine>.viam.cloud --preset lab --ui --tip-offset-mm 130   # dry run
    ... --sink viam                                                                                              # real arm
Chat (--ui): "go to the cup", "get within 5 cm of the cup", "fingertips are 13 cm in front of the camera", "the cup is 9 cm
tall", "calibrate 30 cm", yes / stop / pause / resume / unlock / reset / open the gripper / "what do you see".
Keys in the video window: space start, p re-probe, c calibrate, t teach stop distance, x e-stop, l unlock, r re-acquire.
There is NO contact sensor and the distance comes from the object's size on screen: keep a hand on the e-stop.
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
import webbrowser

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_mimic import PRESETS, floats  # noqa: E402
from local_detect import (COLORS, Lock, Range, color_candidates, crop_gray, detect, draw_lock,  # noqa: E402
                          load_detector, object_candidates, roi_candidates, template_candidates)
from mimic.chat import HELP, parse  # noqa: E402
from mimic.kinematics import load_validated  # noqa: E402
from mimic.overlay import draw_plan  # noqa: E402
from mimic.pipeline import FrameGrabber, Perception  # noqa: E402
from mimic.plan import attach_joints, build_plan, describe, segment_joint_speed  # noqa: E402
from mimic.safety import Obstacles  # noqa: E402
from mimic.servo import FeatureFilter, depth_direction, estimate_jacobian, features, servo_step  # noqa: E402
from mimic.ui import UIServer  # noqa: E402
from reset_position import RESET_JOINTS  # noqa: E402
from tidy.robot import TableToWorld, ViamRobot  # noqa: E402

WIN = "servo object"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
JAC_FILE = os.path.abspath(os.path.join(ROOT, "data", "servo_jacobian.json"))
# only used by the dry run when no probe has been done: image right/down -> gripper -y / -x, a guess
GUESS_J = np.array([[0.0, -0.003, 0.0], [-0.003, 0.0, 0.0], [0.0, 0.0, 0.0008]])


class Abort(Exception):
    pass


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", choices=list(PRESETS))
    preset = pre.parse_known_args()[0].preset
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS))
    ap.add_argument("--viam-address", required=True)
    ap.add_argument("--camera", default="cam")
    ap.add_argument("--arm-name", default="arm")
    ap.add_argument("--gripper-name", default="gripper")
    ap.add_argument("--viam-insecure", action="store_true")
    ap.add_argument("--sink", choices=["sim", "viam"], default="sim", help="sim = dry run, nothing moves")
    ap.add_argument("--table-to-world", default="0,0,0,0")
    ap.add_argument("--lock", default="cup", metavar="NAME", help="object to go to (default cup)")
    ap.add_argument("--lock-color", choices=sorted(COLORS), help="optional: use a colour blob instead")
    ap.add_argument("--lock-min-area", type=float, default=400)
    ap.add_argument("--model", help="YOLO weights (default: yolov8s)")
    ap.add_argument("--prompts")
    ap.add_argument("--conf", type=float)
    ap.add_argument("--imgsz", type=int, default=960, help="detector input size")
    ap.add_argument("--workers", type=int, default=4, help="overlapping camera requests")
    # --- approach profile
    ap.add_argument("--stop-distance-mm", type=float, default=10.0, help="stop at this gripper-to-object distance (10 = 1 cm)")
    ap.add_argument("--slow-below-cm", type=float, default=5.0, help="fast until this far, then ease down")
    ap.add_argument("--open-at-cm", type=float, default=10.0, help="open the gripper at this distance")
    ap.add_argument("--fast-speed", type=float, default=150.0, help="tool speed while far, mm/s")
    ap.add_argument("--slow-speed", type=float, default=20.0, help="tool speed near the stop distance, mm/s")
    ap.add_argument("--tip-offset-mm", type=float, default=0.0,
                    help="distance from the camera to the fingertips along the viewing direction (0 = measure from the camera)")
    ap.add_argument("--object-height-mm", type=float, default=100.0, help="true height of the object (for distance)")
    ap.add_argument("--hfov", type=float, default=70.0, help="camera horizontal field of view in degrees, if unknown")
    ap.add_argument("--calib-cm", type=float, default=30.0, help="press c with the object this far from the camera")
    # --- centring / plan / execution
    ap.add_argument("--gain", type=float, default=0.7, help="fraction of the centring error corrected per step")
    ap.add_argument("--max-step", type=float, default=60.0, help="largest centring step, mm")
    ap.add_argument("--deadband", type=float, default=0.05, help="close enough to start moving in when within this fraction of the frame")
    ap.add_argument("--stable-frames", type=int, default=2, help="centred frames in a row before planning")
    ap.add_argument("--settle", type=float, default=0.35, help="seconds to wait after a move before trusting a frame")
    ap.add_argument("--centre-timeout", type=float, default=60.0)
    ap.add_argument("--plan-step-mm", type=float, default=10.0, help="waypoint spacing of the plan")
    ap.add_argument("--max-travel-mm", type=float, default=400.0)
    ap.add_argument("--max-joint-jump", type=float, default=15.0, help="largest joint change between waypoints, deg")
    ap.add_argument("--joint-accel", type=float, default=3.0, help="joint acceleration limit = this x the leg's joint speed, per s")
    ap.add_argument("--no-joint-space", action="store_true", help="skip the kinematics and use Cartesian moves")
    ap.add_argument("--confirm-plan", action="store_true", help="wait for 'yes' before executing the plan")
    ap.add_argument("--countdown", type=float, default=3.0, help="seconds between showing the plan and executing it")
    ap.add_argument("--replans", type=int, default=3)
    ap.add_argument("--probe-mm", type=float, default=20.0, help="size of the learning moves")
    ap.add_argument("--probe", action="store_true", help="re-learn the camera response even if a saved one exists")
    # --- obstacles
    ap.add_argument("--bounds", default="250,600,-300,300,0,450")
    ap.add_argument("--margin-mm", type=float, default=60.0, help="clearance kept from obstacles")
    ap.add_argument("--obstacle", action="append", default=[], metavar="XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX",
                    help="extra keep-out box in the robot frame (repeatable)")
    ap.add_argument("--no-world-obstacles", action="store_true", help="do not load the machine's table/walls")
    ap.add_argument("--planner", action="store_true", help="send the centring moves through Viam's collision-planning motion service")
    ap.add_argument("--ui", action="store_true", help="serve the chat page on http://127.0.0.1:PORT/ and open it")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="with --ui: do not open the page automatically")
    ap.add_argument("--auto-engage", action="store_true", help="start the mission for --lock right away")
    ap.add_argument("--no-display", action="store_true")
    if preset:
        ap.set_defaults(bounds=PRESETS[preset]["bounds"])
    args = ap.parse_args()
    bounds = floats(args.bounds)
    lo, hi = np.array(bounds[0::2]), np.array(bounds[1::2])
    real = args.sink == "viam"

    det = {"model": None, "dargs": None, "name": args.lock}
    if not args.lock_color:
        det["model"], det["dargs"] = load_detector(args.lock, args.model, args.prompts, args.conf, args.imgsz,
                                                   standard="yolov8s.pt")
    robot = ViamRobot(args.viam_address, args.arm_name, args.gripper_name, None, TableToWorld(*floats(args.table_to_world)),
                      workspace=bounds[:4], max_z_mm=bounds[5], insecure=args.viam_insecure)
    robot.use_planner = args.planner
    lock = Lock(max_jump_frac=0.5, lost_after=10 ** 6)

    def current():
        p = robot.get_pose()
        return np.array(robot.table.to_robot(p.x, p.y, p.z))

    shared = {"cur": current(), "alive": True}
    if args.tip_offset_mm == 0:
        print("note: --tip-offset-mm is 0, so the 10 cm / 1 cm distances are measured from the CAMERA, not the fingertips. "
              "Measure camera-to-fingertip distance and pass it (or tell the chat: fingertips are 13 cm in front of the camera).",
              flush=True)

    # ---- obstacles ------------------------------------------------------------------------------------------------
    obs = Obstacles(margin_mm=args.margin_mm)
    if not args.no_world_obstacles:
        obs, loaded, notes = Obstacles.from_viam(robot, args.margin_mm, skip_names=(args.arm_name, args.gripper_name))
        if not loaded and preset == "lab":
            obs, loaded = Obstacles.lab_world(args.margin_mm), ["built-in lab world: table, walls, ceiling"]
        print("obstacles:", ", ".join(loaded) or "none loaded", flush=True)
        for n_ in notes:
            print("  note:", n_, flush=True)
    for spec in args.obstacle:
        obs.add(floats(spec), "custom")
    inside = obs.blocking(shared["cur"])
    if inside:
        print("warning: the arm starts inside the keep-out margin of", [b.name for b in inside],
              "- dropping those so it can move out", flush=True)
        obs.boxes = [b for b in obs.boxes if b not in inside]

    J = None
    if os.path.exists(JAC_FILE) and not args.probe:
        J = np.array(json.load(open(JAC_FILE))["J"])
        print("loaded the saved camera response from", JAC_FILE)
    ddir = depth_direction(J) if J is not None else None
    target = np.zeros(3)
    click = {}
    memo = {"template": None, "how": "detector"}

    def analyse(frame):
        """Runs on the detector thread for every new frame. If the full-frame pass misses the locked object, look
        again in a zoomed crop around where it was, then fall back to matching its last appearance."""
        cands, dets = [], []
        model, dargs, name = det["model"], det["dargs"], det["name"]
        if model is not None:
            dets = detect(model, frame, dargs)
            cands += object_candidates(dets, name)
            memo["how"] = "detector"
            if not cands and lock.locked:
                cands = roi_candidates(model, dargs, frame, lock.box, name)
                memo["how"] = "zoom"
                if not cands:
                    cands = template_candidates(frame, memo["template"], lock.box, name)
                    memo["how"] = "template"
        if args.lock_color:
            cands += color_candidates(frame, args.lock_color, args.lock_min_area)
        h, w = frame.shape[:2]
        live = lock.update(cands, w)
        if live and memo["how"] == "detector":
            memo["template"] = crop_gray(frame, lock.box)
        return dict(feats=features(lock.box, w, h) if live else None, live=live, dets=dets)

    robot.get_frame(args.camera)                      # create the camera client before threads share it
    grabber = FrameGrabber(lambda: robot.get_frame(args.camera), args.workers).start()
    perc = Perception(grabber, analyse)
    perc.start()

    if real:
        def pose_loop():                               # the arm's real Viam pose, a few times a second
            while shared["alive"]:
                try:
                    shared["cur"] = current()
                except Exception:
                    pass
                time.sleep(0.1)
        threading.Thread(target=pose_loop, daemon=True).start()

    rng = Range(args.object_height_mm, args.hfov)
    try:  # use the camera's own focal length when the machine reports it
        _, intr = robot.camera_info(args.camera)
        if intr:
            rng.focal_px, rng.focal_w = intr["fx"], intr["w"]
    except Exception:
        pass
    cfg = {"tip_off_cm": args.tip_offset_mm / 10.0, "stop_cm": args.stop_distance_mm / 10.0,
           "open_cm": args.open_at_cm, "slow_cm": args.slow_below_cm}

    # ---- shared state between the display loop and the mission thread -----------------------------------------------
    mu = threading.Lock()
    snap = {"x": None, "t": 0.0, "tip": None, "dist": None, "clipped": False, "hpx": None}
    S = {"phase": "idle", "detail": "", "plan": [], "abort": threading.Event(), "confirm": threading.Event(),
         "waiting": False, "opened": False, "thread": None, "progress": None, "plan_obj": None, "plan_ctx": None}
    kin_cache = {}

    def snapshot():
        with mu:
            return dict(snap)

    def check():
        if S["abort"].is_set():
            raise Abort()

    def say(text):
        print("robot:", text, flush=True)
        if ui:
            ui.say(text)

    def set_gripper(open_):
        if real:
            threading.Thread(target=lambda: robot.set_gripper(open_), daemon=True).start()

    def wait_fresh(after, timeout):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            check()
            sn = snapshot()
            if sn["x"] is not None and sn["t"] > after:
                return sn
            time.sleep(0.02)
        return None

    def wait_result(after, timeout=8.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            check()
            r_ = perc.result
            if r_ is not None and r_.t > after:
                return r_
            time.sleep(0.01)
        return None

    def move_blocking(pos):
        """Move there and wait (real arm), or just pretend (dry run)."""
        if real:
            robot.move_xyz(*pos)
            shared["cur"] = current()
        else:
            shared["cur"] = np.array(pos, float)
        return time.monotonic()

    def probe():
        """Learn J: nudge the gripper along x, y, z and watch how the object moves in the image."""
        say("Learning the camera response with three small moves (keep the object in view) ...")
        r0 = wait_result(time.monotonic())
        if r0 is None or r0.feats is None:
            say("I can't see the object, so I can't learn the camera response.")
            return None
        s0, p0, after, deltas = r0.feats, current(), [], []
        for i in range(3):
            tgt = p0.copy()
            d = args.probe_mm if p0[i] + args.probe_mm <= hi[i] else -args.probe_mm
            tgt[i] += d
            tgt = np.clip(tgt, np.minimum(lo, p0), np.maximum(hi, p0))
            safe, hit = obs.constrain(p0, tgt)
            if hit:
                say(f"The learning move along {'xyz'[i]} would enter {hit}; move the arm clear and try again.")
                return None
            r_ = wait_result(move_blocking(safe) + 0.3)
            s_ = None if r_ is None else r_.feats
            move_blocking(p0)
            if s_ is None:
                say(f"The object left the view when moving along {'xyz'[i]}.")
                return None
            after.append(s_)
            deltas.append(float(safe[i] - p0[i]) or d)
        wait_result(time.monotonic() + 0.3)
        Jn = estimate_jacobian(s0, after, deltas)
        os.makedirs(os.path.dirname(JAC_FILE), exist_ok=True)
        json.dump({"J": Jn.tolist(), "probe_mm": args.probe_mm}, open(JAC_FILE, "w"), indent=2)
        print("learned camera response J (frame fraction per mm; rows u, v, h; columns x, y, z):\n", np.round(Jn, 5), flush=True)
        return Jn

    # ---- mission: centre -> plan -> execute -----------------------------------------------------------------------
    def centre():
        """Camera-guided steps until the object is within --deadband of the middle (a few frames in a row). The rest of
        the offset is folded into the plan, so the gripper does not wait for a perfect centre. Checks that each step
        moved the picture the way the learned response J predicted and re-learns J if it did not."""
        nonlocal J, ddir
        S.update(phase="centring", detail="")
        t0, stable, lost, steps, bad, reprobes = time.monotonic(), 0, 0, 0, 0, 0
        gain, prev_e, worse = args.gain, None, 0
        last_end, prev = time.monotonic(), None            # prev: (features before the last move, the move)
        while True:
            check()
            if time.monotonic() - t0 > args.centre_timeout:
                say("I couldn't centre the object in time (is the camera view blocked, or the camera response wrong? "
                    "try 'p' to re-learn it). Stopping.")
                return False
            sn = wait_fresh(last_end + args.settle + 0.5 * grabber.latency, 2.5)
            if sn is None:
                lost += 1
                if lost >= 3:
                    say(f"I lost sight of the {det['name']} while centring. Stopping.")
                    return False
                continue
            lost = 0
            s = sn["x"]
            e = float(np.max(np.abs(target[:2] - s[:2])))
            S["detail"] = f"offset u{s[0]:+.2f} v{s[1]:+.2f} (need < {args.deadband:.2f})"
            if prev is not None and real:                 # did the last move do what J said it would?
                s_before, d_prev = prev
                pred, obs_ = (J @ d_prev)[:2], (s - s_before)[:2]
                if np.linalg.norm(pred) > 0.02 and np.linalg.norm(obs_) > 0.01:
                    cos = float(pred @ obs_ / (np.linalg.norm(pred) * np.linalg.norm(obs_)))
                    ratio = float(np.linalg.norm(obs_) / np.linalg.norm(pred))
                    bad = bad + 1 if (cos < 0.3 or not 0.3 < ratio < 3.0) else 0
                elif np.linalg.norm(pred) > 0.04 and np.linalg.norm(obs_) < 0.005:
                    bad += 1                              # moved a lot and the picture did not change
                if bad >= 2 and reprobes < 2:
                    say("The picture is not moving the way my camera model predicts, so I'll re-learn it.")
                    Jn = probe()
                    if Jn is None:
                        return False
                    J, ddir, bad, reprobes, prev = Jn, depth_direction(Jn), 0, reprobes + 1, None
                    last_end = time.monotonic()
                    continue
            if e < args.deadband:
                stable += 1
                last_end, prev = time.monotonic(), None
                if stable >= args.stable_frames:
                    say(f"Centred to within {100 * e:.0f}% (limit {100 * args.deadband:.0f}%): planning the approach.")
                    return True
                continue
            stable = 0
            if prev_e is not None and e > prev_e * 1.1:   # getting worse: take smaller steps
                worse += 1
                if worse >= 2:
                    gain, worse = max(gain * 0.7, 0.25), 0
                    print(f"centring: error rose, step gain now {gain:.2f}", flush=True)
            else:
                worse = 0
            prev_e = e
            d = servo_step(J, s, target, (0, 1), gain, args.max_step)
            cur = shared["cur"]
            new = np.clip(cur + d, np.minimum(lo, cur), np.maximum(hi, cur))
            safe, hit = obs.constrain(cur, new)
            if hit:
                say(f"Centring would enter {hit}; stopping short of it.")
            steps += 1
            if steps % 3 == 1:
                say(f"Centring: object {100 * e:.0f}% off-centre, moving {np.linalg.norm(safe - cur):.0f} mm.")
            if not real:
                say(f"Dry run: I would centre it with a move of {d[0]:+.0f} {d[1]:+.0f} {d[2]:+.0f} mm "
                    f"(offset u{s[0]:+.2f} v{s[1]:+.2f}). Assuming it is centred and planning from here.")
                return True
            prev = (s.copy(), safe - cur)
            last_end = move_blocking(safe)

    def refine_direction(c0_mm):
        """Once centred, the camera axis keeps the object centred. Test that with a short forward move along the planned
        direction: any sideways drift of the object in the image means the direction is slightly off, and J turns the
        drift into the correction. Moves back afterwards."""
        nonlocal ddir
        sn0 = wait_fresh(time.monotonic() + 0.1, 2.0)
        if sn0 is None or not real:
            return
        s0, p_c, step = sn0["x"].copy(), shared["cur"].copy(), 15.0
        tgt = np.clip(p_c + ddir * step, np.minimum(lo, p_c), np.maximum(hi, p_c))
        safe, hit = obs.constrain(p_c, tgt)
        if hit or np.linalg.norm(safe - p_c) < step * 0.8:
            return
        t_end = move_blocking(safe)
        sn1 = wait_fresh(t_end + args.settle, 2.0)
        move_blocking(p_c)
        time.sleep(args.settle)
        if sn1 is None:
            return
        # moving toward an off-centre object pushes it outward in the picture by itself (perspective); take that out
        drift = (sn1["x"][:2] - s0[:2]) - s0[:2] * (step / max(c0_mm, 50.0))
        corr = np.linalg.pinv(J[:2]) @ drift
        new = ddir * step - corr
        new = new / np.linalg.norm(new)
        ang = float(np.degrees(np.arccos(np.clip(new @ ddir, -1, 1))))
        ddir = new
        say(f"Straightened the approach direction by {ang:.1f} deg using a {step:.0f} mm test move.")

    def measure_tip(n=8, timeout=5.0):
        """Median gripper-to-object distance over several fresh frames while the arm is still."""
        rng.smooth = None
        samples, last_t, t0 = [], time.monotonic(), time.monotonic()
        while len(samples) < n and time.monotonic() - t0 < timeout:
            check()
            sn = snapshot()
            if sn["tip"] is not None and sn["t"] > last_t:
                samples.append(sn["tip"])
                last_t = sn["t"]
            time.sleep(0.02)
        if len(samples) < max(3, n // 2):
            return None, 0.0
        return float(np.median(samples)), float(np.std(samples))

    def get_kin():
        if "v" not in kin_cache:
            if args.no_joint_space:
                kin_cache["v"] = (None, "joint-space planning is off (--no-joint-space)")
            else:
                kin_cache["v"] = load_validated(robot) if real else (None, "dry run: no kinematics check")
            print("kinematics:", kin_cache["v"][1], flush=True)
        return kin_cache["v"]

    def make_plan(tip, lateral=None):
        p0 = shared["cur"].copy()
        plan = build_plan(p0, ddir, tip, cfg["stop_cm"], cfg["open_cm"], cfg["slow_cm"], args.fast_speed, args.slow_speed,
                          obstacles=obs, bounds=bounds, step_mm=args.plan_step_mm, max_travel_mm=args.max_travel_mm,
                          lateral=lateral)
        if lateral is not None and float(np.linalg.norm(lateral)) > 2.0:
            plan.notes.append(f"includes a {float(np.linalg.norm(lateral)):.0f} mm sideways correction for the leftover centring error")
        kin, note = get_kin()
        if kin is not None:
            attach_joints(plan, kin, robot.get_joints(), robot.table.apply, args.max_joint_jump)
        else:
            plan.notes.append(note + "; using Cartesian moves")
        return plan

    def speed_at(plan, s):
        for w in plan.waypoints:
            if w.s_mm >= s:
                return max(w.speed, args.slow_speed * 0.5)
        return args.slow_speed

    def maybe_open(plan, prog):
        if not S["opened"] and prog >= plan.open_s_mm:
            S["opened"] = True
            set_gripper(True)
            say(f"Opening the gripper now ({prog / 10:.1f} cm travelled, about {cfg['open_cm']:.0f} cm from the object).")

    def run_segment(plan, a, b, v):
        """Execute waypoints a..b at about v mm/s, watching the arm's real pose."""
        p0, d, end_s = plan.p0, plan.direction, plan.waypoints[b].s_mm
        wps = plan.waypoints[a:b + 1]
        joint_ok = all(w.joints is not None for w in wps) and real
        fut = None
        if joint_ok:
            vmax = segment_joint_speed(plan, a, b, v)
            fut = robot.move_through_joints([w.joints for w in wps], vmax, vmax * args.joint_accel, wait=False)
            S["detail"] = f"joint trajectory, up to {vmax:.1f} deg/s"
        s_cmd = plan.waypoints[a].s_mm
        futs, t_last = [], time.monotonic()
        while True:
            if S["abort"].is_set():
                if real:
                    robot.stop()
                raise Abort()
            now = time.monotonic()
            dt, t_last = now - t_last, now
            if not real:                                               # dry run: walk the path at the planned speed
                s_cmd = min(end_s, s_cmd + speed_at(plan, s_cmd) * dt)
                shared["cur"] = p0 + d * s_cmd
            prog = float(np.dot(shared["cur"] - p0, d))
            S["progress"] = prog
            maybe_open(plan, prog)
            if fut is not None:
                if fut.done():
                    err = fut.exception()
                    if err and "cancel" not in f"{err!r}".lower():
                        raise RuntimeError(f"the arm refused the trajectory: {err}")
                    return
            elif real:                                                  # Cartesian fallback: stream poses along the line
                if prog >= end_s - 1.0:
                    return
                s_cmd = min(end_s, s_cmd + speed_at(plan, s_cmd) * dt, prog + 40.0)
                pos = p0 + d * s_cmd
                safe, hit = obs.constrain(shared["cur"], pos)
                if hit:
                    raise RuntimeError(f"the line would enter {hit}")
                futs = [f for f in futs if not f.done()][-1:] + [robot.move_xyz(*safe, wait=False)]
            elif s_cmd >= end_s - 1e-6:
                return
            time.sleep(0.1 if (real and fut is None) else 0.05)
            S["detail"] = f"{prog / 10:.1f} cm travelled"

    def execute(plan):
        """-> 'done', or ('replan', measured tip cm) / 'recentre'."""
        S["phase"] = "executing"
        if plan.open_s_mm <= 0:
            maybe_open(plan, 0.0)
        segs = plan.segments(cfg["slow_cm"])
        for i, (name, a, b, v) in enumerate(segs):
            say(f"Leg {i + 1}/{len(segs)} ({name}): {plan.waypoints[a].tip_cm:.1f} -> {plan.waypoints[b].tip_cm:.1f} cm at ~{v:.0f} mm/s")
            run_segment(plan, a, b, v)
            if name == "fast" and i + 1 < len(segs) and real:
                time.sleep(0.4)                                          # let the arm and picture settle, then check
                sn = wait_fresh(time.monotonic(), 2.0)
                if sn is None or sn["clipped"]:
                    say("I can't judge the object here, so I'll continue with the plan.")
                    continue
                if float(np.max(np.abs(sn["x"][:2]))) > 2 * args.deadband:
                    say("The object drifted off-centre; re-centring and re-planning.")
                    return "recentre"
                m, _ = measure_tip(n=6)
                exp = plan.waypoints[b].tip_cm
                if m is not None and abs(m - exp) > max(1.5, 0.25 * exp):
                    say(f"The object is {m:.1f} cm away, the plan expected {exp:.1f} cm. Re-planning from the measurement.")
                    return ("replan", m)
        return "done"

    def trim():
        """After the plan: if the camera can still judge the distance, nudge along the approach line to the stop distance
        (at most 30 mm either way, backing off if it is too close)."""
        if not real:
            return
        time.sleep(args.settle)
        sn = wait_fresh(time.monotonic(), 2.5)
        if sn is None or sn["clipped"]:
            return
        m, _ = measure_tip(n=6, timeout=3.0)
        if m is None:
            return
        err_mm = (m - cfg["stop_cm"]) * 10.0
        if abs(err_mm) < 4.0 or abs(err_mm) > 30.0:
            return
        cur = shared["cur"]
        safe, hit = obs.constrain(cur, np.clip(cur + ddir * err_mm, np.minimum(lo, cur), np.maximum(hi, cur)))
        if hit:
            return
        S["detail"] = f"trimming {err_mm:+.0f} mm"
        say(f"The camera estimates {m:.1f} cm; moving {err_mm:+.0f} mm along the line to reach {cfg['stop_cm']:.1f} cm.")
        move_blocking(safe)

    def mission():
        nonlocal J, ddir
        try:
            S.update(plan=[], waiting=False, progress=None, opened=False, plan_obj=None, plan_ctx=None)
            if J is None:
                if real:
                    J = probe()
                    if J is None:
                        S.update(phase="idle", detail="")
                        return
                else:
                    J = GUESS_J
                    print("dry run with a GUESSED camera response (use --sink viam to learn the real one)", flush=True)
                ddir = depth_direction(J)
            if ddir is None:
                say("The learned camera response has no depth direction. Re-learn it with the object in clear view (p).")
                return
            tip_override, replans = None, 0
            while True:
                if not centre():
                    S.update(phase="idle")
                    return
                if tip_override is None:
                    S.update(phase="measuring", detail="")
                    tip, spread = measure_tip()
                    if tip is None:
                        say(f"I can't get a steady distance to the {det['name']}. Stopping.")
                        S.update(phase="idle")
                        return
                    S.update(phase="aligning", detail="")
                    refine_direction(tip * 10.0 + args.tip_offset_mm)
                    time.sleep(args.settle)
                else:
                    tip, tip_override = tip_override, None
                S.update(phase="planning", detail="")
                sn_now = wait_fresh(time.monotonic(), 2.5)
                lat = np.zeros(3)
                if sn_now is not None:                    # the leftover centring error becomes part of the plan
                    lat = servo_step(J, sn_now["x"], target, (0, 1), 1.0, 80.0)
                plan = make_plan(tip, lat)
                lines = describe(plan, cfg["slow_cm"], args.fast_speed, args.slow_speed, cfg["open_cm"], args.tip_offset_mm)
                S["plan"] = lines
                with perc.mu:
                    bx = lock.box.copy() if lock.locked else None
                c0 = tip * 10.0 + args.tip_offset_mm
                S["plan_ctx"] = {"c0_mm": c0, "p_obj": plan.p0 + plan.direction * c0,
                                 "size": (float(bx[2] - bx[0]), float(bx[3] - bx[1])) if bx is not None else (80.0, 120.0)}
                S["progress"] = 0.0
                S["plan_obj"] = plan                                   # the video overlay draws this live
                say("\n".join(lines))
                if not plan.ok or plan.travel_mm < 5:
                    say("I won't execute this plan (it is blocked or there is nothing to do).")
                    S.update(phase="idle")
                    return
                if args.confirm_plan:
                    S.update(phase="waiting for yes", waiting=True)
                    S["confirm"].clear()
                    say("Say 'yes' to execute the plan, or 'stop'.")
                    while not S["confirm"].wait(0.1):
                        check()
                    S["waiting"] = False
                else:
                    for k in range(int(args.countdown * 10), 0, -1):
                        S.update(phase=f"executing in {k / 10:.1f}s (say stop to cancel)")
                        check()
                        time.sleep(0.1)
                result = execute(plan)
                if result == "done":
                    break
                replans += 1
                if replans > args.replans:
                    say("Too many re-plans. Stopping here.")
                    S.update(phase="idle")
                    return
                if isinstance(result, tuple):
                    tip_override = result[1]
            trim()
            m, _ = measure_tip(n=4, timeout=2.0)
            S.update(phase="arrived", detail="")
            say(f"Arrived: stopped {cfg['stop_cm']:.1f} cm from the {det['name']}"
                + (f" (the camera now estimates {m:.1f} cm)." if (m is not None and real) else "."))
        except Abort:
            S.update(phase="stopped", detail="")
        except Exception as e:
            say(f"Mission failed: {type(e).__name__}: {e}")
            if real:
                robot.stop()
            S.update(phase="idle", detail="")

    def start_mission():
        stop_mission()
        S["abort"].clear()
        th = threading.Thread(target=mission, daemon=True)
        S["thread"] = th
        th.start()

    def stop_mission(stop_arm=True):
        S["abort"].set()
        S["confirm"].set()
        if stop_arm and real:
            try:
                robot.stop()
            except Exception:
                pass
        th = S["thread"]
        if th is not None and th.is_alive() and th is not threading.current_thread():
            th.join(timeout=5)

    # ---- chat UI ---------------------------------------------------------------------------------------------------
    cmdq, ui, pending, coco = queue.Queue(), None, {"confirm": None}, {}

    def coco_names():
        if "n" not in coco:
            from ultralytics import YOLO
            coco["n"] = set(YOLO(os.path.join(ROOT, "weights", "yolov8s.pt")).names.values())
        return coco["n"]

    def on_text(text):
        """Runs on the web server thread: answer what is quick, queue what must run on the main loop."""
        c = parse(text)
        k = c["intent"]
        if k == "help":
            return HELP
        if k in ("empty", "unknown"):
            return "I didn't catch that. Try: go to the cup  (or type help)." if k == "unknown" else None
        if k in ("stop", "yes"):
            if k == "yes" and S["waiting"]:
                S["confirm"].set()
                return "Executing the plan."
            cmdq.put(c)
            return None
        if k == "look":
            r_ = perc.result
            if r_ is None or not r_.dets:
                return "I don't see any known objects right now."
            counts = {}
            for d in r_.dets:
                counts[d[5]] = counts.get(d[5], 0) + 1
            return "I see: " + ", ".join(f"{v} {k_}" if v > 1 else k_ for k_, v in sorted(counts.items()))
        if k == "status":
            sn = snapshot()
            return (f"{S['phase']}{(' - ' + S['detail']) if S['detail'] else ''}. Target: {det['name']}."
                    + (f" About {sn['tip']:.1f} cm from the object." if sn["tip"] is not None else "")
                    + f" Stop distance {cfg['stop_cm']:.1f} cm, gripper opens at {cfg['open_cm']:.0f} cm."
                    + (" Gripper open." if S["opened"] else ""))
        cmdq.put(c)
        return None

    def get_status():
        sn = snapshot()
        return {"mode": "LIVE ARM" if real else "DRY RUN", "state": S["phase"] + (f" ({S['detail']})" if S["detail"] else ""),
                "target": det["name"],
                "distance": None if sn["tip"] is None else f"{sn['tip']:.1f} cm" + ("" if rng.calibrated else " (est)"),
                "stop_at": f"{cfg['stop_cm']:.1f} cm", "cam_fps": round(grabber.fps.value, 1),
                "det_fps": round(perc.fps.value, 1), "plan": S["plan"]}

    def choose_object(name):
        """'white cup' -> the model knows 'cup'; keep unknown names for the open-vocabulary model."""
        names = coco_names()
        if name in names or any(name in n for n in names):
            return name, ""
        last = name.split()[-1]
        if last in names:
            return last, f" (I'll look for a '{last}' and ignore '{name.rsplit(' ', 1)[0]}')"
        return name, " (that isn't a standard class, so I'll use the slower open-vocabulary model)"

    def handle(c):
        """Main-loop side of the chat commands."""
        k = c["intent"]
        if k == "target":
            stop_mission()
            name, note = choose_object(c["name"])
            say(f"Looking for {name}{note} ...")
            if name != det["name"] or det["model"] is None:
                model, dargs = load_detector(name, args.model, args.prompts, args.conf, args.imgsz, standard="yolov8s.pt")
                with perc.mu:
                    det.update(model=model, dargs=dargs, name=name)
                    lock.unlock()
                    lock.enabled = True
                    memo["template"] = None
            else:
                with perc.mu:
                    lock.unlock()
                    lock.enabled = True
            flt.reset()
            with mu:
                snap.update(x=None, tip=None, dist=None)
            if "stop_cm" in c:
                cfg["stop_cm"] = max(0.5, c["stop_cm"])
            say(f"Going for the {name}: centre it, then plan a straight approach (fast, slowing below {cfg['slow_cm']:.0f} cm, "
                f"gripper opens at {cfg['open_cm']:.0f} cm, stop {cfg['stop_cm']:.1f} cm away)."
                + ("" if real else " (Dry run: the arm will not move.)"))
            start_mission()
        elif k in ("stop", "pause"):
            stop_mission()
            S.update(phase="stopped", detail="")
            say("Stopped. The arm is holding where it is." if k == "stop" else "Paused. Say 'resume' to start again.")
        elif k == "resume":
            if lock.locked:
                say("Resuming: centre, plan, execute.")
                start_mission()
            else:
                say("Nothing is locked yet. Tell me what to go to.")
        elif k == "unlock":
            stop_mission()
            with perc.mu:
                lock.unlock()
            flt.reset()
            S.update(phase="idle", detail="", plan=[], plan_obj=None, plan_ctx=None)
            say("Unlocked. Tell me what to go to next.")
        elif k == "stop_distance":
            cfg["stop_cm"] = max(0.5, c["cm"])
            say(f"I'll stop {cfg['stop_cm']:.1f} cm from the object.")
        elif k == "set":
            if c["key"] == "tip_offset_mm":
                cfg["tip_off_cm"] = c["value"] / 10.0
                args.tip_offset_mm = c["value"]
                say(f"OK: the fingertips are {c['value'] / 10:.1f} cm in front of the camera; distances are now gripper-to-object.")
            else:
                rng.real_mm, rng.smooth = c["value"], None
                args.object_height_mm = c["value"]
                say(f"OK: the {det['name']} is {c['value'] / 10:.1f} cm tall, which sets the distance estimate.")
        elif k == "calibrate":
            sn = snapshot()
            if sn["x"] is None:
                say("I can't see the object, so I can't calibrate.")
            else:
                rng.calibrate(sn["hpx"], c["cm"])
                say(f"Calibrated: the {det['name']} is {c['cm']:.0f} cm from the camera at this size.")
        elif k == "gripper":
            S["opened"] = c["open"]
            set_gripper(c["open"])
            say("Opening the gripper." if c["open"] else "Closing the gripper.")
        elif k == "reset":
            stop_mission()
            pending["confirm"] = "reset"
            say("Reset moves every joint to the saved default pose in one sweep (wrist joint 5 turns a long way). "
                "Keep the area clear and type 'yes' to go, or 'no' to cancel.")
        elif k == "yes" and pending["confirm"] == "reset":
            pending["confirm"] = None
            if real:
                say("Resetting ...")
                robot.set_joints(RESET_JOINTS)
                shared["cur"] = current()
                say("Reset done.")
            else:
                say("Dry run: I would move to the reset pose now.")
        elif k in ("yes", "no"):
            pending["confirm"] = None
            say("OK." if k == "no" else "Nothing to confirm.")

    show = not args.no_display
    if show:
        cv2.namedWindow(WIN)
        cv2.setMouseCallback(WIN, lambda e, x, y, f, p: click.update(pt=(x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
    if args.ui:
        ui = UIServer(on_text, get_status, args.port).start()
        print(f"chat UI: {ui.url}", flush=True)
        say(f"Ready ({'LIVE ARM' if real else 'dry run'}). Tell me what to go to, e.g. \"go to the cup\". I'll show the plan "
            "before moving. Type help for commands.")
        if not args.no_browser:
            webbrowser.open(ui.url)
    if args.auto_engage:
        cmdq.put({"intent": "target", "name": args.lock})

    flt, flt_t, misses, last_print, last_draw = FeatureFilter(), 0.0, 0, 0.0, 0.0
    try:
        while True:
            while True:
                try:
                    handle(cmdq.get_nowait())
                except queue.Empty:
                    break
            r = perc.result
            if r is None:
                cv2.waitKey(10) if show else time.sleep(0.01)
                continue
            if click.get("pt"):
                mx, my = click.pop("pt")
                hit_ = [d for d in r.dets if d[0] <= mx <= d[2] and d[1] <= my <= d[3]]
                if hit_:
                    d = min(hit_, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
                    with perc.mu:
                        lock.pin(d[:4], d[5])
            now = time.monotonic()
            if r.t > flt_t:                                   # feed every new detection through the filter
                flt_t = r.t
                if r.feats is not None:
                    flt.update(r.feats, r.t)
                    misses = 0
                    hpx = flt.x[2] * r.frame.shape[0]
                    dist = rng.cm(hpx, r.frame.shape[1])
                    with perc.mu:
                        bx = lock.box.copy() if lock.locked else None
                    clipped = bx is not None and (bx[1] <= 2 or bx[3] >= r.frame.shape[0] - 2)
                    with mu:
                        snap.update(x=flt.x.copy(), t=r.t, dist=dist, hpx=hpx, clipped=clipped,
                                    tip=None if dist is None else dist - cfg["tip_off_cm"])
                else:
                    misses += 1
                    if misses > 5:
                        flt.reset()
                        with mu:
                            snap.update(x=None, tip=None, dist=None)
            if now - last_print > 2.0:
                last_print = now
                sn = snapshot()
                print(f"[{S['phase']}] {S['detail']}  " + ("" if sn["tip"] is None else f"{sn['tip']:.1f} cm  ")
                      + f"cam {grabber.fps.value:.0f} fps det {perc.fps.value:.0f} fps", flush=True)

            if (show or ui) and now - last_draw >= 0.05:                    # redraw up to 20x a second, live
                last_draw = now
                frame = r.frame.copy()
                h, w = frame.shape[:2]
                sn = snapshot()
                with perc.mu:
                    draw_lock(frame, lock, r.live, sn["tip"] if r.live else None, rng.calibrated)
                cv2.drawMarker(frame, (w // 2, h // 2), (0, 255, 0), cv2.MARKER_CROSS, 40, 2)
                if r.feats is not None:
                    cv2.line(frame, (w // 2, h // 2), (int((r.feats[0] + 0.5) * w), int((r.feats[1] + 0.5) * h)), (0, 255, 255), 2)
                busy = S["thread"] is not None and S["thread"].is_alive()
                col = (0, 200, 0) if busy else (0, 0, 220)
                lines = [f"{S['phase'].upper() if (busy or S['phase'] in ('arrived', 'stopped')) else 'IDLE - say go to the cup'} | {args.sink}", S["detail"],
                         ("" if sn["tip"] is None else f"gripper-to-object {sn['tip']:.1f} cm" + ("" if rng.calibrated else " (est)")
                          + f"   stop {cfg['stop_cm']:.1f}  open {cfg['open_cm']:.0f}" + ("  GRIPPER OPEN" if S["opened"] else ""))]
                for i, text in enumerate(lines):
                    cv2.putText(frame, text, (14, 34 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
                    cv2.putText(frame, text, (14, 34 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col if i == 0 else (255, 255, 255), 2)
                po = S["plan_obj"]
                if po is not None and S["plan_ctx"] is not None:                # the planned trajectory, drawn on the picture
                    with perc.mu:
                        bx = lock.box.copy() if lock.locked else None
                    ctre = ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2) if bx is not None else (w / 2, h / 2)
                    size_now = (float(bx[2] - bx[0]), float(bx[3] - bx[1])) if bx is not None else None
                    draw_plan(frame, po, S["plan_ctx"], shared["cur"], bounds, obs.boxes, args.slow_speed, ctre,
                              cfg["slow_cm"], cfg["open_cm"], args.tip_offset_mm, progress_mm=S["progress"] or 0.0,
                              size_now=size_now)
                if ui:
                    ok_, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok_:
                        ui.set_frame(buf.tobytes())
                if not show:
                    time.sleep(0.005)
                    continue
                cv2.imshow(WIN, frame)
                key = cv2.waitKey(5) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    cmdq.put({"intent": "target", "name": det["name"]})
                elif key == ord("p"):
                    J, ddir, args.probe = None, None, True
                elif key == ord("c") and sn["tip"] is not None:
                    rng.calibrate(sn["hpx"], args.calib_cm)
                    print(f"calibrated: the object is {args.calib_cm:.0f} cm from the camera at the current size", flush=True)
                elif key == ord("t") and sn["tip"] is not None:
                    cfg["stop_cm"] = max(0.5, sn["tip"])
                    print(f"stop distance taught: {cfg['stop_cm']:.1f} cm", flush=True)
                elif key == ord("x"):
                    cmdq.put({"intent": "stop"})
                elif key == ord("l"):
                    cmdq.put({"intent": "unlock"})
                elif key == ord("r"):
                    with perc.mu:
                        lock.unlock()
                        lock.enabled = True
            else:
                time.sleep(0.005)
    finally:
        shared["alive"] = False
        stop_mission()
        grabber.stop()
        perc.stop()
        try:
            if real:
                robot.stop()
        finally:
            cv2.destroyAllWindows()
            robot.close()


if __name__ == "__main__":
    main()
