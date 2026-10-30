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
                --clearance-mm (3 cm) of clearance for EVERY link of the arm and --obstacle boxes, and printed in the chat / on the page before it runs.
    3. execute  one MoveThroughJointPositions trajectory per leg (fast, slow) at the planned speed; the gripper opens
                when the arm's real Viam pose passes the planned point. After the fast leg the object is re-measured
                and the plan is rebuilt if reality differs from the plan.
    --planner   uses Viam's collision-planning motion service for the centring moves too.

    scripts/run.sh src/servo_object.py --viam-address <machine>.viam.cloud --preset lab --ui --tip-offset-mm 130   # dry run
    ... --sink viam                                                                                              # real arm
Pick and deliver ("pick up the cup and bring it to the closest person"): reach as above but stop with the fingertips
--grasp-depth-mm past the object's front surface; at 10 cm the object's position is MEASURED and REMEMBERED (the object
fills the picture and gets clipped after that, so the last centimetres follow the remembered position); then close the
gripper with a limited force (--grip-force, at most 3), check it really holds the object, LIFT it straight up
--lift-mm, show that plan too, then look again, lock on the CLOSEST person and approach them (--person-stop-cm,
slower, and only after you say yes). Chat (--ui): "go to the cup", "get within 5 cm of the cup", "fingertips are 13 cm in front of the camera", "the cup is 9 cm
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
from mimic.kinematics import load_validated, nearest_equivalent, smooth_path  # noqa: E402
from mimic.overlay import draw_plan  # noqa: E402
from mimic.pipeline import FrameGrabber, Perception  # noqa: E402
from mimic.plan import (attach_joints, attach_joints_down, build_plan, describe, describe_line,  # noqa: E402
                        line_plan, segment_joint_speed)
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
    ap.add_argument("--grasp-depth-mm", type=float, default=40.0,
                    help="pick-up: how far past the object's front surface the fingertips go before closing")
    ap.add_argument("--grip-force", type=float, default=2.5, help="gripper force when closing (limited to 3 at most)")
    ap.add_argument("--grip-do-command", metavar="JSON",
                    help='optional DoCommand sent to the gripper before closing, e.g. \'{"set_force": 2.5}\' (driver specific)')
    ap.add_argument("--lift-mm", type=float, default=100.0, help="pick-up: lift this far straight up after gripping")
    ap.add_argument("--lift-speed", type=float, default=40.0, help="lift speed, mm/s")
    ap.add_argument("--person-stop-cm", type=float, default=50.0, help="stop this far from the person (hand-over distance)")
    ap.add_argument("--person-height-mm", type=float, default=1700.0, help="person height used to estimate their distance")
    ap.add_argument("--person-fast-speed", type=float, default=80.0, help="approach speed toward a person, mm/s")
    ap.add_argument("--person-slow-speed", type=float, default=30.0)
    ap.add_argument("--pan-threshold", type=float, default=0.15,
                    help="if the target is further than this fraction of the width off-centre, turn the base joint toward it first")
    ap.add_argument("--no-person-confirm", action="store_true", help="do not wait for 'yes' before approaching a person")
    ap.add_argument("--pick", action="store_true", help="with --auto-engage: pick the object up instead of just reaching it")
    ap.add_argument("--handover", action="store_true", help="with --pick: then bring it to the closest person")
    ap.add_argument("--aim-frac", type=float, default=0.5,
                    help="which part of the object the gripper lines up with: 0.5 = its middle (default, keeps clear of "
                    "obstacles above and below), 0.75 = middle of the lower half")
    ap.add_argument("--grasp-from", choices=["top", "side"], default="top",
                    help="top: the camera looks straight DOWN at the object and the gripper descends onto it (default); "
                    "side: level gripper, approach from the side")
    ap.add_argument("--down-wrist", default="15.29,-181.92", metavar="J4,J5 (or J3,J4,J5)",
                    help="the camera-facing-down wrist posture (deg). The other joints are solved so the camera points straight down "
                    "(and, when going over an object, sits right above it)")
    ap.add_argument("--camera-down-at", choices=["near", "start", "never"], default="near",
                    help="near: start as the arm is, centre, approach to --turn-down-cm, raise --turn-down-lift-mm, THEN turn the "
                    "camera down above the object and descend; start: face down from the beginning; never")
    ap.add_argument("--turn-down-cm", type=float, default=15.0, help="distance to the object at which the camera is turned down")
    ap.add_argument("--turn-down-lift-mm", type=float, default=100.0, help="raise this much before turning the camera down")
    ap.add_argument("--top-height-range", default="100,450", metavar="MIN,MAX",
                    help="camera height above the object (mm) accepted for the look-down posture")
    ap.add_argument("--lock-joints", default="5", metavar="N[,N]",
                    help="joints that keep their angle during every move while the camera faces down (default 5, the camera "
                    "roll). Joints 3 and 4 may be added, but then the camera can tilt (see --max-tilt-deg)")
    ap.add_argument("--max-tilt-deg", type=float, default=1.0, help="largest tilt from straight down accepted while moving")
    ap.add_argument("--joint-move-speed", type=float, default=45.0, help="joint speed for small positioning moves, deg/s")
    ap.add_argument("--free-roll", action="store_true",
                    help="do not lock any joint: let the arm's own orientation-holding moves turn joint 5 as they like")
    ap.add_argument("--no-face-down", action="store_true", help="do not start by turning the camera to face down")
    ap.add_argument("--camera-axis-tool", default="0,0,1", metavar="X,Y,Z",
                    help="the camera's viewing axis in the tool frame (default: along the tool's z axis)")
    ap.add_argument("--top-height-mm", type=float, default=200.0,
                    help="top grasp: how far above the object the camera sits when it looks down at it")
    ap.add_argument("--top-wrist-roll", type=float, default=-360.0,
                    help="top grasp: joint 5 angle (deg, taken modulo 360 by the shortest turn)")
    ap.add_argument("--object-top-mm", type=float, default=80.0,
                    help="top grasp: the object's size as seen FROM ABOVE (its width/diameter), for the distance estimate")
    ap.add_argument("--approach-wrist", default="90,-360", metavar="J4,J5",
                    help="wrist joint angles (deg) to hold for the approach: joint 4 horizontal, joint 5 turned to match")
    ap.add_argument("--map-scale", type=float, default=0.3,
                    help="size of the two small plan maps on the video, as a fraction of their original size (0 = hide them)")
    ap.add_argument("--no-orient", action="store_true", help="do not turn the wrist before the approach")
    ap.add_argument("--orient-speed", type=float, default=20.0, help="joint speed for the wrist turn, deg/s")
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
    ap.add_argument("--max-travel-mm", type=float, default=700.0)
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
    ap.add_argument("--clearance-mm", type=float, default=30.0, help="every part of the arm stays at least this far from obstacles")
    ap.add_argument("--arm-radius-mm", type=float, default=40.0, help="assumed thickness (radius) of the arm's links")
    ap.add_argument("--gripper-length-mm", type=float, default=120.0, help="how far the gripper reaches past the tool point")
    ap.add_argument("--no-body-guard", action="store_true", help="skip the arm-body obstacle checks (tool point only)")
    ap.add_argument("--orient-confirm-deg", type=float, default=180.0, help="ask before turning any wrist joint more than this")
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

    det = {"model": None, "dargs": None, "name": args.lock, "color": args.lock != "person"}
    if not args.lock_color or args.lock == "person":
        det["model"], det["dargs"] = load_detector(args.lock, args.model, args.prompts, args.conf, args.imgsz,
                                                   standard="yolov8s.pt")
    robot = ViamRobot(args.viam_address, args.arm_name, args.gripper_name, None, TableToWorld(*floats(args.table_to_world)),
                      workspace=bounds[:4], max_z_mm=bounds[5], insecure=args.viam_insecure)
    robot.use_planner = args.planner
    lock = Lock(max_jump_frac=0.5, lost_after=10 ** 6)

    def current():
        p = robot.get_pose()
        return np.array(robot.table.to_robot(p.x, p.y, p.z))

    shared = {"cur": current(), "alive": True, "q": np.array(robot.get_joints(), float), "slack": (float("inf"), "")}
    if args.tip_offset_mm == 0:
        print("note: --tip-offset-mm is 0, so the 10 cm / 1 cm distances are measured from the CAMERA, not the fingertips. "
              "Measure camera-to-fingertip distance and pass it (or tell the chat: fingertips are 13 cm in front of the camera).",
              flush=True)

    # ---- obstacles ------------------------------------------------------------------------------------------------
    obs = Obstacles(margin_mm=args.clearance_mm + args.arm_radius_mm)
    if not args.no_world_obstacles:
        obs, loaded, notes = Obstacles.from_viam(robot, args.clearance_mm + args.arm_radius_mm, skip_names=(args.arm_name, args.gripper_name))
        if not loaded and preset == "lab":
            obs, loaded = Obstacles.lab_world(args.clearance_mm + args.arm_radius_mm), ["built-in lab world: table, walls, ceiling"]
        print("obstacles:", ", ".join(loaded) or "none loaded", flush=True)
        for n_ in notes:
            print("  note:", n_, flush=True)
    for spec in args.obstacle:
        obs.add(floats(spec), "custom")
    near = [f"{'xyz'[k]} {'max' if abs(shared['cur'][k] - hi[k]) < abs(shared['cur'][k] - lo[k]) else 'min'} "
            f"{hi[k] if abs(shared['cur'][k] - hi[k]) < abs(shared['cur'][k] - lo[k]) else lo[k]:.0f}"
            for k in range(3) if min(abs(shared["cur"][k] - lo[k]), abs(shared["cur"][k] - hi[k])) < 60.0]
    if near:
        c = shared["cur"]
        print(f"warning: the tool is at x{c[0]:.0f} y{c[1]:.0f} z{c[2]:.0f}, within 60 mm of the --bounds limit(s) "
              f"{', '.join(near)}; plans will be cut short there. Widen the bounds to give it room, e.g. "
              f"--bounds {c[0] - 300:.0f},{c[0] + 450:.0f},{c[1] - 400:.0f},{c[1] + 400:.0f},{c[2] - 450:.0f},{c[2] + 250:.0f}", flush=True)
    inside = obs.blocking(shared["cur"])
    if inside:
        print("warning: the arm starts inside the keep-out margin of", [b.name for b in inside],
              "- dropping those so it can move out", flush=True)
        obs.boxes = [b for b in obs.boxes if b not in inside]

    J, jref = None, {"c": None}                      # jref: camera-to-object distance (mm) at which J was learned
    if os.path.exists(JAC_FILE) and not args.probe:
        saved = json.load(open(JAC_FILE))
        J, jref["c"] = np.array(saved["J"]), saved.get("c_mm")
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
            if name == "person":                                  # the closest person looks the biggest
                cands.sort(key=lambda d: -(d[3] - d[1]))
            memo["how"] = "detector"
            if not cands and lock.locked:
                cands = roi_candidates(model, dargs, frame, lock.box, name)
                memo["how"] = "zoom"
                if not cands:
                    cands = template_candidates(frame, memo["template"], lock.box, name)
                    memo["how"] = "template"
        if args.lock_color and det["color"]:
            cands += color_candidates(frame, args.lock_color, args.lock_min_area)
        h, w = frame.shape[:2]
        live = lock.update(cands, w)
        if live and memo["how"] == "detector":
            memo["template"] = crop_gray(frame, lock.box)
        return dict(feats=features(lock.box, w, h, args.aim_frac) if live else None, live=live, dets=dets)

    robot.get_frame(args.camera)                      # create the camera client before threads share it
    grabber = FrameGrabber(lambda: robot.get_frame(args.camera), args.workers).start()
    perc = Perception(grabber, analyse)
    perc.start()

    def pose_loop():
        """The arm's real Viam pose and joint angles a few times a second. With the kinematics known it also measures how
        far every link is from the obstacles and stops the arm if it closes in below the clearance."""
        last_slack, last_gen = None, -1
        while shared["alive"]:
            try:
                if real:
                    shared["cur"] = current()
                q = np.array(robot.get_joints(), float)
                shared["q"] = q
                g, nm = body_slack(q)
                shared["slack"] = (g, nm)
                if gctx["gen"] != last_gen:                       # the arm model changed, not the arm: start the comparison afresh
                    last_gen, last_slack = gctx["gen"], None
                moving = S["thread"] is not None and S["thread"].is_alive()
                if (real and moving and g < args.clearance_mm - 5.0 and last_slack is not None
                        and np.isfinite(last_slack) and g < last_slack - 1.0):
                    S["abort"].set()
                    robot.stop()
                    say(f"Guard: the arm is {g:.0f} mm from the {nm} (the limit is {args.clearance_mm:.0f} mm). Stopped.")
                last_slack = g
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
    wrist_targets = {} if args.no_orient else dict(zip((4, 5), floats(args.approach_wrist)))
    args.grip_force = min(max(args.grip_force, 0.0), 3.0)                      # never more than 3
    cfg = {"tip_off_cm": args.tip_offset_mm / 10.0, "stop_cm": args.stop_distance_mm / 10.0,
           "user_stop_cm": args.stop_distance_mm / 10.0, "open_cm": args.open_at_cm, "slow_cm": args.slow_below_cm,
           "fast": args.fast_speed, "slow": args.slow_speed, "open_gripper": True, "confirm": args.confirm_plan,
           "checkpoint": True, "kind": "reach"}

    # ---- shared state between the display loop and the mission thread -----------------------------------------------
    mu = threading.Lock()
    snap = {"x": None, "t": 0.0, "tip": None, "dist": None, "clipped": False, "hpx": None}
    S = {"phase": "idle", "detail": "", "plan": [], "abort": threading.Event(), "confirm": threading.Event(),
         "waiting": False, "opened": False, "thread": None, "progress": None, "plan_obj": None, "plan_ctx": None,
         "holding": None, "phases": None}
    ms = {"mode": "", "locked": {}}                    # wrist orientation now ("down", "level" or "") and the joint angles held fixed
    kin_cache = {}

    def snapshot():
        with mu:
            return dict(snap)

    def J_now():
        """The camera response rescaled to the current distance: the picture shifts ~1/distance per mm of sideways motion
        (and the object's size ~1/distance^2 per mm of forward motion), so J learned near the cup must not be used as is
        for a person a metre away."""
        sn = snapshot()
        if J is not None and jref["c"] and sn["dist"]:
            k = jref["c"] / (float(sn["dist"]) * 10.0)
            Jm = J.copy()
            Jm[:2] *= k
            Jm[2] *= k * k
            return Jm
        return J

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

    def down_wrist_targets():
        v = floats(args.down_wrist)
        return dict(zip((4, 5), v)) if len(v) == 2 else dict(zip((3, 4, 5), v))

    def a_nominal():
        a = np.array(floats(args.camera_axis_tool), float)
        return a / np.linalg.norm(a)

    def ddir_from(Jn):
        """The approach direction from the learned camera response. With the camera facing straight down the true direction is
        straight down: use that when the estimate is missing (a top view close to the object changes its size too little to
        measure) or points more than ~14 degrees away from it."""
        d = depth_direction(Jn)
        if ms["mode"] == "down":
            down = np.array([0.0, 0.0, -1.0])
            if d is None or float(d @ down) < 0.97:
                return down
        return d

    def lock_now():
        """Remember the angles of the joints to hold fixed (e.g. joint 5) at this moment."""
        if args.free_roll or ms["mode"] != "down":
            ms["locked"] = {}
            return
        q = robot.get_joints() if real else None
        ms["locked"] = {} if q is None else {int(k): float(q[int(k)]) for k in args.lock_joints.split(",") if k.strip()}

    def lock_active():
        return bool(ms["locked"]) and ms["mode"] == "down" and kin_cache.get("v") is not None and kin_cache["v"][0] is not None

    def locked_ik(pos, q_seed):
        """Joints that put the camera at `pos` (robot frame) pointing straight down with the locked joints unchanged, or None."""
        res = kin_cache["v"][0].face_down_ik(np.array(robot.table.apply(*pos)), a_nominal(), q_seed, ms["locked"])
        if res is None or res[1] > args.max_tilt_deg or res[2] > 1.0:
            return None
        return res[0]

    def goto(pos):
        """Move the camera to pos and wait. With the camera facing down this is a joint-space move that leaves the locked
        joints (joint 5) exactly as they are; otherwise the arm's own orientation-holding move."""
        if lock_active():
            q_now = np.array(robot.get_joints(), float)
            q_t = locked_ik(pos, q_now)
            if q_t is not None:
                n = max(2, int(np.max(np.abs(q_t - q_now)) / 4.0))
                run_joint_move(smooth_path(q_now, q_t, n), args.joint_move_speed)
                return
            say("The locked-joint solution failed for this small move; using the arm's own move.")
        robot.move_xyz(*pos)

    def move_blocking(pos):
        """Move there and wait (real arm), or just pretend (dry run)."""
        if real:
            goto(pos)
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
        sn_ref = snapshot()
        jref["c"] = None if sn_ref["dist"] is None else float(sn_ref["dist"]) * 10.0
        s0, p0, after, deltas = r0.feats, current(), [], []
        for i in range(3):
            tgt = p0.copy()
            d = args.probe_mm if p0[i] + args.probe_mm <= hi[i] else -args.probe_mm
            tgt[i] += d
            tgt = np.clip(tgt, np.minimum(lo, p0), np.maximum(hi, p0))
            safe, hit = obs.constrain(p0, tgt)
            safe, why = limit_step(p0, safe)
            if hit or why.startswith("blocked"):
                say(f"The learning move along {'xyz'[i]} is not safe ({'it would enter ' + str(hit) if hit else why}); "
                    "move the arm clear and try again.")
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
        json.dump({"J": Jn.tolist(), "probe_mm": args.probe_mm, "c_mm": jref["c"]}, open(JAC_FILE, "w"), indent=2)
        print("learned camera response J (frame fraction per mm; rows u, v, h; columns x, y, z):\n", np.round(Jn, 5), flush=True)
        return Jn

    # ---- mission: centre -> plan -> execute -----------------------------------------------------------------------
    def centre():
        """Camera-guided steps until the object is within --deadband of the middle (a few frames in a row). The rest of
        the offset is folded into the plan, so the gripper does not wait for a perfect centre. Checks that each step
        moved the picture the way the learned response J predicted and re-learns J if it did not."""
        nonlocal J, ddir
        S.update(phase="centring", detail="")
        t0, stable, lost, steps, bad, reprobes, stuck = time.monotonic(), 0, 0, 0, 0, 0, 0
        gain, prev_e, worse, verified, back_pos = args.gain, None, 0, False, None
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
                if lost >= 2 and back_pos is not None and real:       # the last step took it out of view: undo it
                    say(f"I lost sight of the {det['name']} after that step; going back to where I saw it.")
                    last_end = move_blocking(back_pos)
                    back_pos, bad, prev, lost = None, 2, None, 0       # and re-learn how the picture responds
                    continue
                if lost >= 3:
                    say(f"I lost sight of the {det['name']} while centring. Stopping.")
                    return False
                continue
            lost = 0
            if S.get("switch_t", 0.0) > last_end - 1e-6:                   # the user switched objects: the picture jumped
                prev, back_pos, bad, prev_e, worse = None, None, 0, None, 0
            s = sn["x"]
            e = float(np.max(np.abs(target[:2] - s[:2])))
            S["detail"] = f"offset u{s[0]:+.2f} v{s[1]:+.2f} (need < {args.deadband:.2f})"
            if prev is not None and real:                 # did the last move do what J said it would?
                s_before, d_prev = prev
                pred, obs_ = (J_now() @ d_prev)[:2], (s - s_before)[:2]
                if np.linalg.norm(pred) > 0.012 and np.linalg.norm(obs_) > 0.006:
                    cos = float(pred @ obs_ / (np.linalg.norm(pred) * np.linalg.norm(obs_)))
                    ratio = float(np.linalg.norm(obs_) / np.linalg.norm(pred))
                    good = cos > 0.6 and 0.4 < ratio < 2.5
                    verified = verified or good
                    bad = bad + 1 if (cos < 0.3 or not 0.3 < ratio < 3.0) else 0
                elif np.linalg.norm(pred) > 0.04 and np.linalg.norm(obs_) < 0.005:
                    bad += 1                              # moved a lot and the picture did not change
                if bad >= 2 and reprobes < 2:
                    say("The picture is not moving the way my camera model predicts, so I'll re-learn it.")
                    Jn = probe()
                    if Jn is None:
                        return False
                    J, ddir, bad, reprobes, prev = Jn, ddir_from(Jn), 0, reprobes + 1, None
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
            cap = args.max_step if (verified or e < 0.2) else 25.0       # far off-centre: a short trial step first
            d = servo_step(J_now(), s, target, (0, 1), gain, cap)
            cur = shared["cur"]
            new = np.clip(cur + d, np.minimum(lo, cur), np.maximum(hi, cur))
            safe, hit = obs.constrain(cur, new)
            if hit:
                say(f"Centring would enter {hit}; stopping short of it.")
            safe, why = limit_step(cur, safe)
            if why:
                say(f"Guard: this centring step is {why}")
            steps += 1
            if np.linalg.norm(safe - cur) < 1.0:              # the step vanished: the camera model or the limits block progress
                stuck += 1
                if stuck >= 3:
                    if real and reprobes < 2:
                        say("Centring is not making progress, so I'll re-learn the camera response.")
                        Jn = probe()
                        if Jn is None:
                            return False
                        J, ddir, stuck, reprobes, prev = Jn, ddir_from(Jn), 0, reprobes + 1, None
                        last_end = time.monotonic()
                        continue
                    say("Centring is not making progress (the move it wants is blocked or below 1 mm). Stopping.")
                    return False
            else:
                stuck = 0
            if steps % 3 == 1:
                say(f"Centring: object {100 * e:.0f}% off-centre, moving {np.linalg.norm(safe - cur):.0f} mm.")
            if not real:
                say(f"Dry run: I would centre it with a move of {d[0]:+.0f} {d[1]:+.0f} {d[2]:+.0f} mm "
                    f"(offset u{s[0]:+.2f} v{s[1]:+.2f}). Assuming it is centred and planning from here.")
                return True
            prev, back_pos = (s.copy(), safe - cur), cur.copy()
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
        safe, why = limit_step(p_c, safe)
        if hit or why or np.linalg.norm(safe - p_c) < step * 0.8:
            return
        t_end = move_blocking(safe)
        sn1 = wait_fresh(t_end + args.settle, 2.0)
        move_blocking(p_c)
        time.sleep(args.settle)
        if sn1 is None:
            return
        # moving toward an off-centre object pushes it outward in the picture by itself (perspective); take that out
        drift = (sn1["x"][:2] - s0[:2]) - s0[:2] * (step / max(c0_mm, 50.0))
        corr = np.linalg.pinv(J_now()[:2]) @ drift
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
                kin_cache["v"] = (load_validated(robot, dump_dir=os.path.join(ROOT, "data")) if real
                                  else (None, "dry run: no kinematics check"))
            print("kinematics:", kin_cache["v"][1], flush=True)
        return kin_cache["v"]

    # ---- body guard: every link of the arm stays >= --clearance-mm from the obstacles ------------------------------
    gctx = {"a_tool": None, "gen": 0}

    def kin_if_ready():
        v = kin_cache.get("v")
        return v[0] if v else None

    def refresh_body():
        """Remember the camera/gripper axis in the tool frame (from the learned approach direction)."""
        kin = kin_if_ready()
        if kin is None or ddir is None:
            return
        to_arm, cam = robot.table.apply, shared["cur"]
        d_arm = np.array(to_arm(*(cam + ddir))) - np.array(to_arm(*cam))
        gctx["a_tool"] = kin.fk(np.array(robot.get_joints(), float))[1].T @ d_arm
        gctx["gen"] += 1

    def body_slack(q):
        """(gap in mm beyond the arm's own radius, nearest obstacle) for joint angles q; (inf, '') if it cannot be checked."""
        kin = kin_if_ready()
        if kin is None or args.no_body_guard or not obs.boxes:
            return float("inf"), ""
        pts = kin.link_points(q)
        _, R = kin.fk(q)
        axis = R @ (gctx["a_tool"] if gctx["a_tool"] is not None else np.array([0.0, 0.0, 1.0]))
        pts = pts + [pts[-1] + axis * args.gripper_length_mm]                      # the gripper reaches past the tool point
        return obs.arm_clearance([np.array(robot.table.to_robot(*p)) for p in pts], args.arm_radius_mm)

    def body_ok(q_from, q_to, n=8):
        """(ok, worst gap, obstacle) along the straight joint-space path: clear by the clearance, or at least no closer
        than the arm already is."""
        s0, _ = body_slack(q_from)
        floor = min(args.clearance_mm, s0 - 1.0)
        worst, name = float("inf"), ""
        for t in np.linspace(0.0, 1.0, n + 1)[1:]:
            g, nm = body_slack(q_from + t * (q_to - q_from))
            if g < worst:
                worst, name = g, nm
        return worst >= floor, worst, name

    def limit_step(cur, new):
        """A Cartesian move, shortened if the arm's body would come within the clearance of an obstacle on the way.
        -> (position, note); the note is empty when the full move is fine."""
        kin = kin_if_ready()
        if kin is None or args.no_body_guard or not real:
            return new, ""
        q_now = np.array(robot.get_joints(), float)
        R = kin.fk(q_now)[1]
        worst, name = float("inf"), ""
        for sc in (1.0, 0.5, 0.25):
            cand = cur + (new - cur) * sc
            q_t = locked_ik(cand, q_now) if lock_active() else kin.ik(np.array(robot.table.apply(*cand)), R, q_now)
            if q_t is None:
                return cand, ""                                                     # unknown: the arm's own planner decides
            ok, worst, name = body_ok(q_now, q_t)
            if ok:
                return cand, ("" if sc == 1.0 else f"shortened to {int(sc * 100)}%: the arm would come within {worst:.0f} mm of {name}")
        return cur, f"blocked: the arm would come within {worst:.0f} mm of the {name} (limit {args.clearance_mm:.0f} mm)"

    def limit_plan_by_body(plan):
        """Cut the plan where the arm's body (not just the tool) would get closer than the clearance."""
        q_prev = np.array(robot.get_joints(), float)
        s0, _ = body_slack(q_prev)
        floor = min(args.clearance_mm, s0 - 1.0)
        for i, w in enumerate(plan.waypoints):
            g, nm = body_slack(np.array(w.joints, float))
            if g < floor:
                plan.waypoints = plan.waypoints[:max(i, 1)]
                plan.travel_mm, plan.ok = plan.waypoints[-1].s_mm, False
                plan.notes.append(f"the arm would come within {g:.0f} mm of the {nm} after {w.s_mm:.0f} mm "
                                  f"(limit {args.clearance_mm:.0f} mm): stopping the plan there")
                qs = np.array([x.joints for x in plan.waypoints], float)
                plan.joint_travel_deg = np.sum(np.abs(np.diff(qs, axis=0)), axis=0) if len(qs) > 1 else np.zeros(qs.shape[1])
                break

    def attach_and_guard(plan):
        """Solve the plan in joint space (minimum joint movement) and cut it where the arm's body would get too close to an
        obstacle; without validated kinematics it stays a Cartesian plan and says so."""
        kin, note = get_kin()
        if kin is not None:
            q_start = np.array(robot.get_joints(), float)
            solved = (attach_joints_down(plan, kin, q_start, robot.table.apply, a_nominal(), set(ms["locked"]),
                                         args.max_joint_jump, args.max_tilt_deg) if lock_active()
                      else attach_joints(plan, kin, q_start, robot.table.apply, args.max_joint_jump))
            if not solved and lock_active():                      # locked solution impossible: fall back, keep the camera down
                solved = attach_joints(plan, kin, q_start, robot.table.apply, args.max_joint_jump)
            if solved:
                limit_plan_by_body(plan)
        else:
            plan.notes.append(note + "; using Cartesian moves")

    def make_plan(tip, lateral=None):
        p0 = shared["cur"].copy()
        plan = build_plan(p0, ddir, tip, cfg["stop_cm"], cfg["open_cm"], cfg["slow_cm"], cfg["fast"], cfg["slow"],
                          obstacles=obs, bounds=bounds, step_mm=args.plan_step_mm, max_travel_mm=args.max_travel_mm,
                          lateral=lateral)
        if not cfg["open_gripper"]:
            plan.open_s_mm = 1e9                                  # holding something: the gripper stays closed
        if lateral is not None and float(np.linalg.norm(lateral)) > 2.0:
            plan.notes.append(f"includes a {float(np.linalg.norm(lateral)):.0f} mm sideways correction for the leftover centring error")
        attach_and_guard(plan)
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
        split = cfg["open_cm"] if (cfg["open_gripper"] and cfg["checkpoint"]) else None
        segs = plan.segments(cfg["slow_cm"], split_cm=split)
        for i, (name, a, b, v) in enumerate(segs):
            say(f"Leg {i + 1}/{len(segs)} ({name}): {plan.waypoints[a].tip_cm:.1f} -> {plan.waypoints[b].tip_cm:.1f} cm at ~{v:.0f} mm/s")
            run_segment(plan, a, b, v)
            if not real or i + 1 >= len(segs):
                continue
            if name == "approach":                                       # at the open distance: look, MEMORISE the object
                time.sleep(0.4)
                sn = wait_fresh(time.monotonic(), 2.0)
                if sn is None or sn["clipped"]:
                    say(f"I can't see the whole {det['name']} here, so I keep the position measured earlier as its memory.")
                    continue
                m, _ = measure_tip(n=6)
                if m is not None:
                    say(f"Locked on the {det['name']} at {m:.1f} cm: this position is remembered for the last part, when the "
                        "object will fill the picture. Re-planning from it.")
                    return ("replan", m)
            elif name == "fast":                                         # before the slow leg: is reality where the plan thought?
                time.sleep(0.4)
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

    def run_joint_move(q_list, vmax):
        """One joint-space trajectory, watched for an abort."""
        fut = robot.move_through_joints(q_list, vmax, vmax * args.joint_accel, wait=False)
        while not fut.done():
            if S["abort"].is_set():
                robot.stop()
                raise Abort()
            time.sleep(0.05)
        err = fut.exception()
        if err and "cancel" not in f"{err!r}".lower():
            raise RuntimeError(f"the arm refused the wrist move: {err}")

    def tilt_deg(q):
        """Elevation of the camera/gripper axis above the horizontal, from the kinematics (None if unknown)."""
        kin = kin_if_ready()
        a = a_nominal() if ms["mode"] == "down" else gctx["a_tool"]
        if kin is None or a is None:
            return None
        return float(np.degrees(np.arcsin(np.clip((kin.fk(np.asarray(q, float))[1] @ a)[2], -1, 1))))

    def fmt_q(q):
        return " ".join(f"J{i} {v:+.1f}" for i, v in enumerate(q))

    def orient_wrist(tip, mode):
        """Turn the wrist (joints 0-4 adjusting, joint 5 to its target by the shortest turn) so that the camera
          mode 'down':  looks straight DOWN at the object from --top-height-mm above it (top grasp), or
          mode 'level': is horizontal with joints 4 and 5 at --approach-wrist (side approach, looking for a person).
        With tip (the measured distance) the object stays in view; tip=None (nothing to look at) turns in place.
        Eased in and out, checked against bounds and the arm-body obstacle guard. -> 'done' (re-centre next), 'skipped', or False."""
        nonlocal J, ddir
        q_now = np.array(robot.get_joints() if real else [0.0] * 6, float)
        kin, note = get_kin() if real else (None, "dry run")
        limits = kin.limits if kin is not None else np.array([[-360.0, 360.0]] * 6)
        raw = {5: args.top_wrist_roll} if mode == "down" else dict(wrist_targets)
        targets = {k: nearest_equivalent(v, q_now[k], tuple(limits[k])) for k, v in raw.items()}
        names = ", ".join(f"J{k} {q_now[k]:.1f} -> {v:g}" + (f" (same orientation as {raw[k]:g}, the nearest way round)"
                                                            if abs(v - raw[k]) > 1e-6 else "") for k, v in targets.items())
        what = ("camera looking straight down at the object" if mode == "down" else "gripper level")
        if not real:
            say(f"Dry run: I would turn the wrist ({what}; {names}) before the approach.")
            return "skipped"
        if kin is None:
            say(f"I can't turn the wrist safely without the arm's kinematics ({note}), so I'll approach with the "
                "current wrist angles.")
            return "skipped"
        to_arm = robot.table.apply
        cam = shared["cur"].copy()
        d_arm = np.array(to_arm(*(cam + ddir))) - np.array(to_arm(*cam))
        _, R_now = kin.fk(q_now)
        a_tool = R_now.T @ d_arm                                  # the camera axis, fixed in the tool frame
        axis_now = R_now @ a_tool
        if mode == "down" and axis_now[2] < -0.995 and all(abs(q_now[k] - v) < 2.0 for k, v in targets.items()):
            return "skipped"
        if mode == "level" and all(abs(q_now[k] - v) < 2.0 for k, v in targets.items()) and abs(axis_now[2]) < 0.03:
            return "skipped"
        if tip is not None:
            sn = snapshot()
            lat = servo_step(J_now(), sn["x"], target, (0, 1), 1.0, 80.0) if sn["x"] is not None else np.zeros(3)
            c_now = tip * 10.0 + args.tip_offset_mm
            p_obj = np.array(to_arm(*(cam + lat + ddir * c_now)))
            c0 = args.top_height_mm if mode == "down" else c_now
        else:                                                      # nothing in view: keep the camera where it is, change its heading
            hx = np.array([axis_now[0], axis_now[1]])
            hx = hx / np.linalg.norm(hx) if np.linalg.norm(hx) > 0.2 else np.array([cam[0], cam[1]]) / max(np.linalg.norm(cam[:2]), 1.0)
            c0 = 600.0
            p_obj = np.array(to_arm(*cam)) + c0 * np.array([hx[0], hx[1], 0.0])
        q_new = kin.look_at_ik_multi(p_obj, c0, a_tool, q_now, targets, level=(mode == "level"), down=(mode == "down"))
        if q_new is None:
            say(f"No arm pose has the wrist at ({names}) with the {what} and the object in view from here. "
                "Move the arm so the object is nearer the middle of the workspace, then try again. "
                "I'll approach with the current wrist angles.")
            return "skipped"
        tilt = float(np.degrees(np.arcsin(np.clip((kin.fk(q_new)[1] @ a_tool)[2], -1, 1))))
        dq = q_new - q_now
        was_blocked = bool(obs.blocking(cam))
        for t in np.linspace(0.0, 1.0, 30):                       # the swept tool point must stay inside bounds and clear
            pos = np.array(robot.table.to_robot(*kin.fk(q_now + t * dq)[0]))
            if not was_blocked and obs.blocking(pos) or np.any(pos < np.minimum(lo, cam) - 1) or np.any(pos > np.maximum(hi, cam) + 1):
                say(f"Turning the wrist would swing the gripper outside the safe area (near x{pos[0]:.0f} y{pos[1]:.0f} "
                    f"z{pos[2]:.0f}). I won't do it; stopping.")
                return False
        ok, worst, nm = body_ok(q_now, q_new, n=30)               # and so must every link of the arm
        if not ok:
            say(f"Turning the wrist would bring the arm within {worst:.0f} mm of the {nm} (limit {args.clearance_mm:.0f} mm). "
                "I won't do it; stopping.")
            return False
        cam_after = np.array(robot.table.to_robot(*kin.fk(q_new)[0]))
        S["plan"] = [f"Wrist turn before the approach ({what}, axis elevation {tilt:+.1f} deg):", "  " + names,
                     f"  camera moves from x{cam[0]:.0f} y{cam[1]:.0f} z{cam[2]:.0f} to x{cam_after[0]:.0f} y{cam_after[1]:.0f} z{cam_after[2]:.0f}",
                     "  now    " + fmt_q(q_now), "  after  " + fmt_q(q_new), "  change " + fmt_q(dq)]
        say("Turning the wrist for the approach: " + what + ". " + names + f".\n  camera {cam.round(0).tolist()} -> {cam_after.round(0).tolist()}"
            f"\n  now    {fmt_q(q_now)}\n  after  {fmt_q(q_new)}\n"
            f"  change {fmt_q(dq)}  (gripper axis elevation {tilt:+.1f} deg, eased in and out at {args.orient_speed:g} deg/s)")
        big = [k for k in range(len(dq)) if abs(dq[k]) > args.orient_confirm_deg]
        if big:
            S.update(phase="waiting for yes", waiting=True)
            S["confirm"].clear()
            say(f"Joint {big[0]} would turn {abs(dq[big[0]]):.0f} deg (more than {args.orient_confirm_deg:.0f}). "
                "Say 'yes' to allow it, or 'stop'.")
            while not S["confirm"].wait(0.1):
                check()
            S["waiting"] = False
        S.update(phase="turning wrist", detail="")
        run_joint_move(smooth_path(q_now, q_new, 14), args.orient_speed)
        shared["cur"] = current()
        say(f"Wrist turned: now {fmt_q(np.array(robot.get_joints(), float))}")
        time.sleep(args.settle)
        ms["mode"] = mode
        lock_now()
        set_range_model(cfg["kind"])
        if tip is None:
            refresh_body()
            S["plan"] = []
            return "done"
        Jn = probe()                                              # the camera looks a different way now: re-learn
        if Jn is None:
            say("I lost the object after turning the wrist. Stopping.")
            return False
        J, ddir = Jn, ddir_from(Jn)
        refresh_body()
        S["plan"] = []
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
        safe, why = limit_step(cur, safe)
        if hit or why:
            return
        S["detail"] = f"trimming {err_mm:+.0f} mm"
        say(f"The camera estimates {m:.1f} cm; moving {err_mm:+.0f} mm along the line to reach {cfg['stop_cm']:.1f} cm.")
        move_blocking(safe)

    def face_down():
        """Start with the camera facing straight down: the wrist joints 3-5 go to --down-wrist and joints 0-2 are solved so
        the camera axis is vertical while the camera stays as close as possible to where it is. Eased, bounds- and
        body-checked. The moves that follow hold this orientation (the camera keeps looking down). -> False only on a refusal."""
        wrist = down_wrist_targets()
        a_tool = np.array(floats(args.camera_axis_tool), float)
        a_tool = a_tool / np.linalg.norm(a_tool)
        if not real:
            say("Dry run: I would start by turning the camera to face straight down "
                f"({', '.join(f'J{k} {v:g}' for k, v in wrist.items())}).")
            ms["mode"] = "down"
            return True
        kin, note = get_kin()
        if kin is None:
            say(f"I can't start with the camera facing down without the arm's kinematics ({note}); continuing as the arm is.")
            return True
        q_now = np.array(robot.get_joints(), float)
        targets = {k: nearest_equivalent(v, q_now[k], tuple(kin.limits[k])) for k, v in wrist.items()}
        _, R_now = kin.fk(q_now)
        tilt_now = float(np.degrees(np.arccos(np.clip(-(R_now @ a_tool)[2], -1, 1))))
        if tilt_now < 1.0 and all(abs(q_now[k] - v) < 2.0 for k, v in targets.items()):
            gctx["a_tool"] = a_tool
            ms["mode"] = "down"
            lock_now()
            say(f"The camera already faces down ({tilt_now:.1f} deg from vertical)"
                + (f"; joint(s) {args.lock_joints} are held at their current angle from here." if ms["locked"] else "."))
            return True
        p_now = kin.fk(q_now)[0]
        res = kin.face_down_ik_multi(p_now, a_tool, q_now, targets)
        if res is None:
            say("No arm posture has the wrist at the camera-down angles with the camera pointing straight down near here. "
                "Move the arm closer to above the work area and try again; continuing as the arm is.")
            return True
        q_new, tilt, err = res
        dq = q_new - q_now
        cam_now = np.array(robot.table.to_robot(*p_now))
        cam_new = np.array(robot.table.to_robot(*kin.fk(q_new)[0]))
        was_blocked = bool(obs.blocking(cam_now))
        for t in np.linspace(0.0, 1.0, 30):
            pos = np.array(robot.table.to_robot(*kin.fk(q_now + t * dq)[0]))
            if not was_blocked and obs.blocking(pos) or np.any(pos < np.minimum(lo, cam_now) - 1) or np.any(pos > np.maximum(hi, cam_now) + 1):
                say(f"Turning the camera down would swing the gripper outside the safe area (near x{pos[0]:.0f} y{pos[1]:.0f} "
                    f"z{pos[2]:.0f}). I won't do it; stopping.")
                return False
        gctx["a_tool"] = a_tool
        ok, worst, nm = body_ok(q_now, q_new, n=30)
        if not ok:
            say(f"Turning the camera down would bring the arm within {worst:.0f} mm of the {nm} (limit {args.clearance_mm:.0f} mm). "
                "I won't do it; stopping.")
            return False
        names = ", ".join(f"J{k} {q_now[k]:.1f} -> {v:g}" for k, v in targets.items())
        S["plan"] = [f"Start: camera facing straight down (tilt {tilt:.1f} deg from vertical): {names}",
                     f"  camera moves {err:.0f} mm: x{cam_now[0]:.0f} y{cam_now[1]:.0f} z{cam_now[2]:.0f} -> "
                     f"x{cam_new[0]:.0f} y{cam_new[1]:.0f} z{cam_new[2]:.0f}",
                     "  now    " + fmt_q(q_now), "  after  " + fmt_q(q_new), "  change " + fmt_q(dq)]
        say("Starting with the camera facing straight down: " + names + f".\n  camera {cam_now.round(0).tolist()} -> {cam_new.round(0).tolist()}"
            f"\n  now    {fmt_q(q_now)}\n  after  {fmt_q(q_new)}\n  change {fmt_q(dq)}  (tilt from vertical {tilt:.1f} deg)")
        big = [k for k in range(len(dq)) if abs(dq[k]) > args.orient_confirm_deg]
        if big:
            S.update(phase="waiting for yes", waiting=True)
            S["confirm"].clear()
            say(f"Joint {big[0]} would turn {abs(dq[big[0]]):.0f} deg (more than {args.orient_confirm_deg:.0f}). "
                "Say 'yes' to allow it, or 'stop'.")
            while not S["confirm"].wait(0.1):
                check()
            S["waiting"] = False
        S.update(phase="turning the camera down", detail="")
        run_joint_move(smooth_path(q_now, q_new, 14), args.orient_speed)
        shared["cur"] = current()
        time.sleep(args.settle)
        say(f"Camera facing down: now {fmt_q(np.array(robot.get_joints(), float))}")
        gctx["gen"] += 1
        S["plan"] = []
        ms["mode"] = "down"
        lock_now()
        say(f"Holding joint(s) {args.lock_joints} at their current angle for every move from here, so the camera keeps facing down."
            if ms["locked"] else "No joints are locked; the arm's own moves keep the camera orientation.")
        return True

    def raise_up(mm, title="Raise"):
        """Straight up by mm, planned in joint space like every other move and shown the same way."""
        S.update(phase="planning the raise", detail="")
        p0 = shared["cur"].copy()
        plan = line_plan(p0, (0.0, 0.0, 1.0), mm, args.lift_speed, obstacles=obs, bounds=bounds, step_mm=args.plan_step_mm)
        attach_and_guard(plan)
        S["progress"] = 0.0
        S["plan_obj"] = plan
        S["plan_ctx"] = {"c0_mm": 0.0, "p_obj": plan.waypoints[-1].pos, "box": (600.0, 300.0, 680.0, 420.0)}
        lines = describe_line(plan, args.lift_speed, title)
        S["plan"] = lines
        say("\n".join(lines))
        if plan.travel_mm < 5:
            say("There is no room to raise the arm; stopping.")
            return False
        for k in range(int(args.countdown * 10), 0, -1):
            S.update(phase=f"raising in {k / 10:.1f}s (say stop to cancel)")
            check()
            time.sleep(0.1)
        S["phase"] = "raising"
        run_segment(plan, 0, len(plan.waypoints) - 1, args.lift_speed)
        time.sleep(args.settle)
        return True

    def turn_camera_down_above(p_obj):
        """Turn the wrist to the camera-down posture (--down-wrist, e.g. joint 4 = 15.29, joint 5 = -181.92) with the other joints
        solved so the camera points straight down from directly ABOVE the object (p_obj, robot frame). Its height is whatever the
        arm geometry gives for that wrist; it must be inside --top-height-range. Eased, bounds- and body-checked. Then the camera
        response is re-learned (the picture is now a top view). -> True / False."""
        nonlocal J, ddir
        wrist = down_wrist_targets()
        a_tool = a_nominal()
        wnames = ", ".join(f"J{k} {v:g}" for k, v in wrist.items())
        if not real:
            say(f"Dry run: I would raise {args.turn_down_lift_mm:g} mm, then turn the camera down above the object ({wnames}).")
            ms["mode"] = "down"
            return True
        kin, note = get_kin()
        if kin is None:
            say(f"I can't turn the camera down without the arm's kinematics ({note}); stopping.")
            return False
        q_now = np.array(robot.get_joints(), float)
        targets = {k: nearest_equivalent(v, q_now[k], tuple(kin.limits[k])) for k, v in wrist.items()}
        obj_arm = np.array(robot.table.apply(*p_obj))
        zmin, zmax = floats(args.top_height_range)
        rng_ok = (obj_arm[2] + zmin, obj_arm[2] + zmax)
        res = kin.face_down_above_ik_multi(obj_arm[:2], a_tool, q_now, targets, z_range=rng_ok)
        note_fb = ""
        if res is None and 4 in targets:
            # that exact wrist cannot do it here: hold only joint 5 and take the solution whose joint 4 is nearest the wished value
            free4 = {5: targets[5]} if 5 in targets else {}
            res = kin.face_down_above_ik_multi(obj_arm[:2], a_tool, q_now, free4, tries=50, z_range=rng_ok, prefer={4: targets[4]})
            if res is not None:
                note_fb = (f" (joint 4 = {targets[4]:g} cannot look straight down above the object from here; using the nearest "
                           f"possible, joint 4 = {res[0][4]:.2f})")
        if res is None:                                            # maybe only the height window is the problem
            res = kin.face_down_above_ik_multi(obj_arm[:2], a_tool, q_now, {5: targets[5]} if 5 in targets else {}, tries=50,
                                               z_range=(obj_arm[2] + 30.0, obj_arm[2] + 700.0), prefer={4: targets.get(4, q_now[4])})
            if res is not None:
                note_fb = (f" (camera height {res[3] - obj_arm[2]:.0f} mm is outside the preferred {zmin:g}-{zmax:g} mm; "
                           f"joint 4 = {res[0][4]:.2f})")
        if res is None:
            radius = float(np.hypot(obj_arm[0], obj_arm[1]))
            az = obj_arm[:2] / max(radius, 1e-6)
            table_ = []
            for r_ in range(250, 751, 50):                         # what this wrist CAN do, for the object's bearing
                rr = kin.face_down_above_ik_multi(az * r_, a_tool, q_now, targets, tries=6)
                if rr is not None:
                    table_.append(f"{r_}->{rr[3] - obj_arm[2]:.0f}")
            say(f"No arm posture with the wrist at ({wnames}) puts the camera straight above the object. The object is estimated at "
                f"x{p_obj[0]:.0f} y{p_obj[1]:.0f} z{p_obj[2]:.0f} (robot frame), {radius:.0f} mm from the base axis. For this wrist and bearing "
                "the camera can look straight down only at these distances from the base (mm -> height above the object, mm): "
                + (", ".join(table_) or "none") + ". Move the object into that range, or change --down-wrist. Stopping.")
            return False
        q_new, tilt, xy_err, z_cam = res
        dq = q_new - q_now
        cam_now = np.array(robot.table.to_robot(*kin.fk(q_now)[0]))
        cam_new = np.array(robot.table.to_robot(*kin.fk(q_new)[0]))
        for t in np.linspace(0.0, 1.0, 30):
            pos = np.array(robot.table.to_robot(*kin.fk(q_now + t * dq)[0]))
            if (obs.blocking(pos) and not obs.blocking(cam_now)) or np.any(pos < np.minimum(lo, cam_now) - 1) or np.any(pos > np.maximum(hi, cam_now) + 1):
                say(f"Turning the camera down would swing the gripper outside the safe area (near x{pos[0]:.0f} y{pos[1]:.0f} "
                    f"z{pos[2]:.0f}). I won't do it; stopping.")
                return False
        gctx["a_tool"] = a_tool
        ok, worst, nm = body_ok(q_now, q_new, n=30)
        if not ok:
            say(f"Turning the camera down would bring the arm within {worst:.0f} mm of the {nm} (limit {args.clearance_mm:.0f} mm). Stopping.")
            return False
        names = ", ".join(f"J{k} {q_now[k]:.2f} -> {q_new[k]:.2f} ({q_new[k] - q_now[k]:+.2f} deg)" for k in sorted(set(targets) | {4}))
        S["plan"] = [f"Turn the camera down above the object (tilt {tilt:.2f} deg from vertical, {z_cam - obj_arm[2]:.0f} mm above it):",
                     "  " + names, f"  camera x{cam_now[0]:.0f} y{cam_now[1]:.0f} z{cam_now[2]:.0f} -> x{cam_new[0]:.0f} y{cam_new[1]:.0f} z{cam_new[2]:.0f}",
                     "  now    " + fmt_q(q_now), "  after  " + fmt_q(q_new), "  change " + fmt_q(dq)]
        say("Turning the camera down above the object: " + names + note_fb + f".\n  camera {cam_now.round(0).tolist()} -> {cam_new.round(0).tolist()} "
            f"({z_cam - obj_arm[2]:.0f} mm above the object)\n  now    {fmt_q(q_now)}\n  after  {fmt_q(q_new)}\n  change {fmt_q(dq)}")
        big = [k for k in range(len(dq)) if abs(dq[k]) > args.orient_confirm_deg]
        if big:
            S.update(phase="waiting for yes", waiting=True)
            S["confirm"].clear()
            say(f"Joint {big[0]} would turn {abs(dq[big[0]]):.0f} deg (more than {args.orient_confirm_deg:.0f}). Say 'yes' to allow it, or 'stop'.")
            while not S["confirm"].wait(0.1):
                check()
            S["waiting"] = False
        S.update(phase="turning the camera down", detail="")
        run_joint_move(smooth_path(q_now, q_new, 16), args.orient_speed)
        shared["cur"] = current()
        time.sleep(args.settle)
        say(f"Camera facing down: now {fmt_q(np.array(robot.get_joints(), float))}")
        ms["mode"] = "down"
        lock_now()
        gctx["gen"] += 1
        set_range_model(cfg["kind"])
        S["plan"] = []
        Jn = probe()                                              # a top view now: re-learn how the picture responds
        if Jn is None:
            say("I lost the object after turning the camera down. Stopping.")
            return False
        J, ddir = Jn, ddir_from(Jn)
        refresh_body()
        return True

    # ---- several objects of the same kind: number them, switch between them by chat or by clicking ------------------
    def candidates():
        """The detections of the target class in the latest picture, left to right."""
        r_ = perc.result
        if r_ is None:
            return []
        return sorted(object_candidates(r_.dets, det["name"]), key=lambda d: (d[0] + d[2]) / 2)

    def current_index(cands):
        with perc.mu:
            bx = lock.box.copy() if lock.locked else None
        if bx is None or not cands:
            return None
        cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
        return int(np.argmin([np.hypot((d[0] + d[2]) / 2 - cx, (d[1] + d[3]) / 2 - cy) for d in cands]))

    def pin_candidate(d):
        with perc.mu:
            lock.pin(d[:4], d[5])
            lock.enabled = True
        flt.reset()
        with mu:
            snap.update(x=None, tip=None, dist=None)
        S["switch_t"] = time.monotonic()                          # centring forgets its last step: the picture just jumped

    def busy_moving():
        ph = S["phase"]
        return (S["thread"] is not None and S["thread"].is_alive()
                and (ph in ("executing", "lifting", "raising", "gripping", "turning wrist", "turning the camera down",
                            "turning toward the target") or ph.startswith(("executing in", "raising in", "lifting in"))))

    def select_candidate(which):
        """Lock on another object of the target class. which: 'other', 'prev', 'left', 'right', 'closest', 'farthest' or 1-based number."""
        if busy_moving():
            say("The arm is moving; say 'stop' first, then pick the other one.")
            return False
        cands = candidates()
        if len(cands) < 2:
            say(f"I only see {len(cands)} {det['name']} right now, so there is nothing to switch to.")
            return False
        idx = current_index(cands)
        n = len(cands)
        if which == "other":
            k = ((idx if idx is not None else -1) + 1) % n
        elif which == "prev":
            k = ((idx if idx is not None else 0) - 1) % n
        elif which == "left":
            k = 0
        elif which == "right":
            k = n - 1
        elif which == "closest":
            k = int(np.argmax([d[3] - d[1] for d in cands]))
        elif which == "farthest":
            k = int(np.argmin([d[3] - d[1] for d in cands]))
        elif isinstance(which, int):
            if not 1 <= which <= n:
                say(f"I see {n} {det['name']}s numbered 1 to {n}; there is no number {which}.")
                return False
            k = which - 1
        else:
            return False
        pin_candidate(cands[k])
        say(f"Locked on {det['name']} #{k + 1} of {n} (numbered left to right on the video).")
        return True

    def on_click(xf, yf):
        cmdq.put({"intent": "click", "x": xf, "y": yf})

    def set_detector(name):
        """Switch the detector to `name` (loads the model if needed) and clear the lock."""
        if name != det["name"] or det["model"] is None:
            model, dargs = load_detector(name, args.model, args.prompts, args.conf, args.imgsz, standard="yolov8s.pt")
            with perc.mu:
                det.update(model=model, dargs=dargs, name=name, color=(name != "person"))
                lock.unlock()
                lock.enabled = True
                memo["template"] = None
        else:
            with perc.mu:
                det["color"] = (name != "person")
                lock.unlock()
                lock.enabled = True
        flt.reset()
        with mu:
            snap.update(x=None, tip=None, dist=None)

    def set_range_model(kind):
        """Distance estimate from the object's apparent size: its height seen from the side, its width seen from above."""
        if kind == "person":
            rng.real_mm = args.person_height_mm
        else:
            rng.real_mm = args.object_top_mm if ms["mode"] == "down" else args.object_height_mm
        rng.smooth = None

    def configure_phase(kind):
        """Speeds, stop distance, gripper policy and distance model for the kind of target."""
        cfg["kind"] = kind
        if kind == "person":
            cfg.update(stop_cm=args.person_stop_cm, open_gripper=False, fast=args.person_fast_speed,
                       slow=args.person_slow_speed, confirm=not args.no_person_confirm, checkpoint=False)
        else:
            cfg.update(stop_cm=(-args.grasp_depth_mm / 10.0) if kind == "pick" else cfg["user_stop_cm"], open_gripper=True,
                       fast=args.fast_speed, slow=args.slow_speed, confirm=args.confirm_plan, checkpoint=True)
        set_range_model(kind)

    def pan_align():
        """A target far to one side is reached by turning the base joint (J0), not by sliding the whole arm sideways. The
        picture does not move in proportion to the turn (the camera also swings on an arc around the base), so measure it:
        a trial turn gives the picture shift per degree, then each further turn uses the latest two measurements (secant
        steps, damped) until the target is near the middle. Eased, body-guarded, level gripper. Re-learns the camera
        response afterwards. -> False only if it had to give up."""
        nonlocal J, ddir
        sn = wait_fresh(time.monotonic(), 3.0)
        if sn is None or abs(sn["x"][0]) < args.pan_threshold:
            return True
        if not real:
            say(f"Dry run: the target is {100 * abs(sn['x'][0]):.0f}% off to one side; I would turn the base joint toward it.")
            return True
        kin, note = get_kin()
        if kin is None:
            return True                                   # centring will slide the arm sideways instead
        lim = kin.limits[0]

        def measure_u():
            s_ = wait_fresh(time.monotonic(), 3.0)
            return None if s_ is None else float(s_["x"][0])

        def turn_to(q_from, angle, n):
            q_to = q_from.copy()
            q_to[0] = float(np.clip(angle, lim[0], lim[1]))
            ok, worst, nm = body_ok(q_from, q_to, n=max(n, 4))
            if not ok:
                say(f"Turning the base would bring the arm within {worst:.0f} mm of the {nm} (limit {args.clearance_mm:.0f} mm); "
                    "stopping the turn here.")
                return None
            S.update(phase="turning toward the target")
            run_joint_move(smooth_path(q_from, q_to, n), args.orient_speed)
            shared["cur"] = current()
            time.sleep(args.settle)
            return q_to

        q = np.array(robot.get_joints(), float)
        a_last, u_last = float(q[0]), float(sn["x"][0])
        step = 8.0 if a_last + 8.0 <= lim[1] else -8.0
        q1 = turn_to(q, a_last + step, 4)                 # trial turn: gives the sign and size of the response
        if q1 is None:
            return True
        u1 = measure_u()
        if u1 is None:
            say("I lost sight of the target during the trial turn.")
            return False
        k, a_last, u_last, q = (u1 - u_last) / step, float(q1[0]), u1, q1
        total = abs(step)
        for _ in range(6):
            if abs(u_last) < args.pan_threshold / 2:
                break
            if abs(k) < 1e-3:
                say("Turning the base barely moves the picture, so I'll centre by sliding the arm instead.")
                break
            delta = float(np.clip(-0.8 * u_last / k, -30.0, 30.0))
            say(f"The target is {100 * abs(u_last):.0f}% off to the side: turning the base {delta:+.1f} deg toward it "
                f"(J0 {a_last:+.1f} -> {a_last + delta:+.1f}).")
            q_new = turn_to(q, a_last + delta, 10)
            if q_new is None:
                break
            u_new = measure_u()
            if u_new is None:
                say("I lost sight of the target while turning toward it.")
                return False
            moved = float(q_new[0]) - a_last
            total += abs(moved)
            if abs(moved) > 1.5 and (u_new - u_last) / moved * k > 0:      # secant update, keep the sign
                k = (u_new - u_last) / moved
            a_last, u_last, q = float(q_new[0]), u_new, q_new
        if total > 2.0:                                    # the camera heading changed: the camera response is stale
            Jn = probe()
            if Jn is None:
                say("I lost the target after turning. Stopping.")
                return False
            J, ddir = Jn, ddir_from(Jn)
            refresh_body()
        return True

    def run_phase(kind, name, pre=False):
        """Centre -> (turn the wrist once) -> plan -> show -> execute for one target. -> True if the plan reached its end.
        pre=True: only the first stage of a top grasp: approach to --turn-down-cm (no gripper action), keeping the arm's pose."""
        nonlocal J, ddir
        configure_phase(kind)
        if pre:
            cfg.update(stop_cm=args.turn_down_cm, open_gripper=False, checkpoint=False)
            say(f"Stage 1 of 2: centre the {name}, approach to {args.turn_down_cm:g} cm from it, raise {args.turn_down_lift_mm:g} mm, "
                "then turn the camera down above it and descend.")
        S.update(plan=[], waiting=False, progress=None, plan_obj=None, plan_ctx=None, opened=(kind == "person" and S["opened"]))
        tip_override, replans = None, 0
        need = "level" if (kind == "person" or args.grasp_from == "side") else "down"
        if pre:
            need = ms["mode"]                                       # keep the posture as it is for this first stage
        if kind == "person" and not args.no_orient and ms["mode"] != "level":
            res = orient_wrist(None, "level")                      # after the lift the camera looks down: face forward again
            if res is False:
                S.update(phase="idle")
                return False
            ms["mode"] = "level"
            set_range_model(kind)
            time.sleep(args.settle)
        if not pan_align():
            S.update(phase="idle")
            return False
        while True:
            if tip_override is None or not snapshot()["clipped"]:
                if not centre():
                    S.update(phase="idle")
                    return False
            if not args.no_orient and ms["mode"] != need and kind != "person":
                S.update(phase="measuring", detail="")
                tip0, _ = measure_tip()
                if tip0 is None:
                    say(f"I can't get a steady distance to the {det['name']}. Stopping.")
                    S.update(phase="idle")
                    return False
                res = orient_wrist(tip0, need)
                if res is False:
                    S.update(phase="idle")
                    return False
                ms["mode"] = need
                set_range_model(kind)
                if res == "done":
                    continue                                  # the view changed: centre again before planning
            if tip_override is None:
                S.update(phase="measuring", detail="")
                tip, spread = measure_tip()
                if tip is None:
                    say(f"I can't get a steady distance to the {det['name']}. Stopping.")
                    S.update(phase="idle")
                    return False
                if kind != "person":
                    S.update(phase="aligning", detail="")
                    refine_direction(tip * 10.0 + args.tip_offset_mm)
                    time.sleep(args.settle)
            else:
                tip, tip_override = tip_override, None
            if pre and tip <= args.turn_down_cm + 1.0:
                c0_ = tip * 10.0 + args.tip_offset_mm
                S["plan_ctx"] = {"c0_mm": c0_, "p_obj": shared["cur"] + ddir * c0_, "box": (600.0, 300.0, 680.0, 420.0)}
                say(f"Already {tip:.1f} cm from the {det['name']} (inside {args.turn_down_cm:g} cm): skipping the first approach.")
                return True
            S.update(phase="planning", detail="")
            sn_now = wait_fresh(time.monotonic(), 2.5)
            lat = np.zeros(3)
            if sn_now is not None:                            # the leftover centring error becomes part of the plan
                lat = servo_step(J_now(), sn_now["x"], target, (0, 1), 1.0, 80.0)
            plan = make_plan(tip, lat)
            lines = describe(plan, cfg["slow_cm"], cfg["fast"], cfg["slow"], cfg["open_cm"], args.tip_offset_mm)
            S["plan"] = lines
            with perc.mu:
                bx = lock.box.copy() if lock.locked else None
            c0 = tip * 10.0 + args.tip_offset_mm
            S["plan_ctx"] = {"c0_mm": c0, "p_obj": plan.p0 + plan.direction * c0,
                             "box": tuple(float(v) for v in bx) if bx is not None else (600.0, 300.0, 680.0, 420.0)}
            S["progress"] = 0.0
            S["plan_obj"] = plan                              # the video overlay draws this live
            say("\n".join(lines))
            if plan.travel_mm < 5:
                if plan.ok:
                    say("Already at the stop distance: nothing to move.")
                    break                                       # treat it as arrived
                say("I won't execute this plan: it is blocked right away. Widen --bounds, or move the arm so the object is "
                    "within reach (see the notes above).")
                S.update(phase="idle")
                return False
            if not plan.ok:
                say(f"Heads up: the plan stops {plan.final_tip_cm:.1f} cm from the {det['name']}, short of the "
                    f"{cfg['stop_cm']:.1f} cm target, because of the workspace/obstacle limit above. "
                    "I'll execute the shorter plan (widen --bounds if that area is safe).")
            if cfg["confirm"]:
                S.update(phase="waiting for yes", waiting=True)
                S["confirm"].clear()
                say("Say 'yes' to execute the plan, or 'stop'." if kind != "person" else
                    "I will move toward the closest person with the object. Check the area is clear, then say 'yes' (or 'stop').")
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
                return False
            if isinstance(result, tuple):
                tip_override = result[1]
        if kind == "reach" and not pre:
            trim()
        m, _ = measure_tip(n=4, timeout=2.0) if kind == "reach" else (None, 0.0)
        S.update(phase="arrived", detail="")
        say((f"Arrived: stopped {max(plan.final_tip_cm, cfg['stop_cm']):.1f} cm from the {det['name']}" if plan.ok else
             f"Stopped at the workspace limit, {plan.final_tip_cm:.1f} cm from the {det['name']} (the target was {cfg['stop_cm']:.1f} cm)")
            + (f" (the camera now estimates {m:.1f} cm)." if (m is not None and real) else "."))
        return plan.ok

    def grasp():
        """Close the gripper with a limited force and check, from the gripper and the picture, that it holds the object."""
        S.update(phase="gripping", detail="", holding=None)
        sn = snapshot()
        fill = None if sn["x"] is None else sn["x"][2]
        say(f"At the object: closing the gripper with force {args.grip_force:g} (limit 3)."
            + ("" if fill is None else f" The camera sees the object filling {100 * fill:.0f}% of the picture height."))
        if not real:
            say("Dry run: the gripper would close here and report whether it holds the object.")
            S["opened"], S["holding"] = False, True
            return True
        cmd = json.loads(args.grip_do_command) if args.grip_do_command else None
        ok = robot.grab(args.grip_force, cmd)
        S["opened"] = False
        holding = robot.is_holding()
        t_end = time.monotonic() + 2.0
        while holding is not True and time.monotonic() < t_end:    # give the fingers a moment to settle on the object
            check()
            time.sleep(0.15)
            holding = robot.is_holding()
        S["holding"] = holding if holding is not None else (True if ok else None)
        if holding is False or (holding is None and not ok):
            say("The gripper did not close on anything (it reports it is not holding the object). Stopping; "
                "say 'open the gripper' to release, and reach again.")
            return False
        say("The gripper reports it is holding the object." if holding else
            "The gripper cannot report whether it holds the object; I will confirm with the camera while lifting.")
        return True

    def lift():
        """Straight up --lift-mm with the object in the gripper, planned in joint space like the approach and shown the same way."""
        S.update(phase="planning the lift", detail="")
        sn0 = snapshot()
        before = None if (sn0["x"] is None or sn0["clipped"]) else sn0["x"].copy()
        p0 = shared["cur"].copy()
        plan = line_plan(p0, (0.0, 0.0, 1.0), args.lift_mm, args.lift_speed, obstacles=obs, bounds=bounds,
                         step_mm=args.plan_step_mm)
        attach_and_guard(plan)
        S["progress"] = 0.0
        S["plan_obj"] = plan
        S["plan_ctx"] = {"c0_mm": 0.0, "p_obj": plan.waypoints[-1].pos, "box": (600.0, 300.0, 680.0, 420.0)}
        lines = describe_line(plan, args.lift_speed, "Lift the object")
        S["plan"] = lines
        say("\n".join(lines))
        if plan.travel_mm < 5:
            say("The lift is blocked; stopping with the object held.")
            return False
        if not plan.ok:
            say(f"The lift stops after {plan.travel_mm:.0f} mm because of the limit above.")
        for k in range(int(args.countdown * 10), 0, -1):
            S.update(phase=f"lifting in {k / 10:.1f}s (say stop to cancel)")
            check()
            time.sleep(0.1)
        S["phase"] = "lifting"
        run_segment(plan, 0, len(plan.waypoints) - 1, args.lift_speed)
        time.sleep(args.settle)
        holding = robot.is_holding() if real else True
        sn = wait_fresh(time.monotonic(), 2.0) if real else None
        moved = None
        if sn is not None and before is not None and not sn["clipped"]:
            moved = float(max(np.max(np.abs(sn["x"][:2] - before[:2])), abs(sn["x"][2] - before[2]) * 0.5))
        if holding is False or (moved is not None and moved > 0.12):
            say("The object did not come up with the gripper"
                + (" (the camera sees it stayed behind)." if moved is not None and moved > 0.12 else " (the gripper lost it).")
                + " Stopping; say 'open the gripper' to release.")
            return False
        say("Lifted: " + ("the camera sees the object fixed in the gripper"
                          if moved is not None else "the object is held" if holding else
                          "the grasp could not be confirmed by the gripper or the camera, but nothing slipped that I can see")
            + f" ({plan.travel_mm / 10:.0f} cm up).")
        return True

    def mission(phases):
        """Run the phases in order: ('reach'|'pick'|'person', object name)."""
        nonlocal J, ddir
        try:
            S.update(plan=[], waiting=False, progress=None, opened=False, holding=None, plan_obj=None, plan_ctx=None)
            ms["mode"], ms["locked"] = "", {}
            if real:
                kin0, note0 = get_kin()                   # load the kinematics first: it switches on the arm-body guard
                say(("Arm-body guard on: every link stays >= %.0f mm from the obstacles (kinematics checked: %s)."
                     % (args.clearance_mm, note0)) if (kin0 is not None and not args.no_body_guard) else
                    f"Arm-body guard is OFF ({note0}); only the tool point is kept clear of obstacles.")
            if (phases[0][0] in ("reach", "pick") and args.grasp_from == "top" and not args.no_orient
                    and not args.no_face_down and args.camera_down_at == "start"):
                if not face_down():
                    S.update(phase="idle")
                    return
            for n_, (kind, name) in enumerate(phases):
                if n_ > 0 or det["name"] != name or det["model"] is None:
                    S.update(phase="looking", detail="")
                    say(("Looking again for the closest person ..." if kind == "person" else f"Looking for the {name} ..."))
                    set_detector(name)
                    t_end = time.monotonic() + 8.0          # wait for a lock on it
                    while snapshot()["x"] is None and time.monotonic() < t_end:
                        check()
                        time.sleep(0.1)
                    if snapshot()["x"] is None:
                        say(f"I can't see a {name} in the camera. Stopping.")
                        S.update(phase="idle")
                        return
                    if kind == "person":
                        say("Locked on the closest person (the largest one in view).")
                which = S.pop("which", None) if n_ == 0 else None
                if which is not None:
                    select_candidate(which)
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
                refresh_body()
                if ddir is None:
                    say("The learned camera response has no depth direction. Re-learn it with the object in clear view (p).")
                    return
                if (kind in ("reach", "pick") and args.grasp_from == "top" and args.camera_down_at == "near"
                        and not args.no_orient and not args.no_face_down and ms["mode"] != "down"):
                    if not run_phase(kind, name, pre=True):                  # centre, approach to --turn-down-cm
                        return
                    p_obj = (S["plan_ctx"] or {}).get("p_obj")               # where the object is (plan-time estimate)
                    m15, _ = measure_tip(n=6, timeout=3.0) if real else (None, 0.0)
                    sn15 = snapshot()
                    if m15 is not None and sn15["x"] is not None:            # better: measured now, centred, 15 cm away
                        lat15 = servo_step(J_now(), sn15["x"], target, (0, 1), 1.0, 80.0)
                        p_obj = shared["cur"] + lat15 + ddir * (m15 * 10.0 + args.tip_offset_mm)
                    if p_obj is None or not raise_up(args.turn_down_lift_mm, "Raise before turning the camera down"):
                        S.update(phase="idle")
                        return
                    if not turn_camera_down_above(np.array(p_obj, float)):
                        S.update(phase="idle")
                        return
                if not run_phase(kind, name):
                    return
                if kind == "pick":
                    if not grasp() or not lift():
                        S.update(phase="idle", detail="")
                        return
            S.update(phase="done", detail="")
            if phases[-1][0] == "person":
                say("At the hand-over position. Say 'open the gripper' to release the object, or give me a new command.")
            elif phases[-1][0] == "pick":
                say("The object is lifted and held. Say 'bring it to the closest person' or 'open the gripper'.")
        except Abort:
            S.update(phase="stopped", detail="")
        except Exception as e:
            import traceback
            traceback.print_exc()                                   # the details go to the console; the chat gets the short form
            say(f"Mission failed: {type(e).__name__}: {e}")
            if real:
                robot.stop()
            S.update(phase="idle", detail="")

    def start_mission(phases):
        stop_mission()
        S["abort"].clear()
        S["phases"] = phases
        th = threading.Thread(target=mission, args=(phases,), daemon=True)
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
                "det_fps": round(perc.fps.value, 1), "plan": S["plan"],
                "joints": " ".join(f"J{i} {v:+.1f}" for i, v in enumerate(shared["q"])),
                "clearance": (None if shared["slack"][0] == float("inf") else f"{shared['slack'][0]:.0f} mm ({shared['slack'][1]})"),
                "tilt": (None if tilt_deg(shared["q"]) is None else f"{tilt_deg(shared['q']):+.1f} deg")}

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
        if k == "switch":
            if c.get("name") and c["name"] != det["name"] and c["name"] not in det["name"]:
                say(f"I'm looking for a {det['name']}; to change what I look for say 'go to the {c['name']}'.")
            else:
                select_candidate(c["which"])
        elif k == "click":
            r_ = perc.result
            if r_ is None:
                return
            px, py = c["x"] * r_.frame.shape[1], c["y"] * r_.frame.shape[0]
            hit = [d for d in r_.dets if d[0] <= px <= d[2] and d[1] <= py <= d[3]]
            same = [d for d in hit if det["name"] in d[5].lower()]
            if same:
                if busy_moving():
                    say("The arm is moving; say 'stop' first, then click the object you want.")
                else:
                    d = min(same, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
                    pin_candidate(d)
                    cands = candidates()
                    idx = current_index(cands)
                    say(f"Locked on the {det['name']} you clicked" + (f" (#{idx + 1} of {len(cands)})." if idx is not None else "."))
            elif hit:
                d = min(hit, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
                say(f"That is a {d[5]}, not a {det['name']}. Say 'go to the {d[5]}' to look for that instead.")
            else:
                say("There is no detected object where you clicked.")
        elif k in ("target", "pick") and c.get("which") is not None and k == "target" and (
                c["name"] == det["name"] or c["name"] in det["name"]) and S["thread"] is not None and S["thread"].is_alive():
            select_candidate(c["which"])                                    # "the left cup" while already working on a cup: just switch
        elif k in ("target", "pick"):
            stop_mission()
            S["which"] = c.get("which")
            name, note = choose_object(c["name"])
            if "stop_cm" in c:
                cfg["user_stop_cm"] = max(0.5, c["stop_cm"])
            if k == "pick":
                phases = [("pick", name)] + ([("person", "person")] if c.get("then") == "person" else [])
                say(f"Picking up the {name}{note}: reach it (fast, slowing below {cfg['slow_cm']:.0f} cm, gripper opens at "
                    f"{cfg['open_cm']:.0f} cm where I remember where it is), go {args.grasp_depth_mm:g} mm past its front "
                    f"surface, close the gripper (force {args.grip_force:g}), check the grip, lift {args.lift_mm:g} mm"
                    + (", then find the closest person and bring it to them." if len(phases) > 1 else ".")
                    + ("" if real else " (Dry run: the arm will not move.)"))
            else:
                phases = [("reach", name)]
                say(f"Going for the {name}{note}: centre it, then plan a straight approach (fast, slowing below "
                    f"{cfg['slow_cm']:.0f} cm, gripper opens at {cfg['open_cm']:.0f} cm, stop {cfg['user_stop_cm']:.1f} cm away)."
                    + ("" if real else " (Dry run: the arm will not move.)"))
            start_mission(phases)
        elif k == "person":
            stop_mission()
            if S["holding"] is not True:
                say("Note: I am not sure I am holding anything right now; going to the closest person anyway.")
            say("Bringing it to the closest person: I will look again, lock on the largest person in view, show the plan and "
                "wait for your 'yes' before moving.")
            start_mission([("person", "person")])
        elif k in ("stop", "pause"):
            stop_mission()
            S.update(phase="stopped", detail="")
            say("Stopped. The arm is holding where it is." if k == "stop" else "Paused. Say 'resume' to start again.")
        elif k == "resume":
            if S["phases"]:
                say("Resuming from the beginning of the last request: centre, plan, execute.")
                start_mission(S["phases"])
            else:
                say("Nothing to resume yet. Tell me what to go to.")
        elif k == "unlock":
            stop_mission()
            with perc.mu:
                lock.unlock()
            flt.reset()
            S.update(phase="idle", detail="", plan=[], plan_obj=None, plan_ctx=None)
            say("Unlocked. Tell me what to go to next.")
        elif k == "stop_distance":
            cfg["user_stop_cm"] = cfg["stop_cm"] = max(0.5, c["cm"])
            say(f"I'll stop {cfg['stop_cm']:.1f} cm from the object.")
        elif k == "set":
            if c["key"] == "tip_offset_mm":
                cfg["tip_off_cm"] = c["value"] / 10.0
                args.tip_offset_mm = c["value"]
                say(f"OK: the fingertips are {c['value'] / 10:.1f} cm in front of the camera; distances are now gripper-to-object.")
            elif c["key"] == "object_top_mm":
                args.object_top_mm = c["value"]
                set_range_model(cfg["kind"])
                say(f"OK: the {det['name']} is {c['value'] / 10:.1f} cm across seen from above (used for the top grasp distance).")
            else:
                args.object_height_mm = c["value"]
                set_range_model(cfg["kind"])
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
            q_reset, q_here = np.array(RESET_JOINTS, float), np.array(robot.get_joints(), float) if real else None
            ok_r, worst_r, nm_r = body_ok(q_here, q_reset, n=40) if real else (True, 0.0, "")
            if real and not ok_r:
                say(f"The reset sweep would bring the arm within {worst_r:.0f} mm of the {nm_r} (limit {args.clearance_mm:.0f} mm). "
                    "Not doing it; move the arm clear first.")
            elif real:
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
        ui = UIServer(on_text, get_status, args.port, on_click=on_click).start()
        print(f"chat UI: {ui.url}", flush=True)
        say(f"Ready ({'LIVE ARM' if real else 'dry run'}). Tell me what to go to, e.g. \"go to the cup\". I'll show the plan "
            "before moving. Type help for commands.")
        if not args.no_browser:
            webbrowser.open(ui.url)
    if args.auto_engage:
        cmdq.put({"intent": "pick", "name": args.lock, "then": "person" if args.handover else None} if args.pick
                 else {"intent": "target", "name": args.lock})

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
                cmdq.put({"intent": "click", "x": mx / r.frame.shape[1], "y": my / r.frame.shape[0]})
            now = time.monotonic()
            if r.t > flt_t:                                   # feed every new detection through the filter
                flt_t = r.t
                if r.feats is not None:
                    fx = flt.update(r.feats, r.t)
                    misses = 0
                    hpx = fx[2] * r.frame.shape[0]
                    dist = rng.cm(hpx, r.frame.shape[1])
                    with perc.mu:
                        bx = lock.box.copy() if lock.locked else None
                    clipped = bx is not None and (bx[1] <= 2 or bx[3] >= r.frame.shape[0] - 2)
                    with mu:
                        snap.update(x=fx, t=r.t, dist=dist, hpx=hpx, clipped=clipped,
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
                with perc.mu:
                    bxa = lock.box.copy() if (lock.locked and r.live) else None
                if bxa is not None:                                           # the part of the object the gripper lines up with
                    ax_, ay_ = int((bxa[0] + bxa[2]) / 2), int(bxa[1] + args.aim_frac * (bxa[3] - bxa[1]))
                    cv2.drawMarker(frame, (ax_, ay_), (0, 230, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
                    cv2.line(frame, (int(bxa[0]), ay_), (int(bxa[2]), ay_), (0, 230, 255), 1)
                if r.feats is not None:
                    cv2.line(frame, (w // 2, h // 2), (int((r.feats[0] + 0.5) * w), int((r.feats[1] + 0.5) * h)), (0, 255, 255), 2)
                busy = S["thread"] is not None and S["thread"].is_alive()
                col = (0, 200, 0) if busy else (0, 0, 220)
                lines = [f"{S['phase'].upper() if (busy or S['phase'] in ('arrived', 'stopped')) else 'IDLE - say go to the cup'} | {args.sink}", S["detail"],
                         ("" if sn["tip"] is None else f"gripper-to-object {sn['tip']:.1f} cm" + ("" if rng.calibrated else " (est)")
                          + f"   stop {cfg['stop_cm']:.1f}  open {cfg['open_cm']:.0f}" + ("  GRIPPER OPEN" if S["opened"] else "") + ("  HOLDING" if S["holding"] else ""))]
                for i, text in enumerate(lines):
                    cv2.putText(frame, text, (14, 34 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
                    cv2.putText(frame, text, (14, 34 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col if i == 0 else (255, 255, 255), 2)
                cands_ = candidates()
                if len(cands_) >= 2:                                      # several of the target kind: number them, left to right
                    idx_ = current_index(cands_)
                    for i_, d_ in enumerate(cands_):
                        mine = i_ == idx_
                        cv2.rectangle(frame, (int(d_[0]), int(d_[1])), (int(d_[2]), int(d_[3])), (0, 255, 255) if mine else (200, 200, 200),
                                      2 if mine else 1)
                        lab = f"#{i_ + 1}"
                        cv2.putText(frame, lab, (int(d_[0]) + 3, int(d_[1]) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                        cv2.putText(frame, lab, (int(d_[0]) + 3, int(d_[1]) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 255, 255) if mine else (255, 255, 255), 2)
                po = S["plan_obj"]
                if po is not None and S["plan_ctx"] is not None:                # the planned trajectory, drawn on the picture
                    with perc.mu:
                        bx = lock.box.copy() if lock.locked else None
                    draw_plan(frame, po, S["plan_ctx"], shared["cur"], bounds, obs.boxes, args.slow_speed, (w / 2, h / 2),
                              cfg["slow_cm"], cfg["open_cm"], args.tip_offset_mm, progress_mm=S["progress"] or 0.0,
                              box_now=tuple(float(v) for v in bx) if bx is not None else None, map_scale=args.map_scale)
                q = shared["q"]                                                # live joint angles, for debugging (compact)
                tl, (gap, gname) = tilt_deg(q), shared["slack"]
                rows = ["  ".join(f"J{i}{v:+7.1f}" for i, v in enumerate(q[:3])), "  ".join(f"J{i}{v:+7.1f}" for i, v in enumerate(q[3:], 3))]
                extra = ([f"tilt {tl:+.1f}"] if tl is not None else []) + ([f"clr {gap:.0f}mm"] if gap != float("inf") else [])
                if extra:
                    rows.append("  ".join(extra))
                fs, lh = 0.38, 15
                bw = int(max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0] for t in rows)) + 10
                x0 = w - bw - 4
                cv2.rectangle(frame, (x0 - 4, 4), (w - 2, lh * len(rows) + 10), (0, 0, 0), -1)
                for i, text in enumerate(rows):
                    wrist_row = i == 1
                    colr = (80, 80, 255) if (text.startswith("tilt") is False and "clr" in text and gap < args.clearance_mm) else \
                        ((0, 230, 255) if wrist_row else (255, 255, 255))
                    cv2.putText(frame, text, (x0, 16 + lh * i), cv2.FONT_HERSHEY_SIMPLEX, fs, colr, 1)
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
