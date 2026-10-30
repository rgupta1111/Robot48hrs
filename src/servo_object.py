"""Camera-on-gripper visual servoing: move the gripper so the cup stays CENTRED in the camera image.

Closed loop (pipelined, so it runs as fast as the camera allows):
    camera grabbers (several requests in flight) -> detector thread (light YOLO, every new frame) -> control loop
    -> gripper move on its own thread -> next frame ... The screen shows the live camera / detector fps.

How the image offset turns into robot motion is LEARNED, not assumed: at engage the arm makes three small probe
moves (along x, y, z, then back) and watches how the cup shifts, giving the matrix J (see mimic/servo.py). That
covers however the camera is mounted. The tool orientation is never changed, so the camera keeps its view.
If the cup leaves the picture it keeps moving toward where it was last seen (a few steps), then holds.

    scripts/run.sh src/servo_object.py --viam-address <machine>.viam.cloud --preset lab --lock cup            # dry run
    scripts/run.sh src/servo_object.py --viam-address <machine>.viam.cloud --preset lab --lock cup --sink viam
Keys (click the window): space engage/disengage, p re-probe, c calibrate distance, t teach stop distance, x e-stop, l unlock, r re-acquire,
click a box to lock it, q / Esc quit.  Dry run (--sink sim) sends nothing to the arm.
Chat UI (--ui): open a local web page with the live camera view and a chat box. Type what to go to ("go to the cup",
"lock on the bottle", "get within 5 cm of the cup"), or stop / pause / resume / unlock / reset / "what do you see".
The arm only moves while engaged; STOP and the word "stop" always halt it. Add --sink viam for the real arm.
    scripts/run.sh src/servo_object.py --viam-address <machine>.viam.cloud --preset lab --ui --sink viam
Distance: the camera-to-object distance is estimated from the object's size on screen (pinhole model) and shown on
the video; give the true height with --object-height-mm, or hold the object at a known distance and press c
(--calib-cm) to calibrate.  Approach (default; --no-approach turns it off): once the cup is centred, the gripper moves
toward it (--approach-speed) until the estimated distance is under --stop-distance-mm (default 100 = 10 cm), then
holds. Press t with the gripper at the distance you want to teach a different stop distance. There is NO contact
sensor: the stop distance and the workspace bounds are the only safeguards, so keep a hand on the e-stop.
Smooth motion: detections are filtered and the cup's image velocity is estimated so the arm aims slightly AHEAD
(--lead); successive moves are blended (--smooth); --preempt sends the next move before the previous one finishes
(the arm retargets without stopping) and subtracts moves that the camera has not shown yet (--lag).
Speed: the camera is read over the network. If "cam" fps stays low, raise --workers, connect on the local network
address, or lower the camera's resolution in the Viam config; the detector fps is shown separately.
"""
import argparse
import json
import os
import queue
import sys
import time
import webbrowser

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_mimic import PRESETS, floats  # noqa: E402
from local_detect import (COLORS, Lock, Range, color_candidates, crop_gray, detect, draw_lock,  # noqa: E402
                          load_detector, object_candidates, roi_candidates, template_candidates)
from mimic.chat import HELP, parse  # noqa: E402
from mimic.pipeline import FrameGrabber, MoveWorker, Perception  # noqa: E402
from mimic.servo import FeatureFilter, approach_factor, estimate_jacobian, features, servo_step, tip_target_v  # noqa: E402
from mimic.ui import UIServer  # noqa: E402
from reset_position import RESET_JOINTS  # noqa: E402
from tidy.robot import TableToWorld, ViamRobot  # noqa: E402

WIN = "servo object"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
JAC_FILE = os.path.abspath(os.path.join(ROOT, "data", "servo_jacobian.json"))
# only used by the dry run when no probe has been done: image right/down -> gripper -y / -x, a guess
GUESS_J = np.array([[0.0, -0.003, 0.0], [-0.003, 0.0, 0.0], [0.0, 0.0, 0.0008]])


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
    ap.add_argument("--lock", default="cup", metavar="NAME", help="object to keep centred (default cup)")
    ap.add_argument("--lock-color", choices=sorted(COLORS), help="optional: centre a colour blob instead")
    ap.add_argument("--lock-min-area", type=float, default=400)
    ap.add_argument("--model", help="YOLO weights (default: yolov8s)")
    ap.add_argument("--prompts")
    ap.add_argument("--conf", type=float)
    ap.add_argument("--imgsz", type=int, default=960, help="detector input size (smaller = faster, but misses small/far cups)")
    ap.add_argument("--workers", type=int, default=4, help="overlapping camera requests (more = higher frame rate)")
    ap.add_argument("--rate", type=float, default=15.0, help="control loop rate limit, Hz")
    ap.add_argument("--hold-size", type=float, metavar="FRAC",
                    help="also keep the cup this fraction of the frame height (e.g. 0.3): moves closer/farther")
    ap.add_argument("--gain", type=float, default=0.8, help="fraction of the error corrected per move")
    ap.add_argument("--max-step", type=float, default=60.0, help="largest single move, mm")
    ap.add_argument("--max-speed", type=float, default=400.0, help="gripper speed limit, mm/s")
    ap.add_argument("--deadband", type=float, default=0.03, help="stop moving when centred within this fraction of the frame")
    ap.add_argument("--latency", type=float, default=0.05, help="extra seconds to wait after a move before trusting a frame")
    ap.add_argument("--smooth", type=float, default=0.2, help="0..1 blend with the previous move (higher = smoother, laggier)")
    ap.add_argument("--lead", type=float, default=0.15, help="extra seconds to aim ahead of the cup's image motion")
    ap.add_argument("--preempt", action="store_true",
                    help="keep moving: send the next move before the last one ends (try it; turn off if the arm complains)")
    ap.add_argument("--lag", type=float, default=0.4, help="--preempt: seconds before a move shows up in the camera image")
    ap.add_argument("--no-approach", action="store_true", help="only keep the object centred; do not move toward it")
    ap.add_argument("--touch", action="store_true", help=argparse.SUPPRESS)   # old name: approaching is now the default
    ap.add_argument("--stop-distance-mm", type=float, default=100.0, help="stop moving forward at this camera-to-object distance")
    ap.add_argument("--approach-speed", type=float, default=80.0, help="closing speed while approaching, mm/s")
    ap.add_argument("--object-height-mm", type=float, default=100.0, help="true height of the object (for distance)")
    ap.add_argument("--hfov", type=float, default=70.0, help="camera horizontal field of view in degrees, if unknown")
    ap.add_argument("--calib-cm", type=float, default=30.0, help="press c with the object this far from the camera")
    ap.add_argument("--probe-mm", type=float, default=20.0, help="size of the learning moves")
    ap.add_argument("--probe", action="store_true", help="re-learn the camera response even if a saved one exists")
    ap.add_argument("--search-steps", type=int, default=5, help="moves toward where the cup vanished before holding")
    ap.add_argument("--bounds", default="250,600,-300,300,0,450")
    ap.add_argument("--ui", action="store_true", help="serve the chat page on http://127.0.0.1:PORT/ and open it")
    ap.add_argument("--tip-offset-mm", type=float, default=0.0, help="the gripper sits this far BELOW the camera: aim so the "
                    "gripper, not the camera axis, points at the object (0 = centre the object in the image, as before)")
    ap.add_argument("--tip-flip", action="store_true", help="the gripper is ABOVE the camera in the image instead")
    ap.add_argument("--grasp", action="store_true", help="on arrival: ask in the chat, then close the gripper (force-capped) and lift")
    ap.add_argument("--lift-mm", type=float, default=120.0, help="--grasp: how far to lift after closing")
    ap.add_argument("--max-torque", type=int, default=30, help="--grasp: grip force limit 1-100%% (UFactory G2 gripper only)")
    ap.add_argument("--arm-speed", type=float, help="real arm joint speed in deg/s, set at start (UFactory: 15 slow, 30 medium, 60 fast)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="with --ui: do not open the page automatically")
    ap.add_argument("--auto-engage", action="store_true")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    if preset:
        ap.set_defaults(bounds=PRESETS[preset]["bounds"])
    args = ap.parse_args()
    args.touch = not args.no_approach
    bounds = floats(args.bounds)
    lo, hi = np.array(bounds[0::2]), np.array(bounds[1::2])
    real = args.sink == "viam"

    det = {"model": None, "dargs": None, "name": args.lock}
    if not args.lock_color:
        det["model"], det["dargs"] = load_detector(args.lock, args.model, args.prompts, args.conf, args.imgsz,
                                                   standard="yolov8s.pt")
    robot = ViamRobot(args.viam_address, args.arm_name, args.gripper_name, None, TableToWorld(*floats(args.table_to_world)),
                      workspace=bounds[:4], max_z_mm=bounds[5], insecure=args.viam_insecure)
    if real and args.arm_speed:
        try:
            robot.arm_command({"set_speed": args.arm_speed})
            print(f"arm speed set to {args.arm_speed:.0f} deg/s", flush=True)
        except Exception as e:
            print(f"could not set the arm speed ({type(e).__name__}: {e}); using the configured speed", flush=True)
    lock = Lock(max_jump_frac=0.5, lost_after=10 ** 6)   # the picture jumps when the gripper moves; loss is handled below
    J = None
    if os.path.exists(JAC_FILE) and not args.probe:
        J = np.array(json.load(open(JAC_FILE))["J"])
        print("loaded the saved camera response from", JAC_FILE)
    feats = (0, 1, 2) if args.hold_size else (0, 1)
    target = np.array([0.0, 0.0, args.hold_size or 0.0])
    click = {}

    def current():
        p = robot.get_pose()
        return np.array(robot.table.to_robot(p.x, p.y, p.z))

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
            memo["template"] = crop_gray(frame, lock.box)      # what the object looks like, for the fallback
        return dict(feats=features(lock.box, w, h) if live else None, live=live, dets=dets)

    def do_move(job):
        """Runs on the move thread: relative move from where the gripper is now."""
        if real:
            cur = current()
            robot.move_xyz(*np.clip(cur + job, lo, hi))
        shared["cur"] = current()

    robot.get_frame(args.camera)                      # create the camera client before threads share it
    shared = {"cur": current()}
    grabber = FrameGrabber(lambda: robot.get_frame(args.camera), args.workers).start()
    perc = Perception(grabber, analyse)
    perc.start()
    worker = MoveWorker(do_move)
    worker.start()

    def wait_result(after, timeout=8.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            r = perc.result
            if r is not None and r.t > after:
                return r
            time.sleep(0.01)
        return None

    def move_and_wait(pos):
        robot.move_xyz(*pos)
        shared["cur"] = current()
        return time.monotonic() + args.latency

    def probe():
        """Learn J: nudge the gripper along x, y, z and watch how the cup moves in the image."""
        print("probing: three small moves, keep the cup in view ...", flush=True)
        r0 = wait_result(time.monotonic())
        if r0 is None or r0.feats is None:
            print("probe aborted: cup not visible", flush=True)
            return None
        s0, p0, after, deltas = r0.feats, current(), [], []
        for i in range(3):
            tgt = p0.copy()
            d = args.probe_mm if p0[i] + args.probe_mm <= hi[i] else -args.probe_mm
            tgt[i] += d
            r = wait_result(move_and_wait(tgt))
            s = None if r is None else r.feats
            move_and_wait(p0)
            if s is None:
                print(f"probe aborted: cup left the view when moving axis {'xyz'[i]}", flush=True)
                return None
            after.append(s)
            deltas.append(d)
        wait_result(time.monotonic() + args.latency)
        Jn = estimate_jacobian(s0, after, deltas)
        os.makedirs(os.path.dirname(JAC_FILE), exist_ok=True)
        json.dump({"J": Jn.tolist(), "probe_mm": args.probe_mm}, open(JAC_FILE, "w"), indent=2)
        print("learned camera response J (frame fraction per mm; rows u, v, h; columns x, y, z):\n", np.round(Jn, 5), flush=True)
        return Jn

    show = not args.no_display
    if show:
        cv2.namedWindow(WIN)
        cv2.setMouseCallback(WIN, lambda e, x, y, f, p: click.update(pt=(x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
    engaged, status, last_s, lost_n, last_used, last_tick, n = args.auto_engage, "idle", None, 0, 0.0, 0.0, 0
    err_txt, last_print = "", 0.0
    rng = Range(args.object_height_mm, args.hfov)
    try:  # use the camera's own focal length when the machine reports it
        _, intr = robot.camera_info(args.camera)
        if intr:
            rng.focal_px, rng.focal_w = intr["fx"], intr["w"]
    except Exception:
        pass
    dist_cm, stop_cm, touching = None, args.stop_distance_mm / 10.0, False
    flt, flt_t, misses, prev_move, hist, futs = FeatureFilter(), 0.0, 0, np.zeros(3), [], []
    cmd_pos = shared["cur"].copy()

    # ---- chat UI -------------------------------------------------------------------------------------------
    cmdq, ui, info, pending, coco = queue.Queue(), None, {}, {"confirm": None}, {}

    def coco_names():
        if "n" not in coco:
            from ultralytics import YOLO
            coco["n"] = set(YOLO(os.path.join(ROOT, "weights", "yolov8s.pt")).names.values())
        return coco["n"]

    def on_text(text):
        """Runs on the web server thread: answer what is quick, queue what must run on the control loop."""
        c = parse(text)
        k = c["intent"]
        if k == "help":
            return HELP
        if k in ("empty", "unknown"):
            return "I didn't catch that. Try: go to the cup  (or type help)." if k == "unknown" else None
        if k == "stop":
            cmdq.put(c)                       # the loop halts the arm; the reply comes from there
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
            st = info.get("status", "idle")
            return (f"{st}. Target: {det['name']}." + (f" About {info['dist']:.0f} cm away." if info.get("dist") else "")
                    + f" Stop distance {info.get('stop', 0):.0f} cm." + (" Engaged." if info.get("engaged") else " Not moving."))
        cmdq.put(c)
        return None

    def get_status():
        return {"mode": "LIVE ARM" if real else "DRY RUN", "state": info.get("status", "idle"), "target": det["name"],
                "distance": None if not info.get("dist") else f"{info['dist']:.0f} cm" + ("" if rng.calibrated else " (est)"),
                "stop_at": f"{info.get('stop', stop_cm):.0f} cm", "cam_fps": round(grabber.fps.value, 1),
                "det_fps": round(perc.fps.value, 1)}

    def say(text):
        print("robot:", text, flush=True)
        if ui:
            ui.say(text)

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
        """Runs on the control loop so it can change the lock, the detector and the arm safely."""
        nonlocal engaged, stop_cm, last_s, lost_n, prev_move, J, last_used, flt_t, dist_cm, seen
        k = c["intent"]
        if k == "target":
            name, note = choose_object(c["name"])
            say(f"Looking for {name}{note} ...")
            if name != det["name"] or det["model"] is None:
                model, dargs = load_detector(name, args.model, args.prompts, args.conf, args.imgsz, standard="yolov8s.pt")
                with perc.mu:                                   # swap between two analysed frames
                    det.update(model=model, dargs=dargs, name=name)
                    lock.unlock()
                    lock.enabled = True
                    memo["template"] = None
            else:
                with perc.mu:
                    lock.unlock()
                    lock.enabled = True
            flt.reset()
            last_s, lost_n, prev_move, dist_cm, seen = None, 0, np.zeros(3), None, False
            last_used = flt_t = time.monotonic()          # ignore frames analysed for the previous target
            if "stop_cm" in c:
                stop_cm = max(2.0, c["stop_cm"])
            engaged = True
            memo["grasped"] = False
            if args.grasp and real and not memo.get("opened"):
                robot.set_gripper(True)                      # open before going in
                memo["opened"] = True
            say(f"Going for the {name}: centring it, then moving forward to {stop_cm:.0f} cm."
                + ("" if real else " (Dry run: the arm will not move.)")
                + (" First I'll make three small test moves to learn the camera." if real and J is None else ""))
        elif k == "stop":
            engaged = False
            if real:
                robot.stop()
            say("Stopped. The arm is holding where it is.")
        elif k == "pause":
            engaged = False
            say("Paused. Say 'resume' to continue.")
        elif k == "resume":
            if lock.locked:
                engaged, last_s, lost_n = True, None, 0
                say("Resuming.")
            else:
                say("Nothing is locked yet. Tell me what to go to.")
        elif k == "unlock":
            engaged = False
            with perc.mu:
                lock.unlock()
            flt.reset()
            say("Unlocked. Tell me what to go to next.")
        elif k in ("stop_distance",):
            stop_cm = max(2.0, c["cm"])
            say(f"I'll stop {stop_cm:.0f} cm from the object.")
        elif k == "reset":
            engaged = False
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
        elif k == "yes" and pending["confirm"] == "grasp":
            pending["confirm"], memo["grasped"], engaged = None, True, False
            if not real:
                say(f"Dry run: I would close the gripper (force cap {args.max_torque}%) and lift {args.lift_mm:.0f} mm.")
            else:
                try:   # UFactory G2 only; a G1 has no force setting, so the close below is then unlimited
                    robot.gripper_command({"set_gripper_torque": args.max_torque})
                except Exception as e:
                    say(f"(could not cap the grip force: {type(e).__name__}; closing at the default force)")
                robot.set_gripper(False)
                time.sleep(1.5)
                pos = current()
                robot.move_xyz(pos[0], pos[1], min(pos[2] + args.lift_mm, bounds[5]), wait=True)
                shared["cur"] = current()
                try:
                    held = bool(robot._run(robot.gripper.is_holding_something()).value)
                except Exception:
                    held = None
                say("Closed the gripper and lifted." + ("" if held is None else f" Holding something: {held}."))
        elif k in ("yes", "no"):
            pending["confirm"] = None
            say("OK." if k == "no" else "Nothing to confirm.")

    if args.ui:
        ui = UIServer(on_text, get_status, args.port).start()
        print(f"chat UI: {ui.url}", flush=True)
        say(f"Ready ({'LIVE ARM' if real else 'dry run'}). Tell me what to go to, e.g. \"go to the cup\". Type help for commands.")
        if not args.no_browser:
            webbrowser.open(ui.url)
    prev_status, seen = "", False
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
            if args.max_frames and n >= args.max_frames:
                break
            if worker.error:
                print("ARM ERROR:", worker.error)
                break
            if click.get("pt"):
                mx, my = click.pop("pt")
                hit = [d for d in r.dets if d[0] <= mx <= d[2] and d[1] <= my <= d[3]]
                if hit:
                    d = min(hit, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
                    with perc.mu:
                        lock.pin(d[:4], d[5])
            now = time.monotonic()
            if engaged and J is None:
                if real:
                    J = probe()
                    if J is None:
                        engaged = False
                    continue
                J = GUESS_J
                print("dry run with a GUESSED camera response (use --sink viam to learn the real one)", flush=True)
            if r.t > flt_t:                                   # feed every new detection through the filter
                flt_t = r.t
                if r.feats is not None:
                    flt.update(r.feats, r.t)
                    misses, seen = 0, True
                    dist_cm = rng.cm(flt.x[2] * r.frame.shape[0], r.frame.shape[1])
                else:
                    misses += 1
                    if misses > 5:
                        flt.reset()
                        dist_cm = None
            for f in [f for f in futs if f.done()]:
                futs.remove(f)
                if f.exception():
                    print("ARM ERROR:", f.exception())
                    engaged = False
            move = None
            if engaged and now - last_tick >= 1.0 / args.rate:
                blocked = len([f for f in futs if not f.done()]) >= 2 if args.preempt else worker.busy
                if blocked:
                    status = "moving"
                elif r.t <= (last_used if args.preempt else max(worker.end_t + args.latency, last_used)):
                    status = "waiting for a fresh frame"
                else:
                    dtick, last_tick, last_used = now - last_tick, now, r.t
                    if flt.x is not None and r.feats is not None:
                        s = flt.predict(min(now - flt.t + args.lead, 0.6))      # where the cup is heading, now
                        if args.preempt:                                         # moves the camera has not shown yet
                            pend = sum((m for ts, m in hist if ts > r.t - args.lag), np.zeros(3))
                            s = s + J @ pend
                        last_s, lost_n = s, 0
                        if args.tip_offset_mm and dist_cm:          # parallax: the aim point moves down the image as it gets close
                            f_px = (rng.focal_px * r.frame.shape[1] / rng.focal_w if rng.focal_px
                                    else r.frame.shape[1] / (2 * np.tan(np.radians(args.hfov) / 2)))
                            target[1] = (-1 if args.tip_flip else 1) * tip_target_v(dist_cm, args.tip_offset_mm, f_px, r.frame.shape[0])
                        tgt, fu = target, feats
                        status = "tracking"
                        err = tgt[list(fu)] - s[list(fu)]
                        err_txt = (f"offset u{s[0]:+.2f} v{s[1]:+.2f}" + (f" (aim v{target[1]:+.2f})" if args.tip_offset_mm else "")
                                   + ("" if dist_cm is None else f"  {dist_cm:.0f} cm"))
                        near = np.max(np.abs(err[:2])) < args.deadband
                        if args.touch and dist_cm is not None:
                            if abs(J[2]).max() < 1e-5:
                                status, err_txt = "no depth response - re-probe (p)", "cannot approach"
                                near = True
                            elif dist_cm <= stop_cm + 0.5:
                                status, touching = "ARRIVED - holding", True
                            else:
                                touching = False
                                # creep closer in proportion to how well centred it is (a smooth taper, not an on/off
                                # switch: the error hovers near any threshold and flipped the motion every few steps)
                                closer = (args.approach_speed / 10.0 * min(max(dtick, 0.05), 0.5)
                                          * approach_factor(err[:2], args.deadband))
                                d_goal = max(stop_cm, dist_cm - closer)
                                tgt = np.array([target[0], target[1], s[2] * dist_cm / d_goal])   # size grows as the distance shrinks
                                fu, near = (0, 1, 2), False
                        if status.startswith("ARRIVED") or status.startswith("no depth"):
                            cmd_pos = shared["cur"].copy()
                            if status.startswith("ARRIVED"):
                                move = servo_step(J, s, target, feats, args.gain, args.max_step) if not near else None
                        elif near and (not args.hold_size or abs(err[2]) < 0.03) and not args.touch:
                            status = "centred"
                            cmd_pos = shared["cur"].copy()
                        else:
                            status = "approaching" if fu == (0, 1, 2) and args.touch else "tracking"
                            move = servo_step(J, s, tgt, fu, args.gain, args.max_step)
                            if status == "approaching":   # slow, and never faster than the approach speed
                                move = move * min(1.0, args.approach_speed / args.rate / max(np.linalg.norm(move), 1e-9) * 2)
                    elif last_s is not None and lost_n < args.search_steps:
                        lost_n += 1                    # cup is out of the picture: keep going where it left
                        status, err_txt = f"searching {lost_n}/{args.search_steps}", "object not seen"
                        move = servo_step(J, last_s * 1.5, target, feats, args.gain, args.max_step)
                    else:
                        status, err_txt = "lost - holding", "object not seen"
            elif not engaged:
                status, err_txt = "idle", ""
            if move is not None:
                move = args.smooth * prev_move + (1 - args.smooth) * move     # blend: no sudden direction changes
                move = move * min(1.0, args.max_speed * (1.0 / args.rate) / max(np.linalg.norm(move), 1e-9))
                prev_move = move
                n += 1
                cur = shared["cur"]
                if args.preempt:
                    cmd_pos = np.clip(cmd_pos + move, lo, hi)
                    hist = [(ts, m) for ts, m in hist if now - ts < 3.0] + [(now, move)]
                    if real:
                        futs.append(robot.move_xyz(*cmd_pos, wait=False))
                        shared["cur"] = cmd_pos.copy()
                    sent = True
                else:
                    sent = worker.submit(move)
                    if sent and not real:
                        worker.end_t = time.monotonic()
                if sent:
                    print(f"[{n:3}] {status:16} {err_txt:32} gripper x{cur[0]:.0f} y{cur[1]:.0f} z{cur[2]:.0f}  "
                          f"move {move[0]:+.0f} {move[1]:+.0f} {move[2]:+.0f} mm" + ("" if real else " (dry run)"), flush=True)
            elif status != "tracking":
                prev_move = np.zeros(3)
            info.update(status=status, dist=dist_cm, stop=stop_cm, engaged=engaged)
            if status != prev_status:
                if status.startswith("ARRIVED"):
                    say(f"Arrived: the {det['name']} is about {dist_cm:.0f} cm away. Holding here.")
                    if args.grasp and not memo.get("grasped"):
                        pending["confirm"] = "grasp"
                        say(f"Close the gripper on it (force cap {args.max_torque}%) and lift {args.lift_mm:.0f} mm? "
                            "Type 'yes' to go, or 'no' to leave it.")
                elif status.startswith("lost"):
                    say(f"I lost sight of the {det['name']}. Holding position." if seen else
                        f"I can't see a {det['name']} in the camera. Move it into view or try another name.")
                prev_status = status
            if now - last_print > 2.0:
                last_print = now
                print(f"   cam {grabber.fps.value:.1f} fps, detector {perc.fps.value:.1f} fps, frame age "
                      f"{now - r.t:.2f}s, {status}", flush=True)

            if show or ui:
                frame = r.frame.copy()
                h, w = frame.shape[:2]
                with perc.mu:
                    draw_lock(frame, lock, r.live, dist_cm if r.live else None, rng.calibrated)
                cv2.drawMarker(frame, (w // 2, h // 2), (0, 255, 0), cv2.MARKER_CROSS, 40, 2)   # where the cup should be
                if r.feats is not None:
                    cv2.line(frame, (w // 2, h // 2), (int((r.feats[0] + 0.5) * w), int((r.feats[1] + 0.5) * h)), (0, 255, 255), 2)
                col = (0, 200, 0) if engaged else (0, 0, 220)
                lines = [("ENGAGED" if engaged else "IDLE - space to engage") + f" | {status} | {args.sink}", err_txt,
                         ("" if dist_cm is None else f"distance {dist_cm:.1f} cm" + ("" if rng.calibrated else " (est)")
                          + (f"  stop at {stop_cm:.0f} cm" if args.touch else "")),
                         f"cam {grabber.fps.value:.1f} fps   detector {perc.fps.value:.1f} fps"]
                for i, text in enumerate(lines):
                    cv2.putText(frame, text, (14, 34 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
                    cv2.putText(frame, text, (14, 34 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col if i == 0 else (255, 255, 255), 2)
                if ui:
                    ok_, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok_:
                        ui.set_frame(buf.tobytes())
                if not show:
                    time.sleep(0.01)
                    continue
                cv2.imshow(WIN, frame)
                key = cv2.waitKey(5) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    engaged = not engaged
                    last_s, lost_n, prev_move = None, 0, np.zeros(3)
                    cmd_pos = current() if real else shared["cur"].copy()
                elif key == ord("p"):
                    J, args.probe = None, True
                elif key == ord("c") and dist_cm is not None:
                    rng.calibrate(flt.x[2] * r.frame.shape[0], args.calib_cm)
                    print(f"calibrated: the object is {args.calib_cm:.0f} cm away at the current size", flush=True)
                elif key == ord("t") and dist_cm is not None:
                    stop_cm = dist_cm
                    print(f"stop distance taught: {stop_cm:.1f} cm (the camera-to-object distance when touching)", flush=True)
                elif key == ord("x"):
                    engaged = False
                    if real:
                        robot.stop()
                elif key == ord("l"):
                    with perc.mu:
                        lock.unlock()
                        lock.enabled = False
                elif key == ord("r"):
                    with perc.mu:
                        lock.unlock()
                        lock.enabled = True
            else:
                time.sleep(0.01)
    finally:
        grabber.stop()
        perc.stop()
        worker.stop()
        try:
            if real:
                robot.stop()
        finally:
            cv2.destroyAllWindows()
            robot.close()


if __name__ == "__main__":
    main()
