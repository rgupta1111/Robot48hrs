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
Distance: the camera-to-object distance is estimated from the object's size on screen (pinhole model) and shown on
the video; give the true height with --object-height-mm, or hold the object at a known distance and press c
(--calib-cm) to calibrate.  --touch: once the cup is centred, creep toward it (--approach-speed) until the estimated
distance reaches --stop-distance-mm (camera-to-cup distance when the fingers touch; hold the gripper touching the cup
and press t to teach it), then hold. There is NO contact sensor: the stop distance and the workspace bounds are
the only safeguards, so test the dry run first and keep a hand on the e-stop.
Smooth motion: detections are filtered and the cup's image velocity is estimated so the arm aims slightly AHEAD
(--lead); successive moves are blended (--smooth); --preempt sends the next move before the previous one finishes
(the arm retargets without stopping) and subtracts moves that the camera has not shown yet (--lag).
Speed: the camera is read over the network. If "cam" fps stays low, raise --workers, connect on the local network
address, or lower the camera's resolution in the Viam config; the detector fps is shown separately.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_mimic import PRESETS, floats  # noqa: E402
from local_detect import (COLORS, Lock, Range, color_candidates, crop_gray, detect, draw_lock,  # noqa: E402
                          load_detector, object_candidates, roi_candidates, template_candidates)
from mimic.pipeline import FrameGrabber, MoveWorker, Perception  # noqa: E402
from mimic.servo import FeatureFilter, estimate_jacobian, features, servo_step  # noqa: E402
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
    ap.add_argument("--rate", type=float, default=10.0, help="control loop rate limit, Hz")
    ap.add_argument("--hold-size", type=float, metavar="FRAC",
                    help="also keep the cup this fraction of the frame height (e.g. 0.3): moves closer/farther")
    ap.add_argument("--gain", type=float, default=0.5, help="fraction of the error corrected per move")
    ap.add_argument("--max-step", type=float, default=30.0, help="largest single move, mm")
    ap.add_argument("--max-speed", type=float, default=200.0, help="gripper speed limit, mm/s")
    ap.add_argument("--deadband", type=float, default=0.03, help="stop moving when centred within this fraction of the frame")
    ap.add_argument("--latency", type=float, default=0.1, help="extra seconds to wait after a move before trusting a frame")
    ap.add_argument("--smooth", type=float, default=0.3, help="0..1 blend with the previous move (higher = smoother, laggier)")
    ap.add_argument("--lead", type=float, default=0.15, help="extra seconds to aim ahead of the cup's image motion")
    ap.add_argument("--preempt", action="store_true",
                    help="keep moving: send the next move before the last one ends (try it; turn off if the arm complains)")
    ap.add_argument("--lag", type=float, default=0.4, help="--preempt: seconds before a move shows up in the camera image")
    ap.add_argument("--touch", action="store_true", help="after centring, approach the object until the stop distance")
    ap.add_argument("--stop-distance-mm", type=float, default=120.0, help="camera-to-object distance at which to stop")
    ap.add_argument("--approach-speed", type=float, default=30.0, help="closing speed while approaching, mm/s")
    ap.add_argument("--object-height-mm", type=float, default=100.0, help="true height of the object (for distance)")
    ap.add_argument("--hfov", type=float, default=70.0, help="camera horizontal field of view in degrees, if unknown")
    ap.add_argument("--calib-cm", type=float, default=30.0, help="press c with the object this far from the camera")
    ap.add_argument("--probe-mm", type=float, default=20.0, help="size of the learning moves")
    ap.add_argument("--probe", action="store_true", help="re-learn the camera response even if a saved one exists")
    ap.add_argument("--search-steps", type=int, default=5, help="moves toward where the cup vanished before holding")
    ap.add_argument("--bounds", default="250,600,-300,300,0,450")
    ap.add_argument("--auto-engage", action="store_true")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    if preset:
        ap.set_defaults(bounds=PRESETS[preset]["bounds"])
    args = ap.parse_args()
    bounds = floats(args.bounds)
    lo, hi = np.array(bounds[0::2]), np.array(bounds[1::2])
    real = args.sink == "viam"

    model = dargs = None
    if not args.lock_color:
        model, dargs = load_detector(args.lock, args.model, args.prompts, args.conf, args.imgsz, standard="yolov8s.pt")
    robot = ViamRobot(args.viam_address, args.arm_name, args.gripper_name, None, TableToWorld(*floats(args.table_to_world)),
                      workspace=bounds[:4], max_z_mm=bounds[5], insecure=args.viam_insecure)
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
        if model is not None:
            dets = detect(model, frame, dargs)
            cands += object_candidates(dets, args.lock)
            memo["how"] = "detector"
            if not cands and lock.locked:
                cands = roi_candidates(model, dargs, frame, lock.box, args.lock)
                memo["how"] = "zoom"
                if not cands:
                    cands = template_candidates(frame, memo["template"], lock.box, args.lock)
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
    try:
        while True:
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
                    misses = 0
                    dist_cm = rng.cm(flt.x[2] * r.frame.shape[0], r.frame.shape[1])
                else:
                    misses += 1
                    if misses > 5:
                        flt.reset()
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
                        tgt, fu = target, feats
                        status = "tracking"
                        err = tgt[list(fu)] - s[list(fu)]
                        err_txt = f"offset u{s[0]:+.2f} v{s[1]:+.2f}" + ("" if dist_cm is None else f"  {dist_cm:.0f} cm")
                        near = np.max(np.abs(err[:2])) < args.deadband
                        if args.touch and dist_cm is not None:
                            if abs(J[2]).max() < 1e-5:
                                status, err_txt = "no depth response - re-probe (p)", "cannot approach"
                                near = True
                            elif dist_cm <= stop_cm + 0.5:
                                status, touching = "TOUCH - holding", True
                            else:
                                touching = False
                                closer = 0.0                                  # creep closer only while centred
                                if np.max(np.abs(err[:2])) < 2 * args.deadband:
                                    closer = args.approach_speed / 10.0 * min(max(dtick, 0.05), 0.5)
                                d_goal = max(stop_cm, dist_cm - closer)
                                tgt = np.array([0.0, 0.0, s[2] * dist_cm / d_goal])   # size grows as the distance shrinks
                                fu, near = (0, 1, 2), False
                        if status.startswith("TOUCH") or status.startswith("no depth"):
                            cmd_pos = shared["cur"].copy()
                            if status.startswith("TOUCH"):
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
                        status, err_txt = f"searching {lost_n}/{args.search_steps}", "cup not seen"
                        move = servo_step(J, last_s * 1.5, target, feats, args.gain, args.max_step)
                    else:
                        status, err_txt = "lost - holding", "cup not seen"
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
            if now - last_print > 2.0:
                last_print = now
                print(f"   cam {grabber.fps.value:.1f} fps, detector {perc.fps.value:.1f} fps, frame age "
                      f"{now - r.t:.2f}s, {status}", flush=True)

            if show:
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
