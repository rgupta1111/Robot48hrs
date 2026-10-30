"""The gripper follows a locked object (found by name with YOLO, e.g. a cup) seen by the machine's fixed camera.

Pipeline: camera frame -> YOLO detection of --lock NAME -> lock -> pixel -> table (x, y) via the calibration homography
          -> target = table point + hover height -> proportional move of the gripper (tool pointing down).

No calibration needed by default: when you engage, the gripper stays where it is and then copies the object's motion
in the image (mm_per_px scale, see --mm-per-px/--swap-xy/--flip-u/--flip-v to match your camera). For an exact
"gripper goes to the object" use --calibrated after running calibrate_camera.py.
Safe first run - shows where the gripper WOULD go, sends nothing to the arm:
    scripts/run.sh src/follow_object.py --viam-address <machine>.viam.cloud --preset lab --lock cup
Move the real arm (low speed limit in the Viam config, hand on the e-stop):
    scripts/run.sh src/follow_object.py --viam-address <machine>.viam.cloud --preset lab --lock-color blue --sink viam

Keys (click the window first): space engage / disengage (nothing moves until you engage), c re-zero (current object
position = current gripper position), x emergency stop,
l unlock, r re-acquire, click a box to lock it (--lock NAME), q / Esc quit.
The camera is read over the network, so expect only a few updates per second: moves are small steps, closed on the
arm's real pose each update.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_mimic import PRESETS, floats  # noqa: E402
from local_detect import (COLORS, Lock, color_candidates, detect, draw_lock, load_detector,  # noqa: E402
                          object_candidates)
from mimic.controller import Command  # noqa: E402
from mimic.follower import FollowConfig, ObjectFollower  # noqa: E402
from mimic.sinks import SimSink, ViamSink  # noqa: E402
from tidy.geometry import DEFAULT_CALIBRATION, load_calibration, pixel_to_xy, relative_target  # noqa: E402
from tidy.robot import TableToWorld, ViamRobot  # noqa: E402

WIN = "follow object"


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
    ap.add_argument("--sink", choices=["sim", "viam"], default="sim", help="sim only prints the planned moves")
    ap.add_argument("--calibrated", action="store_true", help="use calibrate_camera.py's table mapping instead of "
                    "following the object's relative motion")
    ap.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    ap.add_argument("--mm-per-px", type=float, default=0.8, help="robot mm moved per pixel the object moves")
    ap.add_argument("--swap-xy", action="store_true", help="camera turned 90 degrees: image right/down map to x/y")
    ap.add_argument("--flip-u", action="store_true", help="reverse the image left/right direction")
    ap.add_argument("--flip-v", action="store_true", help="reverse the image up/down direction")
    ap.add_argument("--table-to-world", default="0,0,0,0", help="dx,dy,dz,yaw_deg of the robot frame in the arm frame")
    ap.add_argument("--lock", default="cup", metavar="NAME",
                    help="object to follow, matched against the model's class names (default cup). A name the standard "
                    "model does not know (e.g. 'blue cylinder') automatically uses the open-vocabulary model")
    ap.add_argument("--lock-color", choices=sorted(COLORS), help="optional: follow a colour blob instead of a named object")
    ap.add_argument("--model", help="YOLO weights (default: yolov8s, or yolov8s-worldv2 for unknown names)")
    ap.add_argument("--prompts", help="open-vocabulary prompts for a *world*.pt model (default: the --lock name)")
    ap.add_argument("--conf", type=float, help="detection confidence threshold (default 0.25, 0.15 open-vocabulary)")
    ap.add_argument("--lock-min-area", type=float, default=400)
    ap.add_argument("--table-z-mm", type=float, default=-123.0, help="table top height in the robot frame")
    ap.add_argument("--object-height-mm", type=float, default=100.0)
    ap.add_argument("--hover-mm", type=float, default=100.0, help="gripper height above the object's top (0 = touch)")
    ap.add_argument("--start", default="400,0,250", help="x,y,z used by --sink sim as the gripper's start position")
    ap.add_argument("--bounds", default="250,600,-300,300,0,450")
    ap.add_argument("--max-speed", type=float, default=100.0)
    ap.add_argument("--max-step", type=float, default=60.0)
    ap.add_argument("--deadband", type=float, default=10.0)
    ap.add_argument("--gain", type=float, default=0.5)
    ap.add_argument("--smoothing", type=float, default=0.5)
    ap.add_argument("--auto-engage", action="store_true")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    if preset:
        d = PRESETS[preset]
        ap.set_defaults(start=d["start"], bounds=d["bounds"], max_speed=d["max_speed"])
    args = ap.parse_args()

    H = None
    if args.calibrated:
        H, cal = load_calibration(args.calibration)
        print(f"calibration: rms {cal['rms_mm']:.1f} mm over {len(cal['points'])} points")
    else:
        print(f"no calibration: the gripper copies the object's motion at {args.mm_per_px} mm/px "
              f"(engage, then move the object; use --flip-u/--flip-v/--swap-xy if it goes the wrong way)")
    bounds = floats(args.bounds)
    cfg = FollowConfig(bounds=bounds, deadband_mm=args.deadband, gain=args.gain, max_step_mm=args.max_step,
                       max_speed_mm_s=args.max_speed, smoothing=args.smoothing)
    follower = ObjectFollower(cfg)
    z_target = args.table_z_mm + args.object_height_mm + args.hover_mm
    if args.calibrated and not bounds[4] <= z_target <= bounds[5]:
        print(f"note: target height {z_target:.0f} mm is outside the z bounds {bounds[4]:.0f}..{bounds[5]:.0f}; it will be clamped")

    model = dargs = None
    if not args.lock_color:
        model, dargs = load_detector(args.lock, args.model, args.prompts, args.conf)

    table = TableToWorld(*floats(args.table_to_world))
    robot = ViamRobot(args.viam_address, args.arm_name, args.gripper_name, None, table,
                      workspace=bounds[:4], max_z_mm=bounds[5], insecure=args.viam_insecure)
    sink = ViamSink(robot) if args.sink == "viam" else SimSink()
    sim_pos = np.array(floats(args.start), float)

    def current():
        if args.sink != "viam":
            return tuple(sim_pos)
        p = robot.get_pose()
        return robot.table.to_robot(p.x, p.y, p.z)

    lock, state, ref = Lock(), {"click": None, "dets": []}, None   # ref: (object pixel, gripper pos) at engage
    cv2_ok = not args.no_display
    if cv2_ok:
        cv2.namedWindow(WIN)
        cv2.setMouseCallback(WIN, lambda e, x, y, f, p: state.update(click=(x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
    t0, n, status = time.monotonic(), 0, "idle"
    if args.auto_engage:
        follower.engage(0.0)
    try:
        while True:
            frame = robot.get_frame(args.camera)
            if frame is None or (args.max_frames and n >= args.max_frames):
                break
            t = time.monotonic() - t0
            n += 1
            cands = []
            if model is not None:
                state["dets"] = detect(model, frame, dargs)
                cands += object_candidates(state["dets"], args.lock)
            if state["click"]:
                mx, my = state["click"]
                state["click"] = None
                hit = [d for d in state["dets"] if d[0] <= mx <= d[2] and d[1] <= my <= d[3]]
                if hit:
                    d = min(hit, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
                    lock.pin(d[:4], d[5])
            if args.lock_color:
                cands += color_candidates(frame, args.lock_color, args.lock_min_area)
            live = lock.update(cands, frame.shape[1])

            target = xy = None
            if lock.locked and live:
                u, v = (lock.box[0] + lock.box[2]) / 2, (lock.box[1] + lock.box[3]) / 2
                cur_now = current()
                if args.calibrated:
                    xy = pixel_to_xy(H, (u, v))
                    target = (float(xy[0]), float(xy[1]), z_target)
                else:
                    if ref is None and follower.engaged:
                        ref = ((u, v), cur_now)   # zero point: where the object is now vs where the gripper is now
                    if ref is not None:
                        target = relative_target(ref[0], ref[1], (u, v), args.mm_per_px, args.swap_xy, args.flip_u,
                                                 args.flip_v)
                        xy = target[:2]
            if not follower.engaged:
                ref = None
            cur = current()
            if follower.engaged:
                new, status = follower.update(target, cur, t)
                if new is not None:
                    sink.send(Command(*new, 0.0, 0.0))
                    if args.sink != "viam":
                        sim_pos = np.array(new)
            else:
                status = "idle"
            if getattr(sink, "error", None):
                print("ARM ERROR:", sink.error)
                break
            print(f"[{t:6.1f}s] {status:10} " + (f"object px ({u:.0f},{v:.0f}) -> target x{xy[0]:.0f} y{xy[1]:.0f}; "
                  if target else "object not seen; ") + "gripper x{:.0f} y{:.0f} z{:.0f}".format(*cur), flush=True)

            if cv2_ok:
                view = frame
                draw_lock(view, lock, live)
                lines = [("ENGAGED" if follower.engaged else "IDLE - space to engage") + f" | {status} | sink {args.sink}",
                         "gripper x{:.0f} y{:.0f} z{:.0f} mm".format(*cur)]
                if target:
                    lines.append("target  x{:.0f} y{:.0f} z{:.0f} mm".format(*target))
                col = (0, 200, 0) if follower.engaged else (0, 0, 220)
                for i, text in enumerate(lines):
                    pos = (14, 34 + 34 * i)
                    cv2.putText(view, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
                    cv2.putText(view, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, col if i == 0 else (255, 255, 255), 2)
                cv2.imshow(WIN, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    follower.disengage() if follower.engaged else follower.engage(t)
                elif key == ord("c"):
                    ref = None            # re-zero on the next frame the object is seen
                elif key == ord("x"):
                    follower.disengage()
                    sink.stop()
                elif key == ord("l"):
                    lock.unlock()
                    lock.enabled = False
                elif key == ord("r"):
                    lock.unlock()
                    lock.enabled = True
    finally:
        follower.disengage()
        try:
            sink.stop()
        finally:
            sink.close() if args.sink == "viam" else robot.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
