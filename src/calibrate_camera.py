"""One-time calibration: teach the system where table points are in the fixed camera's image.

For 4+ spots spread over the workspace: put the object on the table, bring the gripper directly above its centre
(jog the arm in the Viam app, or use --jog keys here), then press SPACE. Each sample pairs the object's pixel
position with the gripper's (x, y). Press s to fit and save data/camera_calibration.json.

    scripts/run.sh src/calibrate_camera.py --viam-address <machine>.viam.cloud --lock cup
    scripts/run.sh src/calibrate_camera.py --viam-address <machine>.viam.cloud --lock cup --jog --preset lab
(use the same --lock object you will later follow; the YOLO detector finds it by name)

Keys: SPACE record sample   u undo last   s fit + save   q quit   x stop arm
      with --jog:  w/s = x +/-   a/d = y +/-   r/f = z +/-   [ ] = step size.  Jogging points the tool straight down,
      so the first jog can rotate the wrist: start from a pose that is already roughly tool-down, with the area clear.
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
from tidy.geometry import DEFAULT_CALIBRATION, fit_homography, save_calibration  # noqa: E402
from tidy.robot import TableToWorld, ViamRobot  # noqa: E402

WIN = "calibrate camera"


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
    ap.add_argument("--lock", default="cup", metavar="NAME", help="object to detect by name (default cup)")
    ap.add_argument("--lock-color", choices=sorted(COLORS), help="optional: use a colour blob instead of a named object")
    ap.add_argument("--model")
    ap.add_argument("--prompts")
    ap.add_argument("--conf", type=float)
    ap.add_argument("--lock-min-area", type=float, default=400)
    ap.add_argument("--table-to-world", default="0,0,0,0")
    ap.add_argument("--bounds", default="250,600,-300,300,0,450")
    ap.add_argument("--jog", action="store_true", help="enable keyboard jogging (moves the real arm)")
    ap.add_argument("--out", default=DEFAULT_CALIBRATION)
    if preset:
        ap.set_defaults(bounds=PRESETS[preset]["bounds"])
    args = ap.parse_args()
    bounds = floats(args.bounds)

    robot = ViamRobot(args.viam_address, args.arm_name, args.gripper_name, None, TableToWorld(*floats(args.table_to_world)),
                      workspace=bounds[:4], max_z_mm=bounds[5], insecure=args.viam_insecure)
    model = dargs = None
    if not args.lock_color:
        model, dargs = load_detector(args.lock, args.model, args.prompts, args.conf)
    lock, samples, step, size, msg = Lock(), [], 10.0, None, "jog the gripper above the object, then SPACE"
    lo, hi = np.array(bounds[0::2]), np.array(bounds[1::2])
    try:
        while True:
            frame = robot.get_frame(args.camera)
            if frame is None:
                break
            size = frame.shape[1], frame.shape[0]
            if model is not None:
                cands = object_candidates(detect(model, frame, dargs), args.lock)
            else:
                cands = color_candidates(frame, args.lock_color, args.lock_min_area)
            live = lock.update(cands, frame.shape[1])
            p = robot.get_pose()
            pos = np.array(robot.table.to_robot(p.x, p.y, p.z))
            draw_lock(frame, lock, live)
            for (u, v), (x, y) in samples:
                cv2.circle(frame, (int(u), int(v)), 6, (0, 255, 0), -1)
            rms = ""
            if len(samples) >= 4:
                try:
                    rms = f"  fit rms {fit_homography(*zip(*samples))[1]:.1f} mm"
                except ValueError:
                    rms = "  (points collinear)"
            lines = [f"samples {len(samples)} (need 4+, spread out){rms}", "gripper x{:.0f} y{:.0f} z{:.0f}".format(*pos), msg]
            for i, text in enumerate(lines):
                cv2.putText(frame, text, (14, 34 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
                cv2.putText(frame, text, (14, 34 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(WIN, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if not (lock.locked and live):
                    msg = "object not locked - cannot record"
                else:
                    uv = ((lock.box[0] + lock.box[2]) / 2, (lock.box[1] + lock.box[3]) / 2)
                    samples.append((uv, (float(pos[0]), float(pos[1]))))
                    msg = f"recorded {len(samples)}: pixel ({uv[0]:.0f},{uv[1]:.0f}) = x{pos[0]:.0f} y{pos[1]:.0f}"
                    print(msg, flush=True)
            elif key == ord("u") and samples:
                samples.pop()
                msg = "removed the last sample"
            elif key == ord("s"):
                try:
                    H, err = fit_homography(*zip(*samples))
                except ValueError as e:
                    msg = str(e)
                    continue
                save_calibration(args.out, H, err, samples, size)
                msg = f"saved {args.out} (rms {err:.1f} mm)"
                print(msg, flush=True)
            elif key == ord("x"):
                robot.stop()
            elif args.jog and key in map(ord, "wsadrf[]"):
                if key == ord("["):
                    step = max(step / 2, 2.0)
                elif key == ord("]"):
                    step = min(step * 2, 80.0)
                else:
                    d = {"w": (1, 0, 0), "s": (-1, 0, 0), "a": (0, 1, 0), "d": (0, -1, 0), "r": (0, 0, 1), "f": (0, 0, -1)}[chr(key)]
                    new = np.clip(pos + step * np.array(d), lo, hi)
                    robot.move_to(*new, 0.0, 0.0)
                msg = f"jog step {step:.0f} mm"
    finally:
        cv2.destroyAllWindows()
        robot.close()


if __name__ == "__main__":
    main()
