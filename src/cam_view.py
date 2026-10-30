"""Show the live feed of a Viam camera component (default "cam") in a window on this machine.

    scripts/run.sh src/cam_view.py --viam-address <machine>.viam.cloud
    scripts/run.sh src/cam_view.py --viam-address <machine>.viam.cloud --camera cam --fps 15
Object detection runs on the machine's own vision service (no local model needed):
    scripts/run.sh src/cam_view.py --viam-address <machine>.viam.cloud --detect            (uses yolo-detector)
    scripts/run.sh src/cam_view.py --viam-address <machine>.viam.cloud --detect yolo-detector --min-conf 0.3
Distance to every object (including ones YOLO has no class for, like a blue cylinder), from the machine's
segmenter + the camera's depth:
    scripts/run.sh src/cam_view.py --viam-address <machine>.viam.cloud --objects            (uses object-segmenter)
Keys: q / Esc quit, s save a snapshot to data/. (VIAM_API_KEY / VIAM_API_KEY_ID set, or --viam-insecure.)
"""
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tidy.robot import ViamRobot  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viam-address", required=True)
    ap.add_argument("--camera", default="cam", help="camera component name on the machine")
    ap.add_argument("--arm-name", default="arm")
    ap.add_argument("--fps", type=float, default=15.0, help="max frames per second to request")
    ap.add_argument("--detect", nargs="?", const="yolo-detector", metavar="VISION_SERVICE",
                    help="draw boxes from this Viam vision service (default yolo-detector)")
    ap.add_argument("--objects", nargs="?", const="object-segmenter", metavar="VISION_SERVICE",
                    help="draw every segmented object with its distance from the camera (needs a depth camera)")
    ap.add_argument("--min-conf", type=float, default=0.4, help="hide detections below this confidence")
    ap.add_argument("--viam-insecure", action="store_true")
    args = ap.parse_args()

    robot = ViamRobot(args.viam_address, args.arm_name, "gripper", insecure=args.viam_insecure)
    try:
        print("cameras on this machine:", robot.camera_names() or "none")
        intr = None
        if args.objects:
            has_depth, intr = robot.camera_info(args.camera)
            print(f"camera depth/point cloud: {'yes' if has_depth else 'NO - distances need a depth camera'};"
                  f" intrinsics: {'yes' if intr else 'no (labels shown as a list)'}")
        if args.detect:
            print("vision services:", robot.vision_names() or "none", "-> using", args.detect)
        win, n, t_prev = f"viam camera: {args.camera}", 0, time.monotonic()
        while True:
            t0 = time.monotonic()
            dets, objs = [], []
            if args.objects:
                frame, objs = robot.objects(args.objects, args.camera)
            elif args.detect:
                frame, dets = robot.detect(args.detect, args.camera, args.min_conf)
            else:
                frame = robot.get_frame(args.camera)
            if frame is None:
                print("could not decode a frame")
                break
            fps = 1.0 / max(t0 - t_prev, 1e-6)
            t_prev = t0
            for d in dets:
                p1, p2 = (int(d.x_min), int(d.y_min)), (int(d.x_max), int(d.y_max))
                cv2.rectangle(frame, p1, p2, (0, 200, 255), 2)
                label = f"{d.class_name} {d.confidence:.2f}"
                cv2.putText(frame, label, (p1[0], max(p1[1] - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(frame, label, (p1[0], max(p1[1] - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            for i, (label, (x, y, z)) in enumerate(sorted(objs, key=lambda o: sum(c * c for c in o[1]))):
                dist_cm = (x * x + y * y + z * z) ** 0.5 / 10.0
                text = f"{label} {dist_cm:.0f} cm"
                if intr and z > 1:
                    u, v = int(intr["fx"] * x / z + intr["cx"]), int(intr["fy"] * y / z + intr["cy"])
                    cv2.circle(frame, (u, v), 8, (0, 200, 255), -1)
                    cv2.putText(frame, text, (u + 12, v), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                    cv2.putText(frame, text, (u + 12, v), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                else:
                    cv2.putText(frame, text, (10, 60 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            if args.objects and n % 10 == 0:
                print("objects:", "; ".join(f"{l} x{x:.0f} y{y:.0f} z{z:.0f} mm ({(x*x+y*y+z*z)**0.5/10:.0f} cm)"
                                            for l, (x, y, z) in objs) or "none")
            if args.detect and n % 10 == 0:
                print("detected:", ", ".join(f"{d.class_name} {d.confidence:.2f}" for d in dets) or "nothing")
            cv2.putText(frame, f"{args.camera}  {frame.shape[1]}x{frame.shape[0]}  {fps:.0f} fps", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                os.makedirs("data", exist_ok=True)
                path = f"data/{args.camera}_{int(time.time())}.jpg"
                cv2.imwrite(path, frame)
                print("saved", path)
            n += 1
            time.sleep(max(0.0, 1.0 / args.fps - (time.monotonic() - t0)))
    finally:
        cv2.destroyAllWindows()
        robot.close()


if __name__ == "__main__":
    main()
