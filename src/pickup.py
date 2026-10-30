"""Pick up a named object with the gripper (camera on the gripper, detection by a Viam vision service).

Sequence: open gripper -> centre the object in the image (visual servoing, response learned by small probe moves) ->
descend while staying centred -> close gripper at --grasp-distance-mm -> lift. See mimic/pickup.py.

DRY RUN by default (--sink sim): reads the camera and prints what it WOULD do; the arm and gripper do not move.
Real arm (low speed limit in the Viam config, hand on the e-stop):
    scripts/run.sh src/pickup.py --viam-address <machine>.viam.cloud --preset lab --vision cup-detector --object cup
    scripts/run.sh src/pickup.py --viam-address <machine>.viam.cloud --preset lab --vision cup-detector --object cup --sink viam
With --sink viam it asks for a typed "yes" before it moves at all and again before it closes the gripper.
Tune --grasp-distance-mm (camera-to-object distance at which the gripper surrounds the object) and --target-uv (where the
object appears in the image when the gripper is over it; the camera is offset from the gripper) in dry runs first.
Distance comes from the object's apparent height: d = fy * --object-height-mm / height_px.
Ctrl+C stops the arm.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_mimic import PRESETS, floats  # noqa: E402
from mimic.pickup import PickupConfig, PickupPlanner  # noqa: E402
from mimic.servo import estimate_jacobian, features  # noqa: E402
from tidy.robot import TableToWorld, ViamRobot  # noqa: E402


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", choices=list(PRESETS))
    preset = pre.parse_known_args()[0].preset
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS))
    ap.add_argument("--viam-address", required=True)
    ap.add_argument("--vision", required=True, help="name of the Viam vision service that detects the object")
    ap.add_argument("--object", default="cup", help="class name to pick up (matched against the detector's labels)")
    ap.add_argument("--camera", default="cam")
    ap.add_argument("--arm-name", default="arm")
    ap.add_argument("--gripper-name", default="gripper")
    ap.add_argument("--viam-insecure", action="store_true")
    ap.add_argument("--sink", choices=["sim", "viam"], default="sim", help="sim = dry run, nothing moves")
    ap.add_argument("--table-to-world", default="0,0,0,0")
    ap.add_argument("--bounds", default="250,600,-300,300,0,450", help="xmin,xmax,ymin,ymax,zmin,zmax mm; moves are clamped")
    ap.add_argument("--min-confidence", type=float, default=0.35)
    ap.add_argument("--object-height-mm", type=float, default=100.0)
    ap.add_argument("--grasp-distance-mm", type=float, default=150.0)
    ap.add_argument("--target-uv", default="0,0", help="image offset (fractions of the frame) of the object at grasp")
    ap.add_argument("--probe-mm", type=float, default=20.0)
    ap.add_argument("--gain", type=float, default=0.4)
    ap.add_argument("--max-step", type=float, default=30.0)
    ap.add_argument("--descend-step", type=float, default=25.0)
    ap.add_argument("--lift-mm", type=float, default=120.0)
    ap.add_argument("--settle", type=float, default=0.8, help="seconds to wait after a move before reading the camera")
    ap.add_argument("--max-cycles", type=int, default=200)
    ap.add_argument("--yes", action="store_true", help="skip the typed confirmations (not recommended)")
    if preset:
        ap.set_defaults(bounds=PRESETS[preset]["bounds"])
    args = ap.parse_args()

    real = args.sink == "viam"
    bounds = floats(args.bounds)
    robot = ViamRobot(args.viam_address, args.arm_name, args.gripper_name, None, TableToWorld(*floats(args.table_to_world)),
                      workspace=bounds[:4], max_z_mm=bounds[5], insecure=args.viam_insecure)
    _, intr = robot.camera_info(args.camera)
    fy = intr["fy"] if intr else None
    lo, hi = np.array(bounds[0::2], float), np.array(bounds[1::2], float)

    def see():
        """-> (features, distance mm or None) of the best matching detection, or (None, None)."""
        frame, dets = robot.detect(args.vision, args.camera, args.min_confidence)
        dets = [d for d in dets if args.object.lower() in d.class_name.lower()]
        if not dets:
            return None, None
        d = max(dets, key=lambda d: d.confidence)
        h_img, w_img = frame.shape[:2] if frame is not None else ((intr["h"], intr["w"]) if intr else (720, 1280))
        s = features((d.x_min, d.y_min, d.x_max, d.y_max), w_img, h_img)
        dist = fy * args.object_height_mm / (s[2] * h_img) if fy and s[2] > 0 else None
        return s, dist

    def position():
        p = robot.get_pose()
        return np.array(robot.table.to_robot(p.x, p.y, p.z))

    def confirm(question):
        return args.yes or input(f"{question} Type 'yes' to continue: ").strip().lower() == "yes"

    def move_by(delta):
        target = np.clip(position() + np.asarray(delta, float), lo, hi)
        robot.move_xyz(*target, wait=True)
        time.sleep(args.settle)

    def probe():
        """Learn how the object's image features respond to moving +x, +y, +z (then undo each move)."""
        s0 = [see()[0] for _ in range(3)]
        s0 = [s for s in s0 if s is not None]
        if not s0:
            raise SystemExit("probe: object not visible")
        s0, after = np.mean(s0, axis=0), []
        for i in range(3):
            d = np.zeros(3)
            d[i] = args.probe_mm
            move_by(d)
            s, _ = see()
            move_by(-d)
            if s is None:
                raise SystemExit("probe: lost the object during a probe move; centre it in view first")
            after.append(s)
        J = estimate_jacobian(s0, after, [args.probe_mm] * 3)
        print("learned image response J (per mm):\n", np.round(J, 5))
        if np.abs(J[:2]).max() < 1e-5:
            raise SystemExit("probe: the object did not move in the image; cannot servo")
        return J

    try:
        if real:
            if not confirm(f"The arm WILL move to pick up '{args.object}' (bounds {bounds}). E-stop in reach?"):
                return
            J = probe()
        else:
            J = np.array([[0.0, -0.004, 0.0], [-0.003, 0.0, 0.0], [0.0, 0.0, 0.001]])   # placeholder, never used to move
            print("DRY RUN: nothing will move; showing what the planner would do each cycle")
        cfg = PickupConfig(target_uv=floats(args.target_uv), gain=args.gain, max_step_mm=args.max_step,
                           descend_step_mm=args.descend_step, grasp_distance_mm=args.grasp_distance_mm,
                           lift_mm=args.lift_mm, max_cycles=args.max_cycles)
        planner = PickupPlanner(J, cfg)
        while True:
            s, dist = see()
            act = planner.step(s, dist)
            seen = "not seen" if s is None else f"u{s[0]:+.2f} v{s[1]:+.2f} h{s[2]:.2f}" + (f" {dist:.0f}mm" if dist else "")
            print(f"[{planner.cycles:3}] {planner.state:8} {act.kind:10} {act.note:28} {seen}"
                  + (f"  move {act.move[0]:+.0f} {act.move[1]:+.0f} {act.move[2]:+.0f}" if act.kind == "move" else ""), flush=True)
            if act.kind in ("done", "abort"):
                break
            if not real:
                if act.kind == "grip_close" or planner.state in ("DONE", "ABORT"):
                    pass
                time.sleep(0.3)
                continue
            if act.kind == "grip_open":
                robot.set_gripper(True)
                time.sleep(1.0)
            elif act.kind == "grip_close":
                if not confirm("About to CLOSE the gripper."):
                    break
                robot.set_gripper(False)
                time.sleep(1.5)
            elif act.kind == "move":
                move_by(act.move)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        if real:
            try:
                robot.stop()
            except Exception as e:
                print("stop failed:", e)
        robot.close()


if __name__ == "__main__":
    main()
