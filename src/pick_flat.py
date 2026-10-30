"""Pick up the nearest cup from a FLAT-gripper pose: short steps forward along the gripper axis, base turns to keep the cup
centred, then a force-capped close and a lift. Meant to start from a taught pick pose (gripper flat, cup roughly ahead).

    camera depth -> how far the cup is; fingertips are ~--tip-cm (10) ahead of the camera, so it closes at --close-cm (8)
    each step: turn joint 1 a little so the cup is centred left/right, then advance along the tool axis (never more than
    --step-mm), re-measure; if the cup slips out of the bottom of the picture it dead-reckons for a few steps.

Dry run by default (--sink sim): measures and prints the plan, nothing moves.
    scripts/run.sh src/pick_flat.py --viam-address sfhack16-main.ty1cywh7tq.local.viam.cloud:8080
Real arm (asks for a typed 'yes' before moving and before closing; Ctrl+C stops):
    scripts/run.sh src/pick_flat.py --viam-address ... --sink viam
"""
import argparse
import math
import os
import struct
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_detect import detect, load_detector, object_candidates  # noqa: E402
from mimic.pickup import advance_along  # noqa: E402
from mimic.size import box_depth_mm  # noqa: E402
from tidy.robot import ViamRobot  # noqa: E402


def read_depth(robot, cam):
    """Latest aligned depth image in mm (H x W), or None."""
    for im in robot._run(robot._cams[cam].get_images())[0]:
        if "dep" in str(im.mime_type).lower() and im.data[:8] == b"DEPTHMAP":
            w, h = struct.unpack(">QQ", im.data[8:24])
            return np.frombuffer(im.data[24:24 + w * h * 2], ">u2").reshape(h, w).astype(float)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viam-address", required=True)
    ap.add_argument("--object", default="cup")
    ap.add_argument("--camera", default="cam")
    ap.add_argument("--sink", choices=["sim", "viam"], default="sim", help="sim = dry run, nothing moves")
    ap.add_argument("--bounds", default="250,720,-300,300,-13,450", help="xmin,xmax,ymin,ymax,zmin,zmax mm for the tool")
    ap.add_argument("--tip-cm", type=float, default=10.0, help="fingertips are this far ahead of the camera")
    ap.add_argument("--close-cm", type=float, default=8.0, help="close when the cup is this far from the camera")
    ap.add_argument("--step-mm", type=float, default=30.0, help="longest forward step")
    ap.add_argument("--center-tol", type=float, default=0.04, help="turn joint 1 until the cup is within this of centre (u)")
    ap.add_argument("--max-torque", type=int, default=30, help="grip force limit 1-100%% (UFactory G2 gripper only)")
    ap.add_argument("--lift-mm", type=float, default=120.0)
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--yes", action="store_true", help="skip the typed confirmations (not recommended)")
    args = ap.parse_args()

    real, b = args.sink == "viam", [float(v) for v in args.bounds.split(",")]
    lo, hi = np.array(b[0::2]), np.array(b[1::2])
    model, dargs = load_detector(args.object)
    robot = ViamRobot(args.viam_address, "arm", "gripper", workspace=b[:4], max_z_mm=b[5])
    _, intr = robot.camera_info(args.camera)
    robot.get_frame(args.camera)                                  # creates the camera client

    def confirm(q):
        return args.yes or input(f"{q} Type 'yes' to continue: ").strip().lower() == "yes"

    def position():
        p = robot.get_pose()
        return np.array(robot.table.to_robot(p.x, p.y, p.z))

    def measure():
        """-> (u, v, depth cm or None, box touches the bottom edge) of the nearest cup, or None."""
        frame = robot.get_frame(args.camera)
        cands = object_candidates(detect(model, frame, dargs), args.object)
        if not cands:
            return None
        h, w = frame.shape[:2]
        depth = read_depth(robot, args.camera)
        best = None
        for x1, y1, x2, y2, conf, _ in cands:
            d = None if depth is None else box_depth_mm(cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST), (x1, y1, x2, y2))
            if d is not None and (best is None or d < best[2] * 10):
                best = ((x1 + x2) / 2 / w - 0.5, (y1 + y2) / 2 / h - 0.5, d / 10.0, y2 > h - 5, w)
        return best

    try:
        m = measure()
        if m is None or m[2] is None:
            raise SystemExit("no cup with a depth reading in view; point the camera at it first")
        d_cm = m[2]
        print(f"cup: {d_cm:.1f} cm from the camera, offset u{m[0]:+.2f} v{m[1]:+.2f}; tool at x{position()[0]:.0f} "
              f"y{position()[1]:.0f} z{position()[2]:.0f} mm", flush=True)
        total = max(d_cm - args.close_cm, 0.0) * 10.0
        print(f"plan: advance about {total:.0f} mm along the gripper axis in steps of <= {args.step_mm:.0f} mm, centring with joint 1, "
              f"then close (force cap {args.max_torque}%) at {args.close_cm:.0f} cm and lift {args.lift_mm:.0f} mm", flush=True)
        if not real:
            print("DRY RUN: nothing moved.")
            return
        if not confirm(f"The arm WILL move forward about {total:.0f} mm to the cup (x limit {hi[0]:.0f} mm). E-stop in reach?"):
            return
        robot.set_gripper(True)
        time.sleep(1.0)
        # phase 1: centre the cup left/right with the base joint while it is still in view (it drops out of the bottom
        # of the picture once the camera moves forward, so this must happen BEFORE the forward move)
        sign, last_abs_u = -1.0, None
        for it in range(1, 7):
            m = measure()
            if m is None or m[2] is None:
                raise SystemExit("cannot see the cup to centre on it; stopping before any forward move")
            u, v, d_cm, clipped, w = m
            if abs(u) <= args.center_tol:
                print(f"[centre {it}] cup u{u:+.2f}: centred, {d_cm:.1f} cm away", flush=True)
                break
            if last_abs_u is not None and abs(u) > last_abs_u + 0.02:
                sign = -sign                                         # that made it worse: turn the other way
            lateral_mm = u * (intr["w"] / intr["fx"] if intr else 1.4) * d_cm * 10.0      # sideways offset of the cup, mm
            dth = float(np.clip(sign * math.degrees(lateral_mm / max(math.hypot(*position()[:2]), 1.0)) * 0.7, -4.0, 4.0))
            robot.set_joint(0, robot.get_joints()[0] + dth)
            last_abs_u = abs(u)
            print(f"[centre {it}] cup u{u:+.2f} at {d_cm:.1f} cm: turned joint 1 by {dth:+.1f} deg", flush=True)
            time.sleep(0.6)
        else:
            raise SystemExit("could not centre the cup in 6 turns; stopping before any forward move")
        # phase 2: forward along the gripper axis. The cup slips out of the bottom of the picture on the way, so the
        # distance is dead-reckoned (updated from the camera whenever the cup is visible and the reading is believable)
        total_mm = max(d_cm - args.close_cm, 0.0) * 10.0
        if not confirm(f"Centred. ADVANCE {total_mm:.0f} mm along the gripper axis (blind once the cup leaves the picture)."):
            return
        for step in range(1, args.max_steps + 1):
            if d_cm <= args.close_cm + 0.5:
                break
            adv = min(args.step_mm, (d_cm - args.close_cm) * 10.0)
            pn = robot.get_pose()
            robot._orient = None
            goal = np.clip(advance_along(position(), (pn.o_x, pn.o_y, pn.o_z), adv), lo, hi)
            if np.linalg.norm(goal - position()) < adv - 2.0:
                raise SystemExit("the next step is limited by the workspace bounds; stopping short of the cup")
            robot.move_xyz(*goal, wait=True)
            d_cm -= adv / 10.0
            time.sleep(0.4)
            m = measure()
            seen = ""
            if m is not None and m[2] is not None and abs(m[2] - d_cm) <= 4.0:
                d_cm, seen = m[2], f" (camera agrees: {m[2]:.1f} cm, u{m[0]:+.2f})"
            print(f"[advance {step:2}] {adv:.0f} mm -> about {d_cm:.1f} cm to go{seen}; tool x{goal[0]:.0f} y{goal[1]:.0f} z{goal[2]:.0f}", flush=True)
        else:
            raise SystemExit("did not reach the cup within the step limit")
        if not confirm(f"At about {d_cm:.0f} cm from the cup. CLOSE the gripper (force cap {args.max_torque}%) and lift."):
            return
        try:
            robot.gripper_command({"set_gripper_torque": args.max_torque})
        except Exception as e:
            print(f"could not cap the grip force ({type(e).__name__}); closing at the default force")
        robot.set_gripper(False)
        time.sleep(1.5)
        pos = position()
        robot.move_xyz(pos[0], pos[1], min(pos[2] + args.lift_mm, hi[2]), wait=True)
        try:
            print("holding something:", bool(robot._run(robot.gripper.is_holding_something()).value))
        except Exception:
            pass
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
