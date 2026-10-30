"""Reset the arm's target to where it is right now: cancel any move in progress and hold the current pose.

Use it after an interrupted run (arm_mimic, joint_test) so the arm stops chasing an old target. It only reads the
current joint angles and commands exactly those, so nothing moves apart from the arm stopping.

    scripts/run.sh src/hold_position.py --viam-address <machine>.viam.cloud
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tidy.robot import ViamRobot  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viam-address", required=True)
    ap.add_argument("--arm-name", default="arm")
    ap.add_argument("--gripper-name", default="gripper")
    ap.add_argument("--viam-insecure", action="store_true")
    args = ap.parse_args()

    robot = ViamRobot(args.viam_address, args.arm_name, args.gripper_name, insecure=args.viam_insecure)
    try:
        before = robot.get_joints()
        held = robot.hold()
        p = robot.get_pose()
        print(f"arm {robot.arm_name!r} stopped and holding its current position")
        for i, (b, h) in enumerate(zip(before, held)):
            print(f"  joint {i}: {h:8.2f} deg")
        print(f"  tool: x {p.x:.0f}  y {p.y:.0f}  z {p.z:.0f} mm  (arm frame)")
        return 0
    finally:
        robot.close()


if __name__ == "__main__":
    sys.exit(main())
