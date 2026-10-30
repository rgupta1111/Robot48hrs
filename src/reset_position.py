"""Send the arm back to its default (reset) joint position, taken from the Viam app's MoveToJointPositions panel.

    scripts/run.sh src/reset_position.py --viam-address <machine>.viam.cloud           # shows the move, asks to confirm
    scripts/run.sh src/reset_position.py --viam-address <machine>.viam.cloud --yes     # no confirmation
Other ways:  --pose --preset lab  (move to the preset's tool-down start pose)   --switch  (the reset-position resource)
Keep the area clear and a hand on the e-stop. Joint moves go straight there in joint space, so check the path is free.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_mimic import PRESETS, floats  # noqa: E402
from tidy.robot import ViamRobot  # noqa: E402

# degrees, joint 0..5 - read from the Viam app's joint panel at the arm's reset pose
RESET_JOINTS = (-0.95, 33.21, -28.59, -86.05, 8.90, -272.49)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viam-address", required=True)
    ap.add_argument("--arm-name", default="arm")
    ap.add_argument("--gripper-name", default="gripper")
    ap.add_argument("--viam-insecure", action="store_true")
    ap.add_argument("--joints", default=",".join(str(j) for j in RESET_JOINTS), help="six joint angles in degrees")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    ap.add_argument("--pose", action="store_true", help="move to the preset's start pose instead")
    ap.add_argument("--preset", choices=list(PRESETS), default="lab")
    ap.add_argument("--switch", action="store_true", help="try the machine's reset-position switch instead")
    args = ap.parse_args()

    robot = ViamRobot(args.viam_address, args.arm_name, args.gripper_name, insecure=args.viam_insecure)
    try:
        if args.pose:
            start = floats(PRESETS[args.preset]["start"])
            print(f"moving to the {args.preset} start pose x{start[0]:.0f} y{start[1]:.0f} z{start[2]:.0f} mm ...")
            robot.move_to(*start, 0.0, 0.0)
        elif args.switch:
            current, labels = robot.switch_info("reset-position")
            print(f"switch positions {list(enumerate(labels))}, now {current}")
            sys.exit("pass the position with the Viam app; this flag only lists it")
        else:
            target = floats(args.joints)
            now = robot.get_joints()
            if len(target) != len(now):
                sys.exit(f"the arm has {len(now)} joints but {len(target)} angles were given")
            print("joint    now   ->  reset    change")
            for i, (a, b) in enumerate(zip(now, target)):
                print(f"  J{i}  {a:8.2f} -> {b:8.2f}  {b - a:+8.2f} deg")
            if not args.yes and input("move the arm to the reset position? [y/N] ").strip().lower() != "y":
                print("cancelled")
                return 1
            robot.set_joints(target)
        print("done. joints now: " + ", ".join(f"{j:.1f}" for j in robot.get_joints()))
        return 0
    finally:
        robot.close()


if __name__ == "__main__":
    sys.exit(main())
