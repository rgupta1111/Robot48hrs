"""Move one joint of a Viam arm +45 deg, then -45 deg, then back -- a quick check that this code can
drive the arm you see in the Viam app's 3D SCENE tab (no camera or hand tracking involved).

Open the machine's CONTROL / 3D SCENE tab in the browser first, then (VIAM_API_KEY / VIAM_API_KEY_ID set):
    python src/joint_test.py --viam-address <machine>.viam.cloud
    python src/joint_test.py --viam-address <machine>.viam.cloud --joint wrist --degrees 30
    python src/joint_test.py --viam-address <machine>.viam.cloud --joint all --degrees 20

--joint is a number (as the Viam arm panel numbers them: UR5e 0 = base ... 5 = last wrist joint) or a role,
counted from the end of the arm:
    pinch  = the last joint          (the one the thumb-index pinch drives)
    wrist  = the second-last joint   (wrist movement)
    elbow  = the third-last joint    (elbow movement)
    all    = pinch, then wrist, then elbow, one after the other
While each move runs it prints the joint angle every --poll seconds together with the arm's is_moving flag, so a
slow arm (low speed limit) or a move that never starts is visible. Every joint ends back at the angle it started
from, even if the run is interrupted.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tidy.robot import ViamRobot  # noqa: E402

ROLES = {"pinch": 1, "wrist": 2, "elbow": 3}  # position counted from the last joint


def role_of(index, n):
    for name, back in ROLES.items():
        if index == n - back:
            return name
    return ""


def show(joints, title, mark=None):
    n = len(joints)
    print(f"  {title}")
    for i, v in enumerate(joints):
        role = role_of(i, n)
        print(f"    joint {i}: {v:8.2f} deg   {role:<6}{'  <-- testing' if i == mark else ''}")


def resolve(spec, n):
    if spec == "all":
        return [n - ROLES[r] for r in ("pinch", "wrist", "elbow") if n - ROLES[r] >= 0]
    if spec in ROLES:
        if n - ROLES[spec] < 0:
            sys.exit(f"the arm has only {n} joints, so there is no {spec} joint")
        return [n - ROLES[spec]]
    index = int(spec)
    if not 0 <= index < n:
        sys.exit(f"the arm has {n} joints (0-{n - 1}); --joint {spec} does not exist")
    return [index]


def settle(robot, index, target, args):
    """Poll until the arm reports it has stopped (or --timeout), printing the joint as it goes."""
    t0 = time.time()
    time.sleep(0.2)
    still = 0
    while time.time() - t0 < args.timeout:
        moving = robot.is_moving()
        now = robot.get_joints()
        print(f"        t={time.time() - t0:4.1f}s  joint {index} = {now[index]:8.2f}  "
              f"(target {target:8.2f}, {target - now[index]:+7.2f} to go)  is_moving={moving}")
        still = 0 if moving else still + 1
        if still >= 2:
            break
        time.sleep(args.poll)
    time.sleep(args.pause)
    return robot.get_joints()


def test_joint(robot, index, args):
    start = robot.get_joints()
    n = len(start)
    base = start[index]
    print(f"\n=== joint {index} ({role_of(index, n) or 'no role'}): {base:.2f} deg, moving +/-{args.degrees:g} deg ===")
    show(start, "joints before:", index)
    ok = True
    try:
        for label, target in (("+", base + args.degrees), ("-", base - args.degrees), ("start", base)):
            before = robot.get_joints()
            print(f"\n  move {label}: commanding joint {index} {before[index]:.2f} -> {target:.2f} "
                  f"(other joints left as they are)")
            t0 = time.time()
            robot.set_joint(index, target)
            print(f"      command accepted after {time.time() - t0:.2f}s")
            now = settle(robot, index, target, args)
            moved = now[index] - before[index]
            want = target - before[index]
            others = [(i, a - b) for i, (a, b) in enumerate(zip(now, start)) if i != index and abs(a - b) >= 1.0]
            good = abs(now[index] - target) < 1.0 and not others
            ok &= good
            print(f"      result: joint {index} = {now[index]:.2f}; moved {moved:+.2f} of {want:+.2f} commanded  "
                  f"{'ok' if good else 'MISMATCH'}")
            if others:
                print(f"      other joints drifted: {[(i, round(d, 1)) for i, d in others]}")
            if abs(want) >= 1.0 and abs(moved) < 0.2 * abs(want):
                print("      the joint barely moved: check the arm is not in a fault / e-stop, that its speed limit "
                      "is not ~0, and the joint limits in its config; try a larger --degrees or --timeout")
            elif not good:
                print("      it moved but not all the way: raise --timeout (slow arm) or check the joint limits")
        show(robot.get_joints(), "joints after:", index)
    finally:
        try:
            robot.set_joint(index, base)  # leave the joint where it started
        except Exception as e:
            print(f"  could not restore joint {index}: {e}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viam-address", required=True)
    ap.add_argument("--arm-name", default="arm")
    ap.add_argument("--gripper-name", default="gripper")
    ap.add_argument("--joint", default="pinch", help="a joint number, or pinch / wrist / elbow / all (default pinch)")
    ap.add_argument("--degrees", type=float, default=45.0, help="how far to move each way")
    ap.add_argument("--pause", type=float, default=1.0, help="extra seconds to wait after each move has stopped")
    ap.add_argument("--timeout", type=float, default=30.0, help="max seconds to wait for each move to finish")
    ap.add_argument("--poll", type=float, default=0.5, help="seconds between progress lines while moving")
    ap.add_argument("--viam-insecure", action="store_true", help="local server without an API key")
    args = ap.parse_args()

    robot = ViamRobot(args.viam_address, args.arm_name, args.gripper_name, insecure=args.viam_insecure)
    try:
        n = len(robot.get_joints())
        print(f"connected: arm {robot.arm_name!r} with {n} joints "
              f"(pinch = joint {n - 1}, wrist = joint {n - 2}, elbow = joint {n - 3})")
        ok = True
        for index in resolve(args.joint, n):
            ok &= test_joint(robot, index, args)
        print("\nPASS: the arm followed every move" if ok else "\nFAIL: the arm did not reach a target (see above)")
        return 0 if ok else 1
    finally:
        robot.close()


if __name__ == "__main__":
    sys.exit(main())
