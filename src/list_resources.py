"""List everything the machine offers (components and services), grouped by type, so you can find exact names.

    scripts/run.sh src/list_resources.py --viam-address <machine>.viam.cloud
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tidy.robot import ViamRobot  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viam-address", required=True)
    ap.add_argument("--viam-insecure", action="store_true")
    args = ap.parse_args()
    robot = ViamRobot(args.viam_address, insecure=args.viam_insecure)
    try:
        groups = defaultdict(list)
        for r in robot._robot.resource_names:
            groups[f"{r.namespace}:{r.type}:{r.subtype}"].append(r.name)
        for kind in sorted(groups):
            print(f"{kind:40} {', '.join(sorted(groups[kind]))}")
    finally:
        robot.close()


if __name__ == "__main__":
    main()
