#!/usr/bin/env bash
# Reset the arm (asks y/N), then open the control page and WAIT there: type "pick up the cup" or "pick up the bottle" in its chat.
# After that the whole pick runs with no other prompts: approach, grip, lift, scan for a person, hold out. The page asks once more before the
# release (press Yes). Then it returns to the reset pose and asks for the next object; it keeps running until you press STOP. Keep the e-stop in reach. Extra arguments go to viam_pick.py.
cd "$(dirname "$0")/.." || exit 1
scripts/reset_real.sh || exit 1
exec scripts/viam_pick_real.sh cup --ui --voxel --approach horizontal --no-sweep --aim-trim-mm 0,-50,0 --scan-person --box-obstacle --yes --keep-running --retries 0 "$@"
