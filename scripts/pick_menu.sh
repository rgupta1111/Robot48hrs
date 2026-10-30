#!/usr/bin/env bash
# Choose cup or bottle, reset the arm (asks y/N), then run the whole pick: approach, grip, lift, scan for a person, hand over.
# No other confirmations; it asks once more before holding the object out. Keep the e-stop in reach. Extra arguments go to viam_pick.py.
cd "$(dirname "$0")/.." || exit 1
case "${1:-}" in
  cup|bottle) obj="$1"; shift ;;
  *) echo "Pick up which object?"; select obj in cup bottle quit; do [ -n "$obj" ] && break; done
     [ "$obj" = quit ] && exit 0 ;;
esac
scripts/reset_real.sh || exit 1
grip=(); [ "$obj" = cup ] && grip=(--grip-width-mm 70)      # the cup's taught width; the bottle is sized from the camera
exec scripts/viam_pick_real.sh "$obj" --ui --voxel --approach horizontal --no-sweep --aim-trim-mm 0,-50,0 --scan-person --box-obstacle --yes --ask-handover --retries 0 "${grip[@]}" "$@"
