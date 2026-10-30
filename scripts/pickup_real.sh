#!/usr/bin/env bash
# Real-arm pick-up (track_pickup.py). Asks for a typed 'yes' before the arm moves and before the gripper closes.
#   scripts/pickup_real.sh [object] [extra track_pickup.py args, e.g. --ui]   default object: cup
# Loads only VIAM_API_KEY / VIAM_API_KEY_ID from /home/adi274903/Documents/.env (nothing else in that file is read).
cd "$(dirname "$0")/.." || exit 1
ENV_FILE="${ENV_FILE:-/home/adi274903/Documents/.env}"
export VIAM_API_KEY="$(grep -E '^VIAM_API_KEY[[:space:]]*=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d ' \r')"
export VIAM_API_KEY_ID="$(grep -E '^VIAM_API_KEY_ID[[:space:]]*=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d ' \r')"
exec scripts/run.sh src/track_pickup.py --viam-address sfhack16-main.ty1cywh7tq.local.viam.cloud:8080 --preset lab \
    --object "${1:-cup}" --tip-offset-mm 60 --arm-speed 30 --sink viam "${@:2}"
