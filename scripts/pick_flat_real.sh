#!/usr/bin/env bash
# Real-arm pick from a flat-gripper pose (pick_flat.py). Asks for a typed 'yes' before moving and before closing.
#   scripts/pick_flat_real.sh [extra pick_flat.py args]
# Loads only VIAM_API_KEY / VIAM_API_KEY_ID from /home/adi274903/Documents/.env (nothing else in that file is read).
cd "$(dirname "$0")/.." || exit 1
ENV_FILE="${ENV_FILE:-/home/adi274903/Documents/.env}"
export VIAM_API_KEY="$(grep -E '^VIAM_API_KEY[[:space:]]*=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d ' \r')"
export VIAM_API_KEY_ID="$(grep -E '^VIAM_API_KEY_ID[[:space:]]*=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d ' \r')"
exec scripts/run.sh src/pick_flat.py --viam-address sfhack16-main.ty1cywh7tq.local.viam.cloud:8080 --sink viam "$@"
