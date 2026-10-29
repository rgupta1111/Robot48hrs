#!/usr/bin/env bash
# Create .venv, install requirements, fetch models. Use: source scripts/setup_env.sh  (so the venv stays active)
# Optional: PYTHON=python3.12 source scripts/setup_env.sh   (mediapipe may lack wheels for the newest Python)
_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
_py="${PYTHON:-python3}"

if [ ! -d "$_root/.venv" ]; then
    "$_py" -m venv "$_root/.venv" || return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
source "$_root/.venv/bin/activate"

if [ ! -f "$_root/.venv/.deps_installed" ] || [ "$_root/requirements.txt" -nt "$_root/.venv/.deps_installed" ]; then
    pip install -q --upgrade pip
    # mediapipe is installed --no-deps so it doesn't pull opencv-contrib over opencv-python; its runtime
    # dependencies (absl-py, certifi, matplotlib, ...) are listed in requirements.txt
    grep -v '^mediapipe' "$_root/requirements.txt" | sed 's/#.*//' | pip install -q -r /dev/stdin \
        && pip install -q mediapipe==0.10.35 --no-deps \
        && touch "$_root/.venv/.deps_installed" || { echo "install failed"; return 1 2>/dev/null || exit 1; }
fi
python "$_root/scripts/download_models.py"
python -c "import viam, cv2, mediapipe" || { echo "import check failed - rerun after: rm .venv/.deps_installed"; return 1 2>/dev/null || exit 1; }
echo "ready: $(python --version), venv active ($VIRTUAL_ENV)"
echo "set VIAM_API_KEY and VIAM_API_KEY_ID in your shell before using --sink viam"
