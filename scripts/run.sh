#!/usr/bin/env bash
# Run a project script with the project's .venv, no activation needed.
#   scripts/run.sh src/joint_test.py --viam-address <machine>.viam.cloud --joint pinch --degrees 20
root="$(cd "$(dirname "$0")/.." && pwd)"
[ -x "$root/.venv/bin/python" ] || { echo "no .venv yet - run: source scripts/setup_env.sh"; exit 1; }
cd "$root" && exec "$root/.venv/bin/python" "$@"
