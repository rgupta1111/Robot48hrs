#!/bin/sh
# Entrypoint viam-server runs. Creates a venv on first start and installs the module's (small) requirements.
cd "$(dirname "$0")"
VENV=.venv
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -r requirements.txt
exec "$VENV/bin/python" main.py "$@"
