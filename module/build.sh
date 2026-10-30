#!/bin/sh
# Packages the module for `viam module build`. Only numpy-level code from src/ is needed (no OpenCV / YOLO / MediaPipe).
set -e
tar -czf module.tar.gz meta.json module/main.py module/follow_service.py module/follow_logic.py \
  module/run.sh module/requirements.txt src/mimic/__init__.py src/mimic/follower.py src/tidy/__init__.py \
  src/tidy/geometry.py src/tidy/robot.py
