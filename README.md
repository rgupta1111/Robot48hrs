# Robot48hrs

Make a robot arm mimic your arm: webcam -> MediaPipe pose/hand tracking -> Viam arm.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # see the mediapipe note inside
python scripts/download_models.py      # pose + hand landmarker models into models/
```

## Run
```bash
python src/arm_mimic.py --arm right                       # safe: simulator sink, no robot
python src/arm_mimic.py --sink viam --viam-address <machine>.viam.cloud --preset ur5e
python src/joint_test.py --viam-address <machine>.viam.cloud --joint wrist --degrees 30
```
Real arm: export `VIAM_API_KEY` and `VIAM_API_KEY_ID` (never commit them), set a low speed limit in the Viam
config and keep a hand on the e-stop. Full controls and options: `python src/arm_mimic.py --help`.

## Layout
- `src/arm_mimic.py` - main app (HUD, keys, engage/e-stop)
- `src/joint_test.py` - move one joint +/- N degrees to verify the Viam connection
- `src/mimic/` - `arm_tracker`, `hand_tracker`, `controller`, `sinks`
- `src/tidy/robot.py` - blocking `ViamRobot` wrapper
- `tests/` - controller/sink unit tests (`pip install -r requirements-dev.txt && pytest`)
