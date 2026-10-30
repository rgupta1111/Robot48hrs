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

## Viam module (runs on the machine)
`module/` packages the object follower as a Viam generic service, `REPLACE_NAMESPACE:robot48hrs:object-follower`
(replace the namespace in `meta.json` and `module/follow_service.py` with your org's public namespace).
It needs an arm, a camera and a vision service (e.g. a YOLO detector) on the machine. Config attributes:
`arm`, `camera`, `vision` (required); `lock` (class name, default `cup`), `bounds` (6 numbers), `table_to_world` (4),
`mm_per_px`, `swap_xy`, `flip_u`, `flip_v`, `deadband_mm`, `gain`, `max_step_mm`, `max_speed_mm_s`, `smoothing`,
`min_confidence`, `loop_hz`. It starts disengaged. DoCommand: `{"command": "engage" | "disengage" | "stop" | "status" |
"recenter" | "set_lock", "lock": "<class>"}`. Package with `viam module build local`; follow-logic tests are in
`tests/test_module_logic.py`.
