"""Camera pixel -> robot table coordinates for a FIXED camera looking at the table.

The object sits on the table, so its pixel position maps to a table (x, y) through a homography. Fitting it from
a few "gripper above the object" samples needs no camera intrinsics, depth, or camera pose - see calibrate_camera.py.
All robot coordinates are mm in the robot frame (arm base at the origin, x forward, y left, z up).
"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DEFAULT_CALIBRATION = os.path.abspath(os.path.join(ROOT, "data", "camera_calibration.json"))


def fit_homography(pixels, xy):
    """pixels, xy: lists of (u, v) and (x, y), at least 4 non-collinear pairs. -> (3x3 matrix, rms error in mm)."""
    import cv2
    if len(pixels) < 4:
        raise ValueError("need at least 4 calibration points")
    H, _ = cv2.findHomography(np.float32(pixels), np.float32(xy), 0)
    if H is None:
        raise ValueError("points are degenerate (collinear?) - spread them across the workspace")
    pred = pixel_to_xy(H, np.float32(pixels))
    rms = float(np.sqrt(np.mean(np.sum((pred - np.float32(xy)) ** 2, axis=1))))
    return H, rms


def pixel_to_xy(H, pixels):
    """(u, v) or an (N, 2) array of pixels -> table (x, y) in mm, same shape."""
    import cv2
    p = np.float32(pixels).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(p, np.asarray(H, np.float64)).reshape(-1, 2)
    return out[0] if np.ndim(pixels) == 1 else out


def save_calibration(path, H, rms, points, frame_size):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"homography": np.asarray(H).tolist(), "rms_mm": rms, "frame_size": list(frame_size),
                   "points": [{"px": list(map(float, p)), "xy": list(map(float, q))} for p, q in points]}, f, indent=2)


def load_calibration(path=DEFAULT_CALIBRATION):
    if not os.path.exists(path):
        raise SystemExit(f"no camera calibration at {path} - run: scripts/run.sh src/calibrate_camera.py ...")
    with open(path) as f:
        data = json.load(f)
    return np.array(data["homography"]), data


def relative_target(ref_uv, ref_xyz, uv, mm_per_px=0.8, swap_xy=False, flip_u=False, flip_v=False):
    """No-calibration mapping: the gripper moves by the object's pixel motion since engage, scaled by mm_per_px.
    Assumes a camera behind the robot looking forward: image right = robot right (-y), image down = toward the
    robot (-x). swap_xy / flip_u / flip_v fix other mounting orientations. z is left as it was at engage."""
    du, dv = uv[0] - ref_uv[0], uv[1] - ref_uv[1]
    du, dv = (-du if flip_u else du), (-dv if flip_v else dv)
    dx, dy = (-du, -dv) if swap_xy else (-dv, -du)
    return (ref_xyz[0] + mm_per_px * dx, ref_xyz[1] + mm_per_px * dy, ref_xyz[2])
