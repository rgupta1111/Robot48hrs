"""Object width estimation from a detection box (pinhole camera model).

    width_mm = width_px * Z / fx        Z = camera-to-object depth in mm, fx = focal length in pixels

Z comes from the RealSense depth image (box_depth_mm) or from a distance you already have (e.g. the Range estimate
in local_detect.py). fx comes from the camera's intrinsics (ViamRobot.camera_info) - rescale it if the frame you
detect on is not the size the intrinsics were measured at (scale_focal).

Limits: the box is axis-aligned and YOLO boxes sit a few pixels outside the object, so the width is an upper bound
for a cup seen side-on; tilt `tightness` below 1.0 if you have measured that bias. Perspective makes a wide object
read slightly small at the box edges. This is the width across the image, not the object's true depth-wise size.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np


def scale_focal(fx, intrinsics_width_px, frame_width_px):
    """Focal length in pixels for a frame of frame_width_px, given fx measured at intrinsics_width_px."""
    return fx * frame_width_px / intrinsics_width_px


def width_from_depth(box, depth_mm, fx, tightness=1.0):
    """box: (x1, y1, x2, y2) pixels; depth_mm: distance along the optical axis. -> width in mm."""
    if depth_mm is None or not np.isfinite(depth_mm) or depth_mm <= 0 or fx <= 0:
        return None
    return float(max(box[2] - box[0], 0.0) * tightness * depth_mm / fx)


def height_from_depth(box, depth_mm, fy, tightness=1.0):
    if depth_mm is None or not np.isfinite(depth_mm) or depth_mm <= 0 or fy <= 0:
        return None
    return float(max(box[3] - box[1], 0.0) * tightness * depth_mm / fy)


def box_depth_mm(depth_img, box, inner=0.5, min_valid_frac=0.2):
    """Median valid depth (mm) in the central `inner` fraction of the box. Zeros / NaNs are missing readings.
    Uses the centre only so background pixels at the box corners do not pull the depth backwards.
    Returns None when too little of the region has a reading (e.g. a shiny or too-close object)."""
    h, w = depth_img.shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw, hh = (box[2] - box[0]) * inner / 2, (box[3] - box[1]) * inner / 2
    x1, x2 = int(max(cx - hw, 0)), int(min(cx + hw, w))
    y1, y2 = int(max(cy - hh, 0)), int(min(cy + hh, h))
    if x2 <= x1 or y2 <= y1:
        return None
    patch = np.asarray(depth_img[y1:y2, x1:x2], float).ravel()
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size < min_valid_frac * patch.size or valid.size == 0:
        return None
    return float(np.median(valid))


def width_uncertainty_mm(box, depth_mm, fx, depth_err_frac=0.02, px_err=2.0):
    """Rough 1-sigma error: pixel quantisation / box jitter (px_err on each edge) and relative depth error."""
    w_px = max(box[2] - box[0], 1.0)
    width = w_px * depth_mm / fx
    return float(np.hypot(width * depth_err_frac, np.hypot(px_err, px_err) * depth_mm / fx))


@dataclass
class WidthReading:
    width_mm: float
    sigma_mm: float
    depth_mm: float
    n: int


class WidthEstimator:
    """Median of the last `window` per-frame estimates: one bad detection or depth reading cannot move it."""

    def __init__(self, fx, window=15, tightness=1.0):
        self.fx, self.tightness = fx, tightness
        self._widths, self._depths = deque(maxlen=window), deque(maxlen=window)

    def reset(self):
        self._widths.clear()
        self._depths.clear()

    def update(self, box, depth_mm):
        """Feed one frame's box and depth (mm). Returns the current WidthReading, or None before any valid sample."""
        w = width_from_depth(box, depth_mm, self.fx, self.tightness)
        if w is not None:
            self._widths.append(w)
            self._depths.append(depth_mm)
        return self.reading(box)

    def reading(self, box=None):
        if not self._widths:
            return None
        width, depth = float(np.median(self._widths)), float(np.median(self._depths))
        ref = box if box is not None else (0.0, 0.0, width * self.fx / depth, 1.0)
        return WidthReading(width, width_uncertainty_mm(ref, depth, self.fx) / np.sqrt(len(self._widths)),
                            depth, len(self._widths))
