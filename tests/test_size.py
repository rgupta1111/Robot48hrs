import numpy as np

from mimic.size import WidthEstimator, box_depth_mm, grip_position, scale_focal, width_from_depth


FX = 900.0


def box_for(width_mm, depth_mm, cx=640, cy=360, height_px=200):
    w_px = width_mm * FX / depth_mm
    return (cx - w_px / 2, cy - height_px / 2, cx + w_px / 2, cy + height_px / 2)


def test_width_from_depth_recovers_known_width():
    assert abs(width_from_depth(box_for(80, 400), 400, FX) - 80) < 1e-6
    assert abs(width_from_depth(box_for(80, 250), 250, FX) - 80) < 1e-6   # same cup, closer: same width


def test_invalid_depth_or_focal_gives_none():
    b = box_for(80, 400)
    assert width_from_depth(b, 0, FX) is None
    assert width_from_depth(b, float("nan"), FX) is None
    assert width_from_depth(b, 400, 0) is None


def test_scale_focal():
    assert scale_focal(900, 1280, 640) == 450


def test_box_depth_ignores_background_at_corners_and_missing_pixels():
    depth = np.full((720, 1280), 1500.0)          # far wall
    depth[260:460, 560:720] = 400.0               # the cup
    depth[300:320, 600:620] = 0.0                 # holes in the reading
    assert box_depth_mm(depth, (540, 240, 740, 480)) == 400.0
    assert box_depth_mm(np.zeros((720, 1280)), (540, 240, 740, 480)) is None


def test_estimator_median_rejects_outliers_and_reports_sigma():
    rng = np.random.default_rng(1)
    est, reading = WidthEstimator(FX, window=15), None
    for i in range(15):
        depth = 400 + rng.normal(0, 3)
        box = box_for(80, depth)
        if i == 7:
            box = box_for(300, depth)             # one wild detection
        reading = est.update(box, depth)
    assert abs(reading.width_mm - 80) < 3
    assert 0 < reading.sigma_mm < 3 and reading.n == 15


def test_grip_position_squeezes_slightly_and_clamps():
    assert grip_position(60, units_per_mm=10, squeeze_mm=1) == 590
    assert grip_position(200, 10, 1) == 850      # wider than the gripper: clamped open
    assert grip_position(0.5, 10, 1) == 0        # never negative
