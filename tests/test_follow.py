import numpy as np

from mimic.follower import FollowConfig, ObjectFollower
from tidy.geometry import fit_homography, pixel_to_xy, relative_target
from tidy.robot import TableToWorld


def test_homography_round_trip():
    # a made-up camera: pixels -> table mm through a known perspective-ish matrix
    H_true = np.array([[0.4, 0.05, 200.0], [-0.03, 0.5, -300.0], [1e-5, 2e-5, 1.0]])
    px = [(100, 100), (1100, 120), (1000, 650), (150, 600), (600, 380), (300, 200)]
    xy = [tuple(pixel_to_xy(H_true, p)) for p in px]
    H, rms = fit_homography(px, xy)
    assert rms < 1e-2
    assert np.allclose(pixel_to_xy(H, (500, 400)), pixel_to_xy(H_true, (500, 400)), atol=0.05)


def test_table_to_world_inverse():
    t = TableToWorld(10, -20, 5, 30)
    assert np.allclose(t.to_robot(*t.apply(100, 50, 7)), (100, 50, 7))


def make(**kw):
    f = ObjectFollower(FollowConfig(bounds=(0, 1000, -500, 500, 0, 500), **kw))
    f.engage(0.0)
    return f


def test_idle_until_engaged():
    assert ObjectFollower(FollowConfig()).update((1, 2, 3), (0, 0, 0), 0.1) == (None, "idle")


def test_settles_within_deadband():
    assert make(deadband_mm=10).update((100, 0, 100), (102, 3, 99), 0.1)[1] == "settled"


def test_converges_with_step_limits():
    f, cur = make(smoothing=1.0, max_step_mm=50, max_speed_mm_s=1e6, gain=1.0), np.array([0.0, 0, 100])
    for i in range(1, 20):
        new, state = f.update((300, 100, 100), cur, i * 0.5)
        if new:
            assert np.linalg.norm(np.array(new) - cur) <= 50 + 1e-6
            cur = np.array(new)
    assert np.linalg.norm(cur - (300, 100, 100)) < 10


def test_speed_limit():
    f = make(smoothing=1.0, max_step_mm=1e6, max_speed_mm_s=100, gain=1.0)
    new, _ = f.update((1000, 0, 100), (0, 0, 100), 0.1)
    assert abs(new[0] - 10) < 1e-6          # 100 mm/s * 0.1 s


def test_bounds_clamp_target():
    f = make(smoothing=1.0, max_step_mm=1e6, max_speed_mm_s=1e6, gain=1.0)
    new, _ = f.update((5000, 0, 100), (900, 0, 100), 1.0)
    assert new[0] == 1000


def test_lost_holds_then_disengages():
    f = make(lost_disengage_s=2.0)
    assert f.update(None, (0, 0, 0), 1.0) == (None, "lost")
    assert f.update(None, (0, 0, 0), 3.5)[1] == "disengaged" and not f.engaged


def test_relative_target_directions():
    ref = ((600, 400), (400.0, 0.0, 250.0))
    assert relative_target(*ref, (600, 400)) == (400, 0, 250)                      # no motion
    assert relative_target(*ref, (700, 400), mm_per_px=1) == (400, -100, 250)      # object right -> robot right (-y)
    assert relative_target(*ref, (600, 500), mm_per_px=1) == (300, 0, 250)         # object down -> toward robot (-x)
    assert relative_target(*ref, (700, 400), mm_per_px=1, flip_u=True)[1] == 100
    assert relative_target(*ref, (700, 400), mm_per_px=1, swap_xy=True)[:2] == (300, 0)
