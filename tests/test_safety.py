import numpy as np

from mimic.safety import MotionSmoother, Obstacles, approach_speed
from mimic.servo import depth_direction


def test_smoother_limits_acceleration_and_speed():
    s = MotionSmoother([0, 0, 0], max_accel=500, max_speed=200)
    prev, v = 0.0, []
    for _ in range(40):
        s.step([1000, 0, 0], 0.05)
        v.append(s.vel[0])
    assert max(abs(np.diff(v))) <= 500 * 0.05 + 1e-9      # never speeds up faster than max_accel
    assert max(v) <= 200 + 1e-9                           # and never exceeds the speed limit
    for _ in range(20):
        s.step([0, 0, 0], 0.05)
    assert abs(s.vel[0]) < 1e-6                           # comes to rest smoothly


def test_approach_speed_profile():
    f = lambda d: approach_speed(d, stop_cm=1, slow_cm=5, fast=200, slow=20)  # noqa: E731
    assert f(30) == 200 and f(5) == 200
    assert f(1) == 0 and f(0.5) == 0
    assert 20 <= f(2) < f(3) < f(4) < 200                 # eases down as it gets closer
    assert abs(f(1.001) - 20) < 1


def test_push_out_slides_along_surface():
    o = Obstacles(margin_mm=50)
    o.add((-1000, 1000, -1000, 1000, -1000, -100), "table")      # table top at z=-100
    p, hit = o.push_out([300, 200, -120])                        # inside the table
    assert hit == ["table"] and np.allclose(p, [300, 200, -50])
    p, hit = o.constrain([0, 0, 100], [400, 0, -300])            # a descent toward the table
    assert hit == ["table"] and p[2] >= -50 - 1e-9 and p[0] > 300  # stops at the margin, still advances in x


def test_clear_path_is_untouched():
    o = Obstacles.lab_world(60)
    p, hit = o.constrain([400, 0, 250], [450, 100, 200])
    assert hit == [] and np.allclose(p, [450, 100, 200])


def test_lab_world_blocks_walls_and_table():
    o = Obstacles.lab_world(60)
    assert o.blocking([400, 0, -100])                      # 23 mm above the table: inside the 60 mm margin
    assert not o.blocking([400, 0, 0])
    assert o.blocking([-480, 0, 100])                      # near the back wall
    assert o.blocking([300, 600, 100])                     # near the left wall


def test_depth_direction_leaves_image_position_alone():
    J = np.array([[0.0, -0.004, 0.0], [-0.003, 0.0, 0.0], [0.0, 0.0, 0.001]])   # image grows with +z
    n = depth_direction(J)
    assert np.allclose(n, [0, 0, 1]) and np.allclose(J[:2] @ n, 0)
    n2 = depth_direction(-J * [[1], [1], [-1]])             # reversed depth sign -> flipped direction
    assert n2 is not None
    assert depth_direction(np.array([[0.0, -0.004, 0.0], [-0.003, 0.0, 0.0], [0.0, 0.0, 0.0]])) is None
