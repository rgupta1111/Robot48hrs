import numpy as np

from mimic.pickup import PickupConfig, PickupPlanner, check_jacobian, smooth_move, table_floor_z

J = np.array([[0.0, -0.004, 0.0], [-0.003, 0.0, 0.0], [0.0, 0.0, 0.001]])


def run(world, cfg=None, max_iter=300):
    """world(pos) -> (features, distance); simulate the loop against a linear camera. Returns (planner, actions, pos)."""
    planner, pos, acts = PickupPlanner(J, cfg or PickupConfig(max_step_mm=1000)), np.zeros(3), []
    for _ in range(max_iter):
        s, d = world(pos)
        a = planner.step(s, d)
        acts.append(a)
        if a.kind == "move":
            pos = pos + np.array(a.move)
        if a.kind in ("done", "abort"):
            break
    return planner, acts, pos


def test_full_pickup_sequence():
    start = np.array([0.35, 0.25, 0.2])
    world = lambda pos: (start + J @ pos * np.array([1, 1, 0]) + np.array([0, 0, 0]), 400 + pos[2])   # z lowers distance
    planner, acts, pos = run(world)
    kinds = [a.kind for a in acts]
    assert kinds[0] == "grip_open"
    assert "grip_close" in kinds and kinds.index("grip_open") < kinds.index("grip_close")
    assert planner.state == "DONE" and kinds[-1] == "done"
    assert acts[kinds.index("grip_close") + 1].move[2] > 0                      # lifts after the grasp
    assert abs(start[0] + (J @ pos)[0]) < 0.2                                    # no wild drift


def test_aborts_when_object_never_seen():
    planner, acts, _ = run(lambda pos: (None, None), PickupConfig(lost_limit=3))
    assert planner.state == "ABORT" and acts[-1].kind == "abort"


def test_does_not_descend_until_centred():
    p = PickupPlanner(J, PickupConfig(max_step_mm=1000))
    p.step((0, 0, 0.2), 500)                                                     # grip_open
    a = p.step((0.4, 0.3, 0.2), 500)                                             # far off-centre
    assert p.state == "CENTRE" and a.kind == "move" and a.note == "centring"


def test_descend_pauses_when_off_centre_and_stops_at_grasp_distance():
    p = PickupPlanner(J, PickupConfig(max_step_mm=1000, descend_step_mm=25))
    p.step((0, 0, 0.2), 500)
    p.step((0, 0, 0.2), 500)                                                     # centred -> DESCEND
    assert p.state == "DESCEND"
    assert p.step((0, 0, 0.2), 500).move[2] == -25                               # centred: drops
    assert p.step((0.3, 0.3, 0.2), 500).move[2] > -25                            # way off-centre: does not drop
    p.step((0, 0, 0.2), 100)                                                     # within grasp distance
    assert p.state == "GRASP"


def test_max_cycles_aborts():
    p = PickupPlanner(J, PickupConfig(max_cycles=2))
    p.step((0, 0, 0.2), 500)
    p.step((0, 0, 0.2), 500)
    assert p.step((0, 0, 0.2), 500).kind == "abort"


def test_centring_never_moves_z():
    p = PickupPlanner(np.array([[-0.001, 0.0003, -0.0008], [-0.004, -0.0048, -0.0048], [0.008, 0.0085, 0.0085]]),
                      PickupConfig(max_step_mm=1000))
    p.step((0, 0, 0.2), 500)
    assert p.step((0.2, 0.3, 0.2), 500).move[2] == 0.0


def test_check_jacobian_rejects_the_bad_probe_from_the_real_run_and_accepts_a_good_one():
    bad = np.array([[-0.00088, 0.00033, -0.00078], [-0.00396, -0.00476, -0.00479], [0.00792, 0.00854, 0.00847]])
    assert check_jacobian(bad) is not None
    assert check_jacobian(J) is None
    assert check_jacobian(np.zeros((3, 3))) is not None


def test_aborts_when_the_image_error_keeps_growing():
    p = PickupPlanner(J, PickupConfig(max_step_mm=1000))
    p.step((0, 0, 0.2), 500)
    acts = [p.step((0.0, 0.3 + 0.03 * i, 0.2), 500) for i in range(6)]
    assert p.state == "ABORT" and any("growing" in a.note for a in acts)


def test_smooth_move_blends_and_caps_speed():
    assert np.allclose(smooth_move((10, 0, 0), (0, 10, 0), 0.5), (5, 5, 0))           # direction change is eased
    assert np.allclose(smooth_move((0, 0, 0), (30, 40, 0), 0.0), (30, 40, 0))         # no smoothing: unchanged
    capped = smooth_move((0, 0, 0), (30, 40, 0), 0.0, max_mm=10)
    assert abs(np.linalg.norm(capped) - 10) < 1e-9 and np.allclose(capped, (6, 8, 0))  # speed limit keeps the direction


def test_alternating_noise_gives_smaller_steps_than_raw():
    rng, prev, raw, smoothed = np.random.default_rng(0), np.zeros(3), [], []
    for _ in range(40):
        m = np.array([20.0, 0.0, 0.0]) * rng.choice([-1, 1])                           # jittery +/- commands
        prev = smooth_move(prev, m, 0.6)
        raw.append(np.linalg.norm(m))
        smoothed.append(np.linalg.norm(prev))
    assert np.mean(smoothed) < 0.6 * np.mean(raw)


def test_table_floor_keeps_the_fingertips_above_the_table():
    assert table_floor_z(-123, 105, 30) == 12
    assert table_floor_z(-123, 105, 30) - 105 > -123 + 29          # fingertips end up >= 30 mm above the table top
