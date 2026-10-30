import numpy as np

from mimic.pickup import PickupConfig, PickupPlanner

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
