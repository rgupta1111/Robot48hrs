import json
import math

import numpy as np

from mimic.kinematics import Kinematics, Unsupported, ov_to_R
from mimic.plan import attach_joints, build_plan, describe, segment_joint_speed
from mimic.safety import Obstacles

# a UR5e-like 6R arm in Viam's SVA JSON layout (lengths in mm)
SVA = {
    "name": "test6r", "kinematic_param_type": "SVA",
    "links": [
        {"id": "base", "parent": "world", "translation": {"x": 0, "y": 0, "z": 0}},
        {"id": "shoulder", "parent": "j0", "translation": {"x": 0, "y": 0, "z": 162.5}},
        {"id": "upper", "parent": "j1", "translation": {"x": 0, "y": 138.0, "z": 425.0}},
        {"id": "fore", "parent": "j2", "translation": {"x": 0, "y": -131.0, "z": 392.2}},
        {"id": "wrist1", "parent": "j3", "translation": {"x": 0, "y": 0, "z": 100.0}},
        {"id": "wrist2", "parent": "j4", "translation": {"x": 0, "y": 127.0, "z": 0}},
        {"id": "wrist3", "parent": "j5", "translation": {"x": 0, "y": 0, "z": 100.0},
         "orientation": {"type": "ov_degrees", "value": {"x": 0, "y": 0, "z": 1, "th": 0}}},
    ],
    "joints": [
        {"id": "j0", "type": "revolute", "parent": "base", "axis": {"x": 0, "y": 0, "z": 1}, "min": -360, "max": 360},
        {"id": "j1", "type": "revolute", "parent": "shoulder", "axis": {"x": 0, "y": 1, "z": 0}, "min": -360, "max": 360},
        {"id": "j2", "type": "revolute", "parent": "upper", "axis": {"x": 0, "y": 1, "z": 0}, "min": -180, "max": 180},
        {"id": "j3", "type": "revolute", "parent": "fore", "axis": {"x": 0, "y": 1, "z": 0}, "min": -360, "max": 360},
        {"id": "j4", "type": "revolute", "parent": "wrist1", "axis": {"x": 0, "y": 0, "z": 1}, "min": -360, "max": 360},
        {"id": "j5", "type": "revolute", "parent": "wrist2", "axis": {"x": 0, "y": 1, "z": 0}, "min": -360, "max": 360},
    ],
}


def arm():
    return Kinematics.from_json(json.dumps(SVA))


def test_parse_and_fk_at_zero():
    k = arm()
    assert k.n == 6
    p, R = k.fk([0] * 6)
    assert np.allclose(p, [0, 138 - 131 + 127, 162.5 + 425 + 392.2 + 100 + 100])   # all links stacked up the z axis
    assert np.allclose(R, np.eye(3))


def test_fk_rotates_about_the_base():
    k = arm()
    p0, _ = k.fk([0] * 6)
    p, _ = k.fk([90, 0, 0, 0, 0, 0])
    assert np.allclose([p[0], p[1], p[2]], [-p0[1], p0[0], p0[2]], atol=1e-6)


def test_ik_round_trip_stays_near_the_seed():
    k = arm()
    q_true = np.array([-10.0, -60.0, 70.0, -100.0, 90.0, 5.0])
    p, R = k.fk(q_true)
    seed = q_true + np.array([4, -3, 2, 5, -4, 3])
    q = k.ik(p, R, seed)
    assert q is not None
    p2, R2 = k.fk(q)
    assert np.linalg.norm(p2 - p) < 0.1 and np.allclose(R2, R, atol=1e-3)
    assert np.max(np.abs(q - seed)) < 15                  # nearest solution, not a flipped one


def test_straight_line_ik_has_small_joint_steps():
    k = arm()
    q = np.array([0.0, -70.0, 100.0, -120.0, 60.0, 40.0])
    p, R = k.fk(q)
    prev, worst = q, 0.0
    for s in range(0, 201, 10):
        qs = k.ik(p + np.array([0, 0, -1.0]) * s * 0.5 + np.array([s * 0.5, 0, 0]), R, prev)
        assert qs is not None
        worst = max(worst, float(np.max(np.abs(qs - prev))))
        prev = qs
    assert worst < 6.0                                     # no wrist flips along a 20 cm line


def test_unsupported_models():
    bad = dict(SVA, kinematic_param_type="DH")
    try:
        Kinematics.from_json(json.dumps(bad))
        assert False
    except Unsupported:
        pass


def test_orientation_vector_matches_z_axis():
    R = ov_to_R(1, 0, 0, 0)
    assert np.allclose(R @ [0, 0, 1], [1, 0, 0], atol=1e-9)
    assert np.allclose(ov_to_R(0, 0, -1, 0) @ [0, 0, 1], [0, 0, -1], atol=1e-9)


def test_plan_profile_and_open_point():
    plan = build_plan([300, 0, 150], [1, 0, 0], tip_cm=20.0, stop_cm=1.0, open_cm=10.0, slow_cm=5.0, fast=200, slow=20)
    assert abs(plan.travel_mm - 190) < 1e-6 and plan.ok
    assert abs(plan.open_s_mm - 100) < 1e-6                # opens after 100 mm = at 10 cm from the object
    tips = [w.tip_cm for w in plan.waypoints]
    assert tips[0] == 20.0 and abs(tips[-1] - 1.0) < 1e-6 and tips == sorted(tips, reverse=True)
    assert plan.waypoints[0].speed == 200 and plan.waypoints[-1].speed == 0
    assert [s[0] for s in plan.segments(5.0)] == ["fast", "slow"]
    lines = describe(plan, 5.0, 200, 20, 10.0)
    assert any("open the gripper after 100 mm" in l for l in lines)


def test_plan_stops_before_an_obstacle_and_workspace():
    obs = Obstacles(margin_mm=50)
    obs.add((420, 450, -500, 500, -500, 500), "box")
    plan = build_plan([300, 0, 150], [1, 0, 0], 20.0, 1.0, 10.0, 5.0, 200, 20, obstacles=obs)
    assert not plan.ok and plan.travel_mm < 100 and any("box" in n for n in plan.notes)
    plan2 = build_plan([300, 0, 150], [1, 0, 0], 20.0, 1.0, 10.0, 5.0, 200, 20, bounds=(250, 400, -300, 300, 0, 450))
    assert not plan2.ok and plan2.waypoints[-1].pos[0] <= 400 + 1e-6


def test_already_close_opens_first():
    plan = build_plan([300, 0, 150], [1, 0, 0], tip_cm=8.0, stop_cm=1.0, open_cm=10.0, slow_cm=5.0, fast=200, slow=20)
    assert plan.open_s_mm == 0


def test_joint_space_plan_for_a_straight_approach():
    k = arm()
    q0 = np.array([0.0, -70.0, 100.0, -120.0, 60.0, 40.0])
    p, R = k.fk(q0)
    d = R[:, 2]                                             # straight along the tool's own axis
    plan = build_plan(p, d, tip_cm=15.0, stop_cm=1.0, open_cm=10.0, slow_cm=5.0, fast=200, slow=20)
    assert attach_joints(plan, k, q0, lambda x, y, z: (x, y, z))
    assert plan.method.startswith("joint-space") and plan.waypoints[0].joints is not None
    assert np.allclose(plan.waypoints[0].joints, q0)
    end_p, end_R = k.fk(plan.waypoints[-1].joints)
    assert np.linalg.norm(end_p - plan.waypoints[-1].pos) < 0.5 and np.allclose(end_R, R, atol=0.02)
    direct = np.abs(plan.waypoints[-1].joints - q0)         # the least any path could move each joint
    assert float(np.sum(plan.joint_travel_deg)) <= float(np.sum(direct)) * 1.03 + 0.5   # essentially no wasted motion
    assert segment_joint_speed(plan, 0, len(plan.waypoints) - 1, 100.0) > 2.0
    assert any("joint travel" in l for l in describe(plan, 5.0, 200, 20, 10.0))
