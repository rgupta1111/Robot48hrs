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


def test_plan_blends_the_lateral_correction():
    lat = np.array([0.0, 12.0, -6.0])
    plan = build_plan([300, 0, 150], [1, 0, 0], tip_cm=20.0, stop_cm=1.0, open_cm=10.0, slow_cm=5.0, fast=150, slow=20, lateral=lat)
    assert np.allclose(plan.waypoints[0].pos, [300, 0, 150])
    assert np.allclose(plan.waypoints[-1].pos, [300 + 190, 12.0, 144.0])      # ends exactly in front of the object
    mid = plan.waypoints[len(plan.waypoints) // 2].pos
    assert 0 < mid[1] < 12 and mid[2] < 150                                  # drifts there gradually


def sva_to_urdf(scale=0.001, rpy_first="0 0 0"):
    """The same toy arm written as URDF (metres)."""
    out = ['<?xml version="1.0"?><robot name="toy">']
    prev = "world_link"
    out.append(f'<link name="{prev}"/>')
    links = {l["id"]: l for l in SVA["links"]}
    joints = {j["parent"]: j for j in SVA["joints"]}
    cur = "base"
    n = 0
    while True:
        t = links[cur]["translation"]
        n += 1
        out.append(f'<link name="{cur}"/><joint name="fix{n}" type="fixed"><parent link="{prev}"/><child link="{cur}"/>'
                   f'<origin xyz="{t["x"] * scale} {t["y"] * scale} {t["z"] * scale}" rpy="{rpy_first}"/></joint>')
        j = joints.get(cur)
        if j is None:
            break
        ax = j["axis"]
        out.append(f'<link name="after_{j["id"]}"/><joint name="{j["id"]}" type="revolute"><parent link="{cur}"/>'
                   f'<child link="after_{j["id"]}"/><origin xyz="0 0 0"/><axis xyz="{ax["x"]} {ax["y"]} {ax["z"]}"/>'
                   f'<limit lower="{np.radians(j["min"])}" upper="{np.radians(j["max"])}"/></joint>')
        prev = f"after_{j['id']}"
        cur = next(l["id"] for l in SVA["links"] if l["parent"] == j["id"])
    out.append("</robot>")
    return "".join(out)


def test_urdf_matches_sva():
    q = [10.0, -60.0, 90.0, -110.0, 35.0, 50.0]
    a = arm().fk(q)
    b = Kinematics.from_urdf(sva_to_urdf(), scale=1000.0).fk(q)
    assert np.allclose(a[0], b[0], atol=1e-6) and np.allclose(a[1], b[1], atol=1e-9)
    assert Kinematics.from_bytes(sva_to_urdf().encode()).n == 6
    assert Kinematics.from_bytes(json.dumps(SVA).encode()).n == 6


def test_load_validated_accepts_urdf_in_metres_or_mm_and_rejects_a_mismatch():
    from mimic.kinematics import load_validated, ov_to_R  # noqa: F401
    q = [10.0, -60.0, 90.0, -110.0, 35.0, 50.0]
    k = arm()
    p, R = k.fk(q)
    o = R[:, 2]
    import math
    lat = math.acos(o[2]); lon = math.atan2(o[1], o[0])
    from mimic.kinematics import rot_y, rot_z
    inner = rot_y(lat).T @ rot_z(lon).T @ R
    th = math.degrees(math.atan2(inner[1, 0], inner[0, 0]))

    class Bot:
        def __init__(self, data):
            self.data = data
        def get_kinematics(self): return self.data
        def get_joints(self): return q
        def get_pose(self):
            from types import SimpleNamespace as NS
            return NS(x=p[0], y=p[1], z=p[2], o_x=o[0], o_y=o[1], o_z=o[2], theta=th)
    kin, note = load_validated(Bot(sva_to_urdf().encode()))                      # metres
    assert kin is not None and "OK" in note
    kin, note = load_validated(Bot(sva_to_urdf(scale=1.0).encode()))             # already millimetres
    assert kin is not None and "OK" in note
    kin, note = load_validated(Bot(sva_to_urdf(scale=0.0007).encode()))          # wrong lengths
    assert kin is None and "rejected" in note
    kin, note = load_validated(Bot(b"\x00\x01 not a kinematics file"))
    assert kin is None and "not supported" in note
    kin, note = load_validated(Bot(json.dumps(SVA).encode()))
    assert kin is not None


def test_short_plan_is_described_and_still_executable():
    plan = build_plan([300, 0, 150], [1, 0, 0], tip_cm=41.0, stop_cm=1.0, open_cm=10.0, slow_cm=5.0, fast=150, slow=20,
                      bounds=(250, 600, -300, 300, 0, 450))
    assert not plan.ok and plan.travel_mm > 200 and plan.final_tip_cm > 1.0
    assert any("SHORT PLAN" in l for l in describe(plan, 5.0, 150, 20, 10.0))


def test_look_at_ik_keeps_the_object_in_view_with_a_new_wrist():
    k = arm()
    q0 = np.array([10.0, -60.0, 90.0, -110.0, 35.0, 50.0])
    p, R = k.fk(q0)
    a_tool = np.array([0.0, 0.0, 1.0])                       # camera looks along the tool axis
    c0 = 300.0
    p_obj = p + R @ a_tool * c0
    fixed = {4: 90.0, 5: -360.0}
    q = k.look_at_ik(p_obj, c0, a_tool, q0, fixed)
    assert q is not None
    assert abs(q[4] - 90.0) < 1e-6 and abs(q[5] + 360.0) < 1e-6      # the wrist is where we asked
    p2, R2 = k.fk(q)
    assert np.linalg.norm(p2 + c0 * (R2 @ a_tool) - p_obj) < 1.0     # the object is still dead ahead of the camera, same range
    assert np.max(np.abs(q[:4] - q0[:4])) < 90                       # the other joints moved sensibly, not a different branch


def test_link_points_end_at_the_tool():
    k = arm()
    q = [10.0, -60.0, 90.0, -110.0, 35.0, 50.0]
    pts = k.link_points(q)
    assert len(pts) == 7 and np.allclose(pts[-1], k.fk(q)[0])


def test_nearest_equivalent_never_spins_the_long_way():
    from mimic.kinematics import nearest_equivalent
    assert nearest_equivalent(-360, -272.45) == -360
    assert nearest_equivalent(-360, 10) == 0                  # same orientation, 10 deg away instead of 370
    assert nearest_equivalent(-360, 200) == 360 or nearest_equivalent(-360, 200) == 0
    assert abs(nearest_equivalent(-360, 200) - 200) <= 180
    assert nearest_equivalent(90, 85) == 90
    assert nearest_equivalent(-360, -272, (-360, 360)) == -360
    assert nearest_equivalent(-360, 10, (-360, -10)) == -360  # limits that forbid 0 and +360 keep the literal angle


def test_smooth_path_eases_in_and_out():
    from mimic.kinematics import smooth_path
    path = smooth_path([0, 0], [90, -40], n=12)
    steps = np.diff(np.vstack([[0, 0]] + path), axis=0)
    assert np.allclose(path[-1], [90, -40])
    assert abs(steps[0][0]) < abs(steps[len(steps) // 2][0]) > abs(steps[-1][0])      # slow, fast, slow


def test_level_look_at_makes_the_camera_axis_horizontal():
    k = arm()
    a_tool = np.array([0.0, 0.0, 1.0])
    rng = np.random.default_rng(3)
    q_h = None
    for _ in range(4000):                                     # find any configuration whose camera axis is level
        q = np.array([rng.uniform(-30, 30), rng.uniform(-100, -30), rng.uniform(60, 120), rng.uniform(-150, -60), 60.0, 20.0])
        if abs((k.fk(q)[1] @ a_tool)[2]) < 0.01:
            q_h = q
            break
    assert q_h is not None
    p, R = k.fk(q_h)
    c0 = 300.0
    p_obj = p + R @ a_tool * c0
    seed = q_h + np.array([6.0, -5.0, 8.0, -9.0, 0.0, 0.0])      # tilted start; wrist already where we want it
    q = k.look_at_ik(p_obj, c0, a_tool, seed, {4: 60.0, 5: 20.0}, level=True)
    assert q is not None
    p2, R2 = k.fk(q)
    axis = R2 @ a_tool
    assert abs(axis[2]) < 0.02                                # horizontal to within about a degree
    assert np.linalg.norm(p2 + c0 * axis - p_obj) < 1.0       # and still looking at the object


def test_segments_split_at_the_checkpoint():
    plan = build_plan([300, 0, 150], [1, 0, 0], tip_cm=20.0, stop_cm=-3.0, open_cm=10.0, slow_cm=5.0, fast=150, slow=20)
    segs = plan.segments(5.0, split_cm=10.0)
    assert [s[0] for s in segs] == ["approach", "fast", "slow"]
    assert abs(plan.waypoints[segs[0][2]].tip_cm - 10.0) < 1e-6 and abs(plan.waypoints[segs[1][2]].tip_cm - 5.0) < 1e-6
    assert [s[0] for s in plan.segments(5.0)] == ["fast", "slow"]
    assert [s[0] for s in plan.segments(5.0, split_cm=25.0)] == ["fast", "slow"]          # already past the checkpoint
    assert abs(plan.final_tip_cm + 3.0) < 1e-6 and any("grasp depth" in l for l in describe(plan, 5.0, 150, 20, 10.0))


def test_line_plan_lifts_straight_up_and_stops_at_limits():
    from mimic.plan import describe_line, line_plan
    plan = line_plan([400, 0, 100], [0, 0, 1], 100.0, 40.0, bounds=(250, 600, -300, 300, 0, 450))
    assert plan.ok and abs(plan.travel_mm - 100) < 1e-6 and np.allclose(plan.waypoints[-1].pos, [400, 0, 200])
    assert plan.kind == "lift" and plan.waypoints[0].speed == 40.0 and plan.waypoints[-1].speed == 0.0
    short = line_plan([400, 0, 400], [0, 0, 1], 100.0, 40.0, bounds=(250, 600, -300, 300, 0, 450))
    assert not short.ok and short.travel_mm < 60 and any("workspace" in n for n in short.notes)
    assert "lift straight up" in describe_line(plan, 40.0)[0]


def test_look_at_multi_start_finds_a_solution_the_local_solver_misses():
    k = arm()
    q0 = np.array([10.0, -60.0, 90.0, -110.0, 35.0, 50.0])
    p, R = k.fk(q0)
    a_tool = np.array([0.0, 0.0, 1.0])
    p_obj = p + R @ a_tool * 300.0
    q = k.look_at_ik_multi(p_obj, 300.0, a_tool, q0, {4: 60.0, 5: 20.0}, level=False)
    assert q is not None
    p2, R2 = k.fk(q)
    assert np.linalg.norm(p2 + 300.0 * (R2 @ a_tool) - p_obj) < 1.0


def test_plan_note_names_the_limit_that_stops_it():
    plan = build_plan([300, 0, 440], [0, 0, 1], tip_cm=40.0, stop_cm=1.0, open_cm=10.0, slow_cm=5.0, fast=150, slow=20,
                      bounds=(250, 600, -300, 300, 0, 450))
    assert not plan.ok and any("z would go above the 450 mm limit" in n for n in plan.notes)


def test_camera_redials_once_when_the_video_connection_breaks():
    import threading
    from types import SimpleNamespace as NS

    import cv2

    from tidy.robot import ViamRobot
    ok, buf = cv2.imencode(".jpg", np.full((20, 30, 3), 120, np.uint8))
    calls = {"n": 0, "resets": 0}

    class Cam:
        def get_images(self):
            calls["n"] += 1
            if calls["n"] == 1:
                raise FileNotFoundError(2, "No such file or directory")
            return [NS(data=buf.tobytes(), mime_type="image/jpeg")], None

    r = object.__new__(ViamRobot)
    r._cam_real, r._cam_lock = {"cam": "cam"}, threading.Lock()
    r._run = lambda value, timeout=None: value
    r._cam_client = lambda real: (Cam(), 0)
    r._cam_reset = lambda gen: calls.__setitem__("resets", calls["resets"] + 1)
    frame = r.get_frame("cam")
    assert frame.shape == (20, 30, 3) and calls == {"n": 2, "resets": 1}


def test_look_at_from_straight_above():
    k = arm()
    rng = np.random.default_rng(5)
    a_tool = np.array([0.0, 0.0, 1.0])
    q_down = None
    for _ in range(6000):                                      # any configuration whose camera axis points straight down
        q = np.array([rng.uniform(-180, 180), rng.uniform(-120, 0), rng.uniform(20, 150), rng.uniform(-200, 0),
                      rng.uniform(-90, 90), 0.0])
        if (k.fk(q)[1] @ a_tool)[2] < -0.999:
            q_down = q
            break
    assert q_down is not None
    p, R = k.fk(q_down)
    c0 = 200.0
    p_obj = p + R @ a_tool * c0                                # the object sits 200 mm below the camera
    seed = q_down + np.array([8.0, -6.0, 9.0, -7.0, 5.0, 0.0])
    q = k.look_at_ik_multi(p_obj, c0, a_tool, seed, {5: 0.0}, down=True)
    assert q is not None
    p2, R2 = k.fk(q)
    axis = R2 @ a_tool
    assert axis[2] < -0.99 and np.linalg.norm(p2 + c0 * axis - p_obj) < 1.0
    assert abs(q[5]) < 1e-6


def test_no_checkpoint_split_when_already_at_the_checkpoint():
    plan = build_plan([300, 0, 150], [1, 0, 0], tip_cm=10.04, stop_cm=-4.0, open_cm=10.0, slow_cm=5.0, fast=150, slow=20)
    assert [s[0] for s in plan.segments(5.0, split_cm=10.0)] == ["fast", "slow"]


def test_face_down_holds_the_wrist_and_points_the_camera_straight_down():
    k = arm()
    a_tool = np.array([0.0, 0.0, 1.0])
    rng = np.random.default_rng(5)
    q_ref = None
    for _ in range(6000):                                    # a posture with the camera looking straight down
        q = np.array([rng.uniform(-90, 90), rng.uniform(-120, 0), rng.uniform(20, 150), rng.uniform(-200, 0),
                      rng.uniform(-90, 90), rng.uniform(-90, 90)])
        if (k.fk(q)[1] @ a_tool)[2] < -0.9999:
            q_ref = q
            break
    assert q_ref is not None
    start = q_ref + np.array([5.0, -12.0, 15.0, 0.0, 0.0, 0.0])           # tilted, nearby
    p_now = k.fk(start)[0]
    wrist = {3: float(q_ref[3]), 4: float(q_ref[4]), 5: float(q_ref[5])}
    res = k.face_down_ik_multi(p_now, a_tool, start, wrist)
    assert res is not None
    q, tilt, err = res
    assert tilt < 1.0 and err < 250.0
    assert all(abs(q[i] - wrist[i]) < 1e-6 for i in (3, 4, 5))            # the wrist joints are exactly where we put them


def test_face_down_on_the_real_arm_model_if_available():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "data", "arm_kinematics.xml")
    if not os.path.exists(path):
        return
    k = Kinematics.from_bytes(open(path, "rb").read(), 1000.0)
    q0 = np.array([-0.95, 33.2, -18.66, -86.01, 8.92, -272.45])           # the reset pose: the camera looks 37 deg down
    p0 = k.fk(q0)[0]
    res = k.face_down_ik_multi(p0, np.array([0.0, 0.0, 1.0]), q0, {3: -180.11, 4: 45.05, 5: -191.84})
    assert res is not None and res[1] < 1.0 and res[2] < 120.0           # straight down, near where it was


def test_vertical_descent_with_the_camera_down_and_joint_5_locked():
    from mimic.plan import attach_joints_down, line_plan
    k = arm()
    a_tool = np.array([0.0, 0.0, 1.0])
    seed = np.array([10.0, -60.0, 90.0, -110.0, 35.0, 50.0])
    res = k.face_down_ik_multi(k.fk(seed)[0], a_tool, seed, {5: 50.0}, max_tilt_deg=0.01)      # a camera-down start pose
    assert res is not None
    q0 = res[0]
    p0 = k.fk(q0)[0]
    plan = line_plan(p0, (0, 0, -1), 80.0, 30.0, kind="descent")
    ok = attach_joints_down(plan, k, q0, lambda x, y, z: (x, y, z), a_tool, {5})
    assert ok and "locked" in plan.method
    q_end = plan.waypoints[-1].joints
    assert abs(q_end[5] - q0[5]) < 1e-6 and plan.joint_travel_deg[5] < 1e-6        # joint 5 never moved
    p_end, R_end = k.fk(q_end)
    assert np.linalg.norm(p_end - (p0 + [0, 0, -80.0])) < 1.0 and (R_end @ a_tool)[2] < -0.9999   # still straight down


def test_face_down_above_a_point_with_the_wrist_fixed():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "data", "arm_kinematics.xml")
    if not os.path.exists(path):
        return
    k = Kinematics.from_bytes(open(path, "rb").read(), 1000.0)
    q0 = np.array([-0.95, 33.2, -18.66, -86.01, 8.92, -272.45])
    res = k.face_down_above_ik_multi(np.array([500.0, 25.0]), np.array([0.0, 0.0, 1.0]), q0, {4: 15.29, 5: -181.92})
    assert res is not None
    q, tilt, err, z = res
    assert tilt < 0.5 and err < 1.0 and 100 < z < 400                      # above the point, facing down, at the height the arm gives
    assert abs(q[4] - 15.29) < 1e-6 and abs(q[5] + 181.92) < 1e-6


def test_face_down_above_on_the_toy_arm():
    k = arm()
    seed = np.array([10.0, -60.0, 90.0, -110.0, 35.0, 50.0])
    ref = k.face_down_ik_multi(k.fk(seed)[0], np.array([0.0, 0.0, 1.0]), seed, {5: 50.0}, max_tilt_deg=0.01)[0]
    xy = k.fk(ref)[0][:2]
    res = k.face_down_above_ik_multi(xy, np.array([0.0, 0.0, 1.0]), seed, {4: float(ref[4]), 5: 50.0})
    assert res is not None and res[1] < 1.0 and res[2] < 2.0


def test_face_down_above_falls_back_to_the_nearest_wrist_value():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "data", "arm_kinematics.xml")
    if not os.path.exists(path):
        return
    k = Kinematics.from_bytes(open(path, "rb").read(), 1000.0)
    q0 = np.array([-0.95, 33.2, -18.66, -86.01, 8.92, -272.45])
    a = np.array([0.0, 0.0, 1.0])
    # an object far out: joint 4 = 15.29 and joint 5 = -181.92 together cannot look straight down above it ...
    far = np.array([700.0, 30.0])
    assert k.face_down_above_ik_multi(far, a, q0, {4: 15.29, 5: -181.92}, tries=12) is None
    # ... but with only joint 5 held there is a family of solutions, and we can take the one nearest joint 4 = 15.29
    res = k.face_down_above_ik_multi(far, a, q0, {5: -181.92}, tries=40, prefer={4: 15.29})
    assert res is not None and res[1] < 1.0 and res[2] < 2.0 and abs(res[0][5] + 181.92) < 1e-6
