import numpy as np

from mimic.handeye import compose, quat_to_rot, rot_to_quat, rotation_error_deg, solve_handeye


def rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    th = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def test_quaternion_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        R = rot(rng.normal(size=3), rng.uniform(0, 180))
        assert np.allclose(quat_to_rot(rot_to_quat(R)), R, atol=1e-9)


def test_recovers_a_known_camera_mount_from_synthetic_views():
    rng = np.random.default_rng(0)
    R_cg, t_cg = rot([0.2, 1.0, 0.1], 97.7), np.array([83.0, -14.0, -87.0])      # the TRUE camera pose in the gripper frame
    R_tw, t_tw = rot([0, 0, 1], 20), np.array([700.0, 40.0, 0.0])                 # a marker fixed on the table, in the world
    Rg, tg, Rt, tt = [], [], [], []
    for _ in range(14):                                                           # gripper poses looking roughly at the marker
        R_gw = rot(rng.normal(size=3), rng.uniform(150, 210)) @ rot(rng.normal(size=3), rng.uniform(0, 25))
        t_gw = np.array([450.0, 0.0, 350.0]) + rng.uniform(-70, 70, 3)
        R_cw, t_cw = compose(R_gw, t_gw, R_cg, t_cg)                              # camera in the world
        R_tc = R_cw.T @ R_tw                                                      # marker in the camera frame
        t_tc = R_cw.T @ (t_tw - t_cw)
        Rg.append(R_gw), tg.append(t_gw), Rt.append(R_tc), tt.append(t_tc)
    R, t = solve_handeye(Rg, tg, Rt, tt)
    assert rotation_error_deg(R, R_cg) < 0.5 and np.linalg.norm(t - t_cg) < 3.0   # within half a degree and 3 mm
