import numpy as np

from mimic.controller import MimicConfig, MimicController, PinchGripper
from mimic.sinks import SimSink


class Pose:
    def __init__(self, rel):
        self.rel = np.array(rel, float)


class Hand:
    def __init__(self, ratio=0.5, heading=0.0, elevation=0.0):
        self.ratio, self.heading_deg, self.elevation_deg = ratio, heading, elevation


def engaged(cfg, pose=None, hand=None):
    ctl = MimicController(cfg)
    assert ctl.engage(pose, 0.0, hand)
    return ctl


def test_idle_until_engaged():
    cmd, st = MimicController(MimicConfig()).update(None, 0)
    assert cmd is None and not st.engaged


def test_engage_holds_start_pose():
    cfg = MimicConfig(start=(150, 0, 150))
    ctl = engaged(cfg, Pose([0.5, 0.2, -0.1]))
    cmd, st = ctl.update(Pose([0.5, 0.2, -0.1]), 0.1)
    assert (cmd.x, cmd.y, cmd.z) == (150, 0, 150) and st.state == "tracking"


def test_directions_and_speed_limit():
    cfg = MimicConfig(start=(150, 0, 150), reach_mm=200, depth_gain=1.0, max_speed_mm_s=100,
                      bounds=(0, 400, -400, 400, 0, 400))
    ctl = engaged(cfg, Pose([0, 0, 0]))
    cmd, _ = ctl.update(Pose([0, 0, -1.0]), 0.1)      # reach toward the camera
    assert cmd.x == 160 and cmd.y == 0                 # forward, limited to 100 mm/s * 0.1 s
    for i in range(2, 40):
        cmd, _ = ctl.update(Pose([-0.5, -0.5, -1.0]), i * 0.1)  # your right and up
    assert cmd.x == 350 and cmd.y == -100 and cmd.z == 250


def test_bounds_clamp():
    cfg = MimicConfig(reach_mm=1000, max_speed_mm_s=1e6, bounds=(100, 200, -50, 50, 50, 200))
    ctl = engaged(cfg, Pose([0, 0, 0]))
    cmd, st = ctl.update(Pose([5, 5, 5]), 0.1)
    assert st.clamped and 100 <= cmd.x <= 200 and -50 <= cmd.y <= 50 and 50 <= cmd.z <= 200


def test_lost_then_disengage():
    ctl = engaged(MimicConfig(lost_disengage_s=1.0), Pose([0, 0, 0]))
    assert ctl.update(None, 0.5)[1].state == "lost"
    assert not ctl.update(None, 2.0)[1].engaged


def test_pinch_only_drives_joint():
    cfg = MimicConfig(pinch_only=True, pinch_joint=5, joint_speed=1e6, smoothing=1.0, lost_disengage_s=1e9)
    ctl = engaged(cfg, None, Hand(1.0))
    assert ctl.update(None, 0.1, Hand(1.0))[0].joint_deg == 90
    assert ctl.update(None, 0.2, Hand(0.0))[0].joint_deg == 0
    assert ctl.update(None, 5.0, None)[1].state == "lost"   # hand gone: hold, never disengage


def test_joint_eases_from_seed():
    cfg = MimicConfig(pinch_only=True, pinch_joint=0, joint_speed=10, smoothing=1.0)
    ctl = engaged(cfg, None, Hand(1.0))
    ctl.seed_joint(0.0)
    assert ctl.update(None, 0.1, Hand(1.0))[0].joint_deg == 1.0


def test_pinch_gripper_hysteresis():
    g = PinchGripper(0.35, 0.6)
    assert g.update(0.5) is None and g.update(0.2) is False and g.update(0.5) is None and g.update(0.7) is True


def test_sim_sink_log():
    s = SimSink()
    s.prepare((1, 2, 3))
    assert len(s.log) == 1 and s.log[0][1:4] == (1, 2, 3)
