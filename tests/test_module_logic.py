import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "module"))

from follow_logic import Detection, FollowLogic, FollowSettings  # noqa: E402
from mimic.follower import FollowConfig  # noqa: E402


def cup(cx, cy, conf=0.9, name="cup"):
    return Detection(cx - 20, cy - 20, cx + 20, cy + 20, conf, name)


def make(**kw):
    s = FollowSettings(follow=FollowConfig(bounds=(0, 1000, -500, 500, 0, 500), smoothing=1.0, gain=1.0,
                                           max_step_mm=1000, max_speed_mm_s=1e6), **kw)
    return FollowLogic(s)


def test_idle_and_no_motion_until_engaged():
    lg = make()
    assert lg.step([cup(300, 300)], (400, 0, 200), 0.1) is None and lg.status == "idle"


def test_first_frame_zeroes_then_copies_motion():
    lg = make(mm_per_px=1.0)
    lg.engage(0.0)
    assert lg.step([cup(300, 300)], (400, 0, 200), 0.1) is None      # zeroed: no motion yet
    new = lg.step([cup(300, 340)], (400, 0, 200), 0.2)               # object moved 40 px down -> -x by default
    assert new is not None and abs(new[0] - 360) < 1e-6 and abs(new[1]) < 1e-6 and new[2] == 200


def test_ignores_other_classes_and_low_confidence():
    lg = make(min_confidence=0.5)
    lg.engage(0.0)
    lg.step([cup(300, 300, name="bottle"), cup(100, 100, conf=0.2)], (400, 0, 200), 0.1)
    assert lg.object_px is None and lg.status == "lost"


def test_prefers_detection_near_previous_lock():
    lg = make()
    lg.engage(0.0)
    lg.step([cup(300, 300)], (400, 0, 200), 0.1)
    lg.step([cup(900, 900, conf=0.99), cup(310, 305, conf=0.5)], (400, 0, 200), 0.2)
    assert lg.object_px == (310, 305)


def test_disengages_after_object_lost():
    lg = make()
    lg.engage(0.0)
    lg.step([cup(300, 300)], (400, 0, 200), 0.1)
    assert lg.step([], (400, 0, 200), 5.0) is None and not lg.engaged and lg.status == "disengaged"
