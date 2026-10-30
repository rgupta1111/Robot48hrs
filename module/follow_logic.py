"""Pure follow logic for the Viam module (no Viam SDK, no camera): detections + gripper position -> next target.

Same behaviour as src/follow_object.py's relative mode: on engage the gripper stays put and then copies the locked
object's motion in the image (mm_per_px), through ObjectFollower's smoothing / deadband / speed limits / bounds.
"""
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mimic.follower import FollowConfig, ObjectFollower  # noqa: E402
from tidy.geometry import relative_target  # noqa: E402


@dataclass
class Detection:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    class_name: str

    @property
    def centre(self):
        return (self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2


@dataclass
class FollowSettings:
    lock: str = "cup"
    min_confidence: float = 0.3
    mm_per_px: float = 0.8
    swap_xy: bool = False
    flip_u: bool = False
    flip_v: bool = False
    follow: FollowConfig = field(default_factory=FollowConfig)


class FollowLogic:
    def __init__(self, settings):
        self.s = settings
        self.follower = ObjectFollower(settings.follow)
        self.status, self.object_px, self.target = "idle", None, None
        self._ref = None            # (object pixel, gripper position) at engage
        self._last_px = None

    @property
    def engaged(self):
        return self.follower.engaged

    def engage(self, t):
        self._ref = None
        self.follower.engage(t)

    def disengage(self):
        self.follower.disengage()
        self._ref = None
        self.status = "idle"

    def set_lock(self, name):
        self.s.lock, self._last_px, self._ref = name, None, None

    def _pick(self, detections):
        """Highest-confidence detection of the locked class, preferring the one nearest the previous position."""
        want = self.s.lock.strip().lower()
        cands = [d for d in detections if d.class_name.lower() == want and d.confidence >= self.s.min_confidence]
        if not cands:
            return None
        if self._last_px is None:
            return max(cands, key=lambda d: d.confidence)
        lx, ly = self._last_px
        return min(cands, key=lambda d: (d.centre[0] - lx) ** 2 + (d.centre[1] - ly) ** 2)

    def step(self, detections, cur, t):
        """detections: [Detection] in pixels; cur: gripper (x, y, z) mm in the robot frame; t: seconds.
        -> new (x, y, z) to command, or None. self.status / object_px / target describe what happened."""
        det = self._pick(detections)
        self.object_px = det.centre if det else None
        self._last_px = self.object_px
        self.target = None
        if det and self.follower.engaged:
            if self._ref is None:
                self._ref = (self.object_px, tuple(cur))
            s = self.s
            self.target = relative_target(self._ref[0], self._ref[1], self.object_px, s.mm_per_px,
                                          s.swap_xy, s.flip_u, s.flip_v)
        new, self.status = self.follower.update(self.target, cur, t)
        return new
