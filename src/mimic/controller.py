"""Turns tracked arm / hand state into robot commands: engage, recenter, speed limit, bounds, pinch -> joint."""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Command:
    x: float
    y: float
    z: float
    yaw_deg: float = 0.0
    tilt_deg: float = 0.0
    joint_deg: float = None   # None unless a pinch joint is configured


@dataclass
class Status:
    state: str = "idle"       # idle | tracking | lost | disengaged
    engaged: bool = False
    target: tuple = None
    clamped: bool = False
    yaw_deg: float = 0.0
    tilt_deg: float = 0.0
    joint_deg: float = None


@dataclass
class MimicConfig:
    start: tuple = (150.0, 0.0, 150.0)
    reach_mm: float = 250.0
    depth_gain: float = 0.6
    max_speed_mm_s: float = 150.0
    bounds: tuple = (60, 320, -200, 200, 40, 300)   # xmin, xmax, ymin, ymax, zmin, zmax
    flip_x: bool = False
    flip_y: bool = False
    wrist_yaw: bool = True
    wrist_tilt: bool = False
    flip_yaw: bool = False
    flip_tilt: bool = False
    pinch_only: bool = False
    lost_disengage_s: float = 3.0
    pinch_joint: int = None
    joint_ratios: tuple = (0.0, 1.0)    # pinch ratio when closed, spread
    joint_span: tuple = (0.0, 90.0)     # joint degrees at those ratios
    joint_speed: float = 90.0           # deg/s
    flip_joint: bool = False
    joint_absolute: bool = True         # False: joint_span is an offset from the joint's angle at engage
    smoothing: float = 0.35             # 0..1 weight of the newest sample in the low-pass filters
    max_yaw_deg: float = 90.0
    max_tilt_deg: float = 45.0


class PinchGripper:
    """Hysteresis on the pinch ratio: closes below `close`, reopens above `open_`."""

    def __init__(self, close=0.35, open_=0.6):
        self.close, self.open_, self.open = close, open_, True

    def reset(self):
        self.open = True

    def update(self, ratio):
        """New gripper state (True = open) if it changed, else None."""
        if ratio is None:
            return None
        if self.open and ratio < self.close:
            self.open = False
            return False
        if not self.open and ratio > self.open_:
            self.open = True
            return True
        return None


def _lerp(a, b, f):
    return a + (b - a) * f


class MimicController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.engaged = False
        self.debug = None
        self._ref = np.zeros(3)
        self._last_delta = np.zeros(3)
        self._hand_ref = (0.0, 0.0)
        self._target = np.array(cfg.start, float)
        self._yaw = self._tilt = 0.0
        self._ratio = None
        self._joint = None
        self._last_t = None
        self._last_seen = 0.0

    # --- engage / disengage -------------------------------------------------
    def engage(self, pose, t, hand=None):
        """Start following. The current hand position becomes the robot's start pose."""
        if pose is None and self.cfg.pinch_joint is None:
            return False  # following needs the arm pose; only the pinch joint can run on a hand alone
        self._ref = pose.rel.copy() if pose is not None else np.zeros(3)
        self._target = np.array(self.cfg.start, float)
        self._yaw = self._tilt = 0.0
        self._hand_ref = (hand.heading_deg, hand.elevation_deg) if hand is not None else None
        self._ratio = None
        self._last_delta = np.zeros(3)
        self._last_t, self._last_seen = t, t
        self.engaged = True
        return True

    def disengage(self):
        self.engaged = False

    def recenter(self):
        """Make the current hand position the new zero without moving the robot."""
        self._ref = self._ref + self._last_delta
        self._hand_ref = None  # re-captured from the next hand sample

    def seed_joint(self, base_deg):
        """Start easing the pinch joint from its current angle instead of snapping to the pinch value."""
        if self.cfg.joint_absolute:
            self._joint = float(base_deg)

    # --- per frame ----------------------------------------------------------
    def update(self, pose, t, hand=None):
        """-> (Command or None, Status). Command is None when there is nothing new to send."""
        cfg = self.cfg
        st = Status(state="idle", engaged=self.engaged)
        if not self.engaged:
            st.state = "disengaged" if self._last_t is not None else "idle"
            return None, st
        dt = min(max(t - self._last_t, 0.0), 0.2) if self._last_t is not None else 0.0
        self._last_t = t

        seen = hand is not None if cfg.pinch_only else pose is not None
        if seen:
            self._last_seen = t
        elif t - self._last_seen > cfg.lost_disengage_s:
            self.disengage()
            st.engaged, st.state = False, "disengaged"
            return None, st
        else:
            st.state, st.target = "lost", tuple(self._target)
            st.yaw_deg, st.tilt_deg, st.joint_deg = self._yaw, self._tilt, self._joint
            return None, st

        if not cfg.pinch_only:
            self._follow(pose, hand, dt, st)
        if cfg.pinch_joint is not None and hand is not None:
            self._pinch(hand, dt)
        st.state, st.target = "tracking", tuple(self._target)
        st.yaw_deg, st.tilt_deg, st.joint_deg = self._yaw, self._tilt, self._joint
        return Command(*self._target, self._yaw, self._tilt, self._joint), st

    def _follow(self, pose, hand, dt, st):
        cfg = self.cfg
        d = pose.rel - self._ref
        self._last_delta = d
        # camera frame -> robot frame: reach toward camera = forward, hand to your right = robot's right, up = up
        fwd, lat, up = -d[2] * cfg.depth_gain, d[0], -d[1]
        if cfg.flip_x:
            fwd = -fwd
        if cfg.flip_y:
            lat = -lat
        want = np.array(cfg.start, float) + cfg.reach_mm * np.array([fwd, lat, up])
        lo, hi = np.array(cfg.bounds[0::2], float), np.array(cfg.bounds[1::2], float)
        clamped = np.clip(want, lo, hi)
        st.clamped = bool(np.any(clamped != want))
        step = clamped - self._target
        limit = cfg.max_speed_mm_s * dt
        dist = float(np.linalg.norm(step))
        if dist > limit > 0 or (limit == 0 and dist > 0):
            step *= limit / dist
        self._target = self._target + step
        if hand is not None and (cfg.wrist_yaw or cfg.wrist_tilt):
            if self._hand_ref is None:
                self._hand_ref = (hand.heading_deg, hand.elevation_deg)
            a = cfg.smoothing
            if cfg.wrist_yaw:
                yaw = _wrap(hand.heading_deg - self._hand_ref[0]) * (-1 if cfg.flip_yaw else 1)
                self._yaw = _lerp(self._yaw, float(np.clip(yaw, -cfg.max_yaw_deg, cfg.max_yaw_deg)), a)
            if cfg.wrist_tilt:
                tilt = (hand.elevation_deg - self._hand_ref[1]) * (-1 if cfg.flip_tilt else 1)
                self._tilt = _lerp(self._tilt, float(np.clip(tilt, -cfg.max_tilt_deg, cfg.max_tilt_deg)), a)

    def _pinch(self, hand, dt):
        cfg = self.cfg
        raw = hand.ratio
        self._ratio = raw if self._ratio is None else _lerp(self._ratio, raw, cfg.smoothing)
        r0, r1 = cfg.joint_ratios
        f = float(np.clip((self._ratio - r0) / (r1 - r0), 0.0, 1.0)) if r1 != r0 else 0.0
        if cfg.flip_joint:
            f = 1.0 - f
        goal = _lerp(cfg.joint_span[0], cfg.joint_span[1], f)
        cur = goal if self._joint is None else self._joint
        step = cfg.joint_speed * dt
        self._joint = cur + float(np.clip(goal - cur, -step, step))
        self.debug = {"ratio_raw": raw, "ratio_smooth": self._ratio, "goal": goal}


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0
