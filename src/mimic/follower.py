"""Moves a gripper toward a moving target: smoothing, deadband, proportional steps, speed limit, bounds."""
from dataclasses import dataclass

import numpy as np


@dataclass
class FollowConfig:
    bounds: tuple = (250, 600, -300, 300, 0, 450)   # xmin, xmax, ymin, ymax, zmin, zmax (mm)
    deadband_mm: float = 10.0       # closer than this to the target: do not move (stops jitter)
    gain: float = 0.5               # fraction of the remaining error to cover per update
    max_step_mm: float = 60.0       # largest single move
    max_speed_mm_s: float = 100.0   # mm/s limit on top of the step limit
    smoothing: float = 0.5          # 0..1 weight of the newest target sample
    lost_disengage_s: float = 3.0   # target gone this long: disengage (hold position)


class ObjectFollower:
    def __init__(self, cfg):
        self.cfg, self.engaged = cfg, False
        self._target = None
        self._last_t = self._last_seen = None

    def engage(self, t):
        self.engaged, self._target, self._last_t, self._last_seen = True, None, t, t

    def disengage(self):
        self.engaged = False

    def update(self, target, cur, t):
        """target / cur: (x, y, z) mm in the robot frame, target None when the object is not seen.
        -> (new position to command or None, state)  state: idle | tracking | settled | lost | disengaged"""
        cfg = self.cfg
        if not self.engaged:
            return None, "idle"
        dt = min(max(t - self._last_t, 0.0), 1.0)
        self._last_t = t
        if target is None:
            if t - self._last_seen > cfg.lost_disengage_s:
                self.disengage()
                return None, "disengaged"
            return None, "lost"
        self._last_seen = t
        tgt = np.array(target, float)
        self._target = tgt if self._target is None else (1 - cfg.smoothing) * self._target + cfg.smoothing * tgt
        lo, hi = np.array(cfg.bounds[0::2], float), np.array(cfg.bounds[1::2], float)
        goal = np.clip(self._target, lo, hi)
        cur = np.array(cur, float)
        err = goal - cur
        dist = float(np.linalg.norm(err))
        if dist < cfg.deadband_mm:
            return None, "settled"
        step = min(dist * cfg.gain, cfg.max_step_mm, max(cfg.max_speed_mm_s * dt, 1.0))
        return tuple(np.clip(cur + err / dist * step, lo, hi)), "tracking"
