"""Smooth motion and collision avoidance for the gripper.

MotionSmoother   acceleration-limited velocity -> position stream (no jerks, even when the camera updates unevenly)
approach_speed   fast while far, easing down to a crawl at the stop distance
Obstacles        axis-aligned keep-out boxes (the machine's table / walls from the Viam frame system, plus any you add)
                 inflated by a clearance margin; commanded positions are pushed out of them, sliding along the surface
"""
import math
from dataclasses import dataclass

import numpy as np


class MotionSmoother:
    def __init__(self, pos, max_accel=800.0, max_speed=300.0):
        self.pos, self.vel = np.array(pos, float), np.zeros(3)
        self.max_accel, self.max_speed = max_accel, max_speed

    def reset(self, pos):
        self.pos, self.vel = np.array(pos, float), np.zeros(3)

    def step(self, v_des, dt):
        """Accelerate toward v_des (mm/s) by at most max_accel, integrate, return the new position."""
        v_des = np.asarray(v_des, float)
        speed = np.linalg.norm(v_des)
        if speed > self.max_speed:
            v_des = v_des * (self.max_speed / speed)
        dv = v_des - self.vel
        n = np.linalg.norm(dv)
        limit = self.max_accel * dt
        if n > limit:
            dv = dv * (limit / n)
        self.vel = self.vel + dv
        self.pos = self.pos + self.vel * dt
        return self.pos


def approach_speed(d_cm, stop_cm, slow_cm, fast, slow):
    """mm/s for a gripper-to-object distance: `fast` until slow_cm, then a smooth ramp down to `slow` at stop_cm,
    and 0 once at or inside stop_cm."""
    if d_cm <= stop_cm:
        return 0.0
    if d_cm >= slow_cm:
        return fast
    f = (d_cm - stop_cm) / max(slow_cm - stop_cm, 1e-9)
    f = f * f * (3 - 2 * f)
    return slow + (fast - slow) * f


@dataclass(eq=False)          # compare boxes by identity (they hold numpy arrays)
class Box:
    lo: np.ndarray
    hi: np.ndarray
    name: str = "obstacle"


class Obstacles:
    def __init__(self, boxes=(), margin_mm=60.0):
        self.boxes, self.margin = list(boxes), margin_mm

    def add(self, lo_hi, name="obstacle"):
        xmin, xmax, ymin, ymax, zmin, zmax = lo_hi
        self.boxes.append(Box(np.array([xmin, ymin, zmin], float), np.array([xmax, ymax, zmax], float), name))

    def _inside(self, p, b):
        return bool(np.all(p > b.lo - self.margin) and np.all(p < b.hi + self.margin))

    def blocking(self, p):
        return [b for b in self.boxes if self._inside(np.asarray(p, float), b)]

    def push_out(self, p):
        """Move p out of every inflated box along the shortest way. -> (new point, names that were in the way)."""
        p, hit = np.array(p, float), []
        for _ in range(3):                        # a couple of passes in case pushing out of one box enters another
            moved = False
            for b in self.boxes:
                if self._inside(p, b):
                    lo, hi = b.lo - self.margin, b.hi + self.margin
                    gaps = np.concatenate([p - lo, hi - p])        # distance to each of the six faces
                    i = int(np.argmin(gaps))
                    axis = i % 3
                    p[axis] = lo[axis] if i < 3 else hi[axis]
                    hit.append(b.name)
                    moved = True
            if not moved:
                break
        return p, sorted(set(hit))

    def constrain(self, p_from, p_to, step_mm=10.0):
        """Walk from p_from to p_to; wherever the path would enter a keep-out box, slide along its surface.
        -> (safe end point, names of obstacles that were in the way)."""
        a, b = np.asarray(p_from, float), np.asarray(p_to, float)
        n = max(int(np.linalg.norm(b - a) // step_mm), 1)
        cur, hit = a, []
        for i in range(1, n + 1):
            nxt = a + (b - a) * (i / n)
            cur, h = self.push_out(nxt)
            hit += h
        return cur, sorted(set(hit))

    def arm_clearance(self, points, radius_mm=0.0, step_mm=20.0):
        """Smallest gap (mm) between the arm's skeleton (polyline through `points`, robot frame) thickened by radius_mm and
        any box. Negative = touching or inside. -> (gap, name of the nearest obstacle)."""
        pts = [np.asarray(p, float) for p in points]
        samples = []
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(int(np.linalg.norm(b - a) // step_mm), 1)
            samples += [a + (b - a) * (i / n) for i in range(n)]
        samples.append(pts[-1])
        best, name = float("inf"), ""
        for b in self.boxes:
            d = np.linalg.norm(np.maximum(np.maximum(b.lo - np.array(samples), np.array(samples) - b.hi), 0.0), axis=1)
            g = float(d.min()) - radius_mm
            if g < best:
                best, name = g, b.name
        return best, name

    # ---- loading from the machine ------------------------------------------------------------------------------
    @classmethod
    def from_viam(cls, robot, margin_mm=60.0, skip_names=()):
        """Box obstacles attached to the world frame in the machine's frame system (best effort).
        -> (Obstacles, [names loaded], [notes about things skipped])."""
        obs, loaded, notes = cls(margin_mm=margin_mm), [], []
        try:
            configs = robot._run(robot._robot.get_frame_system_config())
        except Exception as e:
            return obs, loaded, [f"could not read the frame system ({type(e).__name__}: {e})"]
        for cfg in configs:
            tf = cfg.frame
            name = tf.reference_frame
            g = tf.physical_object
            if not g.HasField("box") or name in skip_names or "cam" in name.lower():
                continue
            if tf.pose_in_observer_frame.reference_frame != "world":
                notes.append(f"{name}: not attached to the world frame, skipped")
                continue
            p, o = tf.pose_in_observer_frame.pose, None
            theta = p.theta % 360
            if not (abs(p.o_z - 1) < 1e-6 and any(abs(theta - a) < 1e-3 for a in (0, 90, 180, 270, 360))):
                notes.append(f"{name}: rotated box, skipped")
                continue
            dims = np.array([g.box.dims_mm.x, g.box.dims_mm.y, g.box.dims_mm.z], float)
            if abs(theta - 90) < 1e-3 or abs(theta - 270) < 1e-3:
                dims[0], dims[1] = dims[1], dims[0]
            c = np.array([p.x, p.y, p.z]) + np.array([g.center.x, g.center.y, g.center.z]) * 1.0
            x, y, z = robot.table.to_robot(*c)
            obs.boxes.append(Box(np.array([x, y, z]) - dims / 2, np.array([x, y, z]) + dims / 2, name))
            loaded.append(name)
        return obs, loaded, notes

    @classmethod
    def lab_world(cls, margin_mm=60.0):
        """The hackathon lab world described in arm_mimic's presets: table top z=-123, walls at x=-525 and y=+625,
        ceiling at z=1050 (as half-spaces)."""
        big = 5000.0
        o = cls(margin_mm=margin_mm)
        o.add((-big, big, -big, big, -big, -123.0), "table")
        o.add((-big, -525.0, -big, big, -big, big), "wall-back")
        o.add((-big, big, 625.0, big, -big, big), "wall-left")
        o.add((-big, big, -big, big, 1050.0, big), "ceiling")
        return o
