"""Pick-up sequence for a gripper-mounted camera, as a pure state machine (no robot, no camera: easy to test).

    OPEN -> CENTRE -> DESCEND -> GRASP -> LIFT -> DONE          (ABORT on lost object / too many cycles)

Each cycle the caller gives the object's image features s = (u, v, h) (see servo.features) and the camera-to-object
distance in mm (None if unknown), and gets back an Action to carry out. Moves are small steps along the robot axes
(mm, z up); the caller moves the arm, waits for the camera to catch up, then calls step() again.
"""
from dataclasses import dataclass

import numpy as np

from .servo import servo_step


@dataclass
class PickupConfig:
    target_uv: tuple = (0.0, 0.0)       # where the object should sit in the image when the gripper is over it
    centre_tol: float = 0.04            # image fraction: centred when |u,v error| is below this
    gain: float = 0.4
    max_step_mm: float = 30.0
    descend_step_mm: float = 25.0       # drop per cycle while centred
    grasp_distance_mm: float = 150.0    # camera-to-object distance at which the gripper is around the object
    lift_mm: float = 120.0
    lost_limit: int = 6                 # consecutive cycles without the object before giving up
    max_cycles: int = 200


@dataclass
class Action:
    kind: str                           # move | grip_open | grip_close | hold | done | abort
    move: tuple = (0.0, 0.0, 0.0)       # mm, robot axes (kind == move)
    note: str = ""


def table_floor_z(table_z_mm, tool_length_mm, margin_mm):
    """Lowest arm-tool height (mm, robot frame) that keeps the gripper's fingertips margin_mm above the table.
    The arm reports the pose of its tool flange; with the gripper pointing down its fingertips are tool_length_mm lower."""
    return float(table_z_mm + tool_length_mm + margin_mm)


def advance_along(pos, direction, mm):
    """Point mm millimetres further along a direction vector (e.g. the tool axis), from pos. Zero vector: stays put."""
    d = np.asarray(direction, float)
    n = float(np.linalg.norm(d))
    return np.asarray(pos, float) + (d / n * mm if n > 1e-9 else 0.0)


def smooth_move(prev, move, blend=0.5, max_mm=None):
    """Blend a new move with the previous one (so the direction cannot flip between cycles) and cap its length.
    blend 0 = no smoothing, 0.9 = very smooth but sluggish. max_mm is the longest allowed step (speed limit * dt)."""
    m = blend * np.asarray(prev, float) + (1.0 - blend) * np.asarray(move, float)
    n = float(np.linalg.norm(m))
    return m * (max_mm / n) if max_mm is not None and n > max_mm else m


def check_jacobian(J, min_response=3e-4, max_cos=0.9):
    """Sanity-check a learned image response before servoing with it. -> an error message, or None when usable.
    Moving the gripper in x and y must shift the object in clearly DIFFERENT image directions, and each axis must shift
    it by more than noise (min_response, image fraction per mm; ~0.006 of the frame for a 20 mm probe). If both columns
    point the same way the probe was dominated by drift or a stale frame and servoing on it would drive off.
    A weak axis is fine as long as it is above the noise: the two axes need not respond equally."""
    A = np.asarray(J, float)[:2, :2]
    cx, cy = A[:, 0], A[:, 1]
    nx, ny = np.linalg.norm(cx), np.linalg.norm(cy)
    if min(nx, ny) < min_response:
        return "the object barely moved in the image for one of the probe moves (" + ("x" if nx < ny else "y") + ")"
    if abs(float(cx @ cy)) / (nx * ny) > max_cos:
        return "moving x and y shifted the object the same way (probe unreliable: drift or a stale frame)"
    return None


class PickupPlanner:
    def __init__(self, J, cfg=None):
        self.J, self.cfg = np.asarray(J, float), cfg or PickupConfig()
        self.state, self.cycles, self.lost, self._errs = "OPEN", 0, 0, []

    def _centre_move(self, s):
        c = self.cfg
        target = np.array([c.target_uv[0], c.target_uv[1], s[2]])
        Jxy = self.J.copy()
        Jxy[:, 2] = 0.0                  # centre with x / y only: height is the descent's job, never the centring's
        return servo_step(Jxy, s, target, (0, 1), c.gain, c.max_step_mm)

    def step(self, s, dist_mm):
        """s: (u, v, h) or None when the object is not seen. -> Action."""
        c = self.cfg
        self.cycles += 1
        if self.cycles > c.max_cycles:
            return self._abort("too many cycles")
        if self.state == "OPEN":
            self.state = "CENTRE"
            return Action("grip_open", note="open gripper")
        if self.state == "GRASP":
            self.state = "LIFT"
            return Action("grip_close", note="close gripper")
        if self.state == "LIFT":
            self.state = "DONE"
            return Action("move", (0.0, 0.0, c.lift_mm), "lift")
        if self.state in ("DONE", "ABORT"):
            return Action("done" if self.state == "DONE" else "abort")

        if s is None:                                   # CENTRE / DESCEND need the object in view
            self.lost += 1
            return self._abort("lost the object") if self.lost > c.lost_limit else Action("hold", note="object not seen")
        self.lost = 0
        s = np.asarray(s, float)
        err = np.array(c.target_uv) - s[:2]
        centred = bool(np.max(np.abs(err)) < c.centre_tol)
        self._errs = (self._errs + [float(np.max(np.abs(err)))])[-5:]
        if len(self._errs) == 5 and all(b > a + 0.005 for a, b in zip(self._errs, self._errs[1:])):
            return self._abort("image error keeps growing: moving the wrong way")

        if self.state == "CENTRE":
            if centred:
                self.state = "DESCEND"
                return Action("hold", note="centred: descending")
            return Action("move", tuple(self._centre_move(s)), "centring")

        # DESCEND
        if dist_mm is not None and dist_mm <= c.grasp_distance_mm:
            self.state = "GRASP"
            return Action("hold", note=f"at grasp distance ({dist_mm:.0f} mm)")
        move = self._centre_move(s)                     # keep centred while dropping
        if np.max(np.abs(err)) < 3 * c.centre_tol:      # only drop while roughly centred
            move = move + np.array([0.0, 0.0, -c.descend_step_mm])
        return Action("move", tuple(move), "descending")

    def _abort(self, reason):
        self.state = "ABORT"
        return Action("abort", note=reason)
