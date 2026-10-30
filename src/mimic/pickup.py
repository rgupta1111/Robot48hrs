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


def check_jacobian(J, min_response=1e-4, min_ratio=0.2):
    """Sanity-check a learned image response before servoing with it. -> an error message, or None when usable.
    Moving the gripper in x and y must shift the object differently in the image: if the x and y columns point the
    same way (or barely respond) the probe was dominated by noise or drift and servoing on it would drive off."""
    sv = np.linalg.svd(np.asarray(J, float)[:2, :2], compute_uv=False)
    if sv[0] < min_response:
        return "the object barely moved in the image during the probe moves"
    if sv[1] < min_ratio * sv[0]:
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
