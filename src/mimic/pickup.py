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


class PickupPlanner:
    def __init__(self, J, cfg=None):
        self.J, self.cfg = np.asarray(J, float), cfg or PickupConfig()
        self.state, self.cycles, self.lost = "OPEN", 0, 0

    def _centre_move(self, s):
        c = self.cfg
        target = np.array([c.target_uv[0], c.target_uv[1], s[2]])
        return servo_step(self.J, s, target, (0, 1), c.gain, c.max_step_mm)

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
