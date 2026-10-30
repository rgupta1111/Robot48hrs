"""An execution plan for the final approach: a straight line along the camera's viewing direction from where the
gripper is (object centred) to the stop distance, with a speed profile and the gripper-open point, optionally
solved in joint space so joint movement stays minimal."""
from dataclasses import dataclass, field

import numpy as np

from .safety import approach_speed


@dataclass
class Waypoint:
    s_mm: float                  # distance travelled along the line
    tip_cm: float                # gripper-to-object distance when it gets here
    speed: float                 # mm/s the arm should move at on the way to this point
    pos: np.ndarray              # robot-frame position
    joints: object = None        # joint angles (deg) when planned in joint space


@dataclass
class Plan:
    p0: np.ndarray
    direction: np.ndarray
    waypoints: list
    start_tip_cm: float
    stop_cm: float
    open_s_mm: float             # travelled distance at which the gripper opens (0 = open before moving)
    travel_mm: float
    method: str = "cartesian"    # "joint-space IK" when joint waypoints exist
    notes: list = field(default_factory=list)
    joint_travel_deg: object = None
    ok: bool = True
    kind: str = "approach"       # "approach" (toward an object) or "lift" (straight up)

    @property
    def final_tip_cm(self):
        return self.waypoints[-1].tip_cm

    def segments(self, slow_cm, split_cm=None):
        """[(name, first waypoint index, last waypoint index, speed mm/s)]. Cut at split_cm (a checkpoint, e.g. where the
        gripper opens) and where the slow zone starts: 'approach' up to the checkpoint, then 'fast', then 'slow'."""
        n = len(self.waypoints)

        def first_le(tip):
            return next((i for i, w in enumerate(self.waypoints) if w.tip_cm <= tip), n - 1)
        cuts, split_idx = [0, n - 1], None
        if split_cm is not None and self.start_tip_cm > split_cm + 0.3:           # not when we are already at the checkpoint
            split_idx = first_le(split_cm)
            cuts.append(split_idx)
        i_slow = first_le(slow_cm)
        if i_slow > 0:
            cuts.append(i_slow)
        cuts = sorted(set(cuts))
        out = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            v = float(np.median([w.speed for w in self.waypoints[a + 1:b + 1]]))
            name = "approach" if (split_idx is not None and b == split_idx) else ("fast" if (self.waypoints[a].tip_cm + self.waypoints[b].tip_cm) / 2 >= slow_cm else "slow")
            out.append((name, a, b, v))
        return out


def build_plan(p0, direction, tip_cm, stop_cm, open_cm, slow_cm, fast, slow, obstacles=None, bounds=None,
               step_mm=10.0, max_travel_mm=400.0, min_speed=8.0, lateral=None):
    """Straight line from p0 covering the (tip_cm - stop_cm) still to go along `direction` (unit vector). `lateral`
    (mm, perpendicular to direction) is the small remaining centring correction, blended in along the way so the
    line ends exactly in front of the object."""
    p0, d = np.asarray(p0, float), np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    lat = np.zeros(3) if lateral is None else np.asarray(lateral, float)
    want = max(0.0, (tip_cm - stop_cm) * 10.0)
    travel, notes, ok = min(want, max_travel_mm), [], True
    if want > max_travel_mm:
        notes.append(f"capped at {max_travel_mm:.0f} mm of travel (the estimate says {want:.0f} mm)")
    n = max(int(np.ceil(travel / step_mm)), 1)
    pts, prev = [], p0
    for i in range(n + 1):
        s = travel * i / n
        p = p0 + d * s + lat * (s / max(want if want <= max_travel_mm else max_travel_mm, 1e-6))
        if bounds is not None:
            lo, hi = np.array(bounds[0::2], float), np.array(bounds[1::2], float)
            if np.any(p < np.minimum(lo, p0) - 1e-6) or np.any(p > np.maximum(hi, p0) + 1e-6):
                lo_e, hi_e = np.minimum(lo, p0), np.maximum(hi, p0)
                ax = next(k for k in range(3) if p[k] < lo_e[k] - 1e-6 or p[k] > hi_e[k] + 1e-6)
                side = "below" if p[ax] < lo_e[ax] - 1e-6 else "above"
                lim = lo_e[ax] if side == "below" else hi_e[ax]
                notes.append(f"the line leaves the safe workspace after {s:.0f} mm ({'xyz'[ax]} would go {side} the {lim:.0f} mm "
                             f"limit of --bounds): stopping the plan there")
                travel, ok = s - travel / n if i > 0 else 0.0, False
                break
        if obstacles is not None:
            blocked = obstacles.blocking(p)
            if blocked and i > 0:
                notes.append(f"the line would enter {[b.name for b in blocked]} after {s:.0f} mm: stopping the plan there")
                travel, ok = s - travel / n, False
                break
        pts.append((s, p))
        prev = p
    if not pts:
        pts = [(0.0, p0)]
    waypoints = []
    for s, p in pts:
        t = tip_cm - s / 10.0
        sp = max(approach_speed(t, stop_cm, slow_cm, fast, slow), min_speed) if t > stop_cm else 0.0
        waypoints.append(Waypoint(s, t, sp, p))
    open_s = max(0.0, (tip_cm - open_cm) * 10.0)
    return Plan(p0, d, waypoints, tip_cm, stop_cm, open_s, pts[-1][0], notes=notes, ok=ok)


def describe(plan, slow_cm, fast, slow, open_cm, tip_offset_mm=0.0):
    """Human-readable plan lines (shown in the chat and on the page)."""
    lines = [f"Plan: straight line along the camera axis, {plan.travel_mm:.0f} mm, "
             f"from {plan.start_tip_cm:.1f} cm to {plan.stop_cm:.1f} cm from the object ({plan.method})"]
    for name, a, b, v in plan.segments(slow_cm):
        wa, wb = plan.waypoints[a], plan.waypoints[b]
        lines.append(f"  {name:4} {wa.tip_cm:5.1f} -> {wb.tip_cm:4.1f} cm  ({wb.s_mm - wa.s_mm:4.0f} mm at ~{v:.0f} mm/s)"
                     f"   x{wb.pos[0]:.0f} y{wb.pos[1]:.0f} z{wb.pos[2]:.0f}")
    if abs(plan.final_tip_cm - plan.stop_cm) > 0.05:
        lines.append(f"  SHORT PLAN: it ends {plan.final_tip_cm:.1f} cm from the object, not {plan.stop_cm:.1f} cm")
    if plan.stop_cm < 0:
        lines.append(f"  grasp depth: the fingertips end {abs(plan.stop_cm) * 10:.0f} mm beyond the object's front surface")
    if plan.open_s_mm >= 1e8:
        lines.append("  the gripper stays closed (it is holding the object)")
    elif plan.open_s_mm <= 0:
        lines.append(f"  open the gripper first (already within {open_cm:.0f} cm)")
    else:
        lines.append(f"  open the gripper after {plan.open_s_mm:.0f} mm, at {open_cm:.0f} cm from the object")
    if plan.joint_travel_deg is not None:
        lines.append("  joint travel (deg): " + " ".join(f"J{i} {v:.1f}" for i, v in enumerate(plan.joint_travel_deg))
                     + f"   total {float(np.sum(plan.joint_travel_deg)):.1f}")
    if tip_offset_mm == 0:
        lines.append("  note: fingertip offset is 0, so distances are measured from the CAMERA (say \"fingertips are 13 cm in front of the camera\")")
    lines += [f"  note: {n}" for n in plan.notes]
    return lines


def attach_joints(plan, kin, q0_deg, to_arm, max_jump_deg=15.0):
    """Solve every waypoint in joint space, each from the previous solution (so joint movement stays minimal) with the
    tool orientation held. -> True and fills waypoint.joints / plan.joint_travel_deg, else False and a note."""
    _, R = kin.fk(q0_deg)
    prev, sols = np.asarray(q0_deg, float), []
    for i, w in enumerate(plan.waypoints):
        q = prev.copy() if i == 0 else kin.ik(np.asarray(to_arm(*w.pos), float), R, prev)
        if q is None:
            plan.notes.append(f"no joint solution at {w.s_mm:.0f} mm: using Cartesian moves instead")
            return False
        if float(np.max(np.abs(q - prev))) > max_jump_deg:
            plan.notes.append(f"joint jump of {float(np.max(np.abs(q - prev))):.0f} deg at {w.s_mm:.0f} mm: using Cartesian moves instead")
            return False
        sols.append(q)
        prev = q
    for w, q in zip(plan.waypoints, sols):
        w.joints = q
    plan.joint_travel_deg = np.sum(np.abs(np.diff(np.array(sols), axis=0)), axis=0) if len(sols) > 1 else np.zeros(len(q0_deg))
    plan.method = "joint-space IK, minimum joint movement"
    return True


def segment_joint_speed(plan, a, b, tool_speed):
    """Joint speed limit (deg/s) that gives about tool_speed (mm/s) along waypoints a..b, from the joint path itself."""
    qa = np.array([w.joints for w in plan.waypoints[a:b + 1]], float)
    travel = float(np.max(np.sum(np.abs(np.diff(qa, axis=0)), axis=0))) if len(qa) > 1 else 0.0
    length = plan.waypoints[b].s_mm - plan.waypoints[a].s_mm
    return max(tool_speed * travel / max(length, 1e-6), 2.0)


def line_plan(p0, direction, travel_mm, speed, obstacles=None, bounds=None, step_mm=10.0, kind="lift"):
    """A straight move of travel_mm along `direction` at a constant tool speed (e.g. lifting the object straight up)."""
    p0, d = np.asarray(p0, float), np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    n = max(int(np.ceil(travel_mm / step_mm)), 1)
    waypoints, notes, ok = [], [], True
    for i in range(n + 1):
        s_mm = travel_mm * i / n
        p = p0 + d * s_mm
        if i > 0 and bounds is not None:
            lo, hi = np.array(bounds[0::2], float), np.array(bounds[1::2], float)
            if np.any(p < np.minimum(lo, p0) - 1e-6) or np.any(p > np.maximum(hi, p0) + 1e-6):
                notes.append(f"the line leaves the safe workspace after {s_mm - travel_mm / n:.0f} mm: stopping the plan there")
                ok = False
                break
        if i > 0 and obstacles is not None and obstacles.blocking(p):
            notes.append(f"the line would enter {[b.name for b in obstacles.blocking(p)]} after {s_mm - travel_mm / n:.0f} mm: stopping there")
            ok = False
            break
        waypoints.append(Waypoint(s_mm, (travel_mm - s_mm) / 10.0, speed if i < n else 0.0, p))
    return Plan(p0, d, waypoints, travel_mm / 10.0, 0.0, 1e9, waypoints[-1].s_mm, notes=notes, ok=ok, kind=kind)


def describe_line(plan, speed, what="Lift"):
    lines = [f"Plan: {what.lower()} straight {'up' if plan.direction[2] > 0.9 else 'along the line'}, {plan.travel_mm:.0f} mm at ~{speed:.0f} mm/s "
             f"({plan.method})",
             f"  from x{plan.p0[0]:.0f} y{plan.p0[1]:.0f} z{plan.p0[2]:.0f}  to  x{plan.waypoints[-1].pos[0]:.0f} "
             f"y{plan.waypoints[-1].pos[1]:.0f} z{plan.waypoints[-1].pos[2]:.0f}"]
    if plan.joint_travel_deg is not None:
        lines.append("  joint travel (deg): " + " ".join(f"J{i} {v:.1f}" for i, v in enumerate(plan.joint_travel_deg))
                     + f"   total {float(np.sum(plan.joint_travel_deg)):.1f}")
    lines += [f"  note: {n}" for n in plan.notes]
    return lines


def attach_joints_down(plan, kin, q0_deg, to_arm, a_tool, locked, max_jump_deg=15.0, max_tilt_deg=1.0, max_err_mm=1.0):
    """Joint-space solution for every waypoint with the camera axis held straight DOWN and the `locked` joints (indices)
    kept exactly at their current angles (e.g. joint 5: the camera/wrist roll never turns). Each waypoint is solved from the
    previous one, so joint movement stays minimal. -> True, or False with a note."""
    prev = np.asarray(q0_deg, float)
    fixed = {k: float(prev[k]) for k in locked}
    sols = []
    for i, w in enumerate(plan.waypoints):
        if i == 0:
            q = prev.copy()
        else:
            res = kin.face_down_ik(np.asarray(to_arm(*w.pos), float), a_tool, prev, fixed)
            if res is None or res[1] > max_tilt_deg or res[2] > max_err_mm:
                why = "no solution" if res is None else f"tilt {res[1]:.1f} deg, position error {res[2]:.1f} mm"
                plan.notes.append(f"keeping the camera down with joint(s) {sorted(locked)} locked is not possible at {w.s_mm:.0f} mm "
                                  f"({why}): using the arm's own moves instead")
                return False
            q = res[0]
        if float(np.max(np.abs(q - prev))) > max_jump_deg:
            plan.notes.append(f"joint jump of {float(np.max(np.abs(q - prev))):.0f} deg at {w.s_mm:.0f} mm: using Cartesian moves instead")
            return False
        sols.append(q)
        prev = q
    for w, q in zip(plan.waypoints, sols):
        w.joints = q
    plan.joint_travel_deg = np.sum(np.abs(np.diff(np.array(sols), axis=0)), axis=0) if len(sols) > 1 else np.zeros(len(q0_deg))
    plan.method = f"joint-space IK, camera held straight down, joint(s) {sorted(locked)} locked"
    return True
