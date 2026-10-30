"""Forward / inverse kinematics from the arm's own Viam kinematics file, to plan in JOINT space and keep joint
movement small.

Reads the Viam SVA JSON model (links with translations / orientations, revolute joints with axes and limits).
Inverse kinematics is damped least squares seeded at the current joint angles and pulled toward them, so every
waypoint's solution is the nearest one: no elbow flips, no wrist spins, minimum joint travel along the path.
Unsupported file formats raise Unsupported; callers check fk() against the live Viam pose before trusting it.
"""
import json
import math

import numpy as np


class Unsupported(Exception):
    pass


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def rot_axis(axis, a):
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * (k @ k)


def ov_to_R(ox, oy, oz, theta_deg):
    """Viam orientation vector (a direction plus a twist about it, degrees) -> rotation matrix."""
    n = math.sqrt(ox * ox + oy * oy + oz * oz) or 1.0
    ox, oy, oz = ox / n, oy / n, oz / n
    lat = math.acos(max(-1.0, min(1.0, oz)))
    lon = math.atan2(oy, ox) if 1 - abs(oz) > 1e-8 else 0.0
    return rot_z(lon) @ rot_y(lat) @ rot_z(math.radians(theta_deg))


def quat_to_R(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _orientation(spec):
    if not spec:
        return np.eye(3)
    kind, v = spec.get("type", "ov_degrees"), spec.get("value", {})
    if kind in ("ov_degrees", "orientation_vector_degrees"):
        return ov_to_R(v.get("x", 0), v.get("y", 0), v.get("z", 1), v.get("th", 0))
    if kind in ("ov_radians", "orientation_vector_radians"):
        return ov_to_R(v.get("x", 0), v.get("y", 0), v.get("z", 1), math.degrees(v.get("th", 0)))
    if kind == "quaternion":
        return quat_to_R(v.get("real", 1), v.get("imag", 0), v.get("jmag", 0), v.get("kmag", 0))
    if kind in ("axis_angles", "r4"):
        return rot_axis([v.get("x", 0), v.get("y", 0), v.get("z", 1)], math.radians(v.get("th", 0)))
    raise Unsupported(f"orientation type {kind!r}")


class Kinematics:
    def __init__(self, elements, limits_deg, name=""):
        self.elements, self.limits, self.name = elements, np.array(limits_deg, float), name   # ("link", p, R) / ("joint", axis)
        self.n = len(limits_deg)

    @classmethod
    def from_json(cls, data):
        model = json.loads(data if isinstance(data, (str, bytes)) else json.dumps(data))
        if model.get("kinematic_param_type", "SVA").upper() != "SVA" or "links" not in model or "joints" not in model:
            raise Unsupported("not an SVA kinematics file")
        links = {l["id"]: l for l in model["links"]}
        joints = {j["id"]: j for j in model["joints"]}
        by_parent_link = {j["parent"]: j for j in model["joints"]}
        by_parent_joint = {l["parent"]: l for l in model["links"] if l["parent"] != "world"}
        roots = [l for l in model["links"] if l["parent"] == "world"]
        if len(roots) != 1:
            raise Unsupported("expected one link attached to the world")
        elements, limits, link = [], [], roots[0]
        while True:
            t = link.get("translation") or {}
            elements.append(("link", np.array([t.get("x", 0), t.get("y", 0), t.get("z", 0)], float), _orientation(link.get("orientation"))))
            j = by_parent_link.get(link["id"])
            if j is None:
                break
            if j.get("type", "revolute") != "revolute":
                raise Unsupported(f"joint type {j.get('type')!r}")
            ax = j.get("axis") or {}
            elements.append(("joint", np.array([ax.get("x", 0), ax.get("y", 0), ax.get("z", 1)], float)))
            limits.append([j.get("min", -360.0), j.get("max", 360.0)])
            nxt = by_parent_joint.get(j["id"])
            if nxt is None:
                raise Unsupported("a joint with no child link")
            link = nxt
        if not limits:
            raise Unsupported("no joints")
        return cls(elements, limits, model.get("name", ""))

    def fk(self, q_deg):
        """-> (tool position mm, tool rotation matrix) in the arm's base frame."""
        p, R, i = np.zeros(3), np.eye(3), 0
        for el in self.elements:
            if el[0] == "link":
                p = p + R @ el[1]
                R = R @ el[2]
            else:
                R = R @ rot_axis(el[1], math.radians(q_deg[i]))
                i += 1
        return p, R

    def ik(self, p_target, R_target, q_seed_deg, pull=0.02, damping=0.05, iters=120, tol_mm=0.05):
        """Nearest joint solution to the seed (degrees) for the given tool pose, or None. `pull` ties the solution
        to the seed (smaller joint movement), `damping` keeps steps tame near singularities."""
        q0 = np.radians(np.asarray(q_seed_deg, float))
        q = q0.copy()
        lo, hi = np.radians(self.limits[:, 0]), np.radians(self.limits[:, 1])
        scale = 100.0                                          # mm per radian, to weigh orientation against position
        for _ in range(iters):
            p, R = self.fk(np.degrees(q))
            e_rot = 0.5 * (np.cross(R[:, 0], R_target[:, 0]) + np.cross(R[:, 1], R_target[:, 1]) + np.cross(R[:, 2], R_target[:, 2]))
            e = np.concatenate([p_target - p, scale * e_rot])
            if np.linalg.norm(e[:3]) < tol_mm and np.linalg.norm(e_rot) < 2e-4:
                return np.degrees(q)
            J = np.zeros((6, self.n))
            for k in range(self.n):
                dq = q.copy()
                dq[k] += 1e-5
                p2, R2 = self.fk(np.degrees(dq))
                r2 = 0.5 * (np.cross(R[:, 0], R2[:, 0]) + np.cross(R[:, 1], R2[:, 1]) + np.cross(R[:, 2], R2[:, 2]))
                J[:, k] = np.concatenate([(p2 - p) / 1e-5, scale * r2 / 1e-5])
            A = J.T @ J + (damping ** 2 * scale ** 2 * 1e-2 + pull) * np.eye(self.n)
            step = np.linalg.solve(A, J.T @ e - pull * (q - q0))
            q = np.clip(q + np.clip(step, -0.2, 0.2), lo, hi)
        p, R = self.fk(np.degrees(q))
        e_rot = 0.5 * (np.cross(R[:, 0], R_target[:, 0]) + np.cross(R[:, 1], R_target[:, 1]) + np.cross(R[:, 2], R_target[:, 2]))
        return np.degrees(q) if np.linalg.norm(p_target - p) < 0.5 and np.linalg.norm(e_rot) < 0.02 else None


def load_validated(robot, tol_mm=5.0, tol_rot=0.1):
    """Kinematics from the live arm, trusted only if its forward kinematics reproduce the pose Viam reports for the
    current joint angles. -> (Kinematics or None, a note saying what happened)."""
    try:
        data = robot.get_kinematics()
        kin = Kinematics.from_json(data)
    except Unsupported as e:
        return None, f"kinematics file not supported ({e})"
    except Exception as e:
        return None, f"could not read the kinematics ({type(e).__name__}: {e})"
    q = robot.get_joints()
    if len(q) != kin.n:
        return None, f"the kinematics has {kin.n} joints but the arm reports {len(q)}"
    p, R = kin.fk(q)
    pose = robot.get_pose()
    err = float(np.linalg.norm(p - np.array([pose.x, pose.y, pose.z])))
    rot = float(np.linalg.norm(R - ov_to_R(pose.o_x, pose.o_y, pose.o_z, pose.theta)))
    if err > tol_mm or rot > tol_rot:
        return None, f"the kinematics model disagrees with the arm's reported pose ({err:.1f} mm, orientation {rot:.2f})"
    return kin, f"kinematics OK ({kin.name or 'arm'}, forward-kinematics error {err:.2f} mm)"
