"""Forward / inverse kinematics from the arm's own Viam kinematics file, to plan in JOINT space and keep joint
movement small.

Reads the Viam SVA JSON model (links with translations / orientations, revolute joints with axes and limits).
Inverse kinematics is damped least squares seeded at the current joint angles and pulled toward them, so every
waypoint's solution is the nearest one: no elbow flips, no wrist spins, minimum joint travel along the path.
Unsupported file formats raise Unsupported; callers check fk() against the live Viam pose before trusting it.
"""
import json
import math
import xml.etree.ElementTree as ET

import numpy as np


class Unsupported(Exception):
    pass


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


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

    @classmethod
    def from_urdf(cls, text, scale=1000.0):
        """A URDF model (revolute / continuous / fixed joints). URDF lengths are metres, so scale=1000 gives mm; the
        longest chain from the root link is the arm."""
        root = ET.fromstring(text)
        joints = []
        for j in root.iter("joint"):
            def vec(tag, attr, default):
                el = j.find(tag)
                return [float(v) for v in el.get(attr, default).split()] if el is not None else [float(v) for v in default.split()]
            lim = j.find("limit")
            kind = j.get("type", "fixed")
            lower = math.degrees(float(lim.get("lower", -2 * math.pi))) if lim is not None and lim.get("lower") else -360.0
            upper = math.degrees(float(lim.get("upper", 2 * math.pi))) if lim is not None and lim.get("upper") else 360.0
            if kind == "continuous":
                lower, upper = -360.0, 360.0
            xyz, rpy = vec("origin", "xyz", "0 0 0"), vec("origin", "rpy", "0 0 0")
            R = rot_z(rpy[2]) @ rot_y(rpy[1]) @ rot_x(rpy[0])
            joints.append({"type": kind, "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
                           "p": np.array(xyz, float) * scale, "R": R, "axis": vec("axis", "xyz", "1 0 0"), "lim": [lower, upper]})
        if not joints:
            raise Unsupported("URDF has no joints")
        children = {j["child"] for j in joints}
        roots = {j["parent"] for j in joints} - children
        if len(roots) != 1:
            raise Unsupported("URDF has no single root link")

        def chain(link):
            best = []
            for j in joints:
                if j["parent"] == link:
                    c = [j] + chain(j["child"])
                    if len(c) > len(best):
                        best = c
            return best
        path = chain(roots.pop())
        elements, limits = [], []
        for j in path:
            elements.append(("link", j["p"], j["R"]))
            if j["type"] in ("revolute", "continuous"):
                elements.append(("joint", np.array(j["axis"], float)))
                limits.append(j["lim"])
            elif j["type"] != "fixed":
                raise Unsupported(f"URDF joint type {j['type']!r}")
        if not limits:
            raise Unsupported("URDF chain has no revolute joints")
        return cls(elements, limits, "urdf")

    @classmethod
    def from_bytes(cls, data, scale=1000.0):
        """Whatever the arm returned: Viam SVA JSON or URDF XML."""
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        head = data.lstrip()[:1]
        if head == "{":
            return cls.from_json(data)
        if head == "<":
            return cls.from_urdf(data, scale)
        raise Unsupported(f"unknown kinematics format, starts with {data[:24]!r}")

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

    def link_points(self, q_deg):
        """3D positions (arm base frame, mm) of every link origin along the arm, ending at the tool point: the arm's
        skeleton, used to keep its links clear of obstacles."""
        p, R, i, pts = np.zeros(3), np.eye(3), 0, []
        for el in self.elements:
            if el[0] == "link":
                p = p + R @ el[1]
                R = R @ el[2]
                pts.append(p.copy())
            else:
                R = R @ rot_axis(el[1], math.radians(q_deg[i]))
                i += 1
        return pts

    def look_at_ik(self, p_obj, c0_mm, a_tool, q_seed_deg, fixed, pull=0.02, damping=0.05, iters=200, tol_mm=0.5,
                   level=False, down=False):
        """Joint angles that hold some joints at given values (`fixed`: {joint index: degrees}) while the camera still
        looks at the object: the camera sits c0_mm in front of p_obj along its axis (a_tool = that axis in the tool
        frame). The free joints solve it, seeded at and pulled toward the current angles (minimum joint movement).
        -> degrees, or None if no such configuration exists."""
        q0 = np.radians(np.asarray(q_seed_deg, float))
        q = q0.copy()
        for k, v in fixed.items():
            q[k] = math.radians(v)
        free = [k for k in range(self.n) if k not in fixed]
        lo, hi = np.radians(self.limits[:, 0]), np.radians(self.limits[:, 1])
        a_tool = np.asarray(a_tool, float)

        def residual(qv):
            p, R = self.fk(np.degrees(qv))
            axis = R @ a_tool
            f = p + c0_mm * axis - p_obj
            if level:                                                     # level: the camera axis has no vertical part
                f = np.append(f, c0_mm * axis[2])
            if down:                                                      # down: the camera axis is vertical (checked below it points down)
                f = np.append(f, [c0_mm * axis[0], c0_mm * axis[1]])
            return f
        for _ in range(iters):
            f = residual(q)
            if np.linalg.norm(f) < tol_mm:
                return np.degrees(q)
            J = np.zeros((len(f), len(free)))
            for col, k in enumerate(free):
                dq = q.copy()
                dq[k] += 1e-5
                J[:, col] = (residual(dq) - f) / 1e-5
            A = J.T @ J + (damping ** 2 * 1e4 + pull) * np.eye(len(free))
            step = np.linalg.solve(A, J.T @ (-f) - pull * (q[free] - q0[free]))
            q[free] = np.clip(q[free] + np.clip(step, -0.2, 0.2), lo[free], hi[free])
        if np.linalg.norm(residual(q)) >= 2.0:
            return None
        if down and (self.fk(np.degrees(q))[1] @ a_tool)[2] > -0.98:     # converged to straight UP: not what was asked
            return None
        return np.degrees(q)

    def face_down_ik(self, p_target, a_tool, q_seed_deg, fixed, w_axis=5000.0, iters=300):
        """Joint angles with the `fixed` joints held (e.g. the wrist at a camera-down posture) and the camera axis pointing
        straight DOWN, the tool staying as close to p_target as that allows (the free joints cannot do both exactly).
        -> (q_deg, axis tilt from vertical in degrees, position error in mm) or None."""
        q0 = np.radians(np.asarray(q_seed_deg, float))
        q = q0.copy()
        for k, v in fixed.items():
            q[k] = math.radians(v)
        free = [k for k in range(self.n) if k not in fixed]
        lo, hi = np.radians(self.limits[:, 0]), np.radians(self.limits[:, 1])
        a_tool = np.asarray(a_tool, float)

        def residual(qv):
            p, R = self.fk(np.degrees(qv))
            axis = R @ a_tool
            return np.concatenate([p - p_target, w_axis * axis[:2]])
        lam = 1.0
        for _ in range(iters):
            f = residual(q)
            J = np.zeros((5, len(free)))
            for col, k in enumerate(free):
                dq = q.copy()
                dq[k] += 1e-5
                J[:, col] = (residual(dq) - f) / 1e-5
            step = np.linalg.solve(J.T @ J + lam * np.eye(len(free)), -J.T @ f)
            q_try = q.copy()
            q_try[free] = np.clip(q[free] + np.clip(step, -0.2, 0.2), lo[free], hi[free])
            if np.linalg.norm(residual(q_try)) < np.linalg.norm(f):
                q, lam = q_try, max(lam * 0.5, 1e-6)
            else:
                lam *= 4.0
            if np.linalg.norm(step) < 1e-9:
                break
        p, R = self.fk(np.degrees(q))
        axis = R @ a_tool
        if axis[2] > -0.9:
            return None
        return np.degrees(q), float(np.degrees(np.arccos(np.clip(-axis[2], -1, 1)))), float(np.linalg.norm(p - p_target))

    def face_down_above_ik(self, xy_target, a_tool, q_seed_deg, fixed, w_axis=3000.0, iters=300):
        """Joints (the `fixed` ones held, e.g. wrist joints 4 and 5 at a camera-down posture) with the camera axis straight
        DOWN and the camera exactly above xy_target. The camera's HEIGHT is not asked for: with the wrist fixed it is whatever
        the arm's geometry gives. -> (q_deg, tilt from vertical in deg, horizontal error mm, camera height mm) or None."""
        q = np.radians(np.asarray(q_seed_deg, float))
        for k, v in fixed.items():
            q[k] = math.radians(v)
        free = [k for k in range(self.n) if k not in fixed]
        lo, hi = np.radians(self.limits[:, 0]), np.radians(self.limits[:, 1])
        a_tool = np.asarray(a_tool, float)

        def residual(qv):
            p, R = self.fk(np.degrees(qv))
            return np.concatenate([p[:2] - xy_target, w_axis * (R @ a_tool)[:2]])
        lam = 1.0
        for _ in range(iters):
            f = residual(q)
            J = np.zeros((4, len(free)))
            for col, k in enumerate(free):
                dq = q.copy()
                dq[k] += 1e-5
                J[:, col] = (residual(dq) - f) / 1e-5
            step = np.linalg.solve(J.T @ J + lam * np.eye(len(free)), -J.T @ f)
            q_try = q.copy()
            q_try[free] = np.clip(q[free] + np.clip(step, -0.2, 0.2), lo[free], hi[free])
            if np.linalg.norm(residual(q_try)) < np.linalg.norm(f):
                q, lam = q_try, max(lam * 0.5, 1e-6)
            else:
                lam *= 4.0
            if np.linalg.norm(step) < 1e-9:
                break
        p, R = self.fk(np.degrees(q))
        axis = R @ a_tool
        if axis[2] > -0.9:
            return None
        return (np.degrees(q), float(np.degrees(np.arccos(np.clip(-axis[2], -1, 1)))),
                float(np.linalg.norm(p[:2] - xy_target)), float(p[2]))

    def face_down_above_ik_multi(self, xy_target, a_tool, q_seed_deg, fixed, tries=30, spread_deg=60.0, max_tilt_deg=1.0,
                                 max_xy_mm=2.0, z_range=(-1e9, 1e9), prefer=None):
        """face_down_above_ik from the current angles and perturbed seeds; of the solutions that point down, sit above the
        target and have an acceptable camera height, the one that moves the joints least."""
        q_seed = np.asarray(q_seed_deg, float)
        rng = np.random.default_rng(13)
        free = [k for k in range(self.n) if k not in fixed]
        found = []
        for i in range(tries + 1):
            seed = q_seed.copy()
            if i:
                for k in free:
                    seed[k] = np.clip(seed[k] + rng.uniform(-spread_deg, spread_deg), self.limits[k, 0], self.limits[k, 1])
            res = self.face_down_above_ik(xy_target, a_tool, seed, fixed)
            if res is not None and res[1] <= max_tilt_deg and res[2] <= max_xy_mm and z_range[0] <= res[3] <= z_range[1]:
                found.append(res)
        if not found:
            return None
        prefer = prefer or {}
        return min(found, key=lambda r: float(np.sum(np.abs(r[0] - q_seed)))
                   + 8.0 * sum(abs(r[0][k] - v) for k, v in prefer.items()))      # nearest to the wished joint values first

    def face_down_ik_multi(self, p_target, a_tool, q_seed_deg, fixed, tries=30, spread_deg=60.0, max_tilt_deg=1.0):
        """face_down_ik from the current angles and from perturbed seeds: of the solutions that really point down, the one
        that leaves the tool closest to p_target and moves the joints least."""
        q_seed = np.asarray(q_seed_deg, float)
        rng = np.random.default_rng(11)
        free = [k for k in range(self.n) if k not in fixed]
        found = []
        for i in range(tries + 1):
            seed = q_seed.copy()
            if i:
                for k in free:
                    seed[k] = np.clip(seed[k] + rng.uniform(-spread_deg, spread_deg), self.limits[k, 0], self.limits[k, 1])
            res = self.face_down_ik(p_target, a_tool, seed, fixed)
            if res is not None and res[1] <= max_tilt_deg:
                found.append(res)
        if not found:
            return None
        return min(found, key=lambda r: r[2] + 0.5 * float(np.sum(np.abs(r[0] - q_seed))))

    def look_at_ik_multi(self, p_obj, c0_mm, a_tool, q_seed_deg, fixed, level=False, tries=40, spread_deg=45.0, down=False):
        """look_at_ik from the current angles first, then from perturbed seeds; of all the solutions found, the one that moves
        the joints least. The local solver can stall for a wrist that must turn a long way; other seeds usually find the
        solution right next to it."""
        q_seed = np.asarray(q_seed_deg, float)
        found = []
        first = self.look_at_ik(p_obj, c0_mm, a_tool, q_seed, fixed, level=level, down=down)
        if first is not None:
            found.append(first)
        rng = np.random.default_rng(7)
        free = [k for k in range(self.n) if k not in fixed]
        for _ in range(0 if first is not None else tries):
            seed = q_seed.copy()
            for k in free:
                seed[k] = np.clip(seed[k] + rng.uniform(-spread_deg, spread_deg), self.limits[k, 0], self.limits[k, 1])
            q = self.look_at_ik(p_obj, c0_mm, a_tool, seed, fixed, level=level, down=down)
            if q is not None:
                found.append(q)
        if not found:
            return None
        return min(found, key=lambda q: float(np.sum(np.abs(q - q_seed))))

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


def nearest_equivalent(target_deg, current_deg, limits=(-360.0, 360.0)):
    """The angle that points the joint the same way as target_deg (any multiple of 360 apart) lying inside the joint's
    limits and closest to where the joint is now: -360 and 0 are the same orientation, so from -270 the move is to
    -360 (90 deg), from +10 it is to 0 (10 deg), never a full spin."""
    lo, hi = limits
    base = target_deg - 360.0 * round((target_deg - current_deg) / 360.0)
    cands = [base + 360.0 * k for k in (-1, 0, 1) if lo - 1e-6 <= base + 360.0 * k <= hi + 1e-6]
    if not cands:
        return float(min(max(target_deg, lo), hi))
    return float(min(cands, key=lambda c: abs(c - current_deg)))


def smooth_path(q0, q1, n=14):
    """Joint waypoints from q0 to q1 with a quintic ease: zero speed and acceleration at both ends, no jerk."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    t = np.linspace(0.0, 1.0, n + 1)[1:]
    e = 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5
    return [q0 + (q1 - q0) * x for x in e]


def load_validated(robot, tol_mm=5.0, tol_rot=0.1, dump_dir=None):
    """Kinematics from the live arm, trusted only if its forward kinematics reproduce the pose Viam reports for the
    current joint angles (SVA JSON or URDF, trying metre and millimetre units). The raw file is saved to dump_dir
    (when given) so it can be inspected. -> (Kinematics or None, a note saying what happened)."""
    try:
        data = robot.get_kinematics()
    except Exception as e:
        return None, f"could not read the kinematics ({type(e).__name__}: {e})"
    if dump_dir:
        try:
            import os
            os.makedirs(dump_dir, exist_ok=True)
            raw = data if isinstance(data, bytes) else str(data).encode()
            ext = "json" if raw.lstrip()[:1] == b"{" else "xml" if raw.lstrip()[:1] == b"<" else "bin"
            with open(os.path.join(dump_dir, f"arm_kinematics.{ext}"), "wb") as f:
                f.write(raw)
        except OSError:
            pass
    q = robot.get_joints()
    pose = robot.get_pose()
    target_p = np.array([pose.x, pose.y, pose.z])
    target_R = ov_to_R(pose.o_x, pose.o_y, pose.o_z, pose.theta)
    best, why = None, ""
    for scale in (1000.0, 1.0):
        try:
            kin = Kinematics.from_bytes(data, scale)
        except Unsupported as e:
            return None, f"kinematics file not supported ({e})"
        except Exception as e:
            return None, f"could not parse the kinematics ({type(e).__name__}: {e})"
        if len(q) != kin.n:
            return None, f"the kinematics has {kin.n} joints but the arm reports {len(q)}"
        p, R = kin.fk(q)
        err = float(np.linalg.norm(p - target_p))
        rot = float(np.linalg.norm(R - target_R))
        if err <= tol_mm and rot <= tol_rot:
            best = (kin, err)
            break
        why = f"the model disagrees with the arm's reported pose ({err:.1f} mm, orientation {rot:.2f})"
        if kin.name != "urdf":
            break                                               # SVA is always in mm: no second scale to try
    if best is None:
        return None, f"kinematics rejected: {why}"
    return best[0], f"kinematics OK ({best[0].name or 'arm'}, forward-kinematics error {best[1]:.2f} mm)"
