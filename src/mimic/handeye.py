"""Hand-eye calibration for a wrist camera: find where the camera really is relative to the gripper.

Move the arm to several poses with a marker (ArUco) fixed on the table in view. At each pose we know the gripper's pose in the world (from the
arm) and the marker's pose in the camera (from the image). solve_handeye then gives the camera's pose in the gripper frame. If the
machine config's camera frame is off by a couple of centimetres or degrees, every world position derived from the camera is off by a lot more
(error grows with distance): this removes that error at its source.
All lengths in mm. Rotations are 3x3 matrices; quaternions are (w, x, y, z).
"""
import numpy as np


def rotvec(R):
    """3x3 rotation -> rotation vector (axis * angle, radians)."""
    q = rot_to_quat(R)                                   # w >= 0, so the angle is in [0, pi]
    v = q[1:]
    n = np.linalg.norm(v)
    return np.zeros(3) if n < 1e-12 else v / n * 2.0 * np.arctan2(n, q[0])


def solve_handeye(R_gripper2world, t_gripper2world, R_target2cam, t_target2cam, min_pair_deg=3.0):
    """Eye-in-hand: lists of per-pose gripper->world and marker->camera rotations/translations -> (R_cam2gripper, t_cam2gripper), the camera's pose
    in the gripper frame. Park & Martin: for every pair of poses A X = X B with A = G_j^-1 G_i and B = B_j B_i^-1; the rotation X comes from
    aligning the rotation vectors (Kabsch), the translation from a stacked least squares. Pure numpy (OpenCV 5 dropped calibrateHandEye)."""
    G = [(np.asarray(r, float), np.asarray(t, float).reshape(3)) for r, t in zip(R_gripper2world, t_gripper2world)]
    B = [(np.asarray(r, float), np.asarray(t, float).reshape(3)) for r, t in zip(R_target2cam, t_target2cam)]
    alphas, betas, pairs = [], [], []
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            RA = G[j][0].T @ G[i][0]
            tA = G[j][0].T @ (G[i][1] - G[j][1])
            RB = B[j][0] @ B[i][0].T
            tB = B[j][1] - RB @ B[i][1]
            if np.degrees(np.linalg.norm(rotvec(RA))) < min_pair_deg:      # nearly the same rotation: no information
                continue
            alphas.append(rotvec(RA)), betas.append(rotvec(RB)), pairs.append((RA, tA, RB, tB))
    if len(pairs) < 2:
        raise ValueError("need at least 3 poses with different gripper rotations")
    M = sum(np.outer(b, a) for a, b in zip(alphas, betas))
    U, _, Vt = np.linalg.svd(M)
    RX = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T              # the rotation mapping each beta onto its alpha
    lhs = np.vstack([RA - np.eye(3) for RA, _, _, _ in pairs])
    rhs = np.concatenate([RX @ tB - tA for _, tA, _, tB in pairs])
    tX = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return RX, tX


def rot_to_quat(R):
    """3x3 rotation -> unit quaternion (w, x, y, z), w >= 0."""
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q, float)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def quat_to_rot(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def compose(R_a, t_a, R_b, t_b):
    """Pose b expressed in frame a, then a in its parent -> b in a's parent: (R_a R_b, R_a t_b + t_a)."""
    return np.asarray(R_a) @ np.asarray(R_b), np.asarray(R_a) @ np.asarray(t_b) + np.asarray(t_a)


def rotation_error_deg(R_a, R_b):
    c = (np.trace(np.asarray(R_a).T @ np.asarray(R_b)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
