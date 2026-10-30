"""Image-based visual servoing for a camera mounted on the gripper.

Features s = (u, v, h): the object's centre offset from the image centre (fractions of the frame, +u right, +v down)
and its height as a fraction of the frame. Moving the gripper by d mm changes them by J @ d. J (3x3, per mm) is
LEARNED by small probe moves, so it absorbs the camera mounting, orientation and scale - nothing is assumed.
Each step: d = gain * J+ (s_target - s), a damped least-squares move, limited in size.
"""
import numpy as np


def estimate_jacobian(s0, s_after, deltas_mm):
    """s0: features before; s_after: three feature vectors after moving +deltas_mm[i] along robot x, y, z."""
    J = np.zeros((3, 3))
    for i in range(3):
        J[:, i] = (np.asarray(s_after[i], float) - np.asarray(s0, float)) / deltas_mm[i]
    return J


def servo_step(J, s, s_target, features=(0, 1), gain=0.5, max_step_mm=40.0, damping=0.1):
    """-> move (dx, dy, dz) in mm along the robot axes that brings the chosen features toward the target."""
    rows = list(features)
    Jf = np.asarray(J, float)[rows]
    e = (np.asarray(s_target, float) - np.asarray(s, float))[rows]
    lam = damping * max(np.linalg.svd(Jf, compute_uv=False)[0], 1e-9)
    d = gain * Jf.T @ np.linalg.solve(Jf @ Jf.T + lam ** 2 * np.eye(len(rows)), e)
    n = float(np.linalg.norm(d))
    return d * (max_step_mm / n) if n > max_step_mm else d


def features(box, frame_w, frame_h):
    """(x1, y1, x2, y2) pixels -> (u, v, h)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return np.array([cx / frame_w - 0.5, cy / frame_h - 0.5, (box[3] - box[1]) / frame_h])


class FeatureFilter:
    """Alpha-beta tracker on the features: removes detection jitter and estimates how fast the cup moves in the
    image (per second), so the arm can aim slightly ahead of it instead of chasing."""

    def __init__(self, alpha=0.6, beta=0.04, max_rate=1.0):
        self.alpha, self.beta, self.max_rate = alpha, beta, max_rate
        self.reset()

    def reset(self):
        self.x = self.v = self.t = None

    def update(self, s, t):
        s = np.asarray(s, float)
        if self.x is None:
            self.x, self.v, self.t = s.copy(), np.zeros_like(s), t
            return self.x
        dt = max(t - self.t, 1e-3)
        pred = self.x + self.v * dt
        r = s - pred
        self.x = pred + self.alpha * r
        self.v = np.clip(self.v + self.beta * r / dt, -self.max_rate, self.max_rate)
        self.t = t
        return self.x

    def predict(self, ahead_s):
        """Where the features will be ahead_s seconds after the last update."""
        return self.x + self.v * ahead_s


def approach_factor(err_uv, deadband):
    """0..1 share of the full approach speed to use for a given centring error (max |u|, |v|, fractions of the frame).
    Full speed while the error is under 2 x deadband, none from 4 x deadband up, a straight line in between. Replaces an
    on/off switch at 2 x deadband that flipped every few steps when the error hovered around it (visible jitter)."""
    e = float(np.max(np.abs(np.asarray(err_uv, float))))
    return float(np.clip((4.0 * deadband - e) / (2.0 * deadband), 0.0, 1.0))
