import numpy as np

from mimic.servo import FeatureFilter, estimate_jacobian, features, servo_step


def world(J_true, s0):
    """A camera whose features respond linearly to gripper motion: s = s0 + J_true @ (pos - pos0)."""
    return lambda pos: s0 + J_true @ pos


def test_jacobian_recovered_from_probes():
    J_true = np.array([[0.0, -0.004, 0.0005], [-0.003, 0.0, 0.0002], [0.0, 0.0, 0.001]])
    s0 = np.array([0.3, -0.2, 0.2])
    cam = world(J_true, s0)
    d = (20.0, 20.0, 20.0)
    after = [cam(np.eye(3)[i] * d[i]) for i in range(3)]
    assert np.allclose(estimate_jacobian(s0, after, d), J_true)


def test_servo_centres_object_even_with_wrong_jacobian():
    J_true = np.array([[0.0, -0.004, 0.0], [-0.003, 0.0, 0.0], [0.0, 0.0, 0.001]])
    cam = world(J_true, np.array([0.35, 0.25, 0.2]))
    J_est = J_true * 1.4                       # 40% off
    pos = np.zeros(3)
    for _ in range(25):
        pos = pos + servo_step(J_est, cam(pos), (0, 0, 0.2), gain=0.5, max_step_mm=1000)
    assert np.allclose(cam(pos)[:2], 0, atol=0.01)


def test_step_limit_and_deadzone():
    J = np.array([[0.0, -0.004, 0.0], [-0.003, 0.0, 0.0], [0.0, 0.0, 0.001]])
    assert np.linalg.norm(servo_step(J, (0.5, 0.5, 0.2), (0, 0, 0.2), max_step_mm=30)) <= 30 + 1e-9
    assert np.allclose(servo_step(J, (0, 0, 0.2), (0, 0, 0.2)), 0)


def test_hold_size_moves_along_depth():
    J = np.array([[0.0, -0.004, 0.0], [-0.003, 0.0, 0.0], [0.0, 0.0, 0.001]])
    d = servo_step(J, (0, 0, 0.1), (0, 0, 0.3), features=(0, 1, 2), max_step_mm=1000)
    assert d[2] > 0 and abs(d[0]) < 1e-9 and abs(d[1]) < 1e-9


def test_features():
    assert np.allclose(features((600, 300, 680, 420), 1280, 720), (0.0, 0.0, 120 / 720))


def test_filter_smooths_noise_and_estimates_velocity():
    rng, f, errs, raw = np.random.default_rng(0), FeatureFilter(), [], []
    for i in range(60):
        t = i * 0.1
        truth = np.array([0.05 * t, 0.0, 0.2])                       # moving 0.05 frame/s to the right
        noisy = truth + rng.normal(0, 0.02, 3)
        est = f.update(noisy, t)
        if i > 20:
            errs.append(np.linalg.norm(est - truth))
            raw.append(np.linalg.norm(noisy - truth))
    assert np.mean(errs) < np.mean(raw)                              # less jitter than the raw detections
    assert abs(f.v[0] - 0.05) < 0.04                                 # sees the motion
    assert abs(f.predict(0.5)[0] - (0.05 * 5.9 + 0.025)) < 0.05      # and leads it
