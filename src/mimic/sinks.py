"""Where commands go: SimSink records them, ViamSink drives a real (or Viam-simulated) arm."""
import math
import threading
import time


class SimSink:
    """No robot: keeps a log of (t, x, y, z, yaw, tilt, joint) rows. log[0] is the start pose."""

    def __init__(self, verbose=False):
        self.verbose, self.log, self.stats, self.error = verbose, [], {}, None
        self._t0 = time.monotonic()

    def _row(self, cmd):
        joint = math.nan if cmd.joint_deg is None else cmd.joint_deg
        return (time.monotonic() - self._t0, cmd.x, cmd.y, cmd.z, cmd.yaw_deg, cmd.tilt_deg, joint)

    def prepare(self, start):
        self.log.append((0.0, *start, 0.0, 0.0, math.nan))

    def send(self, cmd):
        self.log.append(self._row(cmd))
        if self.verbose:
            print("sim cmd x{:.0f} y{:.0f} z{:.0f} yaw{:.0f} tilt{:.0f} joint {}".format(
                cmd.x, cmd.y, cmd.z, cmd.yaw_deg, cmd.tilt_deg, cmd.joint_deg))

    def set_gripper(self, is_open):
        if self.verbose:
            print("sim gripper", "open" if is_open else "closed")

    def stop(self):
        pass

    def close(self):
        pass


class ViamSink:
    """Sends commands to a ViamRobot from a worker thread, always the newest one (slow moves never queue up)."""

    JOINT_EPS_DEG = 0.5      # skip joint commands closer than this to the last one sent
    POLL_S = 0.25

    def __init__(self, robot, joint_index=None, joint_absolute=True, joint_only=False):
        self.robot, self.joint_index, self.joint_absolute, self.joint_only = robot, joint_index, joint_absolute, joint_only
        self.stats, self.error = {"joint_moves": 0}, None
        self._joint_base = None
        self._pending, self._gripper = None, None
        self._cv = threading.Condition()
        self._alive = True
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def prepare(self, start):
        """Move to the start pose (unless joint-only) and remember the pinch joint's starting angle."""
        if not self.joint_only:
            self.robot.move_to(*start, 0.0, 0.0)
        if self.joint_index is not None:
            self._joint_base = self.robot.get_joints()[self.joint_index]

    def send(self, cmd):
        with self._cv:
            self._pending = cmd
            self._cv.notify()

    def set_gripper(self, is_open):
        with self._cv:
            self._gripper = bool(is_open)
            self._cv.notify()

    def stop(self):
        with self._cv:
            self._pending = None
        try:
            self.robot.stop()
        except Exception as e:  # keep shutting down
            self.error = self.error or f"stop failed: {e}"

    def close(self):
        self._alive = False
        with self._cv:
            self._cv.notify()
        self._worker.join(timeout=5)
        self.robot.close()

    def _run(self):
        last_poll = 0.0
        while self._alive:
            with self._cv:
                if self._pending is None and self._gripper is None:
                    self._cv.wait(self.POLL_S)
                cmd, grip, self._pending, self._gripper = self._pending, self._gripper, None, None
            try:
                if grip is not None:
                    self.robot.set_gripper(grip)
                if cmd is not None:
                    self._execute(cmd)
                if time.monotonic() - last_poll > self.POLL_S:
                    last_poll = time.monotonic()
                    self.stats["joints"] = self.robot.get_joints()
                    self.stats.pop("poll_error", None)
            except Exception as e:
                self.stats["poll_error"] = str(e)
                if cmd is not None or grip is not None:
                    self.error = f"{type(e).__name__}: {e}"
                    return

    def _execute(self, cmd):
        if not self.joint_only:
            self.robot.move_to(cmd.x, cmd.y, cmd.z, cmd.yaw_deg, cmd.tilt_deg)
        if self.joint_index is not None and cmd.joint_deg is not None:
            base = 0.0 if self.joint_absolute or self._joint_base is None else self._joint_base
            target = base + cmd.joint_deg
            sent = self.stats.get("joint_sent")
            if sent is None or abs(target - sent) >= self.JOINT_EPS_DEG:
                self.robot.set_joint(self.joint_index, target)
                self.stats["joint_sent"] = target
                self.stats["joint_moves"] += 1
