"""Blocking wrapper around a Viam arm (+ optional gripper) for plain synchronous scripts.

Poses are given in the *robot frame* (mm, z up, arm base at the origin, tool pointing down) and mapped into the
arm frame with TableToWorld. Credentials come from VIAM_API_KEY / VIAM_API_KEY_ID unless insecure=True.
"""
import asyncio
import math
import os
import sys
import threading
from dataclasses import dataclass


@dataclass
class TableToWorld:
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    yaw_deg: float = 0.0

    def apply(self, x, y, z):
        c, s = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        return c * x - s * y + self.dx, s * x + c * y + self.dy, z + self.dz


class ViamRobot:
    def __init__(self, address, arm_name="arm", gripper_name="gripper", camera_name=None, table_to_world=None,
                 workspace=None, max_z_mm=None, insecure=False, allow_collisions=None):
        try:
            from viam.components.arm import Arm
            from viam.components.gripper import Gripper
            from viam.robot.client import RobotClient
            from viam.rpc.dial import DialOptions
        except ModuleNotFoundError as e:
            raise SystemExit(f"{e}\nThis Python ({sys.executable}) lacks the Viam SDK. Use the project venv: "
                             "source scripts/setup_env.sh  (or: scripts/run.sh <script> ...)") from None

        self.table = table_to_world or TableToWorld()
        self.workspace, self.max_z_mm = workspace, max_z_mm
        self.allow_collisions = allow_collisions or []
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        if insecure:
            dial = DialOptions(insecure=True, disable_webrtc=True)
        else:
            key, key_id = os.environ.get("VIAM_API_KEY"), os.environ.get("VIAM_API_KEY_ID")
            if not key or not key_id:
                raise SystemExit("set VIAM_API_KEY and VIAM_API_KEY_ID (or pass --viam-insecure for a local server)")
            dial = DialOptions.with_api_key(key, key_id)
        self._robot = self._run(RobotClient.at_address(address, RobotClient.Options(dial_options=dial)))

        arms = [r.name for r in self._robot.resource_names if r.subtype == "arm"]
        if arm_name not in arms:
            if len(arms) != 1:
                self.close()
                raise SystemExit(f"arm {arm_name!r} not found; machine has arms: {arms or 'none'}")
            print(f"arm {arm_name!r} not found, using the only arm: {arms[0]!r}")
            arm_name = arms[0]
        self.arm_name = arm_name
        self.arm = Arm.from_robot(self._robot, arm_name)
        grippers = [r.name for r in self._robot.resource_names if r.subtype == "gripper"]
        self.gripper = Gripper.from_robot(self._robot, gripper_name) if gripper_name in grippers else None
        if self.gripper is None:
            print(f"no gripper {gripper_name!r} on this machine: running arm-only")

    def _run(self, coro, timeout=120):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    # --- joints (degrees, numbered as in the Viam app's arm panel) ---------------
    def get_joints(self):
        return list(self._run(self.arm.get_joint_positions()).values)

    def set_joint(self, index, degrees):
        """Move one joint, leaving the others where they are."""
        from viam.components.arm import JointPositions
        values = self.get_joints()
        values[index] = degrees
        self._run(self.arm.move_to_joint_positions(JointPositions(values=values)))

    def is_moving(self):
        return bool(self._run(self.arm.is_moving()))

    # --- tool pose ----------------------------------------------------------------
    def move_to(self, x, y, z, yaw_deg=0.0, tilt_deg=0.0):
        """Tool to (x, y, z) mm in the robot frame, pointing down, twisted by yaw and leaned by tilt."""
        from viam.proto.common import Pose
        if self.workspace:
            xmin, xmax, ymin, ymax = self.workspace
            x, y = min(max(x, xmin), xmax), min(max(y, ymin), ymax)
        if self.max_z_mm is not None:
            z = min(z, self.max_z_mm)
        wx, wy, wz = self.table.apply(x, y, z)
        tilt = math.radians(tilt_deg)
        pose = Pose(x=wx, y=wy, z=wz, o_x=math.sin(tilt), o_y=0.0, o_z=-math.cos(tilt),
                    theta=yaw_deg + self.table.yaw_deg)
        if self.allow_collisions:
            self._run(self._motion_move(pose))
        else:
            self._run(self.arm.move_to_position(pose))

    async def _motion_move(self, pose):
        from viam.proto.common import PoseInFrame
        from viam.proto.service.motion import CollisionSpecification, Constraints
        from viam.services.motion import MotionClient
        allowed = [CollisionSpecification.AllowedFrameCollisions(frame1=a, frame2=b) for a, b in self.allow_collisions]
        constraints = Constraints(collision_specification=[CollisionSpecification(allows=allowed)])
        motion = MotionClient.from_robot(self._robot, "builtin")
        await motion.move(self.arm.get_resource_name(self.arm_name), PoseInFrame(reference_frame="world", pose=pose),
                          constraints=constraints)

    # --- gripper / lifecycle --------------------------------------------------------
    def set_gripper(self, is_open):
        if self.gripper is not None:
            self._run(self.gripper.open() if is_open else self.gripper.grab())

    def stop(self):
        self._run(self.arm.stop())
        if self.gripper is not None:
            self._run(self.gripper.stop())

    def close(self):
        try:
            if getattr(self, "_robot", None) is not None:
                self._run(self._robot.close(), timeout=10)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
