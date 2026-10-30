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


    def to_robot(self, x, y, z):
        """Inverse of apply(): a point in the arm frame -> the robot frame."""
        c, s = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        px, py = x - self.dx, y - self.dy
        return c * px + s * py, -s * px + c * py, z - self.dz


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
        self.use_planner = False     # True: moves go through Viam's motion service, which plans around the world's obstacles
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
        self._address, self._dial = address, dial
        self._cam_robot, self._cams, self._cam_gen, self._cam_lock = None, {}, 0, threading.Lock()   # camera has its own connection

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

    def set_joints(self, values):
        """Move every joint to the given angles (degrees) in one command."""
        from viam.components.arm import JointPositions
        self._run(self.arm.move_to_joint_positions(JointPositions(values=list(values))))

    def get_kinematics(self):
        """The arm's kinematics file as bytes (Viam SVA JSON for most arms)."""
        return self._run(self.arm.get_kinematics())[1]

    def move_through_joints(self, waypoints_deg, max_vel_degs=None, max_acc_degs=None, wait=True):
        """One smooth trajectory through all the joint waypoints (Viam MoveThroughJointPositions), with optional
        joint speed / acceleration limits. wait=False returns a future."""
        from viam.components.arm import JointPositions, MoveOptions
        opts = MoveOptions(max_vel_degs_per_sec=max_vel_degs or 0.0, max_acc_degs_per_sec2=max_acc_degs or 0.0)
        positions = [JointPositions(values=[float(v) for v in q]) for q in waypoints_deg]
        fut = asyncio.run_coroutine_threadsafe(self.arm.move_through_joint_positions(positions, options=opts), self._loop)
        return fut.result(300) if wait else fut

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
        if self.allow_collisions or self.use_planner:
            self._run(self._motion_move(pose))
        else:
            self._run(self.arm.move_to_position(pose))

    async def _motion_move(self, pose):
        """Plan and execute with Viam's motion service: it knows the obstacles in the machine's frame system
        (table, walls, ...) and the arm's own links, and refuses paths that would collide."""
        from viam.proto.common import PoseInFrame
        from viam.proto.service.motion import CollisionSpecification, Constraints
        from viam.services.motion import MotionClient
        constraints = None
        if self.allow_collisions:
            allowed = [CollisionSpecification.AllowedFrameCollisions(frame1=a, frame2=b) for a, b in self.allow_collisions]
            constraints = Constraints(collision_specification=[CollisionSpecification(allows=allowed)])
        motion = MotionClient("builtin", self._robot._channel)
        await motion.move(self.arm_name, PoseInFrame(reference_frame="world", pose=pose), constraints=constraints)

    def move_xyz(self, x, y, z, wait=True):
        """Move the tool to (x, y, z) mm in the robot frame KEEPING its current orientation (so a camera mounted on
        the gripper keeps looking the same way). Clamped to the workspace like move_to. wait=False returns a future
        immediately (a newer command then takes over from the one still running, which keeps motion continuous)."""
        from viam.proto.common import Pose
        if self.workspace:
            xmin, xmax, ymin, ymax = self.workspace
            x, y = min(max(x, xmin), xmax), min(max(y, ymin), ymax)
        if self.max_z_mm is not None:
            z = min(z, self.max_z_mm)
        cur = getattr(self, "_orient", None) or self.get_pose()   # orientation is held, so read it once
        self._orient = cur
        wx, wy, wz = self.table.apply(x, y, z)
        pose = Pose(x=wx, y=wy, z=wz, o_x=cur.o_x, o_y=cur.o_y, o_z=cur.o_z, theta=cur.theta)
        coro = self._motion_move(pose) if (self.allow_collisions or self.use_planner) else self.arm.move_to_position(pose)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(120) if wait else fut

    def get_pose(self):
        """Current tool pose in the arm frame: Viam Pose (x, y, z mm; o_x, o_y, o_z; theta deg)."""
        return self._run(self.arm.get_end_position())

    def hold(self):
        """Stop whatever the arm is doing and command it to stay exactly where it is now.
        Returns the joint angles it is holding."""
        from viam.components.arm import JointPositions
        self._run(self.arm.stop())
        joints = self.get_joints()
        self._run(self.arm.move_to_joint_positions(JointPositions(values=joints)))
        return joints

    # --- cameras ---------------------------------------------------------------------
    def camera_names(self):
        return [r.name for r in self._robot.resource_names if r.subtype == "camera"]

    def _cam_client(self, real):
        """A camera client on a DEDICATED connection to the machine, so a hiccup of the video link cannot disturb arm commands."""
        from viam.components.camera import CameraClient
        from viam.robot.client import RobotClient
        with self._cam_lock:
            if self._cam_robot is None:
                self._cam_robot = self._run(RobotClient.at_address(self._address, RobotClient.Options(dial_options=self._dial)),
                                            timeout=60)
            if real not in self._cams:
                self._cams[real] = CameraClient(real, self._cam_robot._channel)
            return self._cams[real], self._cam_gen

    def _cam_reset(self, gen):
        """Drop the camera connection (once per failure, however many threads noticed) so the next call dials again."""
        with self._cam_lock:
            if gen != self._cam_gen:
                return
            old, self._cam_robot, self._cams, self._cam_gen = self._cam_robot, None, {}, self._cam_gen + 1
        if old is not None:
            try:
                self._run(old.close(), timeout=5)
            except Exception:
                pass

    def get_frame(self, camera_name):
        """One BGR frame (numpy array) from a Viam camera component. If the video connection breaks it re-dials once."""
        import cv2
        import numpy as np
        real = getattr(self, "_cam_real", {}).get(camera_name)
        if real is None:
            found = self.camera_names()
            real = camera_name
            if camera_name not in found:
                if len(found) != 1:
                    raise SystemExit(f"camera {camera_name!r} not found; machine has cameras: {found or 'none'}")
                real = found[0]
                print(f"camera {camera_name!r} not found, using the only camera: {real!r}")
            if not hasattr(self, "_cam_real"):
                self._cam_real = {}
            self._cam_real[camera_name] = real
        for attempt in (0, 1):
            cam, gen = self._cam_client(real)
            try:
                if hasattr(cam, "get_image"):
                    img = self._run(cam.get_image(mime_type="image/jpeg"), timeout=30)
                else:  # newer SDKs only have get_images -> (list of NamedImage, metadata)
                    imgs = self._run(cam.get_images(), timeout=30)[0]
                    img = next((i for i in imgs if "depth" not in str(i.mime_type).lower()), None)
                    if img is None:
                        return None
                break
            except Exception:
                self._cam_reset(gen)
                if attempt == 1:
                    raise
        data = getattr(img, "data", None)
        if data is None:  # older SDKs return a PIL image
            return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    def vision_names(self):
        return [r.name for r in self._robot.resource_names if r.subtype == "vision"]

    def detect(self, vision_name, camera_name, min_confidence=0.0):
        """Run a Viam vision service (e.g. a YOLO detector) on a camera.
        Returns (BGR frame or None, [Detection with x_min/y_min/x_max/y_max px, confidence, class_name])."""
        import cv2
        import numpy as np
        from viam.services.vision import VisionClient
        cache = getattr(self, "_vision", None)
        if cache is None:
            cache = self._vision = {}
        if vision_name not in cache:
            if vision_name not in self.vision_names():
                raise SystemExit(f"vision service {vision_name!r} not found; machine has: {self.vision_names() or 'none'}")
            cache[vision_name] = VisionClient(vision_name, self._robot._channel)
        cams = self.camera_names()
        cam = camera_name if camera_name in cams or len(cams) != 1 else cams[0]
        res = self._run(cache[vision_name].capture_all_from_camera(cam, return_image=True, return_detections=True))
        frame = None
        if res.image is not None and getattr(res.image, "data", None):
            frame = cv2.imdecode(np.frombuffer(res.image.data, np.uint8), cv2.IMREAD_COLOR)
        dets = [d for d in (res.detections or []) if d.confidence >= min_confidence]
        return frame, dets

    def camera_info(self, camera_name):
        """(supports_pcd, intrinsics dict or None) - whether the camera has depth, and its pinhole parameters."""
        from viam.components.camera import CameraClient
        real = camera_name if camera_name in self.camera_names() else (self.camera_names() or [camera_name])[0]
        props = self._run(CameraClient(real, self._robot._channel).get_properties())
        ip = props.intrinsic_parameters
        intr = None
        if ip.focal_x_px and ip.focal_y_px:
            intr = dict(fx=ip.focal_x_px, fy=ip.focal_y_px, cx=ip.center_x_px, cy=ip.center_y_px,
                        w=ip.width_px, h=ip.height_px)
        return props.supports_pcd, intr

    def objects(self, vision_name, camera_name):
        """Run a segmenter on the camera's point cloud. Returns (BGR frame or None, [(label, (x, y, z) mm)])
        with positions in the camera frame (z = straight ahead), so distance = sqrt(x^2 + y^2 + z^2)."""
        import cv2
        import numpy as np
        from viam.services.vision import VisionClient
        cache = getattr(self, "_vision", None)
        if cache is None:
            cache = self._vision = {}
        if vision_name not in cache:
            if vision_name not in self.vision_names():
                raise SystemExit(f"vision service {vision_name!r} not found; machine has: {self.vision_names() or 'none'}")
            cache[vision_name] = VisionClient(vision_name, self._robot._channel)
        cams = self.camera_names()
        cam = camera_name if camera_name in cams or len(cams) != 1 else cams[0]
        res = self._run(cache[vision_name].capture_all_from_camera(
            cam, return_image=True, return_object_point_clouds=True))
        frame = None
        if res.image is not None and getattr(res.image, "data", None):
            frame = cv2.imdecode(np.frombuffer(res.image.data, np.uint8), cv2.IMREAD_COLOR)
        found = []
        for o in res.objects or []:
            for g in o.geometries.geometries:
                found.append((g.label or "object", (g.center.x, g.center.y, g.center.z)))
        return frame, found

    def switch_info(self, name):
        """(current position, [position labels]) of a Viam switch component."""
        from viam.components.switch import SwitchClient
        sw = SwitchClient(name, self._robot._channel)
        count, labels = self._run(sw.get_number_of_positions())
        return self._run(sw.get_position()), (list(labels) or [str(i) for i in range(count)])

    def set_switch(self, name, position):
        from viam.components.switch import SwitchClient
        self._run(SwitchClient(name, self._robot._channel).set_position(position))

    # --- gripper / lifecycle --------------------------------------------------------
    def set_gripper(self, is_open):
        if self.gripper is not None:
            self._run(self.gripper.open() if is_open else self.gripper.grab())

    def grab(self, force=None, do_command=None):
        """Close the gripper. `force` goes to the driver as extra={"force": force}; `do_command` is an optional dict sent
        with the gripper's DoCommand first (for drivers that set the force that way). -> True if it reports holding."""
        if self.gripper is None:
            return False
        if do_command:
            self._run(self.gripper.do_command(do_command))
        extra = {"force": float(force)} if force is not None else None
        return bool(self._run(self.gripper.grab(extra=extra)))

    def is_holding(self):
        """True / False from the gripper's own sensing, or None if the driver cannot tell."""
        if self.gripper is None:
            return None
        try:
            res = self._run(self.gripper.is_holding_something())
            return bool(getattr(res, "is_holding_something", res))
        except Exception:
            return None

    def stop(self):
        self._run(self.arm.stop())
        if self.gripper is not None:
            self._run(self.gripper.stop())

    def close(self):
        try:
            if getattr(self, "_cam_robot", None) is not None:
                try:
                    self._run(self._cam_robot.close(), timeout=5)
                except Exception:
                    pass
            if getattr(self, "_robot", None) is not None:
                self._run(self._robot.close(), timeout=10)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
