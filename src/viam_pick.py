"""Pick the nearest object: local YOLO (GPU) + Viam's camera, frame-system and motion services, re-planned every step.

    camera.get_images()   colour + depth in one call
    local YOLO + Lock     finds and keeps tracking the object (nearest first, by depth); size = pixels x depth / focal length
    pinhole + depth       pixel -> 3D point in the camera frame (the object's centre = front surface + half its width)
    robot.transform_pose  camera frame -> WORLD frame            (Viam frame system)
    motion.move           SHORT collision-aware planned moves    (Viam builtin motion service; it knows the table / walls)
    gripper.do_command    close to the measured width            (position control: this gripper has no force limit)

DYNAMIC: nothing is planned once. Every cycle it looks again and plans only the next short step toward where the object is NOW,
so a person moving the bottle just changes the next waypoint. It pauses if a person is close to the gripper, follows the same
object if it moves (re-targets if it jumps), and before closing it re-checks the object is still lined up with the gripper.

Stages: A approach a pre-grasp pose (current orientation, so no wrist spin) -> B flatten the gripper with ONE wrist joint in
guarded 10-degree steps -> C advance along the gripper axis, correcting sideways -> close -> lift.

Dry run by default (--sink sim): looks repeatedly and prints the target and the next waypoint (move the bottle by hand to watch it
follow); nothing moves.
    scripts/run.sh src/viam_pick.py --viam-address sfhack16-main.ty1cywh7tq.local.viam.cloud:8080 --object bottle
Real arm (asks for a typed 'yes' before each stage; Ctrl+C stops):
    scripts/run.sh src/viam_pick.py --viam-address ... --object bottle --sink viam
"""
import argparse
import asyncio
import os
import sys
import time
import types
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from viam.components.arm import Arm, JointPositions  # noqa: E402
from viam.components.camera import Camera  # noqa: E402
from viam.components.gripper import Gripper  # noqa: E402
from mimic.runlog import RunLog, push_mongo  # noqa: E402
from viam.proto.common import Geometry, GeometriesInFrame, Pose, PoseInFrame, RectangularPrism, Vector3, WorldState  # noqa: E402
from viam.robot.client import RobotClient  # noqa: E402
from viam.rpc.dial import DialOptions  # noqa: E402
from viam.proto.service.motion import CollisionSpecification, Constraints  # noqa: E402
from viam.services.motion import MotionClient  # noqa: E402

from local_detect import Lock, detect, load_detector, object_candidates  # noqa: E402
from mimic.ui import UIServer  # noqa: E402
from mimic.voxelmap import VoxelMap, depth_to_points  # noqa: E402
from mimic.size import box_depth_mm, height_from_depth, parse_viam_depth, width_from_depth  # noqa: E402


def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


async def connect(address):
    key, key_id = os.environ.get("VIAM_API_KEY"), os.environ.get("VIAM_API_KEY_ID")
    if not key or not key_id:
        raise SystemExit("set VIAM_API_KEY and VIAM_API_KEY_ID")
    dial = DialOptions.with_api_key(key, key_id)
    if ".local.viam.cloud" in address:                  # direct LAN connection (no relay): several times the frame rate
        dial.disable_webrtc = True
    return await RobotClient.at_address(address, RobotClient.Options(dial_options=dial))


class ObjectMemory:
    """Everything known about the ORIGINAL object, kept for the whole run and saved to a file: where it is (world frame, weighted
    toward the closest and so most reliable sightings; replaced outright if it is moved), how big it is, when it was last seen."""

    def __init__(self, path, label, jump_mm=150.0):
        self.path, self.label, self.jump_mm = path, label, jump_mm
        self.world = self.first_world = None
        self.weight, self.width, self.height, self.sightings, self.last_seen = 0.0, None, None, 0, None

    def update(self, t, width=None, height=None):
        """Fold in one sighting (a look() result)."""
        w_new = 1.0 / max(t["depth_mm"], 100.0) ** 2         # closer sightings count for more
        if self.world is None:
            self.world, self.weight, self.first_world = t["world"].copy(), w_new, t["world"].copy()
        elif np.linalg.norm(t["world"] - self.world) > self.jump_mm:   # it was moved (or the estimate jumped): trust the new place
            self.world, self.weight = t["world"].copy(), w_new
        else:
            self.world = (self.weight * self.world + w_new * t["world"]) / (self.weight + w_new)
            self.weight += w_new
        self.width, self.height = width or self.width, height or self.height
        self.sightings, self.last_seen = self.sightings + 1, time.time()
        if self.sightings % 5 == 1:
            self.save()

    def save(self):
        import json
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as f:
                json.dump({"label": self.label, "world_mm": None if self.world is None else [round(float(v), 1) for v in self.world],
                           "first_seen_world_mm": None if self.first_world is None else [round(float(v), 1) for v in self.first_world],
                           "width_mm": self.width, "height_mm": self.height, "sightings": self.sightings,
                           "last_seen": self.last_seen}, f, indent=2)
        except OSError:
            pass

    def load(self):
        import json
        try:
            d = json.load(open(self.path))
        except (OSError, ValueError):
            return False
        if d.get("label") != self.label or not d.get("world_mm"):
            return False
        self.world = np.array(d["world_mm"], float)
        self.first_world = np.array(d.get("first_seen_world_mm") or d["world_mm"], float)
        self.width, self.height, self.weight, self.sightings = d.get("width_mm"), d.get("height_mm"), 1e-6, d.get("sightings", 0)
        return True


class GraspFailed(Exception):
    """The grasp did not work (nothing in the gripper / object moved away): the whole attempt can be retried from the start pose."""


class Perceiver:
    """One call -> where the tracked object is right now (world frame), how big it is, and whether a person is close."""

    def __init__(self, robot, camera_name, gripper_name, cam, intr, model, dargs, label, person_mm, aim_frac=0.5, memory=None):
        self.memory = memory
        self.color, self.cands, self.last_dmap = None, [], None
        self.aim_frac = aim_frac     # where along the object's height to aim: 0.5 = its middle, 0.85 = near the top (the neck of a bottle)
        self.robot, self.camera_name, self.gripper_name, self.cam, self.intr = robot, camera_name, gripper_name, cam, intr
        self.model, self.dargs, self.label, self.person_mm = model, dargs, label, person_mm
        self.lock, self.widths, self.heights = Lock(max_jump_frac=0.5, lost_after=10 ** 6), [], []
        self.expected = None          # world position of the object we are after: re-locking picks the candidate NEAREST TO IT
        self.reject_mm = 300.0        # while tracking: a sudden jump of more than this is a DIFFERENT object (e.g. a can the detector also calls a
                                      # cup), not the one we are after: ignore it and drop the lock
        self.max_sight_mm = 1000.0    # a sighting farther than this from the base axis (or far below the table / above the arm) is a false detection
        self.confirm_n = 2            # the FIRST sighting must be seen this many looks in a row, within 150 mm, before it is trusted
        self.trim = np.zeros(3)       # constant aiming correction in the gripper frame (--aim-trim-mm)
        self.candidate, self.person_hits, self.person_info, self.rejected = None, 0, "", 0
        self.gate_mm = 200.0          # ...and ignores anything further than this from it (a different bottle: ~30 cm apart; the world
                                      # estimate itself shifts ~12 cm with the viewpoint)

    def reset(self):
        self.lock.unlock()                                # re-lock, but KEEP what we know about the object (width, height, position)

    async def _world(self, box, z, w, h):
        """World position of a box's object centre (front surface + half its width), from its box and depth."""
        sx, sy = w / self.intr["w"], h / self.intr["h"]
        fx, fy, cx, cy = self.intr["fx"] * sx, self.intr["fy"] * sy, self.intr["cx"] * sx, self.intr["cy"] * sy
        zc = z + (box[2] - box[0]) * z / fx / 2
        pose = Pose(x=((box[0] + box[2]) / 2 - cx) * zc / fx, y=((box[1] + box[3]) / 2 - cy) * zc / fy, z=zc)
        wp = (await self.robot.transform_pose(PoseInFrame(reference_frame=self.camera_name, pose=pose), "world")).pose
        return np.array([wp.x, wp.y, wp.z])

    async def look(self):
        imgs = (await self.cam.get_images())[0]
        color = depth = None
        for im in imgs:
            if "dep" in str(im.mime_type).lower():
                depth = parse_viam_depth(im.data)
            else:
                color = cv2.imdecode(np.frombuffer(im.data, np.uint8), cv2.IMREAD_COLOR)
        if color is None or depth is None:
            return None
        h, w = color.shape[:2]
        dmap = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        self.last_dmap = dmap                                 # for the voxel map
        dets = await asyncio.to_thread(detect, self.model, color, self.dargs)          # YOLO on the GPU
        near_person, self.person_info = False, ""
        for d in dets:
            if d[5] == "person":
                pd = box_depth_mm(dmap, d[:4])
                if pd and pd < self.person_mm:
                    near_person = True
                    self.person_info = f"person {pd / 10:.0f} cm away, box {int(d[2] - d[0])}x{int(d[3] - d[1])} px, conf {d[4]:.2f}"
        self.person_hits = self.person_hits + 1 if near_person else 0
        person_close = self.person_hits >= 2                 # a one-frame flicker does not pause the arm
        cands = object_candidates(dets, self.label)
        self.color, self.cands = color, list(cands)           # for the web page's live view
        if not self.lock.locked:                         # (re)lock: the one nearest where it was; the first time, the nearest by depth
            scored = []
            for c in cands:
                zc_ = box_depth_mm(dmap, c[:4])
                if not zc_:
                    continue
                if self.expected is None:
                    scored.append((zc_, c))
                else:
                    wd = await self._world(c[:4], zc_, w, h)
                    dd = float(np.linalg.norm(wd - self.expected))
                    if dd <= self.gate_mm:
                        scored.append((dd, c))
            cands = [c for _, c in sorted(scored, key=lambda t: t[0])]
        if not self.lock.update(cands, w):
            return {"seen": False, "person_close": person_close}
        box = tuple(float(v) for v in self.lock.box)
        z = box_depth_mm(dmap, box)
        if not z:
            return {"seen": False, "person_close": person_close}
        sx, sy = w / self.intr["w"], h / self.intr["h"]
        fx, fy, cx, cy = self.intr["fx"] * sx, self.intr["fy"] * sy, self.intr["cx"] * sx, self.intr["cy"] * sy
        width, height = width_from_depth(box, z, fx), height_from_depth(box, z, fy)
        edge = box[0] < 3 or box[2] > w - 3             # a box clipped by the side of the picture reads too narrow
        if width and not edge:
            self.widths = (self.widths + [width])[-15:]
        if height and box[1] > 3 and box[3] < h - 3:
            self.heights = (self.heights + [height])[-15:]
        zc = z + (width or 0.0) / 2                       # centre of the object: front surface + half its width
        u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        hpx = float(np.median(self.heights)) * fy / z if len(self.heights) >= 3 else None      # its true size in pixels
        wpx = float(np.median(self.widths)) * fx / z if len(self.widths) >= 3 else None
        if hpx and box[3] > h - 3 and box[1] > 3:         # cut off by the bottom of the picture: top edge + half its height
            v = box[1] + hpx / 2
        elif hpx and box[1] <= 3 and box[3] < h - 3:      # cut off by the top
            v = box[3] - hpx / 2
        if wpx and box[0] < 3 and box[2] < w - 3:         # cut off by the left / right side
            u = box[2] - wpx / 2
        elif wpx and box[2] > w - 3 and box[0] > 3:
            u = box[0] + wpx / 2
        cam_pt = Pose(x=(u - cx) * zc / fx, y=(v - cy) * zc / fy, z=zc)
        wp = (await self.robot.transform_pose(PoseInFrame(reference_frame=self.camera_name, pose=cam_pt), "world")).pose
        if np.hypot(wp.x, wp.y) > self.max_sight_mm or not -200.0 < wp.z < 900.0:
            self.rejected += 1                                # not a place the arm could ever pick from: a false detection
            self.lock.unlock()
            return {"seen": False, "person_close": person_close, "rejected": f"x{wp.x:.0f} y{wp.y:.0f} z{wp.z:.0f}"}
        if self.confirm_n > 1 and self.expected is None:
            here = np.array([wp.x, wp.y, wp.z])
            ok_again = self.candidate is not None and float(np.linalg.norm(here - self.candidate[0])) < 150.0
            self.candidate = (here, (self.candidate[1] + 1) if ok_again else 1)
            if self.candidate[1] < self.confirm_n:
                return {"seen": False, "person_close": person_close}
        if self.expected is not None and float(np.linalg.norm(np.array([wp.x, wp.y, wp.z]) - self.expected)) > self.reject_mm:
            self.lock.unlock()                                # it jumped: that is a DIFFERENT object. Do not follow it, do not remember it
            return {"seen": False, "person_close": person_close}
        aim_up = (self.aim_frac - 0.5) * float(np.median(self.heights)) if (self.aim_frac != 0.5 and len(self.heights) >= 3) else 0.0
        if aim_up:                                        # aim higher/lower on the object (world z is up), then express it in the gripper frame
            wp.z += aim_up
            rel = (await self.robot.transform_pose(PoseInFrame(reference_frame="world", pose=Pose(x=wp.x, y=wp.y, z=wp.z)), self.gripper_name)).pose
        else:
            rel = (await self.robot.transform_pose(PoseInFrame(reference_frame=self.camera_name, pose=cam_pt), self.gripper_name)).pose
        rel = types.SimpleNamespace(x=rel.x + self.trim[0], y=rel.y + self.trim[1], z=rel.z + self.trim[2])
        self.expected = np.array([wp.x, wp.y, wp.z])      # follow this object from now on
        self.candidate = None
        result = {"seen": True, "world": np.array([wp.x, wp.y, wp.z]), "rel_mm": np.array([rel.x, rel.y, rel.z]), "depth_mm": z, "width_mm": width, "height_mm": height,
                "person_close": person_close}
        if self.memory is not None:
            self.memory.update(result, self.width_estimate(), float(np.median(self.heights)) if len(self.heights) >= 3 else None)
            self.expected = self.memory.world             # follow the remembered object, not just the last sighting
        return result

    async def person_world(self):
        """World position (mm) of the nearest person in view (chest-ish: the box centre at its depth), or None."""
        imgs = (await self.cam.get_images())[0]
        color = depth = None
        for im in imgs:
            if "dep" in str(im.mime_type).lower():
                depth = parse_viam_depth(im.data)
            else:
                color = cv2.imdecode(np.frombuffer(im.data, np.uint8), cv2.IMREAD_COLOR)
        if color is None or depth is None:
            return None
        h, w = color.shape[:2]
        dmap = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        dets = await asyncio.to_thread(detect, self.model, color, self.dargs)
        best = None
        for d in dets:
            if d[5] != "person":
                continue
            z = box_depth_mm(dmap, d[:4])
            if z and (best is None or z < best[0]):
                best = (z, d)
        if best is None:
            return None
        z, d = best
        sx, sy = w / self.intr["w"], h / self.intr["h"]
        fx, fy, cx, cy = self.intr["fx"] * sx, self.intr["fy"] * sy, self.intr["cx"] * sx, self.intr["cy"] * sy
        pose = Pose(x=((d[0] + d[2]) / 2 - cx) * z / fx, y=((d[1] + d[3]) / 2 - cy) * z / fy, z=z)
        wp = (await self.robot.transform_pose(PoseInFrame(reference_frame=self.camera_name, pose=pose), "world")).pose
        return np.array([wp.x, wp.y, wp.z])

    def width_estimate(self):
        if len(self.widths) >= 3:
            return float(np.median(self.widths))
        return self.memory.width if self.memory is not None else None


async def main_async(args):
    robot = await connect(args.viam_address)
    real = args.sink == "viam"
    arm = Arm.from_robot(robot, args.arm)
    try:
        gripper, cam = Gripper.from_robot(robot, args.gripper), Camera.from_robot(robot, args.camera)
        motion = MotionClient.from_robot(robot, "builtin")
        ip = (await cam.get_properties()).intrinsic_parameters
        intr = {"fx": ip.focal_x_px, "fy": ip.focal_y_px, "cx": ip.center_x_px, "cy": ip.center_y_px, "w": ip.width_px, "h": ip.height_px}
        model, dargs = load_detector(args.object)
        RESET_JOINTS = (-0.95, 33.21, -28.59, -86.05, 8.90, -272.49)   # the reset pose: where the pick starts from (cup visible from here)
        WRIST_HOME = RESET_JOINTS[3:6]                   # the reset pose's wrist (joints 4, 5, 6): the horizontal-ish rotations the pick starts from

        async def start_joints():
            """Current joints, but with the wrist (4, 5, 6) set to the reset pose's rotations: what the wrist returns to after the vertical sweep,
            wherever the run itself began (it may begin in the sweep pose)."""
            j = list((await arm.get_joint_positions()).values)
            j[3:6] = list(WRIST_HOME)
            return j

        home = list((await arm.get_joint_positions()).values)          # the ORIGINAL pose: where to come back to if a search pass finds nothing
        memory = ObjectMemory(args.memory_file, args.object)
        if args.use_memory and memory.load():
            print(f"loaded the remembered {args.object}: world {np.round(memory.world)}, width {memory.width}", flush=True)
        eye = Perceiver(robot, args.camera, args.gripper, cam, intr, model, dargs, args.object, args.person_mm, args.aim_height_frac, memory)
        eye.max_sight_mm, eye.confirm_n = args.max_sight_mm, args.confirm_sightings
        eye.trim = np.array([float(v) for v in args.aim_trim_mm.split(',')])
        if memory.world is not None:
            eye.expected = memory.world
        if args.near_world:
            eye.expected = np.array([float(v) for v in args.near_world.split(",")])

        # ---------------- optional web page (--ui): live view, status, chat commands, STOP ----------------
        loop = asyncio.get_running_loop()
        flags = {"stop": False, "paused": False, "yes": None, "go": not args.ui, "grip": False}
        view = {"text": "starting", "state": "starting", "dist": None}
        ui = None

        runlog = RunLog(args)
        runlog.metric("width_mm", memory.width)

        def say(msg, **_):
            print(msg, flush=True)
            runlog.event(msg)
            if ui:
                ui.say(msg)

        def show(state, text, dist=None):
            """Status text + the live camera frame with every detection (grey) and the locked one (green)."""
            view["state"], view["text"], view["dist"] = state, text, dist
            runlog.stage(state, text)
            if not ui or eye.color is None:
                return
            img = eye.color.copy()
            for c in eye.cands:
                cv2.rectangle(img, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (160, 160, 160), 1)
            if eye.lock.locked:
                b = [int(v) for v in eye.lock.box]
                cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 200, 0), 3)
            for k, line in enumerate((("LIVE ARM" if real else "DRY RUN - nothing moves") + f"  |  {state}", text[:64])):
                cv2.putText(img, line, (12, 26 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
                cv2.putText(img, line, (12, 26 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                ui.set_frame(buf.tobytes())

        def check_stop():
            if flags["stop"]:
                raise SystemExit("stopped from the page")

        async def confirm(q):
            """Typed in the terminal, or answered 'yes' / 'no' in the page's chat."""
            if args.yes:
                return True
            if ui:
                flags["yes"] = None
                say(q + " Reply 'yes' to continue or 'no' to cancel.")
                t0 = time.monotonic()
                while flags["yes"] is None and not flags["stop"] and time.monotonic() - t0 < 120:
                    await asyncio.sleep(0.2)
                return bool(flags["yes"])
            return (await asyncio.to_thread(input, f"{q} Type 'yes' to continue: ")).strip().lower() == "yes"

        if args.ui:
            names = {str(n).lower() for n in (model.names.values() if hasattr(model, "names") else [])}

            def on_text(text):
                t = text.lower().strip()
                if "stop" in t:
                    flags["stop"] = True
                    if real:
                        asyncio.run_coroutine_threadsafe(arm.stop(), loop)
                    return "Stopped. Nothing else will move."
                if t.startswith(("grip", "close", "gripper")) or "grip now" in t or "close now" in t:
                    flags["grip"], flags["yes"] = True, True
                    return "OK: closing the gripper now (skipping the rest of the advance and the checks)."
                if t in ("yes", "y", "confirm", "ok", "go ahead"):
                    flags["yes"] = True
                    return "OK."
                if t in ("no", "n", "cancel"):
                    flags["yes"] = False
                    return "Cancelled."
                if "pause" in t:
                    flags["paused"] = True
                    return "Paused."
                if "resume" in t or "continue" in t:
                    flags["paused"] = False
                    return "Resumed."
                if any(w in t for w in ("pick", "grab", "get", "fetch", "go to", "bring")):
                    if flags["go"]:
                        return "Already on it."
                    word = next((n for n in sorted(names, key=len, reverse=True) if n in t), None)   # "pick up the cup" -> cup
                    if word and word != eye.label:
                        eye.label = word
                        eye.reset()
                        eye.expected, eye.widths, eye.heights = None, [], []
                        memory.label, memory.world, memory.first_world, memory.width, memory.height = word, None, None, None, None
                    flags["go"] = True
                    return f"Going to get the {eye.label}: sweep, approach, flatten, advance, grasp, lift." if real else \
                        f"DRY RUN: looking for the {eye.label}; nothing will move."
                if "see" in t or "status" in t:
                    return view["text"]
                return "I understand: pick up the <object>, stop, pause, resume, yes / no, what do you see."

            ui = UIServer(on_text, lambda: {"mode": "LIVE ARM" if real else "DRY RUN", "state": view["state"], "target": eye.label,
                                            "distance": None if view["dist"] is None else f"{view['dist']:.0f} cm"}, args.port).start()
            print("UI:", ui.url, flush=True)
            if not args.no_browser:
                webbrowser.open(ui.url)
            while not flags["go"] and not flags["stop"]:            # watch the scene until told to go
                try:
                    await eye.look()
                except Exception:
                    pass
                seen_ = f"{len(eye.cands)} {eye.label}(s) in view" if eye.cands else f"no {eye.label} in view"
                show("waiting", seen_ + " - say 'pick up the " + eye.label + "'")
                await asyncio.sleep(0.15)
            check_stop()

        async def gripper_pose():
            return (await robot.transform_pose(PoseInFrame(reference_frame=args.gripper, pose=Pose()), "world")).pose

        def at(pos, gp):                                  # a world pose with the gripper's current orientation
            return Pose(x=float(pos[0]), y=float(pos[1]), z=float(max(pos[2], args.min_z_mm)),
                        o_x=gp.o_x, o_y=gp.o_y, o_z=gp.o_z, theta=gp.theta)

        async def go(pos, gp, obstacle_rel=None):
            """Planned move of the gripper to a world position, keeping its orientation (this SDK wants the component name as a string). The
            object is an obstacle when obstacle_rel (its gripper-frame reading) is given. Returns False when the step is unreachable / blocked."""
            check_stop()
            try:
                await motion.move(args.gripper, PoseInFrame(reference_frame="world", pose=at(pos, gp)), world_state=await object_obstacle(obstacle_rel))
            except Exception as e:
                if "IK solution" in str(e) or "unreachable" in str(e) or "obstacle constraint" in str(e):
                    return False
                raise
            await check_j6()
            return True

        async def rel_of(t, known_world):
            """The object in the GRIPPER's own frame (z = along the gripper axis): straight from the latest reading, or, when it is
            out of view, the last known world position converted by the frame system. Independent of world-frame drift."""
            if t:
                return t["rel_mm"]
            p = (await robot.transform_pose(PoseInFrame(reference_frame="world", pose=Pose(x=float(known_world[0]), y=float(known_world[1]),
                                                                                           z=float(known_world[2]))), args.gripper)).pose
            return np.array([p.x, p.y, p.z])

        vox = VoxelMap() if args.voxel else None

        async def cam_to_world():
            """The camera's pose in the world as (R, origin): the frame system maps the camera's origin and unit axes."""
            async def pt(x, y, z):
                p = (await robot.transform_pose(PoseInFrame(reference_frame=args.camera, pose=Pose(x=x, y=y, z=z)), "world")).pose
                return np.array([p.x, p.y, p.z])
            o = await pt(0, 0, 0)
            return np.stack([await pt(1, 0, 0) - o, await pt(0, 1, 0) - o, await pt(0, 0, 1) - o], axis=1), o

        async def vox_update():
            """Fold the latest depth frame into the voxel map (rays carve free space, so an object a person moves is cleared and re-added)."""
            if vox is None or eye.last_dmap is None:
                return
            h_, w_ = eye.last_dmap.shape[:2]
            sx_, sy_ = w_ / intr["w"], h_ / intr["h"]
            pts = depth_to_points(eye.last_dmap, intr["fx"] * sx_, intr["fy"] * sy_, intr["cx"] * sx_, intr["cy"] * sy_, stride=args.voxel_stride)
            R_, o_ = await cam_to_world()
            await asyncio.to_thread(vox.integrate, pts @ R_.T + o_, o_)

        async def object_obstacle(rel):
            """The object as a box obstacle for the planner, placed from the SAME gripper-frame measurement that drives the move (so the
            two always agree; the remembered world position can differ from it by ~10 cm). None if there is no measurement."""
            if rel is None:
                return None
            wp = (await robot.transform_pose(PoseInFrame(reference_frame=args.gripper, pose=Pose(x=float(rel[0]), y=float(rel[1]),
                                                                                                 z=float(rel[2]))), "world")).pose
            wd, ht = memory.width or 90.0, memory.height or 220.0      # defaults when its size is not known (its box was clipped)
            centre, dims = np.array([wp.x, wp.y, wp.z]), np.array([wd + 10, wd + 10, ht + 10])
            if vox is not None:                                       # the map's real extent of whatever stands there now, if it has seen it
                box = vox.object_box(centre, radius_mm=250.0, min_z_mm=args.table_top_mm + 40.0)
                if box is not None:
                    centre, dims = box
                    if memory.width:                                  # a cup is no wider than its measured width: stray voxels must not inflate the box
                        dims = np.array([min(dims[0], memory.width + 25.0), min(dims[1], memory.width + 25.0), dims[2]])
            g = Geometry(center=Pose(x=float(centre[0]), y=float(centre[1]), z=float(centre[2])),
                         box=RectangularPrism(dims_mm=Vector3(x=float(dims[0]), y=float(dims[1]), z=float(dims[2]))), label=args.object)
            return WorldState(obstacles=[GeometriesInFrame(reference_frame="world", geometries=[g])])

        j6_ref = {"v": None}

        async def check_j6():
            """Joint 6 (the final roll) must not wander: the planner holds the gripper's world orientation, so a long base swing can
            make it spin. Stop if it has turned more than --max-j6-deg from where the run began."""
            j6 = (await arm.get_joint_positions()).values[5]
            if j6_ref["v"] is None:
                j6_ref["v"] = j6
            elif args.max_j6_deg and abs(j6 - j6_ref["v"]) > args.max_j6_deg:
                raise SystemExit(f"joint 6 has turned {j6 - j6_ref['v']:+.0f} deg since the start (limit {args.max_j6_deg:.0f}): stopping")

        async def go_down(mm):
            """Planned move straight down (world z) by mm, keeping the gripper's orientation."""
            gp = await gripper_pose()
            await motion.move(args.gripper, PoseInFrame(reference_frame="world", pose=at(np.array([gp.x, gp.y, gp.z - mm]), gp)))
            await check_j6()

        async def go_rel(v, touch_ok=False, obstacle_rel=None):
            check_stop()
            """Plan a move of v (mm, gripper frame) from where the gripper is now, keeping its orientation. The remembered object is
            an obstacle for the planner. For the final advance (touch_ok) the fingers must straddle it, so the obstacle stays in the
            world but the planner is told the claws may touch it (planner geometry names: 'gripper:claws' and the obstacle's label);
            if the planner rejects that, the obstacle is left out for this step."""
            p = (await robot.transform_pose(PoseInFrame(reference_frame=args.gripper, pose=Pose(x=float(v[0]), y=float(v[1]), z=float(v[2]))),
                                            "world")).pose
            goal = PoseInFrame(reference_frame="world", pose=Pose(x=p.x, y=p.y, z=max(p.z, args.min_z_mm), o_x=p.o_x, o_y=p.o_y, o_z=p.o_z, theta=p.theta))
            ws = await object_obstacle(obstacle_rel)
            allow = None
            if touch_ok and ws is not None:
                allow = Constraints(collision_specification=[CollisionSpecification(allows=[
                    CollisionSpecification.AllowedFrameCollisions(frame1="gripper:claws", frame2=args.object)])])
            close_in = obstacle_rel is not None and float(obstacle_rel[2]) < args.free_within_mm
            try:
                try:
                    await motion.move(args.gripper, goal, world_state=ws, constraints=allow)
                except Exception as e:
                    # near the cup the estimated box and the claws overlap by the measurement error (2-5 cm), so the planner refuses every
                    # step. The cup is the only obstacle dropped; the table and walls stay in the planner's world.
                    if ws is None or not (allow is not None or (close_in and "obstacle constraint" in str(e))):
                        raise
                    say(f"  planner: {str(e)[:200]}", flush=True)
                    say("  close to the cup: retrying this step without the cup as an obstacle", flush=True)
                    await motion.move(args.gripper, goal, world_state=None)
            except Exception as e:
                if "IK solution" in str(e) or "unreachable" in str(e) or "obstacle constraint" in str(e):
                    return False
                raise
            await check_j6()
            return True

        async def rel_of(t, known_world):
            """The object in the GRIPPER's own frame (z = along the gripper axis): straight from the latest reading, or, when it is
            out of view, the last known world position converted by the frame system. Independent of world-frame drift."""
            if t:
                return t["rel_mm"]
            p = (await robot.transform_pose(PoseInFrame(reference_frame="world", pose=Pose(x=float(known_world[0]), y=float(known_world[1]),
                                                                                           z=float(known_world[2]))), args.gripper)).pose
            return np.array([p.x, p.y, p.z])

        vox = VoxelMap() if args.voxel else None

        async def cam_to_world():
            """The camera's pose in the world as (R, origin): the frame system maps the camera's origin and unit axes."""
            async def pt(x, y, z):
                p = (await robot.transform_pose(PoseInFrame(reference_frame=args.camera, pose=Pose(x=x, y=y, z=z)), "world")).pose
                return np.array([p.x, p.y, p.z])
            o = await pt(0, 0, 0)
            return np.stack([await pt(1, 0, 0) - o, await pt(0, 1, 0) - o, await pt(0, 0, 1) - o], axis=1), o

        async def vox_update():
            """Fold the latest depth frame into the voxel map (rays carve free space, so an object a person moves is cleared and re-added)."""
            if vox is None or eye.last_dmap is None:
                return
            h_, w_ = eye.last_dmap.shape[:2]
            sx_, sy_ = w_ / intr["w"], h_ / intr["h"]
            pts = depth_to_points(eye.last_dmap, intr["fx"] * sx_, intr["fy"] * sy_, intr["cx"] * sx_, intr["cy"] * sy_, stride=args.voxel_stride)
            R_, o_ = await cam_to_world()
            await asyncio.to_thread(vox.integrate, pts @ R_.T + o_, o_)

        async def object_obstacle(rel):
            """The object as a box obstacle for the planner, placed from the SAME gripper-frame measurement that drives the move (so the
            two always agree; the remembered world position can differ from it by ~10 cm). None if there is no measurement."""
            if rel is None:
                return None
            wp = (await robot.transform_pose(PoseInFrame(reference_frame=args.gripper, pose=Pose(x=float(rel[0]), y=float(rel[1]),
                                                                                                 z=float(rel[2]))), "world")).pose
            wd, ht = memory.width or 90.0, memory.height or 220.0      # defaults when its size is not known (its box was clipped)
            centre, dims = np.array([wp.x, wp.y, wp.z]), np.array([wd + 10, wd + 10, ht + 10])
            if vox is not None:                                       # the map's real extent of whatever stands there now, if it has seen it
                box = vox.object_box(centre, radius_mm=250.0, min_z_mm=args.table_top_mm + 40.0)
                if box is not None:
                    centre, dims = box
            g = Geometry(center=Pose(x=float(centre[0]), y=float(centre[1]), z=float(centre[2])),
                         box=RectangularPrism(dims_mm=Vector3(x=float(dims[0]), y=float(dims[1]), z=float(dims[2]))), label=args.object)
            return WorldState(obstacles=[GeometriesInFrame(reference_frame="world", geometries=[g])])

        j6_ref = {"v": None}

        async def check_j6():
            """Joint 6 (the final roll) must not wander: the planner holds the gripper's world orientation, so a long base swing can
            make it spin. Stop if it has turned more than --max-j6-deg from where the run began."""
            j6 = (await arm.get_joint_positions()).values[5]
            if j6_ref["v"] is None:
                j6_ref["v"] = j6
            elif args.max_j6_deg and abs(j6 - j6_ref["v"]) > args.max_j6_deg:
                raise SystemExit(f"joint 6 has turned {j6 - j6_ref['v']:+.0f} deg since the start (limit {args.max_j6_deg:.0f}): stopping")

        async def go_down(mm):
            """Planned move straight down (world z) by mm, keeping the gripper's orientation."""
            gp = await gripper_pose()
            await motion.move(args.gripper, PoseInFrame(reference_frame="world", pose=at(np.array([gp.x, gp.y, gp.z - mm]), gp)))
            await check_j6()

        async def go_rel(v, touch_ok=False, obstacle_rel=None):
            check_stop()
            """Plan a move of v (mm, gripper frame) from where the gripper is now, keeping its orientation. The remembered object is
            an obstacle for the planner. For the final advance (touch_ok) the fingers must straddle it, so the obstacle stays in the
            world but the planner is told the claws may touch it (planner geometry names: 'gripper:claws' and the obstacle's label);
            if the planner rejects that, the obstacle is left out for this step."""
            p = (await robot.transform_pose(PoseInFrame(reference_frame=args.gripper, pose=Pose(x=float(v[0]), y=float(v[1]), z=float(v[2]))),
                                            "world")).pose
            goal = PoseInFrame(reference_frame="world", pose=Pose(x=p.x, y=p.y, z=max(p.z, args.min_z_mm), o_x=p.o_x, o_y=p.o_y, o_z=p.o_z, theta=p.theta))
            ws = await object_obstacle(obstacle_rel)
            allow = None
            if touch_ok and ws is not None:
                allow = Constraints(collision_specification=[CollisionSpecification(allows=[
                    CollisionSpecification.AllowedFrameCollisions(frame1="gripper:claws", frame2=args.object)])])
            try:
                try:
                    await motion.move(args.gripper, goal, world_state=ws, constraints=allow)
                except Exception as e:
                    if allow is None or not ("collision" in str(e).lower() or "frame" in str(e).lower() or "name" in str(e).lower()):
                        raise
                    say("  the planner did not accept the claws/object exception: this step goes without the obstacle", flush=True)
                    await motion.move(args.gripper, goal, world_state=None)
            except Exception as e:
                if "IK solution" in str(e) or "unreachable" in str(e) or "obstacle constraint" in str(e):
                    say(f"  that step is not reachable / blocked: skipping it ({str(e)[:160]})", flush=True)
                    return False
                raise
            await check_j6()
            return True

        async def next_target(prev, misses):
            """look; wait while a person is close; returns (target, misses)."""
            while True:
                check_stop()
                while flags["paused"] and not flags["stop"]:
                    await asyncio.sleep(0.2)
                t = await eye.look()
                await vox_update()
                if t and t["person_close"]:
                    show("paused", "a person is close: waiting")
                    say(f"a person is close to the camera/gripper: waiting ({eye.person_info})", flush=True)
                    await asyncio.sleep(1.0)
                    continue
                show("tracking" if t and t["seen"] else "searching",
                     f"{eye.label}: {t['depth_mm'] / 10:.0f} cm, ~{t['width_mm']:.0f} mm wide" if t and t["seen"] else f"no {eye.label} in view",
                     t["depth_mm"] / 10 if t and t["seen"] else None)
                break
            if not t or not t["seen"]:
                return None, misses + 1
            if prev is not None and np.linalg.norm(t["world"] - prev) > args.jump_mm:
                say(f"  the object moved {np.linalg.norm(t['world'] - prev):.0f} mm: re-targeting", flush=True)
            return t, 0

        async def reacquire(known):
            """Lost it. It has probably not gone anywhere, it just left the picture: back off along the camera axis, then sweep
            the base joint a few degrees each way, looking after every move. Returns a fresh target, or None."""
            say("lost the object: searching nearby ...", flush=True)
            gp = await gripper_pose()
            gpos, axis = np.array([gp.x, gp.y, gp.z]), unit([gp.o_x, gp.o_y, gp.o_z])
            for retreat in (80.0, 160.0):
                await go(gpos - axis * retreat, gp)
                eye.lock.unlock()
                for _ in range(3):
                    t, _ = await next_target(None, 0)
                    if t:
                        say(f"  found it again after backing off {retreat:.0f} mm", flush=True)
                        return t
                    await asyncio.sleep(0.2)
            t = await wrist_scan()
            if t:
                return t
            j_here = list((await arm.get_joint_positions()).values)
            for d in (6.0, -6.0, 12.0, -12.0):             # base yaw sweep, bounded to +/-12 degrees
                j = list(j_here)
                j[0] += d
                await arm.move_to_joint_positions(JointPositions(values=j))
                eye.lock.unlock()
                for _ in range(3):
                    t, _ = await next_target(None, 0)
                    if t:
                        say(f"  found it again after turning the base {d:+.0f} deg", flush=True)
                        return t
                    await asyncio.sleep(0.2)
            await arm.move_to_joint_positions(JointPositions(values=j_here))
            return None

        async def wrist_scan():
            """Look around by moving ONLY the two wrist joints (4 and 5), which pan/tilt the camera a little each way, looking after every
            pose. Returns a target (the arm stays where it saw it) or None (the wrist is put back where it was)."""
            j_here = list((await arm.get_joint_positions()).values)
            say("  scanning with the wrist (joints 4 and 5) ...", flush=True)
            for d4, d5 in ((0, 10), (0, -10), (10, 0), (-10, 0), (10, 10), (10, -10), (-10, 10), (-10, -10),
                           (0, 20), (0, -20), (20, 0), (-20, 0)):
                j = list(j_here)
                j[3] += d4
                j[4] += d5
                await arm.move_to_joint_positions(JointPositions(values=j))
                eye.lock.unlock()
                for _ in range(2):
                    t, _ = await next_target(None, 0)
                    if t:
                        say(f"  found a {args.object} with the wrist at joint 4 {d4:+d}, joint 5 {d5:+d} deg", flush=True)
                        return t
                    await asyncio.sleep(0.15)
            await arm.move_to_joint_positions(JointPositions(values=j_here))
            return None

        async def flatten():
            """Level the gripper with ONE of the wrist joints (4 or 5; never 6): test +/-10 degrees on each, pick the one that levels it
            most (preferring the gripper pointing forward), then step it in 10-degree increments until level."""
            candidates = [args.wrist_joint] if args.wrist_joint else [4, 5]
            j0 = list((await arm.get_joint_positions()).values)
            p0, results = await gripper_pose(), []
            for jn in candidates:
                for sg in ((args.wrist_sign,) if args.wrist_sign else (1, -1)):
                    if not in_limits(jn, j0[jn - 1] + sg * 10.0):
                        say(f"  test: joint {jn} {sg * 10:+.0f} deg would pass its limit: skipped", flush=True)
                        continue
                    t = list(j0)
                    t[jn - 1] += sg * 10.0                        # 10-degree test move, then straight back
                    await arm.move_to_joint_positions(JointPositions(values=t))
                    p1 = await gripper_pose()
                    await arm.move_to_joint_positions(JointPositions(values=j0))
                    gain = abs(p0.o_z) - abs(p1.o_z)              # how much more level it got
                    results.append((gain, p1.o_x > p0.o_x, jn, sg))
                    say(f"  test: joint {jn} {sg * 10:+.0f} deg -> levels by {gain:+.2f}" + (", pointing more forward" if p1.o_x > p0.o_x else ""), flush=True)
            good = [r for r in results if r[0] > 0.02]
            if not good:
                raise SystemExit(f"neither joint {' nor '.join(str(c) for c in candidates)} levels the gripper from here; not rotating")
            _, _, jn, sign = max(good, key=lambda r: (r[1], r[0]))   # prefer pointing forward, then the biggest levelling
            total, best_total, best_abs = 0.0, 0.0, abs(p0.o_z)
            while total < args.wrist_deg:
                step = min(10.0, args.wrist_deg - total)
                if not in_limits(jn, j0[jn - 1] + sign * (total + step)):
                    say(f"  joint {jn} would pass its limit: stopping the flatten here", flush=True)
                    break
                total += step
                t = list(j0)
                t[jn - 1] += sign * total
                await arm.move_to_joint_positions(JointPositions(values=t))
                oz = abs((await gripper_pose()).o_z)
                if oz < best_abs - 1e-3:
                    best_abs, best_total = oz, total
                if oz < 0.05:
                    break                                     # level
                if oz > best_abs + 0.01:                      # it got LESS level: it overshot past level (it would end up pointing up)
                    break
            t = list(j0)
            t[jn - 1] += sign * best_total                    # settle at the most level angle reached (never the overshoot)
            await arm.move_to_joint_positions(JointPositions(values=t))
            total = best_total
            gp = await gripper_pose()
            say(f"joint {jn} rotated {sign * total:+.0f} deg; gripper axis ({gp.o_x:+.2f}, {gp.o_y:+.2f}, {gp.o_z:+.2f})", flush=True)

        async def _survey_inner():
            """Nothing in view at the start and no remembered position: look around. Sweep the base in --survey-step-deg steps out to
            +/- --survey-deg, looking after each move; stop at the first object seen (nearest by depth in that view)."""
            say(f"no {eye.label} in view: looking around ...", flush=True)
            try:                                               # exploring is done from the saved top-down sweep pose
                await go_sweep_pose()
            except SystemExit as e:
                say(f"  (could not go to the sweep pose for the survey: {e})", flush=True)
            t = await wrist_scan()
            if t:
                return t
            j_here = list((await arm.get_joint_positions()).values)
            steps = [a * sg for a in np.arange(args.survey_step_deg, args.survey_deg + 1e-6, args.survey_step_deg) for sg in (1.0, -1.0)]
            for d in steps:
                j = list(j_here)
                j[args.survey_joint - 1] += d
                await arm.move_to_joint_positions(JointPositions(values=j))
                eye.lock.unlock()
                eye.expected = None                          # nothing to match yet: take the nearest in this view
                for _ in range(3):
                    t, _ = await next_target(None, 0)
                    if t:
                        say(f"  found a {args.object} after turning joint {args.survey_joint} {d:+.0f} deg", flush=True)
                        return t
                    await asyncio.sleep(0.2)
            await arm.move_to_joint_positions(JointPositions(values=j_here))
            return None

        LIMIT_DEG = {4: 120.0, 5: 120.0}      # wrist joint limits (joint 4 is +/-124 on this arm); stay a few degrees inside them

        def in_limits(jn, deg):
            return abs(deg) <= LIMIT_DEG.get(jn, 360.0)

        async def restore_wrist(j_start):
            """Put the wrist joints (4, 5, 6) back where the run started (the horizontal / near-reset rotations), keeping the other joints."""
            j = list((await arm.get_joint_positions()).values)
            for k in (3, 4, 5):
                j[k] = j_start[k]
            await arm.move_to_joint_positions(JointPositions(values=j))

        async def gripper_pos():
            """Gripper opening as the module reports it (position units, 10 per mm), or None if the reply has no number."""
            try:
                for v in (await gripper.do_command({"get": True})).values():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        return float(v)
            except Exception:
                pass
            return None

        async def retry_attempt(why):
            """The grasp failed: open, back away, go to the start pose, and raise GraspFailed so main() starts the whole attempt again
            (up to --retries times). The person confirms the move first."""
            n = args.attempt + 1
            say(f"GRASP FAILED (attempt {n}/{args.retries + 1}): {why}", flush=True)
            runlog.metric("grasp_failures", args.attempt + 1)
            if args.attempt >= args.retries:
                raise SystemExit(f"grasp failed after {n} attempts: {why}")
            if not await confirm(f"RETRY {n + 1}/{args.retries + 1}: open the gripper, back off 12 cm, go to the start pose and look again."):
                raise SystemExit("retry declined")
            await gripper.do_command({"set": 840.0})
            await asyncio.sleep(1.5)
            await go_rel(np.array([0.0, 0.0, -120.0]))
            await arm.move_to_joint_positions(JointPositions(values=list(RESET_JOINTS)))
            raise GraspFailed(f"grasp failed: {why}")

        async def hand_over():
            """Give the object to a person: hold it out toward the nearest person (or at --handover-xyz), wait until they take it (a pull
            on the arm, or 'release' / 'yes' in the chat), then open the gripper and back away."""
            person = await eye.person_world()
            if person is not None:
                reach_dir = unit([person[0], person[1], 0.0])
                radius = float(np.clip(np.hypot(person[0], person[1]) - args.handover_gap_mm, 350.0, args.handover_max_reach_mm))
                spot = np.array([reach_dir[0] * radius, reach_dir[1] * radius, args.handover_z_mm])
                say(f"a person is at world x{person[0]:.0f} y{person[1]:.0f}: I will hold it out at x{spot[0]:.0f} y{spot[1]:.0f} z{spot[2]:.0f}")
            else:
                spot = np.array([float(v) for v in args.handover_xyz.split(",")])
                say(f"no person seen: I will hold it out at x{spot[0]:.0f} y{spot[1]:.0f} z{spot[2]:.0f}")
            if not await confirm(f"HAND OVER: move the {eye.label} to x{spot[0]:.0f} y{spot[1]:.0f} z{spot[2]:.0f} and hold it out for you to take."):
                return
            async def speed(dps):
                try:
                    await arm.do_command({"set_speed": float(dps)})
                except Exception:
                    pass
            await speed(args.handover_speed_dps)                        # slow near the person; restored below
            try:
                gp = await gripper_pose()
                await motion.move(args.gripper, PoseInFrame(reference_frame="world", pose=at(spot, gp)))
                await check_j6()
                show("handover", "holding it out: take it, or say 'release'")
                say(f"Holding the {eye.label} out. Take it: I let go when I feel a steady pull, or reply 'release' / 'yes'.")
                await asyncio.sleep(2.0)                               # let the arm settle, then read the baseline load
                base = np.array((await arm.do_command({"load": ""})).get("load", [0.0] * 6), float)
                flags["yes"] = None
                t0, since, taken = time.monotonic(), None, False
                while time.monotonic() - t0 < args.handover_wait_s and not flags["stop"]:
                    if flags["yes"]:
                        say("released on your command")
                        taken = True
                        break
                    load = np.array((await arm.do_command({"load": ""})).get("load", [0.0] * 6), float)
                    if float(np.max(np.abs(load - base))) > args.pull_load:
                        since = since or time.monotonic()               # a spike does not count: the pull must last
                        if time.monotonic() - since >= args.pull_hold_s:
                            say("felt a steady pull: letting go")
                            taken = True
                            break
                    else:
                        since = None
                    await asyncio.sleep(0.05)
                if not taken:
                    say(f"nobody took it within {args.handover_wait_s:.0f} s: putting it back down")
                    await go_rel(np.array([0.0, 0.0, 0.0]))
                    gp = await gripper_pose()
                    await motion.move(args.gripper, PoseInFrame(reference_frame="world", pose=at(np.array([gp.x, gp.y, max(args.table_top_mm + 120.0, args.min_z_mm)]), gp)))
                    await asyncio.sleep(0.5)
                await gripper.do_command({"set": 840.0})                # open: theirs (or back on the table)
                await asyncio.sleep(args.handover_settle_s)             # let go fully before moving away
                await go_rel(np.array([0.0, 0.0, -100.0]))              # back away 10 cm along the gripper axis
            finally:
                await speed(args.arm_speed_dps)
            say("handed over." if taken else "put it back down.")
            show("done", "handed over" if taken else "put down")

        async def set_roll():
            """Set the final roll joint (--roll-joint, app numbering; 6) to the taught pick value (--roll-target-deg, about -360), which is
            the roll the gripper had in the taught pick pose (jaws opening sideways). Small steps, other joints untouched."""
            idx = args.roll_joint - 1
            j = list((await arm.get_joint_positions()).values)
            cur, target = j[idx], args.roll_target_deg
            if abs(cur - target) <= args.roll_tol_deg:
                say(f"joint {args.roll_joint} is already at {cur:.0f} deg (target {target:.0f}): roll unchanged", flush=True)
                return
            say(f"setting joint {args.roll_joint} (the gripper roll) from {cur:.0f} to {target:.0f} deg for the pick", flush=True)
            while abs(j[idx] - target) > 1.0:
                j[idx] += float(np.clip(target - j[idx], -15.0, 15.0))      # 15-degree steps toward the target
                await arm.move_to_joint_positions(JointPositions(values=j))
                check_stop()
            say(f"joint {args.roll_joint} is now {(await arm.get_joint_positions()).values[idx]:.0f} deg", flush=True)

        async def go_sweep_pose():
            """FIRST step of the sweep: go to the saved sweep pose (data/sweep_pose.json): gripper vertical, camera looking straight down at
            the table. Falls back to pointing down with joint 5 if there is no saved pose."""
            import json
            try:
                joints = [float(v) for v in json.load(open(args.sweep_pose))["joints_deg_J1_to_J6"]]
            except (OSError, ValueError, KeyError):
                say("  no saved sweep pose: pointing the gripper down with joint 5 instead", flush=True)
                await orient_down()
                return
            say("  moving to the saved sweep pose (gripper vertical, looking down at the table) ...", flush=True)
            await arm.move_to_joint_positions(JointPositions(values=joints))
            gp = await gripper_pose()
            say(f"  sweep pose reached: tool axis ({gp.o_x:+.2f}, {gp.o_y:+.2f}, {gp.o_z:+.2f})", flush=True)

        async def orient_down():
            """Point the gripper straight DOWN (tool direction o_z -> -1) with ONE of the wrist joints (4 or 5, whichever points it further
            down; never 6), so the camera looks at the table below during the survey. The later flatten stage levels it for the grasp."""
            p0 = await gripper_pose()
            if p0.o_z < -0.95:
                return
            j0 = list((await arm.get_joint_positions()).values)
            results = []
            for jn in (args.down_joint,):                     # joint 5 by default: the wrist pitch. Joint 4 rolls the gripper: left alone
                for sg in (1, -1):
                    if not in_limits(jn, j0[jn - 1] + sg * 10.0):
                        continue
                    t = list(j0)
                    t[jn - 1] += sg * 10.0                        # 10-degree test move, then straight back
                    await arm.move_to_joint_positions(JointPositions(values=t))
                    p1 = await gripper_pose()
                    await arm.move_to_joint_positions(JointPositions(values=j0))
                    results.append((abs(p1.o_z) - abs(p0.o_z), jn, sg))      # how much MORE vertical it got
                    say(f"  test: joint {jn} {sg * 10:+.0f} deg -> {abs(p1.o_z) - abs(p0.o_z):+.2f} more vertical")
            good = [r for r in results if r[0] > 0.05]            # a real gain per 10 degrees, not noise
            if not good:
                say(f"  joint {args.down_joint} does not point the gripper further down from here; sweeping as it is")
                return
            _, jn, sg = max(good)
            total, last_oz = 0.0, p0.o_z
            while total < 100.0:
                if not in_limits(jn, j0[jn - 1] + sg * (total + 10.0)):
                    break
                total += 10.0
                t = list(j0)
                t[jn - 1] += sg * total
                await arm.move_to_joint_positions(JointPositions(values=t))
                oz = (await gripper_pose()).o_z
                if oz < -0.97:
                    break
                if last_oz - oz < 0.01:                       # it has stopped getting more vertical
                    break
                last_oz = oz
            gp = await gripper_pose()
            say(f"gripper pointed down with joint {jn} ({sg * total:+.0f} deg): axis ({gp.o_x:+.2f}, {gp.o_y:+.2f}, {gp.o_z:+.2f})")

        async def survey():
            """Look-around fallback (vertical gripper, wrist scan, joint sweep); once something is found the wrist goes back to the
            rotations the run started with (the horizontal / near-reset ones) before the approach."""
            j_start = await start_joints()
            t = await _survey_inner()
            if not t:
                return None
            await arm.move_to_joint_positions(JointPositions(values=list(RESET_JOINTS)))      # back to the full reset pose
            eye.lock.unlock()
            eye.widths, eye.heights = [], []
            for _ in range(3):
                t2, _ = await next_target(None, 0)
                if t2:
                    return t2
                await asyncio.sleep(0.2)
            say("  cannot see it from the restored pose: continuing with the position seen during the look-around", flush=True)
            return t

        async def sweep_survey():
            """ALWAYS run first in a real run: world positions change between runs, so nothing remembered is trusted. Sweep the base from the
            starting pose in --survey-step-deg steps out to +/- --survey-deg, record the nearest object at each stop, then go back to the
            stop where the nearest one was seen and lock on it. The memory is reset to that object."""
            say("initial sweep: looking around from the starting pose (object positions change, so always sweep first) ...", flush=True)
            j_start = await start_joints()                         # the wrist's horizontal-ish (reset) rotations to come back to
            await go_sweep_pose()                                 # FIRST: the saved top-down sweep pose, then sweep joint 2 around it
            j_here = list((await arm.get_joint_positions()).values)
            offsets = [0.0] + [a * sg for a in np.arange(args.survey_step_deg, args.survey_deg + 1e-6, args.survey_step_deg) for sg in (1.0, -1.0)]
            seen = []
            for d in offsets:
                j = list(j_here)
                j[args.survey_joint - 1] += d
                await arm.move_to_joint_positions(JointPositions(values=j))
                if (await gripper_pose()).z < args.min_z_mm + 40.0:      # a joint-2 stop can bring the gripper down toward the table: skip it
                    say(f"  joint {args.survey_joint} {d:+.0f} deg: too close to the table, skipping this stop", flush=True)
                    await arm.move_to_joint_positions(JointPositions(values=j_here))
                    continue
                eye.lock.unlock()
                eye.expected = None
                best = None
                for _ in range(3):
                    t, _ = await next_target(None, 0)
                    if t and (best is None or t["depth_mm"] < best["depth_mm"]):
                        best = t
                    await asyncio.sleep(0.15)
                if best:
                    reach = float(np.hypot(*best["world"][:2]))
                    say(f"  joint {args.survey_joint} {d:+.0f} deg: {args.object} {best['depth_mm'] / 10:.0f} cm from the camera (~{reach:.0f} mm from the base)", flush=True)
                    if reach <= 1000:
                        seen.append((best["depth_mm"], d, best))
            if not seen:
                await arm.move_to_joint_positions(JointPositions(values=list(RESET_JOINTS)))
                say("  the sweep saw nothing within reach", flush=True)
                return None
            _, d, chosen = min(seen, key=lambda x: x[0])
            say(f"  nearest: seen at joint {args.survey_joint} {d:+.0f} deg; returning to the reset pose to start the pick", flush=True)
            await arm.move_to_joint_positions(JointPositions(values=list(RESET_JOINTS)))     # the whole reset pose, not a mix of poses
            eye.lock.unlock()
            # COMMIT the chosen object to memory: position, size and distance from the sweep sighting, kept for the whole run and saved to the
            # file. The sweep mixed several objects, so everything is replaced by this one's.
            wmm, hmm = chosen.get("width_mm"), chosen.get("height_mm")
            eye.widths, eye.heights = ([wmm] * 3 if wmm else []), ([hmm] * 3 if hmm else [])    # seeds the size estimates too
            memory.world, memory.first_world = chosen["world"].copy(), chosen["world"].copy()
            memory.width, memory.height = wmm, hmm
            memory.weight, memory.sightings, memory.last_seen = 1.0 / max(chosen["depth_mm"], 100.0) ** 2, 1, time.time()
            memory.save()
            eye.expected = memory.world
            say(f"  committed the {eye.label} to memory: world x{memory.world[0]:.0f} y{memory.world[1]:.0f} z{memory.world[2]:.0f} mm, "
                f"{chosen['depth_mm'] / 10:.0f} cm from the camera, ~{(wmm or 0):.0f} wide x {(hmm or 0):.0f} tall mm (saved to {os.path.basename(args.memory_file)})", flush=True)
            for _ in range(3):
                t, _ = await next_target(None, 0)
                if t:
                    return t
                await asyncio.sleep(0.2)
            say("  cannot see it from the restored pose: continuing with the position seen during the sweep", flush=True)
            return chosen

        async def find_again(known):
            """INITIAL search only (at the start of the run): search; if the first pass finds nothing, go back to the ORIGINAL pose, detect
            again, and search once more. Losing the object mid-run uses the nearby search (reacquire) and does not go back to the start."""
            for attempt in (1, 2):
                t = await (reacquire(known) if known is not None else survey())
                if t:
                    return t
                say(f"not found on search pass {attempt}: returning to the original pose and looking again", flush=True)
                await arm.move_to_joint_positions(JointPositions(values=home))
                eye.lock.unlock()
                if known is None:
                    eye.expected = None
                for _ in range(3):
                    t, _ = await next_target(None, 0)
                    if t:
                        say("  found it from the original pose", flush=True)
                        return t
                    await asyncio.sleep(0.3)
            return None

        # ---- look first (dry run: keep looking, so you can move the bottle and watch the target follow) ----
        tgt, misses, prev = None, 0, None
        for i in range(args.looks if not real else 5):
            tgt, misses = await next_target(prev, misses)
            gp = await gripper_pose()
            gpos = np.array([gp.x, gp.y, gp.z])
            if tgt:
                vec = tgt["world"] - gpos
                say(f"[look {i + 1}] {args.object} at world x{tgt['world'][0]:.0f} y{tgt['world'][1]:.0f} z{tgt['world'][2]:.0f} mm, "
                      f"{tgt['depth_mm'] / 10:.0f} cm from the camera, ~{tgt['width_mm']:.0f} wide x {tgt['height_mm']:.0f} tall mm; "
                      f"{np.linalg.norm(vec) / 10:.0f} cm from the gripper; next waypoint "
                      f"{np.round(gpos + unit(vec) * min(np.linalg.norm(vec) - args.standoff_mm, args.step_mm))}", flush=True)
                prev = tgt["world"]
            else:
                say(f"[look {i + 1}] no {args.object} seen", flush=True)
            await asyncio.sleep(0.5)
        hold = args.hold_mm if args.hold_mm is not None else args.tip_ahead_mm / 2.0
        if vox is not None and tgt:
            box = vox.object_box(tgt["world"], radius_mm=250.0, min_z_mm=args.table_top_mm + 40.0)
            say(f"voxel map: {int((vox.L > 1.0).sum())} occupied voxels; " + ("nothing above the table near the object yet" if box is None else
                f"object box near it: centre x{box[0][0]:.0f} y{box[0][1]:.0f} z{box[0][2]:.0f}, size {box[1][0]:.0f} x {box[1][1]:.0f} x {box[1][2]:.0f} mm"), flush=True)
        not_in_view = tgt is None
        known0 = (eye.expected if not_in_view else tgt["world"])      # None: nothing seen and nothing remembered
        if not_in_view and not real:
            raise SystemExit(f"no {args.object} with a depth reading in view (a real run would look around: sweep joint {args.survey_joint} +/-{args.survey_deg:.0f} deg)")
        if known0 is not None:
            reach = float(np.hypot(*known0[:2]))
            say(f"the object is {reach:.0f} mm from the base axis (arm reach ~850 mm); width estimate {eye.width_estimate()} mm", flush=True)
            if reach > 1000:
                if not real:
                    raise SystemExit(f"{reach:.0f} mm from the base axis: far beyond the arm's reach")
                say(f"the first sighting is {reach:.0f} mm out, beyond reach: the sweep will look for a nearer one", flush=True)
            if reach > args.max_reach_mm:
                say(f"warning: the world estimate ({reach:.0f} mm) is past {args.max_reach_mm:.0f} mm, but it shifts ~10 cm with the viewpoint; "
                      "the approach is measured in the gripper frame and the motion planner refuses anything truly unreachable", flush=True)
        if not real:
            say("DRY RUN: nothing moved.")
            return
        if not args.ui and not await confirm(("The arm WILL start from where it is (no sweep), " if args.no_sweep else
                       f"The arm WILL first SWEEP joint {args.survey_joint} (up to {args.survey_deg:.0f} degrees each way) to find the nearest {args.object}, ")
                       + "then make planned moves toward it, re-planning every step. E-stop in reach?"):
            return
        if args.no_sweep:
            tgt = tgt if not_in_view is False else None
            if tgt is None and known0 is not None:
                tgt = await reacquire(known0)                      # not in view from here: search nearby around the remembered spot
        else:
            show("sweeping", f"sweeping joint {args.survey_joint} to find the nearest " + eye.label)
            tgt = await sweep_survey()                            # ALWAYS: positions change between runs
        if not tgt:                                               # nothing within reach from the sweep: level the gripper, wrist scan,
            tgt = await find_again(None)                          # wider base sweep, back to the original pose, once more
            if not tgt:
                raise SystemExit("could not find a " + args.object + " by looking around; stopping")

        # ---- stage B (now FIRST): flatten with ONE wrist joint (4 or 5, whichever flattens it from this pose; never 6) ----
        if abs((await gripper_pose()).o_z) < 0.08:
            say("the gripper is already level: skipping the flatten stage", flush=True)
        else:
            candidates = [args.wrist_joint] if args.wrist_joint else [4, 5]
            if not await confirm(f"FLATTEN: rotate joint {' or '.join(str(c) for c in candidates)} only (whichever levels the gripper; never 6), "
                           f"in 10-degree steps up to {args.wrist_deg:.0f}."):
                return
            await flatten()

        await set_roll()                                          # the pick happens at the taught gripper roll (joint 6 near -360)

        # ---- stage A: approach the pre-grasp pose in short re-planned steps ----
        show("approaching", "approaching the pre-grasp pose")
        await gripper.do_command({"set": 840.0})
        j6_start = (await arm.get_joint_positions()).values[5]
        j6_ref["v"] = j6_start
        prev, misses, known = None, 0, tgt["world"]
        if args.approach == "horizontal":
            # HORIZONTAL approach, like a person reaching for a cup: the gripper is already level (stage B ran first). Get to the object's HEIGHT and
            # SIDEWAYS line while still far away, then go in straight along the gripper's own level axis. Never a diagonal dive at it.
            for cycle in range(1, args.max_cycles + 1):
                tgt, misses = await next_target(prev, misses)
                if tgt:
                    prev, known = tgt["world"], memory.world
                elif misses > 2:
                    tgt = await reacquire(known)
                    if not tgt:
                        raise SystemExit("could not find the object again after searching nearby; stopping")
                    misses, prev, known = 0, tgt["world"], memory.world
                rel = await rel_of(tgt, known)
                gp = await gripper_pose()
                g = np.array([gp.x, gp.y, gp.z])
                f = np.array([gp.o_x, gp.o_y, 0.0])
                f = unit(f) if np.linalg.norm(f) > 1e-3 else np.array([1.0, 0.0, 0.0])           # the gripper's level forward direction
                p_ = (await robot.transform_pose(PoseInFrame(reference_frame=args.gripper, pose=Pose(x=float(rel[0]), y=float(rel[1]), z=float(rel[2]))),
                                                 "world")).pose
                ow = np.array([p_.x, p_.y, p_.z])                                                   # the object, from the SAME reading that drives the move
                v = ow - g
                ahead = float(v @ f)
                side = v - ahead * f
                side[2] = 0.0                                                                       # sideways error, horizontal
                vert = float(max(ow[2], args.min_z_mm) - g[2])                                      # height error (never below the table-safe floor)
                off = np.array([side[0], side[1], vert])
                if np.linalg.norm(off) < args.line_tol_mm and abs(ahead - args.standoff_mm) < args.tol_mm:
                    say("level with the object and lined up at the pre-grasp distance", flush=True)
                    break
                if ahead < args.standoff_mm * 0.8:                                                  # too close to line up: back off along the axis first
                    step_vec = -f * min(args.step_mm, args.standoff_mm - ahead)
                elif np.linalg.norm(off) > args.line_tol_mm:                                        # line up: height and sideways, no forward motion
                    step_vec = off * min(1.0, args.step_mm / np.linalg.norm(off))
                else:                                                                               # lined up: go straight in
                    step_vec = f * float(np.clip(ahead - args.standoff_mm, -args.step_mm, args.step_mm))
                if not await go(g + step_vec, gp, obstacle_rel=rel):
                    say("  that step is not reachable / blocked: skipping it", flush=True)
                say(f"[A {cycle}] {ahead:.0f} mm ahead, {np.hypot(side[0], side[1]):.0f} mm sideways, {vert:+.0f} mm in height"
                    + ("" if tgt else " (last known: out of view)"), flush=True)
            else:
                raise SystemExit("did not reach the pre-grasp pose within the cycle limit")
        else:
            for cycle in range(1, args.max_cycles + 1):
                tgt, misses = await next_target(prev, misses)
                if tgt:
                    prev, known = tgt["world"], memory.world       # remembered position (weighted by the closest sightings)
                elif misses > 2:                               # still lost after a few tries: go and look for it
                    tgt = await reacquire(known)
                    if not tgt:
                        raise SystemExit("could not find the object again after searching nearby; stopping")
                    misses, prev, known = 0, tgt["world"], memory.world
                rel = await rel_of(tgt, known)                 # object in the gripper frame (a stationary object: last known if out of view)
                pre_g = rel - unit(rel) * args.standoff_mm       # stop standoff_mm short of it, along the line to it
                if np.linalg.norm(pre_g) < args.tol_mm:
                    say("at the pre-grasp pose", flush=True)
                    break
                await go_rel(unit(pre_g) * min(np.linalg.norm(pre_g), args.step_mm), obstacle_rel=rel)
                say(f"[A {cycle}] {np.linalg.norm(pre_g):.0f} mm to the pre-grasp pose; object {rel[2]:.0f} mm ahead, {np.hypot(rel[0], rel[1]):.0f} mm off-axis"
                      + ("" if tgt else " (last known: out of view)"), flush=True)
            else:
                raise SystemExit("did not reach the pre-grasp pose within the cycle limit")
        d6 = (await arm.get_joint_positions()).values[5] - j6_start
        say(f"joint 6 changed {d6:+.1f} deg during the approach", flush=True)
        if abs(d6) > 90:
            raise SystemExit("joint 6 turned a long way; stopping here")

        # ---- stage C: advance along the gripper axis, correcting sideways, still re-planning ----
        show("advancing", "aligning and advancing onto the object")
        if not await confirm("ADVANCE along the gripper axis onto the object (re-planned every step), then CLOSE and lift."):
            return
        eye.reset()
        peek = {"used": 0.0}
        misses, last = 0, memory.world if memory.world is not None else tgt["world"]
        fails = 0
        for cycle in range(1, args.max_cycles + 1):
            if flags["grip"]:                                     # you said "grip": the gripper is at the cup, stop advancing
                break
            if fails >= 3:                                        # it cannot plan any closer: ask the person instead of crashing
                say("I cannot move any closer from here by myself. If the gripper is at the cup, say 'grip' to close now, or 'stop'.", flush=True)
                t0 = time.monotonic()
                while not flags["grip"] and not flags["stop"] and time.monotonic() - t0 < 120:
                    check_stop()
                    await asyncio.sleep(0.3)
                fails = 0
                continue
            t, misses = await next_target(last, misses)
            gp = await gripper_pose()
            gpos, axis = np.array([gp.x, gp.y, gp.z]), unit([gp.o_x, gp.o_y, gp.o_z])
            if t:
                last = memory.world
            elif misses > 3 and np.linalg.norm(last - gpos) > 200:
                t = await reacquire(last)
                if not t:
                    raise SystemExit("lost the object while it was still far away and could not find it again; stopping")
                misses, last = 0, memory.world
                gp = await gripper_pose()
                gpos, axis = np.array([gp.x, gp.y, gp.z]), unit([gp.o_x, gp.o_y, gp.o_z])
            if not t and peek["used"] < args.peek_down_mm:  # out of view: a level gripper looks straight ahead, over the bottle; lower and look again
                step = min(10.0, args.peek_down_mm - peek["used"])
                await go_down(step)
                peek["used"] += step
                say(f"[C {cycle}] object out of view: lowered {step:.0f} mm ({peek['used']:.0f}/{args.peek_down_mm:.0f}) to look for it again", flush=True)
                continue
            rel = await rel_of(t, last)                    # gripper frame; dead-reckons from the last known spot when out of view
            along, lat = float(rel[2]), float(np.hypot(rel[0], rel[1]))
            if not t and lat > args.blind_lat_mm:          # the remembered position is implausibly far off the gripper axis
                raise SystemExit(f"the remembered object is {lat:.0f} mm off the gripper axis (> {args.blind_lat_mm:.0f}): the memory looks wrong; stopping.")
            cap = lambda v: float(np.clip(0.7 * v, -args.lat_step_mm, args.lat_step_mm))
            if lat > args.align_tol_mm:                    # ALIGN FIRST: sideways/vertical only, and stay clear of the object
                back = max(args.clearance_mm - along, 0.0)
                fails = 0 if await go_rel(np.array([cap(rel[0]), cap(rel[1]), -back]), obstacle_rel=rel) else fails + 1
                say(f"[C {cycle}] aligning: {lat:.0f} mm off-axis, {along:.0f} mm ahead" + (f", backing off {back:.0f} mm" if back else "")
                      + ("" if t else " (out of view)"), flush=True)
                continue
            if along <= hold + args.grasp_tol_mm:
                break
            adv = min(args.step_mm, along - hold)
            fails = 0 if await go_rel(np.array([cap(rel[0]), cap(rel[1]), adv]), touch_ok=along <= args.clearance_mm + args.step_mm, obstacle_rel=rel) else fails + 1
            say(f"[C {cycle}] {along:.0f} mm along the axis, {lat:.0f} mm off to the side" + ("" if t else " (out of view)"), flush=True)
        else:
            raise SystemExit("did not reach the object within the cycle limit")
        # ---- recentre: the jaws must straddle the cup's centre, so correct the sideways/vertical error (no forward motion) until it is small ----
        for k in range(1, args.recentre_tries + 1):
            if flags["grip"]:
                break
            t_c, _ = await next_target(last, 0)
            if not t_c:
                say("  recentre: cup not visible from here, closing on the last alignment", flush=True)
                break
            r_c = t_c["rel_mm"]
            off = np.array([float(r_c[0]), float(r_c[1])])
            if np.linalg.norm(off) <= args.recentre_tol_mm:
                say(f"  recentred: {np.linalg.norm(off):.0f} mm off the gripper axis", flush=True)
                break
            step = off * min(1.0, args.lat_step_mm / np.linalg.norm(off))
            say(f"  recentring [{k}/{args.recentre_tries}]: {off[0]:+.0f} / {off[1]:+.0f} mm off the gripper axis", flush=True)
            await go_rel(np.array([step[0], step[1], 0.0]), touch_ok=True, obstacle_rel=r_c)
        # ---- re-check before closing: a person may have moved it ----
        t, _ = (None, 0) if flags["grip"] else await next_target(last, 0)
        gp = await gripper_pose()
        gpos, axis = np.array([gp.x, gp.y, gp.z]), unit([gp.o_x, gp.o_y, gp.o_z])
        if t:
            rel = t["rel_mm"]
            along, lat = float(rel[2]), float(np.hypot(rel[0], rel[1]))
            say(f"final check: {along:.0f} mm along, {lat:.0f} mm to the side", flush=True)
            if lat > args.lat_tol_mm or along > hold + 45:
                await retry_attempt("the object is no longer lined up with the gripper (moved?): not closing")
        width = args.grip_width_mm or (None if eye.width_estimate() is None else eye.width_estimate() + args.grip_trim_mm)
        if width is None or not 25.0 <= width <= 80.0:
            raise SystemExit(f"cannot size the grip (estimated width {width}); pass --grip-width-mm. Not closing.")
        if not flags["grip"] and not await confirm(f"CLOSE the gripper to {width:.0f} mm (position {width * 10:.0f}, no force) and lift {args.lift_mm:.0f} mm."):
            return
        table_spot = memory.world.copy() if memory.world is not None else (t["world"].copy() if t else None)
        await gripper.do_command({"set_gripper_speed": 300})
        await gripper.do_command({"set": float(width * 10)})
        await asyncio.sleep(2.5)
        closed = await gripper_pos()
        say(f"gripper position: {closed} (commanded {width * 10:.0f}; {args.open_pos:.0f} = fully open)", flush=True)
        runlog.metric("gripper_closed_pos", closed)
        if args.grasp_check and closed is not None and closed >= args.open_pos - 40:
            await retry_attempt(f"the fingers did not close (position {closed:.0f}): the gripper is not working or is blocked open")
        gp = await gripper_pose()
        await go(np.array([gp.x, gp.y, gp.z + args.lift_mm]), gp)
        say("lifted.")
        if args.grasp_check and table_spot is not None:          # the cup must have left the table with the gripper
            eye.reset()
            seen_again = None
            for _ in range(3):
                seen_again, _m = await next_target(table_spot, 0)
                if seen_again:
                    break
                await asyncio.sleep(0.2)
            if seen_again is not None:
                moved = float(np.linalg.norm(seen_again["world"][:2] - table_spot[:2]))
                dz = float(seen_again["world"][2] - table_spot[2])
                runlog.metric("after_lift_cup_offset_mm", [round(moved, 1), round(dz, 1)])
                if moved < 80.0 and dz < 40.0:
                    await retry_attempt(f"the {eye.label} is still on the table where it was ({moved:.0f} mm away, {dz:+.0f} mm in height): nothing was lifted")
            else:
                say("  the cup is not on the table any more: taking it as picked up", flush=True)
        show("lifted", "lifted")
        if not args.no_handover:
            await hand_over()
        await asyncio.sleep(2.0)
    finally:
        try:
            memory.save()
        except Exception:
            pass
        try:
            runlog.metric("width_mm", memory.width)
            runlog.metric("object_world_mm", None if memory.world is None else [round(float(v), 1) for v in memory.world])
            runlog.finish(sys.exc_info()[1])
            say(f"run log: {runlog.save()}  (success={runlog.doc['success']}, class={runlog.doc['failure_class']})")
            if os.environ.get("MONGODB_URI") and not args.no_mongo:
                push_mongo(runlog.doc, os.environ["MONGODB_URI"], args.mongo_db, args.mongo_coll)
        except Exception as e:
            print("run log failed:", e, flush=True)
        if real:
            try:
                await arm.stop()
            except Exception as e:
                print("stop failed:", e)
        await robot.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viam-address", required=True)
    ap.add_argument("--object", default="bottle", help="what to pick up (a YOLO class name)")
    ap.add_argument("--camera", default="cam")
    ap.add_argument("--arm", default="arm")
    ap.add_argument("--gripper", default="gripper")
    ap.add_argument("--sink", choices=["sim", "viam"], default="sim", help="sim = dry run, nothing moves")
    ap.add_argument("--memory-file", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "object_memory.json"),
                    help="where the object's remembered position and size are saved")
    ap.add_argument("--use-memory", action="store_true", help="start from the remembered object position/size in --memory-file (it may be stale)")
    ap.add_argument("--ui", action="store_true", help="serve the local web page: live camera view, status, chat ('pick up the cup'), STOP; "
                    "yes/no in the chat answer the confirmations")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="with --ui: do not open the page automatically")
    ap.add_argument("--sweep-pose", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sweep_pose.json"),
                    help="joint pose (JSON) the sweep starts from: gripper vertical, camera looking straight down at the table")
    ap.add_argument("--down-joint", type=int, default=5, help="joint that points the gripper down for the sweep (5 = wrist pitch; 4 rolls the "
                    "gripper, so it is left alone)")
    ap.add_argument("--no-sweep", action="store_true", help="skip the initial sweep: start from where the arm is (use with --use-memory when "
                    "the object is already close)")
    ap.add_argument("--handover-speed-dps", type=float, default=15.0, help="arm speed (deg/s) while holding the object out to a person")
    ap.add_argument("--arm-speed-dps", type=float, default=50.0, help="arm speed (deg/s) restored after the handover")
    ap.add_argument("--handover-wait-s", type=float, default=60.0, help="how long to wait for the person to take it, then put it down")
    ap.add_argument("--pull-hold-s", type=float, default=0.3, help="the pull must last this long before the gripper opens")
    ap.add_argument("--handover-settle-s", type=float, default=0.8, help="pause after opening, before backing away")
    ap.add_argument("--no-mongo", action="store_true", help="do not send the run log to MongoDB even if MONGODB_URI is set")
    ap.add_argument("--mongo-db", default="robot48")
    ap.add_argument("--mongo-coll", default="pick_runs")
    ap.add_argument("--no-handover", action="store_true", help="do not hand the object to a person after lifting it")
    ap.add_argument("--handover-xyz", default="500,0,350", help="where to hold it out when no person is seen (world mm)")
    ap.add_argument("--handover-gap-mm", type=float, default=450.0, help="stay this far short of the person")
    ap.add_argument("--handover-max-reach-mm", type=float, default=650.0, help="never hold it out further than this from the base axis")
    ap.add_argument("--handover-z-mm", type=float, default=350.0, help="height to hold it out at")
    ap.add_argument("--pull-load", type=float, default=1.0, help="joint-load change (arm 'load' units) read as a person pulling the object")
    ap.add_argument("--survey-joint", type=int, default=2, help="joint rotated for the INITIAL sweep, Viam app numbering (2 = shoulder: "
                    "sweeps the camera near/far over the table; 1 = base: left/right)")
    ap.add_argument("--survey-deg", type=float, default=20.0, help="initial look-around: sweep that joint out to +/- this many degrees")
    ap.add_argument("--survey-step-deg", type=float, default=10.0, help="initial look-around: base step between looks")
    ap.add_argument("--voxel", action="store_true", help="build a live voxel map from the depth frames (it follows an object a person moves) and give the "
                    "planner the cup's real extent from it as the obstacle")
    ap.add_argument("--voxel-stride", type=int, default=8, help="--voxel: use every Nth depth pixel")
    ap.add_argument("--table-top-mm", type=float, default=-23.0, help="--voxel: world height of the table top (the machine config's table box top)")
    ap.add_argument("--near-world", help="x,y,z (mm, world): start with the object nearest this point (to pick a particular one)")
    ap.add_argument("--looks", type=int, default=8, help="dry run: how many times to look (move the object to watch it follow)")
    ap.add_argument("--standoff-mm", type=float, default=250.0, help="pre-grasp distance short of the object (the wrist is flattened here, "
                    "well clear of it: the tip swings by several cm)")
    ap.add_argument("--approach", choices=["horizontal", "direct"], default="horizontal",
                    help="horizontal: level the gripper first, line up with the object's height and side from far away, then go straight in "
                    "(like a person's hand); direct: the old diagonal approach")
    ap.add_argument("--line-tol-mm", type=float, default=30.0, help="--approach horizontal: lined up when the height and sideways errors are within this")
    ap.add_argument("--align-tol-mm", type=float, default=30.0, help="off-axis error above which it only aligns (no forward motion)")
    ap.add_argument("--clearance-mm", type=float, default=120.0, help="stay at least this far ahead of the object while aligning")
    ap.add_argument("--lat-step-mm", type=float, default=25.0, help="longest sideways correction per step")
    ap.add_argument("--blind-lat-mm", type=float, default=400.0, help="out of view it goes to the REMEMBERED position in capped, planned steps; "
                    "refuse if that is further than this off the gripper axis (a wrong memory)")
    ap.add_argument("--tip-ahead-mm", type=float, default=35.0, help="the fingertips are this far ahead of the gripper frame's origin "
                    "(measured: camera -> gripper base 100 mm, camera -> fingertips 135 mm; the frame origin is at the base)")
    ap.add_argument("--hold-mm", type=float, help="final stop: the object's centre this far ahead of the gripper frame's origin "
                    "(default: halfway along the claws = tip-ahead / 2, so the object sits between the fingers)")
    ap.add_argument("--grasp-tol-mm", type=float, default=8.0, help="stop the final advance when within this of the hold point")
    ap.add_argument("--step-mm", type=float, default=100.0, help="longest planned step (shorter = reacts faster to a moved object)")
    ap.add_argument("--tol-mm", type=float, default=30.0)
    ap.add_argument("--lat-tol-mm", type=float, default=25.0, help="max sideways error to still close")
    ap.add_argument("--jump-mm", type=float, default=150.0, help="report a re-target when the object moves this much between looks")
    ap.add_argument("--person-mm", type=float, default=500.0, help="pause if a person is this close to the camera")
    ap.add_argument("--min-z-mm", type=float, default=75.0, help="never plan below this world height: table top (-23, from the machine config) + 80 mm gripper half-height + margin")
    ap.add_argument("--max-reach-mm", type=float, default=800.0, help="refuse objects further than this from the base axis")
    ap.add_argument("--max-cycles", type=int, default=60)
    ap.add_argument("--aim-height-frac", type=float, default=0.5, help="where on the object's height the gripper aims: 0.5 = middle, "
                    "0.85 = near the top, e.g. the neck of a bottle (use with --grip-width-mm = the neck's width)")
    ap.add_argument("--grip-width-mm", type=float, help="override the measured width (the fingers close to this opening)")
    ap.add_argument("--grip-trim-mm", type=float, default=-1.5, help="added to the camera's width estimate before closing: the detector's box "
                    "sits slightly outside the object, so the estimate reads wide (79 mm closed 1-2 mm too loose on the bottle). Negative = tighter")
    ap.add_argument("--wrist-joint", type=int, default=0, help="joint that flattens the gripper (Viam app numbering); 0 = try 4 and 5 and pick "
                    "the one that works from the current pose (never 6)")
    ap.add_argument("--roll-joint", type=int, default=6, help="the final roll joint, Viam app numbering (6)")
    ap.add_argument("--roll-target-deg", type=float, default=0.0, help="roll joint (6) angle for the pick: 0 (the same gripper roll as -360, turned the other way: "
                    "mid-range, so far from the joint limits)")
    ap.add_argument("--roll-tol-deg", type=float, default=10.0, help="leave the roll alone if it is already within this of the target")
    ap.add_argument("--wrist-deg", type=float, default=100.0)
    ap.add_argument("--peek-down-mm", type=float, default=20.0, help="when the object is out of view in the final stage, lower the gripper "
                    "in 10 mm steps by up to this much and look again (a level gripper looks straight ahead, over the object)")
    ap.add_argument("--max-j6-deg", type=float, default=0.0, help="stop if joint 6 (final roll) turns more than this from where the run began (0 = no limit)")
    ap.add_argument("--wrist-sign", type=int, choices=[-1, 0, 1], default=0)
    ap.add_argument("--lift-mm", type=float, default=120.0)
    ap.add_argument("--yes", action="store_true", help="skip the typed confirmations (not recommended)")
    ap.add_argument("--free-within-mm", type=float, default=350.0, help="closer than this to the cup (along the gripper axis), a step the planner "
                    "refuses because of the cup's box is retried without the cup as an obstacle")
    ap.add_argument("--aim-trim-mm", default="0,0,0", help="x,y,z (gripper frame, mm) added to where the cup is measured to be: corrects a "
                    "constant aiming error (x = along the jaw opening, y = the other sideways axis, z = along the gripper axis)")
    ap.add_argument("--recentre-tol-mm", type=float, default=6.0, help="before closing, centre the cup between the jaws to within this")
    ap.add_argument("--recentre-tries", type=int, default=4)
    ap.add_argument("--retries", type=int, default=2, help="after a failed grasp, start the whole pick again this many times (each retry is confirmed)")
    ap.add_argument("--no-grasp-check", dest="grasp_check", action="store_false", help="do not verify the grasp (fingers closed, cup left the table)")
    ap.add_argument("--open-pos", type=float, default=840.0, help="gripper position units when fully open (10 per mm)")
    ap.add_argument("--max-sight-mm", type=float, default=1000.0, help="ignore sightings farther than this from the base axis (false detections)")
    ap.add_argument("--confirm-sightings", type=int, default=2, help="the first sighting must repeat this many looks in a row before it is trusted (1 = off)")
    args = ap.parse_args()
    args.attempt = 0
    try:
        while True:
            try:
                asyncio.run(main_async(args))
                break
            except GraspFailed as e:
                args.attempt += 1
                print(f"retrying ({e}): attempt {args.attempt + 1} of {args.retries + 1}", flush=True)
    except KeyboardInterrupt:
        print("interrupted")


if __name__ == "__main__":
    main()
