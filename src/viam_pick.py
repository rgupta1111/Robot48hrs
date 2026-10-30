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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from viam.components.arm import Arm, JointPositions  # noqa: E402
from viam.components.camera import Camera  # noqa: E402
from viam.components.gripper import Gripper  # noqa: E402
from viam.proto.common import Geometry, GeometriesInFrame, Pose, PoseInFrame, RectangularPrism, Vector3, WorldState  # noqa: E402
from viam.robot.client import RobotClient  # noqa: E402
from viam.rpc.dial import DialOptions  # noqa: E402
from viam.services.motion import MotionClient  # noqa: E402

from local_detect import Lock, detect, load_detector, object_candidates  # noqa: E402
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


class Perceiver:
    """One call -> where the tracked object is right now (world frame), how big it is, and whether a person is close."""

    def __init__(self, robot, camera_name, gripper_name, cam, intr, model, dargs, label, person_mm, aim_frac=0.5, memory=None):
        self.memory = memory
        self.aim_frac = aim_frac     # where along the object's height to aim: 0.5 = its middle, 0.85 = near the top (the neck of a bottle)
        self.robot, self.camera_name, self.gripper_name, self.cam, self.intr = robot, camera_name, gripper_name, cam, intr
        self.model, self.dargs, self.label, self.person_mm = model, dargs, label, person_mm
        self.lock, self.widths, self.heights = Lock(max_jump_frac=0.5, lost_after=10 ** 6), [], []
        self.expected = None          # world position of the object we are after: re-locking picks the candidate NEAREST TO IT
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
        dets = await asyncio.to_thread(detect, self.model, color, self.dargs)          # YOLO on the GPU
        person_close = False
        for d in dets:
            if d[5] == "person":
                pd = box_depth_mm(dmap, d[:4])
                person_close = person_close or bool(pd and pd < self.person_mm)
        cands = object_candidates(dets, self.label)
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
        aim_up = (self.aim_frac - 0.5) * float(np.median(self.heights)) if (self.aim_frac != 0.5 and len(self.heights) >= 3) else 0.0
        if aim_up:                                        # aim higher/lower on the object (world z is up), then express it in the gripper frame
            wp.z += aim_up
            rel = (await self.robot.transform_pose(PoseInFrame(reference_frame="world", pose=Pose(x=wp.x, y=wp.y, z=wp.z)), self.gripper_name)).pose
        else:
            rel = (await self.robot.transform_pose(PoseInFrame(reference_frame=self.camera_name, pose=cam_pt), self.gripper_name)).pose
        self.expected = np.array([wp.x, wp.y, wp.z])      # follow this object from now on
        result = {"seen": True, "world": np.array([wp.x, wp.y, wp.z]), "rel_mm": np.array([rel.x, rel.y, rel.z]), "depth_mm": z, "width_mm": width, "height_mm": height,
                "person_close": person_close}
        if self.memory is not None:
            self.memory.update(result, self.width_estimate(), float(np.median(self.heights)) if len(self.heights) >= 3 else None)
            self.expected = self.memory.world             # follow the remembered object, not just the last sighting
        return result

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
        memory = ObjectMemory(args.memory_file, args.object)
        if args.use_memory and memory.load():
            print(f"loaded the remembered {args.object}: world {np.round(memory.world)}, width {memory.width}", flush=True)
        eye = Perceiver(robot, args.camera, args.gripper, cam, intr, model, dargs, args.object, args.person_mm, args.aim_height_frac, memory)
        if memory.world is not None:
            eye.expected = memory.world
        if args.near_world:
            eye.expected = np.array([float(v) for v in args.near_world.split(",")])

        def confirm(q):
            return args.yes or input(f"{q} Type 'yes' to continue: ").strip().lower() == "yes"

        async def gripper_pose():
            return (await robot.transform_pose(PoseInFrame(reference_frame=args.gripper, pose=Pose()), "world")).pose

        def at(pos, gp):                                  # a world pose with the gripper's current orientation
            return Pose(x=float(pos[0]), y=float(pos[1]), z=float(max(pos[2], args.min_z_mm)),
                        o_x=gp.o_x, o_y=gp.o_y, o_z=gp.o_z, theta=gp.theta)

        async def go(pos, gp):
            await motion.move(args.gripper, PoseInFrame(reference_frame="world", pose=at(pos, gp)))   # this SDK wants the name as a string

        async def rel_of(t, known_world):
            """The object in the GRIPPER's own frame (z = along the gripper axis): straight from the latest reading, or, when it is
            out of view, the last known world position converted by the frame system. Independent of world-frame drift."""
            if t:
                return t["rel_mm"]
            p = (await robot.transform_pose(PoseInFrame(reference_frame="world", pose=Pose(x=float(known_world[0]), y=float(known_world[1]),
                                                                                           z=float(known_world[2]))), args.gripper)).pose
            return np.array([p.x, p.y, p.z])

        async def object_obstacle(rel):
            """The object as a box obstacle for the planner, placed from the SAME gripper-frame measurement that drives the move (so the
            two always agree; the remembered world position can differ from it by ~10 cm). None if there is no measurement."""
            if rel is None:
                return None
            wp = (await robot.transform_pose(PoseInFrame(reference_frame=args.gripper, pose=Pose(x=float(rel[0]), y=float(rel[1]),
                                                                                                 z=float(rel[2]))), "world")).pose
            wd, ht = memory.width or 90.0, memory.height or 220.0      # defaults when its size is not known (its box was clipped)
            g = Geometry(center=Pose(x=wp.x, y=wp.y, z=wp.z), box=RectangularPrism(dims_mm=Vector3(x=wd + 10, y=wd + 10, z=ht + 10)),
                         label=args.object)
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
            """Plan a move of v (mm, gripper frame) from where the gripper is now, keeping its orientation. The remembered object is
            an obstacle for the planner, except for the final advance (touch_ok), where the fingers must straddle it: then it is
            left out (a named collision exception failed: the planner did not know the obstacle by that name)."""
            p = (await robot.transform_pose(PoseInFrame(reference_frame=args.gripper, pose=Pose(x=float(v[0]), y=float(v[1]), z=float(v[2]))),
                                            "world")).pose
            await motion.move(args.gripper, PoseInFrame(reference_frame="world", pose=Pose(
                x=p.x, y=p.y, z=max(p.z, args.min_z_mm), o_x=p.o_x, o_y=p.o_y, o_z=p.o_z, theta=p.theta)),
                world_state=None if touch_ok else await object_obstacle(obstacle_rel))
            await check_j6()

        async def next_target(prev, misses):
            """look; wait while a person is close; returns (target, misses)."""
            while True:
                t = await eye.look()
                if t and t["person_close"]:
                    print("a person is close to the camera/gripper: waiting", flush=True)
                    await asyncio.sleep(1.0)
                    continue
                break
            if not t or not t["seen"]:
                return None, misses + 1
            if prev is not None and np.linalg.norm(t["world"] - prev) > args.jump_mm:
                print(f"  the object moved {np.linalg.norm(t['world'] - prev):.0f} mm: re-targeting", flush=True)
            return t, 0

        async def reacquire(known):
            """Lost it. It has probably not gone anywhere, it just left the picture: back off along the camera axis, then sweep
            the base joint a few degrees each way, looking after every move. Returns a fresh target, or None."""
            print("lost the object: searching nearby ...", flush=True)
            gp = await gripper_pose()
            gpos, axis = np.array([gp.x, gp.y, gp.z]), unit([gp.o_x, gp.o_y, gp.o_z])
            for retreat in (80.0, 160.0):
                await go(gpos - axis * retreat, gp)
                eye.lock.unlock()
                for _ in range(3):
                    t, _ = await next_target(None, 0)
                    if t:
                        print(f"  found it again after backing off {retreat:.0f} mm", flush=True)
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
                        print(f"  found it again after turning the base {d:+.0f} deg", flush=True)
                        return t
                    await asyncio.sleep(0.2)
            await arm.move_to_joint_positions(JointPositions(values=j_here))
            return None

        async def wrist_scan():
            """Look around by moving ONLY the two wrist joints (4 and 5), which pan/tilt the camera a little each way, looking after every
            pose. Returns a target (the arm stays where it saw it) or None (the wrist is put back where it was)."""
            j_here = list((await arm.get_joint_positions()).values)
            print("  scanning with the wrist (joints 4 and 5) ...", flush=True)
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
                        print(f"  found a {args.object} with the wrist at joint 4 {d4:+d}, joint 5 {d5:+d} deg", flush=True)
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
                    t = list(j0)
                    t[jn - 1] += sg * 10.0                        # 10-degree test move, then straight back
                    await arm.move_to_joint_positions(JointPositions(values=t))
                    p1 = await gripper_pose()
                    await arm.move_to_joint_positions(JointPositions(values=j0))
                    gain = abs(p0.o_z) - abs(p1.o_z)              # how much more level it got
                    results.append((gain, p1.o_x > p0.o_x, jn, sg))
                    print(f"  test: joint {jn} {sg * 10:+.0f} deg -> levels by {gain:+.2f}" + (", pointing more forward" if p1.o_x > p0.o_x else ""), flush=True)
            good = [r for r in results if r[0] > 0.02]
            if not good:
                raise SystemExit(f"neither joint {' nor '.join(str(c) for c in candidates)} levels the gripper from here; not rotating")
            _, _, jn, sign = max(good, key=lambda r: (r[1], r[0]))   # prefer pointing forward, then the biggest levelling
            total = 0.0
            while total < args.wrist_deg:
                total += min(10.0, args.wrist_deg - total)
                t = list(j0)
                t[jn - 1] += sign * total
                await arm.move_to_joint_positions(JointPositions(values=t))
                if abs((await gripper_pose()).o_z) < 0.05:
                    break
            gp = await gripper_pose()
            print(f"joint {jn} rotated {sign * total:+.0f} deg; gripper axis ({gp.o_x:+.2f}, {gp.o_y:+.2f}, {gp.o_z:+.2f})", flush=True)

        async def survey():
            """No bottle in view at the start and no remembered position: look around. Sweep the base in 15-degree steps out to
            +/-45 degrees, looking after each move; stop at the first bottle seen (nearest by depth in that view)."""
            print("no bottle in view: looking around ...", flush=True)
            if abs((await gripper_pose()).o_z) >= 0.08:        # exploring is easier with the gripper level: use the two flattening joints
                try:
                    await flatten()
                except SystemExit as e:
                    print(f"  (could not level the gripper for the survey: {e})", flush=True)
            t = await wrist_scan()
            if t:
                return t
            j_here = list((await arm.get_joint_positions()).values)
            for d in (15.0, -15.0, 30.0, -30.0, 45.0, -45.0):
                j = list(j_here)
                j[0] += d
                await arm.move_to_joint_positions(JointPositions(values=j))
                eye.lock.unlock()
                eye.expected = None                          # nothing to match yet: take the nearest in this view
                for _ in range(3):
                    t, _ = await next_target(None, 0)
                    if t:
                        print(f"  found a {args.object} after turning the base {d:+.0f} deg", flush=True)
                        return t
                    await asyncio.sleep(0.2)
            await arm.move_to_joint_positions(JointPositions(values=j_here))
            return None

        # ---- look first (dry run: keep looking, so you can move the bottle and watch the target follow) ----
        tgt, misses, prev = None, 0, None
        for i in range(args.looks if not real else 5):
            tgt, misses = await next_target(prev, misses)
            gp = await gripper_pose()
            gpos = np.array([gp.x, gp.y, gp.z])
            if tgt:
                vec = tgt["world"] - gpos
                print(f"[look {i + 1}] {args.object} at world x{tgt['world'][0]:.0f} y{tgt['world'][1]:.0f} z{tgt['world'][2]:.0f} mm, "
                      f"{tgt['depth_mm'] / 10:.0f} cm from the camera, ~{tgt['width_mm']:.0f} wide x {tgt['height_mm']:.0f} tall mm; "
                      f"{np.linalg.norm(vec) / 10:.0f} cm from the gripper; next waypoint "
                      f"{np.round(gpos + unit(vec) * min(np.linalg.norm(vec) - args.standoff_mm, args.step_mm))}", flush=True)
                prev = tgt["world"]
            else:
                print(f"[look {i + 1}] no {args.object} seen", flush=True)
            await asyncio.sleep(0.5)
        hold = args.hold_mm if args.hold_mm is not None else args.tip_ahead_mm / 2.0
        not_in_view = tgt is None
        known0 = (eye.expected if not_in_view else tgt["world"])      # None: nothing seen and nothing remembered
        if not_in_view and not real:
            raise SystemExit(f"no {args.object} with a depth reading in view (a real run would look around: sweep the base +/-45 deg)")
        if known0 is not None:
            reach = float(np.hypot(*known0[:2]))
            print(f"the object is {reach:.0f} mm from the base axis (arm reach ~850 mm); width estimate {eye.width_estimate()} mm", flush=True)
            if reach > 1000:
                raise SystemExit(f"{reach:.0f} mm from the base axis: far beyond the arm's reach")
            if reach > args.max_reach_mm:
                print(f"warning: the world estimate ({reach:.0f} mm) is past {args.max_reach_mm:.0f} mm, but it shifts ~10 cm with the viewpoint; "
                      "the approach is measured in the gripper frame and the motion planner refuses anything truly unreachable", flush=True)
        if not real:
            print("DRY RUN: nothing moved.")
            return
        if not confirm("The arm WILL make planned moves toward the nearest object, re-planning every step"
                       + ((" (it cannot see it from here, so it will first search nearby: back off, sweep the base)" if known0 is not None else
                           " (no bottle in view, so it will first LOOK AROUND: level the gripper with joints 4/5 if needed, scan with the wrist (joints 4 and 5), then sweep the base left and right, up to 45 degrees)")
                          if not_in_view else "")
                       + ". E-stop in reach?"):
            return
        if not_in_view:
            tgt = await (reacquire(known0) if known0 is not None else survey())
            if not tgt:
                raise SystemExit("could not find a " + args.object + " by looking around; stopping")

        # ---- stage A: approach the pre-grasp pose in short re-planned steps ----
        await gripper.do_command({"set": 840.0})
        j6_start = (await arm.get_joint_positions()).values[5]
        j6_ref["v"] = j6_start
        prev, misses, known = None, 0, tgt["world"]
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
                print("at the pre-grasp pose", flush=True)
                break
            await go_rel(unit(pre_g) * min(np.linalg.norm(pre_g), args.step_mm), obstacle_rel=rel)
            print(f"[A {cycle}] {np.linalg.norm(pre_g):.0f} mm to the pre-grasp pose; object {rel[2]:.0f} mm ahead, {np.hypot(rel[0], rel[1]):.0f} mm off-axis"
                  + ("" if tgt else " (last known: out of view)"), flush=True)
        else:
            raise SystemExit("did not reach the pre-grasp pose within the cycle limit")
        d6 = (await arm.get_joint_positions()).values[5] - j6_start
        print(f"joint 6 changed {d6:+.1f} deg during the approach", flush=True)
        if abs(d6) > 90:
            raise SystemExit("joint 6 turned a long way; stopping here")

        # ---- stage B: flatten with ONE wrist joint (4 or 5, whichever flattens it from this pose; never 6) ----
        if abs((await gripper_pose()).o_z) < 0.08:
            print("the gripper is already level: skipping the flatten stage", flush=True)
        else:
            candidates = [args.wrist_joint] if args.wrist_joint else [4, 5]
            if not confirm(f"FLATTEN: rotate joint {' or '.join(str(c) for c in candidates)} only (whichever levels the gripper; never 6), "
                           f"in 10-degree steps up to {args.wrist_deg:.0f}."):
                return
            await flatten()

        # ---- stage C: advance along the gripper axis, correcting sideways, still re-planning ----
        if not confirm("ADVANCE along the gripper axis onto the object (re-planned every step), then CLOSE and lift."):
            return
        eye.reset()
        peek = {"used": 0.0}
        misses, last = 0, memory.world if memory.world is not None else tgt["world"]
        for cycle in range(1, args.max_cycles + 1):
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
                print(f"[C {cycle}] object out of view: lowered {step:.0f} mm ({peek['used']:.0f}/{args.peek_down_mm:.0f}) to look for it again", flush=True)
                continue
            rel = await rel_of(t, last)                    # gripper frame; dead-reckons from the last known spot when out of view
            along, lat = float(rel[2]), float(np.hypot(rel[0], rel[1]))
            if not t and lat > args.blind_lat_mm:          # the remembered position is implausibly far off the gripper axis
                raise SystemExit(f"the remembered object is {lat:.0f} mm off the gripper axis (> {args.blind_lat_mm:.0f}): the memory looks wrong; stopping.")
            cap = lambda v: float(np.clip(0.7 * v, -args.lat_step_mm, args.lat_step_mm))
            if lat > args.align_tol_mm:                    # ALIGN FIRST: sideways/vertical only, and stay clear of the object
                back = max(args.clearance_mm - along, 0.0)
                await go_rel(np.array([cap(rel[0]), cap(rel[1]), -back]), obstacle_rel=rel)
                print(f"[C {cycle}] aligning: {lat:.0f} mm off-axis, {along:.0f} mm ahead" + (f", backing off {back:.0f} mm" if back else "")
                      + ("" if t else " (out of view)"), flush=True)
                continue
            if along <= hold + args.grasp_tol_mm:
                break
            adv = min(args.step_mm, along - hold)
            await go_rel(np.array([cap(rel[0]), cap(rel[1]), adv]), touch_ok=along <= args.clearance_mm + args.step_mm, obstacle_rel=rel)
            print(f"[C {cycle}] {along:.0f} mm along the axis, {lat:.0f} mm off to the side" + ("" if t else " (out of view)"), flush=True)
        else:
            raise SystemExit("did not reach the object within the cycle limit")
        # ---- re-check before closing: a person may have moved it ----
        t, _ = await next_target(last, 0)
        gp = await gripper_pose()
        gpos, axis = np.array([gp.x, gp.y, gp.z]), unit([gp.o_x, gp.o_y, gp.o_z])
        if t:
            rel = t["rel_mm"]
            along, lat = float(rel[2]), float(np.hypot(rel[0], rel[1]))
            print(f"final check: {along:.0f} mm along, {lat:.0f} mm to the side", flush=True)
            if lat > args.lat_tol_mm or along > hold + 45:
                raise SystemExit("the object is no longer lined up with the gripper (moved?): not closing")
        width = args.grip_width_mm or (None if eye.width_estimate() is None else eye.width_estimate() + args.grip_trim_mm)
        if width is None or not 25.0 <= width <= 80.0:
            raise SystemExit(f"cannot size the grip (estimated width {width}); pass --grip-width-mm. Not closing.")
        if not confirm(f"CLOSE the gripper to {width:.0f} mm (position {width * 10:.0f}, no force) and lift {args.lift_mm:.0f} mm."):
            return
        await gripper.do_command({"set_gripper_speed": 300})
        await gripper.do_command({"set": float(width * 10)})
        await asyncio.sleep(2.5)
        print("gripper position:", await gripper.do_command({"get": True}), flush=True)
        gp = await gripper_pose()
        await go(np.array([gp.x, gp.y, gp.z + args.lift_mm]), gp)
        print("lifted.", flush=True)
    finally:
        try:
            memory.save()
        except Exception:
            pass
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
    ap.add_argument("--near-world", help="x,y,z (mm, world): start with the object nearest this point (to pick a particular one)")
    ap.add_argument("--looks", type=int, default=8, help="dry run: how many times to look (move the object to watch it follow)")
    ap.add_argument("--standoff-mm", type=float, default=250.0, help="pre-grasp distance short of the object (the wrist is flattened here, "
                    "well clear of it: the tip swings by several cm)")
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
    ap.add_argument("--wrist-deg", type=float, default=90.0)
    ap.add_argument("--peek-down-mm", type=float, default=20.0, help="when the object is out of view in the final stage, lower the gripper "
                    "in 10 mm steps by up to this much and look again (a level gripper looks straight ahead, over the object)")
    ap.add_argument("--max-j6-deg", type=float, default=0.0, help="stop if joint 6 (final roll) turns more than this from where the run began (0 = no limit)")
    ap.add_argument("--wrist-sign", type=int, choices=[-1, 0, 1], default=0)
    ap.add_argument("--lift-mm", type=float, default=120.0)
    ap.add_argument("--yes", action="store_true", help="skip the typed confirmations (not recommended)")
    args = ap.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("interrupted")


if __name__ == "__main__":
    main()
