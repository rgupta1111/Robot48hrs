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
from viam.proto.common import Pose, PoseInFrame  # noqa: E402
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


class Perceiver:
    """One call -> where the tracked object is right now (world frame), how big it is, and whether a person is close."""

    def __init__(self, robot, camera_name, cam, intr, model, dargs, label, person_mm):
        self.robot, self.camera_name, self.cam, self.intr = robot, camera_name, cam, intr
        self.model, self.dargs, self.label, self.person_mm = model, dargs, label, person_mm
        self.lock, self.widths, self.heights = Lock(max_jump_frac=0.5, lost_after=10 ** 6), [], []

    def reset(self):
        self.lock.unlock()
        self.widths, self.heights = [], []

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
        if not self.lock.locked:                         # choose the NEAREST one first (by depth), then stay on it
            cands = sorted(cands, key=lambda c: box_depth_mm(dmap, c[:4]) or 1e9)
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
        cam_pt = Pose(x=(u - cx) * zc / fx, y=(v - cy) * zc / fy, z=zc)
        wp = (await self.robot.transform_pose(PoseInFrame(reference_frame=self.camera_name, pose=cam_pt), "world")).pose
        return {"seen": True, "world": np.array([wp.x, wp.y, wp.z]), "depth_mm": z, "width_mm": width, "height_mm": height,
                "person_close": person_close}

    def width_estimate(self):
        return float(np.median(self.widths)) if len(self.widths) >= 3 else None


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
        eye = Perceiver(robot, args.camera, cam, intr, model, dargs, args.object, args.person_mm)

        def confirm(q):
            return args.yes or input(f"{q} Type 'yes' to continue: ").strip().lower() == "yes"

        async def gripper_pose():
            return (await robot.transform_pose(PoseInFrame(reference_frame=args.gripper, pose=Pose()), "world")).pose

        def at(pos, gp):                                  # a world pose with the gripper's current orientation
            return Pose(x=float(pos[0]), y=float(pos[1]), z=float(max(pos[2], args.min_z_mm)),
                        o_x=gp.o_x, o_y=gp.o_y, o_z=gp.o_z, theta=gp.theta)

        async def go(pos, gp):
            await motion.move(args.gripper, PoseInFrame(reference_frame="world", pose=at(pos, gp)))   # this SDK wants the name as a string

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
        if not tgt:
            raise SystemExit(f"no {args.object} with a depth reading in view")
        reach = float(np.hypot(*tgt["world"][:2]))
        print(f"the object is {reach:.0f} mm from the base axis (arm reach ~850 mm); width estimate {eye.width_estimate()} mm", flush=True)
        if reach > args.max_reach_mm:
            raise SystemExit(f"{reach:.0f} mm is beyond --max-reach-mm {args.max_reach_mm:.0f}: not reachable from here")
        if not real:
            print("DRY RUN: nothing moved.")
            return
        if not confirm("The arm WILL make planned moves toward the nearest object, re-planning every step. E-stop in reach?"):
            return

        # ---- stage A: approach the pre-grasp pose in short re-planned steps ----
        await gripper.do_command({"set": 840.0})
        j6_start = (await arm.get_joint_positions()).values[5]
        prev, misses = None, 0
        for cycle in range(1, args.max_cycles + 1):
            tgt, misses = await next_target(prev, misses)
            if not tgt:
                if misses > 5:
                    raise SystemExit("lost the object for too long; stopping")
                await asyncio.sleep(0.2)
                continue
            prev = tgt["world"]
            gp = await gripper_pose()
            gpos = np.array([gp.x, gp.y, gp.z])
            pre = tgt["world"] - unit(tgt["world"] - gpos) * args.standoff_mm
            err = pre - gpos
            if np.linalg.norm(err) < args.tol_mm:
                print("at the pre-grasp pose", flush=True)
                break
            await go(gpos + unit(err) * min(np.linalg.norm(err), args.step_mm), gp)
            print(f"[A {cycle}] {np.linalg.norm(err):.0f} mm to the pre-grasp pose; object at x{tgt['world'][0]:.0f} y{tgt['world'][1]:.0f} z{tgt['world'][2]:.0f}", flush=True)
        else:
            raise SystemExit("did not reach the pre-grasp pose within the cycle limit")
        d6 = (await arm.get_joint_positions()).values[5] - j6_start
        print(f"joint 6 changed {d6:+.1f} deg during the approach", flush=True)
        if abs(d6) > 90:
            raise SystemExit("joint 6 turned a long way; stopping here")

        # ---- stage B: flatten with one wrist joint ----
        if not confirm(f"FLATTEN: rotate joint {args.wrist_joint} only, in 10-degree steps (up to {args.wrist_deg:.0f}), until the gripper is level."):
            return
        wj = args.wrist_joint - 1
        j0 = list((await arm.get_joint_positions()).values)
        sign = args.wrist_sign
        if not sign:                                      # 10-degree test each way: which flattens it and points it forward?
            p0, options = await gripper_pose(), []
            for sg in (1, -1):
                t = list(j0)
                t[wj] += sg * 10.0
                await arm.move_to_joint_positions(JointPositions(values=t))
                p1 = await gripper_pose()
                await arm.move_to_joint_positions(JointPositions(values=j0))
                if abs(p1.o_z) < abs(p0.o_z) - 0.02:
                    options.append((sg, p1.o_x > p0.o_x))
            if not options:
                raise SystemExit(f"joint {args.wrist_joint} does not flatten the gripper from here; use --wrist-sign / check the joint")
            sign = next((sg for sg, fwd in options if fwd), options[0][0])
        total = 0.0
        while total < args.wrist_deg:
            total += min(10.0, args.wrist_deg - total)
            t = list(j0)
            t[wj] += sign * total
            await arm.move_to_joint_positions(JointPositions(values=t))
            if abs((await gripper_pose()).o_z) < 0.05:
                break
        gp = await gripper_pose()
        print(f"joint {args.wrist_joint} rotated {sign * total:+.0f} deg; gripper axis ({gp.o_x:+.2f}, {gp.o_y:+.2f}, {gp.o_z:+.2f})", flush=True)

        # ---- stage C: advance along the gripper axis, correcting sideways, still re-planning ----
        if not confirm("ADVANCE along the gripper axis onto the object (re-planned every step), then CLOSE and lift."):
            return
        eye.reset()
        misses, last = 0, tgt["world"]
        for cycle in range(1, args.max_cycles + 1):
            t, misses = await next_target(last, misses)
            gp = await gripper_pose()
            gpos, axis = np.array([gp.x, gp.y, gp.z]), unit([gp.o_x, gp.o_y, gp.o_z])
            if t:
                last = t["world"]
            elif misses > 3 and np.linalg.norm(last - gpos) > 200:
                raise SystemExit("lost the object while it was still far away; stopping")
            rel = last - gpos                              # dead-reckon on the last known position when it is out of view
            along = float(rel @ axis)
            lateral = rel - along * axis
            if along <= args.hold_mm + args.tol_mm:
                break
            adv = min(args.step_mm, along - args.hold_mm)
            await go(gpos + axis * adv + lateral * 0.7, gp)
            print(f"[C {cycle}] {along:.0f} mm along the axis, {np.linalg.norm(lateral):.0f} mm off to the side" + ("" if t else " (out of view)"), flush=True)
        else:
            raise SystemExit("did not reach the object within the cycle limit")
        # ---- re-check before closing: a person may have moved it ----
        t, _ = await next_target(last, 0)
        gp = await gripper_pose()
        gpos, axis = np.array([gp.x, gp.y, gp.z]), unit([gp.o_x, gp.o_y, gp.o_z])
        if t:
            rel = t["world"] - gpos
            along, lat = float(rel @ axis), float(np.linalg.norm(rel - (rel @ axis) * axis))
            print(f"final check: {along:.0f} mm along, {lat:.0f} mm to the side", flush=True)
            if lat > args.lat_tol_mm or along > args.hold_mm + 60:
                raise SystemExit("the object is no longer lined up with the gripper (moved?): not closing")
        width = args.grip_width_mm or eye.width_estimate()
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
    ap.add_argument("--looks", type=int, default=8, help="dry run: how many times to look (move the object to watch it follow)")
    ap.add_argument("--standoff-mm", type=float, default=150.0, help="pre-grasp distance short of the object")
    ap.add_argument("--hold-mm", type=float, default=0.0, help="stop this far short of the object centre along the gripper axis")
    ap.add_argument("--step-mm", type=float, default=100.0, help="longest planned step (shorter = reacts faster to a moved object)")
    ap.add_argument("--tol-mm", type=float, default=30.0)
    ap.add_argument("--lat-tol-mm", type=float, default=25.0, help="max sideways error to still close")
    ap.add_argument("--jump-mm", type=float, default=150.0, help="report a re-target when the object moves this much between looks")
    ap.add_argument("--person-mm", type=float, default=500.0, help="pause if a person is this close to the camera")
    ap.add_argument("--min-z-mm", type=float, default=75.0, help="never plan below this world height: table top (-23, from the machine config) + 80 mm gripper half-height + margin")
    ap.add_argument("--max-reach-mm", type=float, default=800.0, help="refuse objects further than this from the base axis")
    ap.add_argument("--max-cycles", type=int, default=60)
    ap.add_argument("--grip-width-mm", type=float, help="override the measured width (the fingers close to this opening)")
    ap.add_argument("--wrist-joint", type=int, default=4, help="joint that flattens the gripper (Viam app numbering)")
    ap.add_argument("--wrist-deg", type=float, default=90.0)
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
