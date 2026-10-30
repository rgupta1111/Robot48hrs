"""Pick the nearest object using Viam's own perception, frame-system and motion libraries.

    vision service (object-segmenter)  -> 3D objects in the camera frame: centre + bounding-box size (width/height from depth)
    frame system  (robot.transform_pose) -> the same point in the WORLD frame
    motion service (builtin)            -> collision-aware planned moves (it knows the table / walls from the machine config)
    gripper (do_command set)            -> close to the object's measured width (position control: no force limit on this gripper)

Plan: translate to a pre-grasp pose (standoff short of the object, current orientation, so no wrist spin) -> flatten the gripper
with ONE wrist joint in small guarded steps -> motion-planned move to the grasp pose -> close to the width -> lift.

Dry run by default (--sink sim): perceives and prints the plan, nothing moves.
    scripts/run.sh src/viam_pick.py --viam-address sfhack16-main.ty1cywh7tq.local.viam.cloud:8080 --object bottle
Real arm (asks for a typed 'yes' before each stage; Ctrl+C stops):
    scripts/run.sh src/viam_pick.py --viam-address ... --object bottle --sink viam
"""
import argparse
import asyncio
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from viam.components.arm import Arm, JointPositions  # noqa: E402
from viam.components.gripper import Gripper  # noqa: E402
from viam.proto.common import Pose, PoseInFrame  # noqa: E402
from viam.robot.client import RobotClient  # noqa: E402
from viam.rpc.dial import DialOptions  # noqa: E402
from viam.services.motion import MotionClient  # noqa: E402
from viam.services.vision import VisionClient  # noqa: E402



def pose_xyz(p):
    return np.array([p.x, p.y, p.z], float)


async def connect(address):
    key, key_id = os.environ.get("VIAM_API_KEY"), os.environ.get("VIAM_API_KEY_ID")
    if not key or not key_id:
        raise SystemExit("set VIAM_API_KEY and VIAM_API_KEY_ID")
    dial = DialOptions.with_api_key(key, key_id)
    if ".local.viam.cloud" in address:                  # direct LAN connection (no relay): several times the frame rate
        dial.disable_webrtc = True
    return await RobotClient.at_address(address, RobotClient.Options(dial_options=dial))


async def perceive(robot, vision, camera, label):
    """-> list of dicts (nearest first): world position, distance, and size from the segmenter's 3D bounding boxes."""
    res = await vision.capture_all_from_camera(camera, return_image=False, return_object_point_clouds=True)
    found = []
    for obj in res.objects or []:
        for g in obj.geometries.geometries:
            if label.lower() not in (g.label or "").lower():
                continue
            c = g.center                                # camera frame, mm, z straight ahead
            dims = g.box.dims_mm if (g.box.dims_mm.x or g.box.dims_mm.y or g.box.dims_mm.z) else None   # betterproto: no HasField
            world = await robot.transform_pose(PoseInFrame(reference_frame=camera, pose=Pose(x=c.x, y=c.y, z=c.z)), "world")
            found.append({"label": g.label, "cam_mm": (c.x, c.y, c.z), "dist_mm": math.sqrt(c.x ** 2 + c.y ** 2 + c.z ** 2),
                          "world_mm": pose_xyz(world.pose),
                          "box_mm": None if dims is None else (dims.x, dims.y, dims.z)})
    return sorted(found, key=lambda o: o["dist_mm"])


async def main_async(args):
    robot = await connect(args.viam_address)
    real = args.sink == "viam"
    try:
        arm, gripper = Arm.from_robot(robot, args.arm), Gripper.from_robot(robot, args.gripper)
        vision, motion = VisionClient.from_robot(robot, args.vision), MotionClient.from_robot(robot, "builtin")

        def confirm(q):
            return args.yes or input(f"{q} Type 'yes' to continue: ").strip().lower() == "yes"

        async def gripper_world():
            return (await robot.transform_pose(PoseInFrame(reference_frame=args.gripper, pose=Pose()), "world")).pose

        objs = await perceive(robot, vision, args.camera, args.object)
        if not objs:
            raise SystemExit(f"no '{args.object}' found by the {args.vision} service")
        for i, o in enumerate(objs):
            b = o["box_mm"]
            print(f"{'nearest ' if i == 0 else '        '}{o['label']}: {o['dist_mm'] / 10:.0f} cm from the camera, world "
                  f"x{o['world_mm'][0]:.0f} y{o['world_mm'][1]:.0f} z{o['world_mm'][2]:.0f} mm"
                  + ("" if b is None else f", 3D box {b[0]:.0f} x {b[1]:.0f} x {b[2]:.0f} mm"), flush=True)
        tgt = objs[0]
        gp = await gripper_world()
        axis = np.array([gp.o_x, gp.o_y, gp.o_z], float)
        axis /= np.linalg.norm(axis)
        gpos = pose_xyz(gp)
        reach = float(np.hypot(*tgt["world_mm"][:2]))
        print(f"gripper at world x{gpos[0]:.0f} y{gpos[1]:.0f} z{gpos[2]:.0f} mm, axis ({axis[0]:+.2f}, {axis[1]:+.2f}, {axis[2]:+.2f}); "
              f"the object is {reach:.0f} mm from the base axis (arm reach ~850 mm)", flush=True)
        if reach > args.max_reach_mm:
            raise SystemExit(f"the object is {reach:.0f} mm out, beyond --max-reach-mm {args.max_reach_mm:.0f}: not reachable from here")
        width = args.grip_width_mm or (min(tgt["box_mm"][0], tgt["box_mm"][1]) if tgt["box_mm"] else None)
        if width is None or not 25.0 <= width <= 80.0:
            raise SystemExit(f"cannot size the grip (width {width}); pass --grip-width-mm")
        # pre-grasp: stand off from the object along the line from the gripper to it, at the object's height, same orientation
        to_obj = tgt["world_mm"] - gpos
        direction = to_obj / np.linalg.norm(to_obj)
        pre = tgt["world_mm"] - direction * args.standoff_mm
        pre[2] = max(pre[2], args.min_z_mm)
        print(f"plan: open to {min(width + 20, 84):.0f} mm -> planned move to a pre-grasp pose x{pre[0]:.0f} y{pre[1]:.0f} z{pre[2]:.0f} "
              f"({args.standoff_mm:.0f} mm short) -> flatten joint {args.wrist_joint} -> planned move onto the object -> close to {width:.0f} mm "
              f"(position {width * 10:.0f}) -> lift {args.lift_mm:.0f} mm", flush=True)
        if not real:
            print("DRY RUN: nothing moved.")
            return
        if not confirm("The arm WILL make a planned move to the pre-grasp pose (the machine's obstacles are avoided). E-stop in reach?"):
            return
        await gripper.do_command({"set": float(min(width + 20, 84) * 10)})
        j6_before = (await arm.get_joint_positions()).values[5]
        dest = Pose(x=pre[0], y=pre[1], z=pre[2], o_x=gp.o_x, o_y=gp.o_y, o_z=gp.o_z, theta=gp.theta)   # keep the orientation
        await motion.move(Gripper.get_resource_name(args.gripper), PoseInFrame(reference_frame="world", pose=dest))
        j6_after = (await arm.get_joint_positions()).values[5]
        print(f"at the pre-grasp pose; joint 6 changed {j6_after - j6_before:+.1f} deg", flush=True)
        if abs(j6_after - j6_before) > 90:
            raise SystemExit("joint 6 turned a long way during the planned move; stopping here")
        if not confirm(f"FLATTEN: rotate joint {args.wrist_joint} only, in 10-degree steps (up to {args.wrist_deg:.0f}), until the gripper is level."):
            return
        wj = args.wrist_joint - 1
        j0 = list((await arm.get_joint_positions()).values)
        sign = args.wrist_sign
        if not sign:                                      # 10-degree test each way: which flattens it and points it forward?
            p0 = await gripper_world()
            options = []
            for sg in (1, -1):
                t = list(j0)
                t[wj] += sg * 10.0
                await arm.move_to_joint_positions(JointPositions(values=t))
                p1 = await gripper_world()
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
            if abs((await gripper_world()).o_z) < 0.05:
                break
        gp = await gripper_world()
        axis = np.array([gp.o_x, gp.o_y, gp.o_z], float)
        axis /= np.linalg.norm(axis)
        print(f"joint {args.wrist_joint} rotated {sign * total:+.0f} deg; the gripper axis is now ({axis[0]:+.2f}, {axis[1]:+.2f}, {axis[2]:+.2f})", flush=True)
        # re-perceive from the new pose, then a planned move that stops args.hold_mm short of the object's centre along the axis
        objs = await perceive(robot, vision, args.camera, args.object)
        tgt = objs[0] if objs else tgt
        print(f"object now at world x{tgt['world_mm'][0]:.0f} y{tgt['world_mm'][1]:.0f} z{tgt['world_mm'][2]:.0f} mm", flush=True)
        grasp_pos = tgt["world_mm"] - axis * args.hold_mm
        dest = Pose(x=grasp_pos[0], y=grasp_pos[1], z=max(grasp_pos[2], args.min_z_mm), o_x=gp.o_x, o_y=gp.o_y, o_z=gp.o_z, theta=gp.theta)
        if not confirm(f"ADVANCE (planned move) to x{dest.x:.0f} y{dest.y:.0f} z{dest.z:.0f}, then CLOSE the gripper to {width:.0f} mm and lift."):
            return
        await motion.move(Gripper.get_resource_name(args.gripper), PoseInFrame(reference_frame="world", pose=dest))
        await gripper.do_command({"set": float(width * 10)})
        await asyncio.sleep(2.5)
        print("gripper position:", await gripper.do_command({"get": True}), flush=True)
        gp = await gripper_world()
        up = Pose(x=gp.x, y=gp.y, z=gp.z + args.lift_mm, o_x=gp.o_x, o_y=gp.o_y, o_z=gp.o_z, theta=gp.theta)
        await motion.move(Gripper.get_resource_name(args.gripper), PoseInFrame(reference_frame="world", pose=up))
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
    ap.add_argument("--object", default="bottle", help="label to pick up (the segmenter's detector labels)")
    ap.add_argument("--vision", default="object-segmenter", help="vision service that returns 3D object point clouds")
    ap.add_argument("--camera", default="cam")
    ap.add_argument("--arm", default="arm")
    ap.add_argument("--gripper", default="gripper")
    ap.add_argument("--sink", choices=["sim", "viam"], default="sim", help="sim = dry run, nothing moves")
    ap.add_argument("--standoff-mm", type=float, default=150.0, help="pre-grasp distance short of the object")
    ap.add_argument("--hold-mm", type=float, default=0.0, help="stop this far short of the object's centre along the gripper axis "
                    "(fingertips ~10 cm ahead of the camera; the gripper frame already includes that offset)")
    ap.add_argument("--min-z-mm", type=float, default=-13.0, help="never plan below this world height (table-safe for a flat gripper)")
    ap.add_argument("--max-reach-mm", type=float, default=800.0, help="refuse objects further than this from the base axis")
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
