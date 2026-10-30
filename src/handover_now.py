"""Hand the object the gripper is holding to a person, now (no pick): hold it out at --spot, let go on a steady pull on the arm (or when the file
--release-file appears), put it back where it was picked up if nobody takes it in --wait-s.

    scripts/handover_now_real.sh [--spot 500,0,350]
The arm moves slowly (--speed-dps). Keep the e-stop in reach."""
import argparse
import asyncio
import os
import sys
import time

import numpy as np
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions
from viam.services.motion import MotionClient


async def main_async(a):
    d = DialOptions.with_api_key(os.environ["VIAM_API_KEY"], os.environ["VIAM_API_KEY_ID"])
    if ".local.viam.cloud" in a.viam_address:
        d.disable_webrtc = True
    r = await RobotClient.at_address(a.viam_address, RobotClient.Options(dial_options=d))
    try:
        arm, grip, motion = Arm.from_robot(r, "arm"), Gripper.from_robot(r, "gripper"), MotionClient.from_robot(r, "builtin")

        async def pose():
            return (await r.transform_pose(PoseInFrame(reference_frame="gripper", pose=Pose()), "world")).pose

        async def go(x, y, z, p):
            await motion.move("gripper", PoseInFrame(reference_frame="world", pose=Pose(x=x, y=y, z=z, o_x=p.o_x, o_y=p.o_y, o_z=p.o_z, theta=p.theta)))

        async def speed(v):
            try:
                await arm.do_command({"set_speed": float(v)})
            except Exception:
                pass

        async def load():
            return np.array((await arm.do_command({"load": ""})).get("load", [0.0] * 6), float)

        p0 = await pose()
        spot = [float(v) for v in a.spot.split(",")]
        put_down = (p0.x, p0.y, p0.z - a.lift_mm)
        print(f"holding at x{p0.x:.0f} y{p0.y:.0f} z{p0.z:.0f}; moving slowly to x{spot[0]:.0f} y{spot[1]:.0f} z{spot[2]:.0f}", flush=True)
        await speed(a.speed_dps)
        taken = False
        try:
            await go(*spot, p0)
            print("holding it out. Take it (a steady pull) -- or `touch " + a.release_file + "` to release.", flush=True)
            await asyncio.sleep(2.0)
            base = await load()
            t0, since = time.monotonic(), None
            while time.monotonic() - t0 < a.wait_s:
                if os.path.exists(a.release_file):
                    print("released on command", flush=True)
                    taken = True
                    break
                if float(np.max(np.abs(await load() - base))) > a.pull_load:
                    since = since or time.monotonic()
                    if time.monotonic() - since >= a.pull_hold_s:
                        print("felt a steady pull: letting go", flush=True)
                        taken = True
                        break
                else:
                    since = None
                await asyncio.sleep(0.05)
            if not taken:
                print(f"nobody took it in {a.wait_s:.0f} s: putting it back down where it was picked up", flush=True)
                await go(*put_down, p0)
                await asyncio.sleep(0.5)
            await grip.do_command({"set": a.open_pos})
            await asyncio.sleep(a.settle_s)
            q = await pose()
            await go(q.x - 0.0, q.y, q.z + (0 if taken else 80), q)          # clear of the person / the cup
        finally:
            await speed(a.restore_dps)
        print("handed over." if taken else "put it back down.", flush=True)
    finally:
        await r.close()


ap = argparse.ArgumentParser()
ap.add_argument("--viam-address", required=True)
ap.add_argument("--spot", default="500,0,350", help="world x,y,z (mm) to hold the object out at")
ap.add_argument("--lift-mm", type=float, default=120.0, help="how far above the table pose the arm is now (where it puts it back down)")
ap.add_argument("--speed-dps", type=float, default=15.0)
ap.add_argument("--restore-dps", type=float, default=50.0)
ap.add_argument("--wait-s", type=float, default=60.0)
ap.add_argument("--pull-load", type=float, default=1.0)
ap.add_argument("--pull-hold-s", type=float, default=0.3)
ap.add_argument("--settle-s", type=float, default=0.8)
ap.add_argument("--open-pos", type=float, default=840.0)
ap.add_argument("--release-file", default="/tmp/release_cup")
a = ap.parse_args()
if os.path.exists(a.release_file):
    os.remove(a.release_file)
asyncio.run(main_async(a))
