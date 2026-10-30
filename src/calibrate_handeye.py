"""Calibrate where the wrist camera really is (hand-eye calibration) using one ArUco marker fixed on the table.

1. Print data/aruco_marker_id0.png (scripts/make_marker.py makes it) at a KNOWN size (default --marker-mm 80: measure the black square after printing)
   and tape it flat on the table, where the camera can see it from the arm's current pose (30-60 cm away).
2. Put the arm in a pose where the marker is in view, then run this. It moves the arm by SMALL joint changes (<= --max-deg, default 12) around that
   pose, grabs the marker's pose in the camera at each stop, and solves for the camera's pose in the gripper frame.
3. It prints the camera frame to use in the machine config. It does NOT change the config: that is a separate step you approve.

    scripts/run.sh src/calibrate_handeye.py --viam-address sfhack16-main.ty1cywh7tq.local.viam.cloud:8080          # dry: only looks, no motion
    scripts/run.sh src/calibrate_handeye.py --viam-address ... --move                                              # real arm: asks first
"""
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from viam.components.arm import Arm, JointPositions  # noqa: E402
from viam.components.camera import Camera  # noqa: E402
from viam.proto.common import Pose, PoseInFrame  # noqa: E402
from viam.robot.client import RobotClient  # noqa: E402
from viam.rpc.dial import DialOptions  # noqa: E402

from mimic.handeye import compose, quat_to_rot, rot_to_quat, rotation_error_deg, solve_handeye  # noqa: E402

# small joint offsets (deg) applied to the starting pose: (joint index 0-5, delta) pairs; the mix gives the camera a variety of rotations
OFFSETS = [((0, 8),), ((0, -8),), ((1, 6),), ((1, -6),), ((3, 10),), ((3, -10),), ((4, 10),), ((4, -10),),
           ((0, 8), (3, 10)), ((0, -8), (4, 10)), ((1, 6), (4, -10)), ((1, -6), (3, -10)), ((0, 6), (1, 5), (4, 8)), ((0, -6), (1, -5), (3, -8))]


async def frame_to_world(robot, frame):
    """Pose of `frame` in the world as (R, t): the frame system maps the frame's origin and unit axes."""
    async def pt(x, y, z):
        p = (await robot.transform_pose(PoseInFrame(reference_frame=frame, pose=Pose(x=x, y=y, z=z)), "world")).pose
        return np.array([p.x, p.y, p.z])
    o = await pt(0, 0, 0)
    return np.stack([await pt(1, 0, 0) - o, await pt(0, 1, 0) - o, await pt(0, 0, 1) - o], axis=1), o


def detect_marker(color, K, marker_mm, dict_id=cv2.aruco.DICT_4X4_50, want_id=0):
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dict_id), cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(color)
    if ids is None or want_id not in ids.flatten():
        return None
    c = corners[list(ids.flatten()).index(want_id)].reshape(4, 2).astype(np.float32)
    h = marker_mm / 2.0
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)        # ArUco corner order: TL, TR, BR, BL
    ok, rvec, tvec = cv2.solvePnP(obj, c, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3), c


async def grab(cam):
    imgs = (await cam.get_images())[0]
    for im in imgs:
        if "dep" not in str(im.mime_type).lower():
            return cv2.imdecode(np.frombuffer(im.data, np.uint8), cv2.IMREAD_COLOR)
    return None


async def main_async(args):
    d = DialOptions.with_api_key(os.environ["VIAM_API_KEY"], os.environ["VIAM_API_KEY_ID"])
    if ".local.viam.cloud" in args.viam_address:
        d.disable_webrtc = True
    robot = await RobotClient.at_address(args.viam_address, RobotClient.Options(dial_options=d))
    try:
        arm, cam = Arm.from_robot(robot, args.arm), Camera.from_robot(robot, args.camera)
        ip = (await cam.get_properties()).intrinsic_parameters
        K = np.array([[ip.focal_x_px, 0, ip.center_x_px], [0, ip.focal_y_px, ip.center_y_px], [0, 0, 1.0]])
        color = await grab(cam)
        sx = color.shape[1] / ip.width_px
        K[0] *= sx
        K[1] *= sx                                                       # the intrinsics are for ip.width_px; the frame may be a different size
        first = detect_marker(color, K, args.marker_mm)
        if first is None:
            raise SystemExit("no ArUco marker (4x4, id 0) in view from this pose: put the arm where the camera sees the printed marker")
        print(f"marker seen {np.linalg.norm(first[1]) / 10:.0f} cm from the camera", flush=True)
        if not args.move:
            print("DRY RUN: marker visible; nothing moved. Re-run with --move to calibrate.")
            return
        if input(f"The arm WILL move by small joint steps (<= {args.max_deg:.0f} deg) through {len(OFFSETS)} poses around here. E-stop in reach? Type 'yes': ").strip().lower() != "yes":
            return
        start = list((await arm.get_joint_positions()).values)
        Rg, tg, Rt, tt = [], [], [], []
        try:
            for k, offs in enumerate(OFFSETS, 1):
                j = list(start)
                for idx, dd in offs:
                    j[idx] += float(np.clip(dd, -args.max_deg, args.max_deg))
                await arm.move_to_joint_positions(JointPositions(values=j))
                await asyncio.sleep(1.2)                                # let it settle, then take the median of a few frames
                poses = []
                for _ in range(4):
                    r = detect_marker(await grab(cam), K, args.marker_mm)
                    if r:
                        poses.append(r)
                    await asyncio.sleep(0.15)
                if len(poses) < 3:
                    print(f"[{k:2}] marker not seen steadily: skipped", flush=True)
                    continue
                R_t = poses[0][0]
                t_t = np.median([p[1] for p in poses], axis=0)
                R_g, t_g = await frame_to_world(robot, args.gripper)
                Rg.append(R_g), tg.append(t_g), Rt.append(R_t), tt.append(t_t)
                print(f"[{k:2}] ok: marker {np.linalg.norm(t_t) / 10:.0f} cm away", flush=True)
        finally:
            await arm.move_to_joint_positions(JointPositions(values=start))          # always go back to where it began
        if len(Rg) < 6:
            raise SystemExit(f"only {len(Rg)} usable poses (need at least 6): move the marker or start from a pose that sees it better")
        R, t = solve_handeye(Rg, tg, Rt, tt)
        # residual check: the marker's world pose should be the same from every view
        M = [compose(*compose(Rg[i], tg[i], R, t), Rt[i], tt[i]) for i in range(len(Rg))]
        spread_t = np.std([m[1] for m in M], axis=0)
        spread_r = max(rotation_error_deg(M[0][0], m[0]) for m in M)
        print(f"\ncamera in the GRIPPER frame: x{t[0]:.1f} y{t[1]:.1f} z{t[2]:.1f} mm, quaternion (w,x,y,z) {np.round(rot_to_quat(R), 4).tolist()}")
        print(f"consistency: the marker's world position varies by {np.round(spread_t, 1).tolist()} mm and {spread_r:.2f} deg across the {len(Rg)} views "
              "(smaller is better; > 15 mm or > 2 deg means a poor calibration: reprint at a known size / use a flatter marker)")
        # the config puts the camera's frame under the ARM flange: camera in arm = gripper-in-arm composed with camera-in-gripper
        gz = args.gripper_z_mm                                          # the gripper frame is z mm along the arm's tool axis (config: 105), no rotation
        t_arm = np.array([0.0, 0.0, gz]) + t
        q = rot_to_quat(R)
        out = {"parent": args.arm,
               "translation": {"x": round(float(t_arm[0]), 1), "y": round(float(t_arm[1]), 1), "z": round(float(t_arm[2]), 1)},
               "orientation": {"type": "quaternion", "value": {"w": round(float(q[0]), 5), "x": round(float(q[1]), 5), "y": round(float(q[2]), 5), "z": round(float(q[3]), 5)}}}
        os.makedirs("data", exist_ok=True)
        json.dump({"camera_frame": out, "views": len(Rg), "marker_world_spread_mm": np.round(spread_t, 1).tolist(), "marker_world_spread_deg": round(spread_r, 2)},
                  open("data/handeye_result.json", "w"), indent=2)
        print("\nframe for the machine config (NOT applied):\n" + json.dumps(out, indent=2) + "\nsaved to data/handeye_result.json")
    finally:
        await robot.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viam-address", required=True)
    ap.add_argument("--arm", default="arm")
    ap.add_argument("--gripper", default="gripper")
    ap.add_argument("--camera", default="cam")
    ap.add_argument("--marker-mm", type=float, default=80.0, help="side of the printed black square, measured")
    ap.add_argument("--max-deg", type=float, default=12.0, help="largest joint offset from the starting pose")
    ap.add_argument("--gripper-z-mm", type=float, default=105.0, help="the gripper frame's z offset from the arm flange in the machine config")
    ap.add_argument("--move", action="store_true", help="move the arm through the calibration poses (default: only check that the marker is visible)")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
