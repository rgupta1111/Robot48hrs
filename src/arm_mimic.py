"""Make a robot arm mimic your arm: webcam -> MediaPipe pose -> Viam arm.

Safe first run (no robot, shows the window and what the arm would do):
    python src/arm_mimic.py --arm right

Controls (click the video window first):
    space  engage / disengage   - nothing moves until you engage; hold your hand in a comfortable
                                  neutral spot first, that spot becomes the robot's start pose
    c      recenter             - make the current hand position the new zero (robot stays put)
    (joint) T = sweep test of the joint (0>90>0, no hand needed). --pinch-joint N follows your arm AND drives joint N
            from the pinch; add --pinch-only to test the pinch alone (it then engages as soon as a hand is seen)
    (joint) with --pinch-joint N: the thumb-index pinch value (0 closed .. 1 spread) sets joint N to 0..90 degrees
    (wrist) turn your hand about the vertical axis -> the gripper turns; --wrist-tilt adds pitch
    g      open / close gripper by hand (pinch thumb and index finger to close it automatically)
    x      emergency stop       - disengage and halt the arm
    q/Esc  quit

Real arm (VIAM_API_KEY / VIAM_API_KEY_ID in the environment). Set the arm's speed limit low in the
Viam config first, keep the workspace clear, and keep a hand on the machine's power / e-stop:
    python src/arm_mimic.py --sink viam --viam-address <machine>.viam.cloud --preset ur5e
(--preset ur5e fills in start / reach / bounds / speed for the fake UR5e in viam-videorobot-machine.json,
 whose components are arm-1 and gripper-1. --preset lab does the same for the real arm (named "arm") in the
 walls-and-table world. Any of those can still be overridden on the command line. If --arm-name is wrong and the
 machine has exactly one arm, that arm is used; a machine with no gripper runs arm-only.)
A viam-server with no auth on this network (e.g. a local test config) works with --viam-insecure.

Your arm mirrors onto the robot as if you were facing it: reach toward the camera -> robot reaches
forward, move your hand to your right -> robot moves to its right, up is up. Use --flip-x / --flip-y
if the robot is mounted the other way round. Offline check on a recording:
    python src/arm_mimic.py --video data/test_videos/family_at_home.mp4 --auto-engage --no-display
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mimic.arm_tracker import ArmTracker  # noqa: E402
from mimic.controller import Command, MimicConfig, MimicController, PinchGripper  # noqa: E402
from mimic.hand_tracker import HandTracker  # noqa: E402
from mimic.sinks import SimSink, ViamSink  # noqa: E402

WIN = "arm mimic"
# robot-specific defaults, in the robot's frame with the arm base at the origin and z up (mm)
PRESETS = {
    "xarm6": dict(start="300,0,250", reach=200.0, max_speed=100.0, bounds="200,450,-250,250,80,400"),
    "ur5e": dict(start="400,0,300", reach=300.0, max_speed=100.0, bounds="250,650,-300,300,100,500"),
    # the real arm in the hackathon world: base at the world origin, table top at z=-123, walls behind (x=-525)
    # and to the left (y=+625), ceiling at z=1050. z=0 keeps the tool 123 mm above the table.
    "lab": dict(start="400,0,250", reach=250.0, max_speed=80.0, bounds="250,600,-300,300,0,450"),
}


def floats(text):
    return tuple(float(v) for v in text.split(","))


def draw_debug(frame, lines):
    """Small diagnostics panel, bottom-right."""
    h, w = frame.shape[:2]
    scale, thick, step = 0.55, 1, 22
    width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for t in lines) + 16
    x0, y0 = w - width - 6, h - step * len(lines) - 14
    shade = frame.copy()
    cv2.rectangle(shade, (x0 - 4, y0 - 6), (w - 2, h - 4), (0, 0, 0), -1)
    cv2.addWeighted(shade, 0.6, frame, 0.4, 0, frame)
    for i, t in enumerate(lines):
        col = (0, 255, 0) if "pinch joint" in t else (0, 255, 255)
        cv2.putText(frame, t, (x0 + 4, y0 + 12 + step * i), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick)


def joint_rows(stats, joint_index):
    """All joint angles as the arm reports them (the arm's own 0 reference), one row per joint."""
    joints = stats.get("joints")
    if joints is None:
        return ["arm joints: (not read yet)" if not stats.get("poll_error") else f"arm joints: {stats['poll_error']}"]
    return [f"J{i}  {v:8.1f} deg" + ("   <- pinch joint" if i == joint_index else "") for i, v in enumerate(joints)]


def debug_lines(ctl, status, pose, hand, hands, sink, joint_on, cfg):
    """The same facts for the screen panel and the console. Pinch-only mode: hand seen/used, pinch value, the
    joint angle commanded / sent / read back, then every joint of the arm. Full mode adds the arm-pose state."""
    info = getattr(hands, "last_info", {}) if hands else {}
    if not hands:
        hand_txt = "hand: finger tracking OFF"
    elif info.get("hands", 0) == 0:
        hand_txt = "hand: NONE seen by the hand model"
    elif info.get("used"):
        d = info.get("dist_px")
        hand_txt = f"hand: {info['hands']} seen, USED" + (f" ({d:.0f}px from tracked wrist)" if d is not None else "")
    else:
        hand_txt = f"hand: REJECTED ({info['dist_px']:.0f}px > {info['limit_px']:.0f}px from tracked wrist)"
    lines = [f"state: {status.state}" + ("" if cfg.pinch_only else f"   arm pose: {'ok' if pose is not None else 'NOT SEEN'}"),
             hand_txt]
    if joint_on:
        d, stats = ctl.debug, getattr(sink, "stats", {})
        lines.append(f"pinch raw {d['ratio_raw']:.2f}  smooth {d['ratio_smooth']:.2f}  -> goal {d['goal']:.0f} deg"
                     if d else "pinch: (no value yet)")
        sent = stats.get("joint_sent")
        lines.append(f"joint {cfg.pinch_joint}: cmd {status.joint_deg if status.joint_deg is not None else float('nan'):.1f}"
                     f"  sent {'-' if sent is None else format(sent, '.1f')}   moves {stats.get('joint_moves', '-')}")
        lines += joint_rows(stats, cfg.pinch_joint)
    return lines


def draw_pinch_hud(frame, status, cfg, note, hand):
    """Pinch-only screen: just the thumb-index line and the numbers that matter for one gesture -> one joint."""
    h, w = frame.shape[:2]
    if hand is not None:
        a, b = (int(w - hand.thumb_px[0]), int(hand.thumb_px[1])), (int(w - hand.index_px[0]), int(hand.index_px[1]))
        cv2.line(frame, a, b, (0, 255, 255), 4)
        cv2.circle(frame, a, 8, (255, 255, 0), -1)
        cv2.circle(frame, b, 8, (255, 255, 0), -1)
    col = (0, 200, 0) if status.engaged else (0, 0, 220)
    lines = [("ENGAGED" if status.engaged else "IDLE - space to engage") + f" | {status.state}",
             f"pinch {hand.ratio:.2f}  (0 closed .. 1 spread)" if hand is not None else "pinch: no hand seen",
             f"joint {cfg.pinch_joint} target " + (f"{status.joint_deg:.0f} deg" if status.joint_deg is not None else "-")]
    if note:
        lines.append(note)
    scale, thick, step = 0.85, 2, 36
    width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for t in lines)
    shade = frame.copy()
    cv2.rectangle(shade, (0, 0), (width + 28, step * len(lines) + 12), (0, 0, 0), -1)
    cv2.addWeighted(shade, 0.55, frame, 0.45, 0, frame)
    for i, t in enumerate(lines):
        pos = (14, 34 + step * i)
        cv2.putText(frame, t, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
        cv2.putText(frame, t, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, col if i == 0 else (255, 255, 255), thick)


def draw_hud(frame, pose, status, cfg, gripper_open, note="", hand=None):
    if cfg.pinch_only:
        return draw_pinch_hud(frame, status, cfg, note, hand)
    h, w = frame.shape[:2]
    if pose is not None:
        pts = [(int(w - x), int(y)) for x, y in pose.pixels]  # frame is mirrored for display
        cv2.polylines(frame, [np.array(pts)], False, (0, 220, 255), 4)
        for p in pts:
            cv2.circle(frame, p, 7, (0, 140, 255), -1)
    if hand is not None:  # thumb-index line: green = open, red = pinched/closed
        a, b = (int(w - hand.thumb_px[0]), int(hand.thumb_px[1])), (int(w - hand.index_px[0]), int(hand.index_px[1]))
        cv2.line(frame, a, b, (0, 200, 0) if gripper_open else (0, 0, 255), 3)
        cv2.circle(frame, a, 6, (255, 255, 0), -1)
        cv2.circle(frame, b, 6, (255, 255, 0), -1)
    if hand is not None and hand.knuckles_px:  # the knuckle line whose heading drives the gripper yaw
        k1, k2 = ((int(w - x), int(y)) for x, y in hand.knuckles_px)
        cv2.line(frame, k1, k2, (255, 0, 255), 4)
    col = (0, 200, 0) if status.engaged else (0, 0, 220)
    lines = [("ENGAGED" if status.engaged else "IDLE - space to engage") + f" | {status.state}",
             f"gripper {'open' if gripper_open else 'closed'}"
             + (f"   pinch {hand.ratio:.2f}" if hand is not None else "   (no hand seen)")]
    if pose is not None:
        lines.append(f"elbow {pose.elbow_angle_deg:.0f} deg   conf {pose.visibility:.2f}")
    if status.target:
        lines.append("target x{:.0f} y{:.0f} z{:.0f} mm".format(*status.target) + ("  CLAMPED" if status.clamped else ""))
    lines.append(f"wrist  heading {hand.heading_deg:+.0f}  pitch {hand.elevation_deg:+.0f} deg"
                 if hand is not None else "wrist  (no hand seen)")
    if status.engaged and cfg.pinch_joint is None:
        lines.append(f"gripper yaw {status.yaw_deg:+.0f}  tilt {status.tilt_deg:+.0f} deg")
    if cfg.pinch_joint is not None:
        lines.append(f"joint {cfg.pinch_joint}  " + (f"{status.joint_deg:+.0f} deg" + (" from start" if not cfg.joint_absolute else "")
                                                     if status.joint_deg is not None else "-") + (f"   pinch {hand.ratio:.2f}" if hand is not None else ""))
    if note:
        lines.append(note)
    scale, thick, step = 0.85, 2, 36
    width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for t in lines)
    shade = frame.copy()
    cv2.rectangle(shade, (0, 0), (width + 28, step * len(lines) + 12), (0, 0, 0), -1)
    cv2.addWeighted(shade, 0.55, frame, 0.45, 0, frame)  # darken behind the text so it reads on any background
    for i, t in enumerate(lines):
        pos = (14, 34 + step * i)
        cv2.putText(frame, t, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
        cv2.putText(frame, t, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, col if i == 0 else (255, 255, 255), thick)
    # mini maps of the workspace: top view (x up, y left) and side view (x right, z up)
    xmin, xmax, ymin, ymax, zmin, zmax = cfg.bounds
    size, pad = 130, 10

    def panel(x0, label, to_xy):
        y0 = h - size - pad
        cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (30, 30, 30), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (200, 200, 200), 1)
        cv2.putText(frame, label, (x0 + 4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        for pos, c in ((cfg.start, (120, 120, 120)), (status.target, (0, 255, 0))):
            if pos:
                u, v = to_xy(pos)
                cv2.circle(frame, (int(x0 + u * size), int(y0 + v * size)), 5, c, -1)

    panel(pad, "top (fwd up)", lambda p: ((ymax - p[1]) / (ymax - ymin), 1 - (p[0] - xmin) / (xmax - xmin)))
    panel(2 * pad + size, "side (up)", lambda p: ((p[0] - xmin) / (xmax - xmin), 1 - (p[2] - zmin) / (zmax - zmin)))


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", choices=list(PRESETS))
    preset = pre.parse_known_args()[0].preset
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS), help="robot-specific start / reach / bounds / speed defaults")
    ap.add_argument("--arm", choices=["right", "left"], default="right", help="which of YOUR arms to track")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", help="use a video file instead of the webcam")
    ap.add_argument("--sink", choices=["sim", "viam"], default="sim")
    ap.add_argument("--start", default="150,0,150", help="robot x,y,z (mm) held at engage")
    ap.add_argument("--reach", type=float, default=250.0, help="robot travel (mm) for a fully stretched arm")
    ap.add_argument("--depth-gain", type=float, default=0.6, help="0 ignores toward/away-from-camera motion")
    ap.add_argument("--max-speed", type=float, default=150.0, help="mm/s limit on the commanded target")
    ap.add_argument("--bounds", default="60,320,-200,200,40,300", help="xmin,xmax,ymin,ymax,zmin,zmax in mm")
    ap.add_argument("--flip-x", action="store_true"), ap.add_argument("--flip-y", action="store_true")
    ap.add_argument("--viam-address"), ap.add_argument("--arm-name", default="arm")
    ap.add_argument("--gripper-name", default="gripper")
    ap.add_argument("--viam-insecure", action="store_true", help="connect without an API key (local test server)")
    ap.add_argument("--allow-collision", action="append", default=[], metavar="FRAME1,FRAME2",
                    help="let the planner ignore a collision between two frames (repeatable), e.g. the arm base and "
                    "a camera mount that overlap in the config: --allow-collision arm,cam_origin. Moves then go "
                    "through the motion service (slower than direct IK)")
    ap.add_argument("--table-to-world", default="0,0,0,0", help="dx,dy,dz,yaw_deg of the robot frame in the arm frame")
    ap.add_argument("--no-hand", action="store_true", help="skip finger tracking (gripper by key only)")
    ap.add_argument("--pinch-only", action="store_true", help="with --pinch-joint: ignore your arm and test just the "
                    "pinch -> joint link (no elbow/wrist following, arm not moved to the start pose)")
    ap.add_argument("--pinch-joint", type=int, help="drive this robot joint from the thumb-index spread, numbered "
                    "as in the Viam app's arm panel (UR5e: 0 = base ... 5 = last wrist joint)")
    ap.add_argument("--joint-ratios", default="0,1", help="pinch ratio when fingers are closed,spread")
    ap.add_argument("--joint-span", default="0,90", help="joint degrees at those two ratios")
    ap.add_argument("--joint-relative", action="store_true", help="make --joint-span an offset from the joint's angle at engage")
    ap.add_argument("--joint-speed", type=float, default=90.0, help="deg/s limit on the joint")
    ap.add_argument("--flip-joint", action="store_true", help="swap which way is left/right")
    ap.add_argument("--debug", action="store_true", help="print diagnostics to the console and show them on screen "
                    "(always on with --pinch-joint; --no-debug turns it off)")
    ap.add_argument("--no-debug", action="store_true")
    ap.add_argument("--no-wrist-yaw", action="store_true", help="do not turn the gripper with your hand")
    ap.add_argument("--wrist-tilt", action="store_true", help="also lean the gripper when you pitch your hand")
    ap.add_argument("--flip-yaw", action="store_true"), ap.add_argument("--flip-tilt", action="store_true")
    ap.add_argument("--pinch-close", type=float, default=0.35, help="close the gripper below this pinch ratio")
    ap.add_argument("--pinch-open", type=float, default=0.6, help="reopen above this pinch ratio")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--auto-engage", action="store_true", help="engage on the first good pose (for recordings)")
    ap.add_argument("--sweep-test", action="store_true", help="with --pinch-joint: sweep the joint 0>90>0 at start, no hand")
    ap.add_argument("--manual-engage", action="store_true",
                    help="with --pinch-joint: wait for the space key instead of engaging as soon as a hand is seen")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--csv", help="write every command to this CSV")
    if preset:
        ap.set_defaults(**PRESETS[preset])
    args = ap.parse_args()

    joint_on = args.pinch_joint is not None
    pinch_only = joint_on and args.pinch_only
    # in combined mode the last wrist joint is also the tool twist that wrist yaw drives, so leave that to the pinch
    yaw_conflict = joint_on and not pinch_only and args.pinch_joint >= 5
    cfg = MimicConfig(start=floats(args.start), reach_mm=args.reach, depth_gain=args.depth_gain,
                      max_speed_mm_s=args.max_speed, bounds=floats(args.bounds),
                      flip_x=args.flip_x, flip_y=args.flip_y, wrist_yaw=not args.no_wrist_yaw and not pinch_only and not yaw_conflict,
                      pinch_only=pinch_only,
                      lost_disengage_s=1e9 if pinch_only else 3.0,  # pinch-only: just hold the joint when the hand leaves
                      pinch_joint=args.pinch_joint, joint_ratios=floats(args.joint_ratios),
                      joint_span=floats(args.joint_span), joint_speed=args.joint_speed, flip_joint=args.flip_joint,
                      joint_absolute=not args.joint_relative,
                      wrist_tilt=args.wrist_tilt, flip_yaw=args.flip_yaw, flip_tilt=args.flip_tilt)
    ctl = MimicController(cfg)
    tracker = None if pinch_only else ArmTracker(args.arm)  # pinch-only mode needs neither the arm tracker nor its pose
    hands, pinch = None, PinchGripper(args.pinch_close, args.pinch_open)
    if not args.no_hand:
        try:
            hands = HandTracker()
        except FileNotFoundError as e:
            print(f"finger tracking off: {e}")
    if joint_on and hands is None:
        sys.exit("--pinch-joint needs finger tracking (models/hand_landmarker.task)")

    if args.sink == "viam":
        from tidy.robot import TableToWorld, ViamRobot
        xmin, xmax, ymin, ymax, zmin, zmax = cfg.bounds
        robot = ViamRobot(args.viam_address, args.arm_name, args.gripper_name, None,
                          TableToWorld(*floats(args.table_to_world)), workspace=(xmin, xmax, ymin, ymax),
                          max_z_mm=zmax, insecure=args.viam_insecure,
                          allow_collisions=[p.split(",") for p in args.allow_collision])
        sink = ViamSink(robot, joint_index=args.pinch_joint, joint_absolute=not args.joint_relative,
                        joint_only=pinch_only)
    else:
        sink = SimSink(verbose=args.no_display)

    debug = (args.debug or joint_on) and not args.no_debug
    if joint_on and not pinch_only:
        if yaw_conflict:
            print("note: wrist yaw is off because the pinch drives the last wrist joint (the tool twist)", flush=True)
        if args.sink == "viam" and args.pinch_joint < len(robot.get_joints()) - 1:
            print(f"note: joint {args.pinch_joint} also moves the tool, so following your arm re-solves it on every "
                  f"position update and you may see it jitter; --pinch-only isolates the pinch, or use the last wrist joint",
                  flush=True)
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        sys.exit("cannot open " + (args.video or f"camera {args.camera} (grant camera access to your terminal/app)"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = csv.writer(open(args.csv, "w", newline="")) if args.csv else None
    if out:
        out.writerow(["t", "x", "y", "z", "yaw", "tilt", "joint", "state"])

    gripper_open, frame_i, tracked, t0 = True, 0, 0, time.monotonic()
    want_engage = args.auto_engage or (pinch_only and not args.manual_engage)  # pinch-only: engage on the first hand
    sweep_t0, prepared, sweep_pending = None, False, args.sweep_test and joint_on
    last_dbg = -1e9
    last_pose, status = None, ctl.update(None, 0)[1]
    note = ("engages when a hand is seen.  T = sweep test (joint 0>90>0, no hand needed)" if pinch_only and want_engage
            else "SPACE = engage (click this window first).  T = sweep test" if joint_on else "")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_frames and frame_i >= args.max_frames):
                break
            t = frame_i / fps if args.video else time.monotonic() - t0
            frame_i += 1
            pose = tracker.process(frame, t) if tracker else None
            last_pose = pose if pose is not None else last_pose
            hand = (hands.process(frame, t, pose.pixels[2] if pose is not None else None,
                                  None if joint_on else 0.25) if hands else None)  # pinch-joint: the nearest hand will do
            tracked += (hand is not None) if pinch_only else (pose is not None)

            if want_engage:
                if pose is not None or (joint_on and hand is not None):
                    if not ctl.engaged:
                        sink.prepare(cfg.start)  # may take a moment on a real arm: it moves to the start pose
                        prepared = True
                        if ctl.engage(pose, t):
                            note = ("pinch -> joint %d" % args.pinch_joint if pinch_only
                                    else "following + pinch -> joint %d" % args.pinch_joint if joint_on else "following")
                            pinch.reset()
                            if joint_on and getattr(sink, "_joint_base", None) is not None:
                                ctl.seed_joint(sink._joint_base)  # ease from the joint's current angle
                            if not joint_on:
                                gripper_open = True
                                sink.set_gripper(True)
                            if debug:
                                print(f"[dbg] ENGAGED at t={t:.1f}s ({note})", flush=True)
                    want_engage = False
                elif debug and t - last_dbg > 1.5:
                    last_dbg = t
                    print("[dbg] waiting to engage: " + ("no hand seen yet" if joint_on else "arm pose not visible - "
                          "keep your shoulder, elbow and wrist in view"), flush=True)
            cmd, status = ctl.update(pose, t, hand)
            if sweep_pending:
                if not prepared:
                    sink.prepare(cfg.start)
                    prepared = True
                sweep_t0, sweep_pending = t, False
                print(f"[dbg] sweep test started on joint {args.pinch_joint}", flush=True)
            if sweep_t0 is not None:  # sweep test: drive the joint 0 -> 90 -> 0 over 6 s with no hand involved
                phase = t - sweep_t0
                if phase > 6.0:
                    sweep_t0, note = None, "sweep test done"
                else:
                    angle = 90.0 * (1.0 - abs(1.0 - phase / 3.0))
                    cmd = Command(*cfg.start, 0.0, 0.0, angle)
                    status.joint_deg = angle
            if cmd is not None:
                sink.send(cmd)
                if out:
                    out.writerow([f"{t:.3f}", f"{cmd.x:.1f}", f"{cmd.y:.1f}", f"{cmd.z:.1f}",
                                  f"{cmd.yaw_deg:.1f}", f"{cmd.tilt_deg:.1f}",
                                  "" if cmd.joint_deg is None else f"{cmd.joint_deg:.1f}", status.state])
            if ctl.engaged and status.state == "tracking" and not joint_on:  # the pinch drives a joint instead
                change = pinch.update(hand.ratio if hand else None)
                if change is not None:
                    gripper_open = change
                    sink.set_gripper(change)
            if getattr(sink, "error", None):
                print("ARM ERROR:", sink.error)
                break
            dlines = debug_lines(ctl, status, pose, hand, hands, sink, joint_on, cfg) if debug else None
            if debug and (ctl.engaged or sweep_t0 is not None) and t - last_dbg > 0.5:
                last_dbg = t
                print(f"[dbg {t:6.1f}s] " + " | ".join(dlines), flush=True)

            if not args.no_display:
                view = cv2.flip(frame, 1)
                draw_hud(view, pose, status, cfg, gripper_open, note, hand)
                if dlines:
                    draw_debug(view, dlines)
                cv2.imshow(WIN, view)
                key = cv2.waitKey(1) & 0xFF
                if debug and key != 255:
                    print(f"[dbg] key pressed: {chr(key) if 32 <= key < 127 else key!r}", flush=True)
                if key in (ord("q"), 27):
                    break
                if key == ord("t") and joint_on:
                    if not prepared:
                        sink.prepare(cfg.start)
                        prepared = True
                    sweep_t0, note = t, "SWEEP TEST: joint %d 0 > 90 > 0" % args.pinch_joint
                    print(f"[dbg] sweep test started on joint {args.pinch_joint}", flush=True)
                if key == ord(" "):
                    if ctl.engaged:
                        ctl.disengage()
                        note = "disengaged (arm holds position)"
                    else:
                        want_engage, note = True, "hold still, engaging..."
                elif key == ord("c"):
                    ctl.recenter()
                    note = "recentered"
                elif key == ord("g"):
                    gripper_open = pinch.open = not gripper_open
                    sink.set_gripper(gripper_open)
                elif key == ord("x"):
                    ctl.disengage()
                    sink.stop()
                    note = "EMERGENCY STOP"
    finally:
        ctl.disengage()
        try:
            sink.stop()
        finally:
            sink.close()
            if tracker:
                tracker.close()
            if hands:
                hands.close()
            cap.release()
            cv2.destroyAllWindows()
    frac = tracked / max(frame_i, 1)
    msg = f"{frame_i} frames, {'hand' if pinch_only else 'arm'} tracked in {frac:.0%}"
    if isinstance(sink, SimSink) and len(sink.log) > 1:
        a = np.array([r[1:] for r in sink.log])
        msg += (f"; {len(sink.log) - 1} commands, x {a[:, 0].min():.0f}..{a[:, 0].max():.0f} "
                f"y {a[:, 1].min():.0f}..{a[:, 1].max():.0f} z {a[:, 2].min():.0f}..{a[:, 2].max():.0f} mm, "
                f"yaw {a[:, 3].min():.0f}..{a[:, 3].max():.0f} deg"
                + (f", joint {a[:, 5].min():+.0f}..{a[:, 5].max():+.0f} deg" if args.pinch_joint is not None else ""))
    print(msg)


if __name__ == "__main__":
    main()
