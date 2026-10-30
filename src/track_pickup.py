"""Basic pick-up by tracking: find the named object, move the gripper to it, close the gripper, lift. No UI, no threads.

    track (YOLO on this computer's GPU) -> keep the object lined up with the GRIPPER -> creep forward until it is
    --stop-cm away -> close the gripper (force-capped) -> lift

Dry run by default (--sink sim): reads the camera and prints what it would do, the arm and gripper do not move.
    scripts/run.sh src/track_pickup.py --viam-address sfhack16-main.ty1cywh7tq.local.viam.cloud:8080 --preset lab --object cup
Real arm (keep a hand on the e-stop; it asks you to type 'yes' before moving and again before closing the gripper):
    scripts/run.sh src/track_pickup.py --viam-address sfhack16-main.ty1cywh7tq.local.viam.cloud:8080 --preset lab --object cup \\
        --sink viam --tip-offset-mm 60 --arm-speed 30
The gripper sits below the camera, so the object is not centred in the picture: it is lined up where the gripper points
(--tip-offset-mm, --tip-flip if the gripper is above the camera). Ctrl+C stops the arm.
"""
import argparse
import os
import sys
import time
import webbrowser

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_mimic import PRESETS, floats  # noqa: E402
from local_detect import Lock, Range, detect, load_detector, object_candidates  # noqa: E402
from mimic.pickup import advance_along, check_jacobian, smooth_move, table_floor_z  # noqa: E402
from mimic.servo import (FeatureFilter, approach_factor, estimate_jacobian, features, servo_step,  # noqa: E402
                         tip_target_v)
from mimic.ui import UIServer  # noqa: E402
from tidy.robot import TableToWorld, ViamRobot  # noqa: E402


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", choices=list(PRESETS))
    preset = pre.parse_known_args()[0].preset
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS))
    ap.add_argument("--viam-address", required=True)
    ap.add_argument("--object", default="cup", help="what to pick up (a detector class name, e.g. cup, bottle)")
    ap.add_argument("--camera", default="cam")
    ap.add_argument("--arm-name", default="arm")
    ap.add_argument("--gripper-name", default="gripper")
    ap.add_argument("--sink", choices=["sim", "viam"], default="sim", help="sim = dry run, nothing moves")
    ap.add_argument("--bounds", default="250,600,-300,300,0,450", help="xmin,xmax,ymin,ymax,zmin,zmax mm; moves are clamped")
    ap.add_argument("--table-z-mm", type=float, default=-123.0, help="table top height in the robot frame (the lab preset's)")
    ap.add_argument("--tool-length-mm", type=float, default=105.0, help="flange-to-fingertip length of the gripper")
    ap.add_argument("--table-margin-mm", type=float, default=30.0, help="keep the fingertips at least this far above the table")
    ap.add_argument("--wrist-flat", action="store_true", help="at the grip distance, rotate ONLY one wrist joint "
                    "(--wrist-joint, default 4) until the gripper is flat, then close it (asks before each step)")
    ap.add_argument("--wrist-joint", type=int, default=4, help="--wrist-flat: which joint flattens the gripper, numbered as in the Viam app (1-6); "
                    "4 on this arm (not 5, and never 6, the final roll)")
    ap.add_argument("--wrist-deg", type=float, default=90.0, help="--wrist-flat: most it may rotate, in 10-degree steps")
    ap.add_argument("--wrist-sign", type=int, choices=[-1, 0, 1], default=0, help="--wrist-flat: which way (+1/-1); 0 = try both "
                    "with a 10-degree test and pick the one that flattens the gripper and points it forward")
    ap.add_argument("--flat-floor-mm", type=float, default=80.0, help="--wrist-flat: lowest extent of the gripper below its axis once flat")
    ap.add_argument("--stop-cm", type=float, default=10.0, help="close the gripper at this camera-to-object distance")
    ap.add_argument("--rotate-cm", type=float, default=15.0, help="--wrist-flat: stop and rotate the wrist at this camera-to-object "
                    "distance (the fingertips are ~10 cm ahead of the camera, so this is ~5 cm short of the object)")
    ap.add_argument("--close-cm", type=float, default=8.0, help="--wrist-flat: after flattening, advance along the gripper axis "
                    "until the camera-to-object distance is this, then close")
    ap.add_argument("--tip-offset-mm", type=float, default=0.0, help="the gripper sits this far below the camera")
    ap.add_argument("--tip-flip", action="store_true", help="the gripper is above the camera instead")
    ap.add_argument("--object-height-mm", type=float, default=100.0, help="true height of the object (for distance)")
    ap.add_argument("--hfov", type=float, default=70.0, help="camera horizontal field of view, if the camera does not say")
    ap.add_argument("--gain", type=float, default=0.8)
    ap.add_argument("--deadband", type=float, default=0.03, help="centred within this fraction of the frame")
    ap.add_argument("--smooth", type=float, default=0.3, help="0..0.9 blend of each move with the last")
    ap.add_argument("--smooth-features", type=float, default=0.5, help="0..1 weight of the newest detection (lower = steadier)")
    ap.add_argument("--max-step", type=float, default=40.0, help="longest single move, mm")
    ap.add_argument("--max-speed", type=float, default=150.0, help="gripper speed limit, mm/s")
    ap.add_argument("--approach-speed", type=float, default=80.0, help="closing speed, mm/s")
    ap.add_argument("--arm-speed", type=float, help="real arm joint speed, deg/s (UFactory: 15 slow, 30 medium, 60 fast)")
    ap.add_argument("--probe-mm", type=float, default=20.0)
    ap.add_argument("--grip-width-mm", type=float, help="the object's width: the fingers close to this opening and cannot squeeze "
                    "past it. REQUIRED for a real run. (This gripper has no working force limit; a plain close crushed a cup.)")
    ap.add_argument("--grip-speed", type=int, default=300, help="closing speed 1-5000 (low = gentle)")
    ap.add_argument("--lift-mm", type=float, default=120.0)
    ap.add_argument("--max-cycles", type=int, default=600)
    ap.add_argument("--ui", action="store_true", help="serve the local web page (camera view, chat, STOP) and open it; "
                    "typing 'pick up the cup' starts the run, 'yes' answers the gripper question")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--yes", action="store_true", help="skip the typed confirmations (not recommended)")
    if preset:
        ap.set_defaults(bounds=PRESETS[preset]["bounds"])
    args = ap.parse_args()

    real, bounds = args.sink == "viam", floats(args.bounds)
    if real and args.grip_width_mm is None:
        raise SystemExit("give --grip-width-mm (the object's width in mm) before a real run: the gripper closes to that opening.")
    stop_cm = args.rotate_cm if args.wrist_flat else args.stop_cm       # where the tracking stops
    lo, hi = np.array(bounds[0::2], float), np.array(bounds[1::2], float)
    floor = table_floor_z(args.table_z_mm, args.tool_length_mm, args.table_margin_mm)
    lo[2] = max(lo[2], floor)                          # never command a pose that puts the fingers into the table
    print(f"table-safe floor: the tool never goes below z = {lo[2]:.0f} mm "
          f"(table top {args.table_z_mm:.0f}, fingertips {args.tool_length_mm:.0f} mm below the tool, margin {args.table_margin_mm:.0f})")
    model, dargs = load_detector(args.object)
    robot = ViamRobot(args.viam_address, args.arm_name, args.gripper_name, None, TableToWorld(),
                      workspace=bounds[:4], max_z_mm=bounds[5])
    rng = Range(args.object_height_mm, args.hfov)
    try:
        _, intr = robot.camera_info(args.camera)
        if intr:
            rng.focal_px, rng.focal_w = intr["fx"], intr["w"]
    except Exception:
        pass
    lock, flt = Lock(max_jump_frac=0.5, lost_after=10 ** 6), FeatureFilter(alpha=args.smooth_features, beta=0.0)
    flags = {"stop": False, "paused": False, "yes": None, "go": not args.ui}
    view = {"frame": None, "text": "starting", "state": "starting", "aim": None, "cands": []}
    ui = None

    def confirm(question):
        if args.yes:
            return True
        if ui:                                              # answer in the chat box instead of the terminal
            flags["yes"] = None
            ui.say(question + " Reply 'yes' to continue or 'no' to cancel.")
            t0 = time.monotonic()
            while flags["yes"] is None and not flags["stop"] and time.monotonic() - t0 < 90:
                time.sleep(0.2)
            return bool(flags["yes"])
        return input(f"{question} Type 'yes' to continue: ").strip().lower() == "yes"

    def show(state, text):
        """Update the status and push the annotated camera frame to the page."""
        view["state"], view["text"] = state, text
        if not ui or view["frame"] is None:
            return
        img, h = view["frame"].copy(), view["frame"].shape[0]
        for c in view["cands"]:                             # every detection in grey, the locked one in green
            cv2.rectangle(img, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (160, 160, 160), 1)
        if lock.locked:
            b = [int(v) for v in lock.box]
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 200, 0), 3)
        if view["aim"] is not None:                         # where the gripper points: the cup should sit on this line
            y = int((0.5 + view["aim"]) * h)
            cv2.line(img, (0, y), (img.shape[1], y), (0, 140, 255), 1)
        for i, line in enumerate((("LIVE ARM" if real else "DRY RUN - nothing moves"), text[:60])):
            cv2.putText(img, line, (12, 28 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(img, line, (12, 28 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            ui.set_frame(buf.tobytes())

    def position():
        p = robot.get_pose()
        return np.array(robot.table.to_robot(p.x, p.y, p.z))

    def see(n_tries=1):
        """-> (features, distance cm, frame) of the locked object, or (None, None, frame)."""
        frame = robot.get_frame(args.camera)
        if frame is None:
            return None, None, None
        view["frame"] = frame
        cands = object_candidates(detect(model, frame, dargs), args.object)
        view["cands"] = cands
        lock.update(cands, frame.shape[1])
        if not lock.locked:
            return None, None, frame
        h, w = frame.shape[:2]
        s = features(lock.box, w, h)
        return s, rng.cm(s[2] * h, w), frame

    def at_rest(n=5):
        got = []
        for _ in range(n):
            s, _, _ = see()
            if s is not None:
                got.append(s)
            time.sleep(0.1)
        return np.median(got, axis=0) if len(got) >= n - 1 else None

    def move_by(delta):
        robot.move_xyz(*np.clip(position() + np.asarray(delta, float), lo, hi), wait=True)

    def probe():
        """How the object shifts in the image per mm of gripper motion (x, y, z); returns to the exact start after each."""
        start, J = position(), np.zeros((3, 3))
        if start[2] < lo[2] - 1.0:
            raise SystemExit(f"the tool is at z {start[2]:.0f} mm, below the table-safe floor {lo[2]:.0f} mm: raise the arm first")
        for i in range(3):
            before = at_rest()
            if before is None:
                raise SystemExit("probe: cannot see the object steadily; get it in view and keep it still")
            d = np.zeros(3)
            for sign in (1.0, -1.0):
                d[i] = sign * args.probe_mm
                if abs(np.clip(start + d, lo, hi)[i] - start[i]) >= 5.0:
                    break
            else:
                raise SystemExit(f"probe: no room to move along {'xyz'[i]} inside the bounds")
            target = np.clip(start + d, lo, hi)
            robot.move_xyz(*target, wait=True)
            time.sleep(0.5)
            after = at_rest()
            robot.move_xyz(*start, wait=True)
            time.sleep(0.5)
            if after is None:
                raise SystemExit("probe: lost the object during a probe move; put it nearer the middle of the view")
            J[:, i] = (after - before) / (target - start)[i]
        print("camera response J (per mm):\n", np.round(J, 5))
        problem = check_jacobian(J)
        if problem:
            raise SystemExit("probe rejected: " + problem + ". The arm is back at its start.")
        return J

    def wrist_flat():
        """Rotate ONLY the wrist pitch joint (not the final roll joint) in 10-degree steps until the gripper is flat.
        Tool direction (o_x, o_y, o_z): o_z = -1 is pointing straight down, 0 is flat, o_x > 0 is pointing forward.
        Stops early if the tool gets too low for a flat gripper over the table. Returns True when it is flat."""
        wj = args.wrist_joint - 1
        floor_h = table_floor_z(args.table_z_mm, args.flat_floor_mm, args.table_margin_mm)

        def say(msg):
            print(msg, flush=True)
            if ui:
                ui.say(msg)

        if position()[2] < floor_h:
            say(f"The tool is at z {position()[2]:.0f} mm, lower than {floor_h:.0f} mm: a flat gripper would hit the table. Not rotating.")
            return False
        j0, p0 = robot.get_joints()[wj], robot.get_pose()
        sign = args.wrist_sign
        if not sign:                                         # 10-degree test each way: which flattens it and points forward?
            options = []
            for sg in (1, -1):
                robot.set_joint(wj, j0 + sg * 10.0)
                p1 = robot.get_pose()
                robot.set_joint(wj, j0)
                if abs(p1.o_z) < abs(p0.o_z) - 0.02:
                    options.append((sg, p1.o_x > p0.o_x))
            if not options:
                say(f"Joint {args.wrist_joint} does not flatten the gripper from this pose. Not rotating; use --wrist-sign / check the joint.")
                return False
            sign = next((sg for sg, forward in options if forward), options[0][0])
        total = 0.0
        while total < args.wrist_deg:
            step = min(10.0, args.wrist_deg - total)
            total += step
            robot.set_joint(wj, j0 + sign * total)
            p = robot.get_pose()
            if position()[2] < floor_h:
                say(f"Stopped at {total:.0f} degrees: the tool is too low for a flat gripper.")
                return False
            if abs(p.o_z) < 0.05:
                break
        p = robot.get_pose()
        say(f"Joint {args.wrist_joint} rotated {sign * total:+.0f} degrees; the tool direction is now "
            f"({p.o_x:+.2f}, {p.o_y:+.2f}, {p.o_z:+.2f}).")
        return True

    def grasp():
        if args.wrist_flat:
            if not confirm(f"Arrived at {stop_cm:.0f} cm. ROTATE joint {args.wrist_joint} ONLY (up to {args.wrist_deg:.0f} "
                           "degrees, in 10-degree steps) to make the gripper flat. Keep the area around the arm clear."):
                return
            if not wrist_flat():
                return
            advance_mm = max(args.rotate_cm - args.close_cm, 0.0) * 10.0
            if not confirm(f"ADVANCE {advance_mm:.0f} mm along the gripper axis (camera {args.rotate_cm:.0f} -> {args.close_cm:.0f} cm "
                           f"from the cup), then CLOSE the gripper (to a {args.grip_width_mm:.0f} mm opening) and lift."):
                return
            pn = robot.get_pose()
            robot._orient = None                         # hold the NEW (flat) orientation for the moves that follow
            goal = np.clip(advance_along(position(), (pn.o_x, pn.o_y, pn.o_z), advance_mm), lo, hi)
            robot.move_xyz(*goal, wait=True)
            time.sleep(0.5)
        elif not confirm(f"Arrived at {stop_cm:.0f} cm. About to CLOSE the gripper (to a {args.grip_width_mm:.0f} mm opening) and lift."):
            return
        pos_cmd = robot.close_to_width(args.grip_width_mm, speed=args.grip_speed)   # position, not force
        time.sleep(2.0)
        print(f"gripper commanded to {pos_cmd:.0f}, now {robot.gripper_command({'get': True})}", flush=True)
        pos = position()
        robot.move_xyz(pos[0], pos[1], min(pos[2] + args.lift_mm, bounds[5]), wait=True)
        try:
            print("holding something:", bool(robot._run(robot.gripper.is_holding_something()).value))
        except Exception:
            pass

    if args.ui:
        def on_text(text):
            t = text.lower().strip()
            if "stop" in t:
                flags["stop"] = True
                if real:
                    robot.stop()
                return "Stopped. Nothing else will move."
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
                flags["go"] = True
                return (f"Going to get the {args.object}: probe, line up, approach, grasp, lift." if real else
                        f"DRY RUN: showing how I would get the {args.object}; nothing will move.")
            if "see" in t or "status" in t:
                return view["text"]
            return "I understand: pick up the cup, stop, pause, resume, yes / no, what do you see."

        ui = UIServer(on_text, lambda: {"mode": "LIVE ARM" if real else "DRY RUN", "state": view["state"],
                                        "target": args.object}, args.port).start()
        print("UI:", ui.url, flush=True)
        webbrowser.open(ui.url)

    try:
        if ui:                                              # watch the object until told to go
            while not flags["go"] and not flags["stop"]:
                s0, d0, _ = see()
                show("waiting", ("object in view" if s0 is not None else "object NOT in view") + " - say 'pick up the cup'")
                time.sleep(0.1)
            if flags["stop"]:
                return
        if real:
            if not ui and not confirm(f"The arm WILL move to pick up '{args.object}' (bounds {bounds}). E-stop in reach?"):
                return
            if args.arm_speed:
                try:
                    robot.arm_command({"set_speed": args.arm_speed})
                except Exception as e:
                    print(f"could not set the arm speed ({type(e).__name__}); using the configured speed")
            robot.set_gripper(True)
            J = probe()
        else:
            J = None
            print("DRY RUN: nothing will move; printing where the gripper would go"
                  + ("; at the grip distance it would rotate the wrist joint flat" if args.wrist_flat else ""))
        target, prev, last_t, misses, errs = np.zeros(3), np.zeros(3), time.monotonic(), 0, []
        for cycle in range(1, args.max_cycles + 1):
            if flags["stop"]:
                break
            if flags["paused"]:
                time.sleep(0.2)
                continue
            s, dist, frame = see()
            now = time.monotonic()
            dt, last_t = min(max(now - last_t, 0.05), 1.0), now
            if s is None:
                misses += 1
                show("searching", "object not seen")
                print(f"[{cycle:3}] object not seen ({misses})", flush=True)
                flt.reset()
                if misses > 40:
                    raise SystemExit("lost the object for too long; stopping")
                time.sleep(0.1)
                continue
            misses, s = 0, np.asarray(flt.update(s, now))
            if args.tip_offset_mm and dist:                  # line it up where the GRIPPER points, not the camera axis
                f_px = (rng.focal_px * frame.shape[1] / rng.focal_w if rng.focal_px
                        else frame.shape[1] / (2 * np.tan(np.radians(args.hfov) / 2)))
                target[1] = (-1 if args.tip_flip else 1) * tip_target_v(dist, args.tip_offset_mm, f_px, frame.shape[0])
            view["aim"] = float(target[1])
            err = target[:2] - s[:2]
            e = float(np.max(np.abs(err)))
            errs = (errs + [e])[-8:]
            arrived = dist is not None and dist <= stop_cm and e < 2 * args.deadband
            show("arrived" if arrived else "tracking", f"offset u{s[0]:+.2f} v{s[1]:+.2f} aim v{target[1]:+.2f}"
                 + ("" if dist is None else f" {dist:.0f} cm"))
            print(f"[{cycle:3}] offset u{s[0]:+.2f} v{s[1]:+.2f} aim v{target[1]:+.2f}  "
                  + ("" if dist is None else f"{dist:.0f} cm") + f"  seen {len(view['cands'])}"
                  + ("  ARRIVED" if arrived else ""), flush=True)
            if arrived:
                if real:
                    grasp()
                else:
                    print("DRY RUN: would close the gripper and lift here.")
                break
            if len(errs) == 8 and all(b > a + 0.003 for a, b in zip(errs, errs[1:])):
                raise SystemExit("the offset keeps growing: moving the wrong way. Stopping.")
            if not real:
                time.sleep(0.2)
                continue
            # centre on the aim point and, in proportion to how well it is lined up, close the distance
            closer = (args.approach_speed / 10.0 * dt) * approach_factor(err, args.deadband) if dist else 0.0
            d_goal = max(stop_cm, dist - closer) if dist else None
            tgt = np.array([target[0], target[1], s[2] * dist / d_goal if d_goal else s[2]])
            move = servo_step(J, s, tgt, (0, 1, 2) if d_goal else (0, 1), args.gain, args.max_step)
            move = smooth_move(prev, move, args.smooth, args.max_speed * dt)
            prev = move
            try:
                move_by(move)
            except Exception as e:
                if flags["stop"]:                        # 'stop' cancels the move in flight on purpose
                    break
                raise
        else:
            print("gave up: did not arrive within the cycle limit")
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        if real:
            try:
                robot.stop()
            except Exception as e:
                print("stop failed:", e)
        robot.close()


if __name__ == "__main__":
    main()
