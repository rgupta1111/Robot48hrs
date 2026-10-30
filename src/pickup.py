"""Pick up a named object with the gripper (camera on the gripper, detection by a Viam vision service).

Sequence: open gripper -> centre the object in the image (visual servoing, response learned by small probe moves) ->
descend while staying centred -> close gripper at --grasp-distance-mm -> lift. See mimic/pickup.py.

DRY RUN by default (--sink sim): reads the camera and prints what it WOULD do; the arm and gripper do not move.
Real arm (low speed limit in the Viam config, hand on the e-stop):
    scripts/run.sh src/pickup.py --viam-address <machine>.viam.cloud --preset lab --vision cup-detector --object cup
    scripts/run.sh src/pickup.py --viam-address <machine>.viam.cloud --preset lab --vision cup-detector --object cup --sink viam
With --sink viam it asks for a typed "yes" before it moves at all and again before it closes the gripper.
Tune --grasp-distance-mm (camera-to-object distance at which the gripper surrounds the object) and --target-uv (where the
object appears in the image when the gripper is over it; the camera is offset from the gripper) in dry runs first.
Distance comes from the object's apparent height: d = fy * --object-height-mm / height_px.
Ctrl+C stops the arm.
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
from mimic.pickup import PickupConfig, PickupPlanner, check_jacobian, smooth_move  # noqa: E402
from mimic.servo import FeatureFilter, estimate_jacobian, features  # noqa: E402
from mimic.ui import UIServer  # noqa: E402
from mimic.size import WidthEstimator, grip_position, scale_focal  # noqa: E402
from tidy.robot import TableToWorld, ViamRobot  # noqa: E402


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", choices=list(PRESETS))
    preset = pre.parse_known_args()[0].preset
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS))
    ap.add_argument("--viam-address", required=True)
    ap.add_argument("--vision", required=True, help="name of the Viam vision service that detects the object")
    ap.add_argument("--object", default="cup", help="class name to pick up (matched against the detector's labels)")
    ap.add_argument("--camera", default="cam")
    ap.add_argument("--arm-name", default="arm")
    ap.add_argument("--gripper-name", default="gripper")
    ap.add_argument("--viam-insecure", action="store_true")
    ap.add_argument("--sink", choices=["sim", "viam"], default="sim", help="sim = dry run, nothing moves")
    ap.add_argument("--table-to-world", default="0,0,0,0")
    ap.add_argument("--bounds", default="250,600,-300,300,0,450", help="xmin,xmax,ymin,ymax,zmin,zmax mm; moves are clamped")
    ap.add_argument("--min-confidence", type=float, default=0.35)
    ap.add_argument("--object-height-mm", type=float, default=100.0)
    ap.add_argument("--segmenter", help="Viam vision service that gives 3D object positions (e.g. object-segmenter); "
                    "its depth gives the true camera-to-object distance and lets the width be estimated")
    ap.add_argument("--arm-speed", type=float, default=30.0,
                    help="real arm joint speed in deg/s, set over DoCommand when a run starts (UFactory: 15 slow, 30 medium, 60 fast)")
    ap.add_argument("--max-torque", type=int, default=30, help="grip force limit 1-100%% (UFactory G2 gripper only)")
    ap.add_argument("--grip-speed", type=int, default=1500)
    ap.add_argument("--units-per-mm", type=float, default=10.0, help="gripper position units per mm of opening (check it)")
    ap.add_argument("--squeeze-mm", type=float, default=1.0, help="close this much tighter than the estimated width")
    ap.add_argument("--min-width-mm", type=float, default=10.0)
    ap.add_argument("--max-width-mm", type=float, default=80.0, help="refuse to grasp anything wider than this")
    ap.add_argument("--grasp-distance-mm", type=float, default=150.0)
    ap.add_argument("--target-uv", default="0,0", help="image offset (fractions of the frame) of the object at grasp")
    ap.add_argument("--probe-mm", type=float, default=20.0)
    ap.add_argument("--gain", type=float, default=0.4)
    ap.add_argument("--smooth", type=float, default=0.5, help="0..0.9 blend of each move with the previous one (higher = smoother)")
    ap.add_argument("--feature-alpha", type=float, default=0.5, help="0..1 weight of the newest detection in the filtered "
                    "cup position (lower = steadier, laggier)")
    ap.add_argument("--max-speed", type=float, default=60.0, help="gripper speed limit during moves, mm/s")
    ap.add_argument("--max-step", type=float, default=30.0)
    ap.add_argument("--descend-step", type=float, default=25.0)
    ap.add_argument("--lift-mm", type=float, default=120.0)
    ap.add_argument("--settle", type=float, default=0.8, help="seconds to wait after a move before reading the camera")
    ap.add_argument("--max-cycles", type=int, default=200)
    ap.add_argument("--ui", action="store_true", help="serve the local web page (live camera, status, STOP) and open it")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--yes", action="store_true", help="skip the typed confirmations (not recommended)")
    if preset:
        ap.set_defaults(bounds=PRESETS[preset]["bounds"])
    args = ap.parse_args()

    real = args.sink == "viam"
    bounds = floats(args.bounds)
    robot = ViamRobot(args.viam_address, args.arm_name, args.gripper_name, None, TableToWorld(*floats(args.table_to_world)),
                      workspace=bounds[:4], max_z_mm=bounds[5], insecure=args.viam_insecure)
    _, intr = robot.camera_info(args.camera)
    fy = intr["fy"] if intr else None
    lo, hi = np.array(bounds[0::2], float), np.array(bounds[1::2], float)
    widthest = WidthEstimator(intr["fx"]) if (args.segmenter and intr) else None
    if args.segmenter and not intr:
        print("camera reports no intrinsics: width estimation off, the gripper will close with its default force")

    lock = {"centre": None, "seg_ok": bool(args.segmenter)}
    view = {"frame": None, "box": None, "text": "starting"}
    flags = {"stop": False, "paused": False, "yes": None, "go": not args.ui}

    def see():
        """-> (features, distance mm or None) of the locked cup. The first pick is the NEAREST cup (largest box, i.e.
        closest to the camera); after that it stays on the cup nearest to where the lock last was, so a second cup
        in view never steals it. (None, None) when no cup is seen."""
        frame, dets = robot.detect(args.vision, args.camera, args.min_confidence)
        view["frame"], view["box"] = frame, None
        dets = [d for d in dets if args.object.lower() in d.class_name.lower()]
        if not dets:
            return None, None
        h_img, w_img = frame.shape[:2] if frame is not None else ((intr["h"], intr["w"]) if intr else (720, 1280))
        centre = lambda d: ((d.x_min + d.x_max) / 2, (d.y_min + d.y_max) / 2)
        if lock["centre"] is None:
            d = max(dets, key=lambda d: (d.y_max - d.y_min) * (d.x_max - d.x_min))      # nearest = biggest on screen
        else:
            lx, ly = lock["centre"]
            d = min(dets, key=lambda d: np.hypot(centre(d)[0] - lx, centre(d)[1] - ly))
            if np.hypot(centre(d)[0] - lx, centre(d)[1] - ly) > 0.25 * w_img:           # jumped: lost our cup
                lock["centre"] = None
                return None, None
        lock["centre"] = centre(d)
        box = (d.x_min, d.y_min, d.x_max, d.y_max)
        view["box"] = box
        s = features(box, w_img, h_img)
        dist = fy * args.object_height_mm / (s[2] * h_img) if fy and s[2] > 0 else None
        if lock["seg_ok"]:                                  # true depth from the point cloud beats the height guess
            try:
                _, objs = robot.objects(args.segmenter, args.camera)
            except Exception as e:                          # e.g. a dropped stream: carry on with the height estimate
                print(f"segmenter failed ({type(e).__name__}); using height-based distance, no width estimate", flush=True)
                lock["seg_ok"] = False
                objs = []
            objs = [o for o in objs if args.object.lower() in o[0].lower() and o[1][2] > 0]
            if objs and intr:
                sc = w_img / intr["w"]                      # match the cloud object whose projection is nearest the box
                proj = lambda o: (intr["cx"] * sc + intr["fx"] * sc * o[1][0] / o[1][2],
                                  intr["cy"] * sc + intr["fy"] * sc * o[1][1] / o[1][2])
                cx0, cy0 = centre(d)
                o = min(objs, key=lambda o: np.hypot(proj(o)[0] - cx0, proj(o)[1] - cy0))
                if np.hypot(proj(o)[0] - cx0, proj(o)[1] - cy0) < 0.25 * w_img:
                    dist = o[1][2]
                    if widthest is not None:
                        widthest.fx = scale_focal(intr["fx"], intr["w"], w_img)
                        widthest.update(box, dist)
        return s, dist

    def position():
        p = robot.get_pose()
        return np.array(robot.table.to_robot(p.x, p.y, p.z))

    def confirm(question):
        if args.yes:
            return True
        if ui:                                              # answer in the chat box instead of the terminal
            flags["yes"] = None
            ui.say(question + " Reply 'yes' to continue or 'no' to cancel.")
            t0 = time.monotonic()
            while flags["yes"] is None and not flags["stop"] and time.monotonic() - t0 < 60:
                time.sleep(0.2)
            return bool(flags["yes"])
        return input(f"{question} Type 'yes' to continue: ").strip().lower() == "yes"

    def move_by(delta):
        target = np.clip(position() + np.asarray(delta, float), lo, hi)
        robot.move_xyz(*target, wait=True)
        time.sleep(args.settle)

    def grasp(dry):
        """Close onto the object with just enough squeeze: to the estimated width, force-limited where supported."""
        r = widthest.reading() if widthest else None
        if r is None:
            print("no width estimate (need --segmenter and camera intrinsics): plain close, force capped at "
                  f"{args.max_torque}% where the gripper supports it")
            if not dry and confirm("About to CLOSE the gripper."):
                try:   # UFactory G2 only; a G1 has no force setting, so this just fails and the close is unlimited
                    robot.gripper_command({"set_gripper_torque": args.max_torque})
                except Exception as e:
                    print(f"could not cap the grip force ({type(e).__name__}: {e}); closing at the default force")
                robot.set_gripper(False)
            return
        if not args.min_width_mm <= r.width_mm <= args.max_width_mm:
            print(f"width {r.width_mm:.0f} mm is outside {args.min_width_mm:.0f}-{args.max_width_mm:.0f} mm: NOT grasping")
            return
        pos = grip_position(r.width_mm, args.units_per_mm, args.squeeze_mm)
        print(f"object width {r.width_mm:.1f} +/- {r.sigma_mm:.1f} mm at {r.depth_mm:.0f} mm ({r.n} samples) -> "
              f"close to position {pos:.0f}, torque limit {args.max_torque}%" + (" (dry run)" if dry else ""))
        if dry or not confirm("About to CLOSE the gripper to that width."):
            return
        try:   # UFactory G2: close with a force limit
            robot.gripper_command({"grab_with_torque": {"position": int(pos), "speed": args.grip_speed,
                                                          "torque": args.max_torque}})
        except Exception as e:
            print(f"grab_with_torque unavailable ({type(e).__name__}); closing to the position without a force limit")
            robot.gripper_command({"set": pos})

    def probe():
        """Learn how the object's image features respond to moving +x, +y, +z. Each axis is measured against a fresh
        at-rest reading, and the arm is sent back to the exact start position (absolute, not 'minus the move') so
        clamping or positioning error cannot build up from one probe to the next."""
        def at_rest(n=5):
            got = []
            for _ in range(n):
                s_, _ = see()
                if s_ is not None:
                    got.append(s_)
                time.sleep(0.15)
            return np.median(got, axis=0) if len(got) >= n - 1 else None

        start = position()
        J = np.zeros((3, 3))
        for i in range(3):
            before = at_rest()
            if before is None:
                raise SystemExit("probe: cannot see the object steadily; get it in view and keep it still")
            d = np.zeros(3)
            for sign in (1.0, -1.0):                        # move away from a bound if there is no room this way
                d[i] = sign * args.probe_mm
                if abs(np.clip(start + d, lo, hi)[i] - start[i]) >= 5.0:
                    break
            else:
                raise SystemExit(f"probe: no room to move along axis {'xyz'[i]} inside the bounds")
            target = np.clip(start + d, lo, hi)
            robot.move_xyz(*target, wait=True)
            time.sleep(args.settle + 0.4)                   # let the arm stop and the camera catch up
            after = at_rest()
            try:
                robot.move_xyz(*start, wait=True)
            except Exception as e:
                raise SystemExit(f"probe: could not return to the start position ({type(e).__name__}); the arm is up to "
                                 f"{args.probe_mm:.0f} mm away along {'xyz'[i]} - check it before retrying")
            time.sleep(args.settle + 0.4)
            if after is None:
                raise SystemExit("probe: lost the object during a probe move; centre it in view first")
            J[:, i] = (after - before) / (target - start)[i]
        print("learned image response J (per mm):\n", np.round(J, 5))
        problem = check_jacobian(J)
        if problem:
            raise SystemExit("probe rejected: " + problem + ". The arm was returned to its start; nothing else moved.")
        return J

    ui = None
    if args.ui:
        def on_text(text):
            t = text.lower()
            if "stop" in t:
                flags["stop"] = True
                if real:
                    robot.stop()
                return "Stopped. Nothing else will move."
            if any(w in t for w in ("pick", "get", "grab", "fetch", "go to", "bring")):
                if flags["go"]:
                    return "Already on it."
                flags["go"] = True
                return (f"Going to get the {args.object}: centre, descend, grasp, lift." if real else
                        f"DRY RUN: showing how I would get the {args.object}; nothing will move.")
            if t.strip() in ("yes", "y", "confirm", "ok"):
                flags["yes"] = True
                return "OK."
            if t.strip() in ("no", "n", "cancel"):
                flags["yes"] = False
                return "Cancelled."
            if "pause" in t:
                flags["paused"] = True
                return "Paused."
            if "resume" in t or "continue" in t:
                flags["paused"] = False
                return "Resumed."
            if "see" in t or "status" in t:
                return view["text"]
            return "I understand: pick up the cup, stop, pause, resume, status / what do you see, yes / no."

        def get_status():
            return {"mode": "LIVE ARM" if real else "DRY RUN", "state": view.get("state", "starting"), "target": args.object}

        ui = UIServer(on_text, get_status, args.port).start()
        print("UI:", ui.url, flush=True)
        webbrowser.open(ui.url)

    def show(state, text):
        view["text"], view["state"] = text, state
        if ui and view["frame"] is not None:
            img = view["frame"].copy()
            if view["box"] is not None:
                b = [int(v) for v in view["box"]]
                cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 200, 0), 3)
            for i, line in enumerate((("LIVE ARM" if real else "DRY RUN - nothing moves"), view["text"][:70])):
                cv2.putText(img, line, (14, 34 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
                cv2.putText(img, line, (14, 34 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                ui.set_frame(buf.tobytes())

    cfg = PickupConfig(target_uv=floats(args.target_uv), gain=args.gain, max_step_mm=args.max_step,
                       descend_step_mm=args.descend_step, grasp_distance_mm=args.grasp_distance_mm,
                       lift_mm=args.lift_mm, max_cycles=args.max_cycles)

    def run_pickup(J):
        """One pick-up attempt. Returns how it ended."""
        planner = PickupPlanner(J, cfg)
        lock["centre"] = None
        if widthest is not None:
            widthest.reset()
        flt, misses, prev_move, last_t = FeatureFilter(alpha=args.feature_alpha, beta=0.0), 0, np.zeros(3), time.monotonic()
        while not flags["stop"]:
            if flags["paused"]:
                time.sleep(0.2)
                continue
            s, dist = see()
            now = time.monotonic()
            dt, last_t = min(max(now - last_t, 0.2), 2.0), now
            if s is not None:                               # steady the detection (box jitter) before deciding a move
                s, misses = np.asarray(flt.update(s, now)), 0
            else:
                misses += 1
                if misses > 3:
                    flt.reset()
            act = planner.step(s, dist)
            seen = "not seen" if s is None else f"u{s[0]:+.2f} v{s[1]:+.2f} h{s[2]:.2f}" + (f" {dist:.0f}mm" if dist else "")
            print(f"[{planner.cycles:3}] {planner.state:8} {act.kind:10} {act.note:28} {seen}"
                  + (f"  move {act.move[0]:+.0f} {act.move[1]:+.0f} {act.move[2]:+.0f}" if act.kind == "move" else ""), flush=True)
            show(planner.state, f"{act.kind} {act.note} - {seen}")
            if act.kind in ("done", "abort"):
                return act.kind + (f" ({act.note})" if act.note else "")
            if not real:
                if act.kind == "grip_close":
                    grasp(dry=True)
                time.sleep(0.3)
                continue
            if act.kind == "grip_open":
                robot.set_gripper(True)
                time.sleep(1.0)
            elif act.kind == "grip_close":
                grasp(dry=False)
                time.sleep(1.5)
            elif act.kind == "move":
                m = smooth_move(prev_move, act.move, args.smooth, args.max_speed * dt)
                prev_move = m
                move_by(m)
            if act.kind != "move":
                prev_move = np.zeros(3)
        return "stopped"

    J = None
    try:
        while not flags["stop"]:
            if not flags["go"]:                             # UI mode: watch the cup until told to get it
                s, dist = see()
                show("waiting", ("cup in view" if s is not None else "no cup in view") + " - say 'pick up the cup'")
                time.sleep(0.2)
                continue
            flags["go"] = not args.ui                       # one run per command in UI mode
            if J is None:
                if real:
                    if not ui and not confirm(f"The arm WILL move to pick up '{args.object}' (bounds {bounds}). E-stop in reach?"):
                        return
                    try:
                        robot.arm_command({"set_speed": args.arm_speed})
                        print(f"arm speed set to {args.arm_speed:.0f} deg/s", flush=True)
                    except Exception as e:
                        print(f"could not set the arm speed ({type(e).__name__}: {e}); using the configured speed", flush=True)
                    if ui:
                        ui.say("Learning how the camera responds: three small probe moves (about 20 mm each).")
                    try:
                        J = probe()
                    except SystemExit as e:
                        if not ui:
                            raise
                        ui.say(f"Cannot start: {e}")
                        continue
                else:
                    J = np.array([[0.0, -0.004, 0.0], [-0.003, 0.0, 0.0], [0.0, 0.0, 0.001]])   # placeholder, never used to move
                    print("DRY RUN: nothing will move; showing what the planner would do each cycle")
            try:
                result = run_pickup(J)
            except Exception as e:                          # keep the page alive and say what happened
                if not ui:
                    raise
                try:
                    robot.stop()
                except Exception:
                    pass
                ui.say(f"Run failed and the arm was told to stop: {type(e).__name__}: {e}")
                continue
            print("result:", result, flush=True)
            if not ui:
                break
            ui.say(f"Pickup finished: {result}. Say 'pick up the cup' to try again.")
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
