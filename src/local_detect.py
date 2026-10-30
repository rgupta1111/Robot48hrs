"""Detect objects locally with YOLO (weights in weights/) on the Viam camera, a webcam, an image or a video.

    scripts/run.sh src/local_detect.py --viam-address <machine>.viam.cloud            # Viam camera "cam", yolov8s
    scripts/run.sh src/local_detect.py --webcam
    scripts/run.sh src/local_detect.py --image photo.jpg --no-display --save out.jpg

Small objects (the usual miss): raise --imgsz (960/1280), lower --conf, or add --tile (runs the model on overlapping
crops, slower but finds far smaller things). Things COCO has no class for (a blue cylinder) need the open-vocabulary
model:  --model weights/yolov8s-worldv2.pt --prompts "blue cylinder,cardboard box,paper"
Object lock: put a marker on one object and keep it locked while it moves (re-acquires if it is lost):
    --lock-color blue                      lock the most blue-looking object (fast, no model needed for the target)
    --lock "blue cylinder" --model weights/yolov8s-worldv2.pt --prompts "blue cylinder"     lock by model name
    click any box in the window to lock that object instead; l unlocks, r re-acquires.
Distance to the locked object comes from its size on screen (pinhole model; a flat camera has no depth):
    distance = focal_px * real_height / height_px.  Give the object's true height with --real-height-mm, or hold
    it at a known distance and press c (--calib-cm, default 30) to calibrate - far more accurate than guessing.
Keys: q / Esc quit, s save snapshot, t toggle tiling, +/- change --conf, l unlock, r re-acquire, c calibrate.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_detector(lock_name, model_path=None, prompts=None, conf=None, imgsz=960, standard="yolov8s.pt"):
    """YOLO model + detect() settings for following an object by name. With no model given, use the standard
    yolov8s if it has a class matching lock_name, otherwise the open-vocabulary world model prompted with the name."""
    from argparse import Namespace

    from ultralytics import YOLO
    weights = os.path.join(ROOT, "weights")
    world = False
    if model_path:
        model = YOLO(model_path)
        world = hasattr(model, "set_classes") and "world" in os.path.basename(model_path).lower()
    else:
        model = YOLO(os.path.join(weights, standard))
        if not any(lock_name.lower() in n.lower() for n in model.names.values()):
            print(f"no standard class matches {lock_name!r}: using the open-vocabulary model", flush=True)
            model, world = YOLO(os.path.join(weights, "yolov8s-worldv2.pt")), True
    if world:
        # scores are relative to the prompt set, so a lone prompt scores low; add everyday background classes
        names = [p.strip() for p in prompts.split(",")] if prompts else [lock_name] + [
            b for b in ("person", "laptop", "table", "chair", "box", "bag") if b not in lock_name.lower()]
        model.set_classes(names)
    dargs = Namespace(imgsz=imgsz, conf=conf if conf is not None else (0.15 if world else 0.25), classes=None,
                      class_ids=None, tile=False, tile_size=480, tile_overlap=0.25, device=pick_device())
    print(f"detector: {'open-vocabulary' if world else 'standard'} model, looking for {lock_name!r}, "
          f"conf {dargs.conf}, device {dargs.device}", flush=True)
    return model, dargs


def object_candidates(dets, name):
    """Detections whose class contains name, best first -> [(x1, y1, x2, y2, conf, label)]."""
    return sorted((d for d in dets if name.lower() in d[5].lower()), key=lambda d: -d[4])


def roi_candidates(model, dargs, frame, box, name, conf=0.08, imgsz=640):
    """Look for the object again, only around where it was, at a lower confidence and effectively zoomed in:
    a small cup the full-frame pass drops is usually still found in a crop 3x its size."""
    from copy import copy
    h, w = frame.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    side = int(min(max(3 * max(bw, bh), 256), w, h))
    x0 = int(max(0, min(w - side, (box[0] + box[2]) / 2 - side / 2)))
    y0 = int(max(0, min(h - side, (box[1] + box[3]) / 2 - side / 2)))
    a = copy(dargs)
    a.conf, a.imgsz, a.tile = conf, imgsz, False
    found = detect(model, frame[y0:y0 + side, x0:x0 + side], a)
    return object_candidates([(d[0] + x0, d[1] + y0, d[2] + x0, d[3] + y0, d[4], d[5]) for d in found], name)


def crop_gray(frame, box):
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY) if x2 - x1 >= 8 and y2 - y1 >= 8 else None


def template_candidates(frame, template, box, label, thresh=0.6, reach=0.3):
    """Last resort when no detector finds the object: slide the last picture of it around the old position."""
    if template is None:
        return []
    h, w = frame.shape[:2]
    th, tw = template.shape
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    rx, ry = max(reach * w, 2 * tw), max(reach * h, 2 * th)
    x0, y0 = int(max(0, cx - rx)), int(max(0, cy - ry))
    x1, y1 = int(min(w, cx + rx)), int(min(h, cy + ry))
    window = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    if window.shape[0] <= th or window.shape[1] <= tw:
        return []
    res = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    if score < thresh:
        return []
    return [(x0 + loc[0], y0 + loc[1], x0 + loc[0] + tw, y0 + loc[1] + th, float(score), label)]


def tiles(shape, size, overlap):
    """Top-left corners and size of overlapping square-ish crops covering the frame."""
    h, w = shape[:2]
    step = max(int(size * (1 - overlap)), 1)
    xs = list(range(0, max(w - size, 0) + 1, step)) or [0]
    ys = list(range(0, max(h - size, 0) + 1, step)) or [0]
    if xs[-1] + size < w:
        xs.append(w - size)
    if ys[-1] + size < h:
        ys.append(h - size)
    return [(x, y, min(size, w - x), min(size, h - y)) for y in ys for x in xs]


def detect(model, frame, args):
    """-> list of (x1, y1, x2, y2, conf, class_name)."""
    kw = dict(imgsz=args.imgsz, conf=args.conf, iou=0.5, device=args.device, verbose=False)
    if args.classes:
        kw["classes"] = args.class_ids
    if not args.tile:
        crops, offsets = [frame], [(0, 0)]
        full = True
    else:
        # one full-frame pass for big objects plus overlapping crops for small ones
        boxes = tiles(frame.shape, args.tile_size, args.tile_overlap)
        crops = [frame] + [frame[y:y + h, x:x + w] for x, y, w, h in boxes]
        offsets = [(0, 0)] + [(x, y) for x, y, _, _ in boxes]
        full = False
    out = []
    for res, (ox, oy) in zip(model.predict(crops, **kw), offsets):
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append((x1 + ox, y1 + oy, x2 + ox, y2 + oy, float(b.conf[0]), model.names[int(b.cls[0])]))
    if not full and out:  # merge duplicates from overlapping crops (per class)
        keep = []
        for name in {d[5] for d in out}:
            group = [d for d in out if d[5] == name]
            rects = [[d[0], d[1], d[2] - d[0], d[3] - d[1]] for d in group]
            idx = cv2.dnn.NMSBoxes(rects, [d[4] for d in group], args.conf, 0.5)
            keep += [group[i] for i in np.array(idx).flatten()]
        out = keep
    return out


COLORS = {"blue": ((90, 190, 70), (130, 255, 255)), "red": ((0, 150, 70), (8, 255, 255)),
          "green": ((40, 90, 50), (80, 255, 255)), "yellow": ((20, 120, 100), (35, 255, 255)),
          "orange": ((9, 150, 100), (19, 255, 255))}


def color_candidates(frame, name, min_area):
    """Blobs of a saturated colour -> [(x1, y1, x2, y2, score, label)], the biggest compact ones first."""
    lo, hi = COLORS[name]
    mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), np.array(lo), np.array(hi))
    if name == "red":  # red wraps around the hue circle
        mask |= cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), np.array((172, 150, 70)), np.array((180, 255, 255)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    out = []
    for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        solidity = area / max(w * h, 1)           # how much of its box is the colour: rejects shirts, sky, etc.
        out.append((x, y, x + w, y + h, area * solidity, f"{name} object"))
    return sorted(out, key=lambda d: -d[4])


class Lock:
    """Follows one object: keeps the candidate nearest to where it was, smooths the box, survives short dropouts."""

    def __init__(self, max_jump_frac=0.25, lost_after=15):
        self.box, self.label, self.lost, self.max_jump_frac, self.lost_after = None, "", 0, max_jump_frac, lost_after
        self.enabled = True

    def unlock(self):
        self.box, self.lost = None, 0

    def pin(self, box, label):
        self.box, self.label, self.lost, self.enabled = np.array(box, float), label, 0, True

    @property
    def locked(self):
        return self.box is not None

    def update(self, cands, frame_w):
        """cands: [(x1, y1, x2, y2, score, label)]. Returns True while the object is locked on this frame."""
        if not self.enabled:
            return False
        if self.box is None:
            if cands:
                self.pin(cands[0][:4], cands[0][5])
                return True
            return False
        cx, cy = (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2
        best, best_d = None, self.max_jump_frac * frame_w * (1 + 0.15 * self.lost)
        for c in cands:
            d = np.hypot((c[0] + c[2]) / 2 - cx, (c[1] + c[3]) / 2 - cy)
            if d < best_d:
                best, best_d = c, d
        if best is None:
            self.lost += 1
            if self.lost > self.lost_after:
                self.unlock()                      # gone for good: go back to searching
            return False
        self.box = 0.6 * self.box + 0.4 * np.array(best[:4], float)   # smooth the box
        self.lost = 0
        return True


class Range:
    """Distance from apparent height: d = K / height_px, K = focal_px * real_height (or measured by calibration)."""

    def __init__(self, real_height_mm, hfov_deg, focal_px=None):
        self.real_mm, self.hfov, self.focal_px, self.k, self.calibrated, self.smooth = real_height_mm, hfov_deg, focal_px, None, False, None

    def cm(self, height_px, frame_w):
        if height_px < 2:
            return None
        k = self.k
        if k is None:
            f = self.focal_px * frame_w / self.focal_w if self.focal_px else frame_w / (2 * np.tan(np.radians(self.hfov) / 2))
            k = f * self.real_mm / 10.0
        d = k / height_px
        self.smooth = d if self.smooth is None else 0.7 * self.smooth + 0.3 * d
        return self.smooth

    def calibrate(self, height_px, known_cm):
        self.k, self.calibrated, self.smooth = height_px * known_cm, True, known_cm


def draw_lock(frame, lock, live, dist_cm=None, calibrated=False):
    """Crosshair + corner brackets on the locked object."""
    if not lock.locked:
        return
    x1, y1, x2, y2 = (int(v) for v in lock.box)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    col = (0, 0, 255) if live else (0, 140, 255)
    arm = max(12, min(x2 - x1, y2 - y1) // 3)
    for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (px, py), (px + dx * arm, py), col, 4)
        cv2.line(frame, (px, py), (px, py + dy * arm), col, 4)
    r = max(10, arm // 2)
    cv2.circle(frame, (cx, cy), r, col, 3)
    cv2.line(frame, (cx - r - 8, cy), (cx + r + 8, cy), col, 2)
    cv2.line(frame, (cx, cy - r - 8), (cx, cy + r + 8), col, 2)
    dist = "" if dist_cm is None else f"  {dist_cm:.0f} cm" + ("" if calibrated else " (est)")
    text = f"LOCKED {lock.label}{dist}  ({cx},{cy})" if live else f"lock held, object not seen ({lock.lost})"
    pos = (x1, max(y1 - 12, 36))
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)


def draw(frame, dets, info):
    for x1, y1, x2, y2, conf, name in dets:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, (0, 200, 255), 2)
        label = f"{name} {conf:.2f}"
        pos = (p1[0], max(p1[1] - 6, 14))
        cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(frame, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(frame, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--viam-address", help="use this machine's camera")
    src.add_argument("--webcam", action="store_true")
    src.add_argument("--image")
    src.add_argument("--video")
    ap.add_argument("--camera", default="cam", help="Viam camera component name")
    ap.add_argument("--viam-insecure", action="store_true")
    ap.add_argument("--model", default=os.path.join(ROOT, "weights", "yolov8s.pt"))
    ap.add_argument("--prompts", help="comma-separated names to look for (open-vocabulary --model *world*.pt only)")
    ap.add_argument("--classes", help="comma-separated class names to keep, e.g. person,laptop,cup")
    ap.add_argument("--imgsz", type=int, default=960, help="model input size; bigger finds smaller objects")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tile", action="store_true", help="also run on overlapping crops for small objects")
    ap.add_argument("--tile-size", type=int, default=480)
    ap.add_argument("--tile-overlap", type=float, default=0.25)
    ap.add_argument("--device", default=None, help="cpu / mps / cuda (default: best available)")
    ap.add_argument("--lock", metavar="NAME", help="lock onto the detected object whose class name contains NAME")
    ap.add_argument("--lock-color", choices=sorted(COLORS), help="lock onto the biggest blob of this colour")
    ap.add_argument("--lock-min-area", type=float, default=400, help="smallest colour blob (px^2) worth locking")
    ap.add_argument("--real-height-mm", type=float, default=100.0, help="true height of the locked object (for distance)")
    ap.add_argument("--hfov", type=float, default=70.0, help="camera horizontal field of view in degrees (if unknown)")
    ap.add_argument("--calib-cm", type=float, default=30.0, help="press c with the object this far from the camera")
    ap.add_argument("--show-all", action="store_true", help="keep drawing every detection after the lock is on")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--save", help="write the annotated frame / video to this path")
    args = ap.parse_args()
    if not (args.viam_address or args.webcam or args.image or args.video):
        ap.error("pick a source: --viam-address, --webcam, --image or --video")

    from ultralytics import YOLO
    args.device = args.device or pick_device()
    model = YOLO(args.model)
    if args.prompts:
        if not hasattr(model, "set_classes"):
            sys.exit("--prompts needs an open-vocabulary model, e.g. --model weights/yolov8s-worldv2.pt")
        model.set_classes([p.strip() for p in args.prompts.split(",")])
    args.class_ids = None
    if args.classes:
        want = {c.strip() for c in args.classes.split(",")}
        args.class_ids = [i for i, n in model.names.items() if n in want]
        if not args.class_ids:
            sys.exit(f"none of {sorted(want)} are classes of this model")
    print(f"model {os.path.basename(args.model)} on {args.device}, imgsz {args.imgsz}, conf {args.conf}, "
          f"tiling {'on' if args.tile else 'off'}", flush=True)

    robot = cap = still = None
    if args.viam_address:
        from tidy.robot import ViamRobot
        robot = ViamRobot(args.viam_address, "arm", "gripper", insecure=args.viam_insecure)
    elif args.image:
        still = cv2.imread(args.image)
        if still is None:
            sys.exit(f"cannot read {args.image}")
    else:
        cap = cv2.VideoCapture(args.video if args.video else 0)
        if not cap.isOpened():
            sys.exit("cannot open the video / webcam")
    writer, n, last = None, 0, time.monotonic()
    lock = Lock() if (args.lock or args.lock_color) else None
    if lock is None:
        lock = Lock()
        lock.enabled = False
    state = {"dets": [], "clicked": None}
    rng = Range(args.real_height_mm, args.hfov)
    if robot:  # use the camera's own focal length when the machine reports it
        try:
            _, intr = robot.camera_info(args.camera)
            if intr:
                rng.focal_px, rng.focal_w = intr["fx"], intr["w"]
                print(f"using the camera's reported focal length ({intr['fx']:.0f}px at {intr['w']}px wide)")
        except Exception as e:
            print(f"camera intrinsics unavailable ({type(e).__name__}); assuming a {args.hfov:.0f} degree field of view")

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["clicked"] = (mx, my)
    if not args.no_display:
        cv2.namedWindow("local yolo")
        cv2.setMouseCallback("local yolo", on_mouse)
    try:
        while True:
            if robot:
                frame = robot.get_frame(args.camera)
            elif still is not None:
                frame = still.copy()
            else:
                ok, frame = cap.read()
                frame = frame if ok else None
            if frame is None:
                break
            t0 = time.monotonic()
            # once locked on a colour, the model is not needed at all (much faster); a named lock still needs it
            idle = lock.enabled and lock.locked and not args.lock and not args.show_all
            dets = [] if idle else detect(model, frame, args)
            ms = (time.monotonic() - t0) * 1000
            if (n % 10 == 0 or still is not None) and not idle:
                print(f"{len(dets)} objects ({ms:.0f} ms): " +
                      ", ".join(f"{d[5]} {d[4]:.2f}" for d in sorted(dets, key=lambda d: -d[4])), flush=True)
            now = time.monotonic()
            fps = 1 / max(now - last, 1e-6)
            info = f"{len(dets)} objects  conf>={args.conf:.2f}  imgsz {args.imgsz}{'  TILED' if args.tile else ''}  " \
                   f"{fps:.1f} fps"
            last = now
            state["dets"] = dets
            if state["clicked"]:               # click an object to lock it (smallest box under the cursor)
                mx, my = state["clicked"]
                state["clicked"] = None
                hit = [d for d in dets if d[0] <= mx <= d[2] and d[1] <= my <= d[3]]
                if hit:
                    d = min(hit, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
                    lock.pin(d[:4], d[5])
            cands = []
            if args.lock:
                cands += sorted((d for d in dets if args.lock.lower() in d[5].lower()), key=lambda d: -d[4])
            if args.lock_color:
                cands += color_candidates(frame, args.lock_color, args.lock_min_area)
            live = lock.update(cands, frame.shape[1])
            dist_cm = rng.cm(lock.box[3] - lock.box[1], frame.shape[1]) if lock.locked and live else None
            if lock.locked and n % 10 == 0:
                bx = lock.box
                print(f"lock {'ON ' if live else 'held'} {lock.label}: centre ({(bx[0]+bx[2])/2:.0f},{(bx[1]+bx[3])/2:.0f}) "
                      f"size {bx[2]-bx[0]:.0f}x{bx[3]-bx[1]:.0f}"
                      + ("" if dist_cm is None else f"  distance {dist_cm:.0f} cm{'' if rng.calibrated else ' (estimate)'}"),
                      flush=True)
            if lock.enabled and lock.locked and not args.show_all:
                dets = []                      # locked: ignore (and hide) every other object
                info = f"LOCKED on {lock.label}  -  other objects ignored (l = unlock)  {fps:.1f} fps"
            draw(frame, dets, info)
            draw_lock(frame, lock, live, dist_cm, rng.calibrated)
            if args.save:
                if still is not None:
                    cv2.imwrite(args.save, frame)
                else:
                    if writer is None:
                        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 15, frame.shape[1::-1])
                    writer.write(frame)
            n += 1
            if still is not None and (args.no_display or args.save):
                break
            if not args.no_display:
                cv2.imshow("local yolo", frame)
                key = cv2.waitKey(0 if still is not None else 1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("l"):
                    lock.unlock()
                    lock.enabled = False
                elif key == ord("c") and lock.locked:
                    rng.calibrate(lock.box[3] - lock.box[1], args.calib_cm)
                    print(f"calibrated: object is {args.calib_cm:.0f} cm away at {lock.box[3]-lock.box[1]:.0f}px tall", flush=True)
                elif key == ord("r"):
                    lock.unlock()
                    lock.enabled = True
                elif key == ord("t"):
                    args.tile = not args.tile
                elif key in (ord("+"), ord("=")):
                    args.conf = min(args.conf + 0.05, 0.95)
                elif key in (ord("-"), ord("_")):
                    args.conf = max(args.conf - 0.05, 0.05)
                elif key == ord("s"):
                    os.makedirs("data", exist_ok=True)
                    path = f"data/detect_{int(time.time())}.jpg"
                    cv2.imwrite(path, frame)
                    print("saved", path)
    finally:
        if writer:
            writer.release()
        if cap:
            cap.release()
        if robot:
            robot.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
