"""Draw the execution plan on the camera picture.

Tunnel   the object stays centred while the gripper drives straight at it, so the path appears as the object growing:
         one outline per planned waypoint (green = fast, orange = slow), joined at the corners like a perspective tunnel,
         with the key points labelled (gripper opens, slow zone starts, stop).
Maps     two small plan views of the real path in the robot frame - top (x forward, y left) and side (x forward, z up) -
         with the workspace bounds, keep-out boxes, waypoints coloured by speed, the object, and the gripper right now.
"""
import cv2
import numpy as np

GREEN, ORANGE, RED, WHITE, CYAN, YELLOW, BLUE = (0, 220, 0), (0, 165, 255), (60, 60, 255), (255, 255, 255), (255, 255, 0), (0, 230, 255), (255, 120, 40)


def _label(img, text, org, color, scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def _speed_color(speed, slow_speed):
    if speed <= 0:
        return RED
    return GREEN if speed > slow_speed * 1.6 else ORANGE


def draw_tunnel(frame, plan, c0_mm, tip_offset_mm, aim, box_now, slow_cm, open_cm, slow_speed, progress_mm=0.0):
    """Outlines of how big the object will look at each REMAINING waypoint. The aim point stays on the image centre while
    the gripper drives straight at it, so every outline grows about `aim` from the object's box as it is right now
    (c0_mm = camera-to-object distance when the plan was made, progress_mm = how far the arm has travelled since)."""
    ax, ay = aim
    x1, y1, x2, y2 = box_now
    c_now = c0_mm - progress_mm
    rects = []
    for w in plan.waypoints:
        if w.s_mm < progress_mm - 1e-6:
            continue                                              # already passed: not drawn
        c = c0_mm - w.s_mm
        if c < 20.0 or c_now < 20.0:
            continue
        k = min(c_now / c, 8.0)
        r = (int(ax + (x1 - ax) * k), int(ay + (y1 - ay) * k), int(ax + (x2 - ax) * k), int(ay + (y2 - ay) * k))
        rects.append((r, _speed_color(w.speed, slow_speed), w))
    if not rects:
        return
    for (r, col, w) in rects:
        cv2.rectangle(frame, r[:2], r[2:], col, 1)
    first, last = rects[0][0], rects[-1][0]
    for (xa, ya), (xb, yb) in zip([(first[0], first[1]), (first[2], first[1]), (first[0], first[3]), (first[2], first[3])],
                                  [(last[0], last[1]), (last[2], last[1]), (last[0], last[3]), (last[2], last[3])]):
        cv2.line(frame, (xa, ya), (xb, yb), WHITE, 1, cv2.LINE_AA)       # the perspective rays
    marks = []
    for name, tip in (("slow", slow_cm), ("open", open_cm), ("stop", plan.stop_cm)):
        if plan.start_tip_cm - progress_mm / 10.0 >= tip >= plan.stop_cm - 1e-6:
            best = min(rects, key=lambda e: abs(e[2].tip_cm - tip))
            marks.append((f"{name} {tip:g}cm", best))
    outer = max((e[0] for e in rects), key=lambda r: r[2] - r[0])
    for i, (text, (r, col, w)) in enumerate(marks):
        col = YELLOW if text.startswith("open") else col
        cv2.rectangle(frame, r[:2], r[2:], col, 2)
        tx, ty = outer[2] + 24, outer[1] + 18 * (i + 1)
        cv2.line(frame, (r[2], r[1]), (tx - 2, ty - 4), col, 1, cv2.LINE_AA)       # leader from the outline to its label
        _label(frame, text, (tx, ty), col)
    done = 100.0 * progress_mm / max(plan.travel_mm, 1e-6)
    _label(frame, f"{done:.0f}% of the plan  ({max(plan.travel_mm - progress_mm, 0) / 10:.1f} cm to go)",
           (outer[2] + 24, outer[1]), WHITE, 0.55)


def _panel(frame, x0, y0, size, title, scale=0.4):
    cv2.rectangle(frame, (x0, y0), (x0 + size[0], y0 + size[1]), (25, 25, 25), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + size[0], y0 + size[1]), (170, 170, 170), 1)
    _label(frame, title, (x0 + 3, y0 + int(34 * scale)), (200, 200, 200), scale)


def draw_minimap(frame, plan, p_obj, cur, bounds, boxes, slow_speed, open_s_mm, progress_mm=0.0, origin=None, scale=1.0):
    """Top and side views of the planned path in the robot frame."""
    h, w = frame.shape[:2]
    size = (max(int(210 * scale), 40), max(int(150 * scale), 30))
    gap, tiny = max(int(8 * scale), 3), scale < 0.6
    ox, oy = origin or (10, h - size[1] - 10)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    sw, sh = size
    allp = np.array([wp.pos for wp in plan.waypoints] + [np.asarray(p_obj, float), np.asarray(cur, float)])
    lo = np.minimum(np.array([xmin, ymin, zmin]), allp.min(axis=0)) - 40.0       # the view always contains the plan
    hi = np.maximum(np.array([xmax, ymax, zmax]), allp.max(axis=0)) + 40.0

    def top(p):                       # horizontal = y (left of the robot to the left), vertical = x (forward is up)
        return (hi[1] - p[1]) / (hi[1] - lo[1]), 1 - (p[0] - lo[0]) / (hi[0] - lo[0])

    def side(p):                      # horizontal = x, vertical = z (up is up)
        return (p[0] - lo[0]) / (hi[0] - lo[0]), 1 - (p[2] - lo[2]) / (hi[2] - lo[2])

    for title, proj, px0, other in ((("top" if tiny else "top  (x up, y left)"), top, ox, 2),
                                    (("side" if tiny else "side (x right, z up)"), side, ox + sw + gap, 1)):
        _panel(frame, px0, oy, size, title, 0.3 if tiny else 0.4)

        def pt(p, proj=proj, px0=px0):
            u, v = proj(np.asarray(p, float))
            return int(px0 + np.clip(u, 0, 1) * sw), int(oy + np.clip(v, 0, 1) * sh)
        a, c = pt([xmin, ymax, zmax]), pt([xmax, ymin, zmin])                    # the workspace bounds
        cv2.rectangle(frame, (min(a[0], c[0]), min(a[1], c[1])), (max(a[0], c[0]), max(a[1], c[1])), (110, 110, 110), 1)
        for b in boxes:                                                             # keep-out boxes that sit in this view
            rng_lo, rng_hi = (zmin, zmax) if other == 2 else (ymin, ymax)
            if b.hi[other] < rng_lo or b.lo[other] > rng_hi:
                continue                                                            # the floor / far walls: not visible here
            a, c = pt([b.lo[0], b.hi[1], b.hi[2]]), pt([b.hi[0], b.lo[1], b.lo[2]])
            x1, x2, y1, y2 = min(a[0], c[0]), max(a[0], c[0]), min(a[1], c[1]), max(a[1], c[1])
            if x2 > x1 and y2 > y1:
                over = frame.copy()
                cv2.rectangle(over, (x1, y1), (x2, y2), RED, -1)
                cv2.addWeighted(over, 0.35, frame, 0.65, 0, frame)
        pts = [(pt(wp.pos), (120, 120, 120) if wp.s_mm < progress_mm - 1e-6 else _speed_color(wp.speed, slow_speed))
               for wp in plan.waypoints]                                            # grey = already travelled
        line_w = 1 if tiny else 2
        for (a, _), (b, cb) in zip(pts[:-1], pts[1:]):
            cv2.line(frame, a, b, cb, line_w, cv2.LINE_AA)
        if not tiny:
            for i, (q, _) in enumerate(pts):
                if i % 3 == 0 or i == len(pts) - 1:
                    cv2.circle(frame, q, 2, WHITE, -1)
        r1, r2, r3 = (max(int(5 * scale), 2), max(int(6 * scale), 2), max(int(4 * scale), 1))
        wo = min(plan.waypoints, key=lambda wp: abs(wp.s_mm - open_s_mm))
        cv2.circle(frame, pt(wo.pos), r1, BLUE, 1 if tiny else 2)                  # where the gripper opens
        cv2.circle(frame, pt(p_obj), r2, YELLOW, -1)                               # the object
        cv2.circle(frame, pt(cur), r3, CYAN, -1)                                   # the gripper now
    if not tiny:
        _label(frame, "yellow = object   blue ring = gripper opens   cyan = gripper", (ox + 4, oy - 6), (200, 200, 200), 0.4)


def draw_plan(frame, plan, ctx, cur, bounds, boxes, slow_speed, aim, slow_cm, open_cm, tip_offset_mm,
              progress_mm=0.0, box_now=None, map_scale=0.3):
    """Both overlays, live: progress_mm is how far the arm has actually travelled along the plan, box_now the object's
    current box (x1, y1, x2, y2) in pixels, aim the image point the gripper is steering at (the image centre).
    ctx: dict(c0_mm=, box=(x1, y1, x2, y2) when planned, p_obj=robot-frame object position)."""
    if getattr(plan, "kind", "approach") == "lift":                # no perspective tunnel for a straight lift: maps + progress
        done = 100.0 * progress_mm / max(plan.travel_mm, 1e-6)
        _label(frame, f"LIFT  {done:.0f}% ({max(plan.travel_mm - progress_mm, 0) / 10:.1f} cm to go)", (frame.shape[1] // 2 - 110, 40), YELLOW, 0.8)
    else:
        draw_tunnel(frame, plan, ctx["c0_mm"], tip_offset_mm, aim, box_now or ctx["box"], slow_cm, open_cm, slow_speed, progress_mm)
    if map_scale > 0:
        draw_minimap(frame, plan, ctx["p_obj"], cur, bounds, boxes, slow_speed, plan.open_s_mm, progress_mm, scale=map_scale)
