import numpy as np

from mimic.overlay import draw_plan
from mimic.plan import build_plan
from mimic.safety import Obstacles


def test_overlay_draws_tunnel_and_maps():
    plan = build_plan([300, 0, 150], [1, 0, 0], tip_cm=20.0, stop_cm=1.0, open_cm=10.0, slow_cm=5.0, fast=150, slow=20)
    obs = Obstacles.lab_world(60)
    frame = np.full((720, 1280, 3), 90, np.uint8)
    before = frame.copy()
    c0 = 20.0 * 10 + 130
    ctx = {"c0_mm": c0, "p_obj": np.array([300 + c0, 0, 150.0]), "box": (600.0, 300.0, 680.0, 420.0)}
    draw_plan(frame, plan, ctx, np.array([300.0, 0, 150]), (250, 600, -300, 300, 0, 450), obs.boxes, 20, (640, 360), 5.0, 10.0, 130.0)
    changed = np.any(frame != before, axis=2)
    assert changed[250:470, 540:740].sum() > 500         # the tunnel around the centre of the picture
    assert changed[720 - 160:720 - 10, 10:430].sum() > 5000   # the two maps, bottom left


def test_passed_waypoints_disappear_from_the_tunnel():
    plan = build_plan([300, 0, 150], [1, 0, 0], tip_cm=20.0, stop_cm=1.0, open_cm=10.0, slow_cm=5.0, fast=150, slow=20)
    c0 = 20.0 * 10 + 130
    ctx = {"c0_mm": c0, "p_obj": np.array([300 + c0, 0, 150.0]), "box": (600.0, 300.0, 680.0, 420.0)}
    obs = Obstacles.lab_world(60)
    start, late = np.full((720, 1280, 3), 90, np.uint8), np.full((720, 1280, 3), 90, np.uint8)
    draw_plan(start, plan, ctx, np.array([300.0, 0, 150]), (250, 600, -300, 300, 0, 450), obs.boxes, 20, (640, 360), 5.0, 10.0, 130.0)
    draw_plan(late, plan, ctx, np.array([420.0, 0, 150]), (250, 600, -300, 300, 0, 450), obs.boxes, 20, (640, 360), 5.0, 10.0, 130.0,
              progress_mm=120.0, box_now=(640 - 40 * 310 / 190, 360 - 60 * 310 / 190, 640 + 40 * 310 / 190, 360 + 60 * 310 / 190))
    assert np.any(start != late)                      # the picture changes as the arm moves
