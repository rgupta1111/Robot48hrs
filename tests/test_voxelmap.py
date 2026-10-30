import numpy as np

from mimic.voxelmap import VoxelMap, depth_to_points


def scene_points(cup_xy, n=2500, seed=0):
    """What the camera at ORIGIN actually SEES: the table (z=0), and the cup's camera-facing side and top (not the far side)."""
    rng = np.random.default_rng(seed)
    table = np.column_stack([rng.uniform(300, 800, n), rng.uniform(-300, 300, n), np.zeros(n)])
    ang = rng.uniform(0, 2 * np.pi, n)
    side = np.column_stack([cup_xy[0] + 40 * np.cos(ang), cup_xy[1] + 40 * np.sin(ang), rng.uniform(0, 120, n)])
    normal = np.column_stack([np.cos(ang), np.sin(ang), np.zeros(n)])
    side = side[np.einsum("ij,ij->i", normal, ORIGIN - side) > 0][: n // 4]          # faces the camera
    top = np.column_stack([cup_xy[0] + rng.uniform(-40, 40, n // 8), cup_xy[1] + rng.uniform(-40, 40, n // 8), np.full(n // 8, 120.0)])
    pts = np.vstack([table, side, top])
    inside = (np.hypot(pts[:, 0] - cup_xy[0], pts[:, 1] - cup_xy[1]) < 40) & (pts[:, 2] < 1)      # the cup stands on the table
    return pts[~inside]


ORIGIN = np.array([100.0, 0.0, 500.0])                      # a camera above and behind the scene


def test_cup_appears_in_the_map_with_its_real_extent():
    vm = VoxelMap()
    for k in range(6):
        vm.integrate(scene_points((550, 0), seed=k), ORIGIN)
    assert vm.is_occupied((515, 0, 60)) or vm.is_occupied((520, 0, 60))     # the camera-facing wall
    c, dims = vm.object_box((550, 0, 60), radius_mm=200, min_z_mm=40)
    assert np.allclose(c[:2], (550, 0), atol=60) and 20 < dims[0] < 140 and 60 < dims[1] < 140 and dims[2] > 90      # its real extent, not a guess


def test_moving_the_cup_clears_the_old_place_and_fills_the_new_one():
    vm = VoxelMap()
    for k in range(6):
        vm.integrate(scene_points((550, 0), seed=k), ORIGIN)
    assert vm.object_box((550, 0, 60), radius_mm=150, min_z_mm=40) is not None and vm.object_box((650, 100, 60), radius_mm=60, min_z_mm=40) is None
    for k in range(10):                                      # a person moves the cup 15 cm: rays now pass through the old place
        vm.integrate(scene_points((650, 100), seed=10 + k), ORIGIN)
    assert vm.object_box((550, 0, 60), radius_mm=60, min_z_mm=40) is None             # old place cleared (free-space carving)
    c, _ = vm.object_box((650, 100, 60), radius_mm=150, min_z_mm=40)                  # new place filled
    assert np.allclose(c[:2], (650, 100), atol=60)


def test_unrefreshed_readings_fade():
    vm = VoxelMap(decay=0.9)
    vm.integrate(scene_points((550, 0)), ORIGIN)
    assert vm.object_box((550, 0, 60), radius_mm=150, min_z_mm=40) is not None
    for _ in range(60):
        vm.L *= vm.decay                                       # nothing new seen: only the slow decay
    assert vm.object_box((550, 0, 60), radius_mm=150, min_z_mm=40) is None


def test_object_box_ignores_the_table_and_far_things():
    vm = VoxelMap()
    for k in range(6):
        vm.integrate(scene_points((550, 0), seed=k), ORIGIN)
    assert vm.object_box((550, 0, 60), radius_mm=200, min_z_mm=40, min_voxels=3) is not None
    assert vm.object_box((300, -250, 0), radius_mm=60, min_z_mm=40) is None        # nothing but table out there


def test_depth_to_points_backprojects_a_flat_wall():
    depth = np.full((72, 128), 1000.0)
    pts = depth_to_points(depth, fx=100.0, fy=100.0, cx=64.0, cy=36.0, stride=8)
    assert np.allclose(pts[:, 2], 1000.0) and abs(pts[:, 0].mean()) < 60 and len(pts) == 9 * 16
    far = depth_to_points(np.full((72, 128), 50.0), 100.0, 100.0, 64.0, 36.0)                # inside the blind zone: dropped
    assert len(far) == 0
