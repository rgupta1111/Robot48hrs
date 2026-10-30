"""A persistent 3D occupancy map (voxel grid) built from depth frames at known camera poses (no SLAM: the arm's own pose gives the camera pose).

Every depth frame does two things, in the world frame:
  * the surface point each ray hits becomes more OCCUPIED (log-odds up), and
  * every voxel along the ray IN FRONT of that surface becomes more FREE (log-odds down).
So when a person moves the cup, rays that now pass through its old place clear it out and its new place fills in: the map follows the
scene. A slow decay forgets readings that are never refreshed. Log-odds are clamped, so a changed voxel flips within a few frames.

The map is dense over a bounded box (default 1.3 x 1.4 x 1.15 m at 25 mm = ~100k voxels) and updated with vectorised numpy.
"""
import numpy as np


def depth_to_points(depth_mm, fx, fy, cx, cy, stride=8, min_mm=300.0, max_mm=3000.0):
    """Depth image (H x W, mm) -> (N, 3) points in the CAMERA frame (mm, z straight ahead), subsampled by `stride`.
    Points closer than min_mm (the gripper's own claws, the sensor's blind zone) or farther than max_mm, or without a reading, are dropped."""
    h, w = depth_mm.shape[:2]
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_mm[0:h:stride, 0:w:stride].astype(float)
    ok = np.isfinite(z) & (z >= min_mm) & (z <= max_mm)
    z, u, v = z[ok], u[ok].astype(float), v[ok].astype(float)
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)


class VoxelMap:
    def __init__(self, res_mm=25.0, lo=(0.0, -700.0, -250.0), hi=(1300.0, 700.0, 900.0),
                 l_occ=0.85, l_free=-0.4, l_min=-3.0, l_max=4.0, decay=0.998):
        self.res, self.lo = float(res_mm), np.array(lo, float)
        self.shape = tuple(int(np.ceil((h - l) / self.res)) for l, h in zip(lo, hi))
        self.L = np.zeros(self.shape, np.float32)
        self.l_occ, self.l_free, self.l_min, self.l_max, self.decay = l_occ, l_free, l_min, l_max, decay

    def _index(self, pts):
        i = np.floor((np.asarray(pts, float) - self.lo) / self.res).astype(int)
        ok = np.all((i >= 0) & (i < np.array(self.shape)), axis=1)
        return i[ok]

    def integrate(self, hits_world, origin_world, max_range_mm=3000.0):
        """Fold in one frame: hits_world (N, 3) surface points and the camera origin (3,), both in the world frame, mm."""
        hits = np.asarray(hits_world, float)
        origin = np.asarray(origin_world, float)
        if hits.size == 0:
            return
        d = hits - origin
        dist = np.linalg.norm(d, axis=1)
        keep = (dist > 100.0) & (dist < max_range_mm)
        hits, d, dist = hits[keep], d[keep], dist[keep]
        if len(hits) == 0:
            return
        step = self.res * 0.7
        k = int(np.ceil(dist.max() / step))
        t = (np.arange(k)[None, :] * step) / dist[:, None]                 # fraction along each ray
        in_front = t < (1.0 - self.res * 1.0 / dist[:, None])               # stop one voxel short of the surface it hit
        free = (origin + d[:, None, :] * t[:, :, None])[in_front]
        np.add.at(self.L, tuple(self._index(free).T), self.l_free)          # empty space in front of every surface
        np.add.at(self.L, tuple(self._index(hits).T), self.l_occ)           # the surfaces themselves
        self.L *= self.decay
        np.clip(self.L, self.l_min, self.l_max, out=self.L)

    def is_occupied(self, pt, thresh=1.0):
        i = self._index(np.array([pt], float))
        return bool(len(i)) and bool(self.L[tuple(i[0])] > thresh)

    def occupied_centres(self, thresh=1.0):
        idx = np.argwhere(self.L > thresh)
        return self.lo + (idx + 0.5) * self.res

    def object_box(self, near, radius_mm=250.0, min_z_mm=-100.0, thresh=1.0, min_voxels=3, margin_mm=15.0):
        """Bounding box (centre, dims) of the occupied voxels within radius_mm of `near` and above min_z_mm (above the table), or None.
        The planner's obstacle for the thing we are about to reach: its real extent, not a guessed box."""
        c = self.occupied_centres(thresh)
        if len(c) == 0:
            return None
        c = c[(np.linalg.norm(c - np.asarray(near, float), axis=1) <= radius_mm) & (c[:, 2] >= min_z_mm)]
        if len(c) < min_voxels:
            return None
        lo, hi = c.min(axis=0) - self.res / 2 - margin_mm, c.max(axis=0) + self.res / 2 + margin_mm
        return (lo + hi) / 2, hi - lo
