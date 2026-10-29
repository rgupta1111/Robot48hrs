"""Shoulder / elbow / wrist of one of the user's arms from a webcam frame (MediaPipe pose)."""
from dataclasses import dataclass

import numpy as np

from ._mp import landmarker, to_mp_image

# MediaPipe pose landmark indices: (shoulder, elbow, wrist)
ARMS = {"right": (12, 14, 16), "left": (11, 13, 15)}
MIN_VISIBILITY = 0.5


@dataclass
class ArmPose:
    pixels: list            # [(x, y)] shoulder, elbow, wrist in image pixels (un-mirrored frame)
    rel: np.ndarray         # wrist minus shoulder in arm lengths; x = image right, y = down, z = away from camera
    elbow_angle_deg: float
    visibility: float       # weakest of the three landmarks


class ArmTracker:
    def __init__(self, side="right"):
        self.ids = ARMS[side]
        self.model, self._mp = landmarker("PoseLandmarker", "pose_landmarker.task", num_poses=1)
        self._last_ms = -1

    def process(self, frame, t):
        """ArmPose for this frame, or None if the arm is not clearly visible. t is seconds."""
        ms = max(int(t * 1000), self._last_ms + 1)  # MediaPipe needs strictly increasing timestamps
        self._last_ms = ms
        res = self.model.detect_for_video(to_mp_image(self._mp, frame), ms)
        if not res.pose_landmarks:
            return None
        lm, world = res.pose_landmarks[0], res.pose_world_landmarks[0]
        vis = min(min(lm[i].visibility, lm[i].presence) for i in self.ids)
        if vis < MIN_VISIBILITY:
            return None
        h, w = frame.shape[:2]
        pixels = [(lm[i].x * w, lm[i].y * h) for i in self.ids]
        s, e, wr = (np.array([world[i].x, world[i].y, world[i].z]) for i in self.ids)
        upper, fore = np.linalg.norm(e - s), np.linalg.norm(wr - e)
        if upper < 1e-6 or fore < 1e-6:
            return None
        cosang = np.dot(s - e, wr - e) / (upper * fore)
        return ArmPose(pixels, (wr - s) / (upper + fore), float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))), vis)

    def close(self):
        self.model.close()
