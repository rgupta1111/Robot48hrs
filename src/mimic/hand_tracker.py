"""Thumb-index pinch and hand orientation from a webcam frame (MediaPipe hands)."""
import math
from dataclasses import dataclass, field

import numpy as np

from ._mp import landmarker, model_path, to_mp_image

WRIST, THUMB_TIP, INDEX_TIP, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 4, 8, 5, 9, 17


@dataclass
class HandState:
    thumb_px: tuple
    index_px: tuple
    ratio: float               # thumb-index gap: 0 closed .. 1 spread
    heading_deg: float         # hand turned about the vertical axis (knuckle line)
    elevation_deg: float       # hand pitch (wrist -> middle knuckle)
    knuckles_px: tuple = field(default_factory=tuple)  # (index MCP, pinky MCP) pixels


class HandTracker:
    def __init__(self):
        model_path("hand_landmarker.task")  # raises FileNotFoundError early so the caller can run without fingers
        self.model, self._mp = landmarker("HandLandmarker", "hand_landmarker.task", num_hands=2)
        self.last_info = {}
        self._last_ms = -1

    def process(self, frame, t, wrist_px=None, max_dist_frac=None):
        """Pick the hand nearest to wrist_px (if given). With max_dist_frac, reject hands further than that
        fraction of the frame width from it. Returns a HandState or None; last_info explains the choice."""
        ms = max(int(t * 1000), self._last_ms + 1)
        self._last_ms = ms
        res = self.model.detect_for_video(to_mp_image(self._mp, frame), ms)
        h, w = frame.shape[:2]
        self.last_info = {"hands": len(res.hand_landmarks), "used": False}
        if not res.hand_landmarks:
            return None
        best, best_d = 0, None
        if wrist_px is not None:
            ds = [math.hypot(lm[WRIST].x * w - wrist_px[0], lm[WRIST].y * h - wrist_px[1]) for lm in res.hand_landmarks]
            best = int(np.argmin(ds))
            best_d = ds[best]
            self.last_info["dist_px"] = best_d
            if max_dist_frac is not None:
                self.last_info["limit_px"] = max_dist_frac * w
                if best_d > max_dist_frac * w:
                    return None
        self.last_info["used"] = True
        lm, world = res.hand_landmarks[best], res.hand_world_landmarks[best]
        px = lambda i: (lm[i].x * w, lm[i].y * h)  # noqa: E731
        p = lambda i: np.array([world[i].x, world[i].y, world[i].z])  # noqa: E731
        palm = np.linalg.norm(p(MIDDLE_MCP) - p(WRIST)) or 1e-6
        ratio = float(np.clip(np.linalg.norm(p(THUMB_TIP) - p(INDEX_TIP)) / palm, 0.0, 1.0))
        k = p(PINKY_MCP) - p(INDEX_MCP)
        heading = math.degrees(math.atan2(k[2], k[0]))
        up = p(MIDDLE_MCP) - p(WRIST)
        elevation = math.degrees(math.atan2(-up[1], math.hypot(up[0], up[2])))
        return HandState(px(THUMB_TIP), px(INDEX_TIP), ratio, heading, elevation, (px(INDEX_MCP), px(PINKY_MCP)))

    def close(self):
        self.model.close()
