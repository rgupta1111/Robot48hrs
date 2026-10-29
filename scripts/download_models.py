"""Download the MediaPipe models arm_mimic.py needs into models/ (about 15 MB)."""
import os
import urllib.request

BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "pose_landmarker.task": f"{BASE}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "hand_landmarker.task": f"{BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}
dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
os.makedirs(dest, exist_ok=True)
for name, url in MODELS.items():
    path = os.path.join(dest, name)
    if os.path.exists(path):
        print("have", name)
        continue
    print("downloading", name)
    urllib.request.urlretrieve(url, path)
