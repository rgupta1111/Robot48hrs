"""Shared MediaPipe Tasks plumbing (imported lazily so the controller and sim sink work without mediapipe)."""
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")


def model_path(name):
    path = os.path.abspath(os.path.join(MODELS_DIR, name))
    if not os.path.exists(path):
        raise FileNotFoundError(f"models/{name} not found - run: python scripts/download_models.py")
    return path


def landmarker(kind, filename, **options):
    """Build a VIDEO-mode landmarker; kind is 'PoseLandmarker' or 'HandLandmarker'."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    cls = getattr(vision, kind)
    opts = getattr(vision, kind + "Options")(
        base_options=mp_python.BaseOptions(model_asset_path=model_path(filename)),
        running_mode=vision.RunningMode.VIDEO, **options)
    return cls.create_from_options(opts), mp


def to_mp_image(mp, frame_bgr):
    import cv2
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
