# Re-export primary classes for 2–3 line imports
from .image_ops import ImageOps
from .camera import Camera
from .detectors import YOLODetector, FaceDetector, FaceLandmarks, HandDetector, PoseEstimator
from .landmark_draw import draw_face_landmarks, draw_hands, draw_pose
from .sort_tracker import SortTracker
from .line_count import LineCounter
from .draw import boxes, tracks, lines, texts
from .color_tracking import ColorTracker
from .bg import CleanPlate, ForegroundExtractor, PlateFiller


__all__ = [
    "ImageOps", "Camera",
    "YOLODetector", "FaceDetector", "FaceLandmarks", "HandDetector", "PoseEstimator",
    "draw_face_landmarks", "draw_hands", "draw_pose",
    "SortTracker", "LineCounter", "boxes", "tracks", "lines", "texts",
    "ColorTracker", "CleanPlate", "ForegroundExtractor", "PlateFiller",
]

