"""
Detectors: wrappers for YOLOv8, faces, hands, pose, and face landmarks.
"""
from __future__ import annotations
from typing import Iterable, List, Optional, Tuple
import numpy as np
import cv2

# YOLO (Ultralytics)
try:
    from ultralytics import YOLO  # L13 style
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore

# MediaPipe (hands, pose, face mesh)
try:
    import mediapipe as mp
    _mp_draw = mp.solutions.drawing_utils
    _mp_hands = mp.solutions.hands
    _mp_pose = mp.solutions.pose
    _mp_face = mp.solutions.face_mesh
except Exception:  # pragma: no cover
    mp = None  # type: ignore
    _mp_draw = _mp_hands = _mp_pose = _mp_face = None  # type: ignore



# -------------------- YOLOv8 --------------------
# modules/detectors.py  (replace YOLODetector with this version)

class YOLODetector:
    """Ultralytics YOLOv8 wrapper producing xyxy boxes, confidences, and class names."""

    def __init__(self, weights: str, classes: Optional[Iterable[str | int]] = None,
        conf: float = 0.25, iou: float = 0.45):

        if YOLO is None:
            raise ImportError("ultralytics not available")
        self.model = YOLO(weights)
        self.conf = float(conf)
        self.iou = float(iou)
        self._filter_classes = set(classes) if classes is not None else None

    def _name(self, cid: int, result=None) -> str:
        """Map class id to name if available, else string id."""
        try:
            if hasattr(self.model, "names"):
                return str(self.model.names[cid])
        except Exception:
            pass
        try:
            if result is not None and hasattr(result, "names"):
                return str(result.names[cid])
        except Exception:
            pass
        return str(cid)

    def infer(self, bgr: np.ndarray):
        """
        Run inference on a BGR frame.
        Returns:
            boxes:  Nx4 float32 [x1,y1,x2,y2]
            scores: N float32
            labels: list[str] class names
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = self.model.predict(rgb, verbose=False, conf=self.conf, iou=self.iou)

        boxes, scores, labels = [], [], []
        for r in out:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                score = float(b.conf[0])
                cid = int(b.cls[0])
                name = self._name(cid, r)

                if self._filter_classes is not None:
                    if cid not in self._filter_classes and name not in self._filter_classes:
                        continue

                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                labels.append(name)

        if not boxes:
            return (np.empty((0, 4), dtype=float),
                    np.empty((0,), dtype=float),
                    [])
        return (np.array(boxes, dtype=float),
                np.array(scores, dtype=float),
                labels)




# -------------------- Faces (Haar) --------------------
class FaceDetector:
    """OpenCV Haar cascade face detector producing xyxy boxes."""

    def __init__(self, scaleFactor: float = 1.1, minNeighbors: int = 5, minSize: Tuple[int, int] = (30, 30)):
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(xml)
        if self.cascade.empty():
            raise RuntimeError("failed to load Haar cascade")
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.minSize = minSize

    def detect(self, bgr: np.ndarray) -> np.ndarray:
        """Return boxes (N x 4) in xyxy."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rects = self.cascade.detectMultiScale(gray, scaleFactor=self.scaleFactor,
                                              minNeighbors=self.minNeighbors, minSize=self.minSize)
        if len(rects) == 0:
            return np.empty((0, 4), dtype=int)
        xyxy = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects], dtype=int)
        return xyxy




# -------------------- Face Landmarks (MediaPipe Face Mesh) --------------------
class FaceLandmarks:
    """MediaPipe face mesh wrapper. Returns landmarks for one or more faces."""

    def __init__(self, max_num_faces: int = 1, refine_landmarks: bool = True):
        if _mp_face is None:
            raise ImportError("mediapipe not available")
        self.mesh = _mp_face.FaceMesh(max_num_faces=max_num_faces, refine_landmarks=refine_landmarks)

    def detect(self, bgr: np.ndarray):
        """Return list of landmarks arrays (468 x 3) in image coordinates."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        h, w = bgr.shape[:2]
        out = []
        if res.multi_face_landmarks:
            for lm in res.multi_face_landmarks:
                pts = np.array([[p.x * w, p.y * h, p.z] for p in lm.landmark], dtype=float)
                out.append(pts)
        return out



# -------------------- Hands (MediaPipe) --------------------
class HandDetector:
    """MediaPipe Hands wrapper returning per-hand landmarks."""

    def __init__(self, max_num_hands: int = 2):
        if _mp_hands is None:
            raise ImportError("mediapipe not available")
        self.hands = _mp_hands.Hands(max_num_hands=max_num_hands)

    def detect(self, bgr: np.ndarray):
        """Return list of 21x3 landmark arrays per hand."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        h, w = bgr.shape[:2]
        out = []
        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                pts = np.array([[p.x * w, p.y * h, p.z] for p in lm.landmark], dtype=float)
                out.append(pts)
        return out



# -------------------- Pose (MediaPipe) --------------------
class PoseEstimator:
    """MediaPipe Pose wrapper returning 33 landmarks."""

    def __init__(self):
        if _mp_pose is None:
            raise ImportError("mediapipe not available")
        self.pose = _mp_pose.Pose()

    def detect(self, bgr: np.ndarray):
        """Return 33x3 landmark array or empty list."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        h, w = bgr.shape[:2]
        if not res.pose_landmarks:
            return []
        pts = np.array([[p.x * w, p.y * h, p.z] for p in res.pose_landmarks.landmark], dtype=float)
        return pts
