"""Landmark drawing helpers for faces, hands, and pose.

Merged drawing parts from L9/L10/L11.
"""
from __future__ import annotations
import cv2

try:
    import mediapipe as mp
    _mp_draw = mp.solutions.drawing_utils
    _mp_styles = mp.solutions.drawing_styles
    _mp_hands = mp.solutions.hands
    _mp_pose = mp.solutions.pose
    _mp_face = mp.solutions.face_mesh
except Exception:  # pragma: no cover
    mp = None
    _mp_draw = _mp_styles = _mp_hands = _mp_pose = _mp_face = None


def draw_face_landmarks(bgr, face_landmarks_list) -> None:
    """Draw MediaPipe face mesh landmarks if available. No-ops if empty."""
    if _mp_draw is None or _mp_face is None:
        return
    for lm in face_landmarks_list or []:
        # Recreate the protobuf-like structure only for drawing utils
        # Simple polyline over landmark points
        for (x, y, _) in lm.astype(int):
            cv2.circle(bgr, (int(x), int(y)), 1, (0, 255, 0), 2)


# modules/landmark_draw.py
def draw_hands(bgr, hands_landmarks_list, connect: bool = True) -> None:
    """Draw hand landmarks and optional connections."""
    # fallback edges if MediaPipe is unavailable
    _FALLBACK_EDGES = [
        (0,5),(5,6),(6,7),(7,8),        # index
        (0,9),(9,10),(10,11),(11,12),   # middle
        (0,13),(13,14),(14,15),(15,16), # ring
        (0,17),(17,18),(18,19),(19,20), # pinky
        (0,1),(1,2),(2,3),(3,4)         # thumb
    ]
    for pts in hands_landmarks_list or []:
        pts = pts.astype(int)
        # points
        for (x, y, _) in pts:
            cv2.circle(bgr, (x, y), 2, (0, 0, 255), 2)
        if not connect:
            continue
        # connections
        if _mp_hands is not None:
            for i, j in _mp_hands.HAND_CONNECTIONS:
                x1, y1, _ = pts[int(i)]
                x2, y2, _ = pts[int(j)]
                cv2.line(bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
        else:
            for i, j in _FALLBACK_EDGES:
                x1, y1, _ = pts[i]
                x2, y2, _ = pts[j]
                cv2.line(bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)



def draw_pose(bgr, pose_landmarks, connect: bool = True) -> None:
    """Draw pose landmarks and optional connections."""
    if not hasattr(pose_landmarks, "shape"):
        return
    pts = pose_landmarks.astype(int)

    # points
    for (x, y, _) in pts:
        cv2.circle(bgr, (int(x), int(y)), 3, (0, 0, 255), 2)

    if not connect:
        return

    # use MediaPipe's edges if available, else a minimal fallback skeleton
    _FALLBACK_EDGES = [
        (11,12),(11,13),(13,15),(12,14),(14,16),      # arms + shoulders
        (11,23),(12,24),(23,24),                      # torso
        (23,25),(25,27),(27,29),(29,31),              # left leg/foot
        (24,26),(26,28),(28,30),(30,32)               # right leg/foot
    ]
    if _mp_pose is not None and hasattr(_mp_pose, "POSE_CONNECTIONS"):
        edges = list(_mp_pose.POSE_CONNECTIONS)
    else:
        edges = _FALLBACK_EDGES

    for i, j in edges:
        x1, y1, _ = pts[int(i)]
        x2, y2, _ = pts[int(j)]
        cv2.line(bgr, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)



