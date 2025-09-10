"""Draw: simple overlays for boxes, tracks, lines, and text.

Merged visualization from L14 and L15 demos.
"""
from __future__ import annotations
from typing import Iterable, List, Tuple
import cv2
import numpy as np


def boxes(img, boxes_xyxy: Iterable[Iterable[float]], labels: Iterable[str] | None = None, thickness: int = 2):
    """Draw axis-aligned boxes with optional labels."""
    if labels is None:
        labels = []
    for i, b in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = map(int, b[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), thickness)
        try:
            label = list(labels)[i]
        except Exception:
            label = None
        if label:
            cv2.putText(img, str(label), (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)


def tracks(img, tracks_xyxy_id: Iterable[Iterable[float]], thickness: int = 2):
    """Draw tracked boxes with ID and center point."""
    for t in tracks_xyxy_id:
        x1, y1, x2, y2, tid = map(int, t)
        w, h = x2 - x1, y2 - y1
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), thickness)
        cv2.putText(img, f"{tid}", (x1, max(30, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, f"{tid}", (x1, max(30, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 4, (255, 0, 255), -1)


def lines(img, lines_dict: dict, color=(0, 0, 255), thickness: int = 3):
    """Draw named lines from dict {name: (x1,y1,x2,y2)}."""
    for name, (x1, y1, x2, y2) in lines_dict.items():
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        cv2.putText(img, name, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def texts(img, items: List[Tuple[str, Tuple[int, int], float, Tuple[int, int, int], int]]):
    """Draw a list of text items: (text, org(x,y), scale, color(BGR), thickness)."""
    for text, org, scale, color, thick in items:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
