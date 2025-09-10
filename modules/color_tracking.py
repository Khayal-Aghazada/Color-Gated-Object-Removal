"""Color-based multi-object tracking via HSV masks and contours.

Merged from:
- L7/coloredObjTracking.py
- L7/multColoredObjTr.py
- L4/L5 HSV and RGB extraction examples
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import cv2
import numpy as np
from .image_ops import ImageOps


class ColorTracker:
    """Track colored objects defined by HSV ranges."""

    def __init__(self, color_ranges: Dict[str, Tuple[Iterable[int], Iterable[int]]]):
        """
        Args:
            color_ranges: name -> (lowHSV, highHSV) where low/high are 3-int iterables
        """
        self.color_ranges = {k: (np.array(v[0], dtype=np.uint8), np.array(v[1], dtype=np.uint8))
                             for k, v in color_ranges.items()}

    def update(self, bgr: np.ndarray):
        """
        Returns:
            dict name -> list of dicts with keys: bbox(x1,y1,x2,y2), center(x,y), contour
        """
        out: Dict[str, List[dict]] = {}
        for name, (low, high) in self.color_ranges.items():
            mask = ImageOps.hsv_mask(bgr, low, high)
            mask = ImageOps.open(mask, 3, 1)
            cnts = ImageOps.find_contours(mask)
            items = []
            for c in cnts:
                if cv2.contourArea(c) < 200:  # ignore tiny
                    continue
                x, y, w, h = cv2.boundingRect(c)
                cx, cy = x + w // 2, y + h // 2
                items.append({
                    "bbox": (x, y, x + w, y + h),
                    "center": (cx, cy),
                    "contour": c
                })
            out[name] = items
        return out
