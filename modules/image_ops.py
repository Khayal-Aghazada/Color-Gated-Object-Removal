"""
ImageOps: basic image I/O, color ops, channel ops, masking, and morphology.
"""
from __future__ import annotations
from typing import Iterable, Tuple
import cv2
import numpy as np


class ImageOps:
    """Stateless helpers for common image operations."""

    # -------- I/O --------
    @staticmethod
    def load(path: str, color: bool = True) -> np.ndarray:
        """Load image from disk. BGR if color=True else single-channel."""
        flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
        img = cv2.imread(path, flag)
        if img is None:
            raise FileNotFoundError(f"cannot read image: {path}")
        return img

    @staticmethod
    def save(path: str, img: np.ndarray) -> None:
        """Save image to disk."""
        ok = cv2.imwrite(path, img)
        if not ok:
            raise IOError(f"cannot write image: {path}")
        


    # -------- Mask --------

    @staticmethod
    def load_roi_mask(path: str, shape: tuple[int,int,int] | None = None, binary: bool = True) -> np.ndarray:
        """Load grayscale ROI mask. Resize to frame shape if provided. Returns uint8 [0,255]."""
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"cannot read mask: {path}")
        if shape is not None and m.shape[:2] != shape[:2]:
            h, w = shape[:2]
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        if binary:
            _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
        return m

    @staticmethod
    def apply_roi(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply ROI mask. Works with 1-channel (as 'mask=') or 3-channel masks."""
        if mask.ndim == 2:
            return cv2.bitwise_and(bgr, bgr, mask=mask)
        return cv2.bitwise_and(bgr, mask)



    # -------- Color and channels --------
    @staticmethod
    def to_gray(bgr: np.ndarray) -> np.ndarray:
        """BGR → Gray."""
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def to_rgb(bgr: np.ndarray) -> np.ndarray:
        """BGR → RGB for display."""
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def to_hsv(bgr: np.ndarray) -> np.ndarray:
        """BGR → HSV."""
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    @staticmethod
    def to_ycrcb(bgr: np.ndarray) -> np.ndarray:
        """BGR → YCrCb."""
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

    @staticmethod
    def split_channels(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split B, G, R channels."""
        return cv2.split(bgr)



    # -------- Resize --------
    @staticmethod
    def resize(img: np.ndarray, size: Tuple[int, int] | None = None, max_side: int | None = None,
               interp: int = cv2.INTER_AREA) -> np.ndarray:
        """Resize by fixed size (w,h) or by keeping aspect with max_side."""
        if size is not None:
            w, h = size
            return cv2.resize(img, (w, h), interpolation=interp)
        if max_side is None:
            return img
        h, w = img.shape[:2]
        scale = max_side / float(max(h, w))
        if scale >= 1.0:
            return img
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(img, new_size, interpolation=interp)



    # -------- Binary masks (HSV and “RGB”) --------
    @staticmethod
    def hsv_mask(bgr: np.ndarray, low: Iterable[int], high: Iterable[int]) -> np.ndarray:
        """Threshold by HSV range. low/high are 3-int iterables in HSV space."""
        hsv = ImageOps.to_hsv(bgr)
        low = np.array(list(low), dtype=np.uint8)
        high = np.array(list(high), dtype=np.uint8)
        return cv2.inRange(hsv, low, high)

    @staticmethod
    def rgb_mask(bgr: np.ndarray, low: Iterable[int], high: Iterable[int]) -> np.ndarray:
        """Threshold by RGB range. Converts BGR → RGB then inRange."""
        rgb = ImageOps.to_rgb(bgr)
        low = np.array(list(low), dtype=np.uint8)
        high = np.array(list(high), dtype=np.uint8)
        return cv2.inRange(rgb, low, high)

    @staticmethod
    def find_contours(mask: np.ndarray):
        """Find external contours on a binary mask."""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cnts



    # -------- Morphology --------
    @staticmethod
    def kernel(ksize: int | Tuple[int, int]) -> np.ndarray:
        """Create rectangular kernel."""
        if isinstance(ksize, int):
            ksize = (ksize, ksize)
        return cv2.getStructuringElement(cv2.MORPH_RECT, ksize)

    @staticmethod
    def erode(img: np.ndarray, ksize: int | Tuple[int, int] = 3, iterations: int = 1) -> np.ndarray:
        """Erode binary or grayscale image."""
        return cv2.erode(img, ImageOps.kernel(ksize), iterations=iterations)

    @staticmethod
    def dilate(img: np.ndarray, ksize: int | Tuple[int, int] = 3, iterations: int = 1) -> np.ndarray:
        """Dilate binary or grayscale image."""
        return cv2.dilate(img, ImageOps.kernel(ksize), iterations=iterations)

    @staticmethod
    def open(img: np.ndarray, ksize: int | Tuple[int, int] = 3, iterations: int = 1) -> np.ndarray:
        """Morphological opening = erode then dilate."""
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, ImageOps.kernel(ksize), iterations=iterations)

    @staticmethod
    def close(img: np.ndarray, ksize: int | Tuple[int, int] = 3, iterations: int = 1) -> np.ndarray:
        """Morphological closing = dilate then erode."""
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, ImageOps.kernel(ksize), iterations=iterations)
