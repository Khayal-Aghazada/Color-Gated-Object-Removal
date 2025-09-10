"""
bg.py: Clean-plate capture, plate differencing, and hole filling.

Classes
-------
CleanPlate
    Hold or build a clean background image for a fixed camera.
ForegroundExtractor
    Foreground mask via absolute difference vs clean plate, with morphology.
PlateFiller
    Fill masked pixels by copying from plate or via OpenCV inpainting.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import cv2
import numpy as np


@dataclass
class CleanPlate:
    """Hold and build a clean background plate for a fixed camera.

    Attributes
    ----------
    plate : np.ndarray | None
        BGR image of the empty scene. None until set.
    """
    plate: Optional[np.ndarray] = None

    def set(self, img: np.ndarray) -> None:
        """Set the clean plate from a provided BGR image.

        Parameters
        ----------
        img : np.ndarray
            BGR image to store as the clean plate.
        """
        if img is None:
            raise ValueError("img is None")
        self.plate = img.copy()

    def from_median(self, frames: List[np.ndarray]) -> np.ndarray:
        """Compute a clean plate as the per-pixel median of frames.

        Assumes moving objects do not occupy the same pixel >50% of the time.

        Parameters
        ----------
        frames : list of np.ndarray
            Sequence of BGR frames.

        Returns
        -------
        np.ndarray
            Median-composited clean plate.
        """
        if not frames:
            raise ValueError("no frames provided")
        stack = np.stack(frames, axis=0).astype(np.float32)
        med = np.median(stack, axis=0).astype(np.uint8)
        self.plate = med
        return med

    def available(self) -> bool:
        """True if a plate is set."""
        return self.plate is not None


@dataclass
class ForegroundExtractor:
    """Compute a binary foreground mask by differencing frame and clean plate.

    Parameters
    ----------
    thresh : int
        Threshold on absolute grayscale difference [0..255].
    k_open : int
        Kernel size for morphological opening. 0 disables.
    k_close : int
        Kernel size for morphological closing. 0 disables.
    """
    thresh: int = 30
    k_open: int = 3
    k_close: int = 7

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        """Convert BGR to grayscale."""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def mask(self, frame: np.ndarray, plate: np.ndarray) -> np.ndarray:
        """Return uint8 mask (0/255) of foreground pixels.

        Parameters
        ----------
        frame : np.ndarray
            Current BGR frame.
        plate : np.ndarray
            Clean plate BGR image aligned to frame.

        Returns
        -------
        np.ndarray
            Binary mask where 255 indicates foreground.
        """
        if frame.shape != plate.shape:
            raise ValueError(f"shape mismatch: {frame.shape} vs {plate.shape}")
        g_frame = self._to_gray(frame)
        g_plate = self._to_gray(plate)
        diff = cv2.absdiff(g_frame, g_plate)
        _, m = cv2.threshold(diff, int(self.thresh), 255, cv2.THRESH_BINARY)
        if self.k_open > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((self.k_open, self.k_open), np.uint8))
        if self.k_close > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((self.k_close, self.k_close), np.uint8))
        return m


@dataclass
class PlateFiller:
    """Fill masked regions on a frame.

    Parameters
    ----------
    method : str
        'copy' to paste pixels from the clean plate; 'inpaint' to use cv2.inpaint.
    inpaint_radius : int
        Radius for cv2.inpaint when method == 'inpaint'.
    """
    method: str = "copy"
    inpaint_radius: int = 3

    def fill(self, frame: np.ndarray, mask: np.ndarray, plate: np.ndarray) -> np.ndarray:
        """Return a new image with mask==255 replaced accordingly.

        Parameters
        ----------
        frame : np.ndarray
            Current BGR frame.
        mask : np.ndarray
            Binary mask 0/255 of regions to remove.
        plate : np.ndarray
            Clean plate BGR image.

        Returns
        -------
        np.ndarray
            Output BGR image with regions filled.
        """
        if self.method == "copy":
            out = frame.copy()
            out[mask > 0] = plate[mask > 0]
            return out
        if self.method == "inpaint":
            return cv2.inpaint(frame, mask, float(self.inpaint_radius), cv2.INPAINT_TELEA)
        raise ValueError(f"unknown method: {self.method}")
