"""
Camera: unified video capture with mirroring and key handling.
"""
from __future__ import annotations
from typing import Iterator, Union
import cv2


class Camera:
    """OpenCV VideoCapture wrapper with optional mirroring and waitKey control."""

    def __init__(self, src: Union[int, str] = 0, mirror: bool = True, wait: int = 1):
        """
        Args:
            src: 0/1 for webcams or file/RTSP path.
            mirror: flip horizontally on read.
            wait: milliseconds for cv2.waitKey.
        """
        self.src = src
        self.mirror = mirror
        self.wait = int(wait)
        self.cap = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"cannot open source: {self.src}")

    def read(self):
        """Read one frame. Returns (ret, frame). Applies mirroring if enabled."""
        if self.cap is None:
            self.open()
        ret, frame = self.cap.read()
        if not ret:
            return ret, None
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return ret, frame

    def iterate(self) -> Iterator:
        """Generator over frames until stream ends or 'q' pressed."""
        if self.cap is None:
            self.open()
        while True:
            ret, frame = self.read()
            if not ret:
                break
            yield frame
            if self.wait_key() == ord('q'):
                break

    def show(self, title: str, frame) -> None:
        """Display a frame in a window."""
        cv2.imshow(title, frame)

    def wait_key(self) -> int:
        """Key code from cv2.waitKey(self.wait)."""
        return cv2.waitKey(self.wait) & 0xFF

    def release(self) -> None:
        """Release capture and destroy windows."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()




