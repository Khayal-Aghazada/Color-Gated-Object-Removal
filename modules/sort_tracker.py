from __future__ import annotations
import numpy as np

try:
    from ._sort_impl import Sort
except Exception as e:  # pragma: no cover
    Sort = None  # type: ignore
    _IMPORT_ERR = e


class SortTracker:
    """Thin wrapper around your SORT implementation."""

    def __init__(self, max_age: int = 20, min_hits: int = 3, iou_threshold: float = 0.3):
        if Sort is None:
            raise ImportError(
                "SORT implementation missing. Ensure modules/_sort_impl.py exists and defines class `Sort`. "
                f"Import error: {_IMPORT_ERR}"
            )
        self._tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

    def update(self, detections: np.ndarray) -> np.ndarray:
        if detections is None or len(detections) == 0:
            dets = np.empty((0, 5), dtype=float)
        else:
            dets = detections.astype(float, copy=False)
        return self._tracker.update(dets)
