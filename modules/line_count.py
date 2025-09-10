"""LineCounter: line-crossing counts for tracks.

Merged crossing logic from L14/carCounter.py and L15/peopleCounter.py.
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import numpy as np
import cv2


class LineCounter:
    """Count unique track IDs crossing named line segments with a vertical tolerance."""

    def __init__(self, lines: Dict[str, Tuple[int, int, int, int]], tol: int = 15):
        """
        Args:
            lines: mapping name -> (x1,y1,x2,y2)
            tol: half-thickness band for crossing check
        """
        self.lines = dict(lines)
        self.tol = int(tol)
        self._seen: Dict[str, set] = {k: set() for k in self.lines}

    @staticmethod
    def _center_xyxy(x1, y1, x2, y2):
        cx = int(x1 + (x2 - x1) / 2)
        cy = int(y1 + (y2 - y1) / 2)
        return cx, cy

    def update(self, tracks: Iterable[Iterable[float]]):
        """
        Args:
            tracks: iterable of [x1,y1,x2,y2,id]
        Returns:
            counts: dict name -> total unique IDs
            events: list of (name, id)
        """
        events: List[Tuple[str, int]] = []
        for t in tracks:
            x1, y1, x2, y2, tid = map(int, t)
            cx, cy = self._center_xyxy(x1, y1, x2, y2)
            for name, (lx1, ly1, lx2, ly2) in self.lines.items():
                # band check around the line's y for horizontal lines, or x for vertical lines
                if ly1 == ly2:  # horizontal
                    if lx1 <= cx <= lx2 and (ly1 - self.tol) <= cy <= (ly1 + self.tol):
                        if tid not in self._seen[name]:
                            self._seen[name].add(tid)
                            events.append((name, tid))
                elif lx1 == lx2:  # vertical
                    if ly1 <= cy <= ly2 and (lx1 - self.tol) <= cx <= (lx1 + self.tol):
                        if tid not in self._seen[name]:
                            self._seen[name].add(tid)
                            events.append((name, tid))
                else:
                    # generic distance to segment <= tol
                    if _distance_point_to_segment((cx, cy), (lx1, ly1), (lx2, ly2)) <= self.tol:
                        if tid not in self._seen[name]:
                            self._seen[name].add(tid)
                            events.append((name, tid))
        counts = {k: len(v) for k, v in self._seen.items()}
        return counts, events

    def totals(self) -> Dict[str, int]:
        """Current totals per line."""
        return {k: len(v) for k, v in self._seen.items()}



def _distance_point_to_segment(p, a, b) -> float:
    """Euclidean distance from point p to segment ab."""
    px, py = p
    ax, ay = a
    bx, by = b
    apx, apy = px - ax, py - ay
    abx, aby = bx - ax, by - ay
    ab2 = abx * abx + aby * aby
    if ab2 == 0:
        return float((apx * apx + apy * apy) ** 0.5)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    cx = ax + t * abx
    cy = ay + t * aby
    dx = px - cx
    dy = py - cy
    return float((dx * dx + dy * dy) ** 0.5)
