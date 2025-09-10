"""Simple time-based triggers."""
from __future__ import annotations
import time


class HoldTimer:
    """Fires once after a condition stays True for hold_seconds."""
    def __init__(self, hold_seconds: float = 5.0):
        self.hold = float(hold_seconds)
        self._t0: float | None = None
        self._fired = False

    def reset(self):
        self._t0 = None
        self._fired = False

    def tick(self, condition: bool, now: float | None = None) -> bool:
        """Return True exactly once when held long enough."""
        if now is None:
            now = time.monotonic()
        if not condition:
            self._t0 = None
            self._fired = False
            return False
        if self._t0 is None:
            self._t0 = now
            return False
        if not self._fired and (now - self._t0) >= self.hold:
            self._fired = True
            return True
        return False

    def progress(self, now: float | None = None) -> float:
        """0..1 progress while holding; 0 if not holding."""
        if now is None:
            now = time.monotonic()
        if self._t0 is None:
            return 0.0
        return max(0.0, min(1.0, (now - self._t0) / self.hold))
