"""Window hover locking and screenshot selection geometry; no desktop side effects."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import Rect


@dataclass(frozen=True)
class WindowTarget:
    hwnd: int
    pid: int
    title: str
    client: Rect
    executable: str = ""


@dataclass(frozen=True)
class RegionCalibration:
    executable: str
    client_width: int
    client_height: int
    roi: Rect
    guard: Rect | None = None

    def matches(self, target: WindowTarget) -> bool:
        if (self.executable != target.executable
                or (self.client_width, self.client_height) != (target.client.width, target.client.height)):
            return False
        for box in (self.roi, self.guard):
            if box is not None and (box.x < 0 or box.y < 0 or box.right > self.client_width or box.bottom > self.client_height):
                return False
        if self.roi.width < 160 or self.roi.height < 120:
            return False
        guard = self.guard
        return guard is None or not (guard.x < self.roi.right and guard.right > self.roi.x
                                     and guard.y < self.roi.bottom and guard.bottom > self.roi.y)

    @classmethod
    def from_dict(cls, value: dict) -> RegionCalibration:
        return cls(value["executable"], value["client_width"], value["client_height"],
                   Rect(**value["roi"]), Rect(**value["guard"]) if value.get("guard") else None)


@dataclass(frozen=True)
class HoverState:
    target: WindowTarget | None
    progress: float
    locked: bool


class HoverLock:
    """Only a continuously unchanged target can lock. Locking never follows the pointer."""

    def __init__(self, dwell_seconds: float = 1.5):
        if not math.isfinite(dwell_seconds) or dwell_seconds < 0.3:
            raise ValueError("Hover dwell must be at least 0.3 seconds")
        self.dwell_seconds = dwell_seconds
        self.reset()

    def reset(self) -> None:
        self.target: WindowTarget | None = None
        self.started_at: float | None = None
        self.locked = False

    def observe(self, target: WindowTarget | None, now: float) -> HoverState:
        if self.locked:
            return HoverState(self.target, 1.0, True)
        if target is None:
            self.reset()
            return HoverState(None, 0, False)
        if target != self.target or self.started_at is None or now < self.started_at:
            self.target, self.started_at = target, now
        progress = min(1.0, max(0.0, (now - self.started_at) / self.dwell_seconds))
        self.locked = progress >= 1.0
        return HoverState(target, progress, self.locked)


def selection_to_pixels(
    start: tuple[float, float], end: tuple[float, float],
    display_size: tuple[int, int], image_size: tuple[int, int],
) -> Rect:
    """Map a selection in a scaled screenshot back to native pixels, in either drag direction."""
    dw, dh = display_size
    iw, ih = image_size
    if min(dw, dh, iw, ih) <= 0:
        raise ValueError("Image and display sizes must be positive")
    x1, x2 = sorted((max(0.0, min(dw, start[0])), max(0.0, min(dw, end[0]))))
    y1, y2 = sorted((max(0.0, min(dh, start[1])), max(0.0, min(dh, end[1]))))
    left, top = math.floor(x1 * iw / dw), math.floor(y1 * ih / dh)
    right, bottom = min(iw, math.ceil(x2 * iw / dw)), min(ih, math.ceil(y2 * ih / dh))
    return Rect(left, top, right - left, bottom - top)
