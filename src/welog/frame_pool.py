"""Explicit ownership of bounded application image buffers (not model/native RSS)."""
from __future__ import annotations

import threading
import numpy as np


class ImageBudgetExceeded(RuntimeError):
    pass


class ImageBudget:
    def __init__(self, limit_bytes: int = 64 * 1024 * 1024):
        if limit_bytes <= 0:
            raise ValueError("Image budget must be positive")
        self.limit_bytes = limit_bytes
        self.used_bytes = self.peak_bytes = 0
        self.lock = threading.Lock()

    def reserve(self, size: int) -> Reservation:
        with self.lock:
            if size < 0 or self.used_bytes + size > self.limit_bytes:
                raise ImageBudgetExceeded("Image memory budget exhausted; capture paused")
            self.used_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.used_bytes)
        return Reservation(self, size)

    def copy(self, array: np.ndarray) -> FrameLease:
        reservation = self.reserve(array.nbytes)
        try:
            return FrameLease(array.copy(order="C"), reservation)
        except BaseException:
            reservation.release()
            raise


class Reservation:
    def __init__(self, budget: ImageBudget, size: int):
        self.budget, self.size = budget, size
        self.active = True

    def release(self):
        with self.budget.lock:
            if self.active:
                self.budget.used_bytes -= self.size
                self.active = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


class FrameLease:
    def __init__(self, image: np.ndarray, reservation: Reservation):
        self._image = image
        self.reservation = reservation

    @property
    def image(self) -> np.ndarray:
        if self._image is None:
            raise RuntimeError("Image lease has been released")
        return self._image

    def release(self):
        self._image = None
        self.reservation.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
