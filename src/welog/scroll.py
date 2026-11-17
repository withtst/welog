"""Learn pixels per individual wheel event, with conservative overlap and bounded growth."""

from __future__ import annotations

import math
from collections import deque


class ScrollPlanner:
    def __init__(self, target_ratio: float = 0.55, max_events: int = 8):
        self.target_ratio, self.max_events = target_ratio, max_events
        self.events = 1
        self.recent_rates: deque[float] = deque(maxlen=4)

    def observe(self, delta_px: float, sent_events: int, viewport_height: int) -> None:
        if delta_px == 0 or sent_events < 1:
            return
        self.recent_rates.append(abs(delta_px) / sent_events)
        # Keep a margin against acceleration and partial movement near boundaries.
        conservative_rate = max(self.recent_rates) * 1.25
        desired = max(1, math.floor(viewport_height * self.target_ratio / conservative_rate))
        growth_limit = max(1, math.ceil(self.events * 1.6))
        self.events = min(self.max_events, growth_limit, desired)

    def recover(self) -> None:
        self.events = 1
        self.recent_rates.clear()
