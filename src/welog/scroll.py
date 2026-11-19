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
        # A media guard can override the planned batch to one event. Growing
        # from an unexecuted plan accumulates a hidden burst on guard exit.
        growth_limit = max(1, math.ceil(sent_events * 1.6))
        self.events = min(self.max_events, growth_limit, desired)

    def recover(self) -> None:
        self.events = 1
        self.recent_rates.clear()


class ScrollCalibration:
    """An empirical upper bound, never a substitute for measuring image motion.

    Only independently registered scrolls train it. Single-event guarded steps
    tolerate partial movement and zero at an endpoint; a predicted exact delta
    would incorrectly exclude both. Evidence is scoped to one live source.
    """
    def __init__(self):
        self.rates = deque(maxlen=8)
        self.batches = {}

    def observe(self, delta_px, events, method):
        if events < 1 or not delta_px or method == 'bounded_layout':
            return
        self.rates.append(abs(delta_px) / events)
        self.batches.setdefault(events,deque(maxlen=8)).append(abs(delta_px))

    def planning_upper(self,events):
        """Forecast only: fewer samples may slow capture, never authorize a match."""
        samples=self.batches.get(events,())
        if events>1 and len(samples)>=3:return math.ceil(max(samples)*1.25+8)
        return math.ceil(max(self.rates)*1.5+8)*events if self.rates else None

    def batch_bounds(self,events,direction,height,min_overlap_ratio=.25):
        if events==1:return self.bounds(direction,height)
        samples=self.batches.get(events,())
        if len(samples)<3:return None
        maximum=math.ceil(max(samples)*1.25+8)
        if maximum>height*(1-min_overlap_ratio):return None
        return {'minimum':-maximum if direction=='up' else 0,
                'maximum':maximum if direction=='down' else 0,
                'samples':len(samples),'events':events,'max_observed_delta':max(samples),
                'basis':'verified_scrolls_same_batch'}

    def bounds(self, direction, height):
        if len(self.rates) < 3 or min(self.rates) <= 0:
            return None
        # Startup and boundary steps can be short. They cannot invalidate an
        # upper bound, whereas a large observation must increase it (or disable
        # it via the viewport cap below). Never trim high values as outliers.
        maximum = math.ceil(max(self.rates) * 1.5 + 8)
        if maximum >= height * .25:
            return None
        return {'minimum': -maximum if direction == 'up' else 0,
                'maximum': maximum if direction == 'down' else 0,
                'samples': len(self.rates), 'max_observed_rate': max(self.rates),
                'basis': 'verified_scrolls_single_event'}


def retained_anchor_count(boxes,height,direction,step,margin=8):
    """Count reference lines that will remain fully inside the next overlap."""
    if direction=='up':return sum(box[3]+margin<=height-step for box in boxes)
    return sum(box[1]-margin>=step for box in boxes)
