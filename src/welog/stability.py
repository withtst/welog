"""Bounded local animation tolerance, with stationary content outside the changed region."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from .alignment import OverlapAligner
from .imageio import gray
from .models import AlignmentConfig, Rect


@dataclass
class StabilityResult:
    stable: bool
    reason: str
    samples: int
    dynamic_region_ratio: float = 0.0
    static_matches: int = 0
    dynamic_masks: list[Rect] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class LayoutStabilityDetector:
    def __init__(self, alignment: AlignmentConfig, samples: int = 3, max_dynamic_ratio: float = 0.12):
        if samples < 2 or not 0 <= max_dynamic_ratio <= 0.2:
            raise ValueError("Invalid temporal stability window or animation budget")
        self.alignment = alignment
        self.samples = samples
        self.max_dynamic_ratio = max_dynamic_ratio
        self.history: deque[np.ndarray] = deque(maxlen=samples)

    def observe(self, image: np.ndarray) -> StabilityResult:
        current = gray(image).copy()
        if self.history and self.history[-1].shape != current.shape:
            self.history.clear()
        self.history.append(current)
        if len(self.history) < self.samples:
            return StabilityResult(False, "collecting_samples", len(self.history))
        reference = self.history[0]
        valid = np.ones(reference.shape, dtype=bool)
        for box in self.alignment.masks:
            valid[max(0, box.y):box.bottom, max(0, box.x):box.right] = False
        if not valid.any():
            return StabilityResult(False, "no_visible_area", len(self.history))
        # Compare the whole window to its oldest sample to catch gradual scrolling.
        differences = [np.abs(item.astype(np.int16) - reference.astype(np.int16))
                       for item in list(self.history)[1:]]
        if all(np.array_equal(reference[valid], item[valid]) for item in list(self.history)[1:]):
            return StabilityResult(True, "static", len(self.history))
        changed = np.logical_or.reduce([difference > 10 for difference in differences]) & valid
        height, width = reference.shape
        tile = 32
        masks: list[Rect] = []
        area = 0
        # Tile coverage, rather than changed-pixel count, bounds the region being ignored.
        for y in range(0, height, tile):
            run_start = None
            for x in range(0, width + tile, tile):
                active = False
                if x < width:
                    part = changed[y:y + tile, x:x + tile]
                    visible = valid[y:y + tile, x:x + tile]
                    active = int(part.sum()) >= max(4, int(visible.sum() * 0.01))
                if active and run_start is None:
                    run_start = x
                if not active and run_start is not None:
                    box = Rect(run_start, y, min(x, width) - run_start, min(tile, height - y))
                    masks.append(box)
                    area += int(valid[y:box.bottom, run_start:box.right].sum())
                    run_start = None
        ratio = area / int(valid.sum())
        result = StabilityResult(False, "widespread_motion", len(self.history), ratio, dynamic_masks=masks)
        if ratio > self.max_dynamic_ratio:
            return result
        # First check without new masks. A coherent nonzero translation must never
        # be reclassified as a local animation, even when only sparse text moves.
        raw_aligner = OverlapAligner(self.alignment)
        for item in list(self.history)[1:]:
            movement = raw_aligner.estimate(reference, item, direction="both")
            if movement.accepted and movement.delta_px != 0:
                result.reason = "page_still_scrolling"
                return result
        aligner = OverlapAligner(replace(self.alignment, masks=self.alignment.masks + tuple(masks)))
        inliers = []
        for item in list(self.history)[1:]:
            stationary = aligner.estimate(reference, item, direction="both")
            if not stationary.accepted or stationary.delta_px != 0:
                result.reason = "stationary_content_unconfirmed"
                return result
            inliers.append(stationary.inliers)
        result.stable = True
        result.reason = "local_animation" if masks else "minor_pixel_noise"
        result.static_matches = min(inliers)
        return result
