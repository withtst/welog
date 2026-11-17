"""Translation-only alignment. Ambiguity is a failure, never a guessed offset."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .imageio import gray
from .models import AlignmentConfig, AlignmentResult, PatchMatch


@dataclass(frozen=True)
class _Patch:
    x: int
    y: int
    width: int
    height: int
    texture: float


class OverlapAligner:
    def __init__(self, config: AlignmentConfig | None = None):
        self.config = config or AlignmentConfig()

    def _valid_mask(self, shape: tuple[int, int]) -> np.ndarray:
        valid = np.ones(shape, dtype=bool)
        for box in self.config.masks:
            valid[max(0, box.y):box.bottom, max(0, box.x):box.right] = False
        return valid

    def _patches(self, image: np.ndarray, valid: np.ndarray) -> list[_Patch]:
        height, width = image.shape
        pw = min(self.config.patch_width, width - 8)
        ph = min(self.config.patch_height, max(12, height // 6))
        if pw < 16 or ph >= height:
            return []
        xs = sorted(set(int(x) for x in np.linspace(4, width - pw - 4, 5)))
        ys = range(4, height - ph - 3, max(8, ph // 2))
        bands: dict[int, list[_Patch]] = {}
        for y in ys:
            for x in xs:
                if not valid[y:y + ph, x:x + pw].all():
                    continue
                texture = float(image[y:y + ph, x:x + pw].std())
                if texture < self.config.min_texture_std:
                    continue
                detail = cv2.Laplacian(image[y:y + ph, x:x + pw], cv2.CV_32F)
                edge_fraction = float((np.abs(detail) > 25).mean())
                # Flat avatar/color boundaries can match very strongly but carry little identity.
                if edge_fraction < 0.025:
                    continue
                patch = _Patch(x, y, pw, ph, texture * edge_fraction ** 0.5)
                band = min(5, int(y * 6 / height))
                bands.setdefault(band, []).append(patch)
        for patches in bands.values():
            patches.sort(key=lambda patch: patch.texture, reverse=True)
        chosen: list[_Patch] = []
        # Round-robin across vertical bands, so the overlapping end is represented.
        while len(chosen) < self.config.max_patches:
            added = False
            for band in sorted(bands):
                while bands[band]:
                    patch = bands[band].pop(0)
                    if any(self._intersection_ratio(patch, other) > 0.45 for other in chosen):
                        continue
                    chosen.append(patch)
                    added = True
                    break
                if len(chosen) >= self.config.max_patches:
                    break
            if not added:
                break
        return chosen

    @staticmethod
    def _intersection_ratio(a: _Patch, b: _Patch) -> float:
        width = max(0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
        height = max(0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
        return width * height / (a.width * a.height)

    def _match(
        self, previous: np.ndarray, current: np.ndarray, patch: _Patch,
        low_delta: int, high_delta: int, valid: np.ndarray,
    ) -> PatchMatch | None:
        cfg = self.config
        height, width = current.shape
        y_min = max(0, patch.y - high_delta)
        y_max = min(height - patch.height, patch.y - low_delta)
        x_min = max(0, patch.x - cfg.max_horizontal_px)
        x_max = min(width - patch.width, patch.x + cfg.max_horizontal_px)
        if y_max < y_min:
            return None
        template = previous[patch.y:patch.y + patch.height, patch.x:patch.x + patch.width]
        search = current[y_min:y_max + patch.height, x_min:x_max + patch.width]
        scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        # A mask is viewport-relative: exclude both its old and new locations.
        invalid = (~valid[y_min:y_max + patch.height, x_min:x_max + patch.width]).astype(np.uint8)
        integral = cv2.integral(invalid)
        blocked = (integral[patch.height:, patch.width:] - integral[:-patch.height, patch.width:]
                   - integral[patch.height:, :-patch.width] + integral[:-patch.height, :-patch.width]) > 0
        scores[blocked] = -1
        _, best, _, location = cv2.minMaxLoc(scores)
        if best < cfg.min_score:
            return None
        bx, by = location
        rivals = scores.copy()
        radius = max(3, int(cfg.consensus_tolerance_px * 2))
        rivals[max(0, by - radius):by + radius + 1, :] = -1
        margin = float(best - rivals.max())
        if margin < cfg.min_peak_margin:
            return None
        return PatchMatch(
            patch.x, patch.y, x_min + bx, y_min + by,
            patch.y - (y_min + by), float(best), margin,
        )

    def _validate(
        self, previous: np.ndarray, current: np.ndarray, delta: int, valid: np.ndarray,
    ) -> tuple[float, float]:
        height = previous.shape[0]
        p_start, c_start = max(0, delta), max(0, -delta)
        length = height - abs(delta)
        a = previous[p_start:p_start + length]
        b = current[c_start:c_start + length]
        mask = valid[p_start:p_start + length] & valid[c_start:c_start + length]
        errors: list[float] = []
        for y in range(0, length, 48):
            for x in range(0, previous.shape[1], 64):
                aa = a[y:y + 48, x:x + 64].astype(np.float32)
                bb = b[y:y + 48, x:x + 64].astype(np.float32)
                mm = mask[y:y + 48, x:x + 64]
                if mm.sum() < 64 or aa[mm].std() < self.config.min_texture_std:
                    continue
                # Measure informative pixels, rather than letting the blank background dominate.
                ink = ((np.abs(aa - np.median(aa[mm])) > 20)
                       | (np.abs(bb - np.median(bb[mm])) > 20)) & mm
                if ink.sum() < 12:
                    continue
                errors.append(float(np.mean(np.abs(aa[ink] - bb[ink])) / 255.0))
        if not errors:
            return 1.0, 0.0
        return (
            float(np.median(errors)),
            float(np.mean(np.asarray(errors) <= self.config.max_validation_error)),
        )

    def estimate(
        self, previous: np.ndarray, current: np.ndarray, *, direction: str = "down",
    ) -> AlignmentResult:
        if previous.shape != current.shape:
            return AlignmentResult(False, "layout_size_changed")
        a, b = gray(previous), gray(current)
        if a.dtype != np.uint8 or b.dtype != np.uint8:
            raise ValueError("Alignment requires native uint8 images")
        height, width = a.shape
        if height < 64 or width < 48:
            return AlignmentResult(False, "viewport_too_small")
        maximum = int(height * (1 - self.config.min_overlap_ratio))
        bounds = {"down": (0, maximum), "up": (-maximum, 0), "both": (-maximum, maximum)}
        if direction not in bounds:
            raise ValueError("direction must be down, up, or both")
        valid = self._valid_mask(a.shape)
        patches = self._patches(a, valid)
        matches = [match for patch in patches
                   if (match := self._match(a, b, patch, *bounds[direction], valid)) is not None]
        result = AlignmentResult(False, "insufficient_unique_matches", candidates=len(patches), matches=matches)
        if len(matches) < self.config.min_matches:
            return result
        groups = [
            [other for other in matches
             if abs(other.delta_px - match.delta_px) <= self.config.consensus_tolerance_px
             and abs((other.current_x - other.x) - (match.current_x - match.x)) <= 1]
            for match in matches
        ]
        group = max(groups, key=lambda items: (len(items), sum(item.score for item in items)))
        result.inliers = len(group)
        if len(group) < self.config.min_matches or len(group) / len(matches) < 0.6:
            result.reason = "inconsistent_translation"
            return result
        delta = int(round(float(np.median([match.delta_px for match in group]))))
        horizontal = float(np.median([match.current_x - match.x for match in group]))
        if abs(horizontal) > 1:
            result.reason = "horizontal_layout_changed"
            return result
        residual = max(abs(match.delta_px - delta) for match in group)
        error, coverage = self._validate(a, b, delta, valid)
        result.residual_px = float(residual)
        result.validation_error = error
        result.validation_coverage = coverage
        result.overlap_ratio = (height - abs(delta)) / height
        if coverage < self.config.min_validation_coverage or error > self.config.max_validation_error:
            result.reason = "overlap_content_changed"
            return result
        result.accepted = True
        result.reason = "stationary" if delta == 0 else "aligned"
        result.delta_px = float(delta)
        return result
