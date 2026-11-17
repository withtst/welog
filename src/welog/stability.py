"""Bounded local animation tolerance, with stationary content outside the changed region."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import cv2

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
    media_supported_ratio: float = 0.0
    static_method: str = 'templates'

    def to_dict(self) -> dict:
        return asdict(self)


class LayoutStabilityDetector:
    def __init__(self, alignment: AlignmentConfig, samples: int = 3, max_dynamic_ratio: float = 0.12,max_media_dynamic_ratio: float = .5):
        if samples < 2 or not 0 <= max_dynamic_ratio <= 0.2:
            raise ValueError("Invalid temporal stability window or animation budget")
        self.alignment = alignment
        self.samples = samples
        self.max_dynamic_ratio = max_dynamic_ratio
        if not max_dynamic_ratio<=max_media_dynamic_ratio<=.6:raise ValueError('Invalid media animation budget')
        self.max_media_dynamic_ratio=max_media_dynamic_ratio
        self.history: deque[np.ndarray] = deque(maxlen=samples)
        self.media_history=deque(maxlen=samples)

    def observe(self, image: np.ndarray) -> StabilityResult:
        current = gray(image).copy()
        if self.history and self.history[-1].shape != current.shape:
            self.history.clear()
            self.media_history.clear()
        self.history.append(current)
        from .recognition import detect_bubbles
        self.media_history.append([b.box for b in detect_bubbles(image) if b.kind=='image'] if image.ndim==3 else [])
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
        media_aware=ratio>self.max_dynamic_ratio
        media_boxes=[]
        supported=np.zeros(reference.shape,dtype=bool)
        for boxes in self.media_history:
            for x1,y1,x2,y2 in boxes:supported[max(0,y1):y2,max(0,x1):x2]=True
            media_boxes.extend(Rect(x1,y1,x2-x1,y2-y1) for x1,y1,x2,y2 in boxes)
        changed_count=int(changed.sum())
        outside=int((changed&~supported).sum())
        result.media_supported_ratio=1-outside/changed_count if changed_count else 0.
        media_local=bool(changed_count and media_boxes and outside<=64 and result.media_supported_ratio>=.98)
        if media_aware:
            if ratio>self.max_media_dynamic_ratio:return result
            # Larger masks require actual media-shaped regions, not arbitrary moving text.
            # Store only boxes; the rolling grayscale history remains bounded.
            if not media_local:
                result.reason='motion_outside_media';return result
        # First check without new masks. A coherent nonzero translation must never
        # be reclassified as a local animation, even when only sparse text moves.
        # A panning GIF can itself have coherent motion. For the larger, media-
        # supported policy, test page motion outside media instead of treating it as page motion.
        raw_aligner = OverlapAligner(replace(self.alignment,masks=self.alignment.masks+tuple(media_boxes))) if media_local else OverlapAligner(self.alignment)
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
                count=self._static_features(reference,item,aligner) if media_local else 0
                if count:result.static_method='static_features'
                elif media_local:
                    count=self._position_patches(reference,item,raw_aligner)
                    if count:result.static_method='position_patches'
                if not count:
                    result.reason = "stationary_content_unconfirmed";return result
                inliers.append(count)
            else:inliers.append(stationary.inliers)
        result.stable = True
        result.reason = 'multiple_media_animation' if media_aware else "local_animation" if masks else "minor_pixel_noise"
        result.static_matches = min(inliers)
        return result

    @staticmethod
    def _position_patches(previous,current,aligner):
        """Temporal stationarity only. Repeated avatars are usable at fixed positions,
        but this proof is deliberately NOT used for cross-screen translation."""
        valid=aligner._valid_mask(previous.shape)
        error,coverage=aligner._validate(previous,current,0,valid)
        if error>aligner.config.max_validation_error or coverage<aligner.config.min_validation_coverage:return 0
        selected=[]
        for y in range(0,previous.shape[0]-23,16):
            for x in range(0,previous.shape[1]-31,16):
                if not valid[y:y+24,x:x+32].all():continue
                a=previous[y:y+24,x:x+32];b=current[y:y+24,x:x+32]
                if a.std()<aligner.config.min_texture_std:continue
                if (np.abs(cv2.Laplacian(a,cv2.CV_32F))>25).mean()<.025:continue
                diff=np.abs(a.astype(np.int16)-b.astype(np.int16))
                if diff.mean()>1 or (diff>10).mean()>.01:continue
                if any(max(0,min(x+32,px+32)-max(x,px))*max(0,min(y+24,py+24)-max(y,py))>32*24*.1 for px,py in selected):continue
                selected.append((x,y))
                if len(selected)>=3 and max(np.ptp(np.array(selected),axis=0))>=48:return len(selected)
        return 0

    @staticmethod
    def _static_features(previous,current,aligner):
        valid=aligner._valid_mask(previous.shape)
        detector=cv2.SIFT_create(nfeatures=1500)
        a,da=detector.detectAndCompute(previous,valid.astype(np.uint8)*255)
        b,db=detector.detectAndCompute(current,valid.astype(np.uint8)*255)
        if da is None or db is None or len(a)<12 or len(b)<12:return 0
        matches=[pair[0] for pair in cv2.BFMatcher().knnMatch(da,db,k=2) if len(pair)==2 and pair[0].distance<pair[1].distance*.7]
        if len(matches)<12:return 0
        src=np.float32([a[m.queryIdx].pt for m in matches]);dst=np.float32([b[m.trainIdx].pt for m in matches])
        stationary=np.max(np.abs(dst-src),axis=1)<.25
        if stationary.sum()<12 or stationary.mean()<.8:return 0
        if np.max(np.ptp(src[stationary],axis=0))<48:return 0
        error,coverage=aligner._validate(previous,current,0,valid)
        return int(stationary.sum()) if error<=aligner.config.max_validation_error and coverage>=aligner.config.min_validation_coverage else 0
