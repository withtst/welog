from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Rect:
    """An exclusive-end rectangle, in native physical pixels."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Rectangle dimensions must be positive")

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class AlignmentConfig:
    min_overlap_ratio: float = 0.25
    patch_width: int = 176
    patch_height: int = 56
    max_patches: int = 24
    min_matches: int = 3
    min_score: float = 0.88
    min_peak_margin: float = 0.035
    consensus_tolerance_px: float = 2.0
    max_horizontal_px: int = 3
    min_texture_std: float = 9.0
    max_validation_error: float = 0.12
    min_validation_coverage: float = 0.75
    masks: tuple[Rect, ...] = ()

    def __post_init__(self) -> None:
        if not 0.1 <= self.min_overlap_ratio <= 0.9:
            raise ValueError("min_overlap_ratio must be between 0.1 and 0.9")
        if self.patch_width < 16 or self.patch_height < 8:
            raise ValueError("Patches are too small")
        if self.min_matches < 2 or self.max_patches < self.min_matches:
            raise ValueError("Need at least two independent matching patches")
        if not 0 < self.min_score <= 1 or not 0 <= self.min_peak_margin <= 1:
            raise ValueError("Invalid matching thresholds")
        if self.consensus_tolerance_px < 0 or self.max_horizontal_px < 0:
            raise ValueError("Invalid geometric tolerance")
        if not 0 < self.max_validation_error < 1:
            raise ValueError("Invalid overlap validation error")
        if not 0 < self.min_validation_coverage <= 1:
            raise ValueError("Invalid overlap validation coverage")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AlignmentConfig:
        data = dict(value)
        data["masks"] = tuple(Rect(**mask) for mask in data.get("masks", []))
        return cls(**data)


@dataclass
class PatchMatch:
    x: int
    y: int
    current_x: int
    current_y: int
    delta_px: int
    score: float
    peak_margin: float


@dataclass
class AlignmentResult:
    accepted: bool
    reason: str
    delta_px: float | None = None
    overlap_ratio: float = 0.0
    inliers: int = 0
    candidates: int = 0
    residual_px: float | None = None
    validation_error: float | None = None
    validation_coverage: float = 0.0
    matches: list[PatchMatch] = field(default_factory=list)
    feature_matches_total: int | None = None
    feature_matches_overlap: int | None = None
    feature_inlier_ratio: float | None = None
    feature_rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CaptureConfig:
    # Zero means continue until a verified stationary boundary, an error, or user stop.
    max_frames: int = 100
    initial_wheel_units: int = 120
    target_step_ratio: float = 0.55
    sample_interval_s: float = 0.10
    stable_samples: int = 3
    stable_timeout_s: float = 4.0
    settle_delay_s: float = 0.35
    bottom_attempts: int = 3
    recovery_attempts: int = 3
    origin_y_px: float = 0.0
    direction: str = "down"
    stability_max_dynamic_ratio: float = 0.12
    max_scroll_events: int = 8
    scroll_event_interval_s: float = 0.035
    image_memory_limit_bytes: int = 64 * 1024 * 1024
    boundary_wait_s: float = 1.0
    debug_save_images: bool = False
    debug_max_bytes: int = 1024 * 1024 * 1024
    capture_backend: str = 'foreground'
    stability_media_max_dynamic_ratio: float = .5

    def __post_init__(self) -> None:
        if self.direction not in {"up", "down"}:
            raise ValueError("Capture direction must be up or down")
        if self.max_frames < 0 or self.initial_wheel_units < 1:
            raise ValueError("max_frames must be nonnegative and initial_wheel_units must be positive")
        if not 0.1 <= self.target_step_ratio <= 0.7:
            raise ValueError("target_step_ratio must be between 0.1 and 0.7")
        if self.stable_samples < 2 or self.sample_interval_s <= 0:
            raise ValueError("Invalid stability sampling configuration")
        if self.stable_timeout_s <= self.sample_interval_s * self.stable_samples:
            raise ValueError("stable_timeout_s is too short")
        if self.bottom_attempts < 2 or self.recovery_attempts < 0:
            raise ValueError("Invalid retry limits")
        if self.settle_delay_s < 0:
            raise ValueError("settle_delay_s cannot be negative")
        if not 0 <= self.stability_max_dynamic_ratio <= 0.2:
            raise ValueError("Dynamic region budget must be between 0 and 0.2")
        if not 1 <= self.max_scroll_events <= 16 or self.scroll_event_interval_s < 0:
            raise ValueError("Invalid scroll event configuration")
        if self.image_memory_limit_bytes <= 0:
            raise ValueError("Image memory limit must be positive")
        if not 0 <= self.boundary_wait_s <= 30:
            raise ValueError("boundary_wait_s must be between 0 and 30 seconds")
        if self.debug_max_bytes<=0:raise ValueError('debug_max_bytes must be positive')
        if self.capture_backend not in {'foreground','background'}:raise ValueError('Invalid capture backend')
        if not self.stability_max_dynamic_ratio<=self.stability_media_max_dynamic_ratio<=.6:raise ValueError('Invalid media animation budget')
