"""Validate a saved checkpoint before appending to its capture directory."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import cv2

from .archive import read_jsonl
from .imageio import gray, read_image
from .models import CaptureConfig, Rect
from .motion import MotionTracker


class ResumeError(ValueError):
    pass


@dataclass(frozen=True)
class ResumeHeader:
    metadata: dict

    def capture_config(self, max_frames=0):
        return replace(CaptureConfig(**self.metadata['capture_config']), max_frames=max_frames)


@dataclass(frozen=True)
class PendingScroll:
    image_path: Path
    overlay_masks: tuple[Rect, ...]
    motion_bounds: dict

    def matches_viewport(self, current, config_masks):
        from .stability import minor_border_noise
        from .text_rules import visual_notice_layers
        diagnostic = read_image(self.image_path)
        if diagnostic.shape != current.shape:
            return False
        _, current_overlays = visual_notice_layers(current)
        difference = cv2.absdiff(gray(diagnostic), gray(current))
        for mask in (*config_masks, *self.overlay_masks, *current_overlays):
            difference[max(0, mask.y):mask.bottom, max(0, mask.x):mask.right] = 0
        return minor_border_noise(difference)


@dataclass(frozen=True)
class ResumePoint(ResumeHeader):
    frame: dict
    image_path: Path
    masks: tuple[Rect, ...]
    offset: float
    counter: int
    scroll_samples: tuple[tuple[float, int, str], ...]
    pending_scroll: PendingScroll | None = None

    def verify(self, current, masks, config, alignment, budget):
        previous = read_image(self.image_path)
        if previous.shape != current.shape or previous.shape[:2] != (self.frame['height'], self.frame['width']):
            raise ResumeError('当前聊天区域尺寸和续采参考图不一致。')
        tracker = MotionTracker(budget, alignment)
        try:
            tracker.accept(previous, self.frame['frame_id'], [], 'both', self.masks)
            match, _, _ = tracker.accept(current, 'resume-check', [], 'both', masks, media_risk=True)
            # The immediately following diagnostic may identify the viewport
            # reached by one recorded scroll. Only reuse its wheel bounds if
            # the current viewport independently matches that saved image.
            if not match.accepted and self.pending_scroll and self.pending_scroll.matches_viewport(current, alignment.masks):
                match, _, _ = tracker.accept(current, 'resume-check', [], config.direction, masks,
                                             motion_bounds=self.pending_scroll.motion_bounds, media_risk=True)
        finally:
            tracker.close()
        if (not match.accepted or match.overlap_ratio < .5
                or (match.delta_px > 0 if config.direction == 'up' else match.delta_px < 0)):
            raise ResumeError('无法确认当前聊天画面与续采点相连。请打开原对话，滚回参考图附近后重试。')
        return match, previous


def _read_checkpoint(directory):
    path = Path(directory).resolve()
    try:
        metadata = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
        checkpoint = json.loads((path/'checkpoint.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ResumeError('所选记录没有可读取的续采点。') from error
    if (metadata.get('format_version') != 2
            or metadata.get('kind') not in {'live_debug', 'live_event_debug'}
            or metadata.get('status') not in {'needs_attention', 'stopped', 'limit_reached'}
            or not (path/'messages.sqlite3').is_file()):
        raise ResumeError('请选择已有确认帧并已停止的调试采集记录。')
    if (not isinstance(checkpoint.get('frame_id'), str)
            or metadata.get('final_scroll_offset_px') != checkpoint.get('scroll_offset_px')):
        raise ResumeError('续采点与帧账本不一致。')
    return metadata, checkpoint


def load_resume_header(directory):
    """The desktop picker only needs calibration and capture settings."""
    metadata, _ = _read_checkpoint(directory)
    return ResumeHeader(metadata)


def load_resume_point(directory):
    path = Path(directory).resolve()
    metadata, checkpoint = _read_checkpoint(path)
    frames = read_jsonl(path/'frames.jsonl')
    alignments = read_jsonl(path/'alignments.jsonl')
    accepted = [row for row in alignments if row['alignment']['accepted']]
    key = checkpoint['frame_id']
    if (not accepted or accepted[-1]['frame_id'] != key
            or accepted[-1]['scroll_offset_px'] != checkpoint['scroll_offset_px']
            or len(accepted) != metadata.get('accepted')
            or [frame['frame_id'] for frame in frames] != [row['frame_id'] for row in alignments]
            or [frame['frame_id'] for frame in frames] != [f'{i:06d}' for i in range(len(frames))]):
        raise ResumeError('续采点与帧账本不一致。')
    frame = frames[int(key)]
    retained = {row['frame_id']: row for row in read_jsonl(path/'retained_frames.jsonl')}
    image_info = frame if frame.get('image') else retained.get(key, {})
    image_path = _verified_image(path, image_info)
    stability = frame['action'].get('stability', {})
    masks = tuple(Rect(**box) for box in stability.get('dynamic_masks', []) + stability.get('overlay_masks', [])
                  + accepted[-1].get('motion_overlay_masks', []))
    config = CaptureConfig(**metadata['capture_config'])
    sign = -1 if config.direction == 'up' else 1
    pending = _pending_scroll(path, key, frames[-1], alignments[-1], retained, config, sign)
    scroll_samples = []
    # Only use direct observations under the record's calibrated wheel command.
    # A constraint-confirmed move or a retry must not reinforce these bounds.
    for previous, decision in zip(reversed(frames), reversed(alignments)):
        action = previous['action']
        method = decision.get('motion_method')
        result = decision['alignment']
        delta = result.get('delta_px')
        events = action.get('events', 0)
        if (action.get('kind') == 'scroll' and action.get('event_units') == sign * config.initial_wheel_units
                and 0 < events <= config.max_scroll_events and method in {'text', 'template', 'static_features', 'edge_layout'}
                and result['accepted'] and delta is not None and 0 < sign * delta < frame['height']):
            scroll_samples.append((delta, events, method))
            if len(scroll_samples) == 8:
                break
    return ResumePoint(metadata, frame, image_path, masks, checkpoint['scroll_offset_px'], len(frames),
                       tuple(reversed(scroll_samples)), pending)


def _verified_image(path, image_info):
    image_path = (path / image_info.get('image', '')).resolve()
    if not image_path.is_relative_to(path) or not image_path.is_file():
        raise ResumeError('缺少最后确认帧的截图，无法校验接续位置。')
    with image_path.open('rb') as handle:
        if hashlib.file_digest(handle, 'sha256').hexdigest() != image_info.get('sha256'):
            raise ResumeError('续采参考截图的校验和不一致。')
    return image_path


def _pending_scroll(path, key, frame, decision, retained, config, sign):
    action = frame['action']
    if (int(frame['frame_id']) != int(key) + 1 or decision.get('reference_id') != key
            or action.get('kind') != 'unstable_diagnostic' or decision['alignment']['reason'] != 'stability_timeout'):
        return None
    attempted = action.get('attempted_action', {})
    bounds = attempted.get('motion_bounds')
    if (attempted.get('kind') != 'scroll' or attempted.get('event_units') != sign * config.initial_wheel_units
            or not bounds or not 0 < attempted.get('events', 0) <= config.max_scroll_events):
        return None
    minimum, maximum = bounds.get('minimum'), bounds.get('maximum')
    if not isinstance(minimum, (int, float)) or not isinstance(maximum, (int, float)):
        return None
    if not (-frame['height'] < minimum < maximum < frame['height'] and
            (maximum == 0 if sign < 0 else minimum == 0)):
        return None
    try:
        image = _verified_image(path, frame if frame.get('image') else retained.get(frame['frame_id'], {}))
    except (OSError, ResumeError):
        return None
    overlays = action.get('stability', {}).get('overlay_masks', [])
    return PendingScroll(image, tuple(Rect(**box) for box in overlays), bounds)


@contextmanager
def resume_lock(directory):
    # Keep the lock outside the record so failed verification leaves its files alone.
    key = hashlib.sha256(os.path.normcase(str(Path(directory).resolve())).encode()).hexdigest()
    path = Path(tempfile.gettempdir()) / f'welog-resume-{key}.lock'
    with path.open('a+b') as handle:
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ResumeError('此记录正在续采。') from error
        yield
