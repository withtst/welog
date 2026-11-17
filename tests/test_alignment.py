import cv2
import numpy as np
import pytest

from welog.alignment import OverlapAligner
from welog.demo import make_scene
from welog.models import AlignmentConfig, Rect
from welog.replay import CoordinateTracker


@pytest.fixture(scope="module")
def scene():
    return make_scene(5000, 640, seed=19)


@pytest.mark.parametrize("delta", [0, 1, 63, 281, 480, -183])
def test_known_native_pixel_displacement(scene, delta):
    previous = scene[900:1620]
    current = scene[900 + delta:1620 + delta]
    result = OverlapAligner().estimate(previous, current, direction="both")
    assert result.accepted, result.to_dict()
    assert result.delta_px == delta


def test_rejects_unrelated_content(scene):
    result = OverlapAligner().estimate(scene[:720], make_scene(720, 640, seed=928))
    assert not result.accepted
    assert result.delta_px is None


def test_rejects_textureless_and_periodic_screens():
    blank = np.full((720, 640, 3), 240, np.uint8)
    assert not OverlapAligner().estimate(blank, blank).accepted
    tile = np.full((80, 640, 3), 242, np.uint8)
    cv2.putText(tile, "same message same message", (70, 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (20, 20, 20), 1, cv2.LINE_AA)
    periodic = np.tile(tile, (20, 1, 1))
    result = OverlapAligner().estimate(periodic[:720], periodic[80:800])
    assert not result.accepted, result.to_dict()


def test_rejects_resize_and_horizontal_motion(scene):
    previous = scene[:720]
    assert not OverlapAligner().estimate(previous, previous[:, :-2]).accepted
    current = np.roll(scene[260:980], 12, axis=1)
    assert not OverlapAligner().estimate(previous, current).accepted


def test_can_mask_fixed_overlay(scene):
    previous = scene[:720].copy()
    current = scene[240:960].copy()
    previous[240:320, 160:480] = 0
    current[240:320, 160:480] = 255
    config = AlignmentConfig(masks=(Rect(160, 240, 320, 80),))
    result = OverlapAligner(config).estimate(previous, current)
    assert result.accepted, result.to_dict()
    assert result.delta_px == 240


def test_rejects_nonrigid_relayout(scene):
    previous = scene[:720]
    current = previous.copy()
    current[300:] = scene[400:820]
    result = OverlapAligner().estimate(previous, current)
    assert not result.accepted, result.to_dict()


def test_tracker_does_not_advance_after_failure(scene):
    tracker = CoordinateTracker(OverlapAligner(), origin_y_px=40)
    assert tracker.accept(scene[:720], "a")[0].accepted
    assert tracker.accept(scene[250:970], "b")[0].accepted
    bad = np.full((720, 640, 3), 255, np.uint8)
    result, reference = tracker.accept(bad, "bad")
    assert not result.accepted
    assert reference == "b"
    assert tracker.reference_id == "b"
    assert tracker.scroll_offset_px == 250
    assert tracker.accept(scene[450:1170], "c")[0].accepted
    assert tracker.scroll_offset_px == 450
    assert tracker.global_y(120) == 530
