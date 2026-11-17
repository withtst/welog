import dataclasses

import pytest

from welog.models import Rect
from welog.picker import HoverLock, RegionCalibration, WindowTarget, selection_to_pixels


def target(hwnd=21):
    return WindowTarget(hwnd, 100, "微信", Rect(30, 40, 1200, 900), "weixin.exe")


def test_continuous_hover_locks_only_one_target():
    lock = HoverLock(1.5)
    first = target()
    assert lock.observe(first, 10).progress == 0
    assert not lock.observe(first, 11).locked
    assert lock.observe(first, 11.5).locked
    assert lock.observe(target(99), 12).target == first
    assert lock.observe(None, 13).locked
    lock.reset()
    assert not lock.observe(target(99), 15).locked


def test_switching_windows_leaving_or_reusing_handle_resets_dwell():
    lock = HoverLock(1)
    original = target()
    lock.observe(original, 1)
    assert lock.observe(target(22), 1.8).progress == 0
    assert lock.observe(None, 2.5).target is None
    lock.observe(original, 3)
    reused = dataclasses.replace(original, pid=900)
    assert lock.observe(reused, 3.9).progress == 0
    moved = dataclasses.replace(reused, client=Rect(31, 40, 1200, 900))
    assert not lock.observe(moved, 4.8).locked
    assert lock.observe(moved, 5.8).locked


def test_native_selection_coordinates_survive_scaling_and_reverse_drag():
    assert selection_to_pixels((450, 300), (100, 50), (600, 450), (1200, 900)) == Rect(200, 100, 700, 500)
    assert selection_to_pixels((-20, -20), (900, 900), (600, 450), (1200, 900)) == Rect(0, 0, 1200, 900)
    with pytest.raises(ValueError):
        selection_to_pixels((20, 20), (20, 20), (600, 450), (1200, 900))


def test_calibration_reuse_requires_same_layout_and_contained_nonoverlapping_regions():
    saved = RegionCalibration("weixin.exe", 1200, 900, Rect(300, 80, 880, 600), Rect(300, 0, 400, 60))
    assert saved.matches(target(99))
    assert not saved.matches(dataclasses.replace(target(), executable="other.exe"))
    assert not saved.matches(dataclasses.replace(target(), client=Rect(30, 40, 1100, 900)))
    assert not dataclasses.replace(saved, roi=Rect(300, 80, 1000, 600)).matches(target())
    assert not dataclasses.replace(saved, guard=Rect(300, 100, 400, 60)).matches(target())
