"""Offscreen Qt widgets, fake desktop, fake frames: no real windows are touched."""

import os
import time
from types import SimpleNamespace

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from welog.demo import make_scene
from welog.gui import CaptureWindow, CaptureWorker, PickOverlay, RegionDialog, configure_font
from welog.models import CaptureConfig, Rect
from welog.picker import WindowTarget
from welog.picker import RegionCalibration


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    instance.setQuitOnLastWindowClosed(False)
    configure_font(instance)
    return instance


def target():
    return WindowTarget(123, 456, "微信", Rect(40, 60, 640, 720), "weixin.exe")


def test_cancelled_selector_never_activates_or_captures(app, tmp_path):
    def forbidden(*args):
        raise AssertionError("Cancelled selection must never touch target windows")
    desktop = SimpleNamespace(escape_pressed=lambda: True, window_at_cursor=forbidden,
                              activate=forbidden, describe_window=forbidden)
    window = CaptureWindow(desktop, tmp_path)
    window.begin_pick()
    window.overlay.poll()
    assert window.stage == "idle"
    assert window.target is None
    assert not list(tmp_path.iterdir())
    window.hide()


def test_hovering_mock_window_locks_without_activation(app):
    desktop = SimpleNamespace(escape_pressed=lambda: False, window_at_cursor=lambda _: target())
    overlay = PickOverlay(desktop, app.primaryScreen(), dwell_seconds=0.3)
    values = []
    overlay.locked.connect(values.append)
    overlay.active = True
    overlay.hover.observe(target(), time.monotonic() - 1)
    overlay.poll()
    assert values == [target()]
    assert not overlay.timer.isActive()
    overlay.poll()
    assert values == [target()]
    overlay.hide()


def test_screenshot_selection_and_optional_title_calibration(app):
    dialog = RegionDialog(target(), np.full((720, 640, 3), 245, np.uint8))
    canvas = dialog.canvas
    dialog.show()
    app.processEvents()
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(50, 50))
    QTest.mouseMove(canvas, QPoint(canvas.width() - 10, canvas.height() - 50))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(canvas.width() - 10, canvas.height() - 50))
    assert dialog.next_button.isEnabled()
    expected = canvas.selected()
    dialog.confirm_selection()
    assert dialog.roi == expected
    assert dialog.phase == "guard"
    assert not dialog.next_button.isEnabled()
    dialog.skip_button.click()
    assert dialog.result() == dialog.DialogCode.Accepted
    assert dialog.guard is None
    dialog.hide()


def test_worker_uses_fake_source_and_generates_replay(app, tmp_path):
    scene = make_scene(720, 640)

    class FakeSource:
        def read(self):
            return scene.copy()
        def close(self):
            pass
        def scroll(self, units):
            raise AssertionError("One-frame fixture must not scroll")

    desktop = SimpleNamespace(describe_window=lambda _: target())
    worker = CaptureWorker(target(), Rect(0, 0, 640, 720), None, tmp_path,
                           CaptureConfig(max_frames=1, sample_interval_s=0.001, stable_samples=2),
                           desktop_factory=lambda: desktop, source_factory=lambda *args: FakeSource())
    results, errors = [], []
    worker.result_ready.connect(results.append)
    worker.failed.connect(errors.append)
    worker.start()
    deadline = time.monotonic() + 5
    while worker.isRunning() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert worker.wait(1000)
    app.processEvents()
    assert not errors
    assert results[0]["accepted"] == 1
    assert (tmp_path / "capture/report.html").exists()
    assert not list(tmp_path.rglob('*.png'))
    assert results[0]['image_memory_remaining_bytes'] == 0


def test_cached_region_goes_to_countdown_and_cancel_prevents_late_start(app, tmp_path):
    def forbidden(*args):
        raise AssertionError("Cancelled countdown must not activate or capture")
    desktop = SimpleNamespace(activate=forbidden, escape_pressed=lambda: False)
    window = CaptureWindow(desktop, tmp_path)
    window.cached_calibration = RegionCalibration("weixin.exe", 640, 720, Rect(100, 70, 520, 500))
    window.window_locked(target())
    assert window.stage == "countdown"
    assert window.roi == Rect(100, 70, 520, 500)
    window.cancel()
    window.countdown_tick()
    assert window.stage == "idle"
    assert window.worker is None
    assert not list(tmp_path.iterdir())
    window.hide()


def test_reused_window_handle_cannot_start_capture_worker(app, tmp_path):
    def forbidden(*args):
        raise AssertionError("Changed window must not construct a source")
    desktop = SimpleNamespace(describe_window=lambda _: None)
    worker = CaptureWorker(target(), Rect(0, 0, 640, 720), None, tmp_path,
                           CaptureConfig(max_frames=1), desktop_factory=lambda: desktop, source_factory=forbidden)
    errors = []
    worker.failed.connect(errors.append)
    worker.run()
    assert errors
    assert not list(tmp_path.iterdir())


def test_gui_defaults_to_history_and_explains_zero_motion(app, tmp_path):
    window = CaptureWindow(SimpleNamespace(), tmp_path)
    assert window.direction.currentData() == "up"
    window.capture_finished({"status": "needs_attention", "reason": "no_scroll_motion",
                             "accepted": 4, "final_scroll_offset_px": 0})
    assert "未检测到滚动" in window.status.text()
    assert "底部 → 顶部" in window.status.text()
    window.hide()


@pytest.mark.parametrize("direction", ["up", "down"])
@pytest.mark.parametrize("mode",['all','limited'])
@pytest.mark.parametrize('debug',[False,True])
def test_gui_passes_selected_direction_to_capture_worker(app, tmp_path, monkeypatch, direction, mode,debug):
    from welog import gui

    calls = []

    class FakeWorker:
        def __init__(self, target_value, roi, guard, output, config):
            calls.append((config.direction,config.max_frames,config.debug_save_images))
            signal = SimpleNamespace(connect=lambda *args: None)
            self.progress = self.result_ready = self.failed = self.finished = signal
        def start(self):
            calls.append("started")
        def isRunning(self):
            return False

    monkeypatch.setattr(gui, "CaptureWorker", FakeWorker)
    desktop = SimpleNamespace(activate=lambda hwnd: calls.append(("activate_fake", hwnd)))
    window = CaptureWindow(desktop, tmp_path)
    window.target = target()
    window.roi = Rect(100, 70, 520, 500)
    window.stage = "countdown"
    window.direction.setCurrentIndex(window.direction.findData(direction))
    window.capture_mode.setCurrentIndex(window.capture_mode.findData(mode))
    window.frame_limit.setValue(37)
    assert not window.debug_images.isChecked()
    window.debug_images.setChecked(debug)
    assert ('保存聊天区截图' if debug else '不保存截图') in window.storage_hint.text()
    assert ('开始调试' in window.pick_button.text())==debug
    window.start_capture()
    assert calls == [("activate_fake", 123), (direction,0 if mode=='all' else 37,debug), "started"]
    assert window.stage == "capturing"
    window.hide()
