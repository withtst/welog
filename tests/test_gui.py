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
@pytest.mark.parametrize('backend',['foreground','background'])
def test_gui_passes_selected_direction_to_capture_worker(app, tmp_path, monkeypatch, direction, mode,debug,backend):
    from welog import gui

    calls = []

    class FakeWorker:
        def __init__(self, target_value, roi, guard, output, config):
            calls.append((config.direction,config.max_frames,config.debug_save_images,config.capture_backend))
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
    window.backend.setCurrentIndex(window.backend.findData(backend))
    assert not window.debug_images.isChecked()
    window.debug_images.setChecked(debug)
    assert ('保存聊天区截图' if debug else '不保存截图') in window.storage_hint.text()
    assert ('开始调试' in window.pick_button.text())==debug
    window.start_capture()
    assert calls == ([('activate_fake',123)] if backend=='foreground' else [])+[(direction,0 if mode=='all' else 37,debug,backend),'started']
    assert window.stage == "capturing"
    window.hide()


def test_background_countdown_ignores_escape_in_other_software(app,tmp_path):
    def forbidden():raise AssertionError('Global escape must not be read')
    window=CaptureWindow(SimpleNamespace(escape_pressed=forbidden),tmp_path)
    assert window.backend.currentData()=='background'
    window.target=target();window.begin_countdown();window.countdown_tick()
    assert window.stage=='countdown'
    assert window.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    window.cancel();window.hide()


def test_background_calibration_uses_window_source_and_never_activates_target(app,tmp_path,monkeypatch):
    from welog import gui,background
    calls=[]
    def forbidden(*args):raise AssertionError('Target must not be activated')
    desktop=SimpleNamespace(activate=forbidden,describe_window=lambda _:target(),escape_pressed=forbidden,
                            check_background=lambda selected:selected.client)
    class Source:
        def __init__(self,*args):calls.append('window_source')
        def read(self):return np.full((720,640,3),245,np.uint8)
        def close(self):calls.append('closed')
    def factory(backend):
        assert backend=='background'
        return Source
    class Dialog:
        DialogCode=SimpleNamespace(Accepted=1)
        roi=Rect(50,60,500,500);guard=Rect(100,0,100,40)
        def __init__(self,*args):pass
        def setAttribute(self,attribute,value):assert attribute==Qt.WidgetAttribute.WA_ShowWithoutActivating and value
        def exec(self):return 1
    monkeypatch.setattr(background,'source_factory',factory)
    monkeypatch.setattr(gui,'RegionDialog',Dialog)
    window=CaptureWindow(desktop,tmp_path);window.target=target();window.stage='calibrating'
    window.calibrate()
    assert calls==['window_source','closed'] and window.stage=='countdown'
    window.cancel();window.hide()

@pytest.mark.parametrize('code,expected',[
    ('window_title_changed','未读数量以外'),('window_pid_changed','窗口所属进程'),
    ('client_geometry_changed','窗口客户区尺寸'),('scroll_control_changed','滚动控件已隐藏或更换'),
    ('client_dpi_changed','窗口DPI'),('window_mapping_unstable','窗口持续移动'),
    ('scroll_control_pid_changed','滚动控件所属进程'),('conversation_guard_changed','会话校验区域'),
    (None,'窗口布局、聊天控件或会话校验')])
def test_background_target_failure_shows_specific_reason_offscreen(app,tmp_path,code,expected):
    window=CaptureWindow(SimpleNamespace(),tmp_path)
    result={'status':'needs_attention','reason':'BackgroundTargetChanged','accepted':3,
            'final_scroll_offset_px':-141,'report':str(tmp_path/'report.html'),'output':str(tmp_path)}
    if code:result['source_error']={'code':code,'details':{}}
    window.capture_finished(result)
    assert expected in window.status.text()
    window.hide()


def test_background_worker_accepts_translation_since_lock_using_shared_validation(app,tmp_path):
    calls=[];results=[];failures=[]
    locked=target()
    desktop=SimpleNamespace(check_background=lambda selected:calls.append(selected) or Rect(-600,-300,640,720))
    scene=make_scene(720,640)
    class Source:
        def read(self):return scene.copy()
        def close(self):pass
        def scroll(self,units):raise AssertionError('Single-frame test must not scroll')
    worker=CaptureWorker(locked,Rect(0,0,640,720),None,tmp_path,
        CaptureConfig(capture_backend='background',max_frames=1,stable_samples=2,sample_interval_s=.001),
        desktop_factory=lambda:desktop,source_factory=lambda *args:Source())
    worker.result_ready.connect(results.append);worker.failed.connect(failures.append);worker.run()
    assert calls==[locked] and not failures and results[0]['accepted']==1
    assert not list(tmp_path.rglob('*.png'))

@pytest.mark.parametrize('code,label', [('window_hidden', '隐藏'), ('window_minimized', '最小化')])
def test_background_wait_and_timeout_have_distinct_status_messages(app, tmp_path, code, label):
    window = CaptureWindow(SimpleNamespace(), tmp_path)
    window.capture_progress({'waiting_for_target': code})
    assert label in window.status.text() and '自动继续' in window.status.text()
    window.capture_finished({'status': 'needs_attention', 'reason': 'BackgroundTargetUnavailable',
                             'source_error': {'code': code}, 'accepted': 1})
    assert label in window.status.text() and '采集结束' in window.status.text()
    window.hide()


@pytest.mark.parametrize('from_record_page', [False, True])
def test_selected_record_resumes_in_place_with_original_direction_and_region(app, tmp_path, monkeypatch, from_record_page):
    from dataclasses import asdict, replace
    from welog import gui
    from welog.controller import run_capture
    from test_full_capture import Conversation, KnownText

    selected = target()
    region = Rect(21, 31, 400, 500)
    path = tmp_path/'saved'/'capture'
    source = Conversation('up', count=14)
    progress = []
    run_capture(source, path, config=CaptureConfig(direction='up', capture_backend='background',
        max_frames=0, debug_save_images=True, sample_interval_s=.001, stable_samples=2, settle_delay_s=0),
        metadata={'target': asdict(selected), 'roi': region.to_dict()}, ocr_backend=KnownText(source),
        progress=progress.append, stop_requested=lambda: bool(progress))

    calls = []
    def forbidden(*args):
        raise AssertionError('Background resume must not activate the fixture window')

    class Worker:
        def __init__(self, window, roi, guard, output, config, *, resume):
            calls.append((window, roi, guard, output, config.direction, config.capture_backend, resume))
            signal = SimpleNamespace(connect=lambda *args: None)
            self.progress = self.result_ready = self.failed = self.finished = signal

        def start(self):
            pass

    monkeypatch.setattr(gui, 'CaptureWorker', Worker)
    window = CaptureWindow(SimpleNamespace(activate=forbidden, escape_pressed=lambda: False), tmp_path)
    window.sessions.setCurrentRow(0)
    assert window.resume_button.isEnabled()
    window.direction.setCurrentIndex(window.direction.findData('down'))
    window.backend.setCurrentIndex(window.backend.findData('foreground'))
    if from_record_page:
        from welog.exports import RecordSource
        window.sessions.clearSelection()
        window.sessions.setCurrentItem(None)
        window.records_panel.source = RecordSource(path)
        window.records_panel.update_actions()
        window.records_panel.resume_button.click()
    else:
        window.begin_resume()
    assert window.stage == 'picking'
    window.overlay.finish()
    moved = replace(selected, hwnd=789, client=Rect(-500, 90, 640, 720))
    window.window_locked(moved)
    window.countdown_timer.stop()
    window.start_capture()
    assert calls == [(moved, region, None, path, 'up', 'background', True)]
    assert list(tmp_path.glob('pointer-*')) == []
    window.hide()
