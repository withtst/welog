"""User-operated pointer selection UI. Importing it never queries or controls the desktop."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QFontDatabase, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QSpinBox, QStackedWidget,
    QVBoxLayout, QWidget,
)

from .archive import write_json
from .controller import run_capture
from .models import AlignmentConfig, CaptureConfig, Rect
from .picker import HoverLock, HoverState, RegionCalibration, WindowTarget, selection_to_pixels
from .desktop_theme import GlassBackground, GlassCard, STYLESHEET, label, line_icon, primary
from .exports import discover_sessions
from .records_ui import RecordsPanel
from .record_actions import delete_record, plan_deletion
from .record_state import record_name
from .exports import RecordSource

logger = logging.getLogger("welog.gui")


def configure_font(app: QApplication) -> None:
    """Load a local Chinese font into this process, including on the offscreen platform."""
    font_root = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    families = []
    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf"):
        path = font_root / name
        if path.exists():
            identifier = QFontDatabase.addApplicationFont(str(path))
            if identifier >= 0:
                families.extend(QFontDatabase.applicationFontFamilies(identifier))
    if families:
        app.setFont(QFont(families[0], 10))


def image_pixmap(image: np.ndarray) -> QPixmap:
    height, width = image.shape[:2]
    pixels = np.ascontiguousarray(image)
    return QPixmap.fromImage(QImage(pixels.data, width, height, pixels.strides[0], QImage.Format.Format_BGR888).copy())


class PickOverlay(QWidget):
    locked = Signal(object)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, desktop, screen, dwell_seconds: float = 1.5):
        super().__init__()
        self.desktop, self.screen = desktop, screen
        self.hover = HoverLock(dwell_seconds)
        self.state = HoverState(None, 0, False)
        self.active = False
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool | Qt.WindowType.WindowTransparentForInput
                            | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setGeometry(screen.geometry())
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.poll)

    def begin(self) -> None:
        self.hover.reset()
        self.desktop.escape_pressed()  # Discard an old key transition before arming.
        self.active = True
        self.show()
        self.timer.start()

    def poll(self) -> None:
        if not self.active:
            return
        try:
            if self.desktop.escape_pressed():
                self.finish()
                self.cancelled.emit()
                return
            target = self.desktop.window_at_cursor({"weixin.exe", "wechat.exe"})
            self.state = self.hover.observe(target, time.monotonic())
            self.update()
            if self.state.locked:
                self.finish()
                self.locked.emit(self.state.target)
        except Exception as error:
            self.finish()
            logger.exception("window_picker_error")
            self.failed.emit(str(error))

    def finish(self) -> None:
        self.active = False
        self.timer.stop()
        self.hide()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(10, 20, 38, 35))
        target = self.state.target
        if target is not None:
            scale = self.screen.devicePixelRatio()
            origin = self.screen.geometry().topLeft()
            box = target.client
            bounds = QRectF(box.x / scale - origin.x(), box.y / scale - origin.y(),
                            box.width / scale, box.height / scale)
            painter.setPen(QPen(QColor("#4fa3ff"), 3))
            painter.drawRect(bounds.adjusted(1, 1, -1, -1))
            label = f"{target.title} · {math.ceil((1 - self.state.progress) * self.hover.dwell_seconds * 10) / 10:.1f} 秒后锁定"
        else:
            label = "将光标移到微信聊天窗口内，停留片刻锁定 · Esc 取消"
        panel = QRectF(36, 30, min(720, self.width() - 72), 66)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(15, 27, 45, 235))
        painter.drawRoundedRect(panel, 14, 14)
        painter.setPen(QColor("white"))
        painter.drawText(panel.adjusted(20, 0, -20, 0), Qt.AlignmentFlag.AlignVCenter, label)
        if target is not None:
            painter.fillRect(QRectF(50, 88, (panel.width() - 28) * self.state.progress, 3), QColor("#4fa3ff"))


class SelectionCanvas(QWidget):
    selection_changed = Signal()

    def __init__(self, image: np.ndarray, maximum_width: int, maximum_height: int):
        super().__init__()
        self.pixmap = image_pixmap(image)
        self.image_size = (image.shape[1], image.shape[0])
        scale = min(1, maximum_width / self.image_size[0], maximum_height / self.image_size[1])
        self.setFixedSize(max(1, round(self.image_size[0] * scale)), max(1, round(self.image_size[1] * scale)))
        self.start: QPointF | None = None
        self.end: QPointF | None = None
        self.dragging = False
        self.chat_roi: Rect | None = None
        self.setCursor(Qt.CursorShape.CrossCursor)

    def selected(self) -> Rect | None:
        if self.start is None or self.end is None:
            return None
        try:
            return selection_to_pixels((self.start.x(), self.start.y()), (self.end.x(), self.end.y()),
                                       (self.width(), self.height()), self.image_size)
        except ValueError:
            return None

    def reset(self) -> None:
        self.start = self.end = None
        self.update()
        self.selection_changed.emit()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.start = self.end = event.position()
            self.dragging = True
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.dragging:
            self.end = event.position()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.dragging:
            self.end = event.position()
            self.dragging = False
            self.update()
            self.selection_changed.emit()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.pixmap)
        if self.chat_roi is not None:
            sx, sy = self.width() / self.image_size[0], self.height() / self.image_size[1]
            roi = self.chat_roi
            painter.setPen(QPen(QColor("#32b67a"), 2))
            painter.drawRect(QRectF(roi.x * sx, roi.y * sy, roi.width * sx, roi.height * sy))
        if self.start is not None and self.end is not None:
            painter.setBrush(QColor(35, 125, 245, 35))
            painter.setPen(QPen(QColor("#2785ff"), 2))
            painter.drawRect(QRectF(self.start, self.end).normalized())


class RegionDialog(QDialog):
    def __init__(self, target: WindowTarget, image: np.ndarray, parent=None, *, start_after_calibration=True,with_guard=True):
        super().__init__(parent)
        self.setWindowTitle(f"校准聊天区 · {target.title}")
        self.roi: Rect | None = None
        self.guard: Rect | None = None
        self.phase = "chat"
        self.start_after_calibration,self.with_guard=start_after_calibration,with_guard
        self.instructions = QLabel("拖动框选可滚动的消息区域，排除顶部标题、输入框和滚动条。")
        self.instructions.setWordWrap(True)
        available = QApplication.primaryScreen().availableGeometry()
        self.canvas = SelectionCanvas(image, min(1180, available.width() - 100), available.height() - 220)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.next_button = self.buttons.addButton("确认聊天区", QDialogButtonBox.ButtonRole.AcceptRole)
        self.next_button.setEnabled(False)
        self.skip_button = self.buttons.addButton("跳过标题校准并开始" if start_after_calibration else "跳过标题并保存", QDialogButtonBox.ButtonRole.ActionRole)
        self.skip_button.hide()
        self.buttons.rejected.connect(self.reject)
        self.next_button.clicked.connect(self.confirm_selection)
        self.skip_button.clicked.connect(self.accept)
        self.canvas.selection_changed.connect(self.update_selection)
        layout = QVBoxLayout(self)
        layout.addWidget(self.instructions)
        layout.addWidget(self.canvas)
        layout.addWidget(self.buttons)

    def update_selection(self) -> None:
        box = self.canvas.selected()
        valid = box is not None
        if valid and self.phase == "chat":
            valid = box.width >= 160 and box.height >= 120
        if valid and self.phase == "guard" and self.roi is not None:
            valid = not (box.x < self.roi.right and box.right > self.roi.x
                         and box.y < self.roi.bottom and box.bottom > self.roi.y)
        self.next_button.setEnabled(valid)

    def confirm_selection(self) -> None:
        box = self.canvas.selected()
        if box is None or not self.next_button.isEnabled():
            return
        if self.phase == "chat":
            self.roi = self.canvas.chat_roi = box
            if not self.with_guard:
                self.accept();return
            self.phase = "guard"
            self.canvas.reset()
            self.instructions.setText("再框选固定的会话名称，避开未读计数。也可跳过。" +
                                      ("确认后倒计时开始采集。" if self.start_after_calibration else "确认后保存校准。"))
            self.next_button.setText("确认标题并开始" if self.start_after_calibration else "保存校准")
            self.skip_button.show()
        else:
            self.guard = box
            self.accept()


class CaptureWorker(QThread):
    progress = Signal(dict)
    result_ready = Signal(dict)
    failed = Signal(str)

    def __init__(
        self, target: WindowTarget, roi: Rect, guard: Rect | None, output: Path,
        config: CaptureConfig, *, desktop_factory=None, source_factory=None,
    ):
        super().__init__()
        self.target, self.roi, self.guard, self.output, self.config = target, roi, guard, output, config
        self.desktop_factory, self.source_factory = desktop_factory, source_factory
        self.cancel_event = threading.Event()

    def stop(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            if self.cancel_event.is_set():
                self.result_ready.emit({"status": "stopped", "reason": "user_interrupt", "accepted": 0})
                return
            from .windows import WindowsDesktop
            from .background import source_factory
            desktop = (self.desktop_factory or WindowsDesktop)()
            if self.config.capture_backend=='background':desktop.check_background(self.target)
            elif desktop.describe_window(self.target.hwnd)!=self.target:
                raise RuntimeError("锁定的窗口已发生变化，请重新选择窗口。")
            source = (self.source_factory or source_factory(self.config.capture_backend))(desktop, self.target.hwnd, self.roi, self.guard)
            result = run_capture(
                source, str(self.output / "capture"), config=self.config,
                metadata={"selection": "pointer_hover", "target": asdict(self.target),
                          "roi": self.roi.to_dict(), "guard_roi": self.guard.to_dict() if self.guard else None},
                progress=self.progress.emit, stop_requested=self.cancel_event.is_set,
            )
            self.result_ready.emit(result)
        except Exception as error:
            logger.exception("capture_worker_error")
            self.failed.emit(str(error))


class CaptureWindow(QMainWindow):
    def __init__(self, desktop, output_root: Path, *, dwell_seconds: float = 1.5, countdown: int = 3, max_frames: int = 0):
        super().__init__()
        self.desktop, self.output_root = desktop, Path(output_root).resolve()
        self.dwell_seconds, self.countdown_seconds = dwell_seconds, countdown
        self.target: WindowTarget | None = None
        self.roi: Rect | None = None
        self.guard: Rect | None = None
        self.report_path: str | None = None
        self.worker: CaptureWorker | None = None
        self.overlay: PickOverlay | None = None
        self.cached_calibration: RegionCalibration | None = None
        self.cache_path = self.output_root / "pointer-calibration.json"
        try:
            self.cached_calibration = RegionCalibration.from_dict(json.loads(self.cache_path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, KeyError):
            pass
        self.stage = "idle"
        self.remaining = 0
        self.countdown_deadline = 0.0
        self.setWindowTitle("WeLog · 本地聊天记录工作台")
        self.resize(1280, 880)
        self.setMinimumSize(1160, 760)
        self.setStyleSheet(STYLESHEET)
        body = GlassBackground()
        shell = QHBoxLayout(body)
        shell.setContentsMargins(26, 26, 30, 26)
        shell.setSpacing(26)
        sidebar = GlassCard()
        sidebar.setFixedWidth(208)
        rail = QVBoxLayout(sidebar)
        rail.setContentsMargins(15, 28, 15, 20)
        rail.setSpacing(10)
        brand = label("WeLog", "title")
        brand.setStyleSheet("font-size:30px; padding-left:12px; letter-spacing:1px")
        rail.addWidget(brand)
        rail.addSpacing(18)
        navigation = QButtonGroup(self)
        self.capture_nav = QPushButton("  采集工作台")
        self.capture_nav.setIcon(line_icon('capture'))
        self.records_nav = QPushButton("  记录与导出")
        self.records_nav.setIcon(line_icon('records'))
        for index,button in enumerate([self.capture_nav,self.records_nav]):
            button.setCheckable(True)
            button.setProperty('role','nav')
            navigation.addButton(button,index)
            rail.addWidget(button)
        self.capture_nav.setChecked(True)
        navigation.idClicked.connect(self.show_page)
        rail.addSpacing(24)
        recent = label("最近的记录", "eyebrow")
        recent.setContentsMargins(12, 0, 0, 0)
        rail.addWidget(recent)
        self.sessions = QListWidget()
        self.sessions.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sessions.itemClicked.connect(lambda item:self.show_records(item.data(Qt.ItemDataRole.UserRole)))
        rail.addWidget(self.sessions,1)
        self.refresh_button = QPushButton("刷新记录")
        self.refresh_button.clicked.connect(self.refresh_sessions)
        rail.addWidget(self.refresh_button)
        shell.addWidget(sidebar)
        workspace = QVBoxLayout()
        workspace.setSpacing(22)
        heading = QHBoxLayout()
        words = QVBoxLayout()
        words.setSpacing(8)
        self.page_heading = label("采集工作台", "title")
        words.addWidget(self.page_heading)
        heading.addLayout(words,1)
        workspace.addLayout(heading)
        self.pages = QStackedWidget()
        self.pages.addWidget(self.build_capture_page(max_frames))
        self.records_panel = RecordsPanel(self.output_root)
        self.records_panel.delete_requested.connect(self.delete_saved_record)
        self.records_panel.record_changed.connect(self.refresh_sessions)
        self.pages.addWidget(self.records_panel)
        workspace.addWidget(self.pages,1)
        shell.addLayout(workspace,1)
        self.setCentralWidget(body)
        self.refresh_sessions()
        self.countdown_timer = QTimer(self)
        self.countdown_timer.setInterval(80)
        self.countdown_timer.timeout.connect(self.countdown_tick)

    def build_capture_page(self,max_frames):
        page=QWidget()
        layout=QVBoxLayout(page)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(20)
        card=GlassCard()
        content=QVBoxLayout(card)
        content.setContentsMargins(32,30,32,30)
        content.setSpacing(18)
        content.addWidget(label("采集设置", "heading"))
        row=QHBoxLayout()
        row.setSpacing(14)
        row.addWidget(label("模式", "muted"))
        self.capture_mode=QComboBox()
        self.capture_mode.addItem('完整采集','all')
        self.capture_mode.addItem('限量采集','limited')
        self.capture_mode.setCurrentIndex(0 if max_frames==0 else 1)
        row.addWidget(self.capture_mode)
        row.addWidget(label("方向", "muted"))
        self.direction=QComboBox()
        self.direction.addItem("底部 → 顶部", "up")
        self.direction.addItem("顶部 → 底部", "down")
        self.direction.setMinimumWidth(160)
        row.addWidget(self.direction)
        self.frame_label=label("帧数", "muted")
        row.addWidget(self.frame_label)
        self.frame_limit=QSpinBox()
        self.frame_limit.setRange(1,10000)
        self.frame_limit.setValue(max_frames or 100)
        self.frame_limit.setSuffix(" 帧")
        self.frame_limit.setFixedWidth(110)
        row.addWidget(self.frame_limit)
        row.addStretch()
        content.addLayout(row)
        backend_row=QHBoxLayout()
        backend_row.addWidget(label('采集方式','muted'))
        self.backend=QComboBox()
        self.backend.addItem('后台窗口采集（不占用鼠标）','background')
        self.backend.addItem('前台兼容采集','foreground')
        backend_row.addWidget(self.backend)
        backend_row.addStretch()
        content.addLayout(backend_row)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating,True)
        self.start_hint=label('', 'muted')
        self.start_hint.setWordWrap(True)
        content.addWidget(self.start_hint)
        self.capture_mode.currentIndexChanged.connect(self.update_capture_mode)
        self.direction.currentIndexChanged.connect(self.update_capture_mode)
        self.update_capture_mode()
        self.reuse_regions=QCheckBox("复用上次聊天区域")
        self.reuse_regions.setToolTip("窗口尺寸相同时复用；字体、布局变化后请取消勾选并重新框选。")
        self.reuse_regions.setChecked(True)
        content.addWidget(self.reuse_regions)
        self.debug_images=QCheckBox('调试采集（仅异常和未知消息截图）')
        self.debug_images.setToolTip('正常帧不保存；遇到异常保留问题帧、上一确认参考帧和稳定时间窗，未知消息另存区域。达到 1 GiB 停止，每次启动默认关闭。')
        content.addWidget(self.debug_images)
        self.storage_hint=label('','muted')
        content.addWidget(self.storage_hint)
        content.addSpacing(4)
        buttons=QHBoxLayout()
        self.pick_button=primary(QPushButton("选择窗口并开始"))
        self.pick_button.setToolTip("点击后，将光标移到微信聊天窗口，停留片刻锁定并校准。")
        self.cancel_button=QPushButton("取消 / 停止")
        self.report_button=QPushButton("查看本次记录")
        self.debug_images.toggled.connect(self.update_debug_ui)
        self.update_debug_ui()
        self.report_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.pick_button.clicked.connect(self.begin_pick)
        self.cancel_button.clicked.connect(self.cancel)
        self.report_button.clicked.connect(self.open_report)
        buttons.addWidget(self.pick_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.report_button)
        buttons.addStretch()
        content.addLayout(buttons)
        layout.addWidget(card)
        progress=GlassCard()
        status_layout=QVBoxLayout(progress)
        status_layout.setContentsMargins(30,24,30,24)
        status_layout.addWidget(label("采集状态", "heading"))
        self.status=label("未开始", "muted")
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(75)
        status_layout.addWidget(self.status)
        layout.addWidget(progress)
        layout.addStretch()
        return page

    def capture_storage_label(self):
        return '调试采集：保存聊天区截图（仅异常/未知，上限 1 GiB）' if self.debug_images.isChecked() else '普通采集：不保存截图'

    def update_debug_ui(self):
        self.storage_hint.setText(self.capture_storage_label())
        self.pick_button.setText('选择窗口并开始调试' if self.debug_images.isChecked() else '选择窗口并开始')

    def update_capture_mode(self):
        limited=self.capture_mode.currentData()=='limited'
        self.frame_limit.setVisible(limited)
        self.frame_label.setVisible(limited)
        self.frame_limit.setEnabled(limited)
        self.start_hint.setText('请先将聊天定位到最底部。' if self.direction.currentData()=='up' else '请先手动切换到最早一条聊天记录。')

    def show_page(self,index):
        self.pages.setCurrentIndex(index)
        self.capture_nav.setChecked(index==0)
        self.records_nav.setChecked(index==1)
        self.page_heading.setText("采集工作台" if index==0 else "记录与导出")

    def refresh_sessions(self):
        self.sessions.clear()
        try:
            for path in discover_sessions(self.output_root):
                name=record_name(RecordSource(path))
                item=QListWidgetItem(name)
                item.setToolTip(str(path))
                item.setData(Qt.ItemDataRole.UserRole,str(path))
                self.sessions.addItem(item)
        except OSError:
            logger.warning('Unable to enumerate saved records',exc_info=True)
        if not self.sessions.count():
            item=QListWidgetItem('暂无记录')
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.sessions.addItem(item)

    def show_records(self,path):
        if not path:return
        self.show_page(1)
        self.records_panel.search.clear()
        self.records_panel.load_source(path)

    def delete_saved_record(self,path):
        if self.stage!='idle' or self.records_panel.busy() or (self.worker is not None and self.worker.isRunning()):
            self.records_panel.feedback.setText('请先停止采集或等待当前操作完成。')
            return
        try:
            plan=plan_deletion(path,self.output_root)
            scope='目录及其中全部文件' if plan.directory else '记录及其编辑数据和附件'
            targets=[plan.directory] if plan.directory else [*plan.files,*plan.asset_directories]
            answer=QMessageBox.question(self,'删除记录',f'永久删除{scope}？\n'+ '\n'.join(map(str,targets)),
                QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)
            if answer!=QMessageBox.StandardButton.Yes:return
            delete_record(plan)
            source=self.records_panel.source
            if source and (source.path in plan.files or (plan.directory and source.path.is_relative_to(plan.directory))):
                self.records_panel.clear_source()
            if self.report_path and (Path(self.report_path).resolve() in plan.files or
                                     (plan.directory and Path(self.report_path).resolve().is_relative_to(plan.directory))):
                self.report_path=None
                self.report_button.setEnabled(False)
            self.records_panel.feedback.setText('记录已删除')
            self.refresh_sessions()
        except (OSError,ValueError) as error:self.records_panel.feedback.setText('删除失败：'+str(error))

    def begin_pick(self) -> None:
        if self.records_panel.busy():
            self.status.setText('请等待记录操作完成。')
            return
        if self.stage != "idle" or (self.worker is not None and self.worker.isRunning()):
            return
        self.stage = "picking"
        self.target = self.roi = self.guard = None
        self.report_path = None
        self.report_button.setEnabled(False)
        self.pick_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.hide()
        self.overlay = PickOverlay(self.desktop, QApplication.primaryScreen(), self.dwell_seconds)
        self.overlay.locked.connect(self.window_locked)
        self.overlay.cancelled.connect(self.selection_cancelled)
        self.overlay.failed.connect(self.fail)
        self.overlay.begin()

    def selection_cancelled(self) -> None:
        logger.info("selection_cancelled")
        self.stage = "idle"
        self.status.setText("已取消。可以重新指向微信窗口。")
        self.restore_controls()

    def window_locked(self, target: WindowTarget) -> None:
        logger.info("window_locked hwnd=%s pid=%s", target.hwnd, target.pid)
        self.target = target
        cached = self.cached_calibration
        if self.reuse_regions.isChecked() and cached is not None and cached.matches(target):
            self.roi, self.guard = cached.roi, cached.guard
            self.begin_countdown()
            return
        self.stage = "calibrating"
        # Wait for the selector to disappear before taking a calibration snapshot.
        QTimer.singleShot(180, self.calibrate)

    def calibrate(self) -> None:
        if self.stage != "calibrating" or self.target is None:
            return
        from .background import source_factory
        source = None
        try:
            if self.backend.currentData()=='background':self.desktop.check_background(self.target)
            elif self.desktop.describe_window(self.target.hwnd) != self.target:
                raise RuntimeError("窗口位置或内容发生变化，请重新选择。")
            if self.backend.currentData()=='foreground':self.desktop.activate(self.target.hwnd)
            source = source_factory(self.backend.currentData())(self.desktop, self.target.hwnd,
                                 Rect(0, 0, self.target.client.width, self.target.client.height))
            image = None
            deadline = time.monotonic() + 3
            while image is None and time.monotonic() < deadline:
                image = source.read()
                if image is None:
                    time.sleep(0.05)
            if image is None:
                raise RuntimeError("暂时无法读取窗口，请重新选择。")
            source.close()
            source = None
            dialog = RegionDialog(self.target, image)
            if self.backend.currentData()=='background':dialog.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating,True)
            if dialog.exec() != QDialog.DialogCode.Accepted or dialog.roi is None:
                self.selection_cancelled()
                return
            self.roi, self.guard = dialog.roi, dialog.guard
            self.cached_calibration = RegionCalibration(self.target.executable, self.target.client.width,
                                                        self.target.client.height, self.roi, self.guard,self.target.dpi)
            self.output_root.mkdir(parents=True, exist_ok=True)
            write_json(self.cache_path, asdict(self.cached_calibration))
            self.begin_countdown()
        except KeyboardInterrupt:
            self.selection_cancelled()
        except Exception as error:
            logger.exception("calibration_error")
            self.fail(str(error))
        finally:
            if source is not None:
                source.close()

    def begin_countdown(self) -> None:
        self.stage = "countdown"
        self.remaining = self.countdown_seconds
        self.countdown_deadline = time.monotonic() + self.countdown_seconds
        stop_hint='点击停止按钮结束' if self.backend.currentData()=='background' else 'Esc 停止'
        self.status.setText(f"已锁定：{self.target.title}\n{self.capture_storage_label()}\n{self.remaining} 秒后开始，{stop_hint}。")
        self.show()
        self.countdown_timer.start()

    def countdown_tick(self) -> None:
        if self.stage != "countdown":
            return
        if self.backend.currentData()=='foreground' and self.desktop.escape_pressed():
            self.cancel()
            return
        self.remaining = max(0, math.ceil(self.countdown_deadline - time.monotonic()))
        if self.remaining <= 0:
            self.countdown_timer.stop()
            self.start_capture()
        else:
            self.status.setText(f"已锁定：{self.target.title}\n{self.capture_storage_label()}\n{self.remaining} 秒后开始，可点击取消。")

    def start_capture(self) -> None:
        if self.target is None or self.roi is None or self.stage != "countdown":
            return
        try:
            output = self.output_root / f"pointer-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
            output.mkdir(parents=True)
            write_json(output / "calibration.json", {
                "selection": "pointer_hover", "target": asdict(self.target),
                "title": self.target.title, "roi": self.roi.to_dict(),
                "guard_roi": self.guard.to_dict() if self.guard else None,
                "client_size": [self.target.client.width, self.target.client.height],
                "alignment_config": AlignmentConfig().to_dict(),
                "capture_direction": self.direction.currentData(),
                "capture_mode":self.capture_mode.currentData(),
                "debug_save_images":self.debug_images.isChecked(),
                "capture_backend":self.backend.currentData(),
            })
            self.stage = "capturing"
            if self.backend.currentData()=='foreground':
                self.hide()
                self.desktop.activate(self.target.hwnd)
            self.backend.setEnabled(False)
            self.worker = CaptureWorker(self.target, self.roi, self.guard, output,
                                        CaptureConfig(max_frames=0 if self.capture_mode.currentData()=='all' else self.frame_limit.value(),
                                                      direction=self.direction.currentData(),debug_save_images=self.debug_images.isChecked(),
                                                      capture_backend=self.backend.currentData()))
            logger.info("gui_capture_start direction=%s backend=%s debug_save_images=%s output=%s", self.direction.currentData(), self.backend.currentData(), self.debug_images.isChecked(), output)
            self.worker.progress.connect(self.capture_progress, Qt.ConnectionType.QueuedConnection)
            self.worker.result_ready.connect(self.capture_finished, Qt.ConnectionType.QueuedConnection)
            self.worker.failed.connect(self.fail, Qt.ConnectionType.QueuedConnection)
            self.worker.finished.connect(self.worker_finished, Qt.ConnectionType.QueuedConnection)
            self.worker.start()
        except Exception as error:
            logger.exception("gui_capture_start_error")
            self.fail(str(error))

    def capture_progress(self, record: dict) -> None:
        stop_hint='点击停止按钮结束' if self.backend.currentData()=='background' else 'Esc 停止'
        self.status.setText(f"已处理 {int(record['frame_id'])+1} 帧 · 累计位移 {record['scroll_offset_px']} px\n{stop_hint}")

    def capture_finished(self, result: dict) -> None:
        self.stage = "idle"
        self.report_path = result.get("report")
        self.report_button.setEnabled(bool(self.report_path))
        labels = {"limit_reached": "已达到采样上限", "visible_end_candidate": "当前方向可能已到可见边界",
                  "stopped": "已停止", "needs_attention": "采集暂停，需要检查",
                  "boundary_reached":"已到顶部" if result.get('capture_direction',self.direction.currentData())=='up' else "已到底部"}
        detail = result.get('reason', '')
        if detail in {'repeated_verified_zero_motion','max_frames'}:detail=''
        if detail == "no_scroll_motion":
            detail = ("启动后未检测到滚动。可能已在所选方向的边界，或聊天区未响应滚轮。\n"
                      "采集历史请从最底部选择“底部 → 顶部”；也可重新框选聊天区后重试。")
            if result.get('capture_backend',self.backend.currentData())=='background':
                detail+='\n当前微信版本可能未响应后台滚动，已暂停；可自行选择前台兼容采集。'
        elif detail.startswith("StableTimeout"):
            detail = "画面布局仍未稳定或静态内容不足。"+('已保留调试截图。' if result.get('debug_image_count') else '已记录诊断元数据，没有保存截图。')
        elif detail=='continuity_unconfirmed':
            detail="无法确认跨页连续性，已原位停止。"
        elif detail=='DebugImageLimit':detail='调试截图达到存储上限，已停止。'
        elif detail=='BackgroundFrameStale':detail='窗口停止提供新画面，已暂停；未抢夺焦点。'
        elif detail=='BackgroundCaptureUnavailable':detail='窗口采集不可用，已暂停；未切换到前台采集。'
        elif detail=='BackgroundTargetUnavailable':detail='目标窗口已隐藏、关闭或最小化，采集暂停。'
        elif detail=='BackgroundTargetChanged':
            detail={'window_pid_changed':'窗口所属进程发生变化，请重新选择窗口。',
                    'window_title_changed':'窗口标题发生了未读数量以外的变化，请确认会话并重新校准。',
                    'client_geometry_changed':'窗口客户区尺寸发生变化，请重新校准。',
                    'client_dpi_changed':'窗口DPI发生变化或无法读取，请重新校准。',
                    'window_mapping_unstable':'窗口持续移动，暂时无法建立裁剪映射，请稍后重试。',
                    'scroll_control_changed':'原聊天滚动控件已隐藏或更换，请重新选择聊天区域。',
                    'scroll_control_pid_changed':'聊天滚动控件所属进程发生变化，请重新选择窗口。',
                    'conversation_guard_changed':'会话校验区域发生变化，请确认会话并重新校准。'
                   }.get(result.get('source_error',{}).get('code'),'窗口布局、聊天控件或会话校验发生变化，请重新校准。')
        elif detail=='BackgroundScrollRejected':detail='后台滚动消息被系统拒绝，已暂停；未使用全局输入。'
        if result.get('diagnostic_image_save_error') and result.get('reason')!='DebugImageLimit':
            detail+='\n部分诊断图未能保存，请查看诊断索引中的 image_save_error。'
        if result.get('debug_save_images'):
            detail+=f"\n调试截图：{result.get('debug_image_count',0)} 张 · {Path(result['output'])}"
        if result.get("replay_excluded", 0):
            detail += f"\n回放已排除 {result['replay_excluded']} 张未确认截图。"
        self.status.setText(f"{labels.get(result['status'], result['status'])} · 接受 {result.get('accepted', 0)} 帧\n"
                            f"累计位移：{result.get('final_scroll_offset_px', 0)} px\n"
                            f"{detail}")
        logger.info("gui_capture_result status=%s reason=%s report=%s", result['status'], result.get('reason'), self.report_path)
        self.restore_controls()
        if self.report_path and (Path(self.report_path).parent/'messages.sqlite3').is_file():
            self.refresh_sessions()
            self.show_records(Path(self.report_path).parent)

    def worker_finished(self) -> None:
        self.pick_button.setEnabled(self.stage == "idle")

    def restore_controls(self) -> None:
        self.backend.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.pick_button.setEnabled(self.worker is None or not self.worker.isRunning())
        self.show()

    def cancel(self) -> None:
        self.countdown_timer.stop()
        if self.overlay is not None:
            self.overlay.finish()
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.status.setText("正在停止采集并生成回放报告……")
        else:
            self.selection_cancelled()

    def fail(self, message: str) -> None:
        logger.error("gui_failure %s", message)
        self.countdown_timer.stop()
        self.stage = "idle"
        self.status.setText(f"未开始或已停止：{message}")
        self.restore_controls()

    def open_report(self) -> None:
        if self.report_path:
            if (Path(self.report_path).parent/'messages.sqlite3').is_file():
                self.show_records(Path(self.report_path).parent)
            else:QDesktopServices.openUrl(QUrl.fromLocalFile(self.report_path))

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.cancel()
            event.accept()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        if self.records_panel.busy():
            self.records_panel.feedback.setText("正在读取或导出记录，请完成后再关闭。")
            event.ignore()
            return
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.status.setText("正在停止采集，请稍候再关闭。")
            event.ignore()
        else:
            if self.overlay is not None:
                self.overlay.finish()
            self.countdown_timer.stop()
            QApplication.instance().quit()
            event.accept()


def default_output_root() -> Path:
    project = Path(__file__).resolve().parents[2]
    return project / "runs" if (project / "pyproject.toml").exists() else Path.home() / "WeLog" / "runs"


def launch_gui(output_root: Path | None = None, *, auto_pick: bool = True, dwell: float = 1.5, countdown: int = 3, max_frames: int = 0) -> int:
    from .windows import WindowsDesktop
    from .diagnostics import configure_logging
    configure_logging()
    logger.info("gui_start auto_pick=%s dwell=%s countdown=%s max_frames=%s", auto_pick, dwell, countdown, max_frames)
    HoverLock(dwell)
    if countdown < 1 or not 0 <= max_frames <= 10000:
        raise ValueError("countdown must be positive and max_frames must be between 0 and 10000")
    desktop = WindowsDesktop()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)
    configure_font(app)
    app.setApplicationName("WeLog")
    window = CaptureWindow(desktop, output_root if output_root is not None else default_output_root(),
                           dwell_seconds=dwell, countdown=countdown, max_frames=max_frames)
    window.show()
    if auto_pick:
        QTimer.singleShot(350, window.begin_pick)
    code = app.exec()
    logger.info("gui_exit code=%s", code)
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WeLog 指向窗口采集")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--no-auto-pick", action="store_true")
    parser.add_argument("--dwell", type=float, default=1.5)
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=0, help="0：完整采集；正数：限量采集")
    args = parser.parse_args(argv)
    return launch_gui(args.output_root, auto_pick=not args.no_auto_pick, dwell=args.dwell,
                      countdown=args.countdown, max_frames=args.max_frames)


if __name__ == "__main__":
    raise SystemExit(main())
