"""Render only our widgets offscreen with fake windows and synthetic content."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

from welog.demo import make_scene
from welog.gui import CaptureWindow, PickOverlay, RegionDialog, configure_font
from welog.models import Rect
from welog.picker import HoverState, WindowTarget


def forbidden(*args):
    raise AssertionError("Offscreen preview must never query or control desktop windows")


def main():
    output = Path("runs/gui-preview")
    output.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    configure_font(app)
    desktop = SimpleNamespace(activate=forbidden, describe_window=forbidden,
                              window_at_cursor=forbidden, escape_pressed=forbidden)
    window = CaptureWindow(desktop, output / "unused")
    window.show()
    app.processEvents()
    window.grab().save(str(output / "main.png"))
    window.hide()
    target = WindowTarget(123, 456, "微信", Rect(60, 140, 640, 520), "weixin.exe")
    overlay = PickOverlay(desktop, app.primaryScreen())
    overlay.state = HoverState(target, 0.6, False)
    overlay.show()
    app.processEvents()
    overlay.grab().save(str(output / "picker.png"))
    overlay.hide()
    dialog = RegionDialog(target, make_scene(720, 640))
    dialog.show()
    app.processEvents()
    dialog.grab().save(str(output / "calibration.png"))
    dialog.hide()
    print(output.resolve())


if __name__ == "__main__":
    main()
