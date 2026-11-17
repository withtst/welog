"""Fake Win32 functions only: these tests never query or act on the real desktop."""

import os
from types import SimpleNamespace

from welog.models import Rect
from welog.picker import WindowTarget
from welog.windows import WindowsDesktop


class Function:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


def desktop_with_windows(hit, targets, cursor=(50, 50)):
    desktop = WindowsDesktop.__new__(WindowsDesktop)

    def cursor_pos(point):
        point._obj.x, point._obj.y = cursor
        return True

    def process_id(hwnd, pid):
        pid._obj.value = targets[hwnd].pid
        return 1

    def enumerate_windows(callback, _):
        for hwnd in targets:
            if not callback(hwnd, 0):
                return False
        return True

    desktop.api = SimpleNamespace(
        GetCursorPos=Function(cursor_pos), WindowFromPoint=Function(lambda _: hit),
        GetAncestor=Function(lambda hwnd, _: hwnd),
        GetWindowThreadProcessId=Function(process_id), EnumWindows=Function(enumerate_windows),
    )
    desktop.describe_window = targets.get
    return desktop


def test_pointer_selects_target_under_our_overlay_but_not_other_apps_behind_it():
    own = WindowTarget(1, os.getpid(), "selector", Rect(0, 0, 800, 600), "python.exe")
    wechat = WindowTarget(3, os.getpid() + 2, "微信", Rect(0, 0, 800, 600), "weixin.exe")
    desktop = desktop_with_windows(1, {1: own, 3: wechat})
    assert desktop.window_at_cursor({"weixin.exe"}) == wechat
    unrelated = WindowTarget(2, os.getpid() + 1, "other app", Rect(0, 0, 800, 600), "other.exe")
    desktop = desktop_with_windows(1, {1: own, 2: unrelated, 3: wechat})
    assert desktop.window_at_cursor({"weixin.exe"}) is None


def test_pointer_outside_client_or_on_unrelated_app_cannot_lock_wechat():
    wechat = WindowTarget(3, os.getpid() + 2, "微信", Rect(100, 100, 800, 600), "weixin.exe")
    desktop = desktop_with_windows(3, {3: wechat})
    assert desktop.window_at_cursor({"weixin.exe"}) is None


def test_activation_preserves_maximized_window_geometry():
    calls = []
    desktop = WindowsDesktop.__new__(WindowsDesktop)
    desktop.api = SimpleNamespace(IsIconic=lambda _: False,
                                  ShowWindow=lambda *args: calls.append(("restore", args)),
                                  SetForegroundWindow=lambda hwnd: calls.append(("focus", hwnd)))
    desktop.activate(100)
    assert calls == [("focus", 100)]
