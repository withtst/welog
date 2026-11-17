"""Small Win32 adapter. Imports remain usable on non-Windows replay hosts."""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from typing import Any

import numpy as np

from .imageio import gray
from .models import Rect
from .picker import WindowTarget


class DesktopError(RuntimeError):
    pass

class BackgroundTargetUnavailable(DesktopError):pass
class BackgroundTargetChanged(DesktopError):pass
class BackgroundScrollRejected(DesktopError):pass


class WindowsDesktop:
    def __init__(self) -> None:
        if sys.platform != "win32":
            raise DesktopError("Live capture requires Windows; demo and replay are cross-platform")
        self.api = ctypes.WinDLL("user32", use_last_error=True)
        api = self.api
        api.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        api.SetProcessDpiAwarenessContext.restype = wintypes.BOOL
        api.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        # The thread context also handles hosts that already initialized a different process mode.
        api.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        api.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
        if not api.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4)):
            raise DesktopError("Cannot enable physical per-monitor DPI coordinates")
        api.GetForegroundWindow.restype = wintypes.HWND
        api.WindowFromPoint.argtypes = [wintypes.POINT]
        api.WindowFromPoint.restype = wintypes.HWND
        api.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        api.GetAncestor.restype = wintypes.HWND
        api.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        api.GetClientRect.restype = wintypes.BOOL
        api.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        api.ClientToScreen.restype = wintypes.BOOL
        api.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        api.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        api.GetWindowTextLengthW.restype = ctypes.c_int
        for name in ("IsWindow", "IsWindowVisible", "IsIconic", "SetForegroundWindow"):
            getattr(api, name).argtypes = [wintypes.HWND]
            getattr(api, name).restype = wintypes.BOOL
        api.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        api.GetAsyncKeyState.argtypes = [ctypes.c_int]
        api.GetAsyncKeyState.restype = ctypes.c_short
        api.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
        api.GetCursorPos.restype = wintypes.BOOL
        api.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        api.GetWindowThreadProcessId.restype = wintypes.DWORD
        api.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        api.GetClassNameW.restype = ctypes.c_int
        api.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
        api.SetCursorPos.restype = wintypes.BOOL
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                         ctypes.POINTER(wintypes.DWORD)]
        self.kernel.QueryFullProcessImageNameW.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL

    def escape_pressed(self) -> bool:
        return bool(self.api.GetAsyncKeyState(0x1B) & 0x8001)

    def describe_window(self, hwnd: int) -> WindowTarget | None:
        if not hwnd or not self.api.IsWindow(hwnd) or not self.api.IsWindowVisible(hwnd) or self.api.IsIconic(hwnd):
            return None
        pid = wintypes.DWORD()
        if not self.api.GetWindowThreadProcessId(hwnd, ctypes.byref(pid)):
            return None
        title = self.title(hwnd)
        classname = ctypes.create_unicode_buffer(256)
        self.api.GetClassNameW(hwnd, classname, len(classname))
        if not title or classname.value in {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"}:
            return None
        try:
            client = self.client_rect(hwnd)
        except (DesktopError, ValueError):
            return None
        if client.width < 160 or client.height < 120:
            return None
        executable = ""
        handle = self.kernel.OpenProcess(0x1000, False, pid.value)  # QUERY_LIMITED_INFORMATION
        if handle:
            try:
                length = wintypes.DWORD(32768)
                path = ctypes.create_unicode_buffer(length.value)
                if self.kernel.QueryFullProcessImageNameW(handle, 0, path, ctypes.byref(length)):
                    executable = os.path.basename(path.value).casefold()
            finally:
                self.kernel.CloseHandle(handle)
        return WindowTarget(int(hwnd), pid.value, title, client, executable)

    def window_at_cursor(self, allowed_executables: set[str] | None = None) -> WindowTarget | None:
        point = wintypes.POINT()
        if not self.api.GetCursorPos(ctypes.byref(point)):
            raise DesktopError("Cannot read pointer position")

        def eligible(hwnd: int) -> WindowTarget | None:
            target = self.describe_window(hwnd)
            if target is None or target.pid == os.getpid():
                return None
            if not (target.client.x <= point.x < target.client.right and target.client.y <= point.y < target.client.bottom):
                return None
            if allowed_executables is not None and target.executable not in allowed_executables:
                return None
            return target

        hit = self.api.WindowFromPoint(point)
        root = self.api.GetAncestor(hit, 2) if hit else None
        if root:
            own_pid = wintypes.DWORD()
            self.api.GetWindowThreadProcessId(root, ctypes.byref(own_pid))
            if own_pid.value != os.getpid():
                return eligible(root)
        # A topmost selector belongs to us. Look behind only our own overlay, in z order.
        candidates: list[WindowTarget] = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        @callback_type
        def visit(hwnd: int, _: int) -> bool:
            target = self.describe_window(hwnd)
            if target is None or target.pid == os.getpid():
                return True
            if not (target.client.x <= point.x < target.client.right and target.client.y <= point.y < target.client.bottom):
                return True
            # An unrelated app obscuring WeChat must prevent locking the WeChat behind it.
            if allowed_executables is None or target.executable in allowed_executables:
                candidates.append(target)
            return False

        self.api.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
        self.api.EnumWindows.restype = wintypes.BOOL
        self.api.EnumWindows(visit, 0)
        return candidates[0] if candidates else None

    def title(self, hwnd: int) -> str:
        text = ctypes.create_unicode_buffer(self.api.GetWindowTextLengthW(hwnd) + 1)
        self.api.GetWindowTextW(hwnd, text, len(text))
        return text.value

    def windows(self) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        @callback_type
        def visit(hwnd: int, _: int) -> bool:
            title = self.title(hwnd)
            if title and self.api.IsWindowVisible(hwnd):
                values.append({"hwnd": int(hwnd), "title": title, "minimized": bool(self.api.IsIconic(hwnd))})
            return True

        self.api.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
        self.api.EnumWindows.restype = wintypes.BOOL
        if not self.api.EnumWindows(visit, 0):
            raise DesktopError("Cannot enumerate windows")
        return values

    def resolve(self, hwnd: int | None, title: str | None) -> int:
        if hwnd is not None:
            if not self.api.IsWindow(hwnd):
                raise DesktopError(f"Invalid window handle: {hwnd}")
            return hwnd
        matches = [window for window in self.windows() if title and title.casefold() in window["title"].casefold()]
        if len(matches) != 1:
            raise DesktopError(f"Title must match exactly one window (matched {len(matches)}); use --hwnd")
        return matches[0]["hwnd"]

    def client_rect(self, hwnd: int) -> Rect:
        rect, point = wintypes.RECT(), wintypes.POINT(0, 0)
        if not self.api.GetClientRect(hwnd, ctypes.byref(rect)) or not self.api.ClientToScreen(hwnd, ctypes.byref(point)):
            raise DesktopError("Cannot query window geometry")
        return Rect(point.x, point.y, rect.right, rect.bottom)

    def activate(self, hwnd: int) -> None:
        if self.api.IsIconic(hwnd):
            self.api.ShowWindow(hwnd, 9)  # Restore only minimized windows; preserve maximized geometry.
        self.api.SetForegroundWindow(hwnd)

    def check(self, hwnd: int, expected: Rect, roi: Rect, title: str) -> None:
        if self.escape_pressed():
            raise KeyboardInterrupt
        if not self.api.IsWindow(hwnd) or self.api.IsIconic(hwnd):
            raise DesktopError("Target window is missing or minimized")
        if self.title(hwnd) != title or self.client_rect(hwnd) != expected:
            raise DesktopError("Window title, position, size or DPI changed; recalibration is required")
        foreground = self.api.GetAncestor(self.api.GetForegroundWindow(), 2)
        root = self.api.GetAncestor(hwnd, 2)
        if foreground != root:
            raise DesktopError("Target is no longer the foreground window")
        for fx in (0.08, 0.5, 0.92):
            for fy in (0.08, 0.5, 0.92):
                point = wintypes.POINT(expected.x + roi.x + int(roi.width * fx),
                                       expected.y + roi.y + int(roi.height * fy))
                covering = self.api.GetAncestor(self.api.WindowFromPoint(point), 2)
                if covering != root:
                    raise DesktopError("Capture region is occluded by another window")

    def scroll(self, hwnd: int, client: Rect, roi: Rect, units: int, title: str) -> None:
        self.check(hwnd, client, roi, title)
        x = client.x + roi.x + roi.width // 2
        y = client.y + roi.y + roi.height // 2
        if not self.api.SetCursorPos(x, y):
            raise DesktopError("Cannot position the scroll pointer")

        class MouseInput(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

        class InputUnion(ctypes.Union):
            _fields_ = [("mi", MouseInput)]

        class Input(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("data", InputUnion)]

        event = Input(type=0, data=InputUnion(mi=MouseInput(
            0, 0, (-units) & 0xFFFFFFFF, 0x0800, 0, 0)))  # Positive API units mean scroll DOWN.
        self.api.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int]
        self.api.SendInput.restype = wintypes.UINT
        if self.api.SendInput(1, ctypes.byref(event), ctypes.sizeof(Input)) != 1:
            raise DesktopError("Mouse wheel input was rejected by Windows")

    def check_background(self, target: WindowTarget) -> Rect:
        """Identity/layout checks only: no focus, cursor, hit-testing or global Escape."""
        if not self.api.IsWindow(target.hwnd) or not self.api.IsWindowVisible(target.hwnd) or self.api.IsIconic(target.hwnd):
            raise BackgroundTargetUnavailable('Background target is missing, hidden or minimized')
        pid=wintypes.DWORD()
        self.api.GetWindowThreadProcessId(target.hwnd,ctypes.byref(pid))
        if pid.value!=target.pid or self.title(target.hwnd)!=target.title:
            raise BackgroundTargetChanged('Background window identity changed')
        client=self.client_rect(target.hwnd)
        if client!=target.client:raise BackgroundTargetChanged('Background window geometry or DPI changed; recalibrate')
        return client

    def capture_bounds(self, hwnd: int) -> Rect:
        api=ctypes.WinDLL('dwmapi',use_last_error=True)
        api.DwmGetWindowAttribute.argtypes=[wintypes.HWND,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD]
        api.DwmGetWindowAttribute.restype=ctypes.c_long
        rect=wintypes.RECT()
        if api.DwmGetWindowAttribute(hwnd,9,ctypes.byref(rect),ctypes.sizeof(rect))!=0:
            raise DesktopError('Cannot query captured window bounds')
        return Rect(rect.left,rect.top,rect.right-rect.left,rect.bottom-rect.top)

    def background_scroll_target(self, target: WindowTarget, roi: Rect) -> int:
        """Descend only the selected window's child tree, independently of occlusion."""
        self.check_background(target)
        api=self.api
        api.ChildWindowFromPointEx.argtypes=[wintypes.HWND,wintypes.POINT,wintypes.UINT]
        api.ChildWindowFromPointEx.restype=wintypes.HWND
        api.ScreenToClient.argtypes=[wintypes.HWND,ctypes.POINTER(wintypes.POINT)]
        api.ScreenToClient.restype=wintypes.BOOL
        x=target.client.x+roi.x+roi.width//2;y=target.client.y+roi.y+roi.height//2
        parent=target.hwnd
        for _ in range(32):
            point=wintypes.POINT(x,y)
            if not api.ScreenToClient(parent,ctypes.byref(point)):raise DesktopError('Cannot locate background chat control')
            child=api.ChildWindowFromPointEx(parent,point,7)
            if not child or int(child)==parent:return parent
            pid=wintypes.DWORD();api.GetWindowThreadProcessId(child,ctypes.byref(pid))
            if pid.value!=target.pid:raise DesktopError('Background child belongs to another process')
            parent=int(child)
        raise DesktopError('Background child hierarchy is too deep')

    def post_background_scroll(self, target: WindowTarget, child: int, roi: Rect, units: int) -> None:
        self.check_background(target)
        api=self.api
        api.IsChild.argtypes=[wintypes.HWND,wintypes.HWND];api.IsChild.restype=wintypes.BOOL
        if not api.IsWindow(child) or not api.IsWindowVisible(child) or (child!=target.hwnd and not api.IsChild(target.hwnd,child)):
            raise BackgroundTargetChanged('Background chat control was replaced or hidden')
        pid=wintypes.DWORD();api.GetWindowThreadProcessId(child,ctypes.byref(pid))
        if pid.value!=target.pid:raise BackgroundTargetChanged('Background chat control identity changed')
        api.PostMessageW.argtypes=[wintypes.HWND,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM]
        api.PostMessageW.restype=wintypes.BOOL
        x=target.client.x+roi.x+roi.width//2;y=target.client.y+roi.y+roi.height//2
        if not -32768<=x<=32767 or not -32768<=y<=32767:raise DesktopError('Wheel coordinates exceed Win32 range')
        # Public units are positive DOWN; Win32 wheel delta is positive UP.
        wheel=((-units)&0xffff)<<16;point=(x&0xffff)|((y&0xffff)<<16)
        if not api.PostMessageW(child,0x020A,wheel,point):raise BackgroundScrollRejected('Background scroll message rejected')

    def refresh_background(self, target: WindowTarget) -> None:
        self.check_background(target)
        self.api.RedrawWindow.argtypes=[wintypes.HWND,ctypes.c_void_p,ctypes.c_void_p,wintypes.UINT]
        self.api.RedrawWindow.restype=wintypes.BOOL
        # Asynchronous invalidation of this window and its children only; no activation.
        if not self.api.RedrawWindow(target.hwnd,None,None,0x0001|0x0080):
            raise DesktopError('Background window repaint request failed')


class DXcamSource:
    """MVP: primary monitor only; ROI and optional title guard are client-relative."""

    def __init__(self, desktop: WindowsDesktop, hwnd: int, roi: Rect, guard: Rect | None = None):
        try:
            import dxcam
        except ImportError as error:
            raise DesktopError('Install the live backend with: pip install -e ".[windows]"') from error
        self.desktop, self.hwnd, self.roi, self.guard = desktop, hwnd, roi, guard
        self.client = desktop.client_rect(hwnd)
        self.window_title = desktop.title(hwnd)
        for box in (roi, guard):
            if box is not None and (box.x < 0 or box.y < 0 or box.right > self.client.width or box.bottom > self.client.height):
                raise DesktopError("ROI must be inside the window client area")
        self.camera = dxcam.create(output_color="BGR")
        self.guard_reference: np.ndarray | None = None
        self.last_frame: np.ndarray | None = None
        self.monitor_size = (self.camera.width, self.camera.height)
        # DXcam's default output is the primary display, whose desktop origin is (0, 0).
        if (self.client.x < 0 or self.client.y < 0 or self.client.right > self.camera.width
                or self.client.bottom > self.camera.height):
            self.close()
            raise DesktopError("Stage one requires the entire target window on the primary monitor")

    def _grab(self) -> np.ndarray | None:
        return self.camera.grab(new_frame_only=False)

    def _crop(self, frame: np.ndarray, box: Rect) -> np.ndarray:
        x, y = self.client.x + box.x, self.client.y + box.y
        return frame[y:y + box.height, x:x + box.width].copy()

    def read(self) -> np.ndarray | None:
        self.desktop.check(self.hwnd, self.client, self.roi, self.window_title)
        frame = self._grab()
        if frame is None:
            return None  # No sample does not count as evidence of stability or zero movement.
        if (frame.shape[1], frame.shape[0]) != self.monitor_size:
            raise DesktopError("Display resolution changed")
        if self.guard is not None:
            current = gray(self._crop(frame, self.guard))
            if self.guard_reference is None:
                self.guard_reference = current.copy()
            else:
                changed = np.abs(current.astype(np.int16) - self.guard_reference.astype(np.int16)) > 25
                if float(changed.mean()) > 0.025:
                    raise DesktopError("Conversation guard changed; the selected chat may have changed")
        self.last_frame = self._crop(frame, self.roi)
        return self.last_frame

    def scroll(self, units: int) -> None:
        self.desktop.scroll(self.hwnd, self.client, self.roi, units, self.window_title)

    def close(self) -> None:
        try:
            self.camera.release()
        finally:
            self.last_frame = None
            self.guard_reference = None

    def interrupted(self) -> bool:
        return self.desktop.escape_pressed()
