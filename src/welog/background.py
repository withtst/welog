"""HWND-bound WGC capture and targeted scroll; no desktop screenshot or input fallback."""
from __future__ import annotations
import threading
import time
import numpy as np
from .imageio import gray
from .models import Rect
from .windows import DesktopError, BackgroundTargetChanged, BackgroundTargetUnavailable


class BackgroundFrameStale(DesktopError):pass
class BackgroundCaptureUnavailable(DesktopError):pass


class WgcStream:
    """Own one copied ROI/guard pair; callback buffers never escape the capture thread."""
    def __init__(self,hwnd,extract,*,factory=None,clock=time.monotonic):
        if factory is None:
            try:from windows_capture import WindowsCapture
            except ImportError as error:raise BackgroundCaptureUnavailable('Install the Windows extra: pip install -e ".[windows,gui]"') from error
            factory=WindowsCapture
        self.lock=threading.Lock();self.latest=None;self.error=None;self.closed=False
        self.clock=clock;self.created=clock();self.arrived=0.;self.extract=extract
        self.control=None
        try:
            self.capture=factory(window_hwnd=hwnd,cursor_capture=False,secondary_window=False,
                minimum_update_interval=30,dirty_region=False)
            @self.capture.event
            def on_frame_arrived(frame,control):
                with self.lock:
                    if self.closed:control.stop();return
                try:
                    roi,guard=self.extract(frame.frame_buffer)
                    with self.lock:
                        if not self.closed:self.latest=(roi,guard);self.arrived=self.clock()
                except Exception as error:
                    with self.lock:self.error=type(error).__name__
                    control.stop()
            @self.capture.event
            def on_closed():
                with self.lock:
                    if not self.closed:self.error='capture_item_closed'
            self.control=self.capture.start_free_threaded()
        except Exception as error:
            self.close()
            raise BackgroundCaptureUnavailable('Cannot start HWND-bound WGC capture') from error

    def read(self):
        with self.lock:
            if self.error:raise BackgroundCaptureUnavailable('WGC stream stopped: '+self.error)
            if self.closed:raise BackgroundCaptureUnavailable('WGC stream is closed')
            age=self.clock()-(self.arrived if self.latest is not None else self.created)
            if age>2:raise BackgroundFrameStale('Window stopped producing fresh frames')
            if self.latest is None:return None,age
            return tuple(image.copy() if image is not None else None for image in self.latest),age

    def close(self):
        with self.lock:
            if self.closed:return
            self.closed=True;self.latest=None
        if self.control is not None:
            try:self.control.stop()
            except Exception:pass
            self.control=None


class BackgroundSource:
    def __init__(self,desktop,hwnd,roi:Rect,guard:Rect|None=None,*,stream_factory=None):
        self.desktop=desktop;self.roi=roi;self.guard=guard;self.stream=None
        self.target=desktop.describe_window(hwnd)
        if self.target is None:raise BackgroundCaptureUnavailable('Target must be visible and not minimized')
        self.stream_factory = stream_factory or WgcStream
        self.bounds, self.client = self._capture_mapping()
        for box in (roi,guard):
            if box and (box.x<0 or box.y<0 or box.right>self.client.width or box.bottom>self.client.height):
                raise DesktopError('ROI must be inside the client area')
        self.child=desktop.background_scroll_target(self.target,roi)
        self.guard_reference=None;self.last_refresh=0.
        self.stream=self.stream_factory(hwnd,self._extract)

    def _capture_mapping(self):
        # Capture one coherent client/window mapping. Translation never changes
        # its local crop offset; do not mix independently updated screen origins
        # in the concurrent WGC callback.
        for _ in range(3):
            client=self.desktop.check_background(self.target)
            bounds=self.desktop.capture_bounds(self.target.hwnd)
            if self.desktop.check_background(self.target)==client:
                return bounds, client
        raise BackgroundTargetChanged('Cannot establish a stable client mapping',code='window_mapping_unstable')

    def wait_for_target(self, error, deadline, stopped):
        """Wait for the same target to return and obtain a fresh guarded frame."""
        self._close_stream()
        while time.monotonic() < deadline:
            if stopped():
                raise KeyboardInterrupt
            try:
                if self.stream is None:
                    self.bounds, self.client = self._capture_mapping()
                    self.last_refresh = 0.
                    self.stream = self.stream_factory(self.target.hwnd, self._extract)
                if self.read() is not None:
                    return
            except BackgroundTargetUnavailable as unavailable:
                if not unavailable.temporary:
                    raise
                error = unavailable
                self._close_stream()
            time.sleep(min(.1, max(0., deadline - time.monotonic())))
        raise BackgroundTargetUnavailable(
            'Timed out waiting for the background target',
            code=error.code, details={'wait_timed_out': True},
        ) from error

    def _extract(self,buffer):
        if buffer.ndim!=3 or buffer.shape[2] not in (3,4):raise DesktopError('Invalid WGC frame format')
        h,w=buffer.shape[:2]
        if (w,h)==(self.client.width,self.client.height):x=y=0
        elif (w,h)==(self.bounds.width,self.bounds.height):
            x=self.client.x-self.bounds.x;y=self.client.y-self.bounds.y
        else:raise DesktopError('Captured window dimensions changed')
        if x<0 or y<0 or x+self.client.width>w or y+self.client.height>h:raise DesktopError('Invalid client-to-window mapping')
        def crop(box):return buffer[y+box.y:y+box.bottom,x+box.x:x+box.right,:3].copy()
        return crop(self.roi),crop(self.guard) if self.guard else None

    def read(self):
        self.desktop.check_background(self.target)
        pair,age=self.stream.read()
        if pair is None or age>.5:
            now=time.monotonic()
            if now-self.last_refresh>.1:
                self.desktop.refresh_background(self.target);self.last_refresh=now
            return None
        image,guard=pair
        if not image.any():raise BackgroundCaptureUnavailable('Empty black window surface')
        if guard is not None:
            current=gray(guard)
            if self.guard_reference is None:self.guard_reference=current.copy()
            elif float((np.abs(current.astype(np.int16)-self.guard_reference.astype(np.int16))>25).mean())>.025:
                raise BackgroundTargetChanged('Conversation guard changed',code='conversation_guard_changed')
        return image

    def scroll(self,units):
        self.desktop.post_background_scroll(self.target,self.child,self.roi,units)

    def interrupted(self):return False  # Global Escape belongs to the user's other application.

    def _close_stream(self):
        if self.stream:self.stream.close();self.stream=None

    def close(self):
        self._close_stream()
        self.guard_reference=None


def source_factory(backend):
    if backend=='background':return BackgroundSource
    if backend=='foreground':
        from .windows import DXcamSource
        return DXcamSource
    raise ValueError('Unknown capture backend')
