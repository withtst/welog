"""Background capture tests use fake Win32/WGC adapters; never inspect user windows."""
from types import SimpleNamespace
import ctypes
import numpy as np
import pytest

from welog.background import BackgroundSource,WgcStream,BackgroundFrameStale,BackgroundCaptureUnavailable,source_factory
from welog.models import Rect,CaptureConfig
from welog.picker import WindowTarget
from welog.windows import WindowsDesktop,DesktopError


class Function:
    def __init__(self,fn):self.fn=fn
    def __call__(self,*args):return self.fn(*args)


def forbidden(*args,**kwargs):raise AssertionError('Focus, global input and desktop hit-testing are forbidden')


def desktop():
    target=WindowTarget(10,123,'fixture',Rect(100,120,320,240),'fixture.exe')
    d=WindowsDesktop.__new__(WindowsDesktop)
    def pid(hwnd,out):out._obj.value=123;return 1
    messages=[]
    d.api=SimpleNamespace(IsWindow=lambda _:True,IsWindowVisible=lambda _:True,IsIconic=lambda _:False,
        GetWindowThreadProcessId=pid,GetForegroundWindow=forbidden,WindowFromPoint=forbidden,
        SetCursorPos=forbidden,SetForegroundWindow=forbidden,SendInput=forbidden,
        IsChild=Function(lambda root,child:child==11),
        PostMessageW=Function(lambda *args:messages.append(args) or True),
        RedrawWindow=Function(lambda *args:True))
    d.title=lambda _:'fixture';d.client_rect=lambda _:target.client
    d.describe_window=lambda _:target
    d.capture_bounds=lambda _:Rect(92,100,336,268)
    d.escape_pressed=forbidden
    return d,target,messages


def test_background_validation_does_not_require_foreground_or_uncovered_window():
    d,target,_=desktop()
    assert d.check_background(target)==target.client
    d.refresh_background(target)


@pytest.mark.parametrize('change',['minimized','hidden','title','geometry','pid'])
def test_background_validation_rejects_changed_target(change):
    d,target,_=desktop()
    if change=='minimized':d.api.IsIconic=lambda _:True
    if change=='hidden':d.api.IsWindowVisible=lambda _:False
    if change=='title':d.title=lambda _:'different'
    if change=='geometry':d.client_rect=lambda _:Rect(100,120,321,240)
    if change=='pid':
        def pid(hwnd,out):out._obj.value=999;return 1
        d.api.GetWindowThreadProcessId=pid
    with pytest.raises(DesktopError):d.check_background(target)


@pytest.mark.parametrize('units',[120,-120])
def test_targeted_wheel_is_delivered_only_to_bound_child_with_correct_sign(units):
    d,target,messages=desktop()
    d.post_background_scroll(target,11,Rect(0,0,300,200),units)
    child,msg,wparam,lparam=messages[0]
    delta=ctypes.c_short(wparam>>16).value
    assert child==11 and msg==0x020A and delta==-units
    assert (lparam&0xffff)==250 and (lparam>>16)==220


def test_replaced_child_and_rejected_message_never_fall_back_to_input():
    d,target,_=desktop()
    d.api.IsChild=Function(lambda *args:False)
    with pytest.raises(DesktopError):d.post_background_scroll(target,99,Rect(0,0,200,200),120)
    d.api.PostMessageW=Function(lambda *args:False)
    with pytest.raises(DesktopError):d.post_background_scroll(target,10,Rect(0,0,200,200),120)


def test_negative_monitor_coordinates_are_encoded_as_signed_wheel_coordinates():
    d,target,messages=desktop()
    target=WindowTarget(10,123,'fixture',Rect(-600,-300,320,240),'fixture.exe')
    d.client_rect=lambda _:target.client
    d.post_background_scroll(target,10,Rect(0,0,200,200),120)
    point=messages[0][3]
    assert ctypes.c_short(point&0xffff).value==-500
    assert ctypes.c_short((point>>16)&0xffff).value==-200


def test_child_selection_walks_the_locked_tree_not_the_screen():
    d,target,_=desktop();calls=[]
    def screen(hwnd,point):calls.append(hwnd);point._obj.x=20;point._obj.y=20;return True
    d.api.ScreenToClient=Function(screen)
    d.api.ChildWindowFromPointEx=Function(lambda parent,point,flags:11 if parent==10 else 11)
    assert d.background_scroll_target(target,Rect(0,0,200,200))==11 and calls==[10,11]


class Capture:
    def __init__(self,**kwargs):self.kwargs=kwargs;self.handlers={};self.stops=0
    def event(self,handler):self.handlers[handler.__name__]=handler;return handler
    def start_free_threaded(self):return SimpleNamespace(stop=self.stop)
    def stop(self):self.stops+=1
    def emit(self,buffer):self.handlers['on_frame_arrived'](SimpleNamespace(frame_buffer=buffer),SimpleNamespace(stop=lambda:None))


def make_source(guard=None):
    d,target,messages=desktop();d.background_scroll_target=lambda *args:11
    clock=[0.];captures=[]
    def factory(**kwargs):
        value=Capture(**kwargs);captures.append(value);return value
    def stream(hwnd,extract):return WgcStream(hwnd,extract,factory=factory,clock=lambda:clock[0])
    source=BackgroundSource(d,10,Rect(10,20,200,160),guard,stream_factory=stream)
    return source,captures[0],clock,messages


def test_wgc_binds_exact_hwnd_and_maps_dwm_client_offsets_without_capturing_cursor():
    source,capture,clock,messages=make_source()
    assert capture.kwargs['window_hwnd']==10 and capture.kwargs['cursor_capture'] is False
    assert 'window_name' not in capture.kwargs and 'monitor_index' not in capture.kwargs
    buffer=np.full((268,336,4),77,np.uint8)
    buffer[40:200,18:218,:3]=123
    capture.emit(buffer)
    buffer[:]=0
    first=source.read();assert first.shape==(160,200,3) and (first==123).all()
    first[:]=4;assert (source.read()==123).all()
    source.scroll(-120);assert messages[0][0]==11
    assert source.interrupted() is False
    source.close();source.close();assert capture.stops==1


def test_client_only_capture_and_frame_storage_are_bounded():
    source,capture,clock,_=make_source()
    for i in range(100):capture.emit(np.full((240,320,4),100+i%10,np.uint8))
    assert source.stream.latest[0].nbytes==200*160*3
    assert source.stream.latest[0].base is None
    assert source.read().shape==(160,200,3)
    source.close()


def test_guard_changes_pause_and_no_external_escape_is_read():
    source,capture,clock,_=make_source(Rect(0,0,100,10))
    buffer=np.full((240,320,4),120,np.uint8);capture.emit(buffer);source.read()
    buffer[:10,:100,:3]=250;capture.emit(buffer)
    with pytest.raises(DesktopError,match='guard'):source.read()
    source.close()


def test_stale_closed_malformed_or_black_surfaces_fail_instead_of_desktop_fallback():
    source,capture,clock,_=make_source()
    capture.emit(np.full((240,320,4),100,np.uint8));clock[0]=3
    with pytest.raises(BackgroundFrameStale):source.read()
    source.close()
    for bad in [np.zeros((240,320,4),np.uint8),np.zeros((10,10,4),np.uint8)]:
        source,capture,clock,_=make_source();capture.emit(bad)
        with pytest.raises(BackgroundCaptureUnavailable):source.read()
        source.close()
    source,capture,clock,_=make_source();capture.handlers['on_closed']()
    with pytest.raises(BackgroundCaptureUnavailable):source.read()
    source.close()


def test_stream_creation_failure_has_no_capture_fallback():
    def factory(**kwargs):raise OSError('unsupported')
    with pytest.raises(BackgroundCaptureUnavailable):WgcStream(10,lambda frame:frame,factory=factory)
    with pytest.raises(ValueError):source_factory('unknown')
    with pytest.raises(ValueError):CaptureConfig(capture_backend='unknown')
