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
    with pytest.raises(DesktopError,match='guard') as error:source.read()
    assert error.value.code=='conversation_guard_changed'
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

@pytest.mark.parametrize('initial,current',[
    ('微信','微信 (25)'),('微信 (1)','微信 (26)'),('微信 (25)','微信'),
    ('微信','(25)微信'),('微信','微信（99+）'),('WeChat','WeChat [99+]'),
    ('Weixin (1)','Weixin (2)')])
def test_wechat_unread_badges_are_mutable_captions_not_target_identity(initial,current):
    from dataclasses import replace
    d,target,_=desktop();target=replace(target,title=initial,executable='Weixin.exe')
    d.title=lambda _:current
    assert d.check_background(target)==target.client
    d.post_background_scroll(target,11,Rect(0,0,200,200),-120)


@pytest.mark.parametrize('initial,current,executable',[
    ('微信','different','weixin.exe'),('微信','微信 - private conversation','weixin.exe'),
    ('fixture (1)','fixture (2)','weixin.exe'),('微信','微信 (1)','fixture.exe'),
    ('微信','微信 (private text)','weixin.exe')])
def test_other_caption_changes_are_rejected_with_content_free_diagnostics(initial,current,executable):
    from dataclasses import replace
    from welog.windows import BackgroundTargetChanged
    d,target,_=desktop();target=replace(target,title=initial,executable=executable);d.title=lambda _:current
    with pytest.raises(BackgroundTargetChanged) as error:d.check_background(target)
    assert error.value.code=='window_title_changed'
    assert error.value.details=={'expected_title_length':len(initial),'observed_title_length':len(current)}
    assert current not in str(error.value.details)


@pytest.mark.parametrize('change,code',[('pid','window_pid_changed'),('geometry','client_geometry_changed'),
                                       ('child','scroll_control_changed'),('child_pid','scroll_control_pid_changed')])
def test_target_failures_have_distinct_machine_readable_reasons(change,code):
    from welog.windows import BackgroundTargetChanged
    d,target,_=desktop()
    if change=='geometry':d.client_rect=lambda _:Rect(100,120,321,240)
    if change=='child':d.api.IsChild=Function(lambda *args:False)
    if change in {'pid','child_pid'}:
        def pid(hwnd,out):out._obj.value=999 if change=='pid' or hwnd==11 else 123;return 1
        d.api.GetWindowThreadProcessId=pid
    with pytest.raises(BackgroundTargetChanged) as error:d.post_background_scroll(target,11,Rect(0,0,200,200),120)
    assert error.value.code==code


@pytest.mark.parametrize('direction',['up','down'])
def test_full_background_capture_continues_while_unread_caption_changes(tmp_path,direction):
    from dataclasses import replace
    from welog.demo import make_scene
    from welog.controller import run_capture
    from welog.archive import read_jsonl
    d,target,messages=desktop();target=replace(target,title='微信',executable='weixin.exe',client=Rect(100,120,640,720))
    d.describe_window=lambda _:target;d.client_rect=lambda _:target.client
    d.capture_bounds=lambda _:target.client;d.background_scroll_target=lambda *a:11
    state={'caption':'微信'};d.title=lambda _:state['caption'];captures=[]
    def factory(**kwargs):c=Capture(**kwargs);captures.append(c);return c
    def stream(hwnd,extract):return WgcStream(hwnd,extract,factory=factory)
    class Source(BackgroundSource):
        position=300 if direction=='up' else 0
        reads=0
        def read(self):
            self.reads+=1;state['caption']=f'微信 ({self.reads})'
            captures[0].emit(world[self.position:self.position+720])
            return super().read()
        def scroll(self,units):
            super().scroll(units);self.position+=50 if units>0 else -50
    world=make_scene(1500,640,seed=62);source=Source(d,10,Rect(0,0,640,720),stream_factory=stream)
    result=run_capture(source,tmp_path/'capture',config=CaptureConfig(capture_backend='background',direction=direction,
        max_frames=7,max_scroll_events=1,sample_interval_s=.001,stable_samples=2,settle_delay_s=0,scroll_event_interval_s=0))
    assert result['accepted']==7 and result['rejected']==0 and result['reason']=='max_frames'
    assert result['final_scroll_offset_px']==(-300 if direction=='up' else 300)
    assert all(msg[0]==11 for msg in messages) and len(messages)==6 and captures[0].stops==1
    assert result['image_memory_remaining_bytes']==0 and not list(tmp_path.rglob('*.png'))
    rows=read_jsonl(tmp_path/'capture/messages.jsonl')
    assert [r['global_top_px'] for r in rows]==sorted(r['global_top_px'] for r in rows)


@pytest.mark.parametrize('code',['window_title_changed','window_pid_changed','client_geometry_changed',
                               'scroll_control_changed','scroll_control_pid_changed','conversation_guard_changed'])
@pytest.mark.parametrize('debug',[False,True])
def test_controller_persists_target_failure_details_without_title_or_false_height(tmp_path,caplog,code,debug):
    from welog.controller import run_capture
    from welog.demo import make_scene
    from welog.archive import read_jsonl
    from welog.windows import BackgroundTargetChanged
    class Source:
        offset=0;calls=0;closed=False
        def read(self):return world[self.offset:self.offset+720].copy()
        def scroll(self,units):
            self.calls+=1
            if self.calls==2:raise BackgroundTargetChanged('private vendor caption',code=code,
                details={'expected_title_length':2,'observed_title_length':15})
            self.offset+=50
        def close(self):self.closed=True
    world=make_scene(1500,640,seed=62);source=Source()
    result=run_capture(source,tmp_path/'capture',config=CaptureConfig(capture_backend='background',debug_save_images=debug,max_frames=5,max_scroll_events=1,
        stable_samples=2,sample_interval_s=.001,settle_delay_s=0,scroll_event_interval_s=0))
    assert result['reason']=='BackgroundTargetChanged' and result['source_error']['code']==code
    assert result['accepted']==2 and result['final_scroll_offset_px']==50 and source.closed
    events=read_jsonl(tmp_path/'capture/diagnostics.jsonl')
    event=next(e for e in events if e.get('kind')=='source_error')
    assert not event['coordinate_eligible'] and event['images']==[]
    assert event['source_error']['image_evidence']==('last_confirmed_frame' if debug else 'not_available')
    assert f'background_target_changed code={code}' in caplog.text and 'private vendor caption' not in caplog.text
    assert result['image_memory_remaining_bytes']==0
    if not debug:assert not list(tmp_path.rglob('*.png'))

@pytest.mark.parametrize('initial',[True,False])
def test_read_time_target_failure_retains_only_existing_confirmed_evidence(tmp_path,initial):
    from welog.controller import run_capture
    from welog.demo import make_scene
    from welog.archive import read_jsonl
    from welog.windows import BackgroundTargetChanged
    class Source:
        scrolled=False
        def read(self):
            if initial or self.scrolled:raise BackgroundTargetChanged('no raw state',code='window_pid_changed')
            return world.copy()
        def scroll(self,units):self.scrolled=True
        def close(self):pass
    world=make_scene(720,640,seed=62)
    result=run_capture(Source(),tmp_path/'capture',config=CaptureConfig(capture_backend='background',debug_save_images=True,
        max_frames=3,stable_samples=2,sample_interval_s=.001,settle_delay_s=0,scroll_event_interval_s=0))
    assert result['accepted']==(0 if initial else 1) and result['final_scroll_offset_px']==0
    assert result['source_error']['image_evidence']==('not_available' if initial else 'last_confirmed_frame')
    assert result['diagnostic_image_save_error'] is None and result['image_memory_remaining_bytes']==0
    events=read_jsonl(tmp_path/'capture/diagnostics.jsonl')
    images=[i for e in events for i in e.get('images',[])]
    if initial:assert not images and not list(tmp_path.rglob('*.png'))
    else:
        frames=[i for i in images if i['role'] in {'reference','current'}]
        assert frames and all(i['frame_id']=='000000' for i in frames)
        assert [f['frame_id'] for f in read_jsonl(tmp_path/'capture/frames.jsonl')]==['000000']

@pytest.mark.parametrize('origin',[(-600,-300),(700,80),(100,120)])
def test_background_translation_uses_current_screen_coordinates_without_changing_binding(origin):
    from dataclasses import replace
    d,target,messages=desktop();target=replace(target,dpi=96)
    d.api.GetDpiForWindow=Function(lambda _:96)
    current=Rect(*origin,320,240);d.client_rect=lambda _:current
    assert d.check_background(target)==current
    d.post_background_scroll(target,11,Rect(0,0,200,200),-120)
    point=messages[0][3]
    assert (ctypes.c_short(point&0xffff).value,ctypes.c_short(point>>16).value)==(current.x+100,current.y+100)
    assert target.client==Rect(100,120,320,240)


@pytest.mark.parametrize('dpi',[120,144,0])
def test_dpi_changes_remain_fatal_even_when_client_pixel_dimensions_match(dpi):
    from dataclasses import replace
    from welog.windows import BackgroundTargetChanged
    d,target,messages=desktop();target=replace(target,dpi=96)
    d.api.GetDpiForWindow=Function(lambda _:dpi)
    d.client_rect=lambda _:Rect(-600,-300,320,240)
    with pytest.raises(BackgroundTargetChanged) as error:d.post_background_scroll(target,11,Rect(0,0,200,200),120)
    assert error.value.code=='client_dpi_changed' and not messages
    assert error.value.details=={'expected_dpi':96,'observed_dpi':dpi or None}


def test_scroll_rechecks_position_after_child_validation_before_posting():
    d,target,messages=desktop();current=[target.client]
    d.client_rect=lambda _:current[0]
    def child(*args):current[0]=Rect(-800,-500,320,240);return True
    d.api.IsChild=Function(child)
    d.post_background_scroll(target,11,Rect(0,0,200,200),120)
    point=messages[0][3]
    assert (ctypes.c_short(point&0xffff).value,ctypes.c_short(point>>16).value)==(-700,-400)


def test_child_lookup_uses_current_origin_after_translation():
    d,target,_=desktop();d.client_rect=lambda _:Rect(-600,-300,320,240);points=[]
    def screen(hwnd,point):points.append((point._obj.x,point._obj.y));point._obj.x=20;point._obj.y=20;return True
    d.api.ScreenToClient=Function(screen)
    d.api.ChildWindowFromPointEx=Function(lambda parent,*a:11)
    assert d.background_scroll_target(target,Rect(0,0,200,200))==11
    assert points==[(-500,-200)]*2


def test_initial_mapping_retries_translation_and_then_remains_window_local():
    d,target,_=desktop();current=[target.client];calls=[];captures=[]
    d.client_rect=lambda _:current[0];d.background_scroll_target=lambda *a:11
    def bounds(hwnd):
        calls.append(hwnd);current[0]=Rect(-600,-300,320,240)
        return Rect(current[0].x-8,current[0].y-20,336,268)
    d.capture_bounds=bounds
    def factory(**kwargs):c=Capture(**kwargs);captures.append(c);return c
    source=BackgroundSource(d,10,Rect(10,20,200,160),stream_factory=lambda hwnd,extract:WgcStream(hwnd,extract,factory=factory))
    assert len(calls)==2 and source.client==current[0]
    image=np.full((268,336,4),77,np.uint8);image[40:200,18:218,:3]=123
    for position in [Rect(800,900,320,240),Rect(-300,-400,320,240)]:
        current[0]=position;captures[0].emit(image)
        assert (source.read()==123).all()
    source.close()


def test_continuous_startup_movement_does_not_start_an_incoherent_stream():
    from welog.windows import BackgroundTargetChanged
    d,target,_=desktop();current=[target.client];created=[]
    d.client_rect=lambda _:current[0]
    def bounds(hwnd):
        current[0]=Rect(current[0].x+50,current[0].y,320,240)
        return Rect(current[0].x-8,current[0].y-20,336,268)
    d.capture_bounds=bounds
    with pytest.raises(BackgroundTargetChanged) as error:
        BackgroundSource(d,10,Rect(0,0,200,160),stream_factory=lambda *a:created.append(True))
    assert error.value.code=='window_mapping_unstable' and not created


@pytest.mark.parametrize('direction',['up','down'])
@pytest.mark.parametrize('surface',['window','client'])
def test_full_background_capture_while_target_keeps_moving_has_exact_roi_and_height(tmp_path,direction,surface):
    from dataclasses import replace
    from welog.demo import make_scene
    from welog.controller import run_capture
    from welog.archive import read_jsonl
    d,target,messages=desktop();target=replace(target,client=Rect(100,120,640,720),dpi=96)
    d.describe_window=lambda _:target;current=[target.client];d.client_rect=lambda _:current[0]
    d.api.GetDpiForWindow=Function(lambda _:96);d.background_scroll_target=lambda *a:11
    d.capture_bounds=lambda _:Rect(current[0].x-8,current[0].y-20,656,748)
    captures=[];expected_points=[];world=make_scene(1500,640,seed=76)
    def factory(**kwargs):c=Capture(**kwargs);captures.append(c);return c
    class Source(BackgroundSource):
        position=300 if direction=='up' else 0
        reads=0
        def read(self):
            self.reads+=1;current[0]=Rect(-600 if self.reads%2 else 800,-400 if self.reads%3 else 900,640,720)
            expected=world[self.position:self.position+720]
            if surface=='window':
                buffer=np.full((748,656,3),9,np.uint8);buffer[20:740,8:648]=expected
            else:buffer=expected
            captures[0].emit(buffer)
            result=super().read();assert np.array_equal(result,expected)
            return result
        def scroll(self,units):
            expected_points.append((current[0].x+320,current[0].y+360))
            super().scroll(units);self.position+=50 if units>0 else -50
    source=Source(d,10,Rect(0,0,640,720),stream_factory=lambda hwnd,extract:WgcStream(hwnd,extract,factory=factory))
    initial=source.position;errors=[]
    result=run_capture(source,tmp_path/'capture',config=CaptureConfig(capture_backend='background',direction=direction,
        max_frames=7,max_scroll_events=1,stable_samples=2,sample_interval_s=.001,settle_delay_s=0,scroll_event_interval_s=0),
        progress=lambda row:errors.append(row['scroll_offset_px']-(source.position-initial)) if row['accepted'] else None)
    assert result['accepted']==7 and result['rejected']==0 and result['reason']=='max_frames'
    assert result['final_scroll_offset_px']==(-300 if direction=='up' else 300) and errors==[0]*7
    actual=[(ctypes.c_short(m[3]&0xffff).value,ctypes.c_short(m[3]>>16).value) for m in messages]
    assert actual==expected_points and len(messages)==6 and all(m[0]==11 for m in messages)
    assert captures[0].stops==1 and result['image_memory_remaining_bytes']==0 and not list(tmp_path.rglob('*.png'))
    rows=read_jsonl(tmp_path/'capture/messages.jsonl')
    assert [r['global_top_px'] for r in rows]==sorted(r['global_top_px'] for r in rows)

@pytest.mark.parametrize('case', ['read_minimized', 'scroll_hidden', 'closed', 'cancel', 'timeout', 'guard_changed'])
def test_background_capture_handles_target_loss_without_replaying_posted_wheels(tmp_path, monkeypatch, case):
    from dataclasses import replace
    from welog import controller
    from welog.archive import read_jsonl
    from welog.demo import make_scene

    d, target, messages = desktop()
    target = replace(target, client=Rect(100, 120, 640, 744))
    d.describe_window = lambda _: target
    d.client_rect = lambda _: target.client
    d.capture_bounds = lambda _: target.client
    d.background_scroll_target = lambda *args: 11
    state = {'availability': 'ready', 'offset': 0, 'stop': False, 'guard': 100}
    d.api.IsWindow = lambda hwnd: hwnd != 10 or state['availability'] != 'closed'
    d.api.IsWindowVisible = lambda hwnd: hwnd != 10 or state['availability'] != 'hidden'
    d.api.IsIconic = lambda hwnd: hwnd == 10 and state['availability'] == 'minimized'
    scene = make_scene(1300, 640, seed=89)
    streams = []
    waiting = []

    class Stream:
        def __init__(self, hwnd, extract):
            assert hwnd == 10
            self.extract = extract
            self.closed = False
            streams.append(self)

        def read(self):
            assert state['availability'] == 'ready' and not self.closed
            buffer = np.full((744, 640, 3), state['guard'], np.uint8)
            buffer[24:] = scene[state['offset']:state['offset'] + 720]
            return self.extract(buffer), 0

        def close(self):
            assert not self.closed
            self.closed = True

    def post(hwnd, *args):
        assert hwnd == 11 and state['availability'] == 'ready'
        messages.append((hwnd, *args))
        state['offset'] += 50
        if case == 'scroll_hidden' and len(messages) == 2:
            state['availability'] = 'hidden'  # Second event of this batch is still pending.
        elif case != 'scroll_hidden' and len(messages) == 1:
            state['availability'] = 'closed' if case == 'closed' else 'minimized'
        return True

    d.api.PostMessageW = Function(post)
    source = BackgroundSource(d, 10, Rect(0, 24, 640, 720), Rect(0, 0, 80, 20), stream_factory=Stream)

    def on_wait(code):
        waiting.append(code)
        assert len(messages) == (2 if case == 'scroll_hidden' else 1)
        if case == 'cancel':
            state['stop'] = True
        elif case != 'timeout':
            state['availability'] = 'ready'
            if case == 'guard_changed':
                state['guard'] = 220

    if case == 'timeout':
        monkeypatch.setattr(controller, 'BACKGROUND_RETURN_TIMEOUT_S', .01)
    output = tmp_path / 'capture'
    result = controller.run_capture(source, output, config=CaptureConfig(capture_backend='background',
        max_frames=3, stable_samples=2, sample_interval_s=.001, settle_delay_s=0, scroll_event_interval_s=0),
        availability_progress=on_wait, stop_requested=lambda: state['stop'])
    assert result['rejected'] == 0 and result['image_memory_remaining_bytes'] == 0
    assert all(stream.closed for stream in streams)
    assert not list(output.rglob('*.png'))
    if case in {'read_minimized', 'scroll_hidden'}:
        assert waiting == ['window_minimized' if case == 'read_minimized' else 'window_hidden']
        assert len(streams) == 2 and len(messages) == 3
        assert result['reason'] == 'max_frames' and result['accepted'] == 3
        assert result['final_scroll_offset_px'] == 150
        alignments = read_jsonl(output / 'alignments.jsonl')
        assert [row['scroll_offset_px'] for row in alignments] == [0, 50, 150]
        rows = read_jsonl(output / 'messages.jsonl')
        positions = [row['global_top_px'] for row in rows]
        assert positions == sorted(positions)
    else:
        assert len(messages) == 1 and result['accepted'] == 1 and result['final_scroll_offset_px'] == 0
        if case == 'cancel':
            assert result['reason'] == 'user_interrupt'
        else:
            expected = {'closed': 'window_closed', 'timeout': 'window_minimized',
                        'guard_changed': 'conversation_guard_changed'}[case]
            assert result['source_error']['code'] == expected
            assert result['source_error']['phase'] == 'read' and result['source_error']['posted_events'] == 1
            if case == 'timeout':
                assert result['source_error']['details']['wait_timed_out'] is True
            if case == 'closed':
                assert not waiting and len(streams) == 1
