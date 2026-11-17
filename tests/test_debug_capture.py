import json
import hashlib
from dataclasses import replace

import numpy as np
import pytest

from welog.archive import read_jsonl,archive_frames
from welog.controller import run_capture,StableTimeout
from welog.demo import make_scene
from welog.imageio import read_image
from welog.models import CaptureConfig
from welog.recognition import EmptyOcr
from welog.replay import run_replay


class Source:
    def __init__(self):
        self.scene=make_scene(1400,640)
        self.position=680
        self.closed=False
    def read(self):return self.scene[self.position:self.position+720].copy()
    def scroll(self,units):self.position=max(0,min(680,self.position+units))
    def close(self):self.closed=True


def config(**kwargs):
    return CaptureConfig(debug_save_images=True,direction='up',max_frames=4,sample_interval_s=.001,
        stable_samples=2,settle_delay_s=0,scroll_event_interval_s=0,**kwargs)


def test_debug_frames_are_unmasked_lossless_and_replayable(tmp_path):
    source=Source();first=source.read()
    result=run_capture(source,tmp_path/'capture',config=config(),ocr_backend=EmptyOcr())
    assert result['image_storage']=='debug_png' and result['debug_image_count']==4
    metadata,frames=archive_frames(tmp_path/'capture')
    assert metadata['kind']=='live_debug' and metadata['storage_mode']=='debug_images'
    assert np.array_equal(read_image(tmp_path/'capture'/frames[0]['image']),first)
    for frame in frames:
        path=tmp_path/'capture'/frame['image']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==frame['sha256']
    replay=run_replay(tmp_path/'capture',tmp_path/'replay')
    assert replay['accepted']==4 and replay['normal_frame_direction']=='up'
    assert replay['final_scroll_offset_px']==result['final_scroll_offset_px']
    assert not list((tmp_path/'replay').rglob('*.png'))
    assert source.closed and result['image_memory_remaining_bytes']==0


def test_debug_unstable_diagnostic_is_retained_and_excluded_from_coordinates(tmp_path,monkeypatch):
    from welog import controller
    source=Source()
    calls=[]
    def stable(source,settings,alignment,stopped,diagnostics,budget):
        calls.append(1)
        if len(calls)>1:raise StableTimeout(source.read(),{'stable':False})
        diagnostics.update(reason='fixture',stable=True)
        return source.read()
    monkeypatch.setattr(controller,'wait_for_stable',stable)
    result=run_capture(source,tmp_path/'capture',config=config(),ocr_backend=EmptyOcr())
    assert result['reason']=='StableTimeout' and result['debug_image_count']==2
    frames=read_jsonl(tmp_path/'capture/frames.jsonl')
    assert frames[-1]['action']['kind']=='unstable_diagnostic'
    replay=run_replay(tmp_path/'capture',tmp_path/'replay')
    assert replay['excluded']==1 and replay['accepted']==1
    assert replay['final_scroll_offset_px']==0


def test_rejected_match_images_are_retained_but_never_become_replay_origins(tmp_path):
    class BadSource(Source):
        moved=False
        def scroll(self,units):self.moved=True
        def read(self):return np.full((720,640,3),255,np.uint8) if self.moved else super().read()
    result=run_capture(BadSource(),tmp_path/'capture',config=config(),ocr_backend=EmptyOcr())
    assert result['rejected']>0 and result['final_scroll_offset_px']==0
    replay=run_replay(tmp_path/'capture',tmp_path/'replay')
    assert replay['excluded']==result['rejected'] and replay['accepted']==1


def test_debug_quota_stops_and_keeps_previous_images(tmp_path):
    source=Source()
    from welog.imageio import write_image
    reference=tmp_path/'size.png';write_image(reference,source.read())
    result=run_capture(source,tmp_path/'capture',config=config(debug_max_bytes=reference.stat().st_size),ocr_backend=EmptyOcr())
    assert result['reason']=='DebugImageLimit' and result['debug_image_count']==1
    assert len(read_jsonl(tmp_path/'capture/frames.jsonl'))==1
    assert len(list((tmp_path/'capture/frames').glob('*.png')))==1
    assert source.closed and result['image_memory_remaining_bytes']==0


def test_diagnostic_quota_failure_keeps_prior_frames_replayable(tmp_path,monkeypatch):
    from welog import controller
    from welog.imageio import write_image
    source=Source();reference=tmp_path/'size.png';write_image(reference,source.read())
    calls=[]
    def stable(source,settings,alignment,stopped,diagnostics,budget):
        calls.append(1)
        if len(calls)>1:raise StableTimeout(source.read(),{'stable':False})
        diagnostics.update(reason='fixture',stable=True)
        return source.read()
    monkeypatch.setattr(controller,'wait_for_stable',stable)
    result=run_capture(source,tmp_path/'capture',config=config(debug_max_bytes=reference.stat().st_size),ocr_backend=EmptyOcr())
    assert result['diagnostic_image_save_error']=='DebugImageLimit'
    assert len(read_jsonl(tmp_path/'capture/diagnostics.jsonl'))==1
    replay=run_replay(tmp_path/'capture',tmp_path/'replay')
    assert replay['accepted']==1 and replay['final_scroll_offset_px']==0


@pytest.mark.parametrize('failure',['writer','ocr'])
def test_debug_failures_release_memory_and_close_source(tmp_path,monkeypatch,failure):
    source=Source()
    if failure=='writer':
        from welog import debug_capture
        def broken(*args):raise OSError('write failed')
        monkeypatch.setattr(debug_capture,'write_image',broken)
    class Backend:
        def recognize(self,image):raise RuntimeError('private text')
    result=run_capture(source,tmp_path/'capture',config=config(),ocr_backend=Backend())
    assert result['status']=='needs_attention' and source.closed
    assert result['image_memory_remaining_bytes']==0
    assert result['debug_image_count']==(0 if failure=='writer' else 1)


def test_default_mode_does_not_retain_any_images(tmp_path):
    result=run_capture(Source(),tmp_path/'capture',config=replace(config(),debug_save_images=False),ocr_backend=EmptyOcr())
    assert result['image_storage']=='none' and result['debug_image_count']==0
    assert not (tmp_path/'capture/frames').exists()
    assert all('image' not in frame for frame in read_jsonl(tmp_path/'capture/frames.jsonl'))
