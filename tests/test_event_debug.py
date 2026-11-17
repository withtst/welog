import json
from dataclasses import replace
import cv2
import numpy as np
import pytest

from welog.archive import archive_frames,read_jsonl
from welog.controller import run_capture,StableTimeout
from welog.debug_capture import EventDebugSession
from welog.imageio import read_image
from welog.models import CaptureConfig
from welog.recognition import EmptyOcr,TextLine


def picture(unknown=False):
    image=np.full((400,500,3),250,np.uint8)
    image[50:160,45:280]=255 if unknown else 238
    if not unknown:
        cv2.putText(image,'hello world',(60,95),cv2.FONT_HERSHEY_SIMPLEX,.7,(25,25,25),1)
    return image


class Source:
    closed=False
    def __init__(self,unknown=False):self.image=picture(unknown);self.inputs=[]
    def read(self):return self.image.copy()
    def scroll(self,units):self.inputs.append(units)
    def close(self):self.closed=True


class Backend:
    def recognize(self,image):return [TextLine('hello world',(60,75,250,105),.99)]


def config(**kwargs):
    return CaptureConfig(debug_save_images=True,max_frames=2,sample_interval_s=.001,stable_samples=2,
                         stable_timeout_s=.5,settle_delay_s=0,**kwargs)


def test_event_mode_success_saves_no_images_and_releases_raw_reference(tmp_path):
    source=Source()
    result=run_capture(source,tmp_path/'capture',config=config(),ocr_backend=Backend())
    assert result['accepted']==2 and result['debug_image_count']==0 and source.closed
    assert result['image_storage']=='event_png' and result['debug_image_policy']=='events'
    assert not list(tmp_path.rglob('*.png')) and not (tmp_path/'capture/diagnostics.jsonl').exists()
    assert result['image_memory_remaining_bytes']==0
    assert all('image' not in row for row in read_jsonl(tmp_path/'capture/frames.jsonl'))


def test_unknown_message_keeps_unmasked_frame_and_correct_crop(tmp_path):
    source=Source(True)
    result=run_capture(source,tmp_path/'capture',config=replace(config(),max_frames=1),ocr_backend=EmptyOcr())
    assert result['debug_image_count']==2
    events=read_jsonl(tmp_path/'capture/diagnostics.jsonl')
    assert events[0]['reasons']==['unknown_message'] and events[0]['coordinate_eligible']
    crop=next(image for image in events[0]['images'] if image['role']=='unknown_crop')
    x1,y1,x2,y2=crop['box']
    assert np.array_equal(read_image(tmp_path/'capture'/crop['image']),source.image[y1:y2,x1:x2])
    assert crop['message_id']
    with pytest.raises(ValueError,match='Event-only'):archive_frames(tmp_path/'capture')


def test_unknown_duplicates_are_suppressed_but_full_view_can_upgrade_partial(tmp_path):
    session=EventDebugSession(tmp_path,{},1024*1024)
    image=picture(True)
    def outcome(complete):return {'alignment':{'accepted':True},'unknown_observations':[{'message_id':'one','box':[45,50,280,160],
        'top_state':'visible' if complete else 'clipped','bottom_state':'visible'}]}
    for complete in [False,False,True,True]:
        frame=session.capture(image,{'kind':'initial'})
        session.process_event(frame,image,outcome(complete))
    assert len(read_jsonl(tmp_path/'diagnostics.jsonl'))==2
    assert len(list((tmp_path/'unknown').glob('*.png')))==2


def test_long_unknown_keeps_new_partial_coverage_instead_of_losing_later_parts(tmp_path):
    session=EventDebugSession(tmp_path,{},1024*1024);image=picture(True)
    for offset in [0,0,100,100,200]:
        frame=session.capture(image,{'kind':'scroll'})
        session.process_event(frame,image,{'alignment':{'accepted':True},'scroll_offset_px':offset,
            'unknown_observations':[{'message_id':'long','box':[45,50,280,160],'top_state':'clipped','bottom_state':'clipped'}]})
    assert len(read_jsonl(tmp_path/'diagnostics.jsonl'))==3
    assert len(list((tmp_path/'unknown').glob('*.png')))==3


def test_rejected_frame_saves_previous_reference_only_when_problem_occurs(tmp_path,monkeypatch):
    from welog import controller
    class Bad(Source):
        def scroll(self,units):super().scroll(units);self.image[:]=255
    source=Bad()
    result=run_capture(source,tmp_path/'capture',config=replace(config(),max_frames=3,recovery_attempts=1),ocr_backend=Backend())
    events=read_jsonl(tmp_path/'capture/diagnostics.jsonl')
    first=next(event for event in events if 'alignment_rejected' in event['reasons'])
    assert first['reference_frame_id']=='000000' and not first['coordinate_eligible']
    assert {image['role'] for image in first['images']}=={'reference','current'}
    assert result['final_scroll_offset_px']==0 and result['image_memory_remaining_bytes']==0


def test_timeout_keeps_temporal_samples_without_counting_diagnostic_height(tmp_path):
    class Animated(Source):
        def __init__(self):super().__init__();self.rng=np.random.default_rng(121)
        def read(self):return self.rng.integers(0,256,(180,220,3),dtype=np.uint8)
    result=run_capture(Animated(),tmp_path/'capture',config=replace(config(),stable_timeout_s=.2),ocr_backend=EmptyOcr())
    assert result['reason']=='StableTimeout' and result['final_scroll_offset_px']==0
    event=next(row for row in read_jsonl(tmp_path/'capture/diagnostics.jsonl') if 'stability_timeout' in row['reasons'])
    assert not event['coordinate_eligible']
    samples=[row for row in event['images'] if row['role']=='temporal_sample']
    assert samples and len(samples)<=3
    assert result['image_memory_remaining_bytes']==0


def test_event_quota_failure_is_recorded_without_corrupting_coordinates(tmp_path):
    result=run_capture(Source(True),tmp_path/'capture',config=replace(config(),max_frames=1,debug_max_bytes=1),ocr_backend=EmptyOcr())
    assert result['reason']=='DebugImageLimit' and result['image_memory_remaining_bytes']==0
    assert not list(tmp_path.rglob('*.png'))
    assert any(event.get('image_save_error')=='DebugImageLimit' for event in read_jsonl(tmp_path/'capture/diagnostics.jsonl'))


def test_normal_mode_disables_all_event_images_even_on_timeout(tmp_path,monkeypatch):
    from welog import controller
    def unstable(*args,**kwargs):raise StableTimeout(picture(),{'reason':'fixture'})
    monkeypatch.setattr(controller,'wait_for_stable',unstable)
    result=run_capture(Source(),tmp_path/'capture',config=replace(config(),debug_save_images=False),ocr_backend=EmptyOcr())
    assert not list(tmp_path.rglob('*.png')) and result['debug_image_count']==0


def test_ocr_error_retains_frame_and_close_releases_memory(tmp_path):
    class Error:
        def recognize(self,image):raise RuntimeError('private content')
    result=run_capture(Source(),tmp_path/'capture',config=config(),ocr_backend=Error())
    assert result['reason']=='ocr_backend_failed' and result['debug_image_count']==1
    assert result['image_memory_remaining_bytes']==0
    assert any('ocr_error' in event['reasons'] for event in read_jsonl(tmp_path/'capture/diagnostics.jsonl'))


def test_visible_unreadable_text_is_a_problem_even_without_backend_exception(tmp_path):
    result=run_capture(Source(),tmp_path/'capture',config=replace(config(),max_frames=1),ocr_backend=EmptyOcr())
    events=read_jsonl(tmp_path/'capture/diagnostics.jsonl')
    assert result['debug_image_count']==1 and 'ocr_unreadable' in events[0]['reasons']
    assert events[0]['ocr_failed_regions']


def test_media_audit_finds_event_images_and_does_not_claim_pixels_are_missing(tmp_path):
    import importlib.util
    from pathlib import Path
    from welog.exports import RecordSource
    root=tmp_path/'capture'
    run_capture(Source(True),root,config=replace(config(),max_frames=1),ocr_backend=EmptyOcr())
    spec=importlib.util.spec_from_file_location('event_audit',Path(__file__).resolve().parents[1]/'tools/audit_media_records.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report=module.audit(RecordSource(root))
    assert report['retained_frame_count']==1 and report['capture_image_files']==2
    assert report['records'][0]['verdict']=='needs_visual_review'
    assert report['records'][0]['retained_crops']
