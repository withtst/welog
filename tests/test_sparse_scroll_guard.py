"""Repeated animated media, with no unique cross-screen avatar identity."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from welog.alignment import OverlapAligner
from welog.archive import read_jsonl
from welog.controller import run_capture
from welog.edge_layout import estimate_edge_layout
from welog.imageio import gray,read_image
from welog.models import CaptureConfig,AlignmentConfig,Rect
from welog.recognition import EmptyOcr
from welog.scroll import ScrollCalibration


class AnimatedConversation:
    height=672
    def __init__(self,direction='down',period=152):
        self.world=np.full((2300,875,3),250,np.uint8)
        self.period=period
        self.starts=list(range(45,2168,period))
        rng=np.random.default_rng(125)
        avatar=rng.integers(0,256,(36,36,3),dtype=np.uint8)
        for y in self.starts:self.world[y:y+36,833:869]=avatar
        # Unique static anchors at the two ends train calibration independently.
        for y in [40,170,300,430,1840,1970,2100,2230]:
            self.world[y:y+36,16:52]=rng.integers(0,256,(36,36,3),dtype=np.uint8)
        self.end=len(self.world)-self.height
        self.position=self.end if direction=='up' else 0
        self.rng=rng;self.inputs=[];self.closed=False

    def read(self):
        image=self.world[self.position:self.position+self.height].copy()
        for y in self.starts:
            a,b=max(0,y-self.position),min(self.height,y+132-self.position)
            if b>a:
                pixels=self.rng.integers(0,160,(b-a,132,3),dtype=np.uint8)
                pixels[:,:,1]=40;pixels[:,:,2]=240
                image[a:b,687:819]=pixels
        return image

    def scroll(self,units):
        self.inputs.append(units)
        self.position=max(0,min(self.end,self.position+(67 if units>0 else -67)))

    def close(self):self.closed=True


def pair(delta,period=152):
    source=AnimatedConversation(period=period)
    # Remove unique end anchors: the entire overlap is now one repeated column.
    source.world[:,:60]=250
    images=[];valid=[]
    for offset in [500,500+delta]:
        source.position=offset;images.append(gray(source.read()))
        mask=np.ones(images[-1].shape,bool)
        mask[:,687:819]=False
        valid.append(mask)
    return images,valid


def test_calibration_requires_independent_evidence_and_safe_upper_bound():
    calibration=ScrollCalibration()
    for _ in range(10):calibration.observe(-67,1,'bounded_layout')
    assert calibration.bounds('up',672) is None
    for delta,events in [(-201,3),(-300,4)]:calibration.observe(delta,events,'template')
    assert calibration.bounds('up',672) is None
    calibration.observe(-201,3,'static_features')
    bounds=calibration.bounds('up',672)
    assert bounds['minimum']==-121 and bounds['maximum']==0
    assert calibration.bounds('down',672)['maximum']==121
    calibration.observe(-240,1,'template')
    assert calibration.bounds('up',672) is None


def test_partial_startup_steps_do_not_disable_an_upper_bound():
    calibration=ScrollCalibration()
    for delta,events in [(-37,1),(-104,2),(-300,4)]:calibration.observe(delta,events,'text')
    assert calibration.bounds('up',663)['minimum']==-121
    assert calibration.bounds('down',663)['maximum']==121
    calibration.observe(-2,1,'text')
    assert calibration.bounds('up',663)['minimum']==-121
    calibration.observe(-240,1,'text')
    assert calibration.bounds('up',663) is None


def test_accelerated_startup_then_pure_animation_reaches_boundary(tmp_path):
    from welog.recognition import TextLine
    class Source(AnimatedConversation):
        def __init__(self):
            super().__init__('up');self.count=0
            for y,text in [(1900,'alpha'),(2050,'beta')]:
                self.world[y:y+36,65:265]=238
                cv2.putText(self.world,text,(75,y+26),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1)
        def scroll(self,units):
            self.inputs.append(units)
            steps=[37,52,52,75,75,75,75]
            delta=steps[self.count] if self.count<len(steps) else 67
            self.count+=1;self.position=max(0,self.position-delta)
    source=Source()
    class Backend:
        def recognize(self,image):
            if len(image)!=672:return []
            return [TextLine(text,(75,y+10-source.position,180,y+28-source.position),.99)
                    for y,text in [(1900,'alpha'),(2050,'beta')]
                    if 2<y+10-source.position and y+28-source.position<670]
    result=run_capture(source,tmp_path/'capture',config=CaptureConfig(max_frames=0,direction='up',
        sample_interval_s=.001,stable_samples=3,stable_timeout_s=3,settle_delay_s=0,
        scroll_event_interval_s=0,boundary_wait_s=0),ocr_backend=Backend())
    decisions=read_jsonl(tmp_path/'capture/alignments.jsonl');frames=read_jsonl(tmp_path/'capture/frames.jsonl')
    assert result['status']=='boundary_reached',result
    assert [r['alignment']['delta_px'] for r in decisions[:4]]==[0,-37,-104,-300]
    assert frames[4]['action']['events']==1 and frames[4]['action']['motion_bounds']['minimum']==-121
    assert decisions[4]['motion_method']=='bounded_layout'
    assert result['final_scroll_offset_px']==-1628 and result['rejected']==0
    assert result['image_memory_remaining_bytes']==0 and all(s<0 for s in source.inputs)
    rows=read_jsonl(tmp_path/'capture/messages.jsonl')
    assert [r['global_top_px'] for r in rows]==sorted(r['global_top_px'] for r in rows)
    assert [r['text'] for r in rows if r['type']=='text']==['alpha','beta']
    assert not list(tmp_path.rglob('*.png'))


@pytest.mark.parametrize('delta',[-67,67,-13,0])
def test_single_column_uses_all_viewport_anchors_and_calibrated_upper_bound(delta):
    images,valid=pair(delta)
    direction='up' if delta<=0 else 'down'
    bounds={'minimum':-110 if direction=='up' else 0,'maximum':110 if direction=='down' else 0}
    unbounded=estimate_edge_layout(*images,*valid,OverlapAligner(),direction)
    assert not unbounded.accepted and unbounded.reason=='edge_layout_needs_motion_bounds'
    result=estimate_edge_layout(*images,*valid,OverlapAligner(),direction,bounds)
    assert result.accepted and result.delta_px==delta
    assert result.inliers>=6 and result.validation_coverage>=.75


def test_period_shorter_than_upper_bound_and_out_of_range_nonperiodic_motion_fail():
    images,valid=pair(0,period=100)
    result=estimate_edge_layout(*images,*valid,OverlapAligner(),'up',{'minimum':-110,'maximum':0})
    assert not result.accepted and result.reason=='ambiguous_edge_layout'
    images,valid=pair(-50)
    assert not estimate_edge_layout(*images,*valid,OverlapAligner(),'up',{'minimum':-30,'maximum':0}).accepted
    assert not estimate_edge_layout(*images,*valid,OverlapAligner(),'down',{'minimum':0,'maximum':30}).accepted


def test_replay_preserves_bounds_and_does_not_count_diagnostic_frame(tmp_path):
    from welog.archive import Archive
    from welog.models import AlignmentResult
    from welog.replay import run_replay
    images,valid=pair(-67)
    archive=Archive(tmp_path/'source',{'kind':'live_debug','capture_config':{'direction':'up'}})
    bounds={'minimum':-110,'maximum':0,'basis':'verified_scrolls_single_event','samples':3}
    for index,image in enumerate([*images,images[1]]):
        action={'kind':'initial' if index==0 else 'scroll' if index==1 else 'unstable_diagnostic',
            'stability':{'dynamic_masks':[Rect(687,0,132,672).to_dict()]}}
        if index==1:action['motion_bounds']=bounds
        frame=archive.capture(np.repeat(image[:,:,None],3,axis=2),action=action,truth_px=0 if index==0 else -67)
        archive.evaluate(frame,AlignmentResult(index<2,'fixture',delta_px=0 if index==0 else -67 if index==1 else None),
            reference_id=None if index==0 else '000000',cumulative_px=0 if index==0 else -67 if index==1 else None)
    archive.finish('needs_attention',final_scroll_offset_px=-67)
    report=run_replay(archive.path,tmp_path/'replay',allow_failures=True)
    assert report['accepted']==2 and report['excluded']==1 and report['rejected']==0
    assert report['final_scroll_offset_px']==-67 and report['max_abs_cumulative_error_px']==0


def test_frozen_animation_phase_keeps_media_guard(tmp_path):
    source=AnimatedConversation()
    original=source.read
    reads=0;image=None
    def read():
        nonlocal reads,image
        reads+=1
        if reads<=4:image=original()
        return image.copy()
    source.read=read
    output=tmp_path/'capture'
    run_capture(source,output,config=CaptureConfig(max_frames=2,sample_interval_s=.001,
        stable_samples=3,stable_timeout_s=3),ocr_backend=EmptyOcr())
    assert all(row['media_dominated'] for row in read_jsonl(output/'alignments.jsonl'))
    frames=read_jsonl(output/'frames.jsonl')
    assert frames[-1]['action']['stability']['reason']=='static'
    assert frames[-1]['action']['scroll_guard']=='media_dominated'


@pytest.mark.parametrize('direction',['up','down'])
def test_capture_crosses_repeating_animation_and_reaches_boundary_with_exact_height(tmp_path,direction):
    source=AnimatedConversation(direction);initial=source.position
    output=tmp_path/'capture'
    result=run_capture(source,output,config=CaptureConfig(max_frames=0,direction=direction,
        sample_interval_s=.001,stable_samples=3,stable_timeout_s=3,settle_delay_s=0,
        scroll_event_interval_s=0,boundary_wait_s=0),ocr_backend=EmptyOcr())
    assert result['status']=='boundary_reached',result
    assert result['final_scroll_offset_px']==source.position-initial
    assert result['final_scroll_offset_px']==(-1628 if direction=='up' else 1628)
    assert result['rejected']==0 and source.closed and result['image_memory_remaining_bytes']==0
    frames=read_jsonl(output/'frames.jsonl');decisions=read_jsonl(output/'alignments.jsonl')
    assert any(d['motion_method']=='bounded_layout' for d in decisions)
    assert all(f['action'].get('events',0)<=1 for f in frames)
    assert any(f['action'].get('motion_bounds') for f in frames)
    assert all(s<0 if direction=='up' else s>0 for s in source.inputs)
    assert not list(output.rglob('*.png'))
    rows=read_jsonl(output/'messages.jsonl')
    assert [r['global_top_px'] for r in rows]==sorted(r['global_top_px'] for r in rows)
    assert len(rows)==len(source.starts)


ROOT=Path('runs/pointer-20261118-225049-8fa73f/capture')
@pytest.mark.skipif(not ROOT.exists(),reason='Private event screenshots unavailable')
def test_real_three_event_pair_stays_unconfirmed_without_inventing_an_offset():
    from welog.motion import MotionTracker
    from welog.frame_pool import ImageBudget
    frames={f['frame_id']:f for f in read_jsonl(ROOT/'frames.jsonl')}
    decisions={f['frame_id']:f for f in read_jsonl(ROOT/'alignments.jsonl')}
    tracker=MotionTracker(ImageBudget(),AlignmentConfig())
    try:
        for n in [535,536,537,538,539]:
            key=f'{n:06d}';frame=frames[key]
            masks=tuple(Rect(**b) for b in frame['action']['stability']['dynamic_masks']+decisions[key]['motion_overlay_masks'])
            result,_,_=tracker.accept(read_image(ROOT/'frames'/f'{key}.png'),key,[],'up',masks)
            assert result.accepted==(n==535)
        assert tracker.offset==0 and tracker.reference_id=='000535'
    finally:tracker.close()
    assert tracker.budget.used_bytes==0


WARMUP_ROOT=Path('runs/pointer-20261118-235104-29799c/capture')
@pytest.mark.skipif(not WARMUP_ROOT.exists(),reason='Private warmup event screenshots unavailable')
def test_real_variable_startup_and_retained_tail_use_independent_upper_bound():
    from welog.motion import MotionTracker
    from welog.frame_pool import ImageBudget
    frames=read_jsonl(WARMUP_ROOT/'frames.jsonl');decisions=read_jsonl(WARMUP_ROOT/'alignments.jsonl')
    calibration=ScrollCalibration()
    for frame,row in zip(frames[:4],decisions[:4]):
        calibration.observe(row['alignment']['delta_px'],frame['action'].get('events',0),row['motion_method'])
    bounds=calibration.bounds('up',frames[3]['height'])
    assert bounds['minimum']==-121
    tracker=MotionTracker(ImageBudget(),AlignmentConfig());tracker.offset=decisions[2]['scroll_offset_px']
    try:
        for index in range(2,8):
            image=read_image(WARMUP_ROOT/'frames'/f'{index:06d}.png')
            masks=tuple(Rect(**b) for b in frames[index]['action']['stability']['dynamic_masks']+decisions[index]['motion_overlay_masks'])
            if index==4:
                result,_,_=tracker.accept(image,str(index),[],'up',masks)
                assert not result.accepted and result.edge_rejection_reason=='edge_layout_needs_motion_bounds'
                assert tracker.offset==-441 and tracker.reference_id=='3'
            result,_,method=tracker.accept(image,str(index),[],'up',masks,bounds if index>=4 else None)
            assert result.accepted and result.delta_px==({2:0,3:-300,4:-37}.get(index,0))
            if index==4:assert method=='bounded_layout'
        assert tracker.offset==-478
    finally:tracker.close()
    assert read_jsonl(WARMUP_ROOT/'alignments.jsonl')==decisions and tracker.budget.used_bytes==0
