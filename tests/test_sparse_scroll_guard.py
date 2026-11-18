"""Repeated animated media, with no unique cross-screen avatar identity."""
from pathlib import Path

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


def test_calibration_requires_independent_consistent_evidence_and_includes_partial_motion():
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


@pytest.mark.parametrize('delta',[-67,67,-13,0])
def test_single_column_uses_all_viewport_anchors_and_calibrated_upper_bound(delta):
    images,valid=pair(delta)
    direction='up' if delta<=0 else 'down'
    bounds={'minimum':-110 if direction=='up' else 0,'maximum':110 if direction=='down' else 0}
    assert not estimate_edge_layout(*images,*valid,OverlapAligner(),direction).accepted
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
