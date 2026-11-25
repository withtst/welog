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


@pytest.mark.parametrize('direction',['up','down'])
def test_accelerated_startup_then_pure_animation_reaches_boundary(tmp_path,direction):
    from welog.recognition import TextLine
    class Source(AnimatedConversation):
        def __init__(self):
            super().__init__('up');self.count=0
            self.starts.remove(1717);self.world[1717:1753,833:869]=250
            self.texts=[(1710,'alpha'),(1766,'beta')]
            if direction=='down':
                self.world=self.world[::-1].copy();self.world[:,833:869]=250
                self.starts=[len(self.world)-y-132 for y in self.starts]
                self.texts=[(len(self.world)-y-36,text) for y,text in self.texts]
                avatar=np.random.default_rng(126).integers(0,256,(36,36,3),dtype=np.uint8)
                for y in self.starts:self.world[y:y+36,833:869]=avatar
                self.position=0
            for y,text in self.texts:
                self.world[y:y+36,65:265]=238
                cv2.putText(self.world,text,(75,y+26),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1)
        def scroll(self,units):
            self.inputs.append(units)
            steps=[37,52,52,75,75,75,75]
            delta=steps[self.count] if self.count<len(steps) else 67
            self.count+=1;self.position=max(0,min(self.end,self.position+(delta if direction=='down' else -delta)))
    source=Source()
    class Backend:
        def recognize(self,image):
            if len(image)!=672:return []
            return [TextLine(text,(75,y+10-source.position,180,y+28-source.position),.99)
                    for y,text in source.texts
                    if 2<y+10-source.position and y+28-source.position<670]
    result=run_capture(source,tmp_path/'capture',config=CaptureConfig(max_frames=0,direction=direction,
        sample_interval_s=.001,stable_samples=3,stable_timeout_s=3,settle_delay_s=0,
        scroll_event_interval_s=0,boundary_wait_s=0),ocr_backend=Backend())
    decisions=read_jsonl(tmp_path/'capture/alignments.jsonl');frames=read_jsonl(tmp_path/'capture/frames.jsonl')
    assert result['status']=='boundary_reached',result
    sign=1 if direction=='down' else -1
    assert [r['alignment']['delta_px'] for r in decisions[:4]]==[0,37*sign,104*sign,300*sign]
    bounds=frames[4]['action']['motion_bounds']
    assert frames[4]['action']['events']==1 and max(abs(bounds['minimum']),abs(bounds['maximum']))==121
    assert frames[4]['action']['scroll_guard']=='media_anchor_exit'
    assert frames[4]['action']['anchor_forecast']['available']==2 and frames[4]['action']['anchor_forecast']['retained']==0
    assert any(d['motion_method']=='bounded_layout' for d in decisions)
    assert result['final_scroll_offset_px']==1628*sign and result['rejected']==0
    assert result['image_memory_remaining_bytes']==0 and all(s<0 if direction=='up' else s>0 for s in source.inputs)
    rows=read_jsonl(tmp_path/'capture/messages.jsonl')
    assert [r['global_top_px'] for r in rows]==sorted(r['global_top_px'] for r in rows)
    assert [r['text'] for r in rows if r['type']=='text']==[text for _,text in sorted(source.texts)]
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


def test_multi_event_bound_keeps_periodic_aliases_unconfirmed():
    calibration=ScrollCalibration()
    for delta in [-201,-205,-197]:calibration.observe(delta,3,'text')
    images,valid=pair(-205)
    result=estimate_edge_layout(*images,*valid,OverlapAligner(),'up',calibration.batch_bounds(3,'up',672))
    assert not result.accepted and result.reason.startswith('ambiguous')


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


@pytest.mark.parametrize('direction', ['up', 'down'])
def test_white_sticker_run_uses_single_steps_even_when_all_frames_are_static(tmp_path, direction):
    from test_screenshot_layout import white_sticker
    from welog.recognition import TextLine

    height = 663
    world = np.full((2800, 880, 3), 247, np.uint8)
    avatar = np.random.default_rng(97).integers(0, 256, (36, 36, 3), dtype=np.uint8)
    photos = list(range(700, 2010, 152))
    for y in photos:
        world[y:y+132, 70:202] = white_sticker()
        world[y:y+36, 20:56] = avatar
    lines = [(y, f'message {y}') for y in [*range(22, 653, 72), *range(2082, 2780, 72)]]
    for y, text in lines:
        world[y:y+36, 20:56] = avatar
        world[y:y+36, 65:230] = 233
        cv2.putText(world, text, (74, y+25), cv2.FONT_HERSHEY_SIMPLEX, .53, (28,)*3, 1)

    class Source:
        end = len(world) - height
        position = end if direction == 'up' else 0
        pending = 0

        def read(self):
            steps = (0, 37, 104, 201, 300, 400)
            delta = steps[self.pending] * (-1 if direction == 'up' else 1)
            self.position = min(self.end, max(0, self.position + delta))
            self.pending = 0
            return world[self.position:self.position+height].copy()

        def scroll(self, units):
            assert (units < 0) == (direction == 'up')
            self.pending += 1

        def close(self):
            pass

    source = Source()

    class KnownLines:
        def recognize(self, image):
            if len(image) != height:
                return []
            return [TextLine(text, (74, y+7-source.position, 188, y+30-source.position))
                    for y, text in lines if 2 < y+7-source.position and y+30-source.position < height-2]

    path = tmp_path/'capture'
    result = run_capture(source, path, config=CaptureConfig(direction=direction, max_frames=0,
        max_scroll_events=5, stable_samples=2, sample_interval_s=.001, settle_delay_s=0,
        scroll_event_interval_s=0, boundary_wait_s=0), ocr_backend=KnownLines())
    assert result['status'] == 'boundary_reached' and result['rejected'] == 0, result
    assert result['final_scroll_offset_px'] == (-source.end if direction == 'up' else source.end)
    frames = read_jsonl(path/'frames.jsonl')
    guarded = [frame['action'] for frame in frames if frame['action'].get('scroll_guard') == 'media_dominated']
    assert len(guarded) >= 5 and all(a['events'] == 1 and a['stability']['reason'] == 'static' for a in guarded)
    rows = read_jsonl(path/'messages.jsonl')
    assert [r['text'] for r in rows if r['type'] == 'text'] == [text for _, text in lines]
    assert sum(r['type'] == 'image' for r in rows) == len(photos)
    assert all(r['type'] != 'unknown' for r in rows)


def test_frozen_media_risk_survives_two_text_anchors_near_the_leaving_edge(tmp_path):
    from welog.recognition import TextLine
    source=AnimatedConversation('up');positions=[(2160,'alpha'),(2220,'beta')]
    for y,text in positions:
        source.world[y:y+36,65:265]=238
        cv2.putText(source.world,text,(75,y+26),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1)
    original=source.read;reads=0;cached=None;cached_position=None
    def read():
        nonlocal reads,cached,cached_position
        reads+=1
        if reads<=3 or cached_position!=source.position:
            cached=original();cached_position=source.position
        return cached.copy()
    def scroll(units):source.inputs.append(units);source.position=max(0,source.position-37)
    source.read=read;source.scroll=scroll
    class Backend:
        def recognize(self,image):
            if len(image)!=672:return []
            return [TextLine(text,(75,y+10-source.position,180,y+28-source.position),.99) for y,text in positions
                    if 2<y+10-source.position and y+28-source.position<670]
    result=run_capture(source,tmp_path/'capture',config=CaptureConfig(max_frames=3,direction='up',
        stable_samples=3,stable_timeout_s=3,sample_interval_s=.001,settle_delay_s=0,scroll_event_interval_s=0),ocr_backend=Backend())
    decisions=read_jsonl(tmp_path/'capture/alignments.jsonl');frames=read_jsonl(tmp_path/'capture/frames.jsonl')
    assert result['accepted']==3 and result['rejected']==0 and result['final_scroll_offset_px']==-74
    assert frames[1]['action']['stability']['reason']=='static'
    assert decisions[1]['media_guard_eligible'] and not decisions[1]['media_dominated']
    assert len(decisions[1]['motion_anchor_boxes'])==2
    assert frames[2]['action']['scroll_guard']=='media_anchor_exit' and frames[2]['action']['events']==1
    assert result['image_memory_remaining_bytes']==0 and not list(tmp_path.rglob('*.png'))


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
