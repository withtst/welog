"""Opaque notice surfaces occlude page evidence even when the surface is blank."""
from pathlib import Path
import cv2
import numpy as np
import pytest
from welog.text_rules import visual_notice_layers,suppress_visual_notice
from welog.motion import MotionTracker
from welog.replay import CoordinateTracker,run_replay
from welog.alignment import OverlapAligner
from welog.models import Rect,AlignmentConfig,AlignmentResult
from welog.frame_pool import ImageBudget
from welog.archive import Archive,read_jsonl
from welog.imageio import read_image


def notice(image,count=3):
    image=image.copy();y=len(image)-43
    image[y:y+42,764:874]=(image[y:y+42,764:874].astype(float)*.9+20).astype(np.uint8)
    cv2.rectangle(image,(772,y+5),(869,y+30),(250,250,250),-1)
    cv2.putText(image,f'{count} new msg',(779,y+25),cv2.FONT_HERSHEY_SIMPLEX,.42,(120,180,0),1)
    return image


def mask_at(masks,x,y):return any(b.x<=x<b.right and b.y<=y<b.bottom for b in masks)


def test_opaque_pale_surface_is_evidence_masked_while_visible_message_fragments_survive():
    image=np.full((663,880,3),250,np.uint8)
    image[618:654,739:824]=(153,244,151)
    image[618:654,834:870]=np.random.default_rng(930).integers(0,210,(36,36,3),dtype=np.uint8)
    cv2.putText(image,'ok',(741,639),cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,20),1)
    overlay=notice(image);original=overlay.copy();segment,evidence=visual_notice_layers(overlay)
    assert (overlay[630,835]==250).all() and mask_at(evidence,835,630)
    assert not mask_at(evidence,745,640)  # Unoccluded colored message fill.
    assert not mask_at(evidence,750,633)  # Outside the floating layer.
    clean=suppress_visual_notice(overlay,segment)
    assert np.array_equal(clean[620:650,739:760],overlay[620:650,739:760])
    assert np.array_equal(overlay,original)


def world():
    canvas=np.full((1300,880,3),250,np.uint8);rng=np.random.default_rng(931)
    for i,y in enumerate([120,260,408,550,657,713,890,1060]):
        x=834 if i%2==0 or y==713 else 20
        canvas[y:y+36,x:x+36]=rng.integers(0,256,(36,36,3),dtype=np.uint8)
        a,b=(725,824) if x==834 else (70,300)
        canvas[y:y+36,a:b]=(153,244,151) if x==834 else (238,238,238)
        cv2.putText(canvas,'text '+str(i),(a+5,y+25),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1)
    return canvas


def tracker(replay):return CoordinateTracker(OverlapAligner()) if replay else MotionTracker(ImageBudget(),AlignmentConfig())
def accept(t,replay,image,key,direction,bounds=None,masks=()):
    if replay:return t.accept(image,key,direction=direction,dynamic_masks=masks,motion_bounds=bounds,media_risk=True)[0]
    return t.accept(image,key,[],direction,masks,bounds,media_risk=True)[0]
def offset(t,replay):return t.scroll_offset_px if replay else t.offset


@pytest.mark.parametrize('replay',[False,True])
@pytest.mark.parametrize('direction',['up','down'])
@pytest.mark.parametrize('appears',[False,True])
def test_notice_appearing_or_disappearing_over_message_does_not_invent_motion(replay,direction,appears):
    image=world()[95:758].copy();a,b=(image,notice(image)) if appears else (notice(image),image)
    bounds={'minimum':-265,'maximum':0} if direction=='up' else {'minimum':0,'maximum':265}
    t=tracker(replay)
    try:
        assert accept(t,replay,a,'a',direction).accepted
        result=accept(t,replay,b,'b',direction,bounds)
        assert result.accepted and result.delta_px==0 and offset(t,replay)==0
    finally:
        if not replay:t.close()


@pytest.mark.parametrize('replay',[False,True])
@pytest.mark.parametrize('direction',['up','down'])
def test_covered_avatar_and_message_do_not_veto_real_scroll_or_bypass_direction(replay,direction):
    canvas=world();old=notice(canvas[300:963],3);new=notice(canvas[95:758],4)
    if direction=='down':old,new=new,old
    bounds={'minimum':-265,'maximum':0} if direction=='up' else {'minimum':0,'maximum':265}
    t=tracker(replay)
    try:
        accept(t,replay,old,'a',direction)
        result=accept(t,replay,new,'b',direction,bounds)
        assert result.accepted and result.delta_px==(-205 if direction=='up' else 205)
        retry=accept(t,replay,new,'retry',direction,bounds)
        assert retry.accepted and retry.delta_px==0 and offset(t,replay)==result.delta_px
    finally:
        if not replay:t.close()
    t=tracker(replay);wrong='down' if direction=='up' else 'up'
    try:
        accept(t,replay,old,'a',wrong)
        result=accept(t,replay,new,'b',wrong)
        assert not result.accepted and offset(t,replay)==0 and t.reference_id=='a'
    finally:
        if not replay:t.close()


def test_replay_rebuilds_notice_surface_masks_from_pixels_instead_of_trusting_legacy_ink_only(tmp_path):
    canvas=world();source=tmp_path/'source';archive=Archive(source,{'kind':'live_debug','capture_config':{'direction':'up'}})
    for i,image in enumerate([notice(canvas[300:963],3),notice(canvas[95:758],4)]):
        action={'kind':'initial' if i==0 else 'scroll','events':i*3,'event_units':-120,
                'stability':{'dynamic_masks':[],'overlay_masks':[Rect(779,636,83,12).to_dict()]}}
        if i:action['motion_bounds']={'minimum':-265,'maximum':0}
        f=archive.capture(image,action=action,truth_px=-205*i)
        archive.evaluate(f,AlignmentResult(True,'source',delta_px=-205*i),reference_id=None,cumulative_px=-205*i)
    archive.finish('complete');original=(source/'frames.jsonl').read_bytes()
    report=run_replay(source,tmp_path/'replay')
    assert report['accepted']==2 and report['final_scroll_offset_px']==-205 and report['max_abs_cumulative_error_px']==0
    assert (source/'frames.jsonl').read_bytes()==original


ROOT=Path('runs/pointer-20261122-102718-aa4ab7/capture')
@pytest.mark.skipif(not (ROOT/'frames/001537.png').exists(),reason='Private opaque-notice pair unavailable')
@pytest.mark.parametrize('replay',[False,True])
def test_private_notice_occlusion_tail_passes_from_legacy_masks_without_source_rewrites(replay):
    frames=read_jsonl(ROOT/'frames.jsonl');decisions=read_jsonl(ROOT/'alignments.jsonl');t=tracker(replay)
    try:
        for i in range(1537,1542):
            s=frames[i]['action']['stability'];masks=tuple(Rect(**b) for b in s.get('dynamic_masks',[])+s.get('overlay_masks',[])+decisions[i].get('motion_overlay_masks',[]))
            result=accept(t,replay,read_image(ROOT/'frames'/f'{i:06d}.png'),str(i),'up',frames[i]['action'].get('motion_bounds'),masks)
            assert result.accepted and result.delta_px==(-205 if i==1538 else 0)
            if i==1538:assert result.validation_coverage==1
        assert offset(t,replay)==-205
    finally:
        if not replay:t.close()
    assert read_jsonl(ROOT/'alignments.jsonl')==decisions
