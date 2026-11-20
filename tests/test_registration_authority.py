"""Visual proposals cannot bypass command bounds or borrow animated identities."""
from pathlib import Path
import cv2
import numpy as np
import pytest
from welog.alignment import OverlapAligner
from welog.archive import read_jsonl
from welog.frame_pool import ImageBudget
from welog.models import AlignmentConfig,AlignmentResult,Rect
from welog.motion import MotionTracker
from welog.replay import CoordinateTracker
from welog.recognition import TextLine
from welog.scroll import ScrollCalibration


def repeated_media_pair():
    world=np.full((1600,880,3),250,np.uint8);rng=np.random.default_rng(941)
    avatar=rng.integers(0,256,(36,36,3),dtype=np.uint8)
    for y in range(45,1450,152):
        world[y:y+36,20:56]=avatar
        world[y:y+132,70:202]=(20,90,210)
        cv2.putText(world,'same asset',(76,y+25),cv2.FONT_HERSHEY_SIMPLEX,.4,(240,240,240),1)
    return world[400:1063].copy(),world[363:1026].copy()


@pytest.mark.parametrize('bounded',[False,True])
@pytest.mark.parametrize('direction',['up','down'])
def test_risky_media_cannot_accept_a_feature_identity_from_another_copy(monkeypatch,bounded,direction):
    old,new=repeated_media_pair();tracker=MotionTracker(ImageBudget(),AlignmentConfig());calls=[]
    if direction=='down':old,new=new,old
    sign=-1 if direction=='up' else 1
    bounds={'minimum':-122,'maximum':0} if direction=='up' else {'minimum':0,'maximum':122}
    real_features=tracker.features
    def static_only(a,b,*args,**kwargs):
        calls.append(True)
        assert kwargs.get('require_broad')
        # Caption/graphic bytes are removed, including pixels that a SIFT
        # descriptor could otherwise read outside its keypoint mask.
        assert np.all(a[:,70:202]==250) and np.all(b[:,70:202]==250)
        return real_features(a,b,*args,**kwargs)
    monkeypatch.setattr(tracker,'features',static_only)
    try:
        tracker.accept(old,'a',[],direction,media_risk=True)
        result,_,method=tracker.accept(new,'b',[],direction,motion_bounds=bounds if bounded else None,media_risk=True)
        if bounded:assert not calls and result.accepted and result.delta_px==sign*37 and method=='bounded_layout' and tracker.offset==sign*37
        else:assert not result.accepted and tracker.offset==0 and tracker.reference_id=='a'
    finally:tracker.close()
    assert tracker.budget.used_bytes==0


@pytest.mark.parametrize('proposal',['template','static_features'])
@pytest.mark.parametrize('replay',[False,True])
def test_any_out_of_envelope_visual_proposal_is_rejected_without_advancing(monkeypatch,proposal,replay):
    old,new=repeated_media_pair()
    monkeypatch.setattr(OverlapAligner,'estimate',lambda *a,**k:AlignmentResult(proposal=='template','proposal',delta_px=-493 if proposal=='template' else None))
    tracker=CoordinateTracker(OverlapAligner()) if replay else MotionTracker(ImageBudget(),AlignmentConfig())
    owner=tracker.feature_verifier if replay else tracker
    monkeypatch.setattr(owner,'features',lambda *a,**k:AlignmentResult(True,'proposal',delta_px=-493))
    if replay:
        tracker.accept(old,'a',direction='up',media_risk=False)
        result,_=tracker.accept(new,'b',direction='up',motion_bounds={'minimum':-122,'maximum':0},media_risk=False)
        assert tracker.scroll_offset_px==0
    else:
        tracker.accept(old,'a',[],'up',media_risk=False)
        result,_,_=tracker.accept(new,'b',[],'up',motion_bounds={'minimum':-122,'maximum':0},media_risk=False)
        assert tracker.offset==0;tracker.close()
    assert not result.accepted and result.reason=='motion_outside_calibrated_bounds' and tracker.reference_id=='a'


def test_even_native_text_conflict_cannot_be_silently_snapped_inside_the_envelope():
    world=np.full((1200,600,3),250,np.uint8)
    for y,text in [(400,'alpha'),(500,'beta')]:
        world[y:y+50,60:280]=238
        cv2.putText(world,text,(70,y+30),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1)
    a=[TextLine(text,(70,y-300+10,220,y-300+40),.99) for y,text in [(400,'alpha'),(500,'beta')]]
    b=[TextLine(text,(70,y-111+10,220,y-111+40),.99) for y,text in [(400,'alpha'),(500,'beta')]]
    tracker=MotionTracker(ImageBudget(),AlignmentConfig())
    try:
        tracker.accept(world[300:1020],'a',a,'up')
        result,_,_=tracker.accept(world[111:831],'b',b,'up',motion_bounds={'minimum':-122,'maximum':0})
        assert not result.accepted and result.reason=='motion_outside_calibrated_bounds'
        assert tracker.offset==0 and tracker.reference_id=='a'
    finally:tracker.close()


ROOT=Path('runs/pointer-20261120-231009-67f850/capture')
@pytest.mark.skipif(not ROOT.exists(),reason='Private repeated-media chain unavailable')
def test_retained_chain_keeps_verified_envelope_and_excludes_original_absolute_coordinates():
    from welog.imageio import read_image
    frames=read_jsonl(ROOT/'frames.jsonl');decisions=read_jsonl(ROOT/'alignments.jsonl')
    calibration=ScrollCalibration()
    for f,row in zip(frames[:97],decisions[:97]):
        if row['alignment']['accepted']:calibration.observe(row['alignment']['delta_px'],f['action'].get('events',0),row['motion_method'])
    bounds=calibration.bounds('up',663)
    assert bounds['minimum']==-122
    rates=list(calibration.rates);indices=[97,98,99,100,101,102,103,104,106,107,108,109,110]
    expected={98:-37,100:-37,102:-37,106:-37,107:-37}
    tracker=MotionTracker(ImageBudget(),AlignmentConfig());previous=97
    try:
        for i in indices:
            f=frames[i];masks=tuple(Rect(**b) for b in f['action']['stability']['dynamic_masks']+f['action']['stability'].get('overlay_masks',[])+decisions[i].get('motion_overlay_masks',[]))
            result,_,method=tracker.accept(read_image(ROOT/'frames'/f'{i:06d}.png'),str(i),[],'up',masks,bounds,media_risk=True)
            assert result.accepted and result.delta_px==expected.get(i,0)
            events=sum(frames[j]['action'].get('events',0) for j in range(previous+1,i+1))
            calibration.observe(result.delta_px,events,method);previous=i
            assert list(calibration.rates)==rates and calibration.bounds('up',663)['minimum']==-122
        assert tracker.offset==-185
    finally:tracker.close()
    assert read_jsonl(ROOT/'alignments.jsonl')==decisions and tracker.budget.used_bytes==0

@pytest.mark.parametrize('limit,accepted',[(64,True),(420,False),(None,False)])
def test_two_static_avatar_rows_need_a_range_smaller_than_their_spacing(limit,accepted):
    world=np.full((1000,880,3),250,np.uint8)
    avatar=np.random.default_rng(4).integers(0,256,(36,36,3),dtype=np.uint8)
    for y in [180,480]:
        world[y:y+36,20:56]=avatar
        world[y:y+200,70:202]=(20,90,210)
    tracker=MotionTracker(ImageBudget(),AlignmentConfig())
    try:
        tracker.accept(world[100:763],'a',[],'up',media_risk=True)
        result,_,_=tracker.accept(world[63:726],'b',[],'up',motion_bounds={'minimum':-limit,'maximum':0} if limit else None,media_risk=True)
        assert result.accepted==accepted
        assert tracker.offset==(-37 if accepted else 0)
    finally:tracker.close()


def test_replay_preserves_verified_command_bounds_without_learning_legacy_aliases(tmp_path):
    from welog.archive import Archive
    from welog.replay import run_replay
    # The periodic page can be shifted without introducing a novel identity.
    world=np.full((1600,880,3),250,np.uint8)
    avatar=np.random.default_rng(941).integers(0,256,(36,36,3),dtype=np.uint8)
    for y in range(45,1450,152):
        world[y:y+36,20:56]=avatar
        world[y:y+132,70:202]=(20,90,210)
    source=tmp_path/'source';archive=Archive(source,{'kind':'live_debug','capture_config':{'direction':'up'}})
    for i,start in enumerate([400,363,326,289]):
        action={'kind':'scroll' if i else 'initial','events':1 if i else 0,'event_units':-120,
                'stability':{'dynamic_masks':[{'x':70,'y':0,'width':132,'height':663}]}}
        if i==1:action['motion_bounds']={'minimum':-122,'maximum':0}
        frame=archive.capture(world[start:start+663],action=action,truth_px=-37*i)
        archive.evaluate(frame,AlignmentResult(True,'legacy',delta_px=[0,-189,-37,-493][i]),reference_id=None,cumulative_px=[0,-189,-226,-719][i])
    archive.finish('complete')
    original=(source/'frames.jsonl').read_bytes()
    report=run_replay(source,tmp_path/'replay',allow_failures=True)
    assert report['accepted']==4 and report['final_scroll_offset_px']==-111
    assert report['max_abs_cumulative_error_px']==0
    derived=read_jsonl(tmp_path/'replay/frames.jsonl')
    assert [f['action'].get('motion_bounds_source') for f in derived[2:]]==['last_verified_same_command']*2
    assert (source/'frames.jsonl').read_bytes()==original
