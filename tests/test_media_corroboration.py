"""Narrow external evidence may propose motion; media may only corroborate it."""
from pathlib import Path
import numpy as np
import pytest
from welog.models import AlignmentConfig,Rect
from welog.alignment import OverlapAligner
from welog.frame_pool import ImageBudget
from welog.motion import MotionTracker
from welog.replay import CoordinateTracker
from welog.edge_layout import layout_media_regions,without_media,estimate_edge_layout,propose_bounded_static_layout,corroborate_bounded_surface
from welog.imageio import gray,read_image
from welog.archive import read_jsonl
from welog.scroll import ScrollCalibration


def pair(photo=True,avatar=True,photo_delta=201):
    old=np.full((663,880,3),250,np.uint8);new=old.copy();rng=np.random.default_rng(1061)
    if photo:
        pixels=rng.integers(0,180,(247,139,3),dtype=np.uint8);pixels[:,:,2]=220
        old[250:497,680:819]=pixels
        start=250+photo_delta;end=min(663,start+247)
        new[start:end,680:819]=pixels[:end-start]
    if avatar:
        texture=np.random.default_rng(1062).integers(0,256,(36,36,3),dtype=np.uint8)
        old[250:286,834:870]=texture;new[451:487,834:870]=texture
    return old,new


def static_valid(image):return without_media(np.ones(image.shape[:2],bool),layout_media_regions(image))


def test_unique_narrow_proposal_is_not_a_confirmed_coordinate_without_broad_corroboration():
    old,new=pair();a,b=gray(old),gray(new);v,w=static_valid(old),static_valid(new);bounds={'minimum':-260,'maximum':0}
    assert not estimate_edge_layout(a,b,v,w,OverlapAligner(),'up',bounds).accepted
    proposal=propose_bounded_static_layout(a,b,v,w,OverlapAligner(),'up',bounds)
    assert not proposal.accepted and proposal.reason=='bounded_static_proposal' and proposal.delta_px==-201 and proposal.inliers>=6
    full=np.ones(a.shape,bool);confirmed=corroborate_bounded_surface(a,b,full,full,OverlapAligner(),'up',bounds)
    assert confirmed.accepted and confirmed.delta_px==proposal.delta_px and confirmed.validation_coverage==1


@pytest.mark.parametrize('replay',[False,True])
@pytest.mark.parametrize('direction',['up','down'])
def test_unique_external_and_broad_surface_evidence_must_agree_without_calibration_feedback(replay,direction):
    old,new=pair()
    if direction=='down':old,new=new,old
    sign=-1 if direction=='up' else 1
    cal=ScrollCalibration()
    for _ in range(3):cal.observe(sign*201,3,'text')
    bounds=cal.batch_bounds(3,direction,663);rates=list(cal.rates)
    t=CoordinateTracker(OverlapAligner()) if replay else MotionTracker(ImageBudget(),AlignmentConfig())
    try:
        if replay:
            t.accept(old,'a',direction=direction,media_risk=True)
            result,_=t.accept(new,'b',direction=direction,motion_bounds=bounds,media_risk=True);method='bounded_layout'
        else:
            t.accept(old,'a',[],direction,media_risk=True)
            result,_,method=t.accept(new,'b',[],direction,motion_bounds=bounds,media_risk=True)
        assert result.accepted and result.delta_px==sign*201 and result.reason=='bounded_media_corroborated'
        assert (t.scroll_offset_px if replay else t.offset)==sign*201
        cal.observe(result.delta_px,3,method);assert list(cal.rates)==rates
        if replay:retry,_=t.accept(new,'retry',direction=direction,motion_bounds=bounds,media_risk=True)
        else:retry,_,_=t.accept(new,'retry',[],direction,motion_bounds=bounds,media_risk=True)
        assert retry.accepted and retry.delta_px==0
    finally:
        if not replay:t.close()


@pytest.mark.parametrize('photo,avatar',[(False,True),(True,False)])
def test_neither_narrow_static_evidence_nor_media_can_confirm_motion_alone(photo,avatar):
    old,new=pair(photo,avatar);t=MotionTracker(ImageBudget(),AlignmentConfig())
    try:
        t.accept(old,'a',[],'up',media_risk=True)
        result,_,_=t.accept(new,'b',[],'up',motion_bounds={'minimum':-260,'maximum':0},media_risk=True)
        assert not result.accepted and t.offset==0 and t.reference_id=='a'
    finally:t.close()


def test_broad_media_cannot_choose_between_ambiguous_static_avatar_translations(monkeypatch):
    from welog import edge_layout
    rng=np.random.default_rng(1063);world=np.full((1900,880,3),250,np.uint8);avatar=rng.integers(0,256,(36,36,3),dtype=np.uint8)
    for y in range(40,1750,152):
        world[y:y+36,834:870]=avatar
        pixels=rng.integers(0,180,(132,139,3),dtype=np.uint8);pixels[:,:,2]=220
        world[y:y+132,680:819]=pixels
    def forbidden(*args,**kwargs):raise AssertionError('Media must not disambiguate external aliases')
    monkeypatch.setattr(edge_layout,'corroborate_bounded_surface',forbidden)
    t=MotionTracker(ImageBudget(),AlignmentConfig())
    try:
        t.accept(world[600:1263],'a',[],'up',media_risk=True)
        result,_,_=t.accept(world[563:1226],'b',[],'up',motion_bounds={'minimum':-260,'maximum':0},media_risk=True)
        assert not result.accepted and t.offset==0 and t.reference_id=='a'
    finally:t.close()


def test_surface_support_must_be_broad_even_when_a_narrow_proposal_is_unique():
    old,new=pair(photo=False);v=np.ones(old.shape[:2],bool)
    result=corroborate_bounded_surface(gray(old),gray(new),v,v,OverlapAligner(),'up',{'minimum':-260,'maximum':0})
    assert not result.accepted


def test_changed_or_disagreeing_media_cannot_override_the_external_proposal():
    old,new=pair(photo_delta=151);bounds={'minimum':-260,'maximum':0};full=np.ones(old.shape[:2],bool)
    proposal=propose_bounded_static_layout(gray(old),gray(new),static_valid(old),static_valid(new),OverlapAligner(),'up',bounds)
    assert proposal.reason=='bounded_static_proposal' and proposal.delta_px==-201
    surface=corroborate_bounded_surface(gray(old),gray(new),full,full,OverlapAligner(),'up',bounds)
    assert surface.accepted and surface.delta_px==-151
    t=MotionTracker(ImageBudget(),AlignmentConfig())
    try:
        t.accept(old,'a',[],'up',media_risk=True)
        result,_,_=t.accept(new,'b',[],'up',motion_bounds=bounds,media_risk=True)
        assert not result.accepted and t.offset==0 and t.reference_id=='a'
    finally:t.close()


def test_temporally_excluded_media_cannot_supply_broad_confirmation():
    old,new=pair();t=MotionTracker(ImageBudget(),AlignmentConfig())
    try:
        t.accept(old,'a',[],'up',(Rect(680,250,139,247),),media_risk=True)
        result,_,_=t.accept(new,'b',[],'up',(Rect(680,451,139,212),),{'minimum':-260,'maximum':0},media_risk=True)
        assert not result.accepted and t.offset==0
    finally:t.close()


@pytest.mark.parametrize('direction,bounds',[('up',None),('down',{'minimum':0,'maximum':260})])
def test_missing_bounds_or_wrong_direction_cannot_be_rescued_by_media(direction,bounds):
    old,new=pair();t=MotionTracker(ImageBudget(),AlignmentConfig())
    try:
        t.accept(old,'a',[],direction,media_risk=True)
        result,_,_=t.accept(new,'b',[],direction,motion_bounds=bounds,media_risk=True)
        assert not result.accepted and t.offset==0
    finally:t.close()


ROOT=Path('runs/pointer-20261123-004738-690b9c/capture')
@pytest.mark.skipif(not (ROOT/'frames/001469.png').exists(),reason='Private narrow-avatar/photo pair unavailable')
@pytest.mark.parametrize('replay',[False,True])
def test_private_photo_tail_is_confirmed_without_rewriting_source_or_training_bounds(replay):
    f=read_jsonl(ROOT/'frames.jsonl');ds=read_jsonl(ROOT/'alignments.jsonl');t=CoordinateTracker(OverlapAligner()) if replay else MotionTracker(ImageBudget(),AlignmentConfig())
    try:
        for i in range(1469,1474):
            s=f[i]['action']['stability'];m=tuple(Rect(**b) for b in s.get('dynamic_masks',[])+s.get('overlay_masks',[])+ds[i].get('motion_overlay_masks',[]));im=read_image(ROOT/'frames'/f'{i:06d}.png');bounds=f[i]['action'].get('motion_bounds')
            if replay:result,_=t.accept(im,str(i),direction='up',dynamic_masks=m,motion_bounds=bounds,media_risk=True)
            else:result,_,method=t.accept(im,str(i),[],'up',m,bounds,media_risk=True)
            assert result.accepted and result.delta_px==(-201 if i==1470 else 0)
            if i==1470:assert result.reason=='bounded_media_corroborated'
        assert (t.scroll_offset_px if replay else t.offset)==-201
    finally:
        if not replay:t.close()
    assert read_jsonl(ROOT/'alignments.jsonl')==ds
