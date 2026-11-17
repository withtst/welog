from dataclasses import replace
from pathlib import Path
import json
import cv2
import numpy as np
import pytest

from welog.alignment import OverlapAligner
from welog.demo import make_scene
from welog.frame_pool import ImageBudget
from welog.imageio import gray,read_image
from welog.models import AlignmentConfig,Rect
from welog.motion import MotionTracker


def images():
    world=make_scene(1500,640,seed=321)
    old=world[300:1020].copy();new=world[100:820].copy()
    rng=np.random.default_rng(13)
    overlay=rng.integers(0,256,(110,240,3),dtype=np.uint8)
    old[600:710,390:630]=overlay;new[600:710,390:630]=overlay
    return old,new


def feature_result(old,new,direction='up'):
    config=AlignmentConfig();aligner=OverlapAligner(config)
    tracker=MotionTracker(ImageBudget(),config)
    return tracker.features(gray(old),gray(new),np.ones(old.shape[:2],bool),aligner,direction)


def test_fixed_overlay_and_nonoverlap_features_do_not_dilute_page_motion():
    old,new=images();result=feature_result(old,new)
    assert result.accepted and result.delta_px==-200
    assert result.feature_matches_overlap<result.feature_matches_total
    assert result.feature_inlier_ratio>=.5 and result.inliers>=12
    assert result.validation_coverage>=.75


def test_geometry_replay_uses_the_same_feature_verification():
    from welog.replay import CoordinateTracker
    old,new=images()
    tracker=CoordinateTracker(OverlapAligner(AlignmentConfig(min_matches=24,max_patches=24)))
    tracker.accept(old,'a',direction='up')
    result,reference=tracker.accept(new,'b',direction='up')
    assert result.accepted and result.delta_px==-200 and reference=='a'
    assert tracker.scroll_offset_px==-200


def test_size_change_does_not_update_reference_or_invoke_feature_matching():
    from welog.replay import CoordinateTracker
    old,new=images();tracker=MotionTracker(ImageBudget(),AlignmentConfig())
    try:
        tracker.accept(old,'a',[],'up')
        result,reference,method=tracker.accept(new[:-10],'b',[],'up')
        assert not result.accepted and result.reason=='layout_size_changed' and tracker.reference_id=='a' and tracker.offset==0
    finally:tracker.close()
    replay=CoordinateTracker(OverlapAligner())
    replay.accept(old,'a',direction='up')
    result,_=replay.accept(new[:-10],'b',direction='up')
    assert not result.accepted and replay.reference_id=='a' and replay.scroll_offset_px==0


def test_opposite_direction_is_not_rescued_by_stationary_overlay():
    old,new=images()
    assert not feature_result(old,new,'down').accepted


def test_overlay_alone_cannot_align_unrelated_page_content():
    rng=np.random.default_rng(14)
    old=rng.integers(0,256,(720,640,3),dtype=np.uint8)
    new=rng.integers(0,256,old.shape,dtype=np.uint8)
    new[600:710,390:630]=old[600:710,390:630]
    assert not feature_result(old,new).accepted


def test_unread_overlay_masks_keep_real_prose_and_remove_fixed_overlay_features():
    from welog.text_rules import notice_region_masks
    from welog.recognition import TextLine
    masks=notice_region_masks([TextLine('2条新消息',(500,640,590,660)),TextLine('正文2条新消息继续',(20,50,260,80))],640,720)
    assert len(masks)==1 and masks[0].x==476 and masks[0].bottom==684
    assert notice_region_masks([TextLine('99条新消息',(1,1,30,12))],40,20)[0]==Rect(0,0,40,20)


def test_scaling_and_repeated_texture_are_rejected():
    old,new=images()
    scaled=cv2.resize(old,None,fx=1.03,fy=1.03)[:720,:640]
    assert not feature_result(old,scaled,'both').accepted
    rng=np.random.default_rng(15);tile=rng.integers(0,256,(40,40,3),dtype=np.uint8)
    periodic=np.tile(tile,(18,16,1))
    assert not feature_result(periodic,np.roll(periodic,80,axis=0)).accepted


ROOT=Path('runs/pointer-20261117-152112-b8bfa7/capture')
@pytest.mark.skipif(not ROOT.exists(),reason='Private offline fixture unavailable')
def test_real_failed_transition_and_retries_are_recovered_without_changing_source():
    frames=[json.loads(line) for line in (ROOT/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    tracker=MotionTracker(ImageBudget(),AlignmentConfig())
    try:
        for index in range(21,26):
            frame=frames[index]
            masks=tuple(Rect(**box) for box in frame['action']['stability']['dynamic_masks'])
            result,reference,method=tracker.accept(read_image(ROOT/frame['image']),frame['frame_id'],[],'up',masks)
            assert result.accepted
            if index==22:
                assert result.delta_px==-201 and reference=='000021' and method=='static_features'
                assert result.feature_inlier_ratio>=.5 and result.feature_matches_overlap<result.feature_matches_total
            if index>22:assert result.delta_px==0
        assert tracker.offset==-201
    finally:tracker.close()
    assert tracker.budget.used_bytes==0
