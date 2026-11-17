from pathlib import Path
import json
import cv2
import numpy as np
import pytest
from welog.edge_layout import estimate_edge_layout
from welog.models import AlignmentConfig,Rect
from welog.alignment import OverlapAligner
from welog.imageio import gray,read_image


def sparse_pair():
    rng=np.random.default_rng(93);world=np.full((1000,875,3),250,np.uint8)
    right=rng.integers(0,256,(36,36,3),dtype=np.uint8)
    left=rng.integers(0,256,(36,36,3),dtype=np.uint8)
    for y in [45,197,349,501]:world[y:y+36,833:869]=right
    for y in [654,766]:world[y:y+36,20:56]=left
    images=[];valid=[]
    for offset in [201,0]:
        image=world[offset:offset+672].copy();mask=np.ones((672,875),bool)
        for start in [64,216,368,520]:
            a,b=max(0,start-offset),min(672,start+132-offset)
            if b>a:
                image[a:b,687:819]=rng.integers(0,256,(b-a,132,3),dtype=np.uint8)
                mask[a:b,687:819]=False
        images.append(gray(image));valid.append(mask)
    return images,valid


def test_clipped_anchor_proposes_motion_but_distributed_layout_confirms_it():
    images,valid=sparse_pair()
    result=estimate_edge_layout(*images,*valid,OverlapAligner(), 'up')
    assert result.accepted and result.delta_px==-201 and result.inliers>=6
    assert result.validation_coverage>=.75


def test_wrong_direction_and_unrelated_content_with_fixed_border_are_rejected():
    images,valid=sparse_pair()
    assert not estimate_edge_layout(*images,*valid,OverlapAligner(),'down').accepted
    rng=np.random.default_rng(94)
    old=rng.integers(0,256,(672,875),dtype=np.uint8);new=rng.integers(0,256,old.shape,dtype=np.uint8)
    old[:3]=new[:3]=0;old[-3:]=new[-3:]=0
    new[600:660,600:850]=old[600:660,600:850]
    full=np.ones(old.shape,bool)
    assert not estimate_edge_layout(old,new,full,full,OverlapAligner(),'up').accepted


def test_periodic_layout_remains_ambiguous_even_with_many_small_matches():
    rng=np.random.default_rng(95);tile=np.full((152,875),250,np.uint8)
    avatar=rng.integers(0,256,(36,36),dtype=np.uint8)
    tile[40:76,20:56]=avatar;tile[40:76,833:869]=avatar
    image=np.tile(tile,(5,1))[:672];valid=np.ones(image.shape,bool)
    result=estimate_edge_layout(image,image.copy(),valid,valid,OverlapAligner(),'up')
    assert not result.accepted


def test_scaling_cannot_be_explained_as_page_translation():
    images,valid=sparse_pair()
    enlarged=cv2.resize(images[0],None,fx=1.03,fy=1.03)[:672,:875]
    assert not estimate_edge_layout(images[0],enlarged,valid[0],np.ones(enlarged.shape,bool),OverlapAligner(),'both').accepted


ROOT=Path('runs/pointer-20261117-165048-026bc5/capture')
@pytest.mark.skipif(not ROOT.exists(),reason='Private diagnostic pair unavailable')
def test_real_pair_has_independent_spatial_evidence_without_rewriting_capture():
    from welog.motion import MotionTracker
    from welog.frame_pool import ImageBudget
    frames=[json.loads(line) for line in (ROOT/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    tracker=MotionTracker(ImageBudget(),AlignmentConfig())
    try:
        for i in [69,70]:
            frame=frames[i]
            result,_,method=tracker.accept(read_image(ROOT/frame['image']),str(i),[],'up',tuple(Rect(**b) for b in frame['action']['stability']['dynamic_masks']))
            assert result.accepted
        assert method=='edge_layout' and result.delta_px==-201
    finally:tracker.close()


@pytest.mark.skipif(not ROOT.exists(),reason='Private diagnostic pair unavailable')
@pytest.mark.parametrize('freeze_old_zoom',[False,True])
def test_controlled_animation_pair_passes_stability_and_capture_pipeline(tmp_path,freeze_old_zoom):
    from welog.controller import run_capture
    from welog.recognition import detect_bubbles,EmptyOcr
    from welog.models import CaptureConfig
    images=[read_image(ROOT/'frames'/f'{i:06d}.png') for i in [69,70]]
    frames=[json.loads(line) for line in (ROOT/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    boxes=[[(box['x'],box['y'],box['x']+box['width'],box['y']+box['height']) for box in frames[i]['action']['stability']['dynamic_masks']] for i in [69,70]]
    if freeze_old_zoom:boxes[0]=[b.box for b in detect_bubbles(images[0]) if b.kind=='image']
    class Source:
        stage=0
        def __init__(self):self.rng=np.random.default_rng(96)
        def read(self):
            image=images[self.stage].copy()
            for x1,y1,x2,y2 in boxes[self.stage]:
                # The temporal tiles include padding; keep generated media inside
                # the observed sticker lane so synthetic pixels don't engulf avatars.
                x1,x2=max(687,x1),min(819,x2)
                image[y1:y2,x1:x2]=self.rng.integers(0,256,(y2-y1,x2-x1,3),dtype=np.uint8)
            if self.stage:image[3:16,740:766]=(70,80,int(self.rng.integers(150,250)))
            return image
        def scroll(self,units):self.stage=1
        def close(self):pass
    result=run_capture(Source(),tmp_path/'capture',config=CaptureConfig(max_frames=2,direction='up',
        sample_interval_s=.001,stable_samples=3,stable_timeout_s=3,settle_delay_s=0),ocr_backend=EmptyOcr())
    assert result['accepted']==2 and result['rejected']==0 and result['final_scroll_offset_px']==-201
