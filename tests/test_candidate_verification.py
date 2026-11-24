"""Many local proposals still require complete, unambiguous verification."""
from pathlib import Path
import cv2
import numpy as np
import pytest
from welog.alignment import OverlapAligner
from welog.archive import read_jsonl
from welog.edge_layout import estimate_edge_layout
from welog.frame_pool import ImageBudget
from welog.imageio import gray,read_image
from welog.models import AlignmentConfig,Rect
from welog.motion import MotionTracker
from welog.replay import CoordinateTracker


def crowded_prose(seed):
    rng=np.random.default_rng(seed);world=np.full((1500,880,3),250,np.uint8)
    avatar=rng.integers(0,256,(36,36,3),dtype=np.uint8);y=25
    for i in range(23):
        world[y:y+36,20:56]=avatar
        cv2.rectangle(world,(70,y),(int(rng.integers(240,500)),y+36),(238,238,238),-1)
        cv2.putText(world,'common sample '+str(i),(79,y+25),cv2.FONT_HERSHEY_SIMPLEX,.5,(25,25,25),1)
        y+=int(rng.integers(48,75))
    return world


@pytest.mark.parametrize('seed',[2,5,8])
@pytest.mark.parametrize('direction',['up','down'])
def test_many_common_glyph_proposals_do_not_hide_unique_page_translation(seed,direction):
    world=crowded_prose(seed);old,new=world[400:1063],world[100:763]
    if direction=='down':old,new=new,old
    bounds={'minimum':-385,'maximum':0} if direction=='up' else {'minimum':0,'maximum':385}
    tracker=MotionTracker(ImageBudget(),AlignmentConfig())
    try:
        tracker.accept(old,'a',[],direction,media_risk=True)
        result,_,method=tracker.accept(new,'b',[],direction,motion_bounds=bounds,media_risk=True)
        assert result.accepted and result.delta_px==(-300 if direction=='up' else 300)
        assert result.candidates>24 and result.validation_error==0 and method=='bounded_layout'
        assert tracker.offset==result.delta_px
        repeated,_,_=tracker.accept(new,'retry',[],direction,motion_bounds=bounds,media_risk=True)
        assert repeated.accepted and repeated.delta_px==0 and tracker.offset==result.delta_px
    finally:tracker.close()
    assert tracker.budget.used_bytes==0


def test_verifying_more_proposals_does_not_choose_one_of_two_valid_translations():
    tile=crowded_prose(2)[:350]
    image=gray(np.tile(tile,(2,1,1))[:663]);valid=np.ones(image.shape,bool)
    result=estimate_edge_layout(image,image.copy(),valid,valid,OverlapAligner(),'up',{'minimum':-385,'maximum':0})
    assert not result.accepted and result.reason=='ambiguous_edge_layout' and result.candidates>1


@pytest.mark.parametrize('direction', ['up', 'down'])
def test_distinct_bubble_excludes_repeated_bubble_aliases(direction):
    world = np.full((900, 640), 247, np.uint8)
    avatar = np.random.default_rng(17).integers(0, 256, (36, 36), dtype=np.uint8)
    for y in range(22, 864, 56):
        world[y:y+36, 20:56] = avatar
        cv2.rectangle(world, (65, y), (149, y+35), 235, -1)
        cv2.putText(world, 'one' if y == 302 else 'same', (76, y+25),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, 25, 1)
    old, current = world[235:898], world[:663]
    bounds = {'minimum': -350, 'maximum': 0}
    delta = -235
    if direction == 'down':
        old, current = current, old
        bounds = {'minimum': 0, 'maximum': 350}
        delta = 235
    current = current.copy()
    current[420:424, 430:470] -= 8
    valid = np.ones(old.shape, bool)
    result = estimate_edge_layout(old, current, valid, valid, OverlapAligner(), direction, bounds)
    assert result.accepted and result.delta_px == delta
    assert result.reason == 'edge_layout_aligned_static_overlap'


ROOT=Path('runs/pointer-20261121-001917-39602a/capture')
@pytest.mark.skipif(not (ROOT/'frames/000649.png').exists(),reason='Private candidate-overflow pair unavailable')
@pytest.mark.parametrize('replay',[False,True])
def test_retained_candidate_overflow_pair_and_stationary_retries_preserve_coordinates(replay):
    frames=read_jsonl(ROOT/'frames.jsonl');decisions=read_jsonl(ROOT/'alignments.jsonl')
    tracker=CoordinateTracker(OverlapAligner()) if replay else MotionTracker(ImageBudget(),AlignmentConfig())
    try:
        for i in range(649,654):
            s=frames[i]['action']['stability']
            masks=tuple(Rect(**m) for m in s.get('dynamic_masks',[])+s.get('overlay_masks',[])+decisions[i].get('motion_overlay_masks',[]))
            image=read_image(ROOT/'frames'/f'{i:06d}.png');bounds=frames[i]['action'].get('motion_bounds')
            if replay:result,_=tracker.accept(image,str(i),direction='up',dynamic_masks=masks,motion_bounds=bounds,media_risk=True)
            else:result,_,_=tracker.accept(image,str(i),[],'up',masks,bounds,media_risk=True)
            assert result.accepted and result.delta_px==(-300 if i==650 else 0)
            if i==650:assert result.candidates>24
        assert (tracker.scroll_offset_px if replay else tracker.offset)==-300
    finally:
        if not replay:tracker.close()
    assert read_jsonl(ROOT/'alignments.jsonl')==decisions
