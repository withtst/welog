"""A white media background must not become a text bubble when its content darkens."""
from pathlib import Path
import cv2
import numpy as np
import pytest
from welog.recognition import detect_bubbles
from welog.models import AlignmentConfig,Rect,CaptureConfig
from welog.motion import MotionTracker
from welog.frame_pool import ImageBudget
from welog.replay import CoordinateTracker
from welog.alignment import OverlapAligner
from welog.archive import read_jsonl
from welog.imageio import read_image


def white_photo(phase=0):
    panel=np.full((132,132,3),255,np.uint8)
    panel[15:115,12:120]=235
    panel[15:115,44:120]=(35+phase*20,90,165-phase*20)
    return panel


@pytest.mark.parametrize('x',[70,687])
@pytest.mark.parametrize('top',[100,-70,607])
def test_white_photo_background_is_identified_in_full_and_clipped_views(x,top):
    image=np.full((663,880,3),250,np.uint8);panel=white_photo()
    assert np.median(panel)<251
    a,b=max(0,top),min(663,top+132);image[a:b,x:x+132]=panel[a-top:b-top]
    bubbles=detect_bubbles(image)
    found=[m for m in bubbles if m.kind in {'image','unknown'} and m.box[0]<=x and m.box[2]>=x+132 and m.box[1]<=a and m.box[3]>=b]
    assert found
    assert not [m for m in bubbles if m.kind=='text' and m.box[0]>=x and m.box[2]<=x+132 and m.box[1]>=a and m.box[3]<=b]


@pytest.mark.parametrize('top',[100,-15,635])
def test_gray_text_panels_with_colored_inline_emoji_remain_text(top):
    image=np.full((663,880,3),250,np.uint8);panel=np.full((55,240,3),238,np.uint8)
    cv2.putText(panel,'real text',(10,35),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1)
    panel[10:30,200:220]=(40,120,200)
    a,b=max(0,top),min(663,top+55);image[a:b,70:310]=panel[a-top:b-top]
    found=detect_bubbles(image)
    assert len(found)==1 and found[0].kind=='text'


def test_photo_contents_are_masked_before_ocr_and_not_exported_as_chat_text(tmp_path):
    from welog.pipeline import FrameProcessor
    image=np.full((663,880,3),250,np.uint8);image[100:232,687:819]=white_photo()
    class Backend:
        calls=0
        def recognize(self,clean):
            self.calls+=1
            assert (clean[100:232,687:819]==250).all()
            return []
    backend=Backend();budget=ImageBudget();processor=FrameProcessor(tmp_path,budget,backend,AlignmentConfig(),CaptureConfig())
    try:
        result=processor.process(budget.copy(image),'a',(),'up')
        assert result['alignment']['accepted'] and backend.calls==1
        rows=list(processor.store.iter_messages())
        assert len(rows)==1 and rows[0]['type']=='image' and rows[0]['text']=='[图片]'
    finally:processor.finish({'status':'test'})
    assert budget.used_bytes==0 and not list(tmp_path.rglob('*.png'))


@pytest.mark.parametrize('direction',['up','down'])
def test_changed_white_photo_does_not_break_static_page_translation(direction):
    world=np.full((1500,880,3),250,np.uint8)
    for y in range(40,1400,85):
        cv2.putText(world,f'unique line {y}',(70,y),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1)
    old=world[400:1063].copy();new=world[97:760].copy()
    old[250:382,687:819]=white_photo(0);new[553:663,687:819]=white_photo(1)[:110]
    if direction=='down':old,new=new,old
    bounds={'minimum':-387,'maximum':0} if direction=='up' else {'minimum':0,'maximum':387}
    tracker=MotionTracker(ImageBudget(),AlignmentConfig())
    try:
        tracker.accept(old,'a',[],direction,media_risk=True)
        result,_,_=tracker.accept(new,'b',[],direction,motion_bounds=bounds,media_risk=True)
        assert result.accepted and result.delta_px==(-303 if direction=='up' else 303)
        repeated,_,_=tracker.accept(new,'retry',[],direction,motion_bounds=bounds,media_risk=True)
        assert repeated.accepted and repeated.delta_px==0 and tracker.offset==result.delta_px
    finally:tracker.close()


ROOT=Path('runs/pointer-20261121-110517-053ea2/capture')
@pytest.mark.skipif(not (ROOT/'frames/000666.png').exists(),reason='Private white-media failure pair unavailable')
@pytest.mark.parametrize('replay',[False,True])
def test_retained_white_media_failure_and_retries_preserve_relative_height(replay):
    frames=read_jsonl(ROOT/'frames.jsonl');decisions=read_jsonl(ROOT/'alignments.jsonl')
    tracker=CoordinateTracker(OverlapAligner()) if replay else MotionTracker(ImageBudget(),AlignmentConfig())
    try:
        for i in range(666,671):
            s=frames[i]['action']['stability'];masks=tuple(Rect(**b) for b in s.get('dynamic_masks',[])+s.get('overlay_masks',[])+decisions[i].get('motion_overlay_masks',[]))
            image=read_image(ROOT/'frames'/f'{i:06d}.png');bounds=frames[i]['action'].get('motion_bounds')
            if replay:result,_=tracker.accept(image,str(i),direction='up',dynamic_masks=masks,motion_bounds=bounds,media_risk=True)
            else:result,_,_=tracker.accept(image,str(i),[],'up',masks,bounds,media_risk=True)
            assert result.accepted and result.delta_px==(-303 if i==667 else 0)
        assert (tracker.scroll_offset_px if replay else tracker.offset)==-303
    finally:
        if not replay:tracker.close()
    assert read_jsonl(ROOT/'alignments.jsonl')==decisions
