"""A floating notice is a UI layer, never a message or page-motion anchor."""
from pathlib import Path
import cv2
import numpy as np
import pytest
from welog.archive import read_jsonl
from welog.models import AlignmentConfig,CaptureConfig,Rect
from welog.recognition import detect_bubbles
from welog.stability import LayoutStabilityDetector
from welog.text_rules import visual_notice_layers,suppress_visual_notice
from welog.controller import run_capture


def badge(image,count=12,top=False):
    y=8 if top else image.shape[0]-43
    image[y:y+42,764:874]=(image[y:y+42,764:874].astype(float)*.9+20).astype(np.uint8)
    cv2.rectangle(image,(772,y+5),(869,y+30),(250,250,250),-1)
    cv2.putText(image,f'{count} new msg',(779,y+25),cv2.FONT_HERSHEY_SIMPLEX,.42,(120,180,0),1)
    return image


def scene(phase,height=850,anchors=True):
    image=np.full((height,880,3),250,np.uint8);rng=np.random.default_rng(680+phase)
    for y in [65,217,369,521]:
        data=rng.integers(0,160,(132,132,3),dtype=np.uint8)
        data[:,:,1]=50+(phase%3)*30;data[:,:,2]=240
        image[y:y+132,687:819]=data
    if anchors:
        avatar=np.random.default_rng(681).integers(0,256,(36,36,3),dtype=np.uint8)
        for y in [65,217,369,521]:image[y:y+36,834:870]=avatar
        image[300:336,20:56]=np.random.default_rng(682).integers(0,256,(36,36,3),dtype=np.uint8)
    return image


def test_notice_touching_graphic_does_not_remove_the_graphic_or_modify_raw_pixels(monkeypatch):
    from welog import text_rules
    image=badge(scene(0)[:663]);original=image.copy()
    segment,evidence=visual_notice_layers(image)
    assert segment and evidence
    corrected=[b for b in detect_bubbles(image) if b.kind=='image']
    assert len(corrected)==4 and any(b.box[1]>=515 for b in corrected)
    with monkeypatch.context() as patch:
        patch.setattr(text_rules,'visual_notice_layers',lambda image:((),()))
        legacy=[b for b in detect_bubbles(image) if b.kind=='image']
    assert len(legacy)==3
    assert np.array_equal(image,original)


def test_notice_masks_keep_colored_bubble_and_black_text_and_ignore_non_ui_art():
    image=np.full((663,880,3),250,np.uint8)
    cv2.rectangle(image,(753,0),(824,12),(153,244,151),-1)
    cv2.putText(image,'ok',(754,10),cv2.FONT_HERSHEY_SIMPLEX,.3,(20,20,20),1)
    badge(image,top=True);segment,evidence=visual_notice_layers(image)
    assert segment and evidence
    cleaned=suppress_visual_notice(image,segment)
    visible=(image[:12,753:764].min(2)<210)
    assert np.array_equal(cleaned[:12,753:764][visible],image[:12,753:764][visible])
    plain=np.full_like(image,250)
    cv2.rectangle(plain,(700,590),(819,650),(153,244,151),-1)
    cv2.putText(plain,'link',(710,620),cv2.FONT_HERSHEY_SIMPLEX,.5,(140,180,0),1)
    assert visual_notice_layers(plain)==((),())


def test_overlay_count_changes_are_ignored_but_page_and_text_motion_are_not():
    detector=LayoutStabilityDetector(AlignmentConfig())
    result=[detector.observe(badge(scene(i)[:663],12+i)) for i in range(3)][-1]
    assert result.stable and result.overlay_masks and result.outside_media_pixels<=64 and result.media_supported_ratio>=.98,result.to_dict()
    detector=LayoutStabilityDetector(AlignmentConfig())
    for i in range(6):assert not detector.observe(badge(scene(i)[i:i+663],12+i)).stable
    detector=LayoutStabilityDetector(AlignmentConfig())
    for i in range(6):
        image=badge(scene(i)[:663],12+i)
        image[230:285,70:320]=238
        cv2.putText(image,'moving text',(80,255+i),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,20),1)
        assert not detector.observe(image).stable
    detector=LayoutStabilityDetector(AlignmentConfig())
    assert not [detector.observe(badge(scene(i,anchors=False)[:663],12+i)) for i in range(3)][-1].stable
    detector=LayoutStabilityDetector(AlignmentConfig());fixed=scene(0)[:663]
    static=[detector.observe(badge(fixed.copy(),12+i)) for i in range(3)][-1]
    assert static.stable and static.reason=='static' and static.overlay_masks
    for visible in [(False,True,False),(True,False,True)]:
        detector=LayoutStabilityDetector(AlignmentConfig())
        for i,show in enumerate(visible):
            image=scene(i)[:663]
            result=detector.observe(badge(image,12+i) if show else image)
        assert result.stable and result.outside_media_pixels<=64,result.to_dict()
    detector=LayoutStabilityDetector(AlignmentConfig())
    assert not [detector.observe(badge(scene(i,anchors=False)[:663],12+i)) for i in range(3)][-1].stable
    detector=LayoutStabilityDetector(AlignmentConfig());fixed=scene(0)[:663]
    static=[detector.observe(badge(fixed.copy(),12+i)) for i in range(3)][-1]
    assert static.stable and static.reason=='static' and static.overlay_masks


@pytest.mark.parametrize('direction',['up','down'])
def test_full_capture_ocr_and_height_stay_correct_with_overlapping_notice(tmp_path,direction):
    class Source:
        def __init__(self):self.phase=0;self.position=111 if direction=='up' else 0;self.closed=False;self.inputs=[]
        def read(self):
            image=scene(self.phase)[self.position:self.position+663].copy();self.phase+=1
            return badge(image,12+self.phase%3)
        def scroll(self,units):self.inputs.append(units);self.position=max(0,min(111,self.position+(37 if units>0 else -37)))
        def close(self):self.closed=True
    class Inspector:
        def recognize(self,image):
            region=image[630:657,775:862].astype(np.int16)
            assert not ((region[:,:,1]-region[:,:,2]>30)&(region[:,:,0]-region[:,:,2]>15)).any()
            return []
    source=Source();initial=source.position;errors=[]
    result=run_capture(source,tmp_path/'capture',config=CaptureConfig(max_frames=0,direction=direction,
        sample_interval_s=.001,stable_samples=3,stable_timeout_s=3,settle_delay_s=0,
        scroll_event_interval_s=0,boundary_wait_s=0,max_scroll_events=1),ocr_backend=Inspector(),
        progress=lambda r:errors.append(r['scroll_offset_px']-(source.position-initial)) if r['accepted'] else None)
    assert result['status']=='boundary_reached' and result['rejected']==0,result
    assert errors and all(e==0 for e in errors)
    assert result['final_scroll_offset_px']==(-111 if direction=='up' else 111)
    assert result['image_memory_remaining_bytes']==0 and source.closed
    assert all(v<0 if direction=='up' else v>0 for v in source.inputs)
    rows=read_jsonl(tmp_path/'capture/messages.jsonl')
    assert len(rows)==4 and all(r['type']=='image' for r in rows)
    assert [r['global_top_px'] for r in rows]==sorted(r['global_top_px'] for r in rows)
    assert not list(tmp_path.rglob('*.png'))


ROOT=Path('runs/pointer-20261120-212545-e52d70/capture')
@pytest.mark.skipif(not ROOT.exists(),reason='Private overlay/animation temporal window unavailable')
def test_authentic_temporal_window_and_transition_use_the_same_overlay_layer():
    from welog.imageio import read_image,gray
    from welog.motion import MotionTracker
    from welog.frame_pool import ImageBudget
    event=read_jsonl(ROOT/'diagnostics.jsonl')[-1]
    current=next(i for i in event['images'] if i['role']=='current');meta=current['action']['stability']
    image=read_image(ROOT/current['image']);_,overlays=visual_notice_layers(image)
    assert len([b for b in detect_bubbles(image) if b.kind=='image'])==4
    detector=LayoutStabilityDetector(AlignmentConfig())
    for i,item in enumerate(x for x in event['images'] if x['role']=='temporal_sample'):
        result=detector.observe(gray(read_image(ROOT/item['image'])),media_regions=meta['temporal_media_regions'][i],overlay_regions=overlays)
    assert result.stable and result.outside_media_pixels==0 and result.media_supported_ratio==1
    frames=read_jsonl(ROOT/'frames.jsonl');decisions=read_jsonl(ROOT/'alignments.jsonl')
    tracker=MotionTracker(ImageBudget(),AlignmentConfig());tracker.offset=-126880
    try:
        reference=read_image(ROOT/'frames/000699.png');_,previous_overlay=visual_notice_layers(reference)
        tracker.accept(reference,'699',[],'up',tuple(Rect(**b) for b in frames[699]['action']['stability']['dynamic_masks'])+previous_overlay)
        alignment,_,_=tracker.accept(image,'700',[],'up',tuple(result.dynamic_masks)+tuple(result.overlay_masks),current['action']['attempted_action']['motion_bounds'])
        assert alignment.accepted and alignment.delta_px==-37 and tracker.offset==-126917
    finally:tracker.close()
    assert read_jsonl(ROOT/'alignments.jsonl')==decisions and tracker.budget.used_bytes==0
