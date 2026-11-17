import cv2
import numpy as np
import pytest
from welog.models import AlignmentConfig,CaptureConfig
from welog.stability import LayoutStabilityDetector
from welog.controller import run_capture
from welog.recognition import EmptyOcr

BOXES=[(45,20,195,170),(445,210,595,360),(45,400,195,550),(445,570,595,710)]


def scene(height=720,anchors=True):
    image=np.full((height,640,3),242,np.uint8)
    if anchors:
        for i,y in enumerate(range(60,height-10,90)):
            cv2.putText(image,f'anchor {i:03d} unique',(210,y),cv2.FONT_HERSHEY_SIMPLEX,.5,(25,25,25),1)
    return image


def animate(image,rng,offset=0):
    image=image.copy()
    for x1,y1,x2,y2 in BOXES:
        a,b=max(0,y1-offset),min(image.shape[0],y2-offset)
        if b>a:image[a:b,x1:x2]=rng.integers(0,256,(b-a,x2-x1,3),dtype=np.uint8)
    return image


def test_multiple_gifs_above_old_limit_are_supported_by_static_evidence():
    rng=np.random.default_rng(45);detector=LayoutStabilityDetector(AlignmentConfig())
    results=[detector.observe(animate(scene(),rng)) for _ in range(3)]
    result=results[-1]
    assert result.stable and result.reason=='multiple_media_animation'
    assert .12<result.dynamic_region_ratio<=.5 and result.media_supported_ratio>=.98
    assert result.static_matches>=3


def test_media_only_without_static_detail_cannot_be_accepted():
    rng=np.random.default_rng(46);detector=LayoutStabilityDetector(AlignmentConfig())
    assert not [detector.observe(animate(scene(anchors=False),rng)) for _ in range(3)][-1].stable


def test_feature_stationarity_requires_zero_motion_not_just_similar_text():
    from welog.alignment import OverlapAligner
    from welog.imageio import gray
    from welog.models import Rect
    image=gray(scene())
    aligner=OverlapAligner(AlignmentConfig(masks=tuple(Rect(x1,y1,x2-x1,y2-y1) for x1,y1,x2,y2 in BOXES)))
    assert LayoutStabilityDetector._static_features(image,image.copy(),aligner)>=12
    assert LayoutStabilityDetector._static_features(image,np.roll(image,1,axis=0),aligner)==0


def test_multiple_gifs_do_not_hide_page_scrolling():
    rng=np.random.default_rng(47);detector=LayoutStabilityDetector(AlignmentConfig())
    base=scene(1100)
    for y in range(8):assert not detector.observe(animate(base[y:y+720],rng,y)).stable


def test_motion_outside_media_does_not_get_a_larger_mask_budget():
    rng=np.random.default_rng(48);detector=LayoutStabilityDetector(AlignmentConfig())
    for y in range(5):
        base=scene(1100)[y:y+720]
        assert not detector.observe(animate(base,rng)).stable


@pytest.mark.parametrize('direction',['up','down'])
def test_continuous_multi_gif_capture_preserves_signed_coordinates(tmp_path,direction):
    class Source:
        position=0 if direction=='down' else 300
        def __init__(self):self.rng=np.random.default_rng(49);self.closed=False;self.inputs=[]
        def read(self):return animate(scene(1100)[self.position:self.position+720],self.rng,self.position)
        def scroll(self,units):self.inputs.append(units);self.position=max(0,min(300,self.position+(50 if units>0 else -50)))
        def close(self):self.closed=True
    source=Source();initial=source.position;errors=[]
    result=run_capture(source,tmp_path/'capture',config=CaptureConfig(max_frames=7,direction=direction,
        sample_interval_s=.001,stable_samples=3,stable_timeout_s=2,settle_delay_s=0,scroll_event_interval_s=0,max_scroll_events=1),
        ocr_backend=EmptyOcr(),progress=lambda row:errors.append(row['scroll_offset_px']-(source.position-initial)) if row['accepted'] else None)
    assert result['reason'] not in {'StableTimeout','continuity_unconfirmed'}
    assert source.closed and errors and all(error==0 for error in errors)
    assert result['final_scroll_offset_px']==(-300 if direction=='up' else 300)
    assert result['image_memory_remaining_bytes']==0 and not list(tmp_path.rglob('*.png'))
