import os
import cv2
import numpy as np
import pytest
from welog.models import AlignmentConfig
from welog.stability import LayoutStabilityDetector


BOXES=[(710,20+153*i,780,125+153*i) for i in range(4)]


def base(with_avatars=True,caption_shift=0):
    image=np.full((672,875,3),250,np.uint8)
    if with_avatars:
        avatar=np.random.default_rng(81).integers(0,256,(36,36,3),dtype=np.uint8)
        for i in range(4):image[40+153*i:76+153*i,830:866]=avatar
    cv2.putText(image,'ok',(70,635+caption_shift),cv2.FONT_HERSHEY_SIMPLEX,.4,(30,30,30),1)
    return image


def animate(image,rng):
    image=image.copy()
    for x1,y1,x2,y2 in BOXES:image[y1:y2,x1:x2]=rng.integers(0,256,(y2-y1,x2-x1,3),dtype=np.uint8)
    return image


def test_repeated_avatars_support_temporal_stationarity_without_unique_matches():
    detector=LayoutStabilityDetector(AlignmentConfig());rng=np.random.default_rng(82)
    results=[detector.observe(animate(base(),rng)) for _ in range(3)]
    result=results[-1]
    assert result.stable and result.dynamic_region_ratio<.12
    assert result.static_method=='position_patches' and result.static_matches>=3
    assert result.media_supported_ratio>=.98


def test_blank_surroundings_do_not_provide_stationary_evidence():
    detector=LayoutStabilityDetector(AlignmentConfig());rng=np.random.default_rng(83)
    result=[detector.observe(animate(base(False),rng)) for _ in range(3)][-1]
    assert not result.stable


def test_moving_repeated_avatar_page_is_not_stationary():
    detector=LayoutStabilityDetector(AlignmentConfig());rng=np.random.default_rng(84)
    for offset in range(6):
        image=np.roll(base(),offset,axis=0)
        assert not detector.observe(animate(image,rng)).stable


def test_stationary_avatars_do_not_hide_scrolling_body_text():
    detector=LayoutStabilityDetector(AlignmentConfig());rng=np.random.default_rng(85)
    for offset in range(6):
        image=base();cv2.putText(image,'moving content',(250,340+offset),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1)
        assert not detector.observe(animate(image,rng)).stable


@pytest.mark.skipif(not os.getenv('WELOG_REPEAT_GIF_SAMPLE'),reason='Private reference image unavailable')
def test_supplied_reference_with_synthetic_animation_is_temporally_stable():
    from welog.imageio import read_image
    from welog.recognition import detect_bubbles
    image=read_image(os.environ['WELOG_REPEAT_GIF_SAMPLE'])
    boxes=[b.box for b in detect_bubbles(image) if b.kind=='image']
    assert len(boxes)==4
    detector=LayoutStabilityDetector(AlignmentConfig());rng=np.random.default_rng(86)
    for _ in range(3):
        sample=image.copy()
        for x1,y1,x2,y2 in boxes:sample[y1:y2,x1:x2]=rng.integers(0,256,(y2-y1,x2-x1,3),dtype=np.uint8)
        result=detector.observe(sample)
    assert result.stable and result.dynamic_region_ratio<.12


@pytest.mark.skipif(not os.getenv('WELOG_REPEAT_GIF_SAMPLE'),reason='Private reference image unavailable')
def test_reference_layout_stability_and_cross_screen_pose_are_separate_checks():
    from welog.imageio import read_image
    from welog.recognition import detect_bubbles
    from welog.frame_pool import ImageBudget
    from welog.motion import MotionTracker
    from welog.models import Rect
    image=read_image(os.environ['WELOG_REPEAT_GIF_SAMPLE'])
    h,w=image.shape[:2];document=np.full((h+201,w,3),250,np.uint8);document[:h]=image
    boxes=[b.box for b in detect_bubbles(image) if b.kind=='image']
    tracker=MotionTracker(ImageBudget(),AlignmentConfig());rng=np.random.default_rng(87)
    try:
        for i,offset in enumerate([201,0]):
            detector=LayoutStabilityDetector(AlignmentConfig())
            for _ in range(3):
                sample=document[offset:offset+h].copy()
                for x1,y1,x2,y2 in boxes:
                    a,b=max(0,y1-offset),min(h,y2-offset)
                    if b>a:sample[a:b,x1:x2]=rng.integers(0,256,(b-a,x2-x1,3),dtype=np.uint8)
                stability=detector.observe(sample)
            assert stability.stable
            result,_,_=tracker.accept(sample,str(i),[],'up',tuple(stability.dynamic_masks))
            assert result.accepted
        assert tracker.offset==-201
    finally:tracker.close()
