from pathlib import Path
import cv2
import numpy as np
import pytest
from welog.recognition import detect_bubbles
from welog.models import AlignmentConfig
from welog.stability import LayoutStabilityDetector


def samples(moving_text=False):
    rng=np.random.default_rng(91);avatar=rng.integers(0,256,(36,36,3),dtype=np.uint8)
    result=[]
    for i in range(3):
        image=np.full((672,875,3),250,np.uint8)
        image[0,:]=244;image[:,0]=244
        image[0:4,752:754]=30
        image[3:16,742:768]=(60,70,180+i*30)
        for y in [60,210,360,510]:image[y:y+130,690:820]=rng.integers(0,256,(130,130,3),dtype=np.uint8)
        for y in [45,197,349,501]:image[y:y+36,833:869]=avatar
        cv2.putText(image,'quiet',(70,660),cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,20),1)
        if moving_text:cv2.putText(image,'moving text',(70,240+i*2),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1)
        result.append(image)
    return result


def test_border_connected_clipped_tail_is_temporal_media_not_a_new_message():
    image=samples()[0]
    normal=detect_bubbles(image)
    temporal=detect_bubbles(image,include_small_edge_media=True)
    assert not [b for b in normal if b.box[1]<20 and b.kind=='image']
    assert [b for b in temporal if b.box[1]==0 and b.kind=='image' and b.box[0]<=742 and b.box[2]>=768]


def test_narrow_fragment_regression_old_policy_rejects_new_policy_confirms(monkeypatch):
    from welog import recognition
    original=recognition.detect_bubbles
    frames=samples()
    with monkeypatch.context() as scoped:
        scoped.setattr(recognition,'detect_bubbles',lambda image,**kwargs:original(image))
        old=LayoutStabilityDetector(AlignmentConfig())
        result=[old.observe(frame) for frame in frames][-1]
        assert not result.stable and result.reason=='motion_outside_media'
    new=LayoutStabilityDetector(AlignmentConfig())
    result=[new.observe(frame) for frame in frames][-1]
    assert result.stable and result.media_supported_ratio>=.98


def test_moving_text_outside_fragments_is_not_hidden():
    detector=LayoutStabilityDetector(AlignmentConfig())
    assert not [detector.observe(frame) for frame in samples(True)][-1].stable


def test_narrow_interior_shape_is_not_promoted_to_edge_media():
    image=np.full((200,600,3),250,np.uint8)
    image[80:93,500:526]=(40,50,220)
    assert not detect_bubbles(image,include_small_edge_media=True)


def test_thin_colored_edge_stroke_is_localized_without_changing_message_detection():
    image=np.full((200,600,3),250,np.uint8)
    image[0,:]=244;image[:,0]=244;image[:45,500:502]=(40,50,220)
    assert not detect_bubbles(image)
    assert [b for b in detect_bubbles(image,include_small_edge_media=True) if b.box[0]<=500 and b.box[2]>=502]


def test_edge_media_keeps_interior_parts_not_only_the_colored_hat():
    image=np.full((300,700,3),250,np.uint8)
    image[0,:]=244;image[:,0]=244
    image[:120,520:650]=255
    image[:25,520:650]=(20,210,240)
    image[55:100,527:543]=(180,70,30)
    image[55:100,625:641]=(180,70,30)
    image[70:85,575:595]=(20,40,100)
    temporal=detect_bubbles(image,include_small_edge_media=True)
    assert any(b.kind=='image' and b.box[0]<=527 and b.box[1]==0 and b.box[2]>=641 and b.box[3]>=100 for b in temporal)


ROOT=Path('runs/pointer-20261117-165048-026bc5/capture/frames/000070.png')
@pytest.mark.skipif(not ROOT.exists(),reason='Private diagnostic image unavailable')
def test_real_diagnostic_retains_border_connected_foot():
    from welog.imageio import read_image
    image=read_image(ROOT)
    assert not [b for b in detect_bubbles(image) if b.box[1]<20]
    assert [b for b in detect_bubbles(image,include_small_edge_media=True) if b.box[1]==0 and b.box[0]<=740 and b.box[2]>=766]
