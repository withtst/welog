"""Gray silhouettes and separated GIF components require static page evidence."""
from pathlib import Path
import cv2
import numpy as np
import pytest
from welog.archive import read_jsonl
from welog.models import AlignmentConfig,CaptureConfig,Rect
from welog.recognition import achromatic_media_regions,detect_bubbles,EmptyOcr,extend_media_components
from welog.stability import LayoutStabilityDetector
from welog.controller import run_capture


def render(phase,height=663,anchors=True,moving_text=None):
    image=np.full((height,880,3),250,np.uint8)
    rng=np.random.default_rng(481+phase)
    if anchors:
        avatar=np.random.default_rng(480).integers(0,256,(36,36,3),dtype=np.uint8)
        for y in [113,265,417,569]:image[y:y+36,834:870]=avatar
    cv2.ellipse(image,(130+phase%3*2,25),(45+phase%3*2,48),0,0,360,(180,180,180),-1)
    cv2.ellipse(image,(130+phase%3*2,25),(45+phase%3*2,48),0,0,360,(25,25,25),2)
    for y,h in [(113,124),(573,90)]:
        a,b=y,min(height,y+h)
        pixels=rng.integers(0,160,(b-a,132,3),dtype=np.uint8);pixels[:,:,1]=40;pixels[:,:,2]=240
        image[a:b,687:819]=pixels
    cv2.ellipse(image,(755,341),(30,49),0,0,360,(60+phase*17%100,)*3,-1)
    if phase%3!=1:
        cv2.circle(image,(750,276),5,(20,20,220),-1);cv2.circle(image,(760,276),5,(20,20,220),-1)
        cv2.fillPoly(image,[np.array([(745,277),(765,277),(755,290)])],(20,20,220))
    cv2.ellipse(image,(755+phase%3*2,478),(53,48),0,0,360,(30,160,245),-1)
    if moving_text is not None:
        image[150:205,70:320]=238
        cv2.putText(image,'moving text',(80,177+moving_text),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,20),1)
    return image


def test_gray_graphic_is_image_and_separate_heart_stays_with_its_body():
    bubbles=detect_bubbles(render(0,moving_text=0))
    assert any(b.kind=='image' and b.box[0]<=130<b.box[2] and b.box[1]==0 for b in bubbles)
    penguin=[b for b in bubbles if b.kind=='image' and b.box[0]<=755<b.box[2] and b.box[1]<=340<b.box[3]]
    assert len(penguin)==1 and penguin[0].box[1]<=271
    assert any(b.kind=='text' and b.box[0]==70 and b.box[1]==150 for b in bubbles)
    assert not any(b.kind=='image' and b.box[0]<=150<b.box[2] and b.box[1]<=175<b.box[3] for b in bubbles)


def test_gray_animation_and_heart_have_stationary_evidence_and_complete_motion_support():
    detector=LayoutStabilityDetector(AlignmentConfig())
    result=[detector.observe(render(i)) for i in range(3)][-1]
    assert result.stable,result.to_dict()
    assert result.media_supported_ratio>=.98 and result.outside_media_pixels<=64 and result.static_matches>=3


@pytest.mark.parametrize('grayscale', [False, True])
def test_detached_droplets_along_animating_graphics_have_media_support(grayscale):
    boxes = [(110, y, 166, y+38) for y in (95, 247, 399)]
    avatar = np.random.default_rng(271).integers(0, 256, (36, 36, 3), dtype=np.uint8)
    detectors = [LayoutStabilityDetector(AlignmentConfig()) for _ in range(3)]
    for phase in range(3):
        results = []
        for mode, detector in enumerate(detectors):
            image = np.full((663, 880, 3), 247, np.uint8)
            for x1, y1, x2, y2 in boxes:
                image[y1-50:y1-14, 20:56] = avatar
                body = 50 if mode == 2 else 50 + 36 * phase
                image[y1:y2, x1:x2] = (200, body, 70)
                if phase:
                    gap = 7 if mode != 1 else 25
                    for x in (x1-gap, x2+gap):
                        cv2.circle(image, (x, y1+15), 3, (220, 135, 90), -1)
            for y, text in ((498, 'first fixed message'), (565, 'another fixed message')):
                image[y:y+36, 580:819] = (156, 237, 156)
                cv2.putText(image, text, (596, y+25), cv2.FONT_HERSHEY_SIMPLEX, .55, (25,)*3, 1)
            if grayscale:
                result = detector.observe(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), media_regions=boxes)
            else:
                result = detector.observe(image)
            results.append(result)

    # The strip is narrow and belongs only to an actively changing body.
    # Distant motion and the same droplets beside static graphics do not qualify.
    assert results[0].stable and results[0].outside_media_pixels == 0
    assert results[0].media_supported_ratio == 1
    assert all(result.outside_media_pixels > 64 for result in results[1:])


def test_clipped_grayscale_photo_with_bright_interior_keeps_its_full_media_region():
    detector = LayoutStabilityDetector(AlignmentConfig(), max_dynamic_ratio=.08)
    avatar = np.random.default_rng(381).integers(0, 256, (36, 36, 3), dtype=np.uint8)
    for phase in range(3):
        image = np.full((663, 880, 3), 247, np.uint8)
        photo = np.full((132, 132, 3), 67+phase*31, np.uint8)
        photo[:, 15:32] = 166
        photo[:, 46:92] = 224
        image[:54, 687:819] = photo[-54:]
        for y in (74, 226, 378):
            image[y:y+132, 687:819] = photo
            image[y:y+36, 833:869] = avatar
        cv2.rectangle(image, (683, 530), (817, 565), (153, 235, 153), -1)
        cv2.putText(image, 'fixed sent text', (695, 554), cv2.FONT_HERSHEY_SIMPLEX, .48, (25,)*3, 1)
        cv2.rectangle(image, (65, 586), (215, 621), (234,)*3, -1)
        cv2.putText(image, 'fixed reply', (80, 610), cv2.FONT_HERSHEY_SIMPLEX, .48, (25,)*3, 1)
        messages = detect_bubbles(image)
        photos = [b for b in messages if b.kind == 'image']
        assert len(photos) == 4 and photos[0].top_state == 'clipped'
        assert photos[0].box[0] <= 687 and photos[0].box[2] >= 819
        assert [b.box[1] for b in messages if b.kind == 'text'] == [530, 586]
        result = detector.observe(image)
    assert result.stable and result.outside_media_pixels == 0, result.to_dict()
    assert result.reason == 'multiple_media_animation' and result.static_matches >= 3


@pytest.mark.parametrize('edge', ['top', 'bottom'])
def test_short_dense_photo_fragment_requires_independent_edges_in_same_lane(edge):
    height = 663
    image = np.full((height, 880), 247, np.uint8)
    photo = np.tile(np.linspace(38, 180, 132, dtype=np.uint8), (132, 1))
    top = 0 if edge == 'top' else height-26
    image[top:top+26, 70:202] = photo[:26]

    def has_fragment(frame):
        return any(y == 0 if edge == 'top' else bottom == height
                   for _, y, _, bottom in achromatic_media_regions(frame, include_small_edge_media=True))

    assert not has_fragment(image)
    image[158:290, 85:217] = photo
    assert not has_fragment(image)
    image[158:290, 85:217] = 247
    image[158:290, 70:202] = photo
    assert has_fragment(image)
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    bubbles = detect_bubbles(color)
    assert len(bubbles) == 2 and all(b.kind == 'image' for b in bubbles)
    assert any((b.top_state if edge == 'top' else b.bottom_state) == 'clipped' for b in bubbles)

    image[top:top+26, 70:202] = 154
    assert not has_fragment(image)


def test_gray_media_does_not_hide_page_or_message_text_movement():
    detector=LayoutStabilityDetector(AlignmentConfig())
    for offset in range(6):assert not detector.observe(render(offset,900)[offset:offset+663]).stable
    detector=LayoutStabilityDetector(AlignmentConfig())
    for offset in range(6):assert not detector.observe(render(offset,moving_text=offset)).stable


def test_animated_graphics_without_static_detail_remain_unconfirmed():
    detector=LayoutStabilityDetector(AlignmentConfig())
    result=[detector.observe(render(i,anchors=False)) for i in range(3)][-1]
    assert not result.stable


def test_bubbles_and_avatar_lanes_cannot_be_absorbed_as_satellites():
    image=np.full((400,880,3),250,np.uint8)
    cv2.rectangle(image,(744,171),(765,189),(20,20,220),-1)
    cv2.rectangle(image,(834,171),(869,206),(20,20,20),-1)
    regions=[(720,191,790,300)]
    assert extend_media_components(image,regions,[(740,165,775,195)])==regions
    assert extend_media_components(image,regions)[0][1]<=171


def test_satellite_between_distinct_graphics_is_not_merged_into_both_messages():
    image=np.full((400,880,3),250,np.uint8)
    cv2.rectangle(image,(744,190),(765,208),(20,20,220),-1)
    regions=[(720,100,790,180),(720,220,790,300)]
    assert extend_media_components(image,regions)==regions


@pytest.mark.parametrize('edge',['top','bottom'])
def test_clipped_gray_graphics_are_detected_on_both_edges(edge):
    image=np.full((400,880,3),250,np.uint8)
    center=15 if edge=='top' else 385
    cv2.ellipse(image,(130,center),(50,50),0,0,360,(180,180,180),-1)
    bubbles=detect_bubbles(image,include_small_edge_media=True)
    assert any(b.kind=='image' and (b.box[1]==0 if edge=='top' else b.box[3]==400) for b in bubbles)


@pytest.mark.parametrize('direction',['up','down'])
def test_capture_traverses_gray_and_disconnected_animation_without_height_drift(tmp_path,direction):
    class Source:
        def __init__(self):
            self.phase=0;self.position=111 if direction=='up' else 0
            self.inputs=[];self.closed=False
        def read(self):
            image=render(self.phase,900);self.phase+=1
            image[:115,70:215]=250
            cv2.ellipse(image,(130+self.phase%3*2,65),(45+self.phase%3*2,48),0,0,360,(180,180,180),-1)
            marker=np.random.default_rng(482).integers(0,256,(36,36,3),dtype=np.uint8)
            image[500:536,20:56]=marker
            return image[self.position:self.position+663].copy()
        def scroll(self,units):
            self.inputs.append(units);self.position=max(0,min(111,self.position+(37 if units>0 else -37)))
        def close(self):self.closed=True
    source=Source();initial=source.position;errors=[]
    result=run_capture(source,tmp_path/'capture',config=CaptureConfig(max_frames=0,direction=direction,
        stable_samples=3,stable_timeout_s=3,sample_interval_s=.001,settle_delay_s=0,
        scroll_event_interval_s=0,boundary_wait_s=0,max_scroll_events=1),ocr_backend=EmptyOcr(),
        progress=lambda row:errors.append(row['scroll_offset_px']-(source.position-initial)) if row['accepted'] else None)
    assert result['status']=='boundary_reached' and result['rejected']==0,result
    assert errors and all(error==0 for error in errors)
    assert result['final_scroll_offset_px']==(-111 if direction=='up' else 111)
    assert result['image_memory_remaining_bytes']==0 and source.closed
    assert all(s<0 if direction=='up' else s>0 for s in source.inputs)
    rows=read_jsonl(tmp_path/'capture/messages.jsonl')
    assert [r['global_top_px'] for r in rows]==sorted(r['global_top_px'] for r in rows)
    assert len(rows)==5 and all(r['type']=='image' for r in rows)
    assert not list(tmp_path.rglob('*.png'))


ROOT=Path('runs/pointer-20261119-010237-1efb8f/capture')
@pytest.mark.skipif(not ROOT.exists(),reason='Private gray temporal window unavailable')
def test_authentic_timeout_window_and_bounded_transition_are_recovered_read_only():
    from welog.imageio import read_image,gray
    from welog.motion import MotionTracker
    from welog.frame_pool import ImageBudget
    events=read_jsonl(ROOT/'diagnostics.jsonl');event=events[-1]
    current=next(i for i in event['images'] if i['role']=='current')
    hints=current['action']['stability']['temporal_media_regions']
    detector=LayoutStabilityDetector(AlignmentConfig())
    for i,item in enumerate(x for x in event['images'] if x['role']=='temporal_sample'):
        result=detector.observe(gray(read_image(ROOT/item['image'])),media_regions=hints[i])
    assert result.stable and result.outside_media_pixels<=64 and result.media_supported_ratio>=.98
    frames=read_jsonl(ROOT/'frames.jsonl');decisions=read_jsonl(ROOT/'alignments.jsonl')
    tracker=MotionTracker(ImageBudget(),AlignmentConfig());tracker.offset=decisions[575]['scroll_offset_px']
    try:
        tracker.accept(read_image(ROOT/'frames/000575.png'),'575',[],'up',tuple(Rect(**b) for b in frames[575]['action']['stability']['dynamic_masks']))
        alignment,_,_=tracker.accept(read_image(ROOT/'frames/000576.png'),'576',[],'up',tuple(result.dynamic_masks),{'minimum':-64,'maximum':0})
        assert alignment.accepted and alignment.delta_px==-37 and tracker.offset==-118446
    finally:tracker.close()
    assert read_jsonl(ROOT/'alignments.jsonl')==decisions and tracker.budget.used_bytes==0
