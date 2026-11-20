"""A partly occluded unique bubble must bootstrap motion among repeated avatars."""
from pathlib import Path
import importlib.util

import cv2
import numpy as np
import pytest

from welog.archive import read_jsonl
from welog.controller import run_capture
from welog.models import CaptureConfig,AlignmentConfig,Rect
from welog.recognition import TextLine,detect_bubbles
from welog.text_rules import notice_region_masks,clean_notice


class NoticeConversation:
    def __init__(self,direction):
        self.world=np.full((700,880,3),250,np.uint8)
        self.rng=np.random.default_rng(237)
        avatar=self.rng.integers(0,256,(36,36,3),dtype=np.uint8)
        self.starts=[105+152*i for i in range(4)]
        for y in [48,*self.starts]:self.world[y:y+36,834:870]=avatar
        cv2.rectangle(self.world,(753,48),(819,83),(153,244,151),-1)
        cv2.putText(self.world,'ok',(763,72),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1)
        self.position=37 if direction=='up' else 0
        self.inputs=[];self.closed=False

    def read(self):
        image=self.world[self.position:self.position+663].copy()
        for y in self.starts:
            a,b=max(0,y-self.position),min(663,y+132-self.position)
            if b>a:
                pixels=self.rng.integers(0,160,(b-a,132,3),dtype=np.uint8)
                pixels[:,:,1]=40;pixels[:,:,2]=240
                image[a:b,687:819]=pixels
        # Fixed viewport overlay partly covers the top outgoing bubble.
        image[8:49,764:874]=(image[8:49,764:874].astype(float)*.9+20).astype(np.uint8)
        cv2.rectangle(image,(772,12),(869,36),(250,250,250),-1)
        cv2.putText(image,'12 new msg',(779,31),cv2.FONT_HERSHEY_SIMPLEX,.42,(120,180,0),1)
        return image

    def scroll(self,units):
        self.inputs.append(units)
        self.position=0 if units<0 else 37

    def close(self):self.closed=True


class NoticeOcr:
    def recognize(self,image):
        return [TextLine('1条新消息',(778,12,858,36),.99)]


class ClippedNoticeConversation(NoticeConversation):
    def __init__(self,direction):
        super().__init__(direction)
        self.world=np.pad(self.world,((0,37),(0,0),(0,0)),constant_values=250)
        self.position=73 if direction=='up' else 36

    def scroll(self,units):
        self.inputs.append(units)
        self.position=36 if units<0 else 73


def test_clipped_cold_start_continues_through_entire_media_history(tmp_path):
    class Source(NoticeConversation):
        def __init__(self):
            super().__init__('up')
            self.world=np.full((1936,880,3),250,np.uint8)
            self.starts=sorted([1093-152*i for i in range(8)]+[1305+152*i for i in range(4)])
            avatar=self.rng.integers(0,256,(36,36,3),dtype=np.uint8)
            for y in [1248,*self.starts]:self.world[y:y+36,834:870]=avatar
            cv2.rectangle(self.world,(753,1248),(819,1283),(153,244,151),-1)
            cv2.putText(self.world,'ok',(763,1272),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1)
            self.position=1273
        def scroll(self,units):
            self.inputs.append(units)
            self.position=max(0,self.position-(37 if len(self.inputs)==1 else 67))
    source=Source()
    class Backend(NoticeOcr):
        def recognize(self,image):
            if len(image)!=663:return []
            lines=super().recognize(image);y=1248-source.position
            if 50<y and y+36<663:lines.append(TextLine('ok',(763,y+10,807,y+28),.99))
            return lines
    result=run_capture(source,tmp_path/'capture',config=CaptureConfig(max_frames=0,direction='up',
        stable_samples=3,stable_timeout_s=3,sample_interval_s=.001,settle_delay_s=0,
        scroll_event_interval_s=0,boundary_wait_s=0),ocr_backend=Backend())
    assert result['status']=='boundary_reached' and source.position==0,result
    assert result['rejected']==0 and result['final_scroll_offset_px']==-1273
    decisions=read_jsonl(tmp_path/'capture/alignments.jsonl')
    assert decisions[1]['alignment']['reason'] in {'edge_layout_aligned_8px','edge_layout_aligned_6px'}
    assert any(d['motion_method']=='bounded_layout' for d in decisions)
    rows=read_jsonl(tmp_path/'capture/messages.jsonl')
    assert [r['global_top_px'] for r in rows if r['type']=='image']==[y-1273 for y in source.starts]
    assert [r['text'] for r in rows if r['type']=='text']==['ok']
    assert result['image_memory_remaining_bytes']==0 and source.closed and all(s<0 for s in source.inputs)
    assert not list(tmp_path.rglob('*.png'))


@pytest.mark.parametrize('direction',['up','down'])
def test_clipped_anchor_under_twelve_rows_bootstraps_without_calibration(tmp_path,direction):
    source=ClippedNoticeConversation(direction)
    result=run_capture(source,tmp_path/'capture',config=CaptureConfig(max_frames=0,direction=direction,
        stable_samples=3,stable_timeout_s=3,sample_interval_s=.001,settle_delay_s=0,
        scroll_event_interval_s=0,boundary_wait_s=0),ocr_backend=NoticeOcr())
    assert result['status']=='boundary_reached',result
    assert result['accepted']==5 and result['rejected']==0
    assert result['final_scroll_offset_px']==(-37 if direction=='up' else 37)
    decisions=read_jsonl(tmp_path/'capture/alignments.jsonl')
    assert decisions[1]['alignment']['reason'] in {'edge_layout_aligned_8px','edge_layout_aligned_6px'}
    assert not any(f['action'].get('motion_bounds') for f in read_jsonl(tmp_path/'capture/frames.jsonl'))
    assert source.closed and result['image_memory_remaining_bytes']==0
    assert all(s<0 if direction=='up' else s>0 for s in source.inputs)
    rows=read_jsonl(tmp_path/'capture/messages.jsonl')
    assert [r['global_top_px'] for r in rows]==sorted(r['global_top_px'] for r in rows)
    assert not list(tmp_path.rglob('*.png'))


@pytest.mark.parametrize('direction',['up','down'])
def test_first_scroll_from_animation_works_without_existing_calibration(tmp_path,direction):
    source=NoticeConversation(direction)
    result=run_capture(source,tmp_path/'capture',config=CaptureConfig(max_frames=0,direction=direction,
        stable_samples=3,stable_timeout_s=3,sample_interval_s=.001,settle_delay_s=0,
        scroll_event_interval_s=0,boundary_wait_s=0),ocr_backend=NoticeOcr())
    assert result['status']=='boundary_reached',result
    assert result['accepted']==5 and result['rejected']==0
    assert result['final_scroll_offset_px']==(-37 if direction=='up' else 37)
    assert result['image_memory_remaining_bytes']==0 and source.closed
    assert all(s<0 if direction=='up' else s>0 for s in source.inputs)
    frames=read_jsonl(tmp_path/'capture/frames.jsonl')
    assert not any(f['action'].get('motion_bounds') for f in frames)
    assert frames[1]['action']['scroll_guard']=='media_dominated'
    rows=read_jsonl(tmp_path/'capture/messages.jsonl')
    assert [r['global_top_px'] for r in rows]==sorted(r['global_top_px'] for r in rows)
    assert not list(tmp_path.rglob('*.png'))


@pytest.mark.parametrize('scale',[1,1.5,2])
def test_scaled_notice_keeps_nearby_anchor_and_covers_ink(scale):
    box=tuple(round(v*scale) for v in (778,12,858,36))
    mask=notice_region_masks([TextLine('1条新消息',box,.99)],round(880*scale),round(663*scale))[0]
    assert mask.x<=box[0] and mask.y<=box[1] and mask.right>=box[2] and mask.bottom>=box[3]
    assert mask.x>=round(770*scale)  # Unoccluded left part of the green bubble.
    assert mask.bottom<=round(42*scale)
    assert not notice_region_masks([TextLine('正文1条新消息继续',box,.99)],round(880*scale),round(663*scale))


ROOT=Path('runs/pointer-20261118-233455-374981/capture')
@pytest.mark.skipif(not ROOT.exists(),reason='Private cold-start screenshots unavailable')
def test_private_cold_start_with_cached_local_ocr_preserves_source_decisions():
    from welog.imageio import read_image
    from welog.motion import MotionTracker
    from welog.frame_pool import ImageBudget
    from welog.file_cards import mark_candidates
    from welog.pipeline import FrameProcessor
    from welog.recognition import RapidOcrAdapter
    spec=importlib.util.find_spec('rapidocr')
    if spec is None:pytest.skip('Local OCR unavailable')
    model_dir=Path(spec.origin).parent/'models'
    if not all((model_dir/f'PP-OCRv6_{part}_small.onnx').exists() for part in ['det','rec']):
        pytest.skip('Cached local OCR models unavailable; no network download')
    frames=read_jsonl(ROOT/'frames.jsonl');source_decisions=read_jsonl(ROOT/'alignments.jsonl')
    assert sum(r['alignment']['accepted'] for r in source_decisions)==1
    tracker=MotionTracker(ImageBudget(),AlignmentConfig());ocr=RapidOcrAdapter()
    try:
        for index,frame in enumerate(frames):
            image=read_image(ROOT/'frames'/f'{index:06d}.png');bubbles=detect_bubbles(image);mark_candidates(image,bubbles)
            clean=image.copy()
            FrameProcessor.mask_media(clean,[b.box for b in bubbles if b.kind!='text' and '_file_candidate' not in b.metadata])
            notices=[line for line in ocr.recognize(clean) if clean_notice(line.text)[1]]
            assert notices
            masks=tuple(Rect(**b) for b in frame['action']['stability']['dynamic_masks'])+notice_region_masks(notices,*image.shape[1::-1])
            result,_,_=tracker.accept(image,frame['frame_id'],[],'up',masks)
            assert result.accepted and result.delta_px==(0 if index!=1 else -37)
        assert tracker.offset==-37
    finally:tracker.close()
    assert read_jsonl(ROOT/'alignments.jsonl')==source_decisions and tracker.budget.used_bytes==0


CLIPPED_ROOT=Path('runs/pointer-20261119-000531-ba2a7b/capture')
@pytest.mark.skipif(not CLIPPED_ROOT.exists(),reason='Private clipped-anchor images unavailable')
@pytest.mark.parametrize('erase_unique_fragment',[False,True])
def test_real_clipped_anchor_and_counterfactual_without_it(erase_unique_fragment):
    from welog.imageio import read_image
    from welog.motion import MotionTracker
    from welog.frame_pool import ImageBudget
    frames=read_jsonl(CLIPPED_ROOT/'frames.jsonl');decisions=read_jsonl(CLIPPED_ROOT/'alignments.jsonl')
    tracker=MotionTracker(ImageBudget(),AlignmentConfig())
    try:
        for i,frame in enumerate(frames):
            image=read_image(CLIPPED_ROOT/'frames'/f'{i:06d}.png')
            if erase_unique_fragment:image[:13 if i==0 else 50,740:824]=250
            masks=tuple(Rect(**b) for b in frame['action']['stability']['dynamic_masks']+decisions[i]['motion_overlay_masks'])
            result,_,_=tracker.accept(image,str(i),[],'up',masks)
            assert result.accepted==(i==0 or not erase_unique_fragment)
            if i==1 and not erase_unique_fragment:
                assert result.delta_px==-37 and result.reason in {'edge_layout_aligned_8px','edge_layout_aligned_6px'}
        assert tracker.offset==(0 if erase_unique_fragment else -37)
    finally:tracker.close()
    assert read_jsonl(CLIPPED_ROOT/'alignments.jsonl')==decisions and tracker.budget.used_bytes==0
