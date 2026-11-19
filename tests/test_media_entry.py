"""Guard before reference text leaves overlap; retain command evidence on retry."""
from concurrent.futures import Future
from pathlib import Path
import numpy as np
import pytest
from welog.archive import read_jsonl
from welog.models import CaptureConfig,AlignmentConfig,Rect
from welog.scroll import ScrollCalibration,ScrollPlanner,retained_anchor_count


def test_ordinary_batch_and_retry_keep_the_original_command_envelope(tmp_path,monkeypatch):
    from welog import controller
    from welog.message_store import MessageStore
    class Calibration(ScrollCalibration):
        def __init__(self):
            super().__init__()
            for delta in [-201,-205,-201]:self.observe(delta,3,'text')
    class Planner(ScrollPlanner):
        def __init__(self,*args):super().__init__(*args);self.events=3
    class Source:
        calls=0
        def read(self):return np.zeros((663,880,3),np.uint8)
        def scroll(self,units):self.calls+=1
        def close(self):pass
    class Worker:
        def __init__(self,output,budget,*args):self.output,self.budget=output,budget
        def submit(self,lease,frame_id,masks,direction,captured_at,bounds):
            lease.release();index=int(frame_id);accepted=index!=1
            if index:assert bounds['events']==3 and bounds['minimum']==-265
            future=Future();future.set_result({'alignment':{'accepted':accepted,'reason':'fixture','delta_px':None if not accepted else 0 if not index else -205},
                'scroll_offset_px':None if not accepted else 0 if not index else -205,
                'motion_method':'fixture','reference_id':None if not index else '000000','ocr_error':None})
            return future
        def finish(self,summary):
            summary.update(image_memory_remaining_bytes=self.budget.used_bytes,image_memory_peak_bytes=self.budget.peak_bytes)
            store=MessageStore(self.output/'messages.sqlite3')
            try:return store.export(self.output,summary)
            finally:store.close()
    monkeypatch.setattr(controller,'ScrollCalibration',Calibration)
    monkeypatch.setattr(controller,'ScrollPlanner',Planner)
    monkeypatch.setattr(controller,'OcrWorker',Worker)
    source=Source()
    result=controller.run_capture(source,tmp_path/'capture',config=CaptureConfig(max_frames=3,direction='up',
        stable_samples=2,stable_timeout_s=3,sample_interval_s=.001,settle_delay_s=0,scroll_event_interval_s=0))
    frames=read_jsonl(tmp_path/'capture/frames.jsonl')
    assert frames[1]['action']['events']==3 and 'scroll_guard' not in frames[1]['action']
    assert frames[2]['action']['kind']=='retry' and frames[2]['action']['events']==0
    assert frames[2]['action']['motion_bounds']==frames[1]['action']['motion_bounds']
    assert source.calls==3 and result['accepted']==2 and result['rejected']==1 and result['final_scroll_offset_px']==-205
    assert result['image_memory_remaining_bytes']==0 and not list(tmp_path.rglob('*.png'))


ROOT=Path('runs/pointer-20261119-143536-d66386/capture')
@pytest.mark.skipif(not ROOT.exists(),reason='Private media-entry frames unavailable')
def test_authentic_entry_has_an_observed_batch_bound_and_read_only_reassessment():
    from welog.imageio import read_image
    from welog.motion import MotionTracker
    from welog.frame_pool import ImageBudget
    frames=read_jsonl(ROOT/'frames.jsonl');decisions=read_jsonl(ROOT/'alignments.jsonl')
    calibration=ScrollCalibration()
    for frame,row in zip(frames[:249],decisions[:249]):
        if row['alignment']['accepted']:calibration.observe(row['alignment']['delta_px'],frame['action'].get('events',0),row['motion_method'])
    bounds=calibration.batch_bounds(3,'up',frames[248]['height'])
    assert bounds['minimum']==-265 and bounds['samples']==8
    observations=next(r for r in read_jsonl(ROOT/'observations.jsonl') if r['frame_id']=='000248')
    boxes=[line['box'] for b in observations['observations'] if b['kind']=='text' for line in b['lines']
           if line['score']>=.7 and len(line['text'].strip())>=2 and line['box'][1]>2 and line['box'][3]<661]
    assert len(boxes)>=2 and retained_anchor_count(boxes,663,'up',265)==0
    tracker=MotionTracker(ImageBudget(),AlignmentConfig());tracker.offset=-51743
    try:
        for index in range(248,253):
            masks=tuple(Rect(**b) for b in frames[index]['action']['stability']['dynamic_masks']+decisions[index]['motion_overlay_masks'])
            result,_,_=tracker.accept(read_image(ROOT/'frames'/f'{index:06d}.png'),str(index),[],'up',masks,bounds if index>248 else None)
            assert result.accepted and result.delta_px==(-205 if index==249 else 0)
        assert tracker.offset==-51948
    finally:tracker.close()
    assert read_jsonl(ROOT/'alignments.jsonl')==decisions and tracker.budget.used_bytes==0
