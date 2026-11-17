"""Capture a complete invented conversation in either direction, without desktop APIs."""
import json
from concurrent.futures import Future

import cv2
import numpy as np
import pytest

from welog.archive import read_jsonl
from welog.controller import run_capture
from welog.exports import ExportOptions, FORMATS, RecordSource, export_records
from welog.message_store import MessageStore
from welog.models import CaptureConfig
from welog.recognition import TextLine


class Conversation:
    def __init__(self,direction,count=30):
        self.messages=[(40+index*110,f'message {index:03d}') for index in range(count)]
        self.scene=np.full((count*110+50,400,3),250,np.uint8)
        for index,(top,text) in enumerate(self.messages):
            x=45 if index%2==0 else 190
            self.scene[top:top+65,x:x+155]=(238,238,238) if index%2==0 else (100,220,100)
            cv2.putText(self.scene,text,(x+7,top+39),cv2.FONT_HERSHEY_SIMPLEX,.47,(20,20,20),1)
        self.end=self.scene.shape[0]-500
        self.position=self.end if direction=='up' else 0
        self.inputs=[]
        self.closed=False

    def read(self):return self.scene[self.position:self.position+500].copy()

    def scroll(self,units):
        self.inputs.append(units)
        self.position=max(0,min(self.end,self.position+(120 if units>0 else -120)))

    def close(self):self.closed=True


class KnownText:
    def __init__(self,source):self.source=source

    def recognize(self,image):
        if image.shape[0]!=500:return []  # Fixture text is always fully visible on another full frame.
        lines=[]
        for index,(top,text) in enumerate(self.source.messages):
            y=top+20-self.source.position
            x=45 if index%2==0 else 190
            if 0<=y and y+22<=500:lines.append(TextLine(text,(x+7,y,x+145,y+22),.99))
        return lines


@pytest.mark.parametrize('direction',['up','down'])
def test_full_conversation_and_all_exports_are_oldest_first(tmp_path,direction):
    source=Conversation(direction)
    initial=source.position
    output=tmp_path/direction
    result=run_capture(source,output,config=CaptureConfig(max_frames=0,direction=direction,
        sample_interval_s=.001,stable_samples=2,settle_delay_s=0,scroll_event_interval_s=0,boundary_wait_s=0),ocr_backend=KnownText(source))
    assert source.closed
    assert result['status']=='boundary_reached' and result['boundary_reached']
    assert result['boundary']==('top' if direction=='up' else 'bottom')
    assert result['capture_mode']=='until_boundary' and result['timeline_order']=='oldest_first'
    assert source.position==(0 if direction=='up' else source.end)
    assert result['final_scroll_offset_px']==source.position-initial
    assert all(units<0 if direction=='up' else units>0 for units in source.inputs)
    expected=[text for _,text in source.messages]
    rows=read_jsonl(output/'messages.jsonl')
    assert [row['text'] for row in rows]==expected
    assert len(rows)==30 and all(row['complete'] for row in rows)
    assert [row['global_top_px'] for row in rows]==sorted(row['global_top_px'] for row in rows)
    store=RecordSource(output)
    assert [row['text'] for row in store.records()]==expected
    exported=export_records(store,tmp_path/'exports',FORMATS,ExportOptions())
    for file in exported['files']:
        from pathlib import Path
        path=Path(file)
        if path.suffix in {'.json','.jsonl'}:assert [row['text'] for row in RecordSource(path).records()]==expected
        else:
            body=path.read_text(encoding='utf-8-sig')
            positions=[body.index(text) for text in expected]
            assert positions==sorted(positions)
    report=json.loads((output/'report.json').read_text(encoding='utf-8'))
    assert report['boundary_reached'] and report['timeline_order']=='oldest_first'
    assert report['image_memory_remaining_bytes']==0
    assert not list(tmp_path.rglob('*.png'))


@pytest.mark.parametrize('reject_boundary',[False,True])
def test_unlimited_loop_passes_100_frames_and_counts_recovered_zero_motion(tmp_path,monkeypatch,reject_boundary):
    from welog import controller
    class Source:
        position=0
        closed=False
        def read(self):return np.zeros((16,16,3),np.uint8)
        def scroll(self,units):self.position=min(105,self.position+1)
        def close(self):self.closed=True
    source=Source()
    class Worker:
        previous=0
        reject_next=reject_boundary
        def __init__(self,output,budget,*args):self.output,self.budget=output,budget
        def submit(self,lease,*args):
            delta=source.position-self.previous
            accepted=True
            if source.position==105 and delta==0 and reject_boundary:
                accepted=not self.reject_next
                self.reject_next=not self.reject_next
            if accepted:self.previous=source.position
            lease.release()
            future=Future()
            future.set_result({'alignment':{'accepted':accepted,'reason':'aligned','delta_px':delta if accepted else None},
                'scroll_offset_px':source.position if accepted else None,'motion_method':'fixture','reference_id':None,'ocr_error':None})
            return future
        def finish(self,summary):
            summary.update(image_memory_peak_bytes=self.budget.peak_bytes,image_memory_remaining_bytes=self.budget.used_bytes)
            store=MessageStore(self.output/'messages.sqlite3')
            try:return store.export(self.output,summary)
            finally:store.close()
    def stable(source,config,alignment,stopped,diagnostics,budget):
        diagnostics.update(reason='fixture')
        return source.read()
    monkeypatch.setattr(controller,'OcrWorker',Worker)
    monkeypatch.setattr(controller,'wait_for_stable',stable)
    observed=[]
    result=run_capture(source,tmp_path/'capture',config=CaptureConfig(max_frames=0,settle_delay_s=0,
        boundary_wait_s=0,max_scroll_events=1),progress=observed.append,stop_requested=lambda:len(observed)>300)
    assert result['status']=='boundary_reached'
    assert result['accepted']==109 and len(observed)>100
    assert result['rejected']==(3 if reject_boundary else 0)
    assert result['final_scroll_offset_px']==105 and source.closed


def test_full_mode_stop_is_not_reported_as_complete(tmp_path):
    source=Conversation('up')
    observed=[]
    result=run_capture(source,tmp_path/'capture',config=CaptureConfig(max_frames=0,direction='up',
        sample_interval_s=.001,stable_samples=2,settle_delay_s=0,boundary_wait_s=0),
        ocr_backend=KnownText(source),progress=observed.append,stop_requested=lambda:len(observed)>=2)
    assert result['status']=='stopped' and not result['boundary_reached']
    assert result['boundary'] is None and source.closed


def test_boundary_loading_that_resumes_scroll_is_not_an_early_end(tmp_path):
    class LoadingConversation(Conversation):
        def __init__(self):
            super().__init__('down')
            self.final_end=self.end
            self.end=1200
            self.stationary_inputs=0
        def scroll(self,units):
            if self.position==self.end and self.end!=self.final_end:
                self.stationary_inputs+=1
                if self.stationary_inputs==2:self.end=self.final_end
            super().scroll(units)
    source=LoadingConversation()
    result=run_capture(source,tmp_path/'capture',config=CaptureConfig(max_frames=0,
        sample_interval_s=.001,stable_samples=2,settle_delay_s=0,scroll_event_interval_s=0,boundary_wait_s=0),ocr_backend=KnownText(source))
    assert result['status']=='boundary_reached' and source.position==source.final_end
    assert result['messages']==30 and result['complete']==30


def test_invalid_limits_are_rejected_and_zero_means_unlimited():
    assert CaptureConfig(max_frames=0).max_frames==0
    with pytest.raises(ValueError):CaptureConfig(max_frames=-1)
    with pytest.raises(ValueError):CaptureConfig(boundary_wait_s=-1)
