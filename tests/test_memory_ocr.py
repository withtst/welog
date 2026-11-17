import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from welog.assembler import FragmentAssembler
from welog.controller import run_capture
from welog.frame_pool import ImageBudget,ImageBudgetExceeded
from welog.message_store import MessageStore
from welog.models import AlignmentConfig,CaptureConfig,Rect
from welog.pipeline import BoundaryBuffer,OcrWorker
from welog.recognition import Bubble,EmptyOcr,RapidOcrAdapter,TextLine,detect_bubbles
from welog.motion import MotionTracker
from welog.demo import make_scene


def bubble_image(height=500,width=600):
    image=np.full((height,width,3),250,np.uint8)
    cv2.rectangle(image,(55,50),(250,130),(238,238,238),-1)
    cv2.putText(image,'hello',(70,90),cv2.FONT_HERSHEY_SIMPLEX,.7,(20,20,20),1)
    return image


def test_array_adapter_reuses_model_without_temporary_files():
    inputs=[];constructions=[]
    class Engine:
        def __init__(self):constructions.append(1)
        def __call__(self,image,**kwargs):
            assert isinstance(image,np.ndarray)
            inputs.append(image.shape)
            return SimpleNamespace(boxes=np.array([[[1,2],[30,2],[30,20],[1,20]]]),txts=['hello'],scores=[.99])
    adapter=RapidOcrAdapter(Engine)
    assert adapter.recognize(bubble_image())[0].text=='hello'
    adapter.recognize(bubble_image())
    assert len(constructions)==1 and len(inputs)==2


@pytest.mark.parametrize('top,height',[(0,80),(450,50),(10,440),(0,500)])
def test_bubble_detector_keeps_long_and_clipped_bodies(top,height):
    image=np.full((500,600,3),250,np.uint8)
    image[top:top+height,55:500]=(238,238,238)
    found=[b for b in detect_bubbles(image) if b.kind=='text']
    assert len(found)==1
    assert (found[0].top_state=='clipped')==(top==0)
    assert (found[0].bottom_state=='clipped')==(top+height==500)


@pytest.mark.parametrize('offsets',[[0,300,600],[600,300,0]])
def test_three_screen_message_assembles_in_both_directions(tmp_path,offsets):
    store=MessageStore(tmp_path/'data.sqlite3');assembler=FragmentAssembler(store)
    try:
        for index,offset in enumerate(offsets):
            y1,y2=max(0,50-offset),min(500,1050-offset)
            lines=[TextLine(text,(70,y-offset,190,y-offset+18),.99) for y,text in [(100,'one'),(400,'two'),(700,'three'),(1000,'four')]
                   if y>=offset and y+18<=offset+500]
            bubble=Bubble((55,y1,220,y2),'incoming',top_state='visible' if offset<=50 else 'clipped',
                          bottom_state='visible' if offset+500>=1050 else 'clipped',lines=lines,ocr_status='ok')
            assembler.observe(str(index),bubble,offset,500)
        records=list(store.iter_messages())
        assert len(records)==1
        assert records[0]['complete']
        assert records[0]['text']=='one\ntwo\nthree\nfour'
    finally:store.close()


def test_distinct_identical_messages_and_changed_meaning_are_preserved(tmp_path):
    store=MessageStore(tmp_path/'data.sqlite3');assembler=FragmentAssembler(store)
    try:
        texts=['好','好','下午三点开会','下午三点开会取消']
        for offset in [0,20]:
            for i,text in enumerate(texts):
                top=50+i*80
                bubble=Bubble((55,top-offset,300,top+50-offset),'incoming',lines=[TextLine(text,(70,top+10-offset,260,top+30-offset),.99)],ocr_status='ok')
                assembler.observe(str(offset),bubble,offset,500)
        records=list(store.iter_messages())
        assert [r['text'] for r in records]==texts
        assert len({r['message_id'] for r in records})==4
    finally:store.close()


def test_seam_repair_replaces_a_clipped_line_and_keeps_no_parent_image(tmp_path):
    store=MessageStore(tmp_path/'data.sqlite3');assembler=FragmentAssembler(store)
    try:
        bubble=Bubble((55,440,300,500),'incoming',bottom_state='clipped',
                      lines=[TextLine('截',(70,490,260,500),.99)],ocr_status='partial')
        assembler.observe('first',bubble,0,500)
        assembler.observe('next',Bubble((55,140,300,240),'incoming',top_state='clipped',lines=[],ocr_status='failed'),300,500)
        assembler.repair([TextLine('截断文字已恢复',(70,40,260,60),.8)],450,100)
        record=list(store.iter_messages())[0]
        assert record['text']=='截断文字已恢复' and record['complete']
    finally:store.close()
    budget=ImageBudget(4_000_000);buffer=BoundaryBuffer(budget)
    image=np.full((500,600,3),250,np.uint8);buffer.update(image,0,[])
    assert buffer.top.image.base is None
    patch,top=buffer.compose(image,500,'down')
    assert top<500<top+patch.image.shape[0]
    patch.release();buffer.close();assert budget.used_bytes==0


def no_image_writes(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Image output is forbidden')
    monkeypatch.setattr(cv2,'imwrite',forbidden)
    monkeypatch.setattr(cv2,'imencode',forbidden)
    from welog import imageio
    monkeypatch.setattr(imageio,'write_image',forbidden)
    from PIL import Image
    monkeypatch.setattr(Image.Image,'save',forbidden)
    import tempfile
    monkeypatch.setattr(tempfile,'mkstemp',forbidden)
    monkeypatch.setattr(np,'save',forbidden)


@pytest.mark.parametrize('failure',['none','ocr','unstable','budget'])
def test_capture_and_all_failures_never_write_images_and_release_memory(tmp_path,monkeypatch,failure):
    no_image_writes(monkeypatch)
    rng=np.random.default_rng(11)
    class Source:
        closed=False;scrolls=[]
        def read(self):
            return rng.integers(0,256,(500,600,3),dtype=np.uint8) if failure=='unstable' else bubble_image()
        def scroll(self,units):self.scrolls.append(units)
        def close(self):self.closed=True
    class Backend:
        def recognize(self,image):
            if failure=='ocr':raise RuntimeError('PRIVATE CHAT MUST NOT BE LOGGED')
            return [TextLine('hello',(70,70,220,100),.99)]
    source=Source()
    result=run_capture(source,str(tmp_path/'session'),config=CaptureConfig(max_frames=1,sample_interval_s=.001,
                       stable_timeout_s=.12 if failure in {'unstable','budget'} else .5,image_memory_limit_bytes=1000 if failure=='budget' else 64*1024*1024),ocr_backend=Backend())
    assert source.closed and result['image_memory_remaining_bytes']==0
    assert result['image_memory_peak_bytes']<=64*1024*1024
    assert not list(tmp_path.rglob('*.png')) and not list(tmp_path.rglob('*.npy'))
    assert all('image' not in f for f in __import__('welog.archive',fromlist=['read_jsonl']).read_jsonl(tmp_path/'session/frames.jsonl'))
    if failure=='none':
        records=[json.loads(line) for line in (tmp_path/'session/messages.jsonl').read_text(encoding='utf-8').splitlines()]
        assert records[0]['text']=='hello'
    else:
        assert result['status']=='needs_attention'
        if failure=='unstable':assert result['rejected']==1
        if failure=='ocr':assert result['ocr_failed']==1
    for file in (tmp_path/'session').glob('*.json*'):
        assert 'PRIVATE CHAT MUST NOT BE LOGGED' not in file.read_text(encoding='utf-8')


def test_bounded_worker_applies_backpressure_without_dropping_images(tmp_path):
    budget=ImageBudget(10_000_000);started=threading.Event();release=threading.Event()
    class SlowBackend:
        def recognize(self,image):
            started.set();release.wait(3);return []
    worker=OcrWorker(tmp_path,budget,SlowBackend(),AlignmentConfig(),CaptureConfig(max_frames=1),max_pending=2)
    submitted=threading.Event();futures=[]
    try:
        futures.append(worker.submit(budget.copy(bubble_image()),'0',(),'down'))
        assert started.wait(2)
        futures.append(worker.submit(budget.copy(bubble_image()),'1',(),'down'))
        def third():
            futures.append(worker.submit(budget.copy(bubble_image()),'2',(),'down'));submitted.set()
        thread=threading.Thread(target=third);thread.start();time.sleep(.05)
        assert not submitted.is_set()
        release.set();thread.join(5)
        assert submitted.is_set()
        for future in futures:future.result(timeout=5)
    finally:
        release.set();worker.finish({'status':'stopped'})
    assert budget.used_bytes==0


def test_text_box_jitter_does_not_accumulate_into_scroll_coordinates():
    scene=make_scene(1200,640);budget=ImageBudget();tracker=MotionTracker(budget,AlignmentConfig())
    try:
        a=[TextLine('anchor one',(70,100,260,120)),TextLine('anchor two',(70,200,260,220))]
        b=[TextLine('anchor one',(70,64,260,84)),TextLine('anchor two',(70,164,260,184))]
        tracker.accept(scene[:720],'first',a,'down')
        result,_,method=tracker.accept(scene[37:757],'second',b,'down')
        assert result.accepted and result.delta_px==37 and method=='text'
    finally:tracker.close()
    assert budget.used_bytes==0


def test_photo_containing_chat_bubbles_does_not_become_plain_chat_text():
    image=np.full((500,600,3),250,np.uint8)
    image[50:350,55:230]=255
    image[70:95,75:100]=(20,20,180)
    image[150:220,75:200]=(238,238,238)
    found=detect_bubbles(image)
    assert any(b.kind!='text' and b.box[1]<=50 and b.box[3]>=350 for b in found)
    assert not any(b.kind=='text' for b in found)


@pytest.mark.skipif(not Path('runs/pointer-20261116-204033-bf2111/capture/frames/000005.png').exists(),reason='Private offline fixture unavailable')
def test_real_gif_failure_uses_static_features_without_any_window_input():
    from welog.archive import read_jsonl
    from welog.imageio import read_image
    root=Path('runs/pointer-20261116-204033-bf2111/capture')
    records=read_jsonl(root/'frames.jsonl');budget=ImageBudget();tracker=MotionTracker(budget,AlignmentConfig())
    try:
        for index in [4,5]:
            masks=tuple(Rect(**box) for box in records[index]['action']['stability']['dynamic_masks'])
            result,_,method=tracker.accept(read_image(root/records[index]['image']),str(index),[],'up',masks)
        assert result.accepted and result.delta_px==-300 and method=='static_features'
    finally:tracker.close()


def test_full_pipeline_reocr_repairs_a_cut_line_from_rolling_bands(tmp_path):
    scene=np.full((1000,600,3),250,np.uint8);scene[100:650,55:300]=238
    for y,text in [(350,'anchor one'),(400,'anchor two'),(498,'restored')]:
        cv2.putText(scene,text,(70,y),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1)
    class Backend:
        full_calls=0;seam_calls=0
        def recognize(self,image):
            if image.shape[0]==500:
                self.full_calls+=1;offset=0 if self.full_calls==1 else 300
                lines=[TextLine(text,(70,y-offset,260,y-offset+18),.99) for y,text in [(350,'anchor one'),(400,'anchor two')]]
                if offset==0:lines.append(TextLine('cut',(70,490,260,500),.99))
                return lines
            self.seam_calls+=1
            return [TextLine('restored',(70,156,260,176),.9)]
    backend=Backend();budget=ImageBudget();worker=OcrWorker(tmp_path,budget,backend,AlignmentConfig(),CaptureConfig(direction='down'))
    outcomes=[]
    for i,offset in enumerate([0,300]):
        outcomes.append(worker.submit(budget.copy(scene[offset:offset+500]),str(i),(),'down').result())
    worker.finish({'status':'offline'})
    assert all(o['alignment']['accepted'] for o in outcomes)
    assert backend.seam_calls==1
    rows=[json.loads(line) for line in (tmp_path/'messages.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==1 and rows[0]['text']=='anchor one\nanchor two\nrestored'
    assert rows[0]['complete'] and budget.used_bytes==0


def test_thousand_boundary_buffers_have_constant_memory_and_release_all_images():
    budget=ImageBudget(4_000_000);buffer=BoundaryBuffer(budget);image=bubble_image()
    observed=[]
    for index in range(1000):
        with budget.copy(image):buffer.update(image,float(index),[])
        if index%100==0:observed.append(budget.used_bytes)
    assert len(set(observed))==1
    assert budget.peak_bytes<=4_000_000
    buffer.close();assert budget.used_bytes==0


def test_stop_during_ocr_finishes_owned_buffer_without_any_next_scroll(tmp_path):
    stopped=threading.Event()
    class Source:
        scrolls=[]
        def read(self):return bubble_image()
        def scroll(self,units):self.scrolls.append(units)
        def close(self):pass
    class Backend:
        def recognize(self,image):
            stopped.set();time.sleep(.12);return []
    source=Source()
    result=run_capture(source,str(tmp_path/'capture'),config=CaptureConfig(sample_interval_s=.001),
                       ocr_backend=Backend(),stop_requested=stopped.is_set)
    assert result['status']=='stopped' and not source.scrolls
    assert result['image_memory_remaining_bytes']==0
