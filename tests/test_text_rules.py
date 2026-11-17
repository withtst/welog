import json
import sqlite3

import numpy as np
import pytest

from welog.assembler import FragmentAssembler
from welog.frame_pool import ImageBudget
from welog.message_store import MessageStore
from welog.models import AlignmentConfig,CaptureConfig
from welog.pipeline import OcrWorker
from welog.recognition import Bubble,TextLine,assign_lines
from welog.text_rules import audio_duration,clean_notice,sanitize_store


@pytest.mark.parametrize('source,expected',[
    ('3条新消息',''),('１２ 条 新 消 息',''),('99+条新消息',''),
    ('正文\n3条新消息','正文'),('第一句\n3条新消息\n第二句','第一句\n第二句'),
    ('正文3条新消息继续','正文继续'),('正文2条新消息和7条新消息','正文和'),
    ('我收到新消息了','我收到新消息了'),('这是三条新消息','这是三条新消息'),
    ('  保留原空格  ','  保留原空格  '),
])
def test_unread_notice_filter_removes_only_numbered_overlay(source,expected):
    assert clean_notice(source)[0]==expected


@pytest.mark.parametrize('text',["19''",'19‘’','19’‘','19′′','19″','19"','19”',"【19''】",'１９＇＇','19\n‘’','19 ’ ’'])
def test_whole_duration_variants_are_audio(text):
    assert audio_duration(text)==19


@pytest.mark.parametrize('text',["19'",'19','19秒',"等19''再来",'语音19"','19"后见',"19''\n其他正文",'[19"'])
def test_duration_inside_prose_or_incomplete_marker_is_not_audio(text):
    assert audio_duration(text) is None


def test_standalone_overlay_bubble_is_removed_but_real_empty_ocr_candidate_remains():
    bubbles=[Bubble((40,40,250,90),'incoming'),Bubble((40,140,250,190),'incoming')]
    labels=assign_lines(bubbles,[TextLine('3条新消息',(50,50,220,80)),TextLine('4条新消息',(280,280,390,300))])
    assert len(bubbles)==1 and bubbles[0].box[1]==140
    assert bubbles[0].ocr_status=='failed'
    assert labels==[]


def test_embedded_notice_is_removed_before_motion_anchors_and_audio_classification():
    bubbles=[Bubble((40,40,250,120),'incoming')]
    assign_lines(bubbles,[TextLine('19‘’\n3条新消息',(50,50,220,100))])
    assert bubbles[0].kind=='audio'
    assert bubbles[0].lines[0].text=='19‘’'
    assert bubbles[0].lines[0].notice_removed


def test_audio_export_preserves_duration_text_and_separate_occurrences(tmp_path):
    store=MessageStore(tmp_path/'data.sqlite3');assembler=FragmentAssembler(store)
    try:
        for i in range(2):
            y=50+i*90
            assembler.observe(str(i),Bubble((40,y,250,y+50),'incoming',lines=[TextLine('19‘’',(50,y+10,220,y+35),.99)]),0,500)
        records=list(store.iter_messages())
        assert len(records)==2 and len({r['message_id'] for r in records})==2
        assert all(r['type']=='audio' and r['duration_seconds']==19 and r['text']=='19‘’' for r in records)
        assert all(r['complete'] for r in records)
    finally:store.close()


def test_seam_repair_cannot_reintroduce_notice_and_can_classify_duration(tmp_path):
    store=MessageStore(tmp_path/'data.sqlite3');assembler=FragmentAssembler(store)
    try:
        assembler.observe('first',Bubble((40,40,250,120),'incoming',lines=[],ocr_status='failed'),0,500)
        assembler.repair([TextLine('3条新消息',(50,50,220,80))],0,200)
        assert list(store.iter_messages())[0]['text']==''
        assembler.repair([TextLine('19"\n4条新消息',(50,50,220,80),.99)],0,200)
        row=list(store.iter_messages())[0]
        assert row['type']=='audio' and row['duration_seconds']==19 and row['text']=='19"'
    finally:store.close()


def test_existing_structured_results_are_repaired_without_deleting_real_text(tmp_path):
    store=MessageStore(tmp_path/'data.sqlite3')
    try:
        db=store.connection
        for i,text in enumerate(['正文\n3条新消息','4条新消息','19‘’','我收到新消息了']):
            identity=str(i)
            db.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?,?)',(identity,'incoming','text',40,250,i*90,i*90+50,1,1,0))
            db.execute('INSERT INTO lines(message_id,top,bottom,x1,x2,text,score,clipped) VALUES(?,?,?,?,?,?,?,?)',
                       (identity,i*90+10,i*90+35,50,220,text,.99,0))
        db.commit()
        stats=sanitize_store(store)
        assert stats=={'cleaned_lines':2,'removed_messages':1,'classified_audio':1}
        rows=list(store.iter_messages())
        assert [r['text'] for r in rows]==['正文','19‘’','我收到新消息了']
        assert rows[1]['type']=='audio' and rows[1]['duration_seconds']==19
    finally:store.close()


def test_full_ocr_worker_filters_overlay_and_exports_audio_without_images(tmp_path,monkeypatch):
    import cv2
    def forbidden(*args,**kwargs):raise AssertionError('Image persistence forbidden')
    monkeypatch.setattr(cv2,'imwrite',forbidden)
    monkeypatch.setattr(cv2,'imencode',forbidden)
    image=np.full((500,600,3),250,np.uint8)
    for top in [50,150,250]:image[top:top+55,55:260]=238
    class Backend:
        def recognize(self,bgr):
            return [TextLine('3条新消息',(70,60,230,90)),
                    TextLine('正常正文\n2条新消息',(70,160,230,190)),
                    TextLine('19‘’',(70,260,230,290))]
    budget=ImageBudget();worker=OcrWorker(tmp_path,budget,Backend(),AlignmentConfig(),CaptureConfig())
    outcome=worker.submit(budget.copy(image),'0',(),'up').result()
    worker.finish({'status':'offline'})
    assert len(outcome['observations'])==2
    rows=[json.loads(s) for s in (tmp_path/'messages.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['text'] for r in rows]==['正常正文','19‘’']
    assert rows[1]['type']=='audio' and rows[1]['duration_seconds']==19
    assert not list(tmp_path.rglob('*.png')) and budget.used_bytes==0
