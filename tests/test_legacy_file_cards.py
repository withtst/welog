"""Old record viewing/export must use current file-card rules without rewriting OCR data."""
import json
import sqlite3

import pytest

from welog.exports import RecordSource,export_records,preview_records
from welog.file_cards import reclassify_stored_record
from welog.message_store import MessageStore
from welog.record_state import reset_edit,save_edit


@pytest.mark.parametrize('body,name',[
    ('第十组.pptx\nP\n30.7M 未下载\n微信电脑版','第十组.pptx'),
    ('汇报ppt.pptx\nP\n1.9M\n微信电脑版','汇报ppt.pptx'),
    ('长文件名称第一行\n第二行.pdf\nPDF\n268.0K\n微信电脑版','长文件名称第一行第二行.pdf'),
])
def test_retained_file_card_text_can_be_reclassified(body,name):
    raw={'message_id':'m','type':'text','text':body,'status':'partial','evidence':[{'frame_id':'old'}]}
    result=reclassify_stored_record(raw)
    assert result['type']=='file' and result['text']==name
    assert result['file']['ocr_text']==body and result['classification_source']=='stored_ocr'
    assert result['status']=='partial' and result['evidence']==raw['evidence']
    assert raw['type']=='text' and raw['text']==body
    assert reclassify_stored_record(result)==result


@pytest.mark.parametrize('body',[
    '[文件]汇报ppt.pptx','请看看第十组.pptx','第十组.pptx\n30.7M',
    '第十组.pptx\n微信电脑版','19\'\'','[未知消息]',
])
def test_insufficient_legacy_evidence_stays_unchanged(body):
    raw={'type':'text','text':body}
    assert reclassify_stored_record(raw)==raw


def test_legacy_sqlite_and_jsonl_counts_match_without_source_writes(tmp_path):
    path=tmp_path/'messages.sqlite3'
    store=MessageStore(path)
    for identity,body in [('file','第十组.pptx\nP\n30.7M 未下载\n微信电脑版'),('prose','[文件]汇报ppt.pptx')]:
        store.connection.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?)',(identity,'incoming','text',40,300,0,100,1,1,0))
        for index,line in enumerate(body.splitlines()):
            store.connection.execute('INSERT INTO lines(message_id,top,bottom,x1,x2,text,score,clipped) VALUES (?,?,?,?,?,?,?,?)',
                                     (identity,index*20,index*20+15,50,200,line,.99,0))
    store.connection.execute('DROP TABLE message_details')
    store.connection.execute('DROP TABLE time_markers')
    store.close()
    before=path.read_bytes()
    source=RecordSource(path)
    assert preview_records(source)['statistics']['type_counts']=={'file':1,'text':1}
    projected=next(row for row in source.records() if row['message_id']=='file')
    assert projected['file']['filename']=='第十组.pptx'
    exported=export_records(source,tmp_path/'out',['json','jsonl'])
    for file in exported['files']:
        assert preview_records(RecordSource(file))['statistics']['type_counts']=={'file':1,'text':1}
    assert path.read_bytes()==before
    with sqlite3.connect(path) as db:assert dict(db.execute('SELECT kind,count(*) FROM messages GROUP BY kind'))=={'text':2}


def test_manual_type_override_wins_and_ids_remain_stable(tmp_path):
    path=tmp_path/'messages.jsonl'
    original={'type':'text','text':'附件.pptx\nP\n30.7M\n微信电脑版'}
    path.write_text(json.dumps(original,ensure_ascii=False)+'\n',encoding='utf-8')
    before=path.read_bytes()
    source=RecordSource(path)
    raw=next(source.raw_records())
    row=next(source.records())
    assert row['type']=='file' and row['message_id']==raw['message_id']
    save_edit(source,row,{'type':'text','text':original['text']})
    assert next(source.records())['type']=='text'
    exported=export_records(source,tmp_path/'out',['json'])
    assert next(RecordSource(exported['files'][0]).records())['type']=='text'
    reset_edit(source,next(source.records()))
    assert next(source.records())['type']=='file'
    assert path.read_bytes()==before
