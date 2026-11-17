import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from welog.exports import ExportOptions, FORMATS, RecordSource, export_records, preview_records
from welog.record_actions import delete_record,plan_deletion
from welog.record_state import asset_root,describe,rename_record,replace_attachment,reset_edit,resolve_attachment,save_edit,state_path


@pytest.fixture
def source(tmp_path):
    root=tmp_path/'runs'/'pointer-test'/'capture'
    root.mkdir(parents=True)
    path=root/'messages.jsonl'
    rows=[{'message_id':str(i),'direction':'incoming' if i%2==0 else 'outgoing','type':kind,'text':text,
           'status':'complete','complete':True,'global_top_px':100+i*100,'global_bottom_px':160+i*100}
          for i,(kind,text) in enumerate([('text','第一条'),('image','[图片]'),('unknown','[未知消息]')])]
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    (root/'manifest.json').write_text(json.dumps({'created_at':'2026-11-16T14:00:00+00:00','finished_at':'2026-11-16T14:05:00+00:00','capture_direction':'up'}))
    return RecordSource(path)


def test_rename_keeps_source_identity_statistics_and_export_history(source,tmp_path):
    original=hashlib.sha256(source.path.read_bytes()).hexdigest()
    rename_record(source,'对方的名字')
    first=preview_records(source)
    assert first['metadata']['name']=='对方的名字'
    assert first['statistics']['total']==3 and first['statistics']['type_counts']=={'text':1,'image':1,'unknown':1}
    result=export_records(source,tmp_path/'out',['json'],ExportOptions(title='对方的名字'))
    metadata=json.loads(Path(result['metadata_file']).read_text(encoding='utf-8'))
    assert metadata['record']['name']=='对方的名字'
    assert metadata['record']['capture_started_at']=='2026-11-16T14:00:00+00:00'
    assert metadata['record']['statistics']['total']==3
    info=describe(source,first['statistics'])
    assert info['export_history'][0]['count']==3
    assert info['last_export_at']!=info['capture_finished_at']
    assert hashlib.sha256(source.path.read_bytes()).hexdigest()==original
    assert source.path.parent.name=='capture'
    for name in ['','\n','x'*121]:
        with pytest.raises(ValueError):rename_record(source,name)


def test_message_edits_are_persistent_reversible_and_keep_original(source):
    before=source.path.read_bytes()
    row=list(source.records())[0]
    save_edit(source,row,{'text':'人工修正\n第二行','direction':'outgoing','status':'partial'})
    edited=list(RecordSource(source.path).records())[0]
    assert edited['text']=='人工修正\n第二行' and edited['edited']
    assert edited['original_message']['text']=='第一条' and edited['direction']=='outgoing'
    assert not edited['complete']
    assert preview_records(source)['statistics']['edited']==1
    reset_edit(source,edited)
    assert list(source.records())[0]['text']=='第一条'
    assert 'edited' not in list(source.records())[0]
    assert source.path.read_bytes()==before
    with pytest.raises(ValueError):save_edit(source,row,{'global_top_px':-999})


def test_true_image_and_file_attachments_export_portably_in_all_formats(source,tmp_path):
    rows=list(source.records())
    picture=tmp_path/'真实图片.png'
    ok,encoded=cv2.imencode('.png',np.full((30,40,3),(50,160,220),np.uint8))
    assert ok
    picture.write_bytes(encoded.tobytes())
    document=tmp_path/'中文文件.pdf'
    document.write_bytes(b'%PDF-1.4\nEXPLICIT USER ATTACHMENT\n')
    replace_attachment(source,rows[1],picture)
    replace_attachment(source,rows[2],document)
    updated=list(source.records())
    assert updated[1]['type']=='image' and updated[2]['type']=='file'
    assert updated[2]['file']['filename']=='中文文件.pdf'
    assert resolve_attachment(source,updated[1]).read_bytes()==picture.read_bytes()
    picture.unlink()  # Managed copies are independent of the user's original locations.
    document.unlink()
    exported=export_records(source,tmp_path/'out',FORMATS)
    assert exported['attachments']==2
    main=next(Path(file) for file in exported['files'] if Path(file).suffix=='.json')
    portable=list(RecordSource(main).records())
    for row in portable[1:]:
        asset=resolve_attachment(RecordSource(main),row)
        assert asset is not None
        assert hashlib.sha256(asset.read_bytes()).hexdigest()==row['attachment']['sha256']
        assert not Path(row['attachment']['path']).is_absolute()
    html=next(Path(file) for file in exported['files'] if Path(file).suffix=='.html').read_text(encoding='utf-8')
    assert '<img src=' in html and '中文文件.pdf' in html and 'download' in html
    assert str(source.path.parent) not in html
    assert exported['statistics']['type_counts']=={'text':1,'image':1,'file':1}


def test_attachment_validation_does_not_read_outside_record_directory(source,tmp_path):
    row=list(source.records())[0]
    escaped={**row,'attachment':{'path':'../../../../secret.txt'}}
    assert resolve_attachment(source,escaped) is None
    fake=tmp_path/'fake.png';fake.write_text('not an image')
    with pytest.raises(ValueError,match='不是支持的图片'):replace_attachment(source,row,fake)
    assert not list(asset_root(source).iterdir())


def test_imported_record_deletion_removes_its_own_edits_and_assets(tmp_path):
    path=tmp_path/'external'/'record.jsonl'
    path.parent.mkdir()
    path.write_text('{"message_id":"one","type":"unknown","direction":"incoming","status":"partial","text":"[未知消息]"}\n',encoding='utf-8')
    source=RecordSource(path)
    file=tmp_path/'actual.zip';file.write_bytes(b'PK\x03\x04fixture')
    replace_attachment(source,next(source.records()),file)
    rename_record(source,'外部记录')
    assert state_path(source).exists() and asset_root(source).exists()
    plan=plan_deletion(path,tmp_path/'runs')
    delete_record(plan)
    assert not path.exists() and not state_path(source).exists() and not asset_root(source).exists()
    assert file.exists() and path.parent.exists()


def test_audio_attachment_keeps_audio_type_and_duration(source,tmp_path):
    row=list(source.records())[0]
    save_edit(source,row,{'type':'audio','text':"19''",'duration_seconds':19})
    audio=tmp_path/'voice.mp3';audio.write_bytes(b'ID3fixture')
    replace_attachment(source,next(source.records()),audio)
    record=next(source.records())
    assert record['type']=='audio' and record['duration_seconds']==19 and record['attachment']['name']=='voice.mp3'


def test_records_without_ids_can_be_edited_independently(tmp_path):
    path=tmp_path/'import.json'
    path.write_text('[{"text":"one"},{"text":"two"}]',encoding='utf-8')
    source=RecordSource(path)
    first,second=list(source.records())
    assert first['message_id']!=second['message_id']
    save_edit(source,second,{'text':'changed'})
    assert [row['text'] for row in source.records()]==['one','changed']


def test_recent_sessions_skip_offline_validation_artifacts(tmp_path):
    from welog.exports import discover_sessions
    for directory in ['pointer-real','rich-records-validation/snapshot','gui-preview/empty-root/pointer-preview']:
        path=tmp_path/directory/'messages.jsonl'
        path.parent.mkdir(parents=True)
        path.write_text('')
    assert discover_sessions(tmp_path)==[tmp_path/'pointer-real'/'messages.jsonl']
