"""Text-only exports: aliases, complete metadata, bounded previews and source safety."""
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from welog.exports import ExportOptions, FORMATS, RecordSource, discover_sessions, export_records, filename_stem, preview_records
from welog.message_store import MessageStore


@pytest.fixture
def source(tmp_path):
    rows=[
        {'message_id':'a','direction':'incoming','type':'text','status':'complete','complete':True,
         'text':'你好 <script>alert(1)</script>\n第二行 ``` 保留原文','evidence':[{'frame_id':'000001','box':[1,2,30,40]}]},
        {'message_id':'b','direction':'outgoing','type':'audio','status':'complete','complete':True,'text':"19''",'duration_seconds':19},
        {'message_id':'c','direction':'unknown','sender':'原始昵称','type':'text','status':'partial','complete':False,'text':'=SUM(A1:A2)'},
    ]
    path=tmp_path/'原始记录.jsonl'
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    return RecordSource(path)


def test_all_formats_preserve_chinese_multiline_audio_status_and_source(source,tmp_path):
    before=hashlib.sha256(source.path.read_bytes()).hexdigest()
    options=ExportOptions('林间 & <好友>','自己','周末对话')
    result=export_records(source,tmp_path/'导出',FORMATS,options)
    assert result['count']==3
    files={Path(name).suffix[1:]:Path(name) for name in result['files']}
    assert set(files)==set(FORMATS)
    payload=json.loads(files['json'].read_text(encoding='utf-8'))
    rows=payload['messages']
    assert [row['sender'] for row in rows]==['林间 & <好友>','自己','原始昵称']
    assert rows[0]['evidence']==list(source.records())[0]['evidence']
    assert rows[0]['text']==list(source.records())[0]['text']
    assert rows[1]['duration_seconds']==19 and rows[1]['type']=='audio'
    assert rows[2]['status']=='partial' and rows[2]['complete'] is False
    assert list(RecordSource(files['json']).records())==rows
    assert list(RecordSource(files['jsonl']).records())==rows
    html=files['html'].read_text(encoding='utf-8')
    assert '<script>' not in html and '&lt;script&gt;' in html
    assert '林间 &amp; &lt;好友&gt;' in html
    assert html.count('<article ')==3
    assert '19&#x27;&#x27;' in html
    with files['csv'].open(encoding='utf-8-sig',newline='') as file:csv_rows=list(csv.reader(file))
    assert csv_rows[1][4]==rows[0]['text']
    assert csv_rows[2][1]=='自己' and csv_rows[2][5]=='19'
    assert csv_rows[3][4]=="'=SUM(A1:A2)"
    assert rows[0]['text'] in files['txt'].read_text(encoding='utf-8')
    assert '````text\n'+rows[0]['text']+'\n````' in files['md'].read_text(encoding='utf-8')
    assert hashlib.sha256(source.path.read_bytes()).hexdigest()==before
    assert not list((tmp_path/'导出').glob('*.tmp'))
    assert not list(tmp_path.rglob('*.png'))


def test_export_never_overwrites_and_sanitizes_windows_names(source,tmp_path):
    first=export_records(source,tmp_path/'out',['json','json'],ExportOptions(title='../CON:<test>'))
    second=export_records(source,tmp_path/'out',['json'],ExportOptions(title='../CON:<test>'))
    assert len(first['files'])==1
    assert first['files']!=second['files']
    assert all(Path(path).parent==tmp_path/'out' for path in first['files']+second['files'])
    assert filename_stem('CON')=='记录_CON'
    assert filename_stem('...')=='聊天记录'
    assert len(filename_stem('字'*100))==60


def test_malformed_input_cleans_all_partial_outputs(source,tmp_path):
    with source.path.open('a',encoding='utf-8') as file:file.write('{broken\n')
    directory=tmp_path/'out'
    with pytest.raises(ValueError,match='第 4 行'):export_records(source,directory,FORMATS)
    assert list(directory.iterdir())==[]


@pytest.mark.parametrize('formats',[[],['bad'],['json','exe']])
def test_invalid_format_has_no_side_effects(source,tmp_path,formats):
    with pytest.raises(ValueError):export_records(source,tmp_path/'out',formats)
    assert not (tmp_path/'out').exists()


def test_preview_search_is_paginated_without_losing_total_count(tmp_path):
    path=tmp_path/'messages.jsonl'
    path.write_text(''.join(json.dumps({'direction':'income' if i%2 else 'outcome','text':f'消息 {i}',
        'complete':i%4!=0})+'\n' for i in range(450)),encoding='utf-8')
    source=RecordSource(path)
    result=preview_records(source,offset=200)
    assert len(result['rows'])==200 and result['rows'][0]['text']=='消息 200'
    assert result['total']==result['matched']==450 and result['incomplete']==113
    assert result['rows'][0]['direction']=='outgoing'
    assert result['rows'][1]['direction']=='incoming'
    assert preview_records(source,search='消息 449')['matched']==1
    assert preview_records(source,search='无匹配')['rows']==[]


def test_sqlite_is_opened_readonly_and_records_can_be_exported(tmp_path):
    directory=tmp_path/'pointer-test'/'capture'
    directory.mkdir(parents=True)
    path=directory/'messages.sqlite3'
    store=MessageStore(path)
    store.connection.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?)',('m','incoming','text',1,90,-80,-20,1,1,0))
    store.connection.execute('INSERT INTO lines(message_id,top,bottom,x1,x2,text,score,clipped) VALUES (?,?,?,?,?,?,?,?)',('m',-75,-30,1,90,'只读消息',.99,0))
    store.close()
    before=path.read_bytes()
    store=MessageStore(path,initialize=False,readonly=True)
    with pytest.raises(sqlite3.OperationalError,match='readonly'):store.connection.execute('DELETE FROM messages')
    store.close()
    source=RecordSource(directory.parent)
    assert list(source.records())[0]['text']=='只读消息'
    assert export_records(source,tmp_path/'out',['txt'])['count']==1
    assert path.read_bytes()==before
    (directory/'messages.jsonl').write_text('',encoding='utf-8')
    assert discover_sessions(tmp_path)==[path]


def test_empty_source_and_wrong_directory(source,tmp_path):
    empty=tmp_path/'empty.jsonl'
    empty.write_text('',encoding='utf-8')
    result=export_records(RecordSource(empty),tmp_path/'out',['json','html'])
    assert result['count']==0
    assert json.loads(Path(result['files'][0]).read_text(encoding='utf-8'))['messages']==[]
    with pytest.raises(ValueError,match='不是目录'):export_records(source,source.path,['txt'])


def test_sqlite_preview_hydrates_only_visible_evidence_in_legacy_store(tmp_path, monkeypatch):
    path = tmp_path/'messages.sqlite3'
    store = MessageStore(path)
    store.connection.execute('DROP INDEX observation_message')
    for i in range(25):
        message_id = str(i)
        store.connection.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?)',
                                 (message_id, 'incoming', 'image', 1, 90, i*100, i*100+80, 1, 1, 0))
        store.connection.execute('INSERT INTO observations VALUES (?,?,?,?,?,?)',
                                 (message_id, '000001', '[1,2,3,4]', 'visible', 'visible', 'ok'))
    store.close()
    before = path.read_bytes()
    calls = []
    original = MessageStore.evidence_for

    def evidence(self, ids):
        calls.append(ids)
        return original(self, ids)

    monkeypatch.setattr(MessageStore, 'evidence_for', evidence)
    page = preview_records(RecordSource(path), offset=10, limit=3)
    assert calls == [['10', '11', '12']] and page['total'] == page['matched'] == 25
    assert page['rows'][0]['evidence'][0]['box'] == [1, 2, 3, 4]
    assert preview_records(RecordSource(path), limit=0)['total'] == 25
    assert len(calls) == 1 and path.read_bytes() == before
