"""Read existing text results and export copies. No capture or image dependencies."""
from __future__ import annotations

import csv
import html
import json
import os
import re
import hashlib
import sqlite3
from urllib.parse import quote
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime,timezone
from pathlib import Path
from uuid import uuid4, uuid5, NAMESPACE_URL

from .message_store import MessageStore

FORMATS={'txt':'TXT','md':'Markdown','html':'HTML','csv':'CSV','json':'JSON','jsonl':'JSONL'}
TYPE_NAMES={'text':'文字','audio':'语音','image':'图片','file':'文件','share':'分享','unknown':'未知'}
STATUS_NAMES={'complete':'完整','partial':'待补全','ocr_failed':'识别失败'}
DIRECTION_NAMES={'incoming':'对方','outgoing':'我方','unknown':'未知'}


class RecordSource:
    def __init__(self,path):
        path=Path(path).expanduser().resolve()
        if path.is_dir():
            path=next((candidate for candidate in [path/'messages.sqlite3',path/'messages.jsonl',path/'capture/messages.sqlite3',path/'capture/messages.jsonl']
                       if candidate.is_file()),None)
            if path is None:raise ValueError('该目录没有可读取的消息记录')
        if not path.is_file() or path.suffix.lower() not in {'.sqlite3','.db','.jsonl','.json'}:
            raise ValueError('请选择 WeLog SQLite、JSONL 或 JSON 记录')
        self.path=path

    def records(self):
        from .record_state import project_records
        yield from project_records(self)

    def raw_records(self):
        suffix=self.path.suffix.lower()
        if suffix in {'.sqlite3','.db'}:
            store=MessageStore(self.path,initialize=False,readonly=True)
            try:
                for row in store.iter_messages():yield self.normalize(row)
            finally:store.close()
        elif suffix=='.jsonl':
            with self.path.open(encoding='utf-8-sig') as file:
                for index,line in enumerate(file,1):
                    if not line.strip():continue
                    try:yield self.identify(self.normalize(json.loads(line)),index)
                    except (ValueError,TypeError) as error:raise ValueError(f'JSONL 第 {index} 行不是有效消息记录') from error
        else:
            if self.path.stat().st_size>64*1024*1024:raise ValueError('JSON 超过 64 MiB，请改用 JSONL 或 SQLite 查看')
            payload=json.loads(self.path.read_text(encoding='utf-8-sig'))
            rows=payload if isinstance(payload,list) else payload.get('messages') if isinstance(payload,dict) else None
            if not isinstance(rows,list):raise ValueError('JSON 必须包含 messages 数组或是消息数组')
            for index,row in enumerate(rows,1):yield self.identify(self.normalize(row),index)

    def identify(self,row,index):
        if not row.get('message_id'):
            fingerprint=hashlib.sha256(json.dumps(row,ensure_ascii=False,sort_keys=True).encode('utf-8')).hexdigest()
            row={**row,'message_id':str(uuid5(NAMESPACE_URL,self.path.as_uri()+'#'+str(index)+'#'+fingerprint))}
        return row

    @staticmethod
    def normalize(row):
        if not isinstance(row,dict) or not isinstance(row.get('text'),str):raise ValueError('消息缺少文字字段')
        aliases={'income':'incoming','outcome':'outgoing','我方':'outgoing','对方':'incoming'}
        raw_direction=row.get('direction')
        direction=aliases.get(raw_direction,raw_direction or 'unknown') if isinstance(raw_direction,str) else 'unknown'
        if direction not in {'incoming','outgoing','unknown'}:direction='unknown'
        status=row.get('status') or ('complete' if row.get('complete') else 'partial')
        if status not in {'complete','partial','ocr_failed'}:status='partial'
        kind=row.get('type','text')
        return {**row,'direction':direction,'type':kind if isinstance(kind,str) else 'unknown','status':status}


def discover_sessions(root:Path,limit=200):
    found=[]
    if not root.exists():return found
    for directory,children,files in os.walk(root):
        children[:]=[child for child in children if child not in {'frames','.venv','.git','research','gui-preview'}
                     and not child.startswith('replay') and not child.endswith('-validation')]
        name='messages.sqlite3' if 'messages.sqlite3' in files else 'messages.jsonl' if 'messages.jsonl' in files else None
        if name:
            path=Path(directory)/name
            found.append(path)
    return sorted(found,key=lambda path:path.stat().st_mtime,reverse=True)[:limit]


def preview_records(source:RecordSource,*,offset=0,limit=200,search='',time_filter=''):
    from .timeline import interval_label
    rows=[];count=0;total=0;incomplete=0
    type_counts={};direction_counts={};groups={};edited=0
    query=search.casefold().strip()
    for row in source.records():
        total+=1;incomplete+=row['status']!='complete'
        type_counts[row['type']]=type_counts.get(row['type'],0)+1
        direction_counts[row['direction']]=direction_counts.get(row['direction'],0)+1
        edited+=bool(row.get('edited'))
        group=interval_label(row)
        groups[group]=groups.get(group,0)+1
        if time_filter and group!=time_filter:continue
        if query and query not in row['text'].casefold():continue
        if offset<=count<offset+limit:rows.append(row)
        count+=1
    statistics={'total':total,'incomplete':incomplete,'type_counts':type_counts,'direction_counts':direction_counts,'edited':edited,'time_groups':groups}
    from .record_state import describe
    return {'rows':rows,'matched':count,'total':total,'incomplete':incomplete,'offset':offset,'statistics':statistics,'metadata':describe(source,statistics)}


@dataclass(frozen=True)
class ExportOptions:
    incoming_name:str='对方'
    outgoing_name:str='我'
    title:str='聊天记录'

    def sender(self,record):
        return (self.incoming_name.strip() or '对方') if record['direction']=='incoming' else (
            (self.outgoing_name.strip() or '我') if record['direction']=='outgoing' else str(record.get('sender') or '未知'))


def export_record(record,options):
    result={**record}
    if record.get('sender') and record['sender']!=options.sender(record):result['original_sender']=record['sender']
    result['sender']=options.sender(record)
    return result


def safe_csv(value):
    text=str(value)
    return "'"+text if text.lstrip().startswith(('=','+','-','@')) or text.startswith(('\t','\r')) else text


def filename_stem(title):
    stem=re.sub(r'[<>:"/\\|?*\x00-\x1f]','_',title).strip(' .')[:60] or '聊天记录'
    if stem.split('.')[0].upper() in {'CON','PRN','AUX','NUL',*[f'{prefix}{number}' for prefix in ['COM','LPT'] for number in range(1,10)]}:
        stem='记录_'+stem
    return stem


def export_records(source:RecordSource,directory,formats,options:ExportOptions|None=None,progress=None):
    options=options or ExportOptions()
    selected=list(dict.fromkeys(formats))
    if not selected or any(key not in FORMATS for key in selected):raise ValueError('请选择有效的导出格式')
    directory=Path(directory).expanduser().resolve()
    if directory.exists() and not directory.is_dir():raise ValueError('导出位置不是目录')
    directory.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d-%H%M%S')
    stem=f'{filename_stem(options.title)}-{stamp}-{uuid4().hex[:8]}'
    paths={key:directory/f'{stem}.{key}' for key in selected}
    temporary={key:directory/f'.{stem}.{key}.tmp' for key in selected}
    metadata_path=directory/f'{stem}.metadata.json'
    assets=directory/f'{stem}.assets'
    assets_created=False
    created=[]
    for path in [*paths.values(),*temporary.values()]:
        if path.resolve().parent!=directory:raise ValueError('非法导出路径')
    metadata={'created_at':datetime.now(timezone.utc).isoformat(),'title':options.title,
              'incoming_name':options.incoming_name.strip() or '对方','outgoing_name':options.outgoing_name.strip() or '我',
              'source_name':source.path.name}
    from .record_state import describe, note_export, resolve_attachment
    from .timeline import interval_label
    statistics=preview_records(source,limit=0)['statistics']
    metadata['record']=describe(source,statistics)
    copied={}
    def attachment_copy(record):
        nonlocal assets_created
        attachment=record.get('attachment')
        if not attachment:return record,None
        original=resolve_attachment(source,record)
        if original is None:return {**record,'attachment':{**attachment,'path':None,'available':False}},None
        key=str(original)
        if key not in copied:
            if not assets_created:
                assets.mkdir()
                assets_created=True
            target=assets/(uuid4().hex+original.suffix.lower())
            created.append(target)
            digest=hashlib.sha256()
            with original.open('rb') as read,target.open('xb') as write:
                while chunk:=read.read(1024*1024):write.write(chunk);digest.update(chunk)
            if attachment.get('sha256') and digest.hexdigest()!=attachment['sha256']:raise ValueError('附件已变化，请重新替换附件')
            copied[key]=target.relative_to(directory).as_posix()
        relative=copied[key]
        return {**record,'attachment':{**attachment,'path':relative,'available':True}},relative
    try:
        for path in paths.values():
            with path.open('x',encoding='utf-8'):pass
            created.append(path)
        with ExitStack() as stack:
            handles={}
            for key in selected:
                handles[key]=stack.enter_context(temporary[key].open('x',encoding='utf-8-sig' if key=='csv' else 'utf-8',newline=''))
                created.append(temporary[key])
            csv_writer=csv.writer(handles['csv']) if 'csv' in handles else None
            if csv_writer:csv_writer.writerow(['序号','昵称','方向','类型','内容','时长秒','状态','消息ID','时间区间','时区','附件'])
            if 'txt' in handles:handles['txt'].write(options.title+'\n\n')
            if 'md' in handles:handles['md'].write('# '+html.escape(options.title)+'\n\n')
            if 'json' in handles:
                handles['json'].write('{"schema_version":1,"export":')
                json.dump(metadata,handles['json'],ensure_ascii=False)
                handles['json'].write(',"messages":[')
            if 'html' in handles:
                handles['html'].write('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                    '<meta name="viewport" content="width=device-width,initial-scale=1">'
                    '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; img-src \'self\' file: data:">'
                    '<title>'+html.escape(options.title)+'</title><style>'
                    'body{background:#f8f5ef;color:#26352f;font:15px system-ui;max-width:940px;margin:40px auto;padding:0 24px}'
                    '.message{max-width:78%;padding:16px 20px;margin:16px 0;background:#fffdf9;border:1px solid white;border-radius:22px;box-shadow:0 8px 28px #26352f0a}'
                    '.outgoing{margin-left:auto;background:#e2f1eb}.meta{font-size:12px;color:#61776d;margin-bottom:8px}'
                    '.body{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%;max-height:520px;border-radius:12px}h1{font-size:28px;font-weight:500}</style>'
                    '<h1>'+html.escape(options.title)+'</h1>')
            count=0
            for count,original in enumerate(source.records(),1):
                record=export_record(original,options)
                record,attachment=attachment_copy(record)
                sender=record['sender'];body=record['text'];kind=TYPE_NAMES.get(record['type'],record['type'])
                status=STATUS_NAMES.get(record['status'],record['status'])
                time_label=interval_label(record)
                if 'jsonl' in handles:handles['jsonl'].write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                if 'json' in handles:
                    if count>1:handles['json'].write(',')
                    json.dump(record,handles['json'],ensure_ascii=False,allow_nan=False)
                if csv_writer:csv_writer.writerow([safe_csv(value) for value in [count,sender,record['direction'],kind,body,
                                                    record.get('duration_seconds',''),status,record.get('message_id',''),time_label,'UTC+08:00',attachment or '']])
                if 'txt' in handles:handles['txt'].write(f'[{sender}] · {kind} · {status} · {time_label}\n{body}\n'+(f'附件：{attachment}\n' if attachment else '')+'\n')
                if 'md' in handles:
                    fence='`'*(max([len(match) for match in re.findall(r'`+',body)]+[2])+1)
                    handles['md'].write(f'### {count} · {html.escape(sender)} · {kind} · {status}\n\n{time_label}\n\n{fence}text\n{body}\n{fence}\n\n')
                    if attachment:handles['md'].write(f'{"!" if record["type"]=="image" else ""}[附件]({quote(attachment)})\n\n')
                if 'html' in handles:
                    direction=record['direction'] if record['direction'] in {'incoming','outgoing'} else 'unknown'
                    handles['html'].write(f'<article class="message {direction}"><div class="meta">'+
                        html.escape(f'{sender} · {kind} · {status} · {time_label}')+'</div><div class="body">'+html.escape(body)+'</div>')
                    if attachment:
                        href=html.escape(quote(attachment),quote=True)
                        if record['type']=='image' and record['attachment'].get('mime_type') in {'image/png','image/jpeg','image/webp','image/gif','image/bmp'}:
                            handles['html'].write(f'<p><img src="{href}" alt="图片附件" loading="lazy"></p>')
                        handles['html'].write(f'<a href="{href}" download>'+html.escape(record['attachment'].get('name','附件'))+'</a>')
                    handles['html'].write('</article>')
                if progress and count%100==0:progress(count)
            if 'json' in handles:handles['json'].write(']}\n')
            if 'html' in handles:handles['html'].write(f'<p>共 {count} 条记录</p></html>')
        for key in selected:os.replace(temporary[key],paths[key])
        with metadata_path.open('x',encoding='utf-8') as file:
            created.append(metadata_path)
            json.dump(metadata,file,ensure_ascii=False,indent=2)
        result={'count':count,'files':[str(paths[key]) for key in selected],'directory':str(directory),
                'metadata_file':str(metadata_path),'attachments':len(copied),'statistics':statistics}
        try:note_export(source,{**result,'formats':selected})
        except (OSError,sqlite3.Error):result['metadata_saved_to_source']=False
        return result
    except BaseException:
        for path in created:
            if path.parent in {directory,assets} and path.is_file():path.unlink()
        if assets_created and assets.parent==directory:assets.rmdir()
        raise
