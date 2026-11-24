"""Names, edits and user-imported attachments live beside immutable OCR source data."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from .timeline import SCHEMA, interval_for

KINDS=('text','audio','image','file','share','unknown')
IMAGE_EXTENSIONS={'.png','.jpg','.jpeg','.webp','.gif','.bmp'}


def state_path(source):
    path=source.path
    return path.parent/'record-state.sqlite3' if path.name in {'messages.sqlite3','messages.jsonl'} else path.with_name('.'+path.name+'.welog.sqlite3')


def asset_root(source):
    return state_path(source).with_suffix('.assets')


@contextmanager
def connect(source,write=False):
    path=state_path(source)
    if not write and not path.exists():
        yield None
        return
    db=sqlite3.connect(path if write else path.as_uri()+'?mode=ro',uri=not write)
    db.row_factory=sqlite3.Row
    try:
        if write:
            db.executescript('''CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT);
                CREATE TABLE IF NOT EXISTS edits(message_id TEXT PRIMARY KEY,patch TEXT,updated_at TEXT);
                CREATE TABLE IF NOT EXISTS edit_history(id INTEGER PRIMARY KEY,message_id TEXT,previous_patch TEXT,updated_at TEXT);
                CREATE TABLE IF NOT EXISTS export_events(id INTEGER PRIMARY KEY,created_at TEXT,details TEXT);'''+SCHEMA)
        yield db
        if write:db.commit()
    except BaseException:
        if write:db.rollback()
        raise
    finally:db.close()


def now():return datetime.now(timezone.utc).isoformat()


def get_meta(db,key,default=None):
    row=db.execute('SELECT value FROM metadata WHERE key=?',(key,)).fetchone() if db else None
    return json.loads(row[0]) if row else default


def set_meta(db,key,value):
    db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(key,json.dumps(value,ensure_ascii=False)))


def record_name(source):
    with connect(source) as db:name=get_meta(db,'name')
    if name:return name
    path=source.path
    name=path.parent.parent.name if path.parent.name=='capture' else path.parent.name
    if name.startswith('pointer-') and len(name)>=23:
        name=name[8:12]+'-'+name[12:14]+'-'+name[14:16]+' '+name[17:19]+':'+name[19:21]+':'+name[21:23]
    return name or path.stem


def rename_record(source,name):
    name=name.strip()
    if not name or len(name)>120 or any(ord(char)<32 for char in name):raise ValueError('记录名称须为 1–120 个字符')
    with connect(source,True) as db:set_meta(db,'name',name)


def project_records(source, *, include_evidence=True):
    from .file_cards import reclassify_stored_record
    with connect(source) as db:
        markers=bool(db and db.execute('SELECT 1 FROM time_markers LIMIT 1').fetchone())
        for row in source.raw_records(include_evidence=include_evidence):
            original=row
            row=reclassify_stored_record(row)
            edit=db.execute('SELECT patch,updated_at FROM edits WHERE message_id=?',(row['message_id'],)).fetchone() if db else None
            if edit:
                row={**row,**json.loads(edit['patch']),'edited':True,'edited_at':edit['updated_at']}
                row['original_message']={key:original.get(key) for key in ['text','type','direction','status','duration_seconds','file'] if key in original}
            if markers and isinstance(row.get('global_top_px'),(int,float)) and isinstance(row.get('global_bottom_px'),(int,float)):
                row={**row,'time_interval':interval_for(db,row['global_top_px'],row['global_bottom_px'])}
            yield row


def save_edit(source,record,patch):
    allowed={'text','type','direction','status','complete','duration_seconds','attachment','file'}
    if set(patch)-allowed:raise ValueError('存在不支持的编辑字段')
    merged={**record,**patch}
    if merged['type'] not in KINDS or merged['direction'] not in {'incoming','outgoing','unknown'} or merged['status'] not in {'complete','partial','ocr_failed'}:
        raise ValueError('无效的消息类型、方向或状态')
    if not isinstance(merged['text'],str):raise ValueError('消息正文必须为文字')
    patch={**patch,'complete':merged['status']=='complete'}
    if merged['type']!='audio':patch['duration_seconds']=None
    with connect(source,True) as db:
        old=db.execute('SELECT patch FROM edits WHERE message_id=?',(record['message_id'],)).fetchone()
        stamp=now()
        db.execute('INSERT INTO edit_history(message_id,previous_patch,updated_at) VALUES (?,?,?)',(record['message_id'],old['patch'] if old else None,stamp))
        value={**(json.loads(old['patch']) if old else {}),**patch}
        db.execute('INSERT OR REPLACE INTO edits VALUES (?,?,?)',(record['message_id'],json.dumps(value,ensure_ascii=False),stamp))


def reset_edit(source,record):
    with connect(source,True) as db:
        old=db.execute('SELECT patch FROM edits WHERE message_id=?',(record['message_id'],)).fetchone()
        if old:db.execute('INSERT INTO edit_history(message_id,previous_patch,updated_at) VALUES (?,?,?)',(record['message_id'],old['patch'],now()))
        db.execute('DELETE FROM edits WHERE message_id=?',(record['message_id'],))


def replace_attachment(source,record,path):
    path=Path(path).resolve()
    if not path.is_file():raise ValueError('附件文件不存在')
    root=asset_root(source)
    root.mkdir(parents=True,exist_ok=True)
    extension=path.suffix.lower() if re.fullmatch(r'\.[a-zA-Z0-9]{1,12}',path.suffix) else ''
    target=root/(uuid4().hex+extension)
    size=0;digest=hashlib.sha256()
    try:
        with path.open('rb') as original,target.open('xb') as copy:
            while block:=original.read(1024*1024):
                digest.update(block);size+=len(block);copy.write(block)
        kind='image' if extension in IMAGE_EXTENSIONS and record['type']!='file' else 'file'
        mime=mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        if record['type']=='audio' and (mime.startswith('audio/') or extension in {'.mp3','.wav','.ogg','.m4a','.aac','.flac','.opus'}):kind='audio'
        if kind=='image':
            with target.open('rb') as file:header=file.read(16)
            valid=header.startswith((b'\x89PNG\r\n\x1a\n',b'\xff\xd8\xff',b'GIF87a',b'GIF89a',b'BM')) or (header[:4]==b'RIFF' and header[8:12]==b'WEBP')
            if not valid:raise ValueError('所选文件不是支持的图片，请检查文件内容')
        attachment={'path':target.relative_to(source.path.parent).as_posix(),'name':path.name,'size_bytes':size,'mime_type':mime,'sha256':digest.hexdigest()}
        patch={'type':kind,'attachment':attachment}
        if kind=='file':patch.update(file={'filename':path.name,'extension':extension.lstrip('.'),'size_bytes':size},text=path.name)
        elif record['text'] in {'[未知消息]','[图片]','[文件]',''}:patch['text']='[图片]'
        save_edit(source,record,patch)
        return attachment
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def resolve_attachment(source,record):
    attachment=record.get('attachment')
    if not isinstance(attachment,dict) or not isinstance(attachment.get('path'),str):return None
    path=Path(attachment['path'])
    if path.is_absolute():return None
    root=source.path.parent.resolve()
    target=(root/path).resolve()
    return target if target.is_relative_to(root) and target.is_file() else None


def capture_metadata(source):
    path=source.path.parent/'manifest.json'
    try:
        value=json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError):return {}


def describe(source,statistics=None):
    manifest=capture_metadata(source)
    with connect(source) as db:
        exports=[]
        if db:
            for row in db.execute('SELECT * FROM export_events ORDER BY id DESC LIMIT 20'):
                details=json.loads(row['details'])
                exports.append({'created_at':row['created_at'],**{key:details[key] for key in ['count','formats','statistics','attachments'] if key in details}})
    return {'name':record_name(source),'capture_started_at':manifest.get('created_at'),
            'capture_finished_at':manifest.get('finished_at'),'capture_direction':manifest.get('capture_direction') or manifest.get('capture_config',{}).get('direction'),
            'last_export_at':exports[0]['created_at'] if exports else manifest.get('finished_at'),
            'export_history':exports,'statistics':statistics or {},'timezone':'UTC+08:00','timeline_order':'oldest_first','source_name':source.path.name}


def note_export(source,details):
    stats=details.get('statistics',{})
    compact={key:value for key,value in stats.items() if key!='time_groups'}
    if 'time_groups' in stats:compact['time_group_count']=len(stats['time_groups'])
    details={**details,'statistics':compact}
    with connect(source,True) as db:
        db.execute('INSERT INTO export_events(created_at,details) VALUES (?,?)',(now(),json.dumps(details,ensure_ascii=False)))


def rebuild_intervals(source):
    """Backfill old recordings from confirmed text journals; no original image is needed."""
    from .recognition import TextLine
    from .timeline import observe_markers,CHAT_TZ
    root=source.path.parent
    alignment,observations=root/'alignments.jsonl',root/'observations.jsonl'
    if not alignment.is_file() or not observations.is_file():return
    signature=[alignment.stat().st_mtime_ns,alignment.stat().st_size,observations.stat().st_mtime_ns,observations.stat().st_size]
    with connect(source) as db:
        if get_meta(db,'timeline_journal_signature')==signature:return
    manifest=capture_metadata(source)
    reference=datetime.fromisoformat(manifest['created_at'].replace('Z','+00:00')) if manifest.get('created_at') else datetime.fromtimestamp(alignment.stat().st_mtime,CHAT_TZ)
    origin=manifest.get('origin_y_px',0)
    with connect(source,True) as db:
        db.execute('DELETE FROM time_markers')
        db.execute('CREATE TEMP TABLE widths(frame_id TEXT PRIMARY KEY,width INTEGER,captured_at TEXT)')
        frame_file=root/'frames.jsonl'
        if frame_file.exists():
            with frame_file.open(encoding='utf-8') as file:
                for line in file:
                    frame=json.loads(line)
                    db.execute('INSERT OR REPLACE INTO widths VALUES (?,?,?)',(frame['frame_id'],frame.get('width',0),frame.get('captured_at')))
        db.execute('CREATE TEMP TABLE positions(frame_id TEXT PRIMARY KEY,offset REAL)')
        with alignment.open(encoding='utf-8') as file:
            for line in file:
                row=json.loads(line)
                if row.get('alignment',{}).get('accepted') and row.get('scroll_offset_px') is not None:
                    db.execute('INSERT OR REPLACE INTO positions VALUES (?,?)',(row['frame_id'],row['scroll_offset_px']-origin))
        with observations.open(encoding='utf-8') as file:
            for line in file:
                row=json.loads(line)
                position=db.execute('SELECT offset FROM positions WHERE frame_id=?',(row['frame_id'],)).fetchone()
                if not position or row.get('coordinate_status')!='confirmed':continue
                labels=[TextLine(item['text'],tuple(item['box']),item.get('score',1)) for item in row.get('labels',[])]
                frame=db.execute('SELECT width,captured_at FROM widths WHERE frame_id=?',(row['frame_id'],)).fetchone()
                frame_reference=datetime.fromisoformat(frame['captured_at'].replace('Z','+00:00')) if frame and frame['captured_at'] else reference
                observe_markers(db,labels,position[0],frame['width'] if frame else 0,row['frame_id'],frame_reference)
        set_meta(db,'timeline_journal_signature',signature)
