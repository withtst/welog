"""Streaming structured persistence. This module accepts no image arrays."""
from __future__ import annotations

import html
import json
import sqlite3
from pathlib import Path

from .archive import write_json
from .text_rules import audio_duration


class MessageStore:
    def __init__(self, path: Path, *, initialize: bool = True, readonly: bool = False):
        if readonly and initialize:raise ValueError('Readonly stores cannot initialize schema')
        self.connection = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True) if readonly else sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        if initialize:self.connection.executescript('''
            CREATE TABLE messages(id TEXT PRIMARY KEY, direction TEXT, kind TEXT,
                x1 REAL,x2 REAL,top REAL,bottom REAL,head INTEGER,tail INTEGER,needs_review INTEGER DEFAULT 0);
            CREATE TABLE lines(id INTEGER PRIMARY KEY,message_id TEXT,top REAL,bottom REAL,x1 REAL,x2 REAL,
                text TEXT,score REAL,clipped INTEGER);
            CREATE TABLE observations(message_id TEXT,frame_id TEXT,box TEXT,top_state TEXT,bottom_state TEXT,ocr_status TEXT);
            CREATE INDEX message_location ON messages(direction,top,bottom);
            CREATE INDEX line_location ON lines(message_id,top,bottom);
        ''')
        if initialize:
            from .timeline import SCHEMA
            self.connection.executescript(SCHEMA+'''CREATE TABLE IF NOT EXISTS message_details(message_id TEXT PRIMARY KEY,data TEXT);''')
        self.tables={row[0] for row in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def iter_messages(self):
        # Upward capture discovers older messages later at negative coordinates.
        # Ascending global top is chronological in either direction; never reverse by frame order.
        for message in self.connection.execute("SELECT * FROM messages ORDER BY top,x1,id"):
            lines = self.connection.execute("SELECT * FROM lines WHERE message_id=? ORDER BY top,x1", (message['id'],)).fetchall()
            readable = any(line['text'] for line in lines)
            text = '\n'.join(line['text'] for line in lines)
            duration = audio_duration(text) if message['kind']=='audio' else None
            complete = bool(message['head'] and message['tail'] and not message['needs_review'] and message['kind']!='unknown'
                            and (message['kind']!='audio' or duration is not None)
                            and (message['kind'] not in {'text','audio'} or (readable and not any(line['clipped'] or line['score']<.6 for line in lines))))
            status = 'complete' if complete else 'ocr_failed' if message['kind']=='text' and not readable else 'partial'
            details={}
            if 'message_details' in self.tables:
                detail=self.connection.execute('SELECT data FROM message_details WHERE message_id=?',(message['id'],)).fetchone()
                if detail:details=json.loads(detail['data'])
            if 'time_markers' in self.tables:
                from .timeline import interval_for
                details['time_interval']=interval_for(self.connection,message['top'],message['bottom'])
            yield {**details,'message_id':message['id'],'direction':message['direction'],'type':message['kind'],
                   'text':text if message['kind'] in {'text','audio'} else details.get('file',{}).get('filename',text or '[文件]') if message['kind']=='file' else '[图片]' if message['kind']=='image' else '[未知消息]',
                   **({'duration_seconds':duration} if message['kind']=='audio' else {}),
                   'global_top_px':message['top'],'global_bottom_px':message['bottom'],
                   'top_state':'visible' if message['head'] else 'clipped',
                   'bottom_state':'visible' if message['tail'] else 'clipped',
                   'status':status,'complete':complete,'coordinate_status':'confirmed',
                   'coverage_status':'unknown' if message['needs_review'] else 'continuous',
                   'evidence':[{**dict(row),'box':json.loads(row['box'])} for row in self.connection.execute(
                       "SELECT frame_id,box,top_state,bottom_state,ocr_status FROM observations WHERE message_id=?", (message['id'],))]}

    def export(self, output: Path, summary: dict):
        self.connection.commit()
        counts = {'messages':0,'complete':0,'partial':0,'ocr_failed':0,'type_counts':{},'direction_counts':{}}
        with (output/'messages.jsonl').open('w',encoding='utf-8') as jsonl, (output/'report.html').open('w',encoding='utf-8') as page:
            page.write('<!doctype html><meta charset="utf-8"><title>WeLog 文字采集报告</title><style>'
                       'body{font:15px system-ui;max-width:1100px;margin:32px auto}td,th{border:1px solid #ddd;padding:8px}'
                       'table{border-collapse:collapse;width:100%}td{white-space:pre-wrap}</style>'
                       '<h1>WeLog 文字采集报告</h1><p>'+('调试采集：原始截图保存在 frames 目录。' if summary.get('debug_save_images') else '截图仅在内存处理；本报告不含原图。')+'</p>'
                       '<p>未确认位置的文字保留在 <a href="observations.jsonl">观察记录</a>，不计入已确认消息。</p>')
            page.write('<pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>')
            page.write('<table><tr><th>方向</th><th>类型</th><th>正文</th><th>完整性</th><th>位置</th></tr>')
            for record in self.iter_messages():
                counts['messages']+=1;counts[record['status']]+=1
                for key,field in [('type_counts','type'),('direction_counts','direction')]:
                    counts[key][record[field]]=counts[key].get(record[field],0)+1
                jsonl.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                page.write('<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in [
                    record['direction'],record['type'],record['text'],record['status'],record['global_top_px']])+'</tr>')
            page.write('</table>')
        write_json(output/'report.json',{**summary,**counts})
        return counts

    def close(self):
        self.connection.commit()
        self.connection.close()
