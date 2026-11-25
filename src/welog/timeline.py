"""Visible timestamp separators define intervals, never exact per-message send times."""
from __future__ import annotations
from bisect import bisect_left
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import re
import unicodedata
from uuid import uuid4

CHAT_TZ=timezone(timedelta(hours=8))
TZ_NAME='UTC+08:00'
SCHEMA='''CREATE TABLE IF NOT EXISTS time_markers(
 id TEXT PRIMARY KEY,top REAL,bottom REAL,raw TEXT,time_iso TEXT,precision TEXT,evidence TEXT);
 CREATE INDEX IF NOT EXISTS time_marker_position ON time_markers(top,bottom);'''


def parse_timestamp(text,reference=None):
    reference=(reference or datetime.now(CHAT_TZ)).astimezone(CHAT_TZ)
    value=unicodedata.normalize('NFKC',text).strip()
    value=re.sub(r'\s+',' ',value)
    match=re.fullmatch(r'(?:(\d{4})[年/.-])?(\d{1,2})[月/.-](\d{1,2})日?\s*(?:星期[一二三四五六日天]\s*)?(\d{1,2})[:：](\d{2})',value)
    try:
        if match:
            year,month,day,hour,minute=match.groups()
            inferred=year is None
            year=int(year) if year else reference.year
            date=datetime(year,int(month),int(day),int(hour),int(minute),tzinfo=CHAT_TZ)
            if inferred and date>reference+timedelta(days=1):date=date.replace(year=year-1)
            return date.isoformat(timespec='minutes'),'year_inferred' if inferred else 'explicit'
        match=re.fullmatch(r'(?:(今天|昨天|前天|(?:星期|周)[一二三四五六日天])\s*)?(\d{1,2})[:：](\d{2})',value)
        if not match:return None
        label,hour,minute=match.groups()
        days={'今天':0,'昨天':1,'前天':2}.get(label,0)
        if label and label.startswith(('星期','周')):
            weekday='一二三四五六日'.index(label[-1].replace('天','日'))
            days=(reference.weekday()-weekday)%7
        date=(reference-timedelta(days=days)).replace(hour=int(hour),minute=int(minute),second=0,microsecond=0)
        return date.isoformat(timespec='minutes'),'relative_to_capture_date'
    except ValueError:return None


def observe_markers(db,lines,offset,width,frame_id,reference=None):
    for line in lines:
        x1,y1,x2,y2=line.box
        if not width*.28 <= (x1+x2)/2 <= width*.72 or line.score<.65:continue
        parsed=parse_timestamp(line.text,reference)
        if not parsed:continue
        stamp,precision=parsed
        top,bottom=offset+y1,offset+y2
        old=db.execute('SELECT * FROM time_markers WHERE top BETWEEN ? AND ? AND time_iso=? ORDER BY rowid LIMIT 1',
                       (top-5,top+5,stamp)).fetchone()
        evidence={'frame_id':frame_id,'box':list(line.box),'raw':line.text,'score':line.score}
        if old:
            entries=json.loads(old['evidence'])
            if evidence not in entries:entries.append(evidence)
            db.execute('UPDATE time_markers SET evidence=? WHERE id=?',(json.dumps(entries,ensure_ascii=False),old['id']))
        else:db.execute('INSERT INTO time_markers VALUES (?,?,?,?,?,?,?)',
            (str(uuid4()),top,bottom,line.text,stamp,precision,json.dumps([evidence],ensure_ascii=False)))


def interval_for(db,top,bottom):
    before=db.execute('SELECT * FROM time_markers WHERE bottom<=? ORDER BY bottom DESC LIMIT 1',(top,)).fetchone()
    after=db.execute('SELECT * FROM time_markers WHERE top>=? ORDER BY top LIMIT 1',(bottom,)).fetchone()
    return _interval(before, after)


def _interval(before, after, include_evidence=True):
    start,end=(before['time_iso'] if before else None),(after['time_iso'] if after else None)
    status='bounded' if start and end else 'open_end' if start else 'open_start' if end else 'unassigned'
    if start and end and start>end:status='conflict'
    return {'start':start,'end':end,'start_label':before['raw'] if before else None,
            'end_label':after['raw'] if after else None,'timezone':TZ_NAME,'precision':'interval','status':status,
            'start_marker_id':before['id'] if before else None,'end_marker_id':after['id'] if after else None,
            'start_precision':before['precision'] if before else None,'end_precision':after['precision'] if after else None,
            'start_evidence':json.loads(before['evidence']) if before and include_evidence else [],
            'end_evidence':json.loads(after['evidence']) if after and include_evidence else []}


class TimelineIndex:
    """Read marker coordinates once, then find both bounds without scanning SQLite."""

    def __init__(self, db, *, include_evidence=False):
        evidence = ',evidence' if include_evidence else ''
        self.after = list(db.execute(
            f'SELECT rowid,id,top,bottom,raw,time_iso,precision{evidence} FROM time_markers ORDER BY top,bottom,rowid'))
        self.before = sorted(self.after, key=lambda row: (-row['bottom'], row['rowid']))
        self.tops = [row['top'] for row in self.after]
        self.neg_bottoms = [-row['bottom'] for row in self.before]
        self.include_evidence = include_evidence

    def __bool__(self):
        return bool(self.after)

    def bounds(self, top, bottom):
        before = bisect_left(self.neg_bottoms, -top)
        after = bisect_left(self.tops, bottom)
        return (self.before[before] if before < len(self.before) else None,
                self.after[after] if after < len(self.after) else None)

    def interval(self, top, bottom):
        return _interval(*self.bounds(top, bottom), include_evidence=self.include_evidence)

    def group_counts(self, positions):
        groups = Counter()
        labels = {}
        for top, bottom in positions:
            before, after = self.bounds(top, bottom)
            key = (before['id'] if before else None, after['id'] if after else None)
            if key not in labels:
                labels[key] = interval_label({'time_interval': _interval(before, after, include_evidence=False)})
            groups[labels[key]] += 1
        return dict(groups)


def interval_label(row):
    value=row.get('time_interval') or {}
    start,end=value.get('start'),value.get('end')
    def short(stamp):return stamp[:16].replace('T',' ') if stamp else '未知'
    if not start and not end:return '未定位'
    text=short(start)+' ～ '+short(end)
    return text+('（冲突）' if value.get('status')=='conflict' else '')


def interval_brief(row):
    value=row.get('time_interval') or {}
    start,end=value.get('start'),value.get('end')
    if not start and not end:return '未定位'
    def short(stamp):return stamp[5:16].replace('T',' ') if stamp else '未知'
    right=end[11:16] if start and end and start[:10]==end[:10] else short(end)
    return short(start)+' ～ '+right+(' !' if value.get('status')=='conflict' else '')
