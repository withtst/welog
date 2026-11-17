"""Read-only audit of saved media evidence. Never accesses desktop APIs or changes records."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sqlite3

from welog.exports import RecordSource
from welog.timeline import interval_label


def digest(path):
    with path.open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def audit(source):
    root=source.path.parent
    before={path.name:digest(path) for path in root.iterdir() if path.is_file()}
    frames={}
    journal=root/'observations.jsonl'
    if journal.exists():
        with journal.open(encoding='utf-8') as file:
            frames={row['frame_id']:row for row in map(json.loads,file)}
    rows=list(source.records())
    targets=[row for row in rows if row['type'] in {'image','unknown'}]
    results=[]
    db=sqlite3.connect(source.path.as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    try:
        for row in targets:
            identity=row['message_id']
            message=db.execute('SELECT * FROM messages WHERE id=?',(identity,)).fetchone()
            stored=[dict(item) for item in db.execute('SELECT * FROM lines WHERE message_id=? ORDER BY top,x1',(identity,))]
            views=[]
            for observation in row.get('evidence',[]):
                frame=frames.get(observation['frame_id'],{})
                box=observation['box'];x1,y1,x2,y2=box
                inside=[];near=[]
                for label in frame.get('labels',[]):
                    a,b,c,d=label['box'];cx,cy=(a+c)/2,(b+d)/2
                    if x1<=cx<=x2 and y1<=cy<=y2:inside.append(label)
                    elif x1-25<=cx<=x2+25 and y1-35<=cy<=y2+35:near.append(label)
                bubbles=[bubble for bubble in frame.get('observations',[]) if list(bubble['box'])==list(box)]
                views.append({'frame_id':observation['frame_id'],'box':box,'coordinate_status':frame.get('coordinate_status'),
                    'bubble_ocr_lines':[line for bubble in bubbles for line in bubble.get('lines',[])],
                    'inside_labels':inside,'nearby_labels':near,'observation_found':bool(bubbles)})
            # A neighboring message is a lead to inspect, not evidence belonging to this message.
            foreign=[dict(item) for item in db.execute(
                'SELECT message_id,text,top,bottom,x1,x2 FROM lines WHERE message_id<>? AND bottom>? AND top<? AND x2>? AND x1<?',
                (identity,message['top']-20,message['bottom']+20,message['x1']-20,message['x2']+20))]
            has_internal=bool(stored or any(view['bubble_ocr_lines'] or view['inside_labels'] for view in views))
            results.append({'message_id':identity,'type':row['type'],'time_interval':interval_label(row),
                'global_top_px':message['top'],'global_bottom_px':message['bottom'],
                'width_px':message['x2']-message['x1'],'height_px':message['bottom']-message['top'],
                'stored_lines':stored,'views':views,'neighboring_message_lines':foreign,
                'verdict':'needs_text_review' if has_internal else 'insufficient_evidence',
                'reason':'存在内部文字，需进一步核对' if has_internal else '没有内部 OCR 文字及原采集像素，不能由尺寸推断为文件'})
    finally:db.close()
    frame_file=root/'frames.jsonl'
    image_refs=0
    if frame_file.exists():
        with frame_file.open(encoding='utf-8') as file:image_refs=sum(bool(row.get('image')) for row in map(json.loads,file))
    after={path.name:digest(path) for path in root.iterdir() if path.is_file()}
    if before!=after:raise RuntimeError('Source files changed during read-only audit')
    return {'source':str(source.path),'desktop_accessed':False,'images_written':0,'source_files_unchanged':True,
        'source_sha256':before,'total_messages':len(rows),'current_type_counts':dict(Counter(row['type'] for row in rows)),
        'audited_type_counts':dict(Counter(row['type'] for row in targets)),
        'observation_count':sum(len(row['views']) for row in results),'capture_image_references':image_refs,
        'capture_image_files':sum(path.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.webp'} for path in root.rglob('*') if path.is_file()),
        'records':results,'applied_repairs':[]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=RecordSource(args.source)
    output=args.output.resolve()
    if output.is_relative_to(source.path.parent):raise ValueError('Audit output must be outside the source directory')
    report=audit(source)
    output.mkdir(parents=True,exist_ok=True)
    (output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    page=['<!doctype html><meta charset="utf-8"><title>媒体消息离线核查</title>',
        '<style>body{font:15px system-ui;margin:32px}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ddd}</style>',
        '<h1>媒体消息离线核查</h1>',
        f'<p>共核查 {len(report["records"])} 条媒体消息；结论依据逐条证据，不按卡片尺寸猜测文件类型。</p>',
        '<table><tr><th>消息 ID</th><th>原类型</th><th>时间区间</th><th>观察帧</th><th>内部文字行</th><th>结论</th></tr>']
    for row in report['records']:
        count=len(row['stored_lines'])+sum(len(view['bubble_ocr_lines'])+len(view['inside_labels']) for view in row['views'])
        values=[row['message_id'],row['type'],row['time_interval'],', '.join(view['frame_id'] for view in row['views']),count,row['reason']]
        page.append('<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in values)+'</tr>')
    page.append('</table><p>原始文件哈希未变；未操作桌面，未生成图片。详细邻近文字和来源见 audit.json。</p>')
    (output/'audit.html').write_text(''.join(page),encoding='utf-8')
    print(json.dumps({key:report[key] for key in ['total_messages','current_type_counts','audited_type_counts','observation_count','source_files_unchanged']}))


if __name__=='__main__':main()
