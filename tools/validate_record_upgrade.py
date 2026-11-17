"""Copy only existing text results, rebuild intervals, and verify the original stays intact."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

from welog.exports import RecordSource,export_records,preview_records
from welog.record_state import rebuild_intervals,rename_record


def sha(path):
    with path.open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=RecordSource(args.source)
    root=args.output.resolve()/('snapshot-'+uuid4().hex[:8])
    root.mkdir(parents=True)
    copied=[]
    for name in ['messages.sqlite3','messages.jsonl','manifest.json','frames.jsonl','alignments.jsonl','observations.jsonl','report.json']:
        path=source.path.parent/name
        if path.exists():
            copied.append((path,sha(path)))
            shutil.copy2(path,root/name)
    snapshot=RecordSource(root)
    rebuild_intervals(snapshot)
    rename_record(snapshot,'离线验证记录')
    page=preview_records(snapshot,limit=0)
    exported=export_records(snapshot,root/'exports',['json','html','csv'])
    assert all(sha(path)==digest for path,digest in copied)
    time_counts={}
    for row in snapshot.records():
        status=(row.get('time_interval') or {}).get('status','unassigned')
        time_counts[status]=time_counts.get(status,0)+1
    report={'desktop_accessed':False,'images_written':0,'source_files_unchanged':True,
            'statistics':page['statistics'],'time_interval_status_counts':time_counts,'record_name':page['metadata']['name'],
            'exported_count':exported['count']}
    (root/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'messages':page['total'],'time_interval_status_counts':time_counts,'source_files_unchanged':True,'images_written':0}))
    print(root)


if __name__=='__main__':main()
