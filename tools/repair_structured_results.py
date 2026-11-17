"""Clean a copy of image-free results; no OCR, capture or image output."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from welog.archive import read_jsonl,write_json
from welog.message_store import MessageStore
from welog.recognition import Bubble,TextLine
from welog.text_rules import clean_lines,sanitize_bubble,sanitize_store


def fingerprints(root):
    return {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob('*') if path.is_file()}


def repair(source:Path,output:Path):
    source,output=source.resolve(),output.resolve()
    if output==source or output.is_relative_to(source):raise ValueError('Output must be outside source')
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new or empty output directory')
    before=fingerprints(source)
    output.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect((source/'messages.sqlite3').as_uri()+'?mode=ro',uri=True) as original:
        copy=sqlite3.connect(output/'messages.sqlite3')
        try:original.backup(copy)
        finally:copy.close()
    store=MessageStore(output/'messages.sqlite3',initialize=False)
    try:
        changes=sanitize_store(store)
        summary=json.loads((source/'report.json').read_text(encoding='utf-8'))
        summary.update(postprocess_changes=changes,source_results=str(source))
        counts=store.export(output,summary)
    finally:store.close()
    with (output/'observations.jsonl').open('w',encoding='utf-8') as target:
        for record in read_jsonl(source/'observations.jsonl'):
            observations=[]
            for raw in record.get('observations',[]):
                bubble=Bubble(**{**raw,'lines':[TextLine(**line) for line in raw.get('lines',[])]})
                if sanitize_bubble(bubble):observations.append(bubble.to_dict())
            labels=clean_lines([TextLine(**line) for line in record.get('labels',[])])
            target.write(json.dumps({**record,'observations':observations,
                'labels':[{'text':line.text,'box':line.box,'score':line.score} for line in labels]},ensure_ascii=False)+'\n')
    if before!=fingerprints(source):raise RuntimeError('Source results changed during repair')
    assert not list(output.rglob('*.png'))
    result={**changes,**counts,'source_unchanged':True,'image_files_written':0,'report':str(output/'report.html')}
    write_json(output/'postprocess-validation.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(repair(args.source,args.output),ensure_ascii=True,indent=2))


if __name__=='__main__':main()
