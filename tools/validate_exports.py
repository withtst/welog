"""Validate exports against existing text results without accessing any windows."""
import argparse
import hashlib
import json
from itertools import zip_longest
from pathlib import Path

from welog.exports import ExportOptions, FORMATS, RecordSource, export_record, export_records


def digest(path):
    with path.open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reports=[]
    for index,path in enumerate(args.source,1):
        source=RecordSource(path)
        before=digest(source.path)
        options=ExportOptions('对方','我','离线导出验证')
        result=export_records(source,args.output/f'source-{index}',FORMATS,options)
        paths={Path(path).suffix[1:]:Path(path) for path in result['files']}
        for key in ['json','jsonl']:
            for original,exported in zip_longest(source.records(),RecordSource(paths[key]).records()):
                assert original is not None and exported==export_record(original,options),f'{key} roundtrip mismatch'
        assert before==digest(source.path),'Source file was modified'
        reports.append({'source':str(source.path),'count':result['count'],'formats':list(paths),
            'source_sha256_unchanged':True,'json_roundtrip_matches':True,'jsonl_roundtrip_matches':True,
            'audio_count':sum(row['type']=='audio' for row in source.records()),'files':result['files']})
    assert not [path for path in args.output.rglob('*') if path.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.webp'}]
    report={'desktop_accessed':False,'images_written':0,'sources':reports}
    output=args.output/'validation.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'sources':[{'count':row['count'],'audio_count':row['audio_count']} for row in reports],
        'source_files_unchanged':True,'formats':list(FORMATS),'images_written':0},ensure_ascii=False))


if __name__=='__main__':main()
