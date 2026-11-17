"""Reprocess existing screenshots in memory. No window API or screenshot writes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import cv2
from PIL import Image

from welog.archive import archive_frames,image_root,write_json
from welog.frame_pool import ImageBudget
from welog.imageio import read_image
from welog.models import AlignmentConfig,CaptureConfig,Rect
from welog.pipeline import OcrWorker
from welog.recognition import RapidOcrAdapter
from welog.session import MemorySession


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=6)
    args=parser.parse_args()
    meta,frames=archive_frames(args.source);images=image_root(args.source,meta)
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.source.rglob('*') if p.is_file()}
    session=MemorySession(args.output,{'kind':'offline_memory','source':str(args.source.resolve()),
                                      'note':'Offline reprocessing, not retroactive approval of original rejected frames'})
    budget=ImageBudget();config=CaptureConfig(direction=meta.get('capture_config',{}).get('direction','up'))
    worker=OcrWorker(session.path,budget,RapidOcrAdapter(),AlignmentConfig.from_dict(meta.get('alignment_config',{})),config)
    rows=[]
    def forbidden(*args,**kwargs):raise AssertionError('Image persistence forbidden during validation')
    try:
        with patch.object(cv2,'imwrite',forbidden),patch.object(cv2,'imencode',forbidden),patch.object(Image.Image,'save',forbidden):
            for original in frames[:args.frames]:
                image=read_image(images/original['image'])
                frame=session.frame(image.shape,{'kind':'offline_reprocess','source_frame_id':original['frame_id']})
                masks=tuple(Rect(**box) for box in original.get('action',{}).get('stability',{}).get('dynamic_masks',[]))
                direction='both' if original.get('action',{}).get('kind')=='recovery' else config.direction
                outcome=worker.submit(budget.copy(image),frame['frame_id'],masks,direction).result()
                session.evaluate(frame,outcome)
                rows.append({'frame_id':frame['frame_id'],'accepted':outcome['alignment']['accepted'],
                             'delta_px':outcome['alignment']['delta_px'],'cumulative_px':outcome['scroll_offset_px'],
                             'method':outcome['motion_method'],'bubbles':len(outcome['observations']),
                             'seam_ocr_calls':outcome['seam_ocr_calls'],'ocr_error':outcome['ocr_error']})
            summary={'status':'offline_reprocessed','accepted':sum(r['accepted'] for r in rows),
                     'rejected':sum(not r['accepted'] for r in rows),'final_scroll_offset_px':next((r['cumulative_px'] for r in reversed(rows) if r['accepted']),None),
                     'image_storage':'none','desktop_access':False,'rows':rows}
            summary.update(worker.finish(summary))
    finally:
        if not worker.closed:worker.finish({'status':'validation_error'})
    assert budget.used_bytes==0
    assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.source.rglob('*') if p.is_file()}
    assert not list(args.output.rglob('*.png'))
    summary['source_unchanged']=True
    session.finish('offline_reprocessed',**{k:v for k,v in summary.items() if k!='status'})
    write_json(args.output/'validation.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},ensure_ascii=True,indent=2))


if __name__=='__main__':main()
