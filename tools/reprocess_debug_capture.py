"""Rerun the production OCR pipeline on retained debug frames; no desktop access or image copies."""
import argparse
from dataclasses import fields
from pathlib import Path
import time

from welog.archive import archive_frames,write_json
from welog.frame_pool import ImageBudget
from welog.imageio import read_image
from welog.models import AlignmentConfig,AlignmentResult,CaptureConfig,Rect
from welog.pipeline import OcrWorker
from welog.recognition import RapidOcrAdapter
from welog.replay import source_exclusion
from welog.archive import read_jsonl
from welog.session import MemorySession


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    metadata,frames=archive_frames(args.source)
    if metadata.get('kind')!='live_debug':raise ValueError('Input must be a retained debug capture')
    if not frames:raise ValueError('No retained frames to reprocess')
    if args.output.resolve().is_relative_to(args.source.resolve()):raise ValueError('Output must be outside the original capture directory')
    values={key:value for key,value in metadata['capture_config'].items() if key in {field.name for field in fields(CaptureConfig)}}
    config=CaptureConfig(**{**values,'debug_save_images':False})
    alignment=AlignmentConfig.from_dict(metadata['alignment_config'])
    session=MemorySession(args.output,{'kind':'offline_ocr','source':str(args.source.resolve()),
        'capture_config':values,'alignment_config':metadata['alignment_config'],'origin_y_px':config.origin_y_px,'image_storage':'references_only'})
    budget=ImageBudget(config.image_memory_limit_bytes)
    worker=OcrWorker(session.path,budget,RapidOcrAdapter(),alignment,config)
    decisions={row['frame_id']:row for row in read_jsonl(args.source/'alignments.jsonl')}
    accepted=rejected=excluded=0;offset=0.;reference=None;differences=[]
    started=time.perf_counter()
    try:
        for index,frame in enumerate(frames):
            image=read_image(args.source/frame['image'])
            stored=session.frame(image.shape,{**frame.get('action',{}),'source_frame_id':frame['frame_id']})
            exclusion=source_exclusion(metadata,frame,decisions)
            if exclusion:
                excluded+=1
                session.evaluate(stored,{'alignment':AlignmentResult(False,exclusion).to_dict(),'reference_id':reference,'scroll_offset_px':None})
                continue
            masks=tuple(Rect(**box) for box in frame.get('action',{}).get('stability',{}).get('dynamic_masks',[]))
            outcome=worker.submit(budget.copy(image),stored['frame_id'],masks,config.direction,frame['captured_at']).result()
            session.evaluate(stored,outcome)
            if outcome['ocr_error'] or outcome['repair_error']:raise RuntimeError('Offline OCR failed')
            if outcome['alignment']['accepted']:
                accepted+=1;offset=outcome['scroll_offset_px'];reference=stored['frame_id']
                differences.append(offset-decisions[frame['frame_id']]['scroll_offset_px'])
            else:rejected+=1
            if (index+1)%10==0:print(f'Processed {index+1}/{len(frames)}',flush=True)
        summary={'status':'complete' if not rejected else 'needs_attention','accepted':accepted,'rejected':rejected,'excluded':excluded,
            'final_scroll_offset_px':offset,'source_offset_difference_px':offset-metadata['final_scroll_offset_px'],
            'max_frame_offset_difference_px':max(map(abs,differences),default=0),'desktop_accessed':False,'images_written':0,
            'elapsed_seconds':time.perf_counter()-started}
        summary.update(worker.finish(summary))
        session.finish(summary['status'],**{k:v for k,v in summary.items() if k!='status'})
        write_json(session.path/'validation.json',summary)
        print({key:summary[key] for key in ['accepted','rejected','messages','type_counts','final_scroll_offset_px','source_offset_difference_px']})
    except BaseException:
        if not worker.closed:worker.finish({'status':'failed'})
        raise


if __name__=='__main__':main()
