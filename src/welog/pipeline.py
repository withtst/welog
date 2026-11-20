"""Bounded, serial OCR worker and rolling boundary repair. Never writes images."""
from __future__ import annotations

import threading
import json
from collections import Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .assembler import FragmentAssembler
from .frame_pool import FrameLease, ImageBudget
from .imageio import gray
from .message_store import MessageStore
from .motion import MotionTracker
from .recognition import assign_lines, detect_bubbles
from .text_rules import clean_lines,notice_region_masks,visual_notice_layers
from .file_cards import mark_candidates
from .timeline import observe_markers


class BoundaryBuffer:
    def __init__(self,budget):
        self.budget=budget;self.top=self.bottom=None;self.offset=None;self.height=0;self.media=[]

    def compose(self,current,offset,direction):
        if self.offset is None or offset==self.offset:return None
        band=self.top if direction=='up' else self.bottom
        band_top=self.offset if direction=='up' else self.offset+self.height-band.image.shape[0]
        edge=self.offset if direction=='up' else self.offset+self.height
        context=band.image.shape[0]
        start=max(int(edge-context),int(min(band_top,offset)))
        end=min(int(edge+context),int(max(band_top+band.image.shape[0],offset+current.shape[0])))
        intervals=sorted([(band_top,band_top+band.image.shape[0]),(offset,offset+current.shape[0])])
        if end<=start or intervals[1][0]>intervals[0][1]:return None
        shape=(end-start,current.shape[1],3)
        reservation=self.budget.reserve(int(np.prod(shape)))
        try:
            canvas=np.full(shape,250,np.uint8)
            for image,image_top in [(band.image,band_top),(current,offset)]:
                a,b=max(start,int(image_top)),min(end,int(image_top)+image.shape[0])
                if b>a:canvas[a-start:b-start]=image[a-int(image_top):b-int(image_top)]
            return FrameLease(canvas,reservation),float(start)
        except BaseException:
            reservation.release();raise

    def update(self,image,offset,media):
        h=max(1,image.shape[0]//3)
        top=self.budget.copy(image[:h]);bottom=None
        try:bottom=self.budget.copy(image[-h:])
        except BaseException:top.release();raise
        self.close()
        self.top,self.bottom,self.offset,self.height,self.media=top,bottom,offset,image.shape[0],list(media)

    def close(self):
        for item in (self.top,self.bottom):
            if item:item.release()
        self.top=self.bottom=None


class FrameProcessor:
    def __init__(self,output:Path,budget,backend,alignment,config):
        self.output,self.budget,self.backend,self.config=output,budget,backend,config
        self.store=MessageStore(output/'messages.sqlite3')
        self.assembler=FragmentAssembler(self.store)
        self.motion=MotionTracker(budget,alignment)
        self.boundary=BoundaryBuffer(budget)
        self.previous_partial=False
        self.media_guard=False
        self.media_risk=False
        manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8')) if (output/'manifest.json').exists() else {}
        self.capture_reference=datetime.fromisoformat(manifest['created_at'].replace('Z','+00:00')) if manifest.get('created_at') else None

    @staticmethod
    def mask_media(image,media,offset=0):
        background=np.median(image.reshape(-1,3),axis=0).astype(np.uint8)
        for x1,y1,x2,y2 in media:
            a,b=max(0,int(y1-offset)),min(image.shape[0],int(y2-offset))
            if b>a:image[a:b,max(0,int(x1)):min(image.shape[1],int(x2))]=background

    def process(self,lease,frame_id,masks,direction,captured_at=None,motion_bounds=None):
        try:
            image=lease.image
            # Credit for explicit temporary conversion/difference buffers. Model/native workspaces are separate.
            with self.budget.reserve(image.nbytes*3):
                bubbles=detect_bubbles(image)
                notice_segment,notice_evidence=visual_notice_layers(image)
                mark_candidates(image,bubbles)
                ocr_error=None
                with self.budget.copy(image) as clean:
                    self.mask_media(clean.image,[b.box for b in bubbles if b.kind!='text' and '_file_candidate' not in b.metadata])
                    self.mask_media(clean.image,[(b.x,b.y,b.right,b.bottom) for b in notice_segment])
                    try:lines=self.backend.recognize(clean.image)
                    except Exception as error:lines=[];ocr_error=type(error).__name__
                overlay_masks=tuple(dict.fromkeys(tuple(notice_evidence)+notice_region_masks(lines,image.shape[1],image.shape[0])))
                labels=assign_lines(bubbles,lines)
                anchors=[line for b in bubbles if b.kind=='text' for line in b.lines
                         if not line.notice_removed and line.box[1]>2 and line.box[3]<image.shape[0]-2]
                result,reference,method=self.motion.accept(image,frame_id,anchors,direction,tuple(dict.fromkeys(tuple(masks)+overlay_masks)),motion_bounds)
                counts=Counter(line.text for line in anchors if line.score>=.7)
                planning_anchors=[line for line in anchors if line.score>=.7 and len(line.text.strip())>=2 and counts[line.text]==1]
                # Animation risk and the decision to slow down are separate:
                # two visible lines can both leave the next overlap.
                media_risk=sum(b.kind in {'image','unknown'} for b in bubbles)>=2 and (bool(masks) or self.media_risk)
                media_guard=media_risk and len(planning_anchors)<2
                repair_error=None
                seam_calls=0
                message_ids={}
                if result.accepted:
                    self.media_guard=media_guard
                    self.media_risk=media_risk
                    offset=self.motion.offset-self.config.origin_y_px
                    timestamp_reference=datetime.fromisoformat(captured_at.replace('Z','+00:00')) if captured_at else self.capture_reference
                    observe_markers(self.store.connection,labels,offset,image.shape[1],frame_id,timestamp_reference)
                    for index,bubble in enumerate(bubbles):message_ids[index]=self.assembler.observe(frame_id,bubble,offset,image.shape[0])
                    partial=any(b.kind=='text' and (b.top_state!='visible' or b.bottom_state!='visible') for b in bubbles)
                    media=[(b.box[0],offset+b.box[1],b.box[2],offset+b.box[3]) for b in bubbles if b.kind!='text']
                    if partial or self.previous_partial:
                        seam=self.boundary.compose(image,offset,self.config.direction)
                        if seam:
                            patch,patch_top=seam
                            with patch:
                                self.mask_media(patch.image,self.boundary.media+media,patch_top)
                                try:
                                    seam_calls+=1
                                    self.assembler.repair(clean_lines(self.backend.recognize(patch.image)),patch_top,patch.image.shape[0])
                                except Exception as error:repair_error=type(error).__name__
                    self.boundary.update(image,offset,media)
                    self.previous_partial=partial
                return {'alignment':result.to_dict(),'reference_id':reference,'motion_method':method,
                        'media_dominated':media_guard,
                        'media_guard_eligible':media_risk,
                        'motion_anchor_boxes':[list(line.box) for line in planning_anchors],
                        'scroll_offset_px':self.motion.offset if result.accepted else None,
                        'motion_overlay_masks':[box.to_dict() for box in overlay_masks],
                        'observations':[b.to_dict() for b in bubbles],
                        'unknown_observations':[{'box':list(b.box),'message_id':message_ids.get(i),'top_state':b.top_state,'bottom_state':b.bottom_state}
                                                for i,b in enumerate(bubbles) if b.kind=='unknown'],
                        'ocr_failed_observations':[{'box':list(b.box),'message_id':message_ids.get(i)} for i,b in enumerate(bubbles)
                            if b.kind in {'text','audio'} and b.ocr_status=='failed' and b.top_state=='visible' and b.bottom_state=='visible'],
                        'labels':[{'text':line.text,'box':line.box,'score':line.score} for line in labels],
                        'ocr_error':ocr_error,'repair_error':repair_error,'seam_ocr_calls':seam_calls}
        finally:
            lease.release()

    def finish(self,summary):
        self.boundary.close();self.motion.close()
        summary.update(image_memory_peak_bytes=self.budget.peak_bytes,image_memory_remaining_bytes=self.budget.used_bytes)
        try:return self.store.export(self.output,summary)
        finally:self.store.close()


class OcrWorker:
    """One model owner, bounded pending tasks. Image ownership transfers on submit."""
    def __init__(self,output,budget,backend,alignment,config,max_pending=2):
        if max_pending<1:raise ValueError('Worker queue capacity must be positive')
        self.output,self.budget,self.backend,self.alignment,self.config=output,budget,backend,alignment,config
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='welog-ocr')
        self.credit=threading.BoundedSemaphore(max_pending)
        self.processor=None;self.closed=False

    def _process(self,lease,*args):
        if self.processor is None:
            try:self.processor=FrameProcessor(self.output,self.budget,self.backend,self.alignment,self.config)
            except BaseException:lease.release();raise
        return self.processor.process(lease,*args)

    def submit(self,lease,*args):
        self.credit.acquire()
        try:future=self.executor.submit(self._process,lease,*args)
        except BaseException:self.credit.release();lease.release();raise
        future.add_done_callback(lambda _:self.credit.release())
        return future

    def finish(self,summary):
        def finish_owner():
            if self.processor is not None:return self.processor.finish(summary)
            summary.update(image_memory_peak_bytes=self.budget.peak_bytes,image_memory_remaining_bytes=self.budget.used_bytes)
            store=MessageStore(self.output/'messages.sqlite3')
            try:return store.export(self.output,summary)
            finally:store.close()
        try:
            return self.executor.submit(finish_owner).result()
        finally:self.closed=True;self.executor.shutdown(wait=True)
