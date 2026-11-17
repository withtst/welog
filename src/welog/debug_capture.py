"""Explicit opt-in image retention; the normal session never imports an image writer."""
import hashlib
from collections import OrderedDict
from .session import MemorySession
from .imageio import write_image


class DebugImageLimit(RuntimeError):
    pass


class DebugSession(MemorySession):
    def __init__(self,path,metadata,max_bytes):
        super().__init__(path,{**metadata,'kind':'live_debug','storage_mode':'debug_images',
            'debug_save_images':True,'debug_max_bytes':max_bytes})
        self.max_bytes=max_bytes
        self.image_bytes=0
        self.image_count=0
        (self.path/'frames').mkdir()

    def save_image(self,relative,image):
        target=self.path/relative
        if not target.resolve().is_relative_to(self.path):raise ValueError('Invalid diagnostic path')
        if target.is_file():
            with target.open('rb') as file:checksum=hashlib.file_digest(file,'sha256').hexdigest()
            return {'image':relative,'sha256':checksum,'bytes':target.stat().st_size}
        if self.image_bytes>=self.max_bytes:raise DebugImageLimit('Debug image quota reached')
        try:
            write_image(target,image)
            size=target.stat().st_size
            if self.image_bytes+size>self.max_bytes:raise DebugImageLimit('Debug image quota reached')
        except BaseException:
            target.unlink(missing_ok=True)
            target.with_suffix(target.suffix+'.tmp').unlink(missing_ok=True)
            raise
        with target.open('rb') as file:checksum=hashlib.file_digest(file,'sha256').hexdigest()
        self.image_bytes+=size;self.image_count+=1
        return {'image':relative,'sha256':checksum,'bytes':size}

    def capture(self,image,action):
        relative=f'frames/{self.counter:06d}.png'
        info=self.save_image(relative,image)
        return self.frame(image.shape,action,info)


class EventDebugSession(DebugSession):
    def __init__(self,path,metadata,max_bytes):
        super().__init__(path,metadata,max_bytes)
        self.metadata.update(kind='live_event_debug',storage_mode='event_images',debug_image_policy='events')
        from .archive import write_json
        write_json(self.path/'manifest.json',self.metadata)
        self.unknown_seen=OrderedDict()
        self.ocr_seen=OrderedDict()
        self.retained=OrderedDict()

    def capture(self,image,action):return self.frame(image.shape,action)

    def unknown_needed(self,item,offset):
        key=item.get('message_id')
        if not key or key not in self.unknown_seen:return True
        state=self.unknown_seen[key]
        if state['complete']:return False
        if item.get('top_state')=='visible' and item.get('bottom_state')=='visible':return True
        if offset is None or state['range'] is None:return False
        top,bottom=offset+item['box'][1],offset+item['box'][3]
        return top<state['range'][0]-8 or bottom>state['range'][1]+8

    def retain_frame(self,frame,image):
        key=frame['frame_id']
        if key in self.retained:return self.retained[key]
        info={**frame,**self.save_image(f'frames/{key}.png',image)}
        self.append('retained_frames.jsonl',info)
        self.retained[key]=info
        if len(self.retained)>2048:self.retained.popitem(last=False)
        return info

    def retain_event(self,frame,image,reasons,*,reference=None,unknowns=(),temporal=(),coordinate_eligible=False,ocr_failures=(),scroll_offset_px=None):
        event={'frame_id':frame['frame_id'] if frame else None,'reasons':list(reasons),'images':[],
               'coordinate_eligible':coordinate_eligible,'current_image_available':image is not None,
               'temporal_sample_order':'oldest_to_newest','ocr_failed_regions':list(ocr_failures),'scroll_offset_px':scroll_offset_px,
               'unknown_regions':list(unknowns)}
        try:
            if reference:
                ref_frame,ref_image=reference
                event['reference_frame_id']=ref_frame['frame_id']
                event['images'].append({'role':'reference',**self.retain_frame(ref_frame,ref_image)})
            if frame is not None and image is not None:event['images'].append({'role':'current',**self.retain_frame(frame,image)})
            for index,unknown in enumerate(unknowns):
                x1,y1,x2,y2=map(int,unknown['box'])
                x1,y1=max(0,x1),max(0,y1);x2,y2=min(image.shape[1],x2),min(image.shape[0],y2)
                if x2<=x1 or y2<=y1:continue
                complete=unknown.get('top_state')=='visible' and unknown.get('bottom_state')=='visible'
                key=unknown.get('message_id')
                if not self.unknown_needed(unknown,scroll_offset_px):continue
                info=self.save_image(f'unknown/{frame["frame_id"]}-{index:03d}.png',image[y1:y2,x1:x2])
                position={}
                if coordinate_eligible and scroll_offset_px is not None:
                    position={'global_top_px':scroll_offset_px+y1-self.metadata.get('origin_y_px',0),
                              'global_bottom_px':scroll_offset_px+y2-self.metadata.get('origin_y_px',0)}
                event['images'].append({'role':'unknown_crop',**unknown,**info,'crop_box':[x1,y1,x2,y2],**position})
                if key:
                    bounds=None if scroll_offset_px is None else (scroll_offset_px+y1,scroll_offset_px+y2)
                    prior=self.unknown_seen.get(key)
                    if bounds is not None and prior and prior['range'] is not None:bounds=(min(bounds[0],prior['range'][0]),max(bounds[1],prior['range'][1]))
                    self.unknown_seen[key]={'complete':complete,'range':bounds};self.unknown_seen.move_to_end(key)
                    if len(self.unknown_seen)>1024:self.unknown_seen.popitem(last=False)
            for index,lease in enumerate(temporal):
                info=self.save_image(f'diagnostics/{frame["frame_id"]}-sample-{index:02d}-gray.png',lease.image)
                event['images'].append({'role':'temporal_sample','sample_index':index,**info})
        except Exception as error:
            event['image_save_error']=type(error).__name__
            raise
        finally:self.append('diagnostics.jsonl',event)

    def process_event(self,frame,image,outcome,reference=None,budget=None):
        reasons=[]
        if not outcome['alignment']['accepted']:reasons.append('alignment_rejected')
        if outcome.get('ocr_error'):reasons.append('ocr_error')
        if outcome.get('repair_error'):reasons.append('seam_ocr_error')
        ocr_failures=[item for item in outcome.get('ocr_failed_observations',[]) if not item.get('message_id') or item['message_id'] not in self.ocr_seen]
        if ocr_failures:reasons.append('ocr_unreadable')
        unknowns=[]
        for item in outcome.get('unknown_observations',[]):
            if self.unknown_needed(item,outcome.get('scroll_offset_px')):unknowns.append(item)
        if unknowns:reasons.append('unknown_message')
        if reasons:
            if budget is not None:
                with budget.reserve(image.nbytes*2):self.retain_event(frame,image,reasons,reference=reference,unknowns=unknowns,coordinate_eligible=outcome['alignment']['accepted'],ocr_failures=ocr_failures,scroll_offset_px=outcome.get('scroll_offset_px'))
            else:self.retain_event(frame,image,reasons,reference=reference,unknowns=unknowns,coordinate_eligible=outcome['alignment']['accepted'],ocr_failures=ocr_failures,scroll_offset_px=outcome.get('scroll_offset_px'))
            for item in ocr_failures:
                if item.get('message_id'):
                    self.ocr_seen[item['message_id']]=True
                    if len(self.ocr_seen)>1024:self.ocr_seen.popitem(last=False)
