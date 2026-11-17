"""Frame journal. Normal sessions never write pixels; debug subclasses can attach image metadata."""
from __future__ import annotations

import json
from pathlib import Path

from .archive import utc_now,write_json


class MemorySession:
    def __init__(self,path,metadata):
        self.path=Path(path).resolve()
        if self.path.exists() and any(self.path.iterdir()):raise ValueError('Output must be new or empty')
        self.path.mkdir(parents=True,exist_ok=True)
        self.counter=0
        self.last_height=0
        self.metadata={'format_version':2,'kind':'live_memory','storage_mode':'memory_only',
                       'created_at':utc_now(),'status':'running',**metadata}
        write_json(self.path/'manifest.json',self.metadata)

    def append(self,name,value):
        with (self.path/name).open('a',encoding='utf-8') as file:
            file.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')

    def capture(self,image,action):
        return self.frame(image.shape,action)

    def frame(self,shape,action,image_info=None):
        self.last_height=shape[0]
        value={'frame_id':f'{self.counter:06d}','captured_at':utc_now(),'height':shape[0],'width':shape[1],'action':action,**(image_info or {})}
        self.counter+=1;self.append('frames.jsonl',value)
        return value

    def evaluate(self,frame,outcome):
        record={'frame_id':frame['frame_id'],**outcome}
        self.append('alignments.jsonl',{key:value for key,value in record.items() if key not in {'observations','labels'}})
        self.append('observations.jsonl',{'frame_id':frame['frame_id'],'coordinate_status':'confirmed' if outcome['alignment']['accepted'] else 'unknown',
                                        'observations':outcome.get('observations',[]),'labels':outcome.get('labels',[])})
        if outcome['alignment']['accepted']:
            write_json(self.path/'checkpoint.json',{'frame_id':frame['frame_id'],'scroll_offset_px':outcome['scroll_offset_px'],
                       'origin_y_px':self.metadata.get('origin_y_px',0)})

    def finish(self,status,**fields):
        self.metadata.update(status=status,finished_at=utc_now(),**fields)
        write_json(self.path/'manifest.json',self.metadata)
