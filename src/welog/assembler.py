"""Geometry identifies message occurrences; text equality never deletes a message."""
from __future__ import annotations

import json
from uuid import uuid4

from .message_store import MessageStore
from .recognition import Bubble, TextLine
from .text_rules import clean_notice,classify_stored_message,sanitize_bubble


class FragmentAssembler:
    def __init__(self, store: MessageStore):
        self.store = store

    def observe(self, frame_id: str, bubble: Bubble, offset: float, viewport_height: int) -> str | None:
        if not sanitize_bubble(bubble):return None
        db = self.store.connection
        x1,y1,x2,y2=bubble.box
        top,bottom=offset+y1,offset+y2
        candidates=[]
        for row in db.execute("SELECT * FROM messages WHERE direction=? AND (kind=? OR (kind IN ('text','audio') AND ? IN ('text','audio')) OR (kind IN ('text','image','unknown','file') AND ?='file') OR (kind='file' AND ? IN ('text','image','unknown')) OR (kind IN ('image','unknown') AND ? IN ('image','unknown'))) AND top<? AND bottom>?",
                              (bubble.direction,bubble.kind,bubble.kind,bubble.kind,bubble.kind,bubble.kind,bottom,top)):
            intersection=min(bottom,row['bottom'])-max(top,row['top'])
            horizontal=min(x2,row['x2'])-max(x1,row['x1'])
            if {row['kind'],bubble.kind}=={'image','unknown'}:
                clipped=not(row['head'] and row['tail']) or bubble.top_state!='visible' or bubble.bottom_state!='visible'
                if not clipped or abs(x1-row['x1'])>8 or abs(x2-row['x2'])>8 or intersection<.8*min(bottom-top,row['bottom']-row['top']):continue
            x_aligned=(abs(x1-row['x1'])<=8 and abs(x2-row['x2'])<=8) if bubble.kind=='text' else horizontal>=min(x2-x1,row['x2']-row['x1'])*.4
            if x_aligned and intersection>=min(6,(bottom-top)*.3):
                candidates.append((intersection,row))
        candidates.sort(key=lambda item:item[0],reverse=True)
        if candidates and (len(candidates)==1 or candidates[0][0]>candidates[1][0]*1.5):
            old=candidates[0][1];identity=old['id']
            db.execute("UPDATE messages SET top=?,bottom=?,head=?,tail=? WHERE id=?",(
                min(top,old['top']),max(bottom,old['bottom']),old['head'] or bubble.top_state=='visible',
                old['tail'] or bubble.bottom_state=='visible',identity))
            if bubble.kind in {'audio','file'}:db.execute("UPDATE messages SET kind=? WHERE id=?",(bubble.kind,identity))
            if {old['kind'],bubble.kind}=={'image','unknown'}:db.execute("UPDATE messages SET kind='image' WHERE id=?",(identity,))
        else:
            identity=str(uuid4())
            db.execute("INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?,?)",(
                identity,bubble.direction,bubble.kind,x1,x2,top,bottom,bubble.top_state=='visible',
                bubble.bottom_state=='visible',bool(candidates)))
        db.execute("INSERT INTO observations VALUES(?,?,?,?,?,?)",(identity,frame_id,json.dumps(bubble.box),
                    bubble.top_state,bubble.bottom_state,bubble.ocr_status))
        for line in bubble.lines:
            self.add_line(identity,line,offset,line.box[1]<=2 or line.box[3]>=viewport_height-2)
        classify_stored_message(db,identity)
        if bubble.metadata:
            old=db.execute('SELECT data FROM message_details WHERE message_id=?',(identity,)).fetchone()
            details=json.loads(old['data']) if old else {}
            for key,value in bubble.metadata.items():
                if key.startswith('_'):continue
                if key=='file':
                    old_file=details.get('file',{})
                    if (old_file.get('filename_complete',False),len(old_file.get('filename','')))>(value.get('filename_complete',False),len(value.get('filename',''))):continue
                details[key]=value
            db.execute('INSERT OR REPLACE INTO message_details VALUES (?,?)',(identity,json.dumps(details,ensure_ascii=False)))
        db.commit()
        return identity

    def add_line(self, identity: str, line: TextLine, offset: float, clipped: bool=False):
        text,_=clean_notice(line.text)
        if not text.strip():return
        db=self.store.connection
        x1,y1,x2,y2=line.box;top,bottom=offset+y1,offset+y2
        matches=[]
        for old in db.execute("SELECT * FROM lines WHERE message_id=? AND top<? AND bottom>?",(identity,bottom+3,top-3)):
            vertical=min(bottom,old['bottom'])-max(top,old['top'])
            horizontal=min(x2,old['x2'])-max(x1,old['x1'])
            if vertical>=min(bottom-top,old['bottom']-old['top'])*.45 and horizontal>min(x2-x1,old['x2']-old['x1'])*.25:
                matches.append(old)
        if matches:
            best=max(matches,key=lambda row:(not row['clipped'],row['score']))
            if (not clipped,line.score) < (not best['clipped'],best['score']):
                return
            for old in matches:
                db.execute("DELETE FROM lines WHERE id=?",(old['id'],))
        db.execute("INSERT INTO lines(message_id,top,bottom,x1,x2,text,score,clipped) VALUES(?,?,?,?,?,?,?,?)",
                   (identity,top,bottom,x1,x2,text,line.score,clipped))

    def repair(self, lines: list[TextLine], patch_top: float, patch_height: int):
        db=self.store.connection
        for line in lines:
            if line.box[1]<=2 or line.box[3]>=patch_height-2:
                continue
            center=patch_top+(line.box[1]+line.box[3])/2
            cx=(line.box[0]+line.box[2])/2
            candidates=list(db.execute("SELECT id FROM messages WHERE kind IN ('text','audio') AND top<=? AND bottom>=? AND x1<=? AND x2>=?",
                                       (center,center,cx,cx)))
            if len(candidates)==1:
                self.add_line(candidates[0]['id'],line,patch_top,False)
                classify_stored_message(db,candidates[0]['id'])
        db.commit()
