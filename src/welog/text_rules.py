"""Small deterministic rules shared by normal OCR, seam OCR and stored-result repair."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import replace


_NOTICE_TOKEN = r"(?<!\d)\d+\s*[+＋]?\s*条\s*新\s*消\s*息"
_NOTICE = re.compile(_NOTICE_TOKEN)
_NOTICE_LINE = re.compile(r"(?m)^[ \t]*" + _NOTICE_TOKEN + r"[ \t]*(?:\r?\n|$)")
_DURATION = re.compile(r'''(?P<seconds>\d+)\s*(?:['‘’′ʼ]\s*['‘’′ʼ]|["“”″])''')
_BRACKETS = {'[':']','【':'】','(':')'}


def clean_notice(text: str) -> tuple[str, bool]:
    """Only numbered unread-message notices are removed; other text is preserved."""
    if not _NOTICE.search(text):
        return text, False
    cleaned = _NOTICE_LINE.sub('', text)
    cleaned = _NOTICE.sub('', cleaned).strip()
    return cleaned, True


def audio_duration(text: str) -> int | None:
    """A whole message must be a seconds marker; a duration inside prose is not audio."""
    value = unicodedata.normalize('NFKC', text).strip()
    if value and value[0] in _BRACKETS and value.endswith(_BRACKETS[value[0]]):
        value = value[1:-1].strip()
    match = _DURATION.fullmatch(value)
    return int(match['seconds']) if match else None


def notice_region_masks(lines,width,height):
    """Exclude unread ink with scale-aware padding, preserving adjacent anchors."""
    from .models import Rect
    boxes=[]
    for line in lines:
        cleaned,changed=clean_notice(line.text)
        if not changed or cleaned.strip():continue
        x1,y1,x2,y2=line.box
        # A blanket 24px margin can erase a partly covered chat bubble and its
        # avatar. Those fragments may be the only nonperiodic motion evidence.
        line_height=max(1,y2-y1)
        pad_x=max(4,round(line_height/3));pad_y=max(4,round(line_height/4))
        x1,y1=max(0,x1-pad_x),max(0,y1-pad_y)
        x2,y2=min(width,x2+pad_x),min(height,y2+pad_y)
        if x2>x1 and y2>y1:boxes.append(Rect(x1,y1,x2-x1,y2-y1))
    return tuple(boxes)


def _mask_rectangles(mask,x0,y0):
    """Lossless row runs, preserving visible colored/black message fragments."""
    import numpy as np
    from .models import Rect
    active={};result=[]
    for y,row in enumerate(mask):
        edges=np.flatnonzero(np.diff(np.pad(row.astype(np.int8),(1,1))))
        runs=set(zip(edges[::2],edges[1::2]));following={}
        for a,b in runs:
            start,height=active.pop((a,b),(y,0));following[(a,b)]=(start,height+1)
        for (a,b),(start,height) in active.items():result.append(Rect(x0+int(a),y0+start,int(b-a),height))
        active=following
    for (a,b),(start,height) in active.items():result.append(Rect(x0+int(a),y0+start,int(b-a),height))
    return tuple(result)


def visual_notice_layers(image):
    """Separate a pale, teal-ink floating notice from overlapping message art.

    Return segmentation and evidence masks. Opaque pale pixels may cover
    informative page content, so exclude them from registration too. Visible
    colored fills and black message fragments remain available.
    """
    import cv2
    import numpy as np
    if image.ndim!=3:return (),()
    h,w=image.shape[:2];left=int(w*.75)
    stripe=image[:,left:].astype(np.int16)
    teal=(stripe[:,:,1]-stripe[:,:,2]>30)&(stripe[:,:,0]-stripe[:,:,2]>15)&(stripe[:,:,1]>100)
    joined=cv2.morphologyEx(teal.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((1,11),np.uint8))
    contours,_=cv2.findContours(joined,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    segment=[];evidence=[]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    background=float(np.median(np.concatenate([gray[:,:20].ravel(),gray[:,-20:].ravel()])))
    for contour in contours:
        x,y,bw,bh=cv2.boundingRect(contour);x+=left
        if bw<40 or bh<6 or bh>36 or bw<bh*3 or bw>w*.4 or x+bw<w*.94:continue
        if not (y+bh<h*.18 or y>h*.82):continue
        px=max(18,int(bh*1.5));py=max(20,int(bh*1.5))
        a,b,c,d=max(0,x-px),max(0,y-py),min(w,x+bw+px),min(h,y+bh+py)
        values=image[b:d,a:c].astype(np.int16)
        light=(values.min(2)>=210)&(values.max(2)-values.min(2)<=14)
        ink=(values[:,:,1]-values[:,:,2]>12)&(values[:,:,0]-values[:,:,2]>8)&(values[:,:,1]>100)
        strong=(values[:,:,1]-values[:,:,2]>30)&(values[:,:,0]-values[:,:,2]>15)&(values[:,:,1]>100)
        ink |= cv2.dilate(strong.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        inside=(light|ink)[y-b:y-b+bh,x-a:x-a+bw]
        if inside.mean()<.8:continue
        # Preserve ordinary gray chat panels in the content lane. The floating
        # layer crosses into the avatar lane; removing its pale/shadow pixels
        # there breaks contours joined to a sticker without erasing body text.
        outside_lane=np.arange(a,c)[None,:]>=int(w*.94)
        segment_light=light&((values.min(2)>=242)|outside_lane)
        from .models import Rect
        ink_box=Rect(max(0,x-2),max(0,y-2),min(w,x+bw+2)-max(0,x-2),min(h,y+bh+2)-max(0,y-2))
        segment.extend(_mask_rectangles(segment_light,a,b));segment.append(ink_box)
        shadow=light&(gray[b:d,a:c]<background-1)
        evidence.extend(_mask_rectangles(segment_light|shadow,a,b));evidence.append(ink_box)
    return tuple(dict.fromkeys(segment)),tuple(dict.fromkeys(evidence))


def suppress_visual_notice(image,masks):
    """Segmentation/OCR working copy only; never changes a captured source image."""
    import numpy as np
    if not masks:return image
    result=image.copy()
    if image.ndim==2:
        background=np.median(np.concatenate([image[:,:20].ravel(),image[:,-20:].ravel()]))
    else:
        background=np.median(np.concatenate([image[:,:20].reshape(-1,3),image[:,-20:].reshape(-1,3)]),axis=0).astype(np.uint8)
    for box in masks:result[max(0,box.y):box.bottom,max(0,box.x):box.right]=background
    return result


def clean_lines(lines):
    output = []
    for line in lines:
        text, changed = clean_notice(line.text)
        if text.strip():
            output.append(replace(line, text=text, notice_removed=line.notice_removed or changed))
    return output


def sanitize_bubble(bubble) -> bool:
    """False means the candidate consisted only of an unread overlay, not an OCR failure."""
    if bubble.kind not in {'text','audio'}:
        return True
    affected = any(line.notice_removed or clean_notice(line.text)[1] for line in bubble.lines)
    bubble.lines = clean_lines(bubble.lines)
    if affected and not bubble.lines:
        return False
    if audio_duration('\n'.join(line.text for line in bubble.lines)) is not None:
        bubble.kind = 'audio'
    bubble.ocr_status = 'ok' if bubble.lines else 'failed'
    return True


def classify_stored_message(connection, identity):
    row = connection.execute('SELECT kind FROM messages WHERE id=?',(identity,)).fetchone()
    if row is None or row['kind'] not in {'text','audio'}:
        return
    text = '\n'.join(line['text'] for line in connection.execute(
        'SELECT text FROM lines WHERE message_id=? ORDER BY top,x1',(identity,)))
    if audio_duration(text) is not None:
        connection.execute("UPDATE messages SET kind='audio' WHERE id=?",(identity,))


def sanitize_store(store) -> dict:
    """Repair a COPY of existing structured results, without needing or writing images."""
    db = store.connection
    stats = {'cleaned_lines':0,'removed_messages':0,'classified_audio':0}
    for message in db.execute('SELECT id,kind FROM messages').fetchall():
        if message['kind'] not in {'text','audio'}:
            continue
        identity = message['id'];affected = False
        for line in db.execute('SELECT id,text FROM lines WHERE message_id=?',(identity,)).fetchall():
            text, changed = clean_notice(line['text'])
            if not changed:
                continue
            affected = True;stats['cleaned_lines'] += 1
            if text.strip():db.execute('UPDATE lines SET text=? WHERE id=?',(text,line['id']))
            else:db.execute('DELETE FROM lines WHERE id=?',(line['id'],))
        if affected and not db.execute('SELECT 1 FROM lines WHERE message_id=?',(identity,)).fetchone():
            db.execute('DELETE FROM observations WHERE message_id=?',(identity,))
            db.execute('DELETE FROM messages WHERE id=?',(identity,))
            stats['removed_messages'] += 1
            continue
        classify_stored_message(db,identity)
        updated = db.execute('SELECT kind FROM messages WHERE id=?',(identity,)).fetchone()
        if message['kind']!='audio' and updated['kind']=='audio':stats['classified_audio'] += 1
    db.commit()
    return stats
