"""Recognize file-card semantics using layout plus filename and size/footer evidence."""
import re
import unicodedata
import numpy as np

SIZE=re.compile(r'(?<![\w.])(\d+(?:\.\d+)?)\s*(KB|KiB|MB|MiB|GB|GiB|TB|TiB|B|K|M|G|T)(?![A-Za-z])',re.I)
NAME=re.compile(r'.+\.([A-Za-z0-9]{1,12})\s*$')
ICON_LABELS={'P','W','X','PDF','PPT','PPTX','DOC','DOCX','XLS','XLSX','ZIP','RAR','7Z','TXT'}


def _details(filename,raw,size,complete):
    match=NAME.fullmatch(filename)
    details={'filename':filename,'extension':match[1].lower() if match else '',
             'ocr_text':raw,'download_status':'not_downloaded' if '未下载' in unicodedata.normalize('NFKC',raw) else 'unknown',
             'filename_complete':complete}
    if size:
        details['size_display']=size[0].strip()
        unit=size[2].upper().replace('I','').rstrip('B')
        details['size_bytes_estimate']=round(float(size[1])*1024**({'':0,'K':1,'M':2,'G':3,'T':4}[unit]))
    return details


def reclassify_stored_record(record):
    """Upgrade old text cards using retained OCR only; explicit human edits take priority."""
    if record.get('type')!='text' or record.get('edited') or record.get('classification_source')=='manual':return record
    raw=record['text']
    value=unicodedata.normalize('NFKC',raw)
    size=SIZE.search(value)
    if size is None or not any(footer in value for footer in ['微信电脑版','微信 Windows','微信Mac']):return record
    parts=[]
    for line in value.splitlines():
        if SIZE.search(line) or any(footer in line for footer in ['微信电脑版','微信 Windows','微信Mac']):break
        if line.strip():parts.append(line.strip())
    if len(parts)>1:parts=[part for part in parts if part.upper() not in ICON_LABELS]
    filename=''.join(parts)
    if NAME.fullmatch(filename) is None:return record
    complete=record.get('top_state')=='visible' and record.get('bottom_state')=='visible'
    return {**record,'type':'file','text':filename,'file':_details(filename,raw,size,complete),
            'classification_source':'stored_ocr','original_type':'text'}


def mark_candidates(image,bubbles):
    for bubble in bubbles:
        x1,y1,x2,y2=bubble.box
        patch=image[y1:y2,x1:x2].astype(np.int16)
        h,w=patch.shape[:2]
        if bubble.kind=='unknown' and h>=45 and w>=80:
            # Probe before masking: a white card may be an attachment or a screenshot.
            from .media_layout import screenshot_structure
            bubble.metadata['_file_candidate']=bubble.kind
            hint=screenshot_structure(image[y1:y2,x1:x2])
            if hint:bubble.metadata['_screenshot_candidate']=hint
        if h<45 or not 1.3<=w/h<=6 or w<100:continue
        neutral=(patch.max(2)-patch.min(2)<18)&(patch.min(2)>205)
        right=patch[:max(1,int(h*.8)),int(w*.62):]
        color=(right.max(2)-right.min(2)>65)&(right.min(2)<170)
        gray_icon=(right.max(2)-right.min(2)<18)&(right.min(2)<190)
        if neutral.mean()>.60 and (color.mean()>.035 or .025<gray_icon.mean()<.35):
            bubble.metadata['_file_candidate']=bubble.kind


def classify_card(bubble):
    candidate=bubble.metadata.pop('_file_candidate',None)
    screenshot=bubble.metadata.pop('_screenshot_candidate',None)
    if bubble.kind not in {'text','file'} and candidate is None:return
    lines=sorted(bubble.lines,key=lambda line:(line.box[1],line.box[0]))
    raw='\n'.join(line.text for line in lines)
    value=unicodedata.normalize('NFKC',raw)
    size=SIZE.search(value)
    footer='微信电脑版' in value or '微信 Windows' in value or '微信Mac' in value
    parts=[]
    for line in lines:
        if (line.box[0]+line.box[2])/2>bubble.box[0]+(bubble.box[2]-bubble.box[0])*.74:continue
        if SIZE.search(line.text) or '微信电脑版' in line.text or '微信 Windows' in line.text:break
        parts.append(line.text.strip())
    filename=''.join(parts).strip()
    match=NAME.fullmatch(filename)
    generic=bool(filename and size and footer)
    if (match or generic) and (size or footer) and (candidate is not None or footer) and (not screenshot or footer):
        bubble.kind='file'
        bubble.metadata['file']=_details(filename,raw,size,bubble.top_state=='visible' and bubble.bottom_state=='visible')
    elif candidate is not None and candidate not in {'text','audio'}:
        bubble.kind=candidate
        bubble.lines=[]
        if screenshot:
            bubble.kind='image'
            bubble.metadata['image_layout']=screenshot
