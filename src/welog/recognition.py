"""Project-owned OCR/layout contracts; no image files or desktop dependencies."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol
import cv2
import numpy as np

Box = tuple[int, int, int, int]


@dataclass
class TextLine:
    text: str
    box: Box
    score: float = 1.0
    notice_removed: bool = False


@dataclass
class Bubble:
    box: Box
    direction: str
    kind: str = "text"
    top_state: str = "visible"
    bottom_state: str = "visible"
    lines: list[TextLine] = field(default_factory=list)
    ocr_status: str = "pending"
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class OcrBackend(Protocol):
    def recognize(self, bgr: np.ndarray) -> list[TextLine]: ...


class RapidOcrAdapter:
    """One lazily initialized CPU engine. Only ndarray input, never a temporary path."""
    def __init__(self, engine_factory=None):
        self.engine_factory = engine_factory
        self.engine = None

    def recognize(self, bgr: np.ndarray) -> list[TextLine]:
        if not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("OCR input must be a BGR uint8 array")
        if self.engine is None:
            if self.engine_factory is None:
                from rapidocr import RapidOCR
                self.engine_factory = RapidOCR
            self.engine = self.engine_factory()
        result = self.engine(bgr, use_cls=False)
        if result.boxes is None:
            return []
        height, width = bgr.shape[:2]
        lines = []
        for points, text, score in zip(result.boxes, result.txts, result.scores):
            x1, y1 = np.floor(np.min(points, axis=0)).astype(int)
            x2, y2 = np.ceil(np.max(points, axis=0)).astype(int)
            box = (max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2)))
            if text and box[2] > box[0] and box[3] > box[1]:
                lines.append(TextLine(str(text), box, float(score)))
        return sorted(lines, key=lambda line: (line.box[1], line.box[0]))


class EmptyOcr:
    """Explicit geometry-only fixture backend; never selected by the live UI."""
    def recognize(self, bgr):
        return []


def achromatic_media_regions(image, text_boxes=(), *, include_small_edge_media=False):
    """Conservative gray silhouettes, including viewport-clipped graphics.

    Rectangular chat panels and small disconnected glyphs remain text. These
    are layout hints; temporal stability still needs independent static proof.
    """
    mono=image if image.ndim==2 else cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    h,w=mono.shape;margin=max(1,min(24,int(w*.03)))
    background=float(np.median(np.concatenate([mono[:,:margin].ravel(),mono[:,-margin:].ravel()])))
    foreground=(np.abs(mono.astype(np.int16)-background)>12).astype(np.uint8)
    foreground[:4]=0;foreground[-4:]=0;foreground[:,:4]=0;foreground[:,-4:]=0
    contours,_=cv2.findContours(foreground,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    regions=[]
    for contour in contours:
        x,y,bw,bh=cv2.boundingRect(contour);clipped=y<=4 or y+bh>=h-4
        small_edge=include_small_edge_media and clipped
        if bw<(8 if small_edge else 45) or bh<(2 if small_edge else 8 if clipped else 24) or bw*bh<(64 if small_edge else 256):continue
        if x<w*.05 or x+bw>w*.94 or not (x<w*.25 or x+bw>w*.75):continue
        density=float(foreground[y:y+bh,x:x+bw].mean())
        if not .15<=density<=(.95 if small_edge else .85):continue
        if (mono[y:y+bh,x:x+bw]<background-35).mean()<.08:continue
        if image.ndim==3:
            patch=image[y:y+bh,x:x+bw].astype(np.int16)
            if (patch.max(2)-patch.min(2)>35).mean()>=.02:continue
        cx,cy=x+bw/2,y+bh/2
        if any(a<=cx<c and b<=cy<d for a,b,c,d in text_boxes):continue
        regions.append((max(0,x-2),0 if y<=4 else y-2,min(w,x+bw+2),h if y+bh>=h-4 else y+bh+2))
    return regions


def extend_media_components(image, regions, text_boxes=()):
    """Attach nearby compact satellites (e.g. a floating heart) to one graphic.

    Text panels and avatar lanes are excluded. Use original seeds, preventing
    recursive chains from growing into neighboring messages.
    """
    seeds=[tuple(box) for box in regions]
    if not seeds:return []
    mono=image if image.ndim==2 else cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    h,w=mono.shape;margin=max(1,min(24,int(w*.03)))
    bg=float(np.median(np.concatenate([mono[:,:margin].ravel(),mono[:,-margin:].ravel()])))
    mask=(np.abs(mono.astype(np.int16)-bg)>12).astype(np.uint8)
    mask[:4]=0;mask[-4:]=0;mask[:,:4]=0;mask[:,-4:]=0
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    result=[list(box) for box in seeds]
    for contour in contours:
        x,y,bw,bh=cv2.boundingRect(contour)
        if not (16<=bw<=44 and 12<=bh<=44) or bw*bh<192:continue
        if x<w*.05 or x+bw>w*.94:continue
        if mask[y:y+bh,x:x+bw].mean()<.4:continue
        if any(a<=x+bw/2<c and b<=y+bh/2<d for a,b,c,d in text_boxes):continue
        if any(a<=x and b<=y and c>=x+bw and d>=y+bh for a,b,c,d in seeds):continue
        matches=[]
        for i,(a,b,c,d) in enumerate(seeds):
            ox=max(0,min(c,x+bw)-max(a,x));oy=max(0,min(d,y+bh)-max(b,y))
            dx=max(0,a-x-bw,x-c);dy=max(0,b-y-bh,y-d)
            if (ox>=min(bw,c-a)*.5 and dy<=12) or (oy>=min(bh,d-b)*.5 and dx<=8):matches.append(i)
        if not matches:continue
        # Overlapping RGB/gray proposals may describe the same seed. Distinct
        # neighboring graphics cannot share an ambiguous satellite.
        first=seeds[matches[0]]
        if any(max(0,min(first[2],seeds[i][2])-max(first[0],seeds[i][0]))*
               max(0,min(first[3],seeds[i][3])-max(first[1],seeds[i][1]))<
               .5*min((first[2]-first[0])*(first[3]-first[1]),(seeds[i][2]-seeds[i][0])*(seeds[i][3]-seeds[i][1])) for i in matches[1:]):continue
        for i in matches:
            a,b,c,d=result[i];result[i]=[min(a,max(0,x-2)),min(b,max(0,y-2)),max(c,min(w,x+bw+2)),max(d,min(h,y+bh+2))]
    return [tuple(box) for box in result]


def detect_bubbles(bgr: np.ndarray, *, include_small_edge_media: bool = False) -> list[Bubble]:
    """Light-theme private-chat baseline. Keep long and viewport-clipped bodies."""
    h, w = bgr.shape[:2]
    values = bgr.astype(np.int16)
    margin=max(1,min(24,int(w*.03)))
    bg = np.median(np.concatenate([values[:,:margin].reshape(-1,3),values[:,-margin:].reshape(-1,3)]),axis=0)
    neutral = ((values.max(2) - values.min(2) <= 12) & (values.min(2) >= 215)
               & (np.max(np.abs(values - bg), axis=2) >= 5))
    green = ((values[:, :, 1] - values[:, :, 0] > 30)
             & (values[:, :, 1] - values[:, :, 2] > 25) & (values[:, :, 1] > 160))
    candidates = []
    for mask, color in ((neutral, "neutral"), (green, "green")):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 24 or bh < 8 or x < w * .035 or x + bw > w * .97:
                continue
            if mask[y:y+bh, x:x+bw].mean() < .48:
                continue
            direction = "outgoing" if color == "green" else ("incoming" if x < w * .25 else "outgoing" if x+bw > w*.75 else "unknown")
            if direction == "unknown":
                continue
            patch = bgr[y:y+bh, x:x+bw]
            # Very white rectangular content may be an image/card; never export its embedded text as chat.
            kind = "unknown" if color == "neutral" and np.median(patch) > 251 and bh > 65 else "text"
            candidates.append(Bubble((x,y,x+bw,y+bh), direction, kind,
                                     "clipped" if y <= 1 else "visible",
                                     "clipped" if y+bh >= h-1 else "visible"))
    # Non-bubble colored content is a media candidate, not OCR body text.
    foreground = (np.max(np.abs(values-bg),axis=2)>3).astype(np.uint8)
    contours,_ = cv2.findContours(foreground,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x,y,bw,bh=cv2.boundingRect(contour)
        clipped=y<=1 or y+bh>=h-1
        edge_fragment=include_small_edge_media and clipped and bw>=8 and bh>=2 and bw*bh>=64
        if ((bw<45 or bh<8 or (bh<35 and not clipped)) and not edge_fragment) or x<w*.035 or x+bw>w*.97:
            continue
        if not (x<w*.25 or x+bw>w*.75):
            continue
        contained=[b for b in candidates if x<=(b.box[0]+b.box[2])/2<=x+bw and y<=(b.box[1]+b.box[3])/2<=y+bh]
        if any(b.kind=='text' and (b.box[2]-b.box[0])*(b.box[3]-b.box[1])>=bw*bh*.7 for b in contained):
            continue
        patch=values[y:y+bh,x:x+bw]
        colorful=float(np.mean(patch.max(2)-patch.min(2)>35))
        white_panel=np.median(patch)>251 and bh>65
        if colorful < .02 and not white_panel:
            continue
        candidates=[b for b in candidates if not any(b is item for item in contained)]
        candidates.append(Bubble((x,y,x+bw,y+bh),"incoming" if x<w*.25 else "outgoing","image" if colorful>=.02 else "unknown",
                                 "clipped" if y<=1 else "visible","clipped" if y+bh>=h-1 else "visible"))
    if include_small_edge_media:
        # Detach the neutral viewport border without cutting a clipped sticker
        # into separate eyes/hat/mouth regions. This pass is temporal-only.
        interior=foreground.copy()
        interior[:4]=0;interior[-4:]=0;interior[:,:4]=0;interior[:,-4:]=0
        edge_contours,_=cv2.findContours(interior,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in edge_contours:
            x,y,bw,bh=cv2.boundingRect(contour)
            if not (y<=4 or y+bh>=h-4) or bw<2 or bh<2 or bw*bh<64:continue
            if x<w*.035 or x+bw>w*.97 or not (x<w*.25 or x+bw>w*.75):continue
            patch=values[y:y+bh,x:x+bw]
            if (patch.max(2)-patch.min(2)>35).mean()<.02:continue
            if any(item.kind=='text' and item.box[0]<=x and item.box[2]>=x+bw and item.box[1]<=y and item.box[3]>=y+bh for item in candidates):continue
            a,b,c,d=max(0,x-1),0 if y<=4 else y-1,min(w,x+bw+1),h if y+bh>=h-4 else y+bh+1
            candidates.append(Bubble((a,b,c,d),'incoming' if x<w*.25 else 'outgoing','image',
                'clipped' if y<=4 else 'visible','clipped' if y+bh>=h-4 else 'visible'))
        # A clipped colored tail can connect to a full-window neutral border,
        # making its foreground contour unusable. Inspect chroma separately.
        chroma=(values.max(2)-values.min(2)>35).astype(np.uint8)
        fragments,_=cv2.findContours(chroma,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in fragments:
            x,y,bw,bh=cv2.boundingRect(contour)
            if not (y<=8 or y+bh>=h-8) or bw<2 or bh<2 or cv2.contourArea(contour)<32:continue
            if x<w*.035 or x+bw>w*.97 or not (x<w*.25 or x+bw>w*.75):continue
            a,b,c,d=max(0,x-3),max(0,y-3),min(w,x+bw+3),min(h,y+bh+3)
            if any(item.kind=='text' and item.box[0]<=x and item.box[2]>=x+bw and item.box[1]<=y and item.box[3]>=y+bh for item in candidates):continue
            if any(item.kind=='image' and item.box[0]<=a and item.box[1]<=b and item.box[2]>=c and item.box[3]>=d for item in candidates):continue
            candidates.append(Bubble((a,b,c,d),'incoming' if x<w*.25 else 'outgoing','image',
                'clipped' if y<=8 else 'visible','clipped' if y+bh>=h-8 else 'visible'))
    # Gray stickers can carry substantial motion despite having no chroma.
    for box in achromatic_media_regions(bgr,[b.box for b in candidates if b.kind=='text'],include_small_edge_media=include_small_edge_media):
        a,b,c,d=box
        if any(item.kind in {'image','unknown'} and item.box[0]<=a and item.box[1]<=b and item.box[2]>=c and item.box[3]>=d for item in candidates):continue
        candidates.append(Bubble(box,'incoming' if a<w*.25 else 'outgoing','image',
            'clipped' if b==0 else 'visible','clipped' if d==h else 'visible'))
    media=[b for b in candidates if b.kind=='image']
    for bubble,box in zip(media,extend_media_components(bgr,[b.box for b in media],[b.box for b in candidates if b.kind=='text'])):
        bubble.box=box
        if box[1]==0:bubble.top_state='clipped'
        if box[3]==h:bubble.bottom_state='clipped'
    return sorted(candidates,key=lambda b:(b.box[1],b.box[0]))


def assign_lines(bubbles: list[Bubble], lines: list[TextLine]) -> list[TextLine]:
    from .text_rules import clean_lines,sanitize_bubble
    labels = []
    for line in lines:
        cx,cy=(line.box[0]+line.box[2])/2,(line.box[1]+line.box[3])/2
        match=next((b for b in bubbles if (b.kind in {"text","audio","file"} or '_file_candidate' in b.metadata)
                    and b.box[0]<=cx<b.box[2] and b.box[1]<=cy<b.box[3]),None)
        if match is not None:
            match.lines.append(line)
        else:
            labels.append(line)
    from .file_cards import classify_card
    for bubble in bubbles:classify_card(bubble)
    bubbles[:] = [bubble for bubble in bubbles if sanitize_bubble(bubble)]
    for bubble in bubbles:
        bubble.lines.sort(key=lambda line:(line.box[1],line.box[0]))
        bubble.ocr_status = "ok" if bubble.lines else "failed" if bubble.kind in {"text","audio"} else "not_applicable"
    return clean_lines(labels)
