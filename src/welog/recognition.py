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


def detect_bubbles(bgr: np.ndarray) -> list[Bubble]:
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
        if bw<45 or bh<35 or x<w*.035 or x+bw>w*.97:
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
