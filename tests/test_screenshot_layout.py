import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from welog.file_cards import mark_candidates
from welog.media_layout import screenshot_structure
from welog.recognition import Bubble,TextLine,assign_lines


def screenshot():
    image=np.full((160,260,3),255,np.uint8)
    for y in range(20,140,14):
        for x in range(15,240,13):cv2.rectangle(image,(x,y),(x+3,y+4),(80,80,80),-1)
    return image


def test_screenshot_is_image_and_embedded_ocr_is_not_chat_text():
    image=screenshot();bubble=Bubble((0,0,260,160),'incoming','unknown')
    mark_candidates(image,[bubble])
    assign_lines([bubble],[TextLine('document.pdf',(20,20,120,35))])
    assert bubble.kind=='image' and bubble.lines==[]
    assert bubble.metadata['image_layout']['method']=='dense_small_components'


def test_file_card_evidence_takes_priority_over_screenshot_layout():
    image=screenshot();bubble=Bubble((0,0,260,160),'incoming','unknown')
    mark_candidates(image,[bubble])
    assign_lines([bubble],[TextLine('example.pdf',(20,20,140,35)),TextLine('2.0M',(20,50,100,65)),TextLine('微信电脑版',(20,80,140,95))])
    assert bubble.kind=='file' and bubble.metadata['file']['filename']=='example.pdf'


def test_file_listing_inside_a_screenshot_is_not_an_attachment():
    image=screenshot();bubble=Bubble((0,0,260,160),'incoming','unknown')
    mark_candidates(image,[bubble])
    assign_lines([bubble],[TextLine('example.pdf',(20,20,140,35)),TextLine('2.0M',(20,50,100,65))])
    assert bubble.kind=='image' and bubble.lines==[]


def test_blank_panel_and_single_icon_remain_unknown():
    image=np.full((160,260,3),255,np.uint8)
    cv2.rectangle(image,(180,50),(220,95),(80,80,80),-1)
    assert screenshot_structure(image) is None
    bubble=Bubble((0,0,260,160),'incoming','unknown')
    mark_candidates(image,[bubble]);assign_lines([bubble],[])
    assert bubble.kind=='unknown'


@pytest.mark.parametrize('reverse',[False,True])
def test_clipped_unknown_and_full_screenshot_merge_without_duplicate(tmp_path,reverse):
    from welog.assembler import FragmentAssembler
    from welog.message_store import MessageStore
    store=MessageStore(tmp_path/'messages.sqlite3');assembler=FragmentAssembler(store)
    try:
        fragment=Bubble((40,0,280,98),'incoming','unknown',top_state='clipped')
        full=Bubble((40,58,280,198),'incoming','image',metadata={'image_layout':{'method':'dense_small_components'}})
        views=[('clip',fragment,100),('full',full,0)]
        if reverse:views.reverse()
        for frame,bubble,offset in views:assembler.observe(frame,bubble,offset,500)
        rows=list(store.iter_messages())
        assert len(rows)==1 and rows[0]['type']=='image' and rows[0]['complete']
    finally:store.close()


def test_separate_unknown_card_and_image_are_not_merged(tmp_path):
    from welog.assembler import FragmentAssembler
    from welog.message_store import MessageStore
    store=MessageStore(tmp_path/'messages.sqlite3');assembler=FragmentAssembler(store)
    try:
        assembler.observe('a',Bubble((40,40,280,140),'incoming','unknown'),0,500)
        assembler.observe('b',Bubble((40,160,280,260),'incoming','image'),0,500)
        assert len(list(store.iter_messages()))==2
    finally:store.close()


ROOT=Path('runs/pointer-20261117-123853-b3e31c/capture')
@pytest.mark.skipif(not ROOT.exists(),reason='Private offline fixture unavailable')
def test_all_five_real_unknown_screenshots_are_classified_by_pixels_not_ids():
    from welog.exports import RecordSource
    from welog.imageio import read_image
    count=0
    for record in RecordSource(ROOT).raw_records():
        if record['type']!='unknown':continue
        evidence=max(record['evidence'],key=lambda e:(e['top_state']=='visible' and e['bottom_state']=='visible',e['box'][3]-e['box'][1]))
        image=read_image(ROOT/'frames'/f"{evidence['frame_id']}.png")
        bubble=Bubble(tuple(evidence['box']),record['direction'],'unknown')
        mark_candidates(image,[bubble]);assign_lines([bubble],[])
        assert bubble.kind=='image'
        count+=1
    assert count==5
