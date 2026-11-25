from datetime import datetime
import json

import numpy as np
import pytest

from welog.assembler import FragmentAssembler
from welog.exports import RecordSource,preview_records
from welog.file_cards import mark_candidates
from welog.message_store import MessageStore
from welog.recognition import Bubble,TextLine,assign_lines
from welog.record_state import connect,rebuild_intervals
from welog.timeline import CHAT_TZ,TimelineIndex,interval_for,observe_markers,parse_timestamp


@pytest.mark.parametrize('raw,expected',[
    ('2025年12月4日 22:00','2025-12-04T22:00+08:00'),('2025/12/02 14:22','2025-12-02T14:22+08:00'),
    ('昨天 09:30','2026-11-15T09:30+08:00'),('22:01','2026-11-16T22:01+08:00'),
    ('星期日 18:00','2026-11-15T18:00+08:00'),('12月31日 23:50','2025-12-31T23:50+08:00'),
])
def test_timestamp_formats_use_fixed_utc8(raw,expected):
    assert parse_timestamp(raw,datetime(2026,11,16,23,0,tzinfo=CHAT_TZ))[0]==expected


@pytest.mark.parametrize('raw',['明天见','19\'\'','开会时间 12:00','2025年2月30日 22:00','25:00'])
def test_body_text_and_invalid_dates_are_not_timestamps(raw):assert parse_timestamp(raw) is None


@pytest.mark.parametrize('offsets',[[0,-100],[-100,0]])
def test_intervals_are_derived_from_global_position_in_both_scan_directions(tmp_path,offsets):
    store=MessageStore(tmp_path/'messages.sqlite3')
    assembler=FragmentAssembler(store)
    try:
        for index,offset in enumerate(offsets):
            markers=[TextLine('2025年12月2日 12:41',(240,100-offset,360,115-offset)),
                     TextLine('2025年12月2日 14:22',(240,300-offset,360,315-offset))]
            observe_markers(store.connection,markers,offset,600,str(index))
        for top in [20,160,350]:
            assembler.observe('frame',Bubble((45,top,220,top+50),'incoming',lines=[TextLine(str(top),(55,top+10,210,top+30))]),0,500)
        rows=list(store.iter_messages())
        assert [row['time_interval']['status'] for row in rows]==['open_start','bounded','open_end']
        middle=rows[1]['time_interval']
        assert middle['start']=='2025-12-02T12:41+08:00' and middle['end']=='2025-12-02T14:22+08:00'
        assert middle['precision']=='interval' and 'sent_at' not in rows[1]
        assert store.connection.execute('SELECT COUNT(*) FROM time_markers').fetchone()[0]==2
        index = TimelineIndex(store.connection, include_evidence=True)
        for top, bottom in [(0, 30), (115, 300), (109, 305), (320, 400)]:
            assert index.interval(top, bottom) == interval_for(store.connection, top, bottom)
        store.connection.commit()
        source=RecordSource(tmp_path)
        stats=preview_records(source)
        label=next(group for group in stats['statistics']['time_groups'] if '12:41' in group and '14:22' in group)
        assert preview_records(source,time_filter=label)['matched']==1
    finally:store.close()


def test_timestamp_like_message_inside_a_bubble_is_not_a_separator(tmp_path):
    store=MessageStore(tmp_path/'messages.sqlite3')
    try:
        bubbles=[Bubble((40,40,250,100),'incoming')]
        labels=assign_lines(bubbles,[TextLine('12:00',(50,50,100,80))])
        observe_markers(store.connection,labels,0,600,'frame')
        assert interval_for(store.connection,110,150)['status']=='unassigned'
    finally:store.close()


def test_old_journal_backfill_excludes_unconfirmed_frames(tmp_path):
    source_file=tmp_path/'messages.jsonl'
    source_file.write_text('{"message_id":"m","text":"正文","global_top_px":200,"global_bottom_px":250}\n',encoding='utf-8')
    (tmp_path/'frames.jsonl').write_text(''.join(json.dumps({'frame_id':str(i),'width':600})+'\n' for i in range(3)))
    alignments=[{'frame_id':str(i),'alignment':{'accepted':i!=1},'scroll_offset_px':0 if i!=1 else None} for i in range(3)]
    (tmp_path/'alignments.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in alignments))
    observations=[{'frame_id':str(i),'coordinate_status':'confirmed' if i!=1 else 'unknown',
        'labels':[{'text':f'2025年12月2日 {hour}:00','box':[240,y,360,y+15]}]}
        for i,hour,y in [(0,'12',100),(1,'13',170),(2,'14',300)]]
    (tmp_path/'observations.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in observations))
    source=RecordSource(source_file)
    rebuild_intervals(source);rebuild_intervals(source)
    interval=next(source.records())['time_interval']
    assert interval['start']=='2025-12-02T12:00+08:00' and interval['end']=='2025-12-02T14:00+08:00'
    with connect(source) as db:assert db.execute('SELECT COUNT(*) FROM time_markers').fetchone()[0]==2


def test_bare_times_crossing_midnight_use_each_frames_capture_date(tmp_path):
    source_file=tmp_path/'messages.jsonl'
    source_file.write_text('{"text":"message","global_top_px":200,"global_bottom_px":250}\n',encoding='utf-8')
    frames=[{'frame_id':str(i),'width':600,'captured_at':stamp} for i,stamp in enumerate(['2026-11-16T15:59:00+00:00','2026-11-16T16:01:00+00:00'])]
    (tmp_path/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in frames),encoding='utf-8')
    alignments=[{'frame_id':str(i),'alignment':{'accepted':True},'scroll_offset_px':0} for i in range(2)]
    (tmp_path/'alignments.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in alignments),encoding='utf-8')
    rows=[{'frame_id':str(i),'coordinate_status':'confirmed','labels':[{'text':stamp,'box':[240,top,360,top+15]}]}
          for i,stamp,top in [(0,'23:59',100),(1,'00:01',300)]]
    (tmp_path/'observations.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
    source=RecordSource(source_file)
    rebuild_intervals(source)
    interval=next(source.records())['time_interval']
    assert interval['status']=='bounded' and interval['start']=='2026-11-16T23:59+08:00' and interval['end']=='2026-11-17T00:01+08:00'
    assert interval['start_precision']=='relative_to_capture_date' and len(interval['start_evidence'])==1


@pytest.mark.parametrize('extension',['pdf','pptx','docx','xlsx','zip','rar','7z','mp4','png','py','gz'])
def test_file_card_types_preserve_filename_size_and_download_state(extension):
    image=np.full((200,600,3),250,np.uint8)
    image[30:130,45:300]=237
    image[45:90,250:280]=(50,50,230)
    bubble=Bubble((45,30,300,130),'incoming')
    mark_candidates(image,[bubble])
    assert '_file_candidate' in bubble.metadata
    lines=[TextLine(f'中文附件.{extension}',(55,45,230,65)),TextLine('PDF',(251,55,278,75)),
           TextLine('30.7M 未下载',(55,78,200,94)),TextLine('微信电脑版',(65,107,180,123))]
    assign_lines([bubble],lines)
    assert bubble.kind=='file'
    assert bubble.metadata['file']['filename']==f'中文附件.{extension}'
    assert bubble.metadata['file']['extension']==extension
    assert bubble.metadata['file']['size_display']=='30.7M'
    assert bubble.metadata['file']['download_status']=='not_downloaded'


def test_wrapped_pdf_name_and_plain_file_mentions_are_distinct():
    card=Bubble((45,20,300,130),'incoming')
    assign_lines([card],[TextLine('长文件名称第一行',(55,30,230,50)),TextLine('第二行.pdf',(55,55,210,75)),
                        TextLine('268.0K',(55,80,180,95)),TextLine('微信电脑版',(60,105,190,122))])
    assert card.kind=='file' and card.metadata['file']['filename']=='长文件名称第一行第二行.pdf'
    plain=Bubble((45,20,300,90),'incoming')
    assign_lines([plain],[TextLine('请看看附件.pdf',(55,30,230,50))])
    assert plain.kind=='text' and 'file' not in plain.metadata


def test_grayscale_file_icon_and_filename_without_extension():
    image=np.full((200,600,3),250,np.uint8)
    image[30:130,45:300]=237
    image[45:90,250:280]=160
    bubble=Bubble((45,30,300,130),'incoming','unknown')
    mark_candidates(image,[bubble])
    assert '_file_candidate' in bubble.metadata
    assign_lines([bubble],[TextLine('Dockerfile',(55,45,200,65)),TextLine('2.0K',(55,78,200,94)),TextLine('微信电脑版',(65,107,180,123))])
    assert bubble.kind=='file' and bubble.metadata['file']['filename']=='Dockerfile' and bubble.metadata['file']['extension']==''


def test_partial_media_can_be_upgraded_to_file_without_duplicate_message(tmp_path):
    store=MessageStore(tmp_path/'messages.sqlite3');assembler=FragmentAssembler(store)
    try:
        assembler.observe('a',Bubble((45,10,300,90),'incoming','image',top_state='clipped'),0,500)
        card=Bubble((45,0,300,90),'incoming','file',metadata={'file':{'filename':'完整.pdf','extension':'pdf'}})
        assembler.observe('b',card,0,500)
        rows=list(store.iter_messages())
        assert len(rows)==1 and rows[0]['type']=='file' and rows[0]['text']=='完整.pdf'
    finally:store.close()
