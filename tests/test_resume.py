import hashlib
import json

import pytest

from test_full_capture import Conversation, KnownText
from welog.archive import read_jsonl
from welog.controller import resume_capture, run_capture
from welog.exports import RecordSource
from welog.models import CaptureConfig
from welog.record_state import record_name, rename_record, save_edit
from welog.resume import ResumeError, load_resume_point, resume_lock


def start_record(path, direction):
    source = Conversation(direction, count=14)
    initial = source.position
    rows = []
    result = run_capture(source, path, config=CaptureConfig(max_frames=0, direction=direction,
        debug_save_images=True, sample_interval_s=.001, stable_samples=2, settle_delay_s=0,
        scroll_event_interval_s=0, boundary_wait_s=0, max_scroll_events=1),
        ocr_backend=KnownText(source), progress=rows.append, stop_requested=lambda: len(rows) == 3)
    assert result['status'] == 'stopped' and result['accepted'] == 3
    return source, initial


def snapshot(path):
    return {file.relative_to(path): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in path.rglob('*') if file.is_file()}


@pytest.mark.parametrize('direction', ['up', 'down'])
def test_resume_appends_same_record_and_preserves_message_ids_and_edits(tmp_path, direction):
    path = tmp_path / 'capture'
    first, initial = start_record(path, direction)
    before = list(RecordSource(path).records())
    source = RecordSource(path)
    rename_record(source, '续采保持记录名称')
    edited = before[len(before)//2]
    save_edit(source, edited, {'text': '用户手动修订'})
    point = load_resume_point(path)
    old_frames = (path/'frames.jsonl').read_bytes()
    old_alignment = (path/'alignments.jsonl').read_bytes()
    second = Conversation(direction, count=14)
    second.position = first.position
    result = resume_capture(second, path, ocr_backend=KnownText(second))
    assert result['status'] == 'boundary_reached' and result['output'] == str(path)
    assert result['final_scroll_offset_px'] == second.position - initial
    assert result['image_memory_remaining_bytes'] == 0
    assert (path/'frames.jsonl').read_bytes().startswith(old_frames)
    assert (path/'alignments.jsonl').read_bytes().startswith(old_alignment)
    frames = read_jsonl(path/'frames.jsonl')
    assert frames[3]['frame_id'] == '000003' and frames[3]['action']['kind'] == 'resume'
    assert frames[3]['action']['checkpoint_frame_id'] == point.frame['frame_id']
    journal = read_jsonl(path/'alignments.jsonl')
    assert journal[3]['reference_id'] == '000002' and journal[3]['scroll_offset_px'] == point.offset
    assert result['accepted'] == len(journal) == len(frames)
    merged = list(RecordSource(path).records())
    assert len(merged) == len(second.messages)
    assert record_name(source) == '续采保持记录名称'
    assert {r['message_id'] for r in before} <= {r['message_id'] for r in merged}
    texts = [r['text'] for r in merged]
    assert texts == ['用户手动修订' if text == edited['text'] else text for _, text in second.messages]
    assert [row['global_top_px'] for row in merged] == [top - initial for top, _ in second.messages]
    meta = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    assert meta['created_at'] == point.metadata['created_at'] and meta['resume_count'] == 1


def test_already_scrolled_view_joins_at_measured_offset_and_reuses_checkpoint_again(tmp_path):
    path = tmp_path/'capture'
    first, initial = start_record(path, 'down')
    source = Conversation('down', count=14)
    source.position = first.position + 120
    progress = []
    result = resume_capture(source, path, ocr_backend=KnownText(source),
                            progress=progress.append, stop_requested=lambda: bool(progress))
    assert result['status'] == 'stopped' and not source.inputs
    assert result['final_scroll_offset_px'] == source.position - initial
    assert load_resume_point(path).frame['frame_id'] == '000003'
    assert read_jsonl(path/'alignments.jsonl')[-1]['alignment']['delta_px'] == 120


@pytest.mark.parametrize('failure', ['unrelated', 'opposite', 'checksum', 'locked'])
def test_resume_preflight_errors_do_not_write_or_scroll(tmp_path, failure):
    path = tmp_path/'capture'
    first, _ = start_record(path, 'down')
    point = load_resume_point(path)
    source = Conversation('down', count=14)
    source.position = first.position - 120 if failure == 'opposite' else first.position
    if failure == 'unrelated':
        source.scene[:] = 250
    if failure == 'checksum':
        point.image_path.write_bytes(b'bad checksum')
    before = snapshot(path)
    if failure == 'locked':
        with resume_lock(path), pytest.raises(ResumeError, match='正在续采'):
            resume_capture(source, path, ocr_backend=KnownText(source))
    else:
        with pytest.raises(ResumeError):
            resume_capture(source, path, ocr_backend=KnownText(source))
    assert snapshot(path) == before and source.closed and not source.inputs


def test_unstable_resume_does_not_record_diagnostic_frames(tmp_path, monkeypatch):
    from welog import controller
    path = tmp_path/'capture'
    first, _ = start_record(path, 'down')
    before = snapshot(path)
    source = Conversation('down', count=14)
    source.position = first.position

    def unstable(source, config, *args):
        assert not config.debug_save_images
        raise controller.StableTimeout(None)

    monkeypatch.setattr(controller, 'wait_for_stable', unstable)
    with pytest.raises(ResumeError, match='稳定的聊天画面'):
        resume_capture(source, path, ocr_backend=KnownText(source))
    assert snapshot(path) == before and source.closed and not source.inputs
