import json

import numpy as np

from welog.archive import read_jsonl
from welog.controller import run_capture
from welog.demo import make_scene
from welog.models import CaptureConfig


class FakeDesktop:
    def __init__(self, *, bad_after_first_scroll=False, fail_reads=False):
        self.scene = make_scene(1400, 640, seed=42)
        self.offset = 0
        self.scrolls = []
        self.closed = False
        self.bad_after_first_scroll = bad_after_first_scroll
        self.fail_reads = fail_reads

    def read(self):
        if self.fail_reads:
            raise RuntimeError("target lost foreground")
        if self.bad_after_first_scroll and self.scrolls:
            return np.full((720, 640, 3), 255, np.uint8)
        return self.scene[self.offset:self.offset + 720].copy()

    def scroll(self, units):
        self.scrolls.append(units)
        self.offset = max(0, min(680, self.offset + units))

    def close(self):
        self.closed = True


def settings(**kwargs):
    return CaptureConfig(max_frames=20, sample_interval_s=0.001, stable_timeout_s=0.2,
                         stable_samples=2, settle_delay_s=0, **kwargs)


def test_live_loop_calibrates_measured_movement_and_stops_at_visible_end(tmp_path):
    source = FakeDesktop()
    result = run_capture(source, str(tmp_path / "live"), config=settings())
    assert source.closed
    assert result["status"] == "visible_end_candidate"
    assert result["final_scroll_offset_px"] == 680
    assert source.scrolls[0] == 120
    assert all(units == 120 for units in source.scrolls)
    frames = read_jsonl(tmp_path / "live/frames.jsonl")
    assert max(frame["action"].get("events", 0) for frame in frames) > 1
    records = read_jsonl(tmp_path / "live/alignments.jsonl")
    assert all(record["alignment"]["accepted"] for record in records)


def test_alignment_failure_only_retries_in_place(tmp_path):
    source = FakeDesktop(bad_after_first_scroll=True)
    result = run_capture(source, str(tmp_path / "live"), config=settings())
    assert result["status"] == "needs_attention"
    assert result["final_scroll_offset_px"] == 0
    assert result["rejected"] >= 2
    assert source.scrolls[0] > 0
    assert source.scrolls == [120]
    checkpoint = json.loads((tmp_path / "live/checkpoint.json").read_text())
    assert checkpoint["frame_id"] == "000000"
    assert source.closed


def test_foreground_loss_is_archived_and_resources_are_released(tmp_path):
    source = FakeDesktop(fail_reads=True)
    result = run_capture(source, str(tmp_path / "live"), config=settings())
    assert result["status"] == "needs_attention"
    assert result["reason"] == "RuntimeError"
    assert result["accepted"] == 0
    assert source.closed


def test_cancel_before_first_frame_does_not_scroll_or_capture(tmp_path):
    source = FakeDesktop()
    result = run_capture(source, str(tmp_path / "live"), config=settings(), stop_requested=lambda: True)
    assert result["status"] == "stopped"
    assert result["accepted"] == 0
    assert source.scrolls == []
    assert source.closed


def test_cancel_after_first_frame_prevents_next_scroll(tmp_path):
    source = FakeDesktop()
    observations = []
    result = run_capture(source, str(tmp_path / "live"), config=settings(),
                         progress=observations.append, stop_requested=lambda: bool(observations))
    assert result["status"] == "stopped"
    assert result["accepted"] == 1
    assert source.scrolls == []
    assert source.closed


def test_upward_history_capture_uses_negative_motion_and_signed_coordinates(tmp_path):
    source = FakeDesktop()
    source.offset = 680
    result = run_capture(source, str(tmp_path / "history"), config=settings(direction="up"))
    assert result["status"] == "visible_end_candidate"
    assert source.offset == 0
    assert all(units < 0 for units in source.scrolls)
    assert result["final_scroll_offset_px"] == -680
    records = read_jsonl(tmp_path / "history/alignments.jsonl")
    assert all(record["alignment"]["accepted"] for record in records)


def test_starting_at_boundary_is_not_reported_as_successful_capture(tmp_path):
    source = FakeDesktop()
    source.offset = 680
    result = run_capture(source, str(tmp_path / "stationary"), config=settings())
    assert result["status"] == "needs_attention"
    assert result["reason"] == "no_scroll_motion"
    assert result["final_scroll_offset_px"] == 0
    assert result["accepted"] == 4


def test_upward_alignment_failure_never_rolls_back_downward(tmp_path):
    source = FakeDesktop(bad_after_first_scroll=True)
    source.offset = 680
    result = run_capture(source, str(tmp_path / "history"), config=settings(direction="up"))
    assert result["status"] == "needs_attention"
    assert source.scrolls[0] < 0
    assert source.scrolls == [-120]
    assert result["final_scroll_offset_px"] == 0
