import json

import numpy as np

from welog.archive import read_jsonl
from welog.controller import run_capture
from welog.demo import make_scene
from welog.models import CaptureConfig
from welog.scroll import ScrollPlanner
from welog.replay import run_replay


def test_event_planner_leaves_overlap_margin_and_restarts_after_recovery():
    planner = ScrollPlanner()
    for _ in range(8):
        previous = planner.events
        planner.observe(previous * 50, previous, 724)
        assert planner.events * 50 * 1.25 <= 724 * 0.55
        assert planner.events <= 8
    assert planner.events == 6
    planner.recover()
    assert planner.events == 1


def test_separate_wheel_events_cover_more_content_when_each_event_is_clamped(tmp_path):
    scene = make_scene(6000, 640, seed=14)

    class ClampedSource:
        def __init__(self):
            self.position = 0
            self.inputs = []
        def read(self):
            return scene[self.position:self.position + 720].copy()
        def scroll(self, units):
            self.inputs.append(units)
            # Same per-event clamp observed in the saved WeChat run.
            self.position += 50 if units > 0 else -50
        def close(self):
            pass

    results = []
    for maximum in (1, 8):
        source = ClampedSource()
        result = run_capture(source, str(tmp_path / f"events-{maximum}"), config=CaptureConfig(
            max_frames=12, sample_interval_s=0.001, stable_samples=2,
            settle_delay_s=0, scroll_event_interval_s=0, max_scroll_events=maximum))
        assert result["rejected"] == 0
        assert result["final_scroll_offset_px"] == source.position
        assert set(source.inputs) == {120}
        results.append(result)
    assert results[1]["final_scroll_offset_px"] >= results[0]["final_scroll_offset_px"] * 4
    (tmp_path / "comparison.json").write_text(json.dumps(results), encoding="utf-8")


def test_sudden_scroll_acceleration_pauses_without_reversing_or_committing_a_gap(tmp_path):
    scene = make_scene(8000, 640, seed=91)

    class AcceleratingSource:
        def __init__(self):
            self.position = 0
            self.calls = 0
            self.inputs = []
        def read(self):
            return scene[self.position:self.position + 720].copy()
        def scroll(self, units):
            self.calls += 1
            self.inputs.append(units)
            step = 50 if self.calls < 8 else 160
            self.position = max(0, self.position + (step if units > 0 else -step))
        def close(self):
            pass

    source = AcceleratingSource()
    errors = []
    result = run_capture(source, str(tmp_path / "capture"), config=CaptureConfig(
        max_frames=12, sample_interval_s=0.001, stable_samples=2,
        settle_delay_s=0, scroll_event_interval_s=0),
        progress=lambda item: errors.append(item["scroll_offset_px"] - source.position) if item["accepted"] else None)
    assert result["rejected"] > 0
    assert all(units > 0 for units in source.inputs)
    assert result['reason'] == 'continuity_unconfirmed'
    assert errors and all(error == 0 for error in errors)


def test_cancellation_interrupts_a_batch_before_the_next_event(tmp_path):
    scene = make_scene(3000, 640, seed=14)

    class Source:
        def __init__(self):
            self.calls = 0
            self.position = 0
        def read(self):
            return scene[self.position:self.position + 720].copy()
        def scroll(self, units):
            self.calls += 1
            self.position += 50
        def close(self):
            pass

    source = Source()
    result = run_capture(source, str(tmp_path / "capture"), config=CaptureConfig(
        max_frames=12, sample_interval_s=0.001, stable_samples=2,
        settle_delay_s=0, scroll_event_interval_s=0), stop_requested=lambda: source.calls >= 2)
    assert result["status"] == "stopped"
    assert source.calls == 2
    assert result["final_scroll_offset_px"] == 50  # Partial second batch is uncommitted.


def test_scrolling_animated_content_preserves_capture_and_replay_coordinates(tmp_path):
    scene = make_scene(6000, 640, seed=14)
    rng = np.random.default_rng(72)

    class AnimatedScrollingSource:
        def __init__(self):
            self.position = 1000
        def read(self):
            sample = scene[self.position:self.position + 720].copy()
            top = 1100 - self.position
            height = min(80, 720 - top)
            if 0 <= top < 720:
                sample[top:top + height, 500:620] = rng.integers(0, 256, (height, 120, 3), dtype=np.uint8)
            return sample
        def scroll(self, units):
            self.position = max(0, self.position + (50 if units > 0 else -50))
        def close(self):
            pass

    source = AnimatedScrollingSource()
    errors = []
    output = tmp_path / "capture"
    result = run_capture(source, str(output), config=CaptureConfig(
        direction="up", max_frames=8, sample_interval_s=0.001, stable_samples=3,
        settle_delay_s=0, scroll_event_interval_s=0),
        progress=lambda item: errors.append(item["scroll_offset_px"] - (source.position - 1000))
        if item["accepted"] else None)
    assert result["rejected"] == 0
    assert errors and all(error == 0 for error in errors)
    frames = read_jsonl(output / "frames.jsonl")
    assert any(frame["action"]["stability"]["reason"] == "local_animation" for frame in frames)
    report=json.loads((output/'report.json').read_text(encoding='utf-8'))
    assert report['final_scroll_offset_px'] == result['final_scroll_offset_px']
    assert report['image_memory_remaining_bytes']==0
    assert not list(output.rglob('*.png'))
