import json

import cv2
import numpy as np
import pytest

from welog.archive import read_jsonl
from welog.controller import run_capture
from welog.demo import make_scene
from welog.models import CaptureConfig
from welog.scroll import ScrollPlanner,ScrollCalibration,retained_anchor_count
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


def test_command_bounds_use_only_independent_samples_of_the_same_batch():
    calibration=ScrollCalibration()
    for delta in [-201,-205]:calibration.observe(delta,3,'text')
    assert calibration.batch_bounds(3,'up',663) is None
    calibration.observe(-197,3,'text')
    bound=calibration.batch_bounds(3,'up',663)
    assert bound['minimum']==-265 and bound['maximum']==0 and bound['events']==3
    assert calibration.batch_bounds(3,'down',663)['maximum']==265
    assert calibration.batch_bounds(4,'up',663) is None
    calibration.observe(-9,3,'text')  # Partial movement must remain in the range.
    calibration.observe(-350,3,'bounded_layout')  # No self-training.
    assert calibration.batch_bounds(3,'up',663)['minimum']==-265
    calibration.observe(-480,3,'text')
    assert calibration.batch_bounds(3,'up',663) is None


def test_reference_anchors_are_checked_in_the_directional_retained_overlap():
    bottom=[(70,548,95,565),(70,602,95,620)]
    top=[(70,43,95,60),(70,99,95,116)]
    assert retained_anchor_count(bottom,663,'up',265)==0
    assert retained_anchor_count(bottom,663,'down',265)==2
    assert retained_anchor_count(top,663,'down',265)==0
    assert retained_anchor_count(top,663,'up',265)==2


def test_overridden_single_events_do_not_accumulate_an_unexecuted_batch():
    planner=ScrollPlanner()
    for _ in range(8):
        events=planner.events;planner.observe(events*75,events,663)
    assert planner.events==3
    for _ in range(12):
        planner.observe(37,1,663)  # Media guard overrides the requested batch.
        assert planner.events<=2
    planner.observe(104,2,663)
    assert planner.events==4
    planner.observe(300,4,663)
    assert planner.events==3


@pytest.mark.parametrize('direction',['up','down'])
def test_leaving_media_guard_restores_speed_without_losing_overlap(tmp_path,direction):
    from welog.recognition import TextLine
    class Source:
        def __init__(self):
            self.scene=np.full((3000,880,3),250,np.uint8)
            self.rng=np.random.default_rng(146);self.texts=[]
            self.media=[y if direction=='down' else 3000-y-132 for y in [100,280]]
            for y in self.media:self.scene[y:y+36,834:870]=self.rng.integers(0,256,(36,36,3),dtype=np.uint8)
            marker=500 if direction=='down' else 3000-500-36
            self.scene[marker:marker+36,20:56]=self.rng.integers(0,256,(36,36,3),dtype=np.uint8)
            for i,base in enumerate(range(650,2950,70)):
                y=base if direction=='down' else 3000-base-48;text=f'row {i:03d}'
                self.texts.append((y,text));self.scene[y:y+48,70:300]=238
                cv2.putText(self.scene,text,(80,y+32),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1)
            self.end=3000-663;self.position=0 if direction=='down' else self.end
            self.pending=0;self.inputs=[];self.closed=False
        def read(self):
            if self.pending:
                distance={1:37,2:104,3:201,4:300,5:400,6:500,7:595,8:695}[self.pending]
                self.position=max(0,min(self.end,self.position+(distance if direction=='down' else -distance)))
                self.pending=0
            image=self.scene[self.position:self.position+663].copy()
            for y in self.media:
                a,b=max(0,y-self.position),min(663,y+132-self.position)
                if b>a:
                    pixels=self.rng.integers(0,160,(b-a,132,3),dtype=np.uint8);pixels[:,:,1]=40;pixels[:,:,2]=240
                    image[a:b,687:819]=pixels
            return image
        def scroll(self,units):self.pending+=1;self.inputs.append(units)
        def close(self):self.closed=True
    source=Source();initial=source.position
    class Backend:
        def recognize(self,image):
            if len(image)!=663:return []
            return [TextLine(text,(80,y+12-source.position,270,y+30-source.position),.99)
                    for y,text in source.texts if 2<y+12-source.position and y+30-source.position<661]
    result=run_capture(source,tmp_path/'capture',config=CaptureConfig(max_frames=0,direction=direction,
        stable_samples=3,stable_timeout_s=3,sample_interval_s=.001,settle_delay_s=0,
        scroll_event_interval_s=0,boundary_wait_s=0),ocr_backend=Backend())
    frames=read_jsonl(tmp_path/'capture/frames.jsonl');decisions=read_jsonl(tmp_path/'capture/alignments.jsonl')
    assert result['status']=='boundary_reached' and result['rejected']==0,result
    assert result['final_scroll_offset_px']==source.position-initial==(-2337 if direction=='up' else 2337)
    assert [f['action']['events'] for f in frames[1:5]]==[1,1,1,2]
    assert frames[3]['action']['scroll_guard']=='media_dominated' and 'scroll_guard' not in frames[4]['action']
    assert max(abs(d['alignment']['delta_px']) for d in decisions)<=663*.75
    assert max(f['action'].get('events',0) for f in frames)<=4
    rows=read_jsonl(tmp_path/'capture/messages.jsonl')
    assert [r['text'] for r in rows if r['type']=='text']==[text for _,text in sorted(source.texts)]
    assert source.closed and result['image_memory_remaining_bytes']==0
    assert all(v<0 if direction=='up' else v>0 for v in source.inputs)
    assert not list(tmp_path.rglob('*.png'))


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
