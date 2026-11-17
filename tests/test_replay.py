import json
from pathlib import Path

import numpy as np
import pytest

from welog.archive import Archive, archive_frames, read_jsonl
from welog.cli import main
from welog.demo import generate_demo, make_scene
from welog.replay import normal_replay_direction, run_replay
from welog.models import AlignmentResult


@pytest.mark.parametrize("seed", [9, 42])
def test_end_to_end_long_sequence_and_coordinate_cli(tmp_path, capsys, seed):
    source, output = tmp_path / "中文采集", tmp_path / "回放"
    generate_demo(source, frames=100, seed=seed)
    report = run_replay(source, output)
    assert report["rejected"] == 0
    assert report["frames_processed"] == 100
    assert report["max_abs_cumulative_error_px"] == 0
    assert report["final_scroll_offset_px"] > 25_000
    assert (output / "report.html").exists()
    assert main(["coordinate", str(output), "--frame", "000099", "--y", "42"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["global_y"] == report["final_scroll_offset_px"] + 42


def test_failure_stops_default_replay_and_preserves_checkpoint(tmp_path):
    source = tmp_path / "input"
    archive = Archive(source, {})
    scene = make_scene(1800, 640)
    archive.capture(scene[:720], truth_px=0)
    archive.capture(scene[200:920], truth_px=200)
    archive.capture(np.full((720, 640, 3), 255, np.uint8), truth_px=400)
    archive.capture(scene[400:1120], truth_px=400)
    archive.finish("complete")
    report = run_replay(source, tmp_path / "strict")
    assert report["rejected"] == 1
    assert report["frames_processed"] == 3
    assert not report["complete"]
    checkpoint = json.loads((tmp_path / "strict/checkpoint.json").read_text())
    assert checkpoint["scroll_offset_px"] == 200
    report = run_replay(source, tmp_path / "diagnostic", allow_failures=True)
    assert report["complete"]
    assert report["rejected"] == 1
    assert report["final_scroll_offset_px"] == 400
    assert report["steps"][3]["reference_id"] == "000001"


def test_missing_ground_truth_is_not_reported_as_zero_error(tmp_path):
    archive = Archive(tmp_path / "input", {})
    scene = make_scene(1200, 640)
    archive.capture(scene[:720])
    archive.capture(scene[200:920])
    archive.finish("complete")
    report = run_replay(archive.path, tmp_path / "output")
    assert report["accepted"] == 2
    assert not report["ground_truth_available"]
    assert report["max_abs_cumulative_error_px"] is None


def test_evidence_integrity_and_no_overwrite(tmp_path):
    archive = Archive(tmp_path / "input", {})
    frame = archive.capture(make_scene(720, 640))
    with pytest.raises(ValueError, match="empty directory"):
        Archive(archive.path, {})
    path = archive.path / frame["image"]
    path.write_bytes(path.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="checksum"):
        archive_frames(archive.path)


def test_truncated_tail_is_not_trusted(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_text('{"frame_id":"a"}\n{"frame_id":', encoding="utf-8")
    assert read_jsonl(path) == [{"frame_id": "a"}]
    path.write_text('{broken}\n{"frame_id":"b"}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)


def test_unconfirmed_diagnostic_frame_never_advances_replay_or_checkpoint(tmp_path):
    source = tmp_path / "live"
    archive = Archive(source, {"kind": "live"})
    scene = make_scene(1800, 640)
    for index, offset in enumerate((0, 200)):
        frame = archive.capture(scene[offset:offset + 720], action={"kind": "initial" if index == 0 else "scroll"})
        archive.evaluate(frame, AlignmentResult(True, "origin" if index == 0 else "aligned", delta_px=offset),
                         reference_id=None if index == 0 else "000000", cumulative_px=offset)
    frame = archive.capture(scene[250:970], action={"kind": "unstable_diagnostic"})
    archive.evaluate(frame, AlignmentResult(False, "stability_timeout"), reference_id="000001", cumulative_px=None)
    archive.finish("needs_attention", reason="stability_timeout", final_scroll_offset_px=200)
    output = tmp_path / "replay"
    report = run_replay(source, output, allow_failures=True)
    assert report["accepted"] == 2
    assert report["excluded"] == 1
    assert report["rejected"] == 0
    assert report["final_scroll_offset_px"] == 200
    assert report["source_checkpoint_difference_px"] == 0
    assert report["status"] == "needs_attention"
    assert report["steps"][-1]["excluded"]
    assert report["steps"][-1]["scroll_offset_px"] is None
    assert json.loads((output / "checkpoint.json").read_text())["frame_id"] == "000001"
    second = run_replay(output, tmp_path / "second-replay", allow_failures=True)
    assert second["excluded"] == 1
    assert second["final_scroll_offset_px"] == 200


@pytest.mark.parametrize("kind", ["unstable_diagnostic", "initial"])
def test_diagnostic_or_uncommitted_first_frame_cannot_become_origin(tmp_path, kind):
    archive = Archive(tmp_path / "live", {"kind": "live"})
    archive.capture(make_scene(720, 640), action={"kind": kind})
    archive.finish("needs_attention")
    output = tmp_path / "replay"
    report = run_replay(archive.path, output)
    assert report["accepted"] == 0
    assert report["excluded"] == 1
    assert report["final_scroll_offset_px"] is None
    assert not (output / "checkpoint.json").exists()
    assert "尚无确认坐标" in (output / "report.html").read_text(encoding="utf-8")


@pytest.mark.parametrize("direction,offsets", [
    ("up", [600, 400, 450, 200]),
    ("down", [200, 400, 350, 600]),
])
def test_capture_direction_and_opposite_recovery_survive_repeated_replay(tmp_path, direction, offsets):
    source = Archive(tmp_path / "source", {"kind": "synthetic", "capture_config": {"direction": direction}})
    scene = make_scene(1800, 640, seed=29)
    for offset, kind in zip(offsets, ["initial", "scroll", "recovery", "scroll"]):
        source.capture(scene[offset:offset + 720], action={"kind": kind}, truth_px=offset)
    source.finish("complete")
    output = tmp_path / "replay"
    for input_path, output_path in [(source.path, output), (output, tmp_path / "second-replay")]:
        report = run_replay(input_path, output_path)
        assert report["accepted"] == 4
        assert report["rejected"] == report["excluded"] == 0
        assert report["normal_frame_direction"] == direction
        assert [step["search_direction"] for step in report["steps"]] == [direction, direction, "both", direction]
        assert report["final_scroll_offset_px"] == offsets[-1] - offsets[0]
        assert report["max_abs_cumulative_error_px"] == 0
        # Recovery moves against the main traversal direction; one-way search cannot accept it.
        assert report["steps"][2]["alignment"]["delta_px"] == offsets[2] - offsets[1]


@pytest.mark.parametrize("metadata,requested,expected", [
    ({}, "auto", "both"),
    ({"capture_config": {"direction": "up"}}, "auto", "up"),
    ({"normal_frame_direction": "down", "direction": "auto"}, "auto", "down"),
    ({"direction": "up"}, "auto", "up"),
    ({"capture_config": {"direction": "up"}}, "down", "down"),
    ({"capture_config": {"direction": "up"}}, "both", "both"),
])
def test_direction_resolution_and_legacy_fallback(metadata, requested, expected):
    assert normal_replay_direction(metadata, requested) == expected


def test_invalid_replay_direction_cannot_create_output(tmp_path):
    source = Archive(tmp_path / "source", {})
    source.capture(make_scene(720, 640))
    source.finish("complete")
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="Replay direction"):
        run_replay(source.path, output, direction="sideways")
    assert not output.exists()


LOCAL_DIRECTION_REGRESSION = Path(__file__).resolve().parents[1] / "runs/pointer-20261116-202220-4b6bc2/capture"


@pytest.mark.skipif(not (LOCAL_DIRECTION_REGRESSION / "manifest.json").exists(),
                    reason="Private local capture regression archive is not distributed")
def test_local_capture_frame_62_does_not_break_the_replay_chain(tmp_path):
    source_records = {row["frame_id"]: row for row in read_jsonl(LOCAL_DIRECTION_REGRESSION / "alignments.jsonl")}
    report = run_replay(LOCAL_DIRECTION_REGRESSION, tmp_path / "replay")
    assert report["frames_processed"] == 100
    assert (report["accepted"], report["excluded"], report["rejected"]) == (87, 13, 0)
    assert report["final_scroll_offset_px"] == -22681
    assert report["source_checkpoint_difference_px"] == 0
    critical = report["steps"][62]
    assert critical["search_direction"] == "up"
    assert critical["alignment"]["delta_px"] == -300
    assert critical["reference_id"] == "000061"
    for step in report["steps"]:
        original = source_records[step["source_frame_id"]]
        if original["alignment"]["accepted"]:
            assert step["alignment"]["accepted"]
            assert step["scroll_offset_px"] == original["scroll_offset_px"]
        else:
            assert step["excluded"] and step["scroll_offset_px"] is None
