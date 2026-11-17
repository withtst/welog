"""Offline validation only. No desktop imports, window queries, or real input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from welog.archive import archive_frames, read_jsonl, write_json
from welog.controller import run_capture, wait_for_stable
from welog.demo import make_scene
from welog.imageio import read_image
from welog.models import AlignmentConfig, CaptureConfig
from welog.replay import run_replay
from welog.recognition import EmptyOcr


def digest_files(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Existing capture archive")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new or empty output directory")
    before = digest_files(args.source)
    metadata, frames = archive_frames(args.source)
    replay = run_replay(args.source, args.output / "replay", allow_failures=True)
    assert replay["final_scroll_offset_px"] == metadata["final_scroll_offset_px"]
    assert replay["source_checkpoint_difference_px"] == 0

    # Existing screenshot background, with explicitly simulated local animation.
    # The original timeout frame remains excluded; this does not retroactively approve it.
    base = read_image(args.source / frames[-1]["image"])
    height, width = base.shape[:2]
    rng = np.random.default_rng(113)

    class AnimatedFrames:
        def read(self):
            sample = base.copy()
            x, y = int(width * 0.78), 0
            w, h = int(width * 0.18), min(120, height // 5)
            sample[y:y + h, x:x + w] = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            return sample

    stability = {}
    wait_for_stable(AnimatedFrames(), CaptureConfig(sample_interval_s=0.002),
                    AlignmentConfig.from_dict(metadata["alignment_config"]), diagnostics=stability)
    assert stability["stable"] and stability["reason"] == "local_animation"

    scene = make_scene(8000, 640, seed=14)
    comparisons = []
    for max_events in (1, 8):
        class ClampedSource:
            def __init__(self):
                self.position = self.start = 6000
            def read(self):
                return scene[self.position:self.position + 720].copy()
            def scroll(self, units):
                self.position += 50 if units > 0 else -50
            def close(self):
                pass

        source = ClampedSource()
        errors = []
        output = args.output / f"scroll-events-{max_events}"
        result = run_capture(source, str(output), config=CaptureConfig(
            direction="up", max_frames=12, sample_interval_s=0.002, stable_samples=3,
            settle_delay_s=0, max_scroll_events=max_events, scroll_event_interval_s=0),
            ocr_backend=EmptyOcr(),progress=lambda item: errors.append(item["scroll_offset_px"] - (source.position - source.start))
            if item["accepted"] else None)
        assert result["rejected"] == 0 and errors and max(abs(error) for error in errors) == 0
        comparisons.append({"max_events": max_events, "frames": result["accepted"],
                            "covered_px": abs(result["final_scroll_offset_px"]),
                            "max_abs_coordinate_error_px": max(abs(error) for error in errors),
                            "largest_batch": max(frame["action"].get("events", 0) for frame in read_jsonl(output / "frames.jsonl"))})
    assert before == digest_files(args.source)
    summary = {
        "desktop_access": False, "original_archive_unchanged": True,
        "source": str(args.source.resolve()),
        "replay": {key: value for key, value in replay.items() if key != "steps"},
        "simulated_animation_on_saved_background": stability,
        "scroll_comparison": comparisons,
        "coverage_gain_per_same_frame_count": comparisons[1]["covered_px"] / comparisons[0]["covered_px"],
    }
    write_json(args.output / "validation.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
