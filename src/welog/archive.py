"""Append-only raw capture journal plus separately committed alignment results."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .imageio import write_image
from .models import AlignmentResult


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    values = []
    for index, line in enumerate(lines):
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            # A killed writer can leave a truncated tail, never a trusted record.
            if index != len(lines) - 1:
                raise
    return values


def _append(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class Archive:
    def __init__(self, path: str | Path, metadata: dict[str, Any]):
        self.path = Path(path).resolve()
        if self.path.exists() and any(self.path.iterdir()):
            raise ValueError(f"Output must be a new or empty directory: {self.path}")
        (self.path / "frames").mkdir(parents=True, exist_ok=True)
        self.metadata = {
            "format_version": 1, "created_at": utc_now(), "status": "running",
            "coordinate_convention": "global_y = scroll_offset_px + viewport_y - origin_y_px",
            **metadata,
        }
        self.counter = 0
        write_json(self.path / "manifest.json", self.metadata)

    def capture(
        self, image: np.ndarray, *, action: dict[str, Any] | None = None,
        truth_px: float | None = None, source_name: str | None = None,
    ) -> dict[str, Any]:
        frame_id = f"{self.counter:06d}"
        self.counter += 1
        relative = f"frames/{frame_id}.png"
        target = self.path / relative
        write_image(target, image)
        record = {
            "frame_id": frame_id, "image": relative, "captured_at": utc_now(),
            "width": image.shape[1], "height": image.shape[0],
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "action": action or {}, "true_scroll_px": truth_px, "source_name": source_name,
        }
        _append(self.path / "frames.jsonl", record)
        return record

    def evaluate(
        self, frame: dict[str, Any], result: AlignmentResult,
        *, reference_id: str | None, cumulative_px: float | None,
    ) -> dict[str, Any]:
        record = {
            "frame_id": frame["frame_id"], "reference_id": reference_id,
            "scroll_offset_px": cumulative_px if result.accepted else None,
            "alignment": result.to_dict(), "evaluated_at": utc_now(),
        }
        _append(self.path / "alignments.jsonl", record)
        if result.accepted:
            write_json(self.path / "checkpoint.json", {
                **record, "image": frame["image"],
                "origin_y_px": self.metadata.get("origin_y_px", 0.0),
            })
        return record

    def finish(self, status: str, **extra: Any) -> None:
        self.metadata.update(status=status, finished_at=utc_now(), **extra)
        write_json(self.path / "manifest.json", self.metadata)


class ReferenceArchive(Archive):
    """Legacy offline replay references existing evidence; it does not copy image files."""
    def capture_reference(self, frame: dict, action: dict) -> dict:
        record = {**frame,"frame_id":f"{self.counter:06d}","source_name":frame["frame_id"],"action":action}
        self.counter += 1
        _append(self.path / "frames.jsonl", record)
        return record


def image_root(root: Path, metadata: dict) -> Path:
    if metadata.get("kind") == "replay" and metadata.get("image_root"):
        return Path(metadata["image_root"]).resolve()
    return root.resolve()


def archive_frames(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(path).resolve()
    metadata = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if metadata.get('storage_mode')=='event_images':
        raise ValueError('Event-only capture has sparse screenshots; inspect diagnostics.jsonl and retained_frames.jsonl instead of full replay')
    if metadata.get("storage_mode") == "memory_only" or metadata.get("kind") == "live_memory":
        raise ValueError("Memory-only session has no screenshots; open its text report instead")
    frames = read_jsonl(root / "frames.jsonl")
    images = image_root(root,metadata)
    for frame in frames:
        if not frame.get('image'):
            raise ValueError(f"Frame has no retained image: {frame['frame_id']}; check image_save_error in its action")
        target = (images / frame["image"]).resolve()
        if not target.is_relative_to(images):
            raise ValueError("Frame path escapes the archive")
        if hashlib.sha256(target.read_bytes()).hexdigest() != frame["sha256"]:
            raise ValueError(f"Frame checksum mismatch: {frame['frame_id']}")
    return metadata, frames
