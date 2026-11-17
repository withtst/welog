"""Deterministic chat-like fixture; ground truth is withheld from the aligner."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from .archive import Archive
from .models import AlignmentConfig


def make_scene(height: int, width: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    canvas = np.full((height, width, 3), 242, np.uint8)
    y, index = 18, 0
    while y < height - 90:
        side = index % 3 == 0
        lines = int(rng.integers(1, 5))
        bubble_width = int(rng.integers(width // 3, max(width // 3 + 1, width - 125)))
        bubble_height = lines * 24 + 22
        x = width - bubble_width - 62 if side else 62
        avatar_x = width - 48 if side else 12
        color = (int(rng.integers(60, 190)), int(rng.integers(80, 210)), int(rng.integers(60, 200)))
        cv2.rectangle(canvas, (avatar_x, y), (avatar_x + 32, y + 32), color, -1)
        cv2.rectangle(canvas, (x, y), (x + bubble_width, min(height - 1, y + bubble_height)),
                      (135, 232, 160) if side else (255, 255, 255), -1)
        for line in range(lines):
            text = f"message {index:04d} line {line} {rng.integers(10000, 99999)}"
            cv2.putText(canvas, text, (x + 10, y + 25 + 24 * line), cv2.FONT_HERSHEY_SIMPLEX,
                        0.43, (35, 38, 42), 1, cv2.LINE_AA)
        y += bubble_height + int(rng.integers(14, 32))
        index += 1
    return canvas


def generate_demo(
    output: str | Path, *, frames: int = 80, width: int = 640,
    height: int = 720, step: int = 280, seed: int = 42,
) -> dict:
    if not 2 <= frames <= 1000 or not 160 <= width <= 1920 or not 200 <= height <= 2160:
        raise ValueError("Invalid demo dimensions or frame count")
    if not 1 <= step <= height * 0.65:
        raise ValueError("step must leave at least 35% overlap")
    rng = np.random.default_rng(seed + 1)
    positions = [0]
    for _ in range(frames - 1):
        positions.append(positions[-1] + max(1, step + int(rng.integers(-step // 5, step // 5 + 1))))
    total_height = positions[-1] + height
    if total_height * width * 3 > 400_000_000:
        raise ValueError("Demo scene exceeds the 400 MB generation limit")
    scene = make_scene(total_height, width, seed)
    archive = Archive(output, {
        "kind": "synthetic", "seed": seed, "origin_y_px": 0.0,
        "alignment_config": AlignmentConfig().to_dict(),
        "ground_truth_description": "Native pixel crops from one stationary rendered scene",
        "scene_sha256": hashlib.sha256(scene.tobytes()).hexdigest(),
    })
    for offset in positions:
        archive.capture(scene[offset:offset + height].copy(), truth_px=float(offset), action={"kind": "fixture"})
    archive.finish("complete", frame_count=len(positions))
    return {"output": str(archive.path), "frames": len(positions), "true_final_scroll_px": positions[-1]}
