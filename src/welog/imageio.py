"""Unicode-safe, lossless image IO (including Chinese Windows paths)."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    result = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if result is None:
        raise ValueError(f"Cannot decode image: {path}")
    return result


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(encoded.tobytes())
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("Expected a grayscale or BGR image")
