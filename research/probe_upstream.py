"""Offline algorithm probes: selected AST definitions only, no GUI/capture/OCR imports."""

from __future__ import annotations

import ast
import copy
import json
import sys
import types
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
YA = BASE / "sources/yaozhwen-wechatOcr-b2cde637550e"
WC = BASE / "sources/yuanzal-Weclip-d3aa61d521b3"


def load_definitions(path: Path, names: set[str], module: types.ModuleType) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    definitions = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            node = copy.deepcopy(node)
            if isinstance(node, ast.ClassDef) and node.name == "Collector":
                node.body = [item for item in node.body if not isinstance(item, ast.FunctionDef) or item.name != "export"]
            definitions.append(node)
    assert {node.name for node in definitions} == names
    unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
                      + definitions, type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), module.__dict__)


def module(name):
    result = types.ModuleType(name)
    result.__dict__.update(np=np, cv2=cv2, dataclass=dataclass, asdict=asdict, field=field,
                           datetime=datetime, Any=Any, Rect=tuple)
    sys.modules[name] = result
    return result


def main():
    ya = module("welog_research_ya_algorithms")
    load_definitions(YA / "wechat_ocr/recognition.py", {"Message", "FrameResult", "find_bubbles", "content_regions"}, ya)
    load_definitions(YA / "wechat_ocr/tracking.py", {"Collector", "FrameGate"}, ya)
    wc = module("welog_research_wc_algorithms")
    load_definitions(WC / "ocr_core.py", {
        "_clamp", "_hsv_bounds", "_build_color_masks", "_extract_bubbles_from_mask", "merge_scrolled_frames",
        "_message_key", "_texts_equivalent", "_rows_match", "_find_overlap_size", "_find_best_overlap",
        "_merge_by_unique_keys", "_collapse_adjacent_duplicates", "_dedupe_near_repeats", "_normalize_message_text",
    }, wc)
    results = {"execution": {"desktop_access": False, "ocr_models_loaded": False,
                             "image_files_written": False, "method": "Whitelisted AST definitions; synthetic arrays and message records"}}
    bubble_cases = []
    for name, y, height in [("normal", 50, 80), ("top_clipped", 0, 80),
                            ("long_48_percent", 50, 240), ("bottom_clipped", 450, 50),
                            ("long_88_percent", 10, 440)]:
        image = np.full((500, 600, 3), 250, np.uint8)
        image[y:y + height, 70:210] = (238, 238, 240)
        for line_y in range(y + 25, min(500, y + height) - 5, 28):
            cv2.putText(image, "text line", (80, line_y), cv2.FONT_HERSHEY_SIMPLEX,
                        .35, (30, 30, 30), 1, cv2.LINE_AA)
        found = ya.find_bubbles(image)
        regions, _ = ya.content_regions(image)
        mask = np.zeros((500, 600), np.uint8)
        mask[y:y + height, 70:210] = 255
        extracted = wc._extract_bubbles_from_mask(mask, "对方", 600, 500)
        bubble_cases.append({"case": name, "ya_bubble_count": len(found),
                             "ya_complete_flags": [box[2] for box in found],
                             "ya_region_types": [box[3] for box in regions], "weclip_bubble_count": len(extracted)})
    results["bubble_cases"] = bubble_cases

    def frame(texts, complete=True):
        return ya.FrameResult("2026-11-16T00:00:00+08:00", "synthetic", (0, 0, 600, 500), (600, 500),
                              [ya.Message("对方", text, (70, 100 + i * 60, 210, 140 + i * 60), .99, complete)
                               for i, text in enumerate(texts)])

    sequences = []
    for direction, first, second in [("down", ["甲", "乙"], ["乙", "丙"]),
                                      ("up", ["乙", "丙"], ["甲", "乙"])]:
        collector = ya.Collector()
        collector.accept(frame(first))
        collector.accept(frame(second))
        sequences.append({"direction": direction, "segment_count": len(collector.segments),
                          "exported_message_count": sum(len(segment["messages"]) for segment in collector.segments),
                          "last_segment_reason": collector.segments[-1]["reason"]})
    partial = ya.Collector()
    partial.accept(frame(["上半段"], complete=False))
    partial.accept(frame(["上半段\n下半段"], complete=True))
    results["ya_collector"] = {"direction_cases": sequences,
                               "partial_to_complete_segments": len(partial.segments),
                               "partial_to_complete_records": sum(len(s["messages"]) for s in partial.segments)}

    gate = ya.FrameGate()
    gate_results = []
    for index in range(6):
        image = np.full((500, 600, 3), 250, np.uint8)
        image[30:90, 430:510] = 20 if index % 2 else 180
        gate_results.append(bool(gate.ready(image)))
    results["ya_animation_gate"] = {"ready_results": gate_results, "accepted": sum(gate_results)}

    def rows(texts):
        return [{"sender": "对方", "content": text, "confidence": .99} for text in texts]

    repeat_cases = []
    for texts in [["好", "好"], ["好", "明天见", "好"], ["下午三点开会", "下午三点开会取消"]]:
        output = wc.merge_scrolled_frames([rows(texts)])
        repeat_cases.append({"input": texts, "output": [row["content"] for row in output],
                             "input_count": len(texts), "output_count": len(output)})
    results["weclip_repeated_messages"] = repeat_cases
    gray = np.full((500, 600, 3), 250, np.uint8)
    gray[50:130, 70:210] = 240
    white, _ = wc._build_color_masks(gray)
    results["weclip_default_neutral_gray_mask_pixels"] = int(np.count_nonzero(white))
    (BASE / "probe-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
