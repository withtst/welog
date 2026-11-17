from __future__ import annotations

import html
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .alignment import OverlapAligner
from .archive import ReferenceArchive, archive_frames, image_root, read_jsonl, write_json
from .imageio import read_image
from .models import AlignmentConfig, AlignmentResult, Rect
from .motion import MotionTracker
from .frame_pool import ImageBudget
from .imageio import gray


class CoordinateTracker:
    """Rejected frames never become the reference or advance the coordinate."""

    def __init__(self, aligner: OverlapAligner, origin_y_px: float = 0):
        self.aligner = aligner
        self.origin_y_px = origin_y_px
        self.reference: np.ndarray | None = None
        self.reference_id: str | None = None
        self.scroll_offset_px = 0.0
        self.reference_masks: tuple[Rect, ...] = ()
        self.layout_regions=()
        self.feature_verifier=MotionTracker(ImageBudget(),aligner.config)

    def accept(self, image: np.ndarray, frame_id: str, *, direction: str = "down",
               dynamic_masks: tuple[Rect, ...] = ()) -> tuple[AlignmentResult, str | None]:
        previous_id = self.reference_id
        if self.reference is None:
            result = AlignmentResult(True, "origin", delta_px=0.0, overlap_ratio=1.0)
        else:
            masks = self.aligner.config.masks + self.reference_masks + dynamic_masks
            aligner = OverlapAligner(replace(self.aligner.config, masks=masks)) if masks else self.aligner
            result = aligner.estimate(self.reference, image, direction=direction)
            if not result.accepted and self.reference.shape==image.shape:
                old_valid=OverlapAligner(replace(self.aligner.config,masks=self.aligner.config.masks+self.reference_masks))._valid_mask(image.shape[:2])
                new_valid=OverlapAligner(replace(self.aligner.config,masks=self.aligner.config.masks+dynamic_masks))._valid_mask(image.shape[:2])
                feature=self.feature_verifier.features(gray(self.reference),gray(image),aligner._valid_mask(image.shape[:2]),aligner,direction,old_valid,new_valid)
                if feature.accepted:result=feature
                else:
                    from .edge_layout import estimate_edge_layout,layout_media_regions,without_media
                    edge=estimate_edge_layout(gray(self.reference),gray(image),without_media(old_valid,self.layout_regions),
                        without_media(new_valid,layout_media_regions(image)),aligner,direction)
                    if edge.accepted:result=edge
                    else:result.edge_rejection_reason=edge.reason
        if result.accepted:
            from .edge_layout import layout_media_regions
            self.scroll_offset_px += result.delta_px or 0.0
            self.reference = image.copy()
            self.reference_id = frame_id
            self.reference_masks = dynamic_masks
            self.layout_regions=layout_media_regions(image)
        return result, previous_id

    def global_y(self, viewport_y: float) -> float:
        return self.scroll_offset_px + viewport_y - self.origin_y_px


def run_replay(
    source: str | Path, output: str | Path, *, config: AlignmentConfig | None = None,
    direction: str = "auto", allow_failures: bool = False,
) -> dict[str, Any]:
    source, output = Path(source).resolve(), Path(output).resolve()
    metadata, frames = archive_frames(source)
    images = image_root(source,metadata)
    if not frames:
        raise ValueError("Archive contains no frames")
    normal_direction = normal_replay_direction(metadata, direction)
    decisions = {item["frame_id"]: item for item in read_jsonl(source / "alignments.jsonl")}
    configuration = config or AlignmentConfig.from_dict(metadata.get("alignment_config", {}))
    tracker = CoordinateTracker(OverlapAligner(configuration), metadata.get("origin_y_px", 0.0))
    archive = ReferenceArchive(output, {
        "kind": "replay", "source": str(source), "origin_y_px": tracker.origin_y_px,
        "alignment_config": configuration.to_dict(), "direction": direction,
        "normal_frame_direction": normal_direction,
        "direction_policy": "capture_direction_with_bidirectional_recovery",
        "image_root":str(images),"image_storage":"references_only",
    })
    errors: list[float] = []
    steps: list[dict[str, Any]] = []
    truth_origin = None
    origin_initialized = False
    stopped = False
    for frame in frames:
        image = read_image(images / frame["image"])
        action = dict(frame.get("action") or {})
        exclusion = source_exclusion(metadata, frame, decisions)
        frame_direction = "both" if action.get("kind") == "recovery" else normal_direction
        search_direction = None if exclusion else frame_direction
        if exclusion:
            # Persist the exclusion when this replay archive is replayed again.
            action.update(coordinate_eligible=False, exclusion_reason=exclusion)
        captured = archive.capture_reference(frame,action)
        start = time.perf_counter()
        if exclusion:
            result = AlignmentResult(False, "source_unconfirmed")
            reference = tracker.reference_id
        else:
            masks = tuple(Rect(**box) for box in action.get("stability", {}).get("dynamic_masks", [])+decisions.get(frame['frame_id'],{}).get('motion_overlay_masks',[]))
            result, reference = tracker.accept(image, captured["frame_id"], direction=frame_direction, dynamic_masks=masks)
        elapsed = (time.perf_counter() - start) * 1000
        record = archive.evaluate(captured, result, reference_id=reference, cumulative_px=tracker.scroll_offset_px)
        truth = frame.get("true_scroll_px")
        error = None
        if result.accepted and not origin_initialized:
            truth_origin, origin_initialized = truth, True
        if result.accepted and truth is not None and truth_origin is not None:
            error = tracker.scroll_offset_px - (truth - truth_origin)
            errors.append(error)
        steps.append({**record, "elapsed_ms": elapsed, "error_px": error, "image": (images / captured["image"]).as_uri(),
                      "source_frame_id": frame["frame_id"], "excluded": bool(exclusion), "exclusion_reason": exclusion,
                      "search_direction": search_direction})
        if not result.accepted and not exclusion and not allow_failures:
            stopped = True
            break
    excluded = sum(step["excluded"] for step in steps)
    failed = sum(not step["alignment"]["accepted"] and not step["excluded"] for step in steps)
    confirmed = len(steps) - failed - excluded
    final_offset = tracker.scroll_offset_px if tracker.reference_id is not None else None
    source_offset = metadata.get("final_scroll_offset_px")
    durations = [step["elapsed_ms"] for step in steps if not step["excluded"] and step["alignment"]["reason"] != "origin"]
    summary = {
        "frames_total": len(frames), "frames_processed": len(steps),
        "normal_frame_direction": normal_direction,
        "direction_policy": "capture_direction_with_bidirectional_recovery",
        "accepted": confirmed, "rejected": failed, "excluded": excluded,
        "complete": not stopped and len(steps) == len(frames),
        "status": "needs_attention" if failed or excluded or metadata.get("status") in {"needs_attention", "running"} else "complete",
        "source_status": metadata.get("status"), "source_reason": metadata.get("reason"),
        "source_final_scroll_offset_px": source_offset,
        "source_checkpoint_difference_px": final_offset - source_offset if final_offset is not None and source_offset is not None else None,
        "final_scroll_offset_px": final_offset,
        "ground_truth_available": bool(errors),
        "max_abs_cumulative_error_px": float(np.max(np.abs(errors))) if errors else None,
        "p95_abs_cumulative_error_px": float(np.percentile(np.abs(errors), 95)) if errors else None,
        "final_accepted_error_px": errors[-1] if errors else None,
        "alignment_p95_ms": float(np.percentile(durations, 95)) if durations else None,
        "steps": steps,
    }
    write_json(output / "report.json", summary)
    write_report(output, summary, tracker.origin_y_px)
    archive.finish(summary["status"], processed=len(steps), accepted=confirmed, rejected=failed, excluded=excluded,
                   final_scroll_offset_px=final_offset)
    return summary


def normal_replay_direction(metadata: dict, requested: str = "auto") -> str:
    """Match capture constraints by default; retain a fallback for raw/legacy images."""
    if requested not in {"auto", "up", "down", "both"}:
        raise ValueError("Replay direction must be auto, up, down, or both")
    if requested != "auto":
        return requested
    # A replay persists its resolved direction so replaying it cannot lose the policy.
    for candidate in (metadata.get("capture_config", {}).get("direction"),
                      metadata.get("normal_frame_direction"), metadata.get("direction")):
        if candidate in {"up", "down", "both"}:
            return candidate
    return "both"


def source_exclusion(metadata: dict, frame: dict, decisions: dict) -> str | None:
    action = frame.get("action") or {}
    if action.get("coordinate_eligible") is False:
        return action.get("exclusion_reason") or "previously_unconfirmed"
    if action.get("kind") == "unstable_diagnostic" or action.get("stability", {}).get("stable") is False:
        return "stability_timeout"
    if metadata.get("kind") in {"live","live_debug"}:
        decision = decisions.get(frame["frame_id"])
        if decision is None:
            return "capture_decision_missing"
        if not decision["alignment"]["accepted"]:
            return decision["alignment"]["reason"]
    return None


def write_report(output: Path, report: dict[str, Any], origin_y_px: float) -> None:
    rows = []
    for step in report["steps"]:
        result = step["alignment"]
        offset = step["scroll_offset_px"]
        global_top = None if offset is None else offset - origin_y_px
        eligibility = "未确认，排除" if step["excluded"] else ("计入" if result["accepted"] else "对齐拒绝")
        rows.append(
            "<tr>" + "".join(f"<td>{html.escape(str(value)) if value is not None else '—'}</td>" for value in [
                step["frame_id"], eligibility, step.get("search_direction"), step["exclusion_reason"] or result["reason"], result["delta_px"], offset,
                global_top, step["error_px"], result["inliers"], round(step["elapsed_ms"], 1),
            ]) + f'<td><a href="{html.escape(step["image"], quote=True)}">PNG</a></td></tr>'
        )
    safe_data = json.dumps(report["steps"], ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    source_labels = {"needs_attention": "未完成，需检查", "running": "尚在运行", "complete": "已处理",
                     "stopped": "用户停止", "limit_reached": "达到截图上限", "visible_end_candidate": "疑似可见边界"}
    source_value = report.get("source_status")
    source_status = html.escape(source_labels.get(source_value, str(source_value or "未提供")))
    source_reason = html.escape(str(report.get("source_reason") or "无"))
    offset_text = f"{report['final_scroll_offset_px']:g} px" if report['final_scroll_offset_px'] is not None else "尚无确认坐标"
    truth_text = "有" if report['ground_truth_available'] else "无"
    error_text = f"{report['max_abs_cumulative_error_px']:g} px" if report['max_abs_cumulative_error_px'] is not None else "无真实标注，未计算"
    normal_direction_text = html.escape(str(report.get("normal_frame_direction") or "未记录"))
    content = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>WeLog 位移回放</title><style>
body{{font:15px system-ui;max-width:1200px;margin:32px auto;color:#17212c;background:#f6f8fa}}
table{{border-collapse:collapse;background:white;width:100%}}td,th{{padding:8px;border:1px solid #dce2e8;text-align:left}}
img{{max-width:100%;border:1px solid #ccc}}input{{width:100%}}.viewer{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
svg{{width:100%;height:160px;background:white}}pre{{white-space:pre-wrap}}
</style><h1>WeLog 位移回放</h1>
<p>有效 {report['accepted']} 帧，对齐拒绝 {report['rejected']} 帧，诊断/未确认排除 {report['excluded']} 帧。确认累计位移：{offset_text}。</p>
<p>源采集状态：{source_status}；结束原因：{source_reason}。回放处理完已保存帧不表示历史采集完成。</p>
<p>方向策略：普通帧 {normal_direction_text}，恢复帧 both（双向）；排除帧不参与搜索。</p>
<p>真实位移标注：{truth_text}；最大累计误差：{error_text}。
真实采集无标注时，只能检查一致性，不能据此宣称绝对准确。</p>
<svg id="plot" viewBox="0 0 1000 160"></svg>
<input id="slider" type="range" min="0" max="{len(report['steps']) - 1}" value="0">
<div class="viewer"><img id="frame" alt="采集截图"><pre id="details"></pre></div>
<p>坐标换算：global_y = scroll_offset_px + viewport_y - origin_y_px。</p>
<table><thead><tr><th>帧</th><th>坐标资格</th><th>搜索方向</th><th>状态</th><th>位移</th><th>累计位移</th><th>视口全局顶部</th><th>累计误差</th><th>匹配数</th><th>毫秒</th><th>证据</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<script>const steps={safe_data};const origin={origin_y_px};
function show(){{const s=steps[Number(document.getElementById('slider').value)];document.getElementById('frame').src=s.image;
document.getElementById('details').textContent=JSON.stringify({{frame:s.frame_id,reference:s.reference_id,search_direction:s.search_direction,excluded:s.excluded,exclusion_reason:s.exclusion_reason,scroll:s.scroll_offset_px,origin_y_px:origin,error:s.error_px,alignment:s.alignment}},null,2)}}
document.getElementById('slider').oninput=show;show();
const values=steps.filter(s=>s.scroll_offset_px!==null);const max=Math.max(1,...values.map(s=>s.scroll_offset_px));const min=Math.min(0,...values.map(s=>s.scroll_offset_px));
const points=steps.map((s,i)=>s.scroll_offset_px===null?null:[10+i*980/Math.max(1,steps.length-1),150-(s.scroll_offset_px-min)*140/(max-min)]).filter(Boolean);
const line=document.createElementNS('http://www.w3.org/2000/svg','polyline');line.setAttribute('points',points.map(p=>p.join(',')).join(' '));line.setAttribute('fill','none');line.setAttribute('stroke','#1868db');line.setAttribute('stroke-width','2');document.getElementById('plot').appendChild(line);
</script></html>"""
    (output / "report.html").write_text(content, encoding="utf-8")
