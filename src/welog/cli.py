from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .alignment import OverlapAligner
from .archive import archive_frames, write_json
from .controller import run_capture
from .demo import generate_demo
from .imageio import read_image
from .models import AlignmentConfig, CaptureConfig, Rect
from .replay import run_replay


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


def _rect(value: str) -> Rect:
    try:
        parts = [int(part.strip()) for part in value.split(",")]
        if len(parts) != 4:
            raise ValueError
        return Rect(*parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use x,y,width,height in native client pixels") from error


def _alignment(args: argparse.Namespace, base: dict[str, Any] | None = None) -> AlignmentConfig:
    config = AlignmentConfig.from_dict(base or {})
    if getattr(args, "mask", None):
        data = config.to_dict()
        data["masks"] = [mask.to_dict() for mask in args.mask]
        config = AlignmentConfig.from_dict(data)
    return config


def _compact(report: dict[str, Any], output: Path) -> dict[str, Any]:
    return {**{key: value for key, value in report.items() if key != "steps"},
            "report": str(output.resolve() / "report.html")}


def _target_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hwnd", type=lambda value: int(value, 0), help="Window handle, decimal or 0x-prefixed")
    parser.add_argument("--title", help="Unique title substring")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="welog", description="WeLog: capture and verify overlap-corrected scroll coordinates")
    commands = parser.add_subparsers(dest="command", required=True)
    gui = commands.add_parser("gui", help="Point at WeChat to select a window and start the capture UI")
    gui.add_argument("--output-root", type=Path)
    gui.add_argument("--no-auto-pick", action="store_true")
    gui.add_argument("--dwell", type=float, default=1.5)
    gui.add_argument("--countdown", type=int, default=3)
    gui.add_argument("--max-frames", type=int, default=0, help="0: capture until boundary (default); positive: frame limit")
    commands.add_parser("windows", help="List visible Windows window handles and titles")
    setup = commands.add_parser("configure", help="Select the chat ROI and optional fixed conversation title guard")
    _target_args(setup)
    setup.add_argument("--output", type=Path, required=True)
    setup.add_argument("--with-guard", action="store_true", help="Also select a non-scrolling conversation title region")
    capture = commands.add_parser("capture", help="Capture and scroll one foreground window")
    _target_args(capture)
    capture.add_argument("--config", type=Path)
    capture.add_argument("--roi", type=_rect)
    capture.add_argument("--guard-roi", type=_rect)
    capture.add_argument("--mask", type=_rect, action="append", help="Ignore viewport-relative dynamic regions")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--max-frames", type=int, default=0, help="0: capture until boundary (default); positive: frame limit")
    capture.add_argument('--debug-save-images',action='store_true',help='Retain raw ROI screenshots for offline debugging')
    capture.add_argument('--debug-max-mb',type=int,default=1024,help='Debug image quota in MiB; stop when reached')
    capture.add_argument("--wheel-units", type=int, default=120, help="Units per individual wheel event (default 120)")
    capture.add_argument("--direction", choices=["up", "down"], default="down")
    capture.add_argument("--origin-y", type=float, default=0, help="First message top relative to viewport; default origin is viewport top")
    capture.add_argument("--countdown", type=float, default=4)
    capture.add_argument("--activate", action="store_true", help="Ask Windows to activate the selected target")
    replay = commands.add_parser("replay", help="Recompute offsets from saved raw frames, without desktop access")
    replay.add_argument("source", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--mask", type=_rect, action="append")
    replay.add_argument("--direction", choices=["auto", "down", "up", "both"], default="auto",
                        help="Auto uses capture direction for normal frames; recovery always searches both ways")
    replay.add_argument("--continue-on-failure", action="store_true", help="Diagnostic only: keep the last trusted reference after a rejected frame")
    replay.add_argument("--max-error", type=float, default=2)
    demo = commands.add_parser("demo", help="Generate a known-offset chat fixture and run offline verification")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--frames", type=int, default=80)
    demo.add_argument("--width", type=int, default=640)
    demo.add_argument("--height", type=int, default=720)
    demo.add_argument("--step", type=int, default=280)
    demo.add_argument("--seed", type=int, default=42)
    pair = commands.add_parser("align", help="Inspect the displacement between two cropped screenshots")
    pair.add_argument("previous", type=Path)
    pair.add_argument("current", type=Path)
    pair.add_argument("--direction", choices=["down", "up", "both"], default="both")
    pair.add_argument("--mask", type=_rect, action="append")
    coordinate = commands.add_parser("coordinate", help="Convert a saved frame's viewport Y into cumulative global Y")
    coordinate.add_argument("archive", type=Path)
    coordinate.add_argument("--frame", required=True)
    coordinate.add_argument("--y", required=True, type=float)
    return parser


def configure(args: argparse.Namespace) -> int:
    from PySide6.QtWidgets import QApplication,QDialog
    from .gui import RegionDialog,configure_font
    from .windows import DXcamSource, WindowsDesktop

    desktop = WindowsDesktop()
    hwnd = desktop.resolve(args.hwnd, args.title)
    desktop.activate(hwnd)
    time.sleep(0.5)
    client = desktop.client_rect(hwnd)
    source = DXcamSource(desktop, hwnd, Rect(0, 0, client.width, client.height))
    try:
        deadline = time.monotonic() + 3
        image = None
        while image is None and time.monotonic() < deadline:
            image = source.read()
            time.sleep(0.1)
        if image is None:
            raise ValueError("No desktop frame available")
        app=QApplication.instance() or QApplication(sys.argv[:1])
        configure_font(app)
        dialog=RegionDialog(desktop.describe_window(hwnd),image,start_after_calibration=False,with_guard=args.with_guard)
        if dialog.exec()!=QDialog.DialogCode.Accepted or dialog.roi is None:
            return 0
        roi,guard=dialog.roi,dialog.guard
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, {
            "title": desktop.title(hwnd), "roi": roi.to_dict(),
            "guard_roi": guard.to_dict() if guard else None,
            "client_size": [client.width, client.height],
            "alignment_config": AlignmentConfig().to_dict(),
        })
        _print({"config": str(args.output.resolve()), "roi": roi.to_dict()})
        return 0
    finally:
        source.close()


def capture(args: argparse.Namespace) -> int:
    from .windows import DXcamSource, WindowsDesktop

    saved = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    roi = args.roi or (Rect(**saved["roi"]) if "roi" in saved else None)
    guard = args.guard_roi or (Rect(**saved["guard_roi"]) if saved.get("guard_roi") else None)
    if roi is None:
        raise ValueError("Specify --roi or create a configuration with 'welog configure'")
    if args.countdown < 0:
        raise ValueError("countdown cannot be negative")
    config = CaptureConfig(max_frames=args.max_frames, initial_wheel_units=args.wheel_units,
                           origin_y_px=args.origin_y, direction=args.direction,
                           debug_save_images=args.debug_save_images,debug_max_bytes=args.debug_max_mb*1024*1024)
    alignment = _alignment(args, saved.get("alignment_config"))
    desktop = WindowsDesktop()
    hwnd = desktop.resolve(args.hwnd, args.title or saved.get("title"))
    if args.activate:
        desktop.activate(hwnd)
    print(f"Focus the selected chat. Starting in {args.countdown:g}s. Esc or Ctrl+C stops capture.", flush=True)
    time.sleep(args.countdown)
    client = desktop.client_rect(hwnd)
    if saved.get("client_size") and saved["client_size"] != [client.width, client.height]:
        raise ValueError("Window size differs from calibration; run configure again")
    source = DXcamSource(desktop, hwnd, roi, guard)
    report = run_capture(source, str(args.output), config=config, alignment=alignment,
                         metadata={"hwnd": hwnd, "window_title": desktop.title(hwnd), "roi": roi.to_dict(),
                                   "guard_roi": guard.to_dict() if guard else None, "client_rect": client.to_dict()},
                         progress=_print)
    _print(report)
    return 2 if report["status"] == "needs_attention" else 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "gui":
            try:
                from .gui import launch_gui
            except ImportError as error:
                raise RuntimeError('Install the UI with: pip install -e ".[windows,gui]"') from error
            return launch_gui(args.output_root, auto_pick=not args.no_auto_pick, dwell=args.dwell,
                              countdown=args.countdown, max_frames=args.max_frames)
        if args.command == "windows":
            from .windows import WindowsDesktop
            _print(WindowsDesktop().windows())
            return 0
        if args.command == "configure":
            return configure(args)
        if args.command == "capture":
            return capture(args)
        if args.command == "align":
            result = OverlapAligner(_alignment(args)).estimate(
                read_image(args.previous), read_image(args.current), direction=args.direction)
            _print(result.to_dict())
            return 0 if result.accepted else 2
        if args.command == "demo":
            if args.output.exists() and any(args.output.iterdir()):
                raise ValueError("Demo output must be a new or empty directory")
            generate_demo(args.output / "input", frames=args.frames, width=args.width,
                          height=args.height, step=args.step, seed=args.seed)
            report = run_replay(args.output / "input", args.output / "replay")
            _print(_compact(report, args.output / "replay"))
            return 0 if report["rejected"] == 0 and report["max_abs_cumulative_error_px"] <= 2 else 2
        if args.command == "replay":
            if args.max_error < 0:
                raise ValueError("max-error cannot be negative")
            metadata, _ = archive_frames(args.source)
            report = run_replay(args.source, args.output, config=_alignment(args, metadata.get("alignment_config")),
                                direction=args.direction, allow_failures=args.continue_on_failure)
            _print(_compact(report, args.output))
            error = report["max_abs_cumulative_error_px"]
            return 2 if report["status"] == "needs_attention" or (error is not None and error > args.max_error) else 0
        if args.command == "coordinate":
            from .archive import read_jsonl
            metadata=json.loads((args.archive/'manifest.json').read_text(encoding='utf-8'))
            frames=read_jsonl(args.archive/'frames.jsonl') if metadata.get('storage_mode')=='memory_only' else archive_frames(args.archive)[1]
            frame = next((frame for frame in frames if frame["frame_id"] == args.frame), None)
            if frame is None:
                raise ValueError("Unknown frame ID")
            if not 0 <= args.y < frame["height"]:
                raise ValueError("Y is outside the frame")
            record = next((item for item in reversed(read_jsonl(args.archive / "alignments.jsonl"))
                           if item["frame_id"] == args.frame and item["alignment"]["accepted"]), None)
            if record is None:
                raise ValueError("Frame has no accepted coordinate")
            origin = metadata.get("origin_y_px", 0)
            _print({"frame": args.frame, "viewport_y": args.y, "scroll_offset_px": record["scroll_offset_px"],
                    "origin_y_px": origin, "global_y": record["scroll_offset_px"] + args.y - origin})
            return 0
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError) as error:
        print(f"welog: {error}", file=sys.stderr)
        return 2
    return 0
