r"""Opt-in integration check against our OWN synthetic Tk window, never WeChat.

Run from the repository: .venv\Scripts\python.exe tools/native_smoke.py --output runs/native
This temporarily displays a synthetic window and moves the pointer to scroll it.
"""

from __future__ import annotations

import argparse
import base64
import json
import multiprocessing as mp
import os
import time
from pathlib import Path


def fixture(title, offset, ready, stop, request_focus):
    import tkinter as tk
    import cv2
    from welog.demo import make_scene

    scene = make_scene(6000, 640, seed=121)
    root = tk.Tk()
    root.title(title)
    root.geometry("640x720+80+80")
    root.resizable(False, False)
    canvas = tk.Canvas(root, width=640, height=720, borderwidth=0, highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    def draw():
        _, encoded = cv2.imencode(".png", scene[offset.value:offset.value + 720])
        canvas.photo = tk.PhotoImage(data=base64.b64encode(encoded.tobytes()).decode("ascii"))
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=canvas.photo)

    def wheel(event):
        offset.value = max(0, min(5280, offset.value - event.delta))
        draw()

    def check_stop():
        if stop.is_set():
            root.destroy()
        else:
            if request_focus.is_set():
                request_focus.clear()
                root.lift()
                root.focus_force()
            root.after(50, check_stop)

    canvas.bind("<MouseWheel>", wheel)
    draw()
    root.update()
    ready.set()
    root.after(50, check_stop)
    root.mainloop()


def main():
    from welog.archive import write_json
    from welog.controller import run_capture
    from welog.models import CaptureConfig, Rect
    from welog.recognition import EmptyOcr
    from welog.windows import DXcamSource, WindowsDesktop

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new output directory")
    desktop = WindowsDesktop()
    title = f"WeLog synthetic native capture test {os.getpid()}"
    offset, ready, stop = mp.Value("i", 0), mp.Event(), mp.Event()
    request_focus = mp.Event()
    process = mp.Process(target=fixture, args=(title, offset, ready, stop, request_focus))
    process.start()
    observations = []
    try:
        if not ready.wait(15):
            raise RuntimeError("Synthetic test window did not become ready")
        hwnd = desktop.resolve(None, title)
        desktop.activate(hwnd)
        time.sleep(0.7)
        source = DXcamSource(desktop, hwnd, Rect(0, 0, 640, 720))
        request_focus.set()
        time.sleep(0.5)

        def observe(record):
            actual = offset.value
            estimate = record["scroll_offset_px"]
            observations.append({**record, "true_scroll_px": actual,
                                 "error_px": None if estimate is None else estimate - actual})

        result = run_capture(source, str(args.output / "capture"),
                             config=CaptureConfig(max_frames=10, settle_delay_s=0.45),
                             metadata={"fixture": title}, progress=observe,ocr_backend=EmptyOcr())
        errors = [abs(item["error_px"]) for item in observations if item["error_px"] is not None]
        summary = {
            "capture": result, "native_observations": observations,
            "max_abs_cumulative_error_px": max(errors) if errors else None,
            "image_storage": "none",
        }
        write_json(args.output / "native-smoke.json", summary)
        print(json.dumps(summary, ensure_ascii=True, indent=2))
        return 0 if result["accepted"] == 10 and result["rejected"] == 0 and errors and max(errors) == 0 else 2
    finally:
        stop.set()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()  # Only the synthetic process created above.
            process.join(timeout=3)


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
