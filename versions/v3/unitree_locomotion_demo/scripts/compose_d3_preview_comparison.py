#!/usr/bin/env python3
"""Compose a vertical D3 planned-load preview comparison video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from compose_demo_video import ROOT, font
from compose_d1_d2_comparison import draw_push_panel, draw_series, load_log, read_frame


def compose(args: argparse.Namespace) -> None:
    no_preview_video = ROOT / "results" / "unitree_load_no_preview.mp4"
    preview_video = ROOT / "results" / "unitree_load_preview.mp4"
    no_preview_log = load_log(ROOT / "results" / "unitree_load_no_preview_log.npz")
    preview_log = load_log(ROOT / "results" / "unitree_load_preview_log.npz")

    cap0 = cv2.VideoCapture(str(no_preview_video))
    cap1 = cv2.VideoCapture(str(preview_video))
    if not cap0.isOpened() or not cap1.isOpened():
        raise RuntimeError("Missing D3 preview comparison videos. Generate unitree_load_no_preview.mp4 first.")

    fps = int(args.fps)
    width, height = 1280, 720
    duration = min(
        float(cap0.get(cv2.CAP_PROP_FRAME_COUNT) / (cap0.get(cv2.CAP_PROP_FPS) or fps)),
        float(cap1.get(cv2.CAP_PROP_FRAME_COUNT) / (cap1.get(cv2.CAP_PROP_FPS) or fps)),
        float(no_preview_log["time"][-1]),
        float(preview_log["time"][-1]),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args.out), fps=fps, codec="libx264", quality=8)
    try:
        for i in range(int(duration * fps)):
            t = i / fps
            canvas = Image.new("RGB", (width, height), (8, 12, 22))
            draw = ImageDraw.Draw(canvas, "RGBA")
            draw.rectangle([0, 0, width, 76], fill=(5, 8, 14, 255))
            draw.text((26, 18), "D3 Preview Comparison: Reactive MPC vs Preview MPC", font=font(25), fill=(250, 252, 255, 255))
            draw.text((26, 51), "Same planned lateral load. Preview starts correction before the load arrives.", font=font(14), fill=(191, 205, 224, 255))

            logs = [
                ("No preview", no_preview_log, "Reactive MPC", (248, 113, 113, 255)),
                ("Preview", preview_log, "Preview MPC", (52, 211, 153, 255)),
            ]
            curve_x0, curve_x1 = 22, 356
            draw.rectangle([0, 76, curve_x1 + 14, height], fill=(10, 15, 26, 235))
            draw.text((curve_x0, 94), "Live comparison", font=font(22), fill=(255, 255, 255, 255))
            draw_series(draw, (curve_x0, 134, curve_x1, 286), logs, "y_error", t, (-0.75, 0.75), "Lateral error [m]")
            draw_series(draw, (curve_x0, 314, curve_x1, 466), logs, "vy_correction", t, (-0.55, 0.15), "Correction vy [m/s]")
            draw_push_panel(
                draw,
                (curve_x0, 494, curve_x1, 646),
                no_preview_log,
                t,
                title="Planned load input Fy [N]",
                note="same planned load applied to both runs",
            )
            draw.text((curve_x0, 670), f"t = {t:4.1f} s", font=font(16), fill=(148, 163, 184, 255))

            video_x = 388
            video_w, video_h = 866, 292
            top_y, bottom_y = 92, 402
            no_preview_frame = read_frame(cap0, t, video_w, video_h)
            preview_frame = read_frame(cap1, t, video_w, video_h)
            canvas.paste(Image.fromarray(no_preview_frame), (video_x, top_y))
            canvas.paste(Image.fromarray(preview_frame), (video_x, bottom_y))
            draw.rectangle([video_x, top_y, video_x + video_w, top_y + video_h], outline=(248, 113, 113, 255), width=3)
            draw.rectangle([video_x, bottom_y, video_x + video_w, bottom_y + video_h], outline=(52, 211, 153, 255), width=3)
            draw.rectangle([video_x, top_y, video_x + video_w, top_y + 36], fill=(30, 12, 18, 175))
            draw.rectangle([video_x, bottom_y, video_x + video_w, bottom_y + 36], fill=(8, 27, 22, 175))
            draw.text((video_x + 14, top_y + 8), "D3-A: same load, Interaction Dynamics MPC without preview", font=font(19), fill=(255, 225, 225, 255))
            draw.text((video_x + 14, bottom_y + 8), "D3-B: same load with Interaction Dynamics MPC preview", font=font(19), fill=(210, 255, 235, 255))
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
        cap0.release()
        cap1.release()
    print(f"saved: {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "unitree_d3_preview_comparison.mp4")
    ap.add_argument("--fps", type=int, default=30)
    compose(ap.parse_args())


if __name__ == "__main__":
    main()
