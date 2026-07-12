#!/usr/bin/env python3
"""Compose a vertical D0 baseline comparison video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from compose_demo_video import ROOT, fit_image, font
from compose_d1_d2_comparison import draw_series, load_log, read_frame


def compose(args: argparse.Namespace) -> None:
    d0_video = ROOT / "results" / "unitree_base_only.mp4"
    d0_mpc_video = ROOT / "results" / "unitree_base_idmpc.mp4"
    d0_log = load_log(ROOT / "results" / "unitree_base_only_log.npz")
    d0_mpc_log = load_log(ROOT / "results" / "unitree_base_idmpc_log.npz")

    cap0 = cv2.VideoCapture(str(d0_video))
    cap1 = cv2.VideoCapture(str(d0_mpc_video))
    if not cap0.isOpened() or not cap1.isOpened():
        raise RuntimeError("Missing D0 baseline videos. Generate unitree_base_idmpc.mp4 first.")

    fps = int(args.fps)
    width, height = 1280, 720
    duration = min(
        float(cap0.get(cv2.CAP_PROP_FRAME_COUNT) / (cap0.get(cv2.CAP_PROP_FPS) or fps)),
        float(cap1.get(cv2.CAP_PROP_FRAME_COUNT) / (cap1.get(cv2.CAP_PROP_FPS) or fps)),
        float(d0_log["time"][-1]),
        float(d0_mpc_log["time"][-1]),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args.out), fps=fps, codec="libx264", quality=8)
    try:
        for i in range(int(duration * fps)):
            t = i / fps
            canvas = Image.new("RGB", (width, height), (8, 12, 22))
            draw = ImageDraw.Draw(canvas, "RGBA")
            draw.rectangle([0, 0, width, 76], fill=(5, 8, 14, 255))
            draw.text((26, 18), "D0 Baseline Comparison: Unitree Only vs Interaction Dynamics MPC", font=font(25), fill=(250, 252, 255, 255))
            draw.text((26, 51), "No external push. This checks whether Interaction Dynamics MPC reduces nominal lateral drift.", font=font(14), fill=(191, 205, 224, 255))

            logs = [
                ("Unitree", d0_log, "Unitree only", (96, 165, 250, 255)),
                ("ID-MPC", d0_mpc_log, "Interaction Dynamics MPC", (52, 211, 153, 255)),
            ]
            curve_x0, curve_x1 = 22, 356
            draw.rectangle([0, 76, curve_x1 + 14, height], fill=(10, 15, 26, 235))
            draw.text((curve_x0, 94), "Live comparison", font=font(22), fill=(255, 255, 255, 255))
            draw_series(draw, (curve_x0, 134, curve_x1, 286), logs, "y_error", t, (-0.45, 0.15), "Lateral error [m]")
            draw_series(draw, (curve_x0, 314, curve_x1, 466), logs, "vy_correction", t, (-0.15, 0.15), "Correction vy [m/s]")
            draw_series(draw, (curve_x0, 494, curve_x1, 646), logs, "push_y", t, (-1.0, 1.0), "External push Fy [N]")
            draw.text((curve_x0, 670), f"t = {t:4.1f} s", font=font(16), fill=(148, 163, 184, 255))

            video_x = 388
            video_w, video_h = 866, 292
            top_y, bottom_y = 92, 402
            d0_frame = read_frame(cap0, t, video_w, video_h)
            d0_mpc_frame = read_frame(cap1, t, video_w, video_h)
            canvas.paste(Image.fromarray(d0_frame), (video_x, top_y))
            canvas.paste(Image.fromarray(d0_mpc_frame), (video_x, bottom_y))
            draw.rectangle([video_x, top_y, video_x + video_w, top_y + video_h], outline=(96, 165, 250, 255), width=3)
            draw.rectangle([video_x, bottom_y, video_x + video_w, bottom_y + video_h], outline=(52, 211, 153, 255), width=3)
            draw.rectangle([video_x, top_y, video_x + video_w, top_y + 36], fill=(10, 20, 42, 175))
            draw.rectangle([video_x, bottom_y, video_x + video_w, bottom_y + 36], fill=(8, 27, 22, 175))
            draw.text((video_x + 14, top_y + 8), "D0-A: Unitree policy only, no correction", font=font(19), fill=(215, 232, 255, 255))
            draw.text((video_x + 14, bottom_y + 8), "D0-B: same walking with Interaction Dynamics MPC correction", font=font(19), fill=(210, 255, 235, 255))
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
        cap0.release()
        cap1.release()
    print(f"saved: {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "unitree_d0_baseline_comparison.mp4")
    ap.add_argument("--fps", type=int, default=30)
    compose(ap.parse_args())


if __name__ == "__main__":
    main()
