#!/usr/bin/env python3
"""Compose a vertical D1/D2 push comparison video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from compose_demo_video import ROOT, fit_image, font


def load_log(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def interp(log: dict[str, np.ndarray], key: str, t: float) -> float:
    return float(np.interp(t, log["time"], log[key]))


def read_frame(cap: cv2.VideoCapture, t: float, width: int, height: int, crop_top: int = 125) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000.0 * t)
    ok, frame = cap.read()
    if not ok:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if crop_top > 0 and frame.shape[0] > crop_top + 20:
            frame = frame[crop_top:, :, :]
    return fit_image(frame, width, height)


def draw_time_axis(
    draw: ImageDraw.ImageDraw,
    px0: int,
    px1: int,
    y: int,
    t_end: float,
    color: tuple[int, int, int, int] = (148, 163, 184, 255),
) -> None:
    for tt in (0.0, 0.5 * t_end, t_end):
        x = px0 + (px1 - px0) * min(max(tt / max(t_end, 1e-6), 0.0), 1.0)
        draw.line([x, y - 4, x, y], fill=color, width=1)
        label = f"{tt:.0f}s" if tt > 0 else "0s"
        draw.text((x - 11, y + 2), label, font=font(10), fill=color)


def draw_series(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    logs: list[tuple[str, dict[str, np.ndarray], str, tuple[int, int, int, int]]],
    key: str,
    t: float,
    yrange: tuple[float, float],
    title: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], fill=(7, 11, 20, 255), outline=(55, 65, 82, 255), width=1)
    draw.text((x0 + 10, y0 + 8), title, font=font(15), fill=(230, 238, 248, 255))
    px0, px1 = x0 + 10, x1 - 10
    py0, py1 = y0 + 34, y1 - 24
    draw.line([px0, (py0 + py1) / 2, px1, (py0 + py1) / 2], fill=(44, 54, 70, 255), width=1)
    ymin, ymax = yrange
    denom = max(ymax - ymin, 1e-6)
    t_end = max(float(logs[0][1]["time"][-1]), 1e-6)
    for idx, (label, log, _, color) in enumerate(logs):
        vals = log[key]
        ts = log["time"]
        stride = max(1, len(ts) // 520)
        pts = []
        for tt, vv in zip(ts[::stride], vals[::stride]):
            x = px0 + (px1 - px0) * min(max(float(tt) / t_end, 0.0), 1.0)
            y = py1 - (py1 - py0) * min(max((float(vv) - ymin) / denom, 0.0), 1.0)
            pts.append((x, y))
        if len(pts) > 1:
            draw.line(pts, fill=color, width=2)
        now = interp(log, key, t)
        draw.text((x1 - 155, y0 + 8 + 20 * idx), f"{label}: {now:+.3f}", font=font(13), fill=color)
    cursor_x = px0 + (px1 - px0) * min(max(t / t_end, 0.0), 1.0)
    draw.line([cursor_x, py0, cursor_x, py1], fill=(255, 255, 255, 170), width=1)
    draw_time_axis(draw, px0, px1, y1 - 18, t_end)


def draw_push_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    log: dict[str, np.ndarray],
    t: float,
    title: str = "External push input Fy [N]",
    note: str = "same scripted push applied to both runs",
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], fill=(7, 11, 20, 255), outline=(55, 65, 82, 255), width=1)
    draw.text((x0 + 10, y0 + 8), title, font=font(15), fill=(251, 191, 36, 255))
    px0, px1 = x0 + 10, x1 - 10
    py0, py1 = y0 + 34, y1 - 24
    draw.line([px0, py1, px1, py1], fill=(44, 54, 70, 255), width=1)

    ts = log["time"]
    vals = log["push_y"]
    t_end = max(float(ts[-1]), 1e-6)
    ymax = max(float(np.max(vals)) * 1.15, 1.0)
    pts = []
    fill_pts = [(px0, py1)]
    stride = max(1, len(ts) // 520)
    for tt, vv in zip(ts[::stride], vals[::stride]):
        x = px0 + (px1 - px0) * min(max(float(tt) / t_end, 0.0), 1.0)
        y = py1 - (py1 - py0) * min(max(float(vv) / ymax, 0.0), 1.0)
        pts.append((x, y))
        fill_pts.append((x, y))
    fill_pts.append((pts[-1][0] if pts else px1, py1))
    if len(fill_pts) > 2:
        draw.polygon(fill_pts, fill=(251, 191, 36, 42))
    if len(pts) > 1:
        draw.line(pts, fill=(251, 191, 36, 255), width=3)

    now = interp(log, "push_y", t)
    draw.text((x1 - 98, y0 + 8), f"{now:+.1f}", font=font(13), fill=(251, 191, 36, 255))
    cursor_x = px0 + (px1 - px0) * min(max(t / t_end, 0.0), 1.0)
    draw.line([cursor_x, py0, cursor_x, py1], fill=(255, 255, 255, 170), width=1)
    draw_time_axis(draw, px0, px1, y1 - 18, t_end, color=(251, 191, 36, 255))
    draw.text((x0 + 10, y1 - 32), note, font=font(12), fill=(251, 191, 36, 255))


def compose(args: argparse.Namespace) -> None:
    d1_video = ROOT / "results" / "unitree_push_layer_off.mp4"
    d2_video = ROOT / "results" / "unitree_push_layer_on.mp4"
    d1_log = load_log(ROOT / "results" / "unitree_push_layer_off_log.npz")
    d2_log = load_log(ROOT / "results" / "unitree_push_layer_on_log.npz")

    cap1 = cv2.VideoCapture(str(d1_video))
    cap2 = cv2.VideoCapture(str(d2_video))
    if not cap1.isOpened() or not cap2.isOpened():
        raise RuntimeError("Missing D1/D2 generated videos. Run generate_all_open_source_videos.py first.")

    fps = int(args.fps)
    width, height = 1280, 720
    duration = min(
        float(cap1.get(cv2.CAP_PROP_FRAME_COUNT) / (cap1.get(cv2.CAP_PROP_FPS) or fps)),
        float(cap2.get(cv2.CAP_PROP_FRAME_COUNT) / (cap2.get(cv2.CAP_PROP_FPS) or fps)),
        float(d1_log["time"][-1]),
        float(d2_log["time"][-1]),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args.out), fps=fps, codec="libx264", quality=8)
    try:
        for i in range(int(duration * fps)):
            t = i / fps
            canvas = Image.new("RGB", (width, height), (8, 12, 22))
            draw = ImageDraw.Draw(canvas, "RGBA")
            draw.rectangle([0, 0, width, 76], fill=(5, 8, 14, 255))
            draw.text((26, 18), "D1 vs D2 Push Comparison: Unitree Only vs Interaction Dynamics MPC", font=font(25), fill=(250, 252, 255, 255))
            draw.text((26, 51), "Same Unitree G1 policy and same push. The cyan ground line is the nominal reference path.", font=font(14), fill=(191, 205, 224, 255))

            logs = [
                ("D1", d1_log, "Unitree only", (248, 113, 113, 255)),
                ("D2", d2_log, "Correction", (52, 211, 153, 255)),
            ]

            curve_x0, curve_x1 = 22, 356
            draw.rectangle([0, 76, curve_x1 + 14, height], fill=(10, 15, 26, 235))
            draw.text((curve_x0, 94), "Live comparison", font=font(22), fill=(255, 255, 255, 255))
            draw_series(draw, (curve_x0, 134, curve_x1, 286), logs, "y_error", t, (-0.45, 0.45), "Lateral error [m]")
            draw_series(draw, (curve_x0, 314, curve_x1, 466), logs, "vy_correction", t, (-0.5, 0.5), "Correction vy [m/s]")
            draw_push_panel(draw, (curve_x0, 494, curve_x1, 646), d1_log, t)
            draw.text((curve_x0, 670), f"t = {t:4.1f} s", font=font(16), fill=(148, 163, 184, 255))

            video_x = 388
            video_w, video_h = 866, 292
            d1_y, d2_y = 92, 402
            d1_frame = read_frame(cap1, t, video_w, video_h)
            d2_frame = read_frame(cap2, t, video_w, video_h)
            canvas.paste(Image.fromarray(d1_frame), (video_x, d1_y))
            canvas.paste(Image.fromarray(d2_frame), (video_x, d2_y))
            draw.rectangle([video_x, d1_y, video_x + video_w, d1_y + video_h], outline=(248, 113, 113, 255), width=3)
            draw.rectangle([video_x, d2_y, video_x + video_w, d2_y + video_h], outline=(52, 211, 153, 255), width=3)
            draw.rectangle([video_x, d1_y, video_x + video_w, d1_y + 36], fill=(30, 12, 18, 175))
            draw.rectangle([video_x, d2_y, video_x + video_w, d2_y + 36], fill=(8, 27, 22, 175))
            draw.text((video_x + 14, d1_y + 8), "D1: Unitree policy only, no correction", font=font(19), fill=(255, 225, 225, 255))
            draw.text((video_x + 14, d2_y + 8), "D2: same push with Interaction Dynamics MPC correction", font=font(19), fill=(210, 255, 235, 255))

            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
        cap1.release()
        cap2.release()
    print(f"saved: {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "unitree_d1_d2_push_comparison.mp4")
    ap.add_argument("--fps", type=int, default=30)
    compose(ap.parse_args())


if __name__ == "__main__":
    main()
