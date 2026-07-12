#!/usr/bin/env python3
"""Compose a demo storyboard video for Unitree-locomotion integration.

The videos listed in the manifest are generated locally by running Unitree's
open-source G1 MuJoCo policy through `run_unitree_rl_gym_g1.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import cv2
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def font(size: int):
    for p in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def fit_image(img: np.ndarray, width: int, height: int) -> np.ndarray:
    pil = Image.fromarray(img).convert("RGB")
    pil.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (14, 18, 28))
    x = (width - pil.width) // 2
    y = (height - pil.height) // 2
    canvas.paste(pil, (x, y))
    return np.asarray(canvas)


def read_image(path: Path, width: int, height: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return fit_image(np.asarray(img), width, height)


def iter_video(path: Path, width: int, height: int, fps: int, duration_s: float) -> Iterable[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    max_frames = int(duration_s * fps)
    for i in range(max_frames):
        cap.set(cv2.CAP_PROP_POS_MSEC, 1000.0 * i / fps)
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield fit_image(frame, width, height)
    cap.release()


def iter_static(path: Path, width: int, height: int, fps: int, duration_s: float) -> Iterable[np.ndarray]:
    frame = read_image(path, width, height)
    for _ in range(int(duration_s * fps)):
        yield frame


def load_log(scene: dict) -> dict[str, np.ndarray] | None:
    path = scene.get("log")
    if not path:
        return None
    log_path = resolve_path(path)
    if not log_path.exists():
        return None
    data = np.load(log_path)
    return {k: data[k] for k in data.files}


def interp_log(log: dict[str, np.ndarray] | None, key: str, t: float, default: float = 0.0) -> float:
    if not log or key not in log or "time" not in log:
        return default
    return float(np.interp(t, log["time"], log[key]))


def draw_curve_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], log: dict[str, np.ndarray] | None, t: float) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], fill=(12, 17, 29, 245))
    draw.text((x0 + 18, y0 + 18), "Live Signals", font=font(22), fill=(255, 255, 255, 255))

    if not log:
        draw.text((x0 + 18, y0 + 62), "No log found", font=font(15), fill=(248, 113, 113, 255))
        return

    plots = [
        ("lateral error y [m]", "y_error", (-0.75, 0.75), (96, 165, 250, 255)),
        ("correction vy [m/s]", "vy_correction", (-0.5, 0.5), (52, 211, 153, 255)),
        ("external push Fy [N]", "push_y", (-10.0, 70.0), (251, 191, 36, 255)),
    ]
    t_arr = log["time"]
    t_end = max(float(t_arr[-1]), 1e-6)
    panel_w = x1 - x0
    plot_h = 142
    gap = 34
    px0 = x0 + 18
    px1 = x1 - 18

    for idx, (label, key, yrange, color) in enumerate(plots):
        py0 = y0 + 62 + idx * (plot_h + gap)
        py1 = py0 + plot_h
        draw.rectangle([px0, py0, px1, py1], outline=(59, 72, 94, 255), width=1, fill=(7, 11, 20, 255))
        draw.line([px0, (py0 + py1) / 2, px1, (py0 + py1) / 2], fill=(42, 52, 68, 255), width=1)
        draw.text((px0, py0 - 22), label, font=font(14), fill=(226, 232, 240, 255))
        v_now = interp_log(log, key, t)
        draw.text((px1 - 82, py0 - 22), f"{v_now:+.2f}", font=font(14), fill=color)

        vals = log[key]
        ymin, ymax = yrange
        denom = max(ymax - ymin, 1e-6)
        pts = []
        stride = max(1, len(t_arr) // 450)
        for tt, vv in zip(t_arr[::stride], vals[::stride]):
            x = px0 + (px1 - px0) * min(max(float(tt) / t_end, 0.0), 1.0)
            y = py1 - (py1 - py0) * min(max((float(vv) - ymin) / denom, 0.0), 1.0)
            pts.append((x, y))
        if len(pts) > 1:
            draw.line(pts, fill=color, width=2)

        cursor_x = px0 + (px1 - px0) * min(max(t / t_end, 0.0), 1.0)
        draw.line([cursor_x, py0, cursor_x, py1], fill=(255, 255, 255, 170), width=1)

    draw.text(
        (x0 + 18, y1 - 82),
        "What is happening:",
        font=font(16),
        fill=(255, 255, 255, 255),
    )
    notes = [
        "Unitree policy generates gait.",
        "Correction changes high-level vy.",
    ]
    yy = y1 - 56
    for note in notes:
        draw.text((x0 + 18, yy), "- " + note, font=font(13), fill=(203, 213, 225, 255))
        yy += 19


def panel_frame(scene: dict, media: np.ndarray, manifest: dict, log: dict[str, np.ndarray] | None, frame_idx: int) -> np.ndarray:
    width = int(manifest.get("width", 1280))
    height = int(manifest.get("height", 720))
    curve_w = 320
    side_w = 360
    canvas = Image.new("RGB", (width, height), (10, 14, 24))
    media_w = width - curve_w - side_w
    canvas.paste(Image.fromarray(fit_image(media, media_w, height - 72)), (curve_w, 72))

    draw = ImageDraw.Draw(canvas, "RGBA")
    x0 = width - side_w
    draw.rectangle([x0, 0, width, height], fill=(14, 18, 30, 245))
    draw.rectangle([0, 0, width, 72], fill=(5, 8, 14, 235))

    title_font = font(25)
    scene_font = font(22)
    body_font = font(17)
    small_font = font(14)
    fps = int(manifest.get("fps", 30))
    t = frame_idx / max(fps, 1)
    draw_curve_panel(draw, (0, 72, curve_w, height), log, t)

    draw.text((24, 18), manifest["title"], font=title_font, fill=(245, 248, 255, 255))
    draw.text((x0 + 22, 26), scene["id"], font=title_font, fill=(96, 165, 250, 255))
    yy_title = 72
    for line in wrap(scene["title"], 31):
        draw.text((x0 + 22, yy_title), line, font=scene_font, fill=(255, 255, 255, 255))
        yy_title += 27

    y = max(128, yy_title + 24)
    draw.text((x0 + 22, y), "Claim boundary", font=body_font, fill=(251, 191, 36, 255))
    y += 30
    for line in wrap(scene["claim"], 38):
        draw.text((x0 + 22, y), line, font=small_font, fill=(226, 232, 240, 255))
        y += 22

    y += 16
    draw.text((x0 + 22, y), "Demo metrics", font=body_font, fill=(52, 211, 153, 255))
    y += 30
    for item in scene.get("metrics", []):
        for j, line in enumerate(wrap(item, 36)):
            prefix = "- " if j == 0 else "  "
            draw.text((x0 + 22, y), prefix + line, font=small_font, fill=(218, 226, 236, 255))
            y += 21

    footer = "Unitree gait; D1/D2 compare correction."
    draw.text((x0 + 22, height - 42), footer, font=small_font, fill=(148, 163, 184, 255))
    return np.asarray(canvas)


def wrap(text: str, n: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        if sum(len(x) for x in cur) + len(cur) + len(w) > n and cur:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines


def scene_frames(
    scene: dict,
    manifest: dict,
) -> Iterable[np.ndarray]:
    fps = int(manifest.get("fps", 30))
    width = int(manifest.get("width", 1280))
    height = int(manifest.get("height", 720))
    curve_w = 320
    side_w = 360
    media_w = width - curve_w - side_w
    duration_s = float(scene.get("duration_s", 5.0))
    log = load_log(scene)

    path = resolve_path(scene["clip"])
    if not path.exists():
        raise FileNotFoundError(
            f"Missing generated video {path}. Run `mjpython scripts/generate_all_open_source_videos.py`."
        )

    if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        base_iter = iter_video(path, media_w, height, fps, duration_s)
    else:
        base_iter = iter_static(path, media_w, height, fps, duration_s)

    for i, media in enumerate(base_iter):
        yield panel_frame(scene, media, manifest, log, i)


def generated_videos_missing(manifest: dict) -> list[Path]:
    missing = []
    for scene in manifest["scenes"]:
        p = resolve_path(scene["clip"])
        if not p.exists():
            missing.append(p)
    return missing


def title_frames(manifest: dict) -> Iterable[np.ndarray]:
    fps = int(manifest.get("fps", 30))
    width = int(manifest.get("width", 1280))
    height = int(manifest.get("height", 720))
    img = Image.new("RGB", (width, height), (8, 12, 22))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, 0, width, height], fill=(8, 12, 22, 255))
    draw.text((70, 180), manifest["title"], font=font(42), fill=(255, 255, 255, 255))
    draw.text((72, 250), manifest["subtitle"], font=font(23), fill=(203, 213, 225, 255))
    draw.text((72, 335), "Demo-only positioning", font=font(26), fill=(96, 165, 250, 255))
    bullets = [
        "D0: Unitree policy walking baseline.",
        "D1/D2: same lateral push, correction off versus on.",
        "D3: planned lateral load with preview correction.",
        "The gait is Unitree RL Gym; this is not a v3 dynamic-walking proof.",
    ]
    y = 386
    for b in bullets:
        draw.text((96, y), "- " + b, font=font(21), fill=(226, 232, 240, 255))
        y += 36
    frame = np.asarray(img)
    for _ in range(2 * fps):
        yield frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "demo_manifest.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "unitree_locomotion_demo.mp4")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    missing = generated_videos_missing(manifest)
    if missing:
        print("Cannot compose demo because these generated videos are missing:")
        for path in missing:
            print(f"  - {path}")
        print("\nGenerate them with:")
        print("  mjpython scripts/generate_all_open_source_videos.py")
        sys.exit(2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args.out), fps=int(manifest.get("fps", 30)), codec="libx264", quality=8)
    try:
        for frame in title_frames(manifest):
            writer.append_data(frame)
        for scene in manifest["scenes"]:
            for frame in scene_frames(scene, manifest):
                writer.append_data(frame)
    finally:
        writer.close()
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
