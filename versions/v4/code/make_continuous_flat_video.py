#!/usr/bin/env python3
"""Render a complete torque-level flat-walking video with tracking evidence.

The video is deliberately generated from ``run_trial`` rather than from the
root-assisted demonstration.  Its right panel shows the moving lateral
reference and the controlled error relative to the zero-error line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from run_uneven_ground_benchmark import PUBLICATION_GAIT, RESULTS, run_trial


def _font(size: int, bold: bool = False):
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _polyline(draw: ImageDraw.ImageDraw, xy: np.ndarray, color, width: int = 2) -> None:
    if len(xy) >= 2:
        draw.line([tuple(v) for v in xy], fill=color, width=width, joint="curve")


def _map_curve(t: np.ndarray, value: np.ndarray, box, t_end: float,
               y_limit: float) -> np.ndarray:
    x0, y0, x1, y1 = box
    x = x0 + (x1 - x0) * np.clip(t / max(t_end, 1e-9), 0.0, 1.0)
    y = 0.5 * (y0 + y1) - 0.5 * (y1 - y0) * np.clip(value / y_limit, -1.0, 1.0)
    return np.c_[x, y]


def render(controller: str, seed: int, duration: float, output: Path) -> dict:
    log, summary = run_trial(
        controller,
        "flat",
        seed,
        duration=duration,
        video={"fps": 30, "width": 640, "height": 480,
               "distance": 2.7, "azimuth": 135.0, "elevation": -10.0},
    )
    if summary["fell"] or summary["duration_completed_s"] < duration - 0.01:
        raise RuntimeError(f"continuous-video gate failed: {summary}")

    frames = log["frames"]
    fps = int(log["video_fps"])
    t = np.asarray(log["t"], float)
    y = 1000.0 * np.asarray(log["com"], float)[:, 1]
    y_ref = 1000.0 * np.asarray(log["com_ref"], float)[:, 1]
    e_y = 1000.0 * np.asarray(log["task_error"], float)[:, 1]
    t_end = float(t[-1])
    active = t >= 1.0
    tail = t >= max(1.0, t_end - 1.0)
    lat_rms = float(np.sqrt(np.mean(e_y[active] ** 2)))
    lat_peak = float(np.max(np.abs(e_y[active])))
    lat_mean = float(np.mean(e_y[active]))
    lat_tail_rms = float(np.sqrt(np.mean(e_y[tail] ** 2)))
    lat_tail_mean = float(np.mean(e_y[tail]))
    controller_label = {
        "nominal_mpc": "Nominal MPC",
        "interaction_mpc": "ID-MPC",
    }[controller]

    panel_w = 500
    title_font = _font(20, bold=True)
    text_font = _font(15)
    small_font = _font(13)
    out_frames = []
    ref_box = (54, 105, panel_w - 25, 270)
    err_box = (54, 325, panel_w - 25, 445)
    ref_limit = max(125.0, 1.05 * float(np.max(np.abs(np.r_[y, y_ref]))))
    err_limit = max(15.0, 1.05 * float(np.max(np.abs(e_y))))
    ref_curve = _map_curve(t, y_ref, ref_box, t_end, ref_limit)
    actual_curve = _map_curve(t, y, ref_box, t_end, ref_limit)
    err_curve = _map_curve(t, e_y, err_box, t_end, err_limit)

    for fi, frame in enumerate(frames):
        robot = Image.fromarray(frame)
        panel = Image.new("RGB", (panel_w, robot.height), "white")
        draw = ImageDraw.Draw(panel)
        now = t_end * fi / max(len(frames) - 1, 1)
        k = int(np.clip(np.searchsorted(t, now), 1, len(t) - 1))

        draw.text((24, 18), "Continuous flat walking", font=title_font, fill="#111111")
        draw.text((24, 48), f"Torque-level {controller_label} | no root assist",
                  font=text_font, fill="#333333")
        travel_now = float(log["com"][k, 0] - log["com"][0, 0])
        draw.text((24, 72),
                  f"t = {now:4.1f} s   travel = {travel_now:.3f} m",
                  font=small_font, fill="#444444")

        for box, title, limit, unit in (
            (ref_box, "Lateral CoM and moving reference", ref_limit, "mm"),
            (err_box, "Controlled lateral error  e_y = y - y_d", err_limit, "mm"),
        ):
            x0, y0, x1, y1 = box
            draw.rectangle(box, outline="#bbbbbb", width=1)
            draw.line((x0, (y0 + y1) / 2, x1, (y0 + y1) / 2),
                      fill="#aaaaaa", width=1)
            draw.text((x0, y0 - 22), title, font=small_font, fill="#222222")
            draw.text((4, y0 - 5), f"+{limit:.0f}", font=small_font, fill="#666666")
            draw.text((15, (y0 + y1) / 2 - 7), "0", font=small_font, fill="#666666")
            draw.text((7, y1 - 14), f"-{limit:.0f}", font=small_font, fill="#666666")
            draw.text((x1 - 22, y1 + 3), unit, font=small_font, fill="#666666")

        _polyline(draw, ref_curve, "#d55e00", 2)
        _polyline(draw, actual_curve, "#0072b2", 2)
        _polyline(draw, err_curve, "#009e73", 2)
        _polyline(draw, ref_curve[:k], "#d55e00", 3)
        _polyline(draw, actual_curve[:k], "#0072b2", 3)
        _polyline(draw, err_curve[:k], "#009e73", 3)

        cursor_x = ref_box[0] + (ref_box[2] - ref_box[0]) * now / t_end
        draw.line((cursor_x, ref_box[1], cursor_x, ref_box[3]), fill="#222222", width=1)
        draw.line((cursor_x, err_box[1], cursor_x, err_box[3]), fill="#222222", width=1)
        draw.text((65, 276), "reference", font=small_font, fill="#d55e00")
        draw.text((155, 276), "actual", font=small_font, fill="#0072b2")
        draw.text((270, 276), f"current error {e_y[k]:+.1f} mm",
                  font=small_font, fill="#009e73")
        draw.text((65, 451), f"RMS {lat_rms:.2f} mm   peak {lat_peak:.2f} mm   zero line shown",
                  font=small_font, fill="#333333")
        out_frames.append(np.concatenate((np.asarray(robot), np.asarray(panel)), axis=1))

    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(output, out_frames, fps=fps, codec="libx264", quality=8)
    summary = dict(summary)
    summary["lateral_error_rms_mm"] = lat_rms
    summary["lateral_error_peak_mm"] = lat_peak
    summary["lateral_error_mean_mm"] = lat_mean
    summary["lateral_error_final_second_rms_mm"] = lat_tail_rms
    summary["lateral_error_final_second_mean_mm"] = lat_tail_mean
    summary["video"] = str(output)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="interaction_mpc",
                    choices=("nominal_mpc", "interaction_mpc"))
    ap.add_argument("--seed", type=int, default=4300)
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--output", type=Path,
                    default=RESULTS / "continuous_flat_idmpc.mp4")
    args = ap.parse_args()
    summary = render(args.controller, args.seed, args.duration, args.output)
    summary.update({
        "schema_version": 1,
        "kind": "torque_level_continuous_flat_video",
        "root_assist": False,
        "planner": PUBLICATION_GAIT,
        "video_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    })
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
