#!/usr/bin/env python3
"""Render illustrative MuJoCo videos of the Interaction-Dynamics MPC handling
three environmental interactions: an external torso push, a hole (depression),
and a stone (obstacle) in the road.

The controller, planner, realizer, and terrain are exactly those of the paper
benchmark; these runs are for illustration only and write to versions/v3/videos.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from run_uneven_ground_benchmark import run_trial, PushSpec

HERE = Path(__file__).resolve().parent
VIDEODIR = HERE.parent / "videos"
VIDEODIR.mkdir(exist_ok=True)

FPS = 30
SIZE = dict(width=640, height=480)
SEED = 4200
CONTROLLER = "interaction_mpc"

SCENARIOS = {
    # name        terrain        push spec                      camera
    "push": dict(
        terrain="flat",
        push=PushSpec(direction="lateral", phase="single_support",
                      magnitude_n=90.0, duration_s=0.15),
        cam=dict(azimuth=180.0, elevation=-8.0, distance=2.9)),
    "hole": dict(
        terrain="depression", push=None,
        cam=dict(azimuth=90.0, elevation=-6.0, distance=2.9)),
    "stone": dict(
        terrain="obstacle", push=None,
        cam=dict(azimuth=90.0, elevation=-6.0, distance=2.9)),
}


def render(name: str, spec: dict) -> None:
    video = dict(fps=FPS, **SIZE, **spec["cam"])
    log, summary = run_trial(CONTROLLER, spec["terrain"], SEED, duration=4.0,
                             push=spec["push"], video=video)
    frames = log["frames"]
    out = VIDEODIR / f"{name}.mp4"
    imageio.mimwrite(out, frames, fps=FPS, quality=8, macro_block_size=None)
    status = "FELL" if summary["fell"] else "completed"
    print(f"{name:6s}: {len(frames)} frames -> {out.name}  "
          f"[{status}, CoM peak {summary['com_xyz_peak_mm']:.1f} mm]")


if __name__ == "__main__":
    for name, spec in SCENARIOS.items():
        render(name, spec)
