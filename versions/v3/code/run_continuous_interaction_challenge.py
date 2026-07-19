#!/usr/bin/env python3
"""Staged 15 s G1 walking, push, and spatial-platform challenge.

The script deliberately gates the combined vignette behind independently
inspectable flat, sustained-push, and platform trials.  It reuses the same
reference provider, 100 Hz MPC, 500 Hz inverse-dynamics QP, and 1 kHz plant as
the paper benchmark; only the plant disturbance/terrain changes.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from run_uneven_ground_benchmark import (
    MPC_DT,
    RESULTS,
    SIM_DT,
    WBC_DT,
    PushSpec,
    run_trial,
)


GAIT = {
    "n_steps": 12,
    "step_length": 0.03,
    "step_time": 1.40,
    "double_support_time": 1.00,
    "settle_time": 1.00,
    "lateral_zmp_scale": 1.00,
}
DEFAULT_CONTROLLERS = ("nominal_mpc", "interaction_mpc")


def _first_sustained(mask: np.ndarray, samples: int) -> int | None:
    if samples <= 1:
        idx = np.flatnonzero(mask)
        return None if not len(idx) else int(idx[0])
    run = np.convolve(mask.astype(int), np.ones(samples, int), mode="valid")
    idx = np.flatnonzero(run == samples)
    return None if not len(idx) else int(idx[0])


def event_metrics(log: dict, *, push_start_s: float | None = None,
                  push_end_s: float | None = None,
                  terrain_height_mm: float = 0.0) -> dict:
    t = np.asarray(log["t"], float)
    alive = ~np.asarray(log["fell"], bool)
    e = np.asarray(log["task_error"], float)
    rpy = np.asarray(log["rpy"], float)
    feet = np.asarray(log["foot_position"], float)
    out = {
        "peak_abs_lateral_error_mm": float(1000.0 * np.max(np.abs(e[alive, 1]))),
        "peak_abs_roll_deg": float(np.degrees(np.max(np.abs(rpy[alive, 0])))),
    }

    if push_end_s is not None:
        response = alive & (t >= push_start_s) & (t <= push_end_s + 2.0)
        out["push_window_peak_abs_lateral_error_mm"] = float(
            1000.0 * np.max(np.abs(e[response, 1]))
        )
        out["push_window_peak_abs_roll_deg"] = float(
            np.degrees(np.max(np.abs(rpy[response, 0])))
        )
        baseline = alive & (t >= max(1.0, push_end_s - 1.8)) & (t < push_end_s - 1.1)
        score = np.sqrt((e[:, 1] / 0.02) ** 2 + (rpy[:, 0] / 0.10) ** 2)
        threshold = max(1.0, 1.5 * float(np.percentile(score[baseline], 95)))
        after = alive & (t >= push_end_s) & (score <= threshold)
        start = _first_sustained(after, round(0.25 / SIM_DT))
        out["push_recovery_threshold"] = threshold
        out["push_recovery_time_s"] = (
            None if start is None or np.any(~alive) else
            float(max(0.0, t[start] - push_end_s))
        )

    # Detect physical step-up/down from measured foot-site height, independent
    # of the nominal gait timestamps.
    z0 = float(np.median(feet[(t >= 0.2) & (t < 0.8), :, 2]))
    contact = np.asarray(log["contact"], bool)
    height_threshold = max(0.003, 0.35 * terrain_height_mm / 1000.0)
    high = np.any(contact & (feet[:, :, 2] > z0 + height_threshold), axis=1)
    up_idx = np.flatnonzero(alive & high & (t >= 4.0))
    out["measured_step_up_time_s"] = None if not len(up_idx) else float(t[up_idx[0]])
    if len(up_idx):
        down = np.flatnonzero(alive & ~high & (np.arange(len(t)) > up_idx[0]) & (t >= 7.0))
        out["measured_step_down_time_s"] = None if not len(down) else float(t[down[0]])
    else:
        out["measured_step_down_time_s"] = None
    return out


def _run(controller: str, terrain: str, seed: int, *, height_mm: float = 30.0,
         push: PushSpec | None = None, video: dict | None = None,
         platform: tuple[float, float] = (0.025, 0.065)) -> tuple[dict, dict]:
    log, summary = run_trial(
        controller, terrain, seed, duration=15.0, gait=GAIT, push=push,
        video=video, terrain_height_mm=height_mm,
        platform_start_x=platform[0], platform_end_x=platform[1],
    )
    summary["event_metrics"] = event_metrics(
        log, push_start_s=None if push is None else push.start_time_s,
        push_end_s=None if push is None else push.start_time_s + push.duration_s,
        terrain_height_mm=height_mm if terrain == "platform" else 0.0,
    )
    summary["challenge_terrain"] = {
        "height_mm": float(height_mm) if terrain == "platform" else 0.0,
        "platform_start_x_m": float(platform[0]) if terrain == "platform" else None,
        "platform_end_x_m": float(platform[1]) if terrain == "platform" else None,
    }
    return log, summary


def _platform_from_flat(log: dict) -> tuple[float, float]:
    x0 = float(np.mean(log["foot_position"][0, :, 0]))
    step = GAIT["step_length"]
    # Compact stepping block sized from the frozen step length.  Measured
    # contact, rather than these nominal indices, defines entry and exit.
    return x0 + 2.5 * step, x0 + 5.5 * step


def _write_video(logs: dict[str, dict], path: Path, labels: dict[str, str]) -> None:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    names = list(logs)
    n = min(len(logs[name]["frames"]) for name in names)
    fps = int(logs[names[0]]["video_fps"])
    frames = []
    for k in range(n):
        panels = []
        for name in names:
            im = Image.fromarray(logs[name]["frames"][k])
            draw = ImageDraw.Draw(im)
            draw.rectangle((8, 8, 310, 39), fill=(255, 255, 255))
            draw.text((16, 16), labels[name], fill=(0, 0, 0))
            panels.append(np.asarray(im))
        frames.append(np.concatenate(panels, axis=1))
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4300)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--controllers", nargs="+", default=list(DEFAULT_CONTROLLERS))
    ap.add_argument("--stages", nargs="+", choices=("flat", "push", "platform", "combined"),
                    default=["flat", "push", "platform", "combined"])
    ap.add_argument("--push-forces", nargs="+", type=float, default=[30.0, 50.0, 70.0])
    ap.add_argument("--platform-heights", nargs="+", type=float, default=[20.0, 30.0, 40.0])
    ap.add_argument("--combined-force", type=float, default=5.0)
    ap.add_argument("--combined-height", type=float, default=30.0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--artifact", default="continuous_interaction_challenge.json")
    args = ap.parse_args()

    trials = []
    flat_logs: dict[int, dict] = {}
    platform_by_seed: dict[int, tuple[float, float]] = {}
    for i in range(args.seeds):
        seed = args.seed + i
        # Always execute the flat gate once because its measured initial feet
        # define the spatial platform; only save it as a result when requested.
        gate_controller = args.controllers[0]
        log, summary = _run(gate_controller, "flat", seed)
        flat_logs[seed] = log
        platform_by_seed[seed] = _platform_from_flat(log)
        if "flat" in args.stages:
            summary["stage"] = "flat"
            trials.append(summary)
        if summary["fell"]:
            raise RuntimeError(f"flat gate failed for seed {seed}: {summary}")

        for controller in args.controllers:
            if "flat" in args.stages and controller != gate_controller:
                _, s = _run(controller, "flat", seed)
                s["stage"] = "flat"; trials.append(s)
            if "push" in args.stages:
                for force in args.push_forces:
                    push = PushSpec("lateral", "timed", force, duration_s=1.0,
                                    start_time_s=2.0, profile="flat_top", ramp_s=0.10)
                    _, s = _run(controller, "flat", seed, push=push)
                    s["stage"] = "push_sweep"; trials.append(s)
            if "platform" in args.stages:
                for height in args.platform_heights:
                    _, s = _run(controller, "platform", seed, height_mm=height,
                                platform=platform_by_seed[seed])
                    s["stage"] = "platform_sweep"; trials.append(s)
            if "combined" in args.stages:
                push = PushSpec("lateral", "timed", args.combined_force, duration_s=1.0,
                                start_time_s=2.0, profile="flat_top", ramp_s=0.10)
                _, s = _run(controller, "platform", seed, height_mm=args.combined_height,
                            push=push, platform=platform_by_seed[seed])
                s["stage"] = "combined"; trials.append(s)
            print(f"completed seed={seed} controller={controller}", flush=True)

    artifact = {
        "schema_version": 1,
        "scenario": {
            "duration_s": 15.0, "push_window_s": [2.0, 3.0],
            "nominal_step_up_s": 5.2, "nominal_step_down_s": 9.4,
            "gait": GAIT, "platform_coordinates_by_seed": {
                str(k): list(v) for k, v in platform_by_seed.items()
            },
        },
        "metadata": {
            "python": platform.python_version(), "platform": platform.platform(),
            "plant_dt_s": SIM_DT, "wbc_schedule_dt_s": WBC_DT,
            "mpc_schedule_dt_s": MPC_DT,
        },
        "trials": trials,
    }
    out = RESULTS / args.artifact
    out.write_text(json.dumps(artifact, indent=2))
    print(f"saved {out}")

    if args.video:
        seed = args.seed
        push = PushSpec("lateral", "timed", args.combined_force, duration_s=1.0,
                        start_time_s=2.0, profile="flat_top", ramp_s=0.10)
        logs = {}
        for controller in args.controllers:
            logs[controller], _ = _run(
                controller, "platform", seed, height_mm=args.combined_height,
                push=push, platform=platform_by_seed[seed],
                video={"fps": 30, "width": 640, "height": 480},
            )
        video_path = RESULTS / "continuous_interaction_challenge.mp4"
        _write_video(logs, video_path, {name: name.replace("_", " ") for name in logs})
        print(f"saved {video_path}")


if __name__ == "__main__":
    main()
