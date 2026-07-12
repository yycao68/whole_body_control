#!/usr/bin/env python3
"""Generate all Unitree open-source locomotion demo videos."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
import subprocess
import sys

from run_unitree_rl_gym_g1 import DEFAULT_CONFIG, run


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


SCENES = [
    (
        "baseline",
        "baseline",
        ROOT / "results" / "unitree_base_only.mp4",
        ROOT / "results" / "unitree_base_only_summary.json",
        ROOT / "results" / "unitree_base_only_log.npz",
        ["--vx", "0.5"],
    ),
    (
        "baseline_idmpc",
        "baseline",
        ROOT / "results" / "unitree_base_idmpc.mp4",
        ROOT / "results" / "unitree_base_idmpc_summary.json",
        ROOT / "results" / "unitree_base_idmpc_log.npz",
        ["--vx", "0.5", "--push-y", "0", "--layer", "on"],
    ),
    (
        "push_off",
        "push_off",
        ROOT / "results" / "unitree_push_layer_off.mp4",
        ROOT / "results" / "unitree_push_layer_off_summary.json",
        ROOT / "results" / "unitree_push_layer_off_log.npz",
        ["--vx", "0.5", "--push-y", "40"],
    ),
    (
        "push_on",
        "push_on",
        ROOT / "results" / "unitree_push_layer_on.mp4",
        ROOT / "results" / "unitree_push_layer_on_summary.json",
        ROOT / "results" / "unitree_push_layer_on_log.npz",
        ["--vx", "0.5", "--push-y", "40"],
    ),
    (
        "load_no_preview",
        "load_preview",
        ROOT / "results" / "unitree_load_no_preview.mp4",
        ROOT / "results" / "unitree_load_no_preview_summary.json",
        ROOT / "results" / "unitree_load_no_preview_log.npz",
        ["--vx", "0.5", "--push-y", "60", "--layer", "on", "--preview", "off"],
    ),
    (
        "load_preview",
        "load_preview",
        ROOT / "results" / "unitree_load_preview.mp4",
        ROOT / "results" / "unitree_load_preview_summary.json",
        ROOT / "results" / "unitree_load_preview_log.npz",
        ["--vx", "0.5", "--push-y", "60"],
    ),
]


FINAL_VIDEOS = {
    ROOT / "results" / "unitree_d0_baseline_comparison.mp4",
    ROOT / "results" / "unitree_d1_d2_push_comparison.mp4",
    ROOT / "results" / "unitree_d3_preview_comparison.mp4",
}


def remove_intermediate_videos() -> None:
    for path in (ROOT / "results").glob("*.mp4"):
        if path not in FINAL_VIDEOS:
            path.unlink()
            print(f"removed intermediate video: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--skip-compose", action="store_true")
    ap.add_argument("--keep-intermediate-videos", action="store_true")
    args = ap.parse_args()

    summaries = {}
    for name, scenario, video, summary, log, extra in SCENES:
        option_pairs = dict(zip(extra[0::2], extra[1::2]))
        run_args = Namespace(
            config=DEFAULT_CONFIG,
            duration=float(args.duration),
            scenario=scenario,
            vx=float(option_pairs.get("--vx", 0.5)),
            vy=None,
            yaw_rate=None,
            video=video,
            summary=summary,
            log=log,
            fps=30,
            width=1280,
            height=720,
            camera="",
            camera_distance=4.0,
            camera_azimuth=135.0,
            camera_elevation=-18.0,
            camera_x_offset=0.25,
            camera_y_offset=0.0,
            camera_z_lookat=0.8,
            fall_height=0.45,
            fall_tilt=0.75,
            push_body="pelvis",
            push_start=3.0,
            push_duration=0.35,
            push_y=float(option_pairs["--push-y"]) if "--push-y" in option_pairs else None,
            layer=option_pairs.get("--layer", "auto"),
            preview=option_pairs.get("--preview", "auto"),
            preview_lead=0.35,
            preview_gain=0.003,
            y_kp=1.6,
            y_kd=0.45,
            yaw_kp=1.4,
            max_vy_correction=0.45,
            reference_line=True,
        )
        print(f"running: {name} ({scenario}) -> {video}")
        summaries[name] = run(run_args)

    aggregate = ROOT / "results" / "unitree_open_source_demo_summary.json"
    aggregate.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"saved: {aggregate}")

    if not args.skip_compose:
        cmd = [
            sys.executable,
            str(HERE / "compose_d0_baseline_comparison.py"),
            "--out",
            str(ROOT / "results" / "unitree_d0_baseline_comparison.mp4"),
        ]
        print("running:", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)

        cmd = [
            sys.executable,
            str(HERE / "compose_d1_d2_comparison.py"),
            "--out",
            str(ROOT / "results" / "unitree_d1_d2_push_comparison.mp4"),
        ]
        print("running:", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)

        if not args.keep_intermediate_videos:
            remove_intermediate_videos()

        cmd = [
            sys.executable,
            str(HERE / "compose_d3_preview_comparison.py"),
            "--out",
            str(ROOT / "results" / "unitree_d3_preview_comparison.mp4"),
        ]
        print("running:", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
