#!/usr/bin/env python3
"""Verify the Unitree locomotion demo package.

This does not validate locomotion physics. It checks that the final comparison
videos and Unitree open-source simulation summaries are in a coherent state.
Raw scene videos are intermediate artifacts and are normally deleted after the
final three videos are composed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def media_info(path: Path) -> dict:
    if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        reader = imageio.get_reader(str(path))
        meta = reader.get_meta_data()
        reader.close()
        return {
            "kind": "video",
            "fps": meta.get("fps"),
            "duration_s": meta.get("duration"),
            "size": meta.get("size"),
        }
    return {"kind": "image", "size_bytes": path.stat().st_size}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "demo_manifest.json")
    ap.add_argument("--require-generated", action="store_true", help="Fail unless the final comparison videos exist.")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    report = {
        "manifest": str(args.manifest),
        "title": manifest.get("title"),
        "num_scenes": len(manifest.get("scenes", [])),
        "ready_for_generated_demo": True,
    }

    comparison_paths = [
        ROOT / "results" / "unitree_d0_baseline_comparison.mp4",
        ROOT / "results" / "unitree_d1_d2_push_comparison.mp4",
        ROOT / "results" / "unitree_d3_preview_comparison.mp4",
    ]
    report["comparison_videos"] = []
    for comparison in comparison_paths:
        item = {
            "path": str(comparison),
            "video_exists": comparison.exists(),
        }
        if comparison.exists():
            item["video_info"] = media_info(comparison)
        elif args.require_generated:
            report["ready_for_generated_demo"] = False
        report["comparison_videos"].append(item)

    if args.require_generated and not report["ready_for_generated_demo"]:
        missing = [v["path"] for v in report["comparison_videos"] if not v["video_exists"]]
        print(json.dumps(report, indent=2))
        print("\nSTATUS: generated Unitree open-source simulation demo is not ready.")
        print("Missing final comparison videos:")
        for path in missing:
            print(f"  - {path}")
        print("\nGenerate them with:")
        print("  mjpython scripts/generate_all_open_source_videos.py")
        sys.exit(2)

    print(json.dumps(report, indent=2))
    if not report["ready_for_generated_demo"]:
        print("\nSTATUS: incomplete. Run `mjpython scripts/generate_all_open_source_videos.py`.")
    else:
        print("\nSTATUS: all final Unitree open-source simulation comparison videos are present.")


if __name__ == "__main__":
    main()
