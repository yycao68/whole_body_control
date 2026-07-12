#!/usr/bin/env python3
"""Compose one annotated video per Unitree demo scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio

from compose_demo_video import ROOT, scene_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "demo_manifest.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--suffix", default="_with_curves")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for scene in manifest["scenes"]:
        src = Path(scene["clip"])
        out = args.out_dir / f"{src.stem}{args.suffix}.mp4"
        writer = imageio.get_writer(
            str(out),
            fps=int(manifest.get("fps", 30)),
            codec="libx264",
            quality=8,
        )
        try:
            for frame in scene_frames(scene, manifest):
                writer.append_data(frame)
        finally:
            writer.close()
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
