#!/usr/bin/env python3
"""Platform vignette (short physics window): step down / step up at 20/30/40 mm,
and a merged push + platform case.

This is the physics-substrate complement to the kinematic sustained-push
campaign.  Because the shared gait/WBC is stable only for ~4 s, these are short
4 s trials in which the walking foot meets a unilateral step (a depression =
step down, an obstacle = step up), optionally with a transient torso push.
Nominal MPC vs Interaction MPC, ten paired seeds, physics recovery metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_uneven_ground_benchmark import run_trial, PushSpec, RESULTS

CONTROLLERS = ("nominal_mpc", "interaction_mpc")
HEIGHTS_MM = (20.0, 30.0, 40.0)
STEPS = (("depression", "step_down"), ("obstacle", "step_up"))
SEEDS = range(4200, 4210)


def _cell(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "falls": int(sum(r["fell"] for r in rows)),
        "com_peak_mm": float(np.median([r["com_xyz_peak_mm"] for r in rows])),
        "com_rms_mm": float(np.median([r["com_xyz_rms_mm"] for r in rows])),
        "realization_residual_rms_mps2": float(np.median(
            [r["realization_residual_rms_mps2"] for r in rows])),
        "qp_fallbacks": int(sum(r["qp_fallbacks"] for r in rows)),
    }


def main() -> None:
    trials = []
    print("=== step down / step up sweep (physics, 4 s) ===")
    print(f"{'kind':10s} {'mm':>4s} {'controller':16s} {'falls':>6s} "
          f"{'peak_mm':>8s} {'rms_mm':>7s} {'res_rms':>8s}")
    for terrain, kind in STEPS:
        for h in HEIGHTS_MM:
            for controller in CONTROLLERS:
                rows = []
                for seed in SEEDS:
                    _, s = run_trial(controller, terrain, seed, duration=4.0,
                                     terrain_height_mm=h)
                    s["kind"] = kind; s["height_mm"] = h
                    rows.append(s); trials.append(s)
                c = _cell(rows)
                print(f"{kind:10s} {h:4.0f} {controller:16s} {c['falls']:5d}/10 "
                      f"{c['com_peak_mm']:8.1f} {c['com_rms_mm']:7.1f} "
                      f"{c['realization_residual_rms_mps2']:8.3f}")

    print("\n=== merged push + platform (obstacle 30 mm + lateral push) ===")
    merged = []
    spec = PushSpec(direction="lateral", phase="single_support",
                    magnitude_n=90.0, duration_s=0.15)
    for controller in CONTROLLERS:
        rows = []
        for seed in SEEDS:
            _, s = run_trial(controller, "obstacle", seed, duration=4.0,
                             push=spec, terrain_height_mm=30.0)
            s["kind"] = "push+platform"
            rows.append(s); merged.append(s)
        c = _cell(rows)
        print(f"{'push+plat':10s} {'30':>4s} {controller:16s} {c['falls']:5d}/10 "
              f"{c['com_peak_mm']:8.1f} {c['com_rms_mm']:7.1f} "
              f"{c['realization_residual_rms_mps2']:8.3f}")

    def summarize(rows):
        cells = {}
        for r in rows:
            key = f"{r['kind']}|{int(r.get('height_mm', 30))}|{r['controller']}"
            cells.setdefault(key, []).append(r)
        return {k: _cell(v) for k, v in cells.items()}

    out = {
        "benchmark": "platform_vignette",
        "substrate": "physics (4 s window)",
        "heights_mm": list(HEIGHTS_MM), "seeds": list(SEEDS),
        "sweep_cells": summarize(trials),
        "merged_cells": summarize(merged),
        "trials": trials + merged,
    }
    path = RESULTS / "platform_vignette.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved {path}  ({len(trials) + len(merged)} trials)")


if __name__ == "__main__":
    main()
