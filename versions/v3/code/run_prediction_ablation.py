#!/usr/bin/env python3
"""Does the benefit of ID-MPC come from PREDICTING interaction, or just from
having an estimate of it?

Three-stage ablation, run on the same phase-locked external-push protocol as
the paper's primary benchmark (run_external_push_benchmark.py), reusing its
reference provider, realizer, seed pairing, and metrics unchanged:

  1. ``nominal_mpc``       -- no residual estimate at all (d_hat = 0).
  2. ``residual_feedback`` -- the SAME low-pass estimator and the SAME
                              dead-zone/saturation shaping of the residual as
                              ``interaction_mpc``, applied as a direct reactive
                              cancellation on top of the impedance feedback law.
                              No MPC, no horizon propagation: the estimate is
                              used, not predicted forward.
  3. ``interaction_mpc``   -- the full ID-MPC: the identical estimate, now
                              propagated as a persisting disturbance over the
                              MPC's receding horizon (Theorem 1's model).

Stages 2 and 3 share one estimator and one residual-shaping formula; the only
thing that changes between them is whether the estimate is treated as
persisting into the future. If ID-MPC's improvement over nominal_mpc were
just "having an estimate to react to," stage 2 would already capture most of
it and stage 3 would add little. If the improvement instead comes from
predicting that the interaction persists and shaping the correction over the
horizon accordingly, stage 3 should separate clearly from stage 2.
"""

from __future__ import annotations

import argparse
import json
import platform

import numpy as np

from run_uneven_ground_benchmark import PUSH_CONDITIONS, RESULTS, SIM_DT
from run_external_push_benchmark import (
    PUSH_MAGNITUDE_N, PUSH_DURATION_S, SEED_START, N_SEEDS,
    run_condition,
)

ABLATION_STAGES = ("nominal_mpc", "residual_feedback", "interaction_mpc")


def _median(values):
    v = [x for x in values if x is not None]
    return float(np.median(v)) if v else None


def aggregate(trials: list[dict]) -> dict:
    """Same cell layout as run_external_push_benchmark.aggregate, but keyed
    over ABLATION_STAGES instead of the imported (unrelated) CONTROLLERS
    tuple, so the residual_feedback stage isn't silently dropped."""
    cells = {}
    for direction, phase in PUSH_CONDITIONS:
        for controller in ABLATION_STAGES:
            rows = [t for t in trials
                    if t["direction"] == direction and t["phase"] == phase
                    and t["controller"] == controller]
            if not rows:
                continue
            pred10 = [r["post_push_prediction"]["10"] for r in rows if "post_push_prediction" in r]
            cells[f"{direction}|{phase}|{controller}"] = {
                "direction": direction, "phase": phase, "controller": controller,
                "n": len(rows),
                "com_peak_mm": _median([r.get("post_com_peak_mm") for r in rows]),
                "com_rms_mm": _median([r.get("post_com_rms_mm") for r in rows]),
                "roll_pitch_peak_rad": _median([r.get("post_roll_pitch_peak_rad") for r in rows]),
                "recovery_time_s": _median([r.get("recovery_time_s") for r in rows]),
                "recovered_fraction": float(np.mean([r.get("recovered", False) for r in rows])),
                "realization_residual_rms_mps2": _median(
                    [r.get("realization_residual_rms_mps2") for r in rows]),
                "pred10_nominal_com_rmse_mm": _median(
                    [p["nominal_com_rmse_mm"] for p in pred10]),
                "pred10_augmented_com_rmse_mm": _median(
                    [p["augmented_com_rmse_mm"] for p in pred10]),
                "falls": int(sum(r.get("fell", False) for r in rows)),
            }
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--seed-start", type=int, default=SEED_START)
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="subset like lateral:single_support")
    args = ap.parse_args()

    conditions = PUSH_CONDITIONS
    if args.conditions:
        want = set(args.conditions)
        conditions = tuple(c for c in PUSH_CONDITIONS if f"{c[0]}:{c[1]}" in want)

    trials: list[dict] = []
    for direction, phase in conditions:
        for i in range(args.seeds):
            seed = args.seed_start + i
            for controller in ABLATION_STAGES:
                print(f"ABLATION {direction}/{phase} seed={seed} {controller}", flush=True)
                res = run_condition(direction, phase, controller, seed, args.duration)
                row = res[0] if isinstance(res, tuple) else res
                trials.append(row)

    out = {
        "benchmark": "prediction_ablation",
        "description": (
            "nominal_mpc -> residual_feedback (estimate, reactive, no horizon "
            "propagation) -> interaction_mpc (same estimate, predicted over the "
            "MPC horizon), on the paper's external-push protocol."
        ),
        "sim_dt_s": SIM_DT, "duration_s": args.duration,
        "push_magnitude_n": PUSH_MAGNITUDE_N, "push_duration_s": PUSH_DURATION_S,
        "seed_start": args.seed_start, "n_seeds": args.seeds,
        "controllers": list(ABLATION_STAGES),
        "conditions": [f"{d}:{p}" for d, p in conditions],
        "platform": platform.platform(), "python": platform.python_version(),
        "cells": aggregate(trials),
        "trials": trials,
    }
    path = RESULTS / "prediction_ablation.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {path}  ({len(trials)} trials)")


if __name__ == "__main__":
    main()
