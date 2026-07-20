#!/usr/bin/env python3
"""Phase-locked external-push benchmark for the publication experiment.

A finite-duration half-sine torso wrench is applied to the plant while the same
nominal gait is replayed.  The wrench perturbs only the simulated robot; it is
logged at 1 kHz for ground truth but is hidden from the estimator and every
controller in this primary comparison. Three controllers, four direction/phase
conditions, and ten paired seeds give 120 push trials, reusing the terrain
benchmark's reference provider, realizer, controller factory, and seed pairing.

Push claims are supported by prediction, peak-error, and recovery-time metrics
reported separately per direction and phase. Remaining upright is not by
itself evidence.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from run_uneven_ground_benchmark import (
    COMMAND_SLEW_TASK_ACC_PER_UPDATE, CONTROLLERS, PUBLICATION_GAIT, PUSH_CONDITIONS,
    RESIDUAL_CAP_TASK_ACC, RESIDUAL_DEADBAND_TASK_ACC, RESULTS, SIM_DT,
    PushSpec, _prediction_rmse, run_trial,
)

# Frozen evaluation constants (declared before the final seeds are run).
PUSH_MAGNITUDE_N = 90.0        # peak of the half-sine torso force
PUSH_DURATION_S = 0.15
RECOVERY_BAND_MM = 12.0        # CoM planar error band
RECOVERY_DWELL_S = 0.20        # must stay inside the band this long
POST_WINDOW_S = 2.0            # analysis window after onset
SEED_START = 4200
N_SEEDS = 10


def _post_push_metrics(log: dict, onset: float) -> dict:
    t = np.asarray(log["t"])
    fell_any = bool(np.any(log["fell"]))
    end = min(onset + POST_WINDOW_S, float(t[-1]))
    win = (t >= onset) & (t <= end) & (np.asarray(log["fell"]) == 0)
    e = np.asarray(log["task_error"])[win]
    rr = np.asarray(log["realization_residual"])[win]
    rpy = np.asarray(log["rpy"])[win]
    com = np.asarray(log["com"])[win]
    com_ref = np.asarray(log["com_ref"])[win]

    # Recovery: first post-onset time the CoM planar error stays inside the
    # frozen band for RECOVERY_DWELL_S continuously.
    planar_mm = 1000.0 * np.linalg.norm(np.asarray(log["task_error"])[:, :2], axis=1)
    recovery = None
    after = np.flatnonzero((t >= onset) & (np.asarray(log["fell"]) == 0))
    dwell_steps = int(round(RECOVERY_DWELL_S / SIM_DT))
    for k in after:
        if k + dwell_steps >= len(t):
            break
        if np.all(planar_mm[k:k + dwell_steps] < RECOVERY_BAND_MM):
            recovery = float(t[k] - onset)
            break

    def rms(a):
        return float(np.sqrt(np.mean(a ** 2))) if a.size else None

    def peak(a):
        return float(np.max(np.abs(a))) if a.size else None

    return {
        "post_com_peak_mm": None if not e.size else float(1000 * np.max(np.abs(e[:, :3]))),
        "post_com_rms_mm": None if not e.size else float(1000 * np.sqrt(np.mean(e[:, :3] ** 2))),
        "post_roll_pitch_peak_rad": peak(rpy[:, :2]),
        "post_roll_pitch_rms_rad": rms(e[:, 3:5]) if e.size else None,
        "max_com_excursion_mm": None if not com.size else float(
            1000 * np.max(np.linalg.norm((com - com_ref)[:, :2], axis=1))),
        "recovery_time_s": recovery,
        "recovered": recovery is not None,
        "realization_residual_rms_mps2": rms(rr),
        "realization_residual_peak_mps2": peak(rr),
        "fell": fell_any,
    }


def _post_push_prediction(log: dict, onset: float) -> dict:
    """Prediction audit restricted to the post-onset window."""
    sub = dict(log)
    mask = (np.asarray(log["t"]) >= onset) & (np.asarray(log["t"]) <= onset + POST_WINDOW_S)
    sub["mpc_sample"] = np.asarray(log["mpc_sample"]).astype(int) * mask.astype(int)
    return _prediction_rmse(sub, SIM_DT)


def run_condition(direction: str, phase: str, controller: str, seed: int,
                  duration: float) -> dict:
    spec = PushSpec(direction=direction, phase=phase,
                    magnitude_n=PUSH_MAGNITUDE_N, duration_s=PUSH_DURATION_S)
    log, summary = run_trial(controller, "flat", seed, duration=duration, push=spec)
    onset = summary["push"]["onset_s"]
    out = {
        "controller": controller, "direction": direction, "phase": phase,
        "seed": int(seed), "push": summary["push"],
        "qp_fallbacks": summary["qp_fallbacks"],
        "max_torque_utilization": summary["max_torque_utilization"],
        "peak_contact_force_n": summary["peak_contact_force_n"],
    }
    if onset is None:
        out["onset_found"] = False
        return out
    out["onset_found"] = True
    out.update(_post_push_metrics(log, onset))
    out["post_push_prediction"] = _post_push_prediction(log, onset)
    return out, log


def _median(values):
    v = [x for x in values if x is not None]
    return float(np.median(v)) if v else None


def aggregate(trials: list[dict]) -> dict:
    cells = {}
    for direction, phase in PUSH_CONDITIONS:
        for controller in CONTROLLERS:
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
    ap.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=list(CONTROLLERS))
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="subset like lateral:single_support")
    ap.add_argument("--save-representative", action="store_true")
    ap.add_argument("--artifact", default="external_push_benchmark.json")
    args = ap.parse_args()

    conditions = PUSH_CONDITIONS
    if args.conditions:
        want = set(args.conditions)
        conditions = tuple(c for c in PUSH_CONDITIONS if f"{c[0]}:{c[1]}" in want)

    trials: list[dict] = []
    rep_saved = set()
    for direction, phase in conditions:
        for i in range(args.seeds):
            seed = args.seed_start + i
            for controller in args.controllers:
                print(f"PUSH {direction}/{phase} seed={seed} {controller}", flush=True)
                res = run_condition(direction, phase, controller, seed, args.duration)
                if isinstance(res, tuple):
                    row, log = res
                else:
                    row, log = res, None
                trials.append(row)
                key = (direction, phase, controller)
                if args.save_representative and log is not None and key not in rep_saved \
                        and seed == args.seed_start:
                    np.savez_compressed(
                        RESULTS / f"push_{direction}_{phase}_{controller}_seed{seed}.npz", **log)
                    rep_saved.add(key)

    out = {
        "schema_version": 2,
        "benchmark": "external_push",
        "sim_dt_s": SIM_DT, "duration_s": args.duration,
        "push_magnitude_n": PUSH_MAGNITUDE_N, "push_duration_s": PUSH_DURATION_S,
        "recovery_band_mm": RECOVERY_BAND_MM, "recovery_dwell_s": RECOVERY_DWELL_S,
        "post_window_s": POST_WINDOW_S,
        "seed_start": args.seed_start, "n_seeds": args.seeds,
        "controllers": list(args.controllers),
        "gait": PUBLICATION_GAIT,
        "residual_conditioning": {
            "deadband_task_acceleration": RESIDUAL_DEADBAND_TASK_ACC,
            "cap_task_acceleration": RESIDUAL_CAP_TASK_ACC,
            "command_slew_task_acceleration_per_update": COMMAND_SLEW_TASK_ACC_PER_UPDATE,
        },
        "conditions": [f"{d}:{p}" for d, p in conditions],
        "platform": platform.platform(), "python": platform.python_version(),
        "cells": aggregate(trials),
        "trials": trials,
    }
    path = RESULTS / args.artifact
    path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {path}  ({len(trials)} trials)")


if __name__ == "__main__":
    main()
