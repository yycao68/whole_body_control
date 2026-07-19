#!/usr/bin/env python3
"""Fail closed when the uneven-ground manuscript evidence is incomplete."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE.parent / "figures"
MANUSCRIPT = HERE.parent / "wbc_ieee.md"
DATASET = RESULTS / "uneven_ground_benchmark.json"
PUSH_DATASET = RESULTS / "external_push_benchmark.json"
SUSTAINED_DATASET = RESULTS / "sustained_push_benchmark.json"
PLATFORM_DATASET = RESULTS / "platform_vignette.json"
TERRAINS = ("flat", "depression", "obstacle", "rough")
CONTROLLERS = ("impedance", "nominal_mpc", "interaction_mpc", "no_realization_feedback")
PUSH_CONDITIONS = (("lateral", "double_support"), ("lateral", "single_support"),
                   ("forward", "double_support"), ("forward", "single_support"))
PUSH_FIGURES = ("external_push_summary.png", "external_push_prediction.png",
                "external_push_response.png")
SEEDS = set(range(4200, 4210))
METRICS = ("com_xyz_rms_mm", "com_xyz_peak_mm", "roll_pitch_rms_rad",
           "max_abs_roll_pitch_rad", "realization_residual_rms_mps2",
           "realization_residual_peak_mps2", "peak_contact_force_n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_supplementary(manuscript: str) -> dict:
    # Sustained-force sweep (reduced model, Section X-H).
    sd = json.loads(SUSTAINED_DATASET.read_bytes())
    require(len(sd["trials"]) == 60, f"expected 60 sustained-force trials, found {len(sd['trials'])}")
    nom30 = sd["cells"]["nominal|30"]["steady_offset_mm"]
    int30 = sd["cells"]["interaction|30"]["steady_offset_mm"]
    require(int30 < 0.2 * nom30,
            f"offset-free claim fails: 30 N steady offset {int30:.1f} vs {nom30:.1f} mm")
    require(abs(nom30 - 42.8) < 0.2 and abs(int30 - 4.2) < 0.2,
            f"30 N steady-offset drift: {nom30:.2f}->{int30:.2f}")
    require(sd["cells"]["interaction|30"]["recovered_fraction"] == 1.0,
            "30 N ID-MPC should recover 10/10")
    require((FIGURES / "sustained_push_offset.png").stat().st_size > 10_000,
            "missing figure: sustained_push_offset.png")

    # Step-height / combined vignette (physics, Section X-I).
    pv = json.loads(PLATFORM_DATASET.read_bytes())
    require(len(pv["trials"]) == 140, f"expected 140 platform trials, found {len(pv['trials'])}")
    up30_nom = pv["sweep_cells"]["step_up|30|nominal_mpc"]["com_peak_mm"]
    up30_int = pv["sweep_cells"]["step_up|30|interaction_mpc"]["com_peak_mm"]
    require(up30_int < up30_nom,
            f"step-up 30 mm: ID-MPC peak {up30_int:.1f} not below nominal {up30_nom:.1f}")
    down40_int_falls = pv["sweep_cells"]["step_down|40|interaction_mpc"]["falls"]
    require(down40_int_falls >= 5,
            f"40 mm step-down ID-MPC should destabilize; falls={down40_int_falls}")

    for claim in ("Sustained-Force Rejection", "42.8", "4.2",
                  "49.8", "30.1", "19.6"):
        require(claim in manuscript, f"supplementary manuscript statement missing: {claim}")
    return {
        "sustained_dataset": SUSTAINED_DATASET.name,
        "sustained_sha256": hashlib.sha256(SUSTAINED_DATASET.read_bytes()).hexdigest(),
        "sustained_30N_offset_mm": {"nominal": nom30, "interaction": int30},
        "platform_dataset": PLATFORM_DATASET.name,
        "platform_sha256": hashlib.sha256(PLATFORM_DATASET.read_bytes()).hexdigest(),
        "step_up_30mm_peak_mm": {"nominal": up30_nom, "interaction": up30_int},
    }


def verify_push(manuscript: str) -> dict:
    raw = PUSH_DATASET.read_bytes()
    data = json.loads(raw)
    trials = data["trials"]
    require(len(trials) == 160, f"expected 160 push trials, found {len(trials)}")
    require(float(data["push_magnitude_n"]) == 90.0, "push magnitude is not 90 N")
    for direction, phase in PUSH_CONDITIONS:
        for controller in CONTROLLERS:
            cell = [t for t in trials if t["direction"] == direction
                    and t["phase"] == phase and t["controller"] == controller]
            require(len(cell) == 10, f"push {direction}/{phase}/{controller}: expected 10")
            require({t["seed"] for t in cell} == SEEDS,
                    f"push {direction}/{phase}/{controller}: seed mismatch")
            require(all(t.get("onset_found") for t in cell),
                    f"push {direction}/{phase}/{controller}: onset missing")
    require(sum(int(t["qp_fallbacks"]) for t in trials) == 0, "push QP fallback present")
    for name in PUSH_FIGURES:
        require((FIGURES / name).stat().st_size > 10_000, f"missing push figure: {name}")

    def peak(controller: str) -> float:
        return float(np.median([
            t["post_com_peak_mm"] for t in trials
            if t["direction"] == "lateral" and t["phase"] == "single_support"
            and t["controller"] == controller]))
    nominal_peak, interaction_peak = peak("nominal_mpc"), peak("interaction_mpc")
    require(abs(nominal_peak - 57.3) < 0.05 and abs(interaction_peak - 31.0) < 0.05,
            f"lateral SS peak drift: {nominal_peak:.2f}->{interaction_peak:.2f}")
    for claim in ("a 160-trial external-push study", "57.3 to 31.0 mm", "phase-locked"):
        require(claim in manuscript, f"push manuscript statement missing: {claim}")
    return {
        "push_dataset": PUSH_DATASET.name,
        "push_sha256": hashlib.sha256(raw).hexdigest(),
        "push_trials": len(trials),
        "lateral_ss_nominal_peak_mm": nominal_peak,
        "lateral_ss_interaction_peak_mm": interaction_peak,
        "push_figures": list(PUSH_FIGURES),
    }


def main() -> None:
    raw = DATASET.read_bytes()
    data = json.loads(raw)
    trials = data["trials"]
    require(len(trials) == 160, f"expected 160 trials, found {len(trials)}")
    require(data["metadata"]["wbc_dt_s"] == 0.002, "WBC schedule is not 500 Hz")
    require(data["metadata"]["mpc_dt_s"] == 0.01, "MPC schedule is not 100 Hz")
    for terrain in TERRAINS:
        for controller in CONTROLLERS:
            cell = [t for t in trials if t["terrain"] == terrain and t["controller"] == controller]
            require(len(cell) == 10, f"{terrain}/{controller}: expected 10 trials")
            require({t["seed"] for t in cell} == SEEDS, f"{terrain}/{controller}: seed mismatch")
            require(sum(int(t["qp_fallbacks"]) for t in cell) == 0,
                    f"{terrain}/{controller}: QP fallback present")
            for trial in cell:
                for metric in METRICS:
                    require(math.isfinite(float(trial[metric])),
                            f"nonfinite {metric} in {terrain}/{controller}/{trial['seed']}")
                for horizon in ("1", "5", "10"):
                    prediction = trial["prediction"][horizon]
                    for metric in ("nominal_com_rmse_mm", "augmented_com_rmse_mm",
                                   "nominal_roll_pitch_rmse_mrad",
                                   "augmented_roll_pitch_rmse_mrad"):
                        require(math.isfinite(float(prediction[metric])),
                                f"nonfinite prediction metric {metric}")
    figure_names = ("uneven_ground_tracking.png", "uneven_ground_prediction.png",
                    "uneven_ground_timeseries.png", "uneven_ground_timing.png")
    for name in figure_names:
        require((FIGURES / name).stat().st_size > 10_000, f"missing or empty figure: {name}")

    # Recompute the manuscript's paired obstacle result rather than treating a
    # rounded prose value as an independent source.
    obstacle_nominal = np.asarray([
        t["com_xyz_peak_mm"] for t in trials
        if t["terrain"] == "obstacle" and t["controller"] == "nominal_mpc"
    ])
    obstacle_interaction = np.asarray([
        t["com_xyz_peak_mm"] for t in trials
        if t["terrain"] == "obstacle" and t["controller"] == "interaction_mpc"
    ])
    paired_delta = obstacle_interaction - obstacle_nominal
    rng = np.random.default_rng(7)
    resampled = paired_delta[rng.integers(0, len(paired_delta), (20_000, len(paired_delta)))]
    bootstrap_medians = np.median(resampled, axis=1)
    obstacle_delta = float(np.median(paired_delta))
    obstacle_ci = [float(x) for x in np.quantile(bootstrap_medians, [0.025, 0.975])]

    manuscript = MANUSCRIPT.read_text()
    required_claims = (
        "a 160-trial terrain study",
        "320 torque-level runs",
        "-13.06%",
        "-1.79 mm",
        "2.77 ms",
        "7.71 ms",
        "prototype measurement on a non-real-time host",
    )
    for claim in required_claims:
        require(claim in manuscript, f"manuscript evidence statement missing: {claim}")

    push_report = verify_push(manuscript)
    supp_report = verify_supplementary(manuscript)

    wbc_median = sorted(t["timing"]["wbc"]["median_ms"] for t in trials)
    mpc_median = sorted(t["timing"]["mpc"]["median_ms"] for t in trials)
    report = {
        "status": "PASS",
        "dataset": DATASET.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "trials": len(trials),
        "cells": len(TERRAINS) * len(CONTROLLERS),
        "qp_fallbacks": sum(int(t["qp_fallbacks"]) for t in trials),
        "wbc_median_of_trial_medians_ms": 0.5 * (wbc_median[79] + wbc_median[80]),
        "mpc_median_of_trial_medians_ms": 0.5 * (mpc_median[79] + mpc_median[80]),
        "obstacle_interaction_minus_nominal_peak_median_mm": obstacle_delta,
        "obstacle_peak_paired_bootstrap_95pct_mm": obstacle_ci,
        "figures": list(figure_names),
        **push_report,
        **supp_report,
    }
    out = RESULTS / "uneven_ground_verification.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
