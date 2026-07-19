#!/usr/bin/env python3
"""Fail closed unless the manuscript's evidence and experimental configuration hold.

This checks *validity*, not just string presence: it recomputes the reported
medians from the committed JSON, verifies the measured contact phase at every
push onset, and confirms the evaluated configuration (soft realizer, single
shared estimator, three reported controllers).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE.parent / "figures"
MANUSCRIPT = HERE.parent / "wbc_ieee.md"
BENCHMARK_SRC = HERE / "run_uneven_ground_benchmark.py"
DATASET = RESULTS / "uneven_ground_benchmark.json"
PUSH_DATASET = RESULTS / "external_push_benchmark.json"
SUSTAINED_DATASET = RESULTS / "sustained_push_benchmark.json"
PLATFORM_DATASET = RESULTS / "platform_vignette.json"
TERRAINS = ("flat", "depression", "obstacle", "rough")
CONTROLLERS = ("impedance", "nominal_mpc", "interaction_mpc")  # three reported
PUSH_CONDITIONS = (("lateral", "double_support"), ("lateral", "single_support"),
                   ("forward", "double_support"), ("forward", "single_support"))
PUSH_FIGURES = ("external_push_summary.png", "external_push_response.png")
SEEDS = set(range(4200, 4210))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def median_cell(trials, **sel):
    key = sel.pop("metric")
    rows = [t for t in trials if all(t.get(k) == v for k, v in sel.items())]
    vals = [t[key] for t in rows if t.get(key) is not None]
    return float(np.median(vals)) if vals else math.nan


def verify_configuration() -> dict:
    """The evaluated configuration must match what the paper describes."""
    src = BENCHMARK_SRC.read_text()
    # The realizer that runs the benchmark uses the soft (exact_realizer=False)
    # configuration described by Eq. (12); the hard-constraint mode is not used.
    require("InverseDynamicsQPRealizer(model, exact_realizer=False)" in src,
            "benchmark realizer is not the evaluated exact_realizer=False configuration")
    require("exact_realizer=True" not in src,
            "benchmark must not evaluate the exact realizer")
    # The realization-feedback ablation was removed: both controllers use the
    # single FilteredAccelerationResidualEstimator, so no separate random-walk
    # observer confounds the comparison.
    require("RandomWalkDisturbanceObserver" not in src,
            "a second (confounding) observer is present in the benchmark")
    require("est.effective" in src,
            "ID-MPC must use the combined interaction+realization estimate (est.effective)")
    return {"realizer": "soft (exact_realizer=False)",
            "estimator": "single FilteredAccelerationResidualEstimator"}


def verify_push(manuscript: str) -> dict:
    data = json.loads(PUSH_DATASET.read_bytes())
    trials = data["trials"]
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
    # The central validity fix: every push must land in its *measured* contact
    # phase -- one foot down for single support, both for double support.
    want = {"single_support": 1, "double_support": 2}
    for t in trials:
        if not t.get("onset_found"):
            continue
        n_contact = int(t["push"]["onset_contact"][0]) + int(t["push"]["onset_contact"][1])
        require(n_contact == want[t["phase"]],
                f"push onset contact {t['push']['onset_contact']} inconsistent with "
                f"{t['phase']} (seed {t['seed']}, {t['direction']}, {t['controller']})")
    require(sum(int(t["qp_fallbacks"]) for t in trials) == 0, "push QP fallback present")
    for name in PUSH_FIGURES:
        require((FIGURES / name).stat().st_size > 10_000, f"missing push figure: {name}")

    # Recompute the reported lateral single-support medians and the every-condition
    # improvement rather than trusting a rounded prose number.
    ss_nom = median_cell(trials, metric="post_com_peak_mm", direction="lateral",
                         phase="single_support", controller="nominal_mpc")
    ss_int = median_cell(trials, metric="post_com_peak_mm", direction="lateral",
                         phase="single_support", controller="interaction_mpc")
    require(abs(ss_nom - 14.4) < 0.3 and abs(ss_int - 10.6) < 0.3,
            f"lateral SS peak drift: nominal {ss_nom:.2f}, ID-MPC {ss_int:.2f}")
    for direction, phase in PUSH_CONDITIONS:
        n = median_cell(trials, metric="post_com_peak_mm", direction=direction,
                        phase=phase, controller="nominal_mpc")
        i = median_cell(trials, metric="post_com_peak_mm", direction=direction,
                        phase=phase, controller="interaction_mpc")
        require(i < n, f"ID-MPC does not lower peak vs nominal in {direction}/{phase} "
                       f"({i:.2f} vs {n:.2f})")
    max_torque = max(t["max_torque_utilization"] for t in trials if t.get("onset_found"))
    require(max_torque < 1.0,
            f"torque utilization saturated ({max_torque:.2f}); no-saturation claim fails")
    for claim in ("three controllers", "14.4 to 10.6 mm", "measured single- or double-support"):
        require(claim in manuscript, f"push manuscript statement missing: {claim!r}")
    return {
        "push_dataset": PUSH_DATASET.name,
        "push_sha256": hashlib.sha256(PUSH_DATASET.read_bytes()).hexdigest(),
        "lateral_ss_nominal_peak_mm": ss_nom,
        "lateral_ss_interaction_peak_mm": ss_int,
        "push_max_torque_utilization": max_torque,
    }


def verify_supplementary(manuscript: str) -> dict:
    sd = json.loads(SUSTAINED_DATASET.read_bytes())
    require(len(sd["trials"]) == 60, f"expected 60 sustained-force trials, found {len(sd['trials'])}")
    nom30 = sd["cells"]["nominal|30"]["steady_offset_mm"]
    int30 = sd["cells"]["interaction|30"]["steady_offset_mm"]
    require(int30 < 0.2 * nom30,
            f"offset-free claim fails: 30 N steady offset {int30:.1f} vs {nom30:.1f} mm")
    require(abs(nom30 - 42.8) < 0.5 and abs(int30 - 4.2) < 0.5,
            f"30 N steady-offset drift: {nom30:.2f}->{int30:.2f}")
    require((FIGURES / "sustained_push_offset.png").stat().st_size > 10_000,
            "missing figure: sustained_push_offset.png")

    pv = json.loads(PLATFORM_DATASET.read_bytes())
    # Step-down sweep: both controllers stay upright through 40 mm (the earlier
    # ID-MPC destabilization was a random-walk-observer artifact, now resolved).
    for h in (20, 30, 40):
        for c in ("nominal_mpc", "interaction_mpc"):
            require(pv["sweep_cells"][f"step_down|{h}|{c}"]["falls"] == 0,
                    f"step-down {h} mm {c} should not fall")
    # Combined raised-lane + push: ID-MPC lowers the peak.
    mp_nom = pv["merged_cells"]["push+platform|30|nominal_mpc"]["com_peak_mm"]
    mp_int = pv["merged_cells"]["push+platform|30|interaction_mpc"]["com_peak_mm"]
    require(mp_int < mp_nom,
            f"combined push+platform: ID-MPC peak {mp_int:.1f} not below nominal {mp_nom:.1f}")
    require("Sustained-Force Rejection" in manuscript, "supplementary section title missing")
    return {
        "sustained_30N_offset_mm": {"nominal": nom30, "interaction": int30},
        "combined_push_platform_peak_mm": {"nominal": mp_nom, "interaction": mp_int},
    }


def main() -> None:
    raw = DATASET.read_bytes()
    data = json.loads(raw)
    trials = data["trials"]
    require(data["metadata"]["wbc_dt_s"] == 0.002, "WBC schedule is not 500 Hz")
    require(data["metadata"]["mpc_dt_s"] == 0.01, "MPC schedule is not 100 Hz")
    for terrain in TERRAINS:
        for controller in CONTROLLERS:
            cell = [t for t in trials if t["terrain"] == terrain and t["controller"] == controller]
            require(len(cell) == 10, f"{terrain}/{controller}: expected 10 trials")
            require({t["seed"] for t in cell} == SEEDS, f"{terrain}/{controller}: seed mismatch")
            require(sum(int(t["qp_fallbacks"]) for t in cell) == 0,
                    f"{terrain}/{controller}: QP fallback present")
    figure_names = ("uneven_ground_tracking.png", "uneven_ground_prediction.png",
                    "uneven_ground_timeseries.png", "uneven_ground_timing.png")
    for name in figure_names:
        require((FIGURES / name).stat().st_size > 10_000, f"missing or empty figure: {name}")

    # Recompute the headline obstacle peak improvement rather than trusting prose.
    nom = data["cells"]["obstacle/nominal_mpc"]["com_xyz_peak_mm"]["median"]
    inter = data["cells"]["obstacle/interaction_mpc"]["com_xyz_peak_mm"]["median"]
    obstacle_pct = 100.0 * (inter - nom) / nom
    require(abs(obstacle_pct - (-19.8)) < 1.0,
            f"obstacle peak improvement drift: {obstacle_pct:.2f}% (expected ~-19.8%)")

    manuscript = MANUSCRIPT.read_text()
    for claim in ("three controllers", "19.8%",
                  "prototype measurement on a non-real-time host"):
        require(claim in manuscript, f"manuscript evidence statement missing: {claim!r}")

    config_report = verify_configuration()
    push_report = verify_push(manuscript)
    supp_report = verify_supplementary(manuscript)

    report = {
        "status": "PASS",
        "dataset": DATASET.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "reported_controllers": list(CONTROLLERS),
        "obstacle_interaction_vs_nominal_peak_pct": obstacle_pct,
        **config_report,
        **push_report,
        **supp_report,
    }
    out = RESULTS / "uneven_ground_verification.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
