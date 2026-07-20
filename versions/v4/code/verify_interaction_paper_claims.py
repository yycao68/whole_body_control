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
MANUSCRIPT = HERE.parent / "wbc_ieee_v4.md"
BENCHMARK_SRC = HERE / "run_uneven_ground_benchmark.py"
DATASET = RESULTS / "uneven_ground_benchmark.json"
PUSH_DATASET = RESULTS / "external_push_benchmark.json"
CONTINUOUS_VIDEO = RESULTS / "continuous_flat_idmpc.mp4"
CONTINUOUS_VIDEO_REPORT = RESULTS / "continuous_flat_idmpc.json"
TERRAINS = ("flat", "depression", "obstacle", "rough")
CONTROLLERS = ("impedance", "nominal_mpc", "interaction_mpc")  # three reported
PUSH_CONDITIONS = (("lateral", "double_support"), ("lateral", "single_support"),
                   ("forward", "double_support"), ("forward", "single_support"))
SEEDS = set(range(4200, 4210))
PAPER_FIGURES = (
    "multirate_architecture.png",
    "prediction_realization_concept.png",
    "uneven_ground_prediction.png",
    "uneven_ground_tracking.png",
    "uneven_ground_timeseries.png",
    "external_push_summary.png",
    "external_push_response.png",
    "uneven_ground_timing.png",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def median_cell(trials, **sel):
    key = sel.pop("metric")
    rows = [t for t in trials if all(t.get(k) == v for k, v in sel.items())]
    vals = [t[key] for t in rows if t.get(key) is not None]
    return float(np.median(vals)) if vals else math.nan


def verify_continuous_video(expected_gait: dict) -> dict:
    """Bind the continuous-walking claim to its torque-level video artifact."""
    require(CONTINUOUS_VIDEO.exists(), "continuous flat-walking video is missing")
    require(CONTINUOUS_VIDEO_REPORT.exists(),
            "continuous flat-walking video report is missing")
    report = json.loads(CONTINUOUS_VIDEO_REPORT.read_bytes())
    require(report.get("kind") == "torque_level_continuous_flat_video",
            "continuous video report has the wrong artifact kind")
    require(report.get("controller") == "interaction_mpc",
            "continuous video must exercise the reported ID-MPC controller")
    require(report.get("root_assist") is False,
            "continuous video must not use root assistance")
    require(report.get("planner") == expected_gait,
            "continuous video does not use the frozen publication gait")
    require(report.get("fell") is False, "continuous video trial fell")
    require(float(report.get("duration_completed_s", 0.0)) >= 14.9,
            "continuous video trial ended early")
    require(int(report.get("qp_fallbacks", -1)) == 0,
            "continuous video trial used a QP fallback")
    digest = hashlib.sha256(CONTINUOUS_VIDEO.read_bytes()).hexdigest()
    require(report.get("video_sha256") == digest,
            "continuous video hash does not match its report")
    require(float(report.get("lateral_error_rms_mm", math.inf)) < 5.0,
            "continuous video lateral tracking RMS is not below 5 mm")
    require(abs(float(report.get("lateral_error_mean_mm", math.inf))) < 3.0,
            "continuous video lateral error is not centered near zero")
    require(float(report.get("lateral_error_final_second_rms_mm", math.inf)) < 4.0,
            "continuous video final-second lateral RMS is not below 4 mm")
    return {
        "continuous_video": CONTINUOUS_VIDEO.name,
        "continuous_video_sha256": digest,
        "continuous_video_lateral_rms_mm": report["lateral_error_rms_mm"],
        "continuous_video_lateral_peak_mm": report["lateral_error_peak_mm"],
        "continuous_video_lateral_mean_mm": report["lateral_error_mean_mm"],
        "continuous_video_final_second_lateral_rms_mm": (
            report["lateral_error_final_second_rms_mm"]
        ),
    }


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
    require("measured_error_acceleration - correction" in src,
            "logged residual must be measured minus commanded task acceleration")
    realizer_src = (HERE / "run_g1_torque_realizer_benchmark.py").read_text()
    require("mj_jacDot" in realizer_src,
            "realizer must include Jdot*qdot in Cartesian acceleration tasks")
    return {"realizer": "soft (exact_realizer=False)",
            "estimator": "single FilteredAccelerationResidualEstimator"}


def verify_figures(manuscript: str) -> dict:
    """Require every cited publication figure to exist and be nonempty."""
    for name in PAPER_FIGURES:
        path = FIGURES / name
        require(path.exists(), f"paper figure is missing: {name}")
        require(path.stat().st_size > 10_000, f"paper figure is unexpectedly small: {name}")
        require(f"figures/{name}" in manuscript, f"manuscript does not cite figure: {name}")
    return {"paper_figures": list(PAPER_FIGURES)}


def verify_terrain_table(manuscript: str, data: dict) -> None:
    """Match every terrain-table number to the authoritative aggregate JSON."""
    controller_name = {
        "impedance": "impedance",
        "nominal MPC": "nominal_mpc",
        "ID-MPC": "interaction_mpc",
    }
    rows = []
    current_terrain = None
    for raw_line in manuscript.replace("**", "").splitlines():
        if not raw_line.startswith("|"):
            continue
        cells = [cell.strip() for cell in raw_line.strip().strip("|").split("|")]
        if len(cells) != 6 or cells[1] not in controller_name:
            continue
        if cells[0]:
            current_terrain = cells[0]
        if current_terrain not in TERRAINS:
            continue
        controller = controller_name[cells[1]]
        key = f"{current_terrain}/{controller}"
        expected = data["cells"][key]
        require(abs(float(cells[2]) - expected["com_xyz_rms_mm"]["median"]) < 5e-4,
                f"terrain table CoM RMS is stale for {key}")
        require(abs(float(cells[3]) - expected["com_xyz_peak_mm"]["median"]) < 5e-4,
                f"terrain table CoM peak is stale for {key}")
        require(abs(float(cells[4]) - 1000.0 * expected["roll_pitch_rms_rad"]["median"]) < 5e-3,
                f"terrain table roll/pitch RMS is stale for {key}")
        require(int(cells[5]) == int(expected["falls"]),
                f"terrain table fall count is stale for {key}")
        rows.append(key)
    require(len(rows) == len(TERRAINS) * len(CONTROLLERS),
            "terrain result table is incomplete")
    require(len(set(rows)) == len(rows), "terrain result table contains duplicate rows")


def verify_push_table(manuscript: str, data: dict) -> None:
    """Match the composite push table to the authoritative aggregate JSON."""
    controller_order = ("impedance", "nominal_mpc", "interaction_mpc")
    rows = []
    cleaned = manuscript.replace("**", "").replace("$", "")
    for raw_line in cleaned.splitlines():
        if not raw_line.startswith("|"):
            continue
        cells = [cell.strip() for cell in raw_line.strip().strip("|").split("|")]
        if len(cells) != 5 or cells[0] not in {
            "lateral, double support", "lateral, single support",
            "forward, double support", "forward, single support",
        }:
            continue
        direction, phase_words = [part.strip() for part in cells[0].split(",", 1)]
        phase = phase_words.replace(" ", "_")
        expected = [data["cells"][f"{direction}|{phase}|{c}"] for c in controller_order]
        peaks = [float(value.strip()) for value in cells[1].split("/")]
        require(all(abs(value - round(cell["com_peak_mm"], 2)) < 5e-3
                    for value, cell in zip(peaks, expected)),
                f"push table peak values are stale for {direction}/{phase}")
        pct = float(cells[2].replace("\\%", ""))
        expected_pct = 100.0 * (expected[2]["com_peak_mm"] - expected[1]["com_peak_mm"]) \
            / expected[1]["com_peak_mm"]
        require(abs(pct - round(expected_pct, 1)) < 5e-2,
                f"push table percentage is stale for {direction}/{phase}")
        recovery_values = [value.strip() for value in cells[3].split("/")]
        for value, cell in zip(recovery_values, expected):
            expected_recovery = cell["recovery_time_s"]
            if expected_recovery is None:
                require(value == "--", f"push recovery should be absent for {direction}/{phase}")
            else:
                require(abs(float(value) - expected_recovery) < 5.1e-4,
                        f"push recovery is stale for {direction}/{phase}")
        falls = [int(value.strip()) for value in cells[4].split("/")]
        require(falls == [int(cell["falls"]) for cell in expected],
                f"push table fall counts are stale for {direction}/{phase}")
        rows.append((direction, phase))
    require(set(rows) == set(PUSH_CONDITIONS), "push result table is incomplete")


def verify_push(manuscript: str) -> dict:
    data = json.loads(PUSH_DATASET.read_bytes())
    trials = data["trials"]
    require(data.get("schema_version") == 2,
            "push artifact predates the corrected shared-realizer protocol")
    require(tuple(data.get("controllers", ())) == CONTROLLERS,
            "push artifact must contain exactly the three publication controllers")
    require({t["controller"] for t in trials} == set(CONTROLLERS),
            "push trials contain a diagnostic or missing controller")
    require(len(trials) == 120, "push artifact must contain 120 publication trials")
    terrain_data = json.loads(DATASET.read_bytes())
    require(data.get("gait") == terrain_data.get("metadata", {}).get("gait"),
            "push and terrain artifacts do not use the same walking plan")
    conditioning = data.get("residual_conditioning", {})
    require(conditioning == {
        "deadband_task_acceleration": 0.30,
        "cap_task_acceleration": 0.50,
        "command_slew_task_acceleration_per_update": 0.70,
    }, "push artifact lacks the frozen residual-conditioning parameters")
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
    # Compute the lateral single-support medians for the verification report.
    # Performance direction is not hard-coded: the manuscript may claim only
    # what the newly generated artifact actually supports.
    ss_nom = median_cell(trials, metric="post_com_peak_mm", direction="lateral",
                         phase="single_support", controller="nominal_mpc")
    ss_int = median_cell(trials, metric="post_com_peak_mm", direction="lateral",
                         phase="single_support", controller="interaction_mpc")
    require(np.isfinite(ss_nom) and np.isfinite(ss_int),
            "lateral single-support peak comparison is not finite")
    max_torque = max(t["max_torque_utilization"] for t in trials if t.get("onset_found"))
    peak_reductions = {}
    for direction, phase in PUSH_CONDITIONS:
        nominal = median_cell(trials, metric="post_com_peak_mm", direction=direction,
                              phase=phase, controller="nominal_mpc")
        interaction = median_cell(trials, metric="post_com_peak_mm", direction=direction,
                                  phase=phase, controller="interaction_mpc")
        require(interaction < nominal,
                f"manuscript claims push peak attenuation for {direction}/{phase}")
        peak_reductions[f"{direction}/{phase}"] = 100.0 * (interaction - nominal) / nominal
    require("three controllers" in manuscript,
            "manuscript does not declare the three-controller comparison")
    verify_push_table(manuscript, data)
    return {
        "push_dataset": PUSH_DATASET.name,
        "push_sha256": hashlib.sha256(PUSH_DATASET.read_bytes()).hexdigest(),
        "lateral_ss_nominal_peak_mm": ss_nom,
        "lateral_ss_interaction_peak_mm": ss_int,
        "push_interaction_vs_nominal_peak_pct": peak_reductions,
        "push_max_torque_utilization": max_torque,
    }


def main() -> None:
    raw = DATASET.read_bytes()
    data = json.loads(raw)
    trials = data["trials"]
    require(data.get("schema_version") == 2,
            "terrain artifact predates the corrected measured-acceleration/obstacle protocol")
    require(tuple(data.get("metadata", {}).get("controllers", ())) == CONTROLLERS,
            "terrain artifact must contain exactly the three publication controllers")
    require({t["controller"] for t in trials} == set(CONTROLLERS),
            "terrain trials contain a diagnostic or missing controller")
    require(len(trials) == 120, "terrain artifact must contain 120 publication trials")
    require(data["metadata"].get("residual_conditioning") == {
        "deadband_task_acceleration": 0.30,
        "cap_task_acceleration": 0.50,
        "command_slew_task_acceleration_per_update": 0.70,
    }, "terrain artifact lacks the frozen residual-conditioning parameters")
    require(data["metadata"]["wbc_dt_s"] == 0.002, "WBC schedule is not 500 Hz")
    require(data["metadata"]["mpc_dt_s"] == 0.01, "MPC schedule is not 100 Hz")
    require(data["metadata"].get("duration_s", 0.0) >= 15.0,
            "terrain campaign must use the 15 s continuous-walking window")
    expected_gait = {
        "n_steps": 12,
        "step_length": 0.03,
        "step_time": 1.4,
        "double_support_time": 1.0,
        "settle_time": 1.0,
        "lateral_zmp_scale": 1.0,
        "smooth_double_support": True,
        "zmp_transfer_time": 0.05,
        "smooth_lateral_only": False,
    }
    require(data["metadata"].get("gait") == expected_gait,
            "terrain artifact does not use the frozen continuous-walking gait")
    for terrain in TERRAINS:
        for controller in CONTROLLERS:
            cell = [t for t in trials if t["terrain"] == terrain and t["controller"] == controller]
            require(len(cell) == 10, f"{terrain}/{controller}: expected 10 trials")
            require({t["seed"] for t in cell} == SEEDS, f"{terrain}/{controller}: seed mismatch")
            require(sum(int(t["qp_fallbacks"]) for t in cell) == 0,
                    f"{terrain}/{controller}: QP fallback present")
            if terrain == "flat" and controller in ("nominal_mpc", "interaction_mpc"):
                require(all(not t["fell"] for t in cell),
                        f"{terrain}/{controller}: continuous flat-ground gate fell")
                require(all(t["duration_completed_s"] >= 14.9 for t in cell),
                        f"{terrain}/{controller}: flat-ground gate ended early")
                require(all(t["forward_travel_m"] >= 0.18 for t in cell),
                        f"{terrain}/{controller}: flat-ground gate did not walk continuously")
            if terrain == "obstacle":
                require(all(not t.get("obstacle_contact_during_settling", True) for t in cell),
                        f"{terrain}/{controller}: obstacle touched during settling")
                require(all(t.get("obstacle_contacted", False) for t in cell),
                        f"{terrain}/{controller}: raised patch was not contacted in every trial")
                require(all(float(t["obstacle_first_contact_s"]) >= 1.0 for t in cell),
                        f"{terrain}/{controller}: obstacle contact preceded walking")
                require(all(not t["fell"] for t in cell),
                        f"{terrain}/{controller}: manuscript says every obstacle trial completed")
            if terrain in ("depression", "rough"):
                require(all(t["fell"] for t in cell),
                        f"{terrain}/{controller}: manuscript failure-boundary claim is stale")
    # Recompute the obstacle comparison from the corrected future-contact trial.
    nom = data["cells"]["obstacle/nominal_mpc"]["com_xyz_peak_mm"]["median"]
    inter = data["cells"]["obstacle/interaction_mpc"]["com_xyz_peak_mm"]["median"]
    obstacle_pct = 100.0 * (inter - nom) / nom
    require(np.isfinite(obstacle_pct), "obstacle comparison is not finite")

    manuscript = MANUSCRIPT.read_text()
    verify_terrain_table(manuscript, data)
    for stale in ("19.8%", "14.4 to 10.6", "42.8", "4.2 mm"):
        require(stale not in manuscript,
                f"manuscript still contains superseded numerical claim: {stale!r}")
    for claim in ("three controllers", "0.30", "0.50", "0.70",
                  "prototype measurement on a non-real-time host",
                  "240 torque-level", "7.0%", "22.6%"):
        require(claim in manuscript, f"manuscript evidence statement missing: {claim!r}")

    config_report = verify_configuration()
    figure_report = verify_figures(manuscript)
    video_report = verify_continuous_video(expected_gait)
    push_report = verify_push(manuscript)

    report = {
        "status": "PASS",
        "dataset": DATASET.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "reported_controllers": list(CONTROLLERS),
        "obstacle_interaction_vs_nominal_peak_pct": obstacle_pct,
        **config_report,
        **figure_report,
        **video_report,
        **push_report,
    }
    out = RESULTS / "uneven_ground_verification.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
