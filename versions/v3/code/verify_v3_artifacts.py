#!/usr/bin/env python3
"""Verify the currently implemented v3 artifacts.

This is intentionally narrower than the full S1-S5 benchmark plan. It checks
the executable artifacts that currently support the paper draft:

* exact-ZOH normalized MPC construction;
* input-centered disturbance cancellation behavior;
* Unitree G1 MJCF loading;
* H1/H2 representation and fixed-support torque-realizer results;
* canonical no-push dual-MPC root-assisted walking log/summary/video;
* short-push dual-MPC root-assisted walking log/summary/video;
* torque-realizer smoke gates and known gait-extension failure status.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from normalized_mpc import NormalizedMPC


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "g1_wbc.xml"
RESULTS = HERE / "results"


def require(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def load_json(path: Path):
    require(path.exists(), f"missing artifact: {path}")
    require(path.stat().st_size > 0, f"empty artifact: {path}")
    with path.open() as f:
        return json.load(f)


def require_file(path: Path):
    require(path.exists(), f"missing artifact: {path}")
    require(path.stat().st_size > 0, f"empty artifact: {path}")


def verify_normalized_mpc():
    dt = 0.01
    mpc = NormalizedMPC(
        dim=3,
        dt=dt,
        horizon=8,
        q_pos=10.0,
        q_vel=2.0,
        r=0.1,
        qf_pos=20.0,
        qf_vel=4.0,
    )
    A_expected = np.block(
        [
            [np.eye(3), dt * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)],
        ]
    )
    B_expected = np.vstack((0.5 * dt**2 * np.eye(3), dt * np.eye(3)))
    require(np.allclose(mpc.A, A_expected), "normalized MPC A is not exact ZOH")
    require(np.allclose(mpc.B, B_expected), "normalized MPC B is not exact ZOH")

    d_hat = np.array([0.3, -0.2, 0.1])
    u = mpc.solve(np.zeros(6), d_hat=d_hat)
    require(np.allclose(u, -d_hat), "input-centered MPC does not cancel constant d_hat at zero state")
    return {
        "A_shape": list(mpc.A.shape),
        "B_shape": list(mpc.B.shape),
        "zero_state_cancel": u.tolist(),
    }


def verify_model():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    require(model.nq > 0 and model.nv > 0 and model.nu > 0, "G1 model did not load valid dimensions")
    return {"nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu)}


def verify_result_prefix(prefix: str, *, expected_push: bool):
    summary_path = RESULTS / f"{prefix}_summary.json"
    log_path = RESULTS / f"{prefix}_log.npz"
    plot_path = RESULTS / f"{prefix}.png"
    video_path = RESULTS / f"{prefix}_video.mp4"

    for path in (summary_path, log_path, plot_path, video_path):
        require_file(path)

    summary = load_json(summary_path)
    log = dict(np.load(log_path))

    require(summary.get("body_mpc_enabled") is True, "body MPC flag is not true")
    require(summary.get("task_mpc_enabled") is True, "task MPC flag is not true")
    require(summary.get("root_assist_enabled") is True, "root assist flag is not true")
    require(summary.get("push_enabled") is expected_push, f"{prefix} push flag mismatch")
    require(summary.get("passes_visual_demo") is True, "visual demo did not pass")
    require(summary.get("fell") is False, "summary reports a fall")
    require(summary.get("fall_assessment_valid") is False, "root-assisted run must not be marked as valid fall assessment")
    require(abs(summary["duration_s"] - 10.0) < 1e-9, "duration is not the 10 s default")
    require(abs(summary["commanded_distance_m"] - 10.8) < 1e-9, "commanded distance is not the 10.8 m trapezoid default")
    require(abs(summary["cruise_speed_mps"] - 1.2) < 1e-9, "cruise speed is not 1.2 m/s")
    require(abs(summary["accel_ramp_time_s"] - 1.0) < 1e-9, "ramp time is not 1.0 s")
    require("0-1.0 s ramp" in summary["velocity_profile"], "velocity profile metadata is missing the 0-1 s ramp")
    require("1.0-9.0 s cruise" in summary["velocity_profile"], "velocity profile metadata is missing the 1-9 s cruise")
    require(summary["distance_m"] > 10.5, "walking distance is too short for the 10.8 m profile")
    require(min(summary["left_foot_lift_m"], summary["right_foot_lift_m"]) > 0.03, "foot lift is not visible")
    require(summary["max_abs_roll_pitch_rad"] < 0.10, "root-assisted visualization is not upright")

    for key in ("t", "com", "com_ref", "u_body", "u_task", "qpos", "foot_z", "contact", "push_accel"):
        require(key in log, f"missing log key: {key}")
    require(log["qpos"].shape[0] == log["t"].shape[0], "qpos log length mismatch")
    require(log["qpos"].shape[0] >= 9000, "10 s log is unexpectedly short")
    if expected_push:
        require(summary["push_accel_mps2"] > 0.0, "push run reports zero push magnitude")
        require(summary["max_lateral_deviation_m"] > 1e-3, "push run has no lateral deviation")
        require(float(np.max(np.linalg.norm(log["push_accel"][:, :2], axis=1))) > 0.0, "push log contains no disturbance")
    else:
        require(float(np.max(np.linalg.norm(log["push_accel"][:, :2], axis=1))) == 0.0, "no-push log contains disturbance")

    reader = imageio.get_reader(video_path)
    metadata = reader.get_meta_data()
    reader.close()
    require(metadata.get("duration", 0.0) >= 9.5, "video duration is too short")
    require(tuple(metadata.get("size", ())) == (1400, 640), "unexpected video resolution")

    return {
        "prefix": prefix,
        "summary": summary,
        "log_samples": int(log["t"].shape[0]),
        "video": {
            "duration_s": metadata.get("duration"),
            "fps": metadata.get("fps"),
            "size": metadata.get("size"),
        },
    }


def verify_h1_h2_results():
    path = RESULTS / "h1_h2_results.json"
    data = load_json(path)

    h1 = data["H1_structural"]
    for key in (
        "body_A_matches_exact_zoh",
        "body_B_matches_exact_zoh",
        "task_A_matches_exact_zoh",
        "task_B_matches_exact_zoh",
    ):
        require(abs(float(h1[key])) < 1e-12, f"H1 structural error is nonzero: {key}")

    eq = data["H1_equivalence"]
    require(
        float(eq["max_command_diff_normalized_vs_forceinput"]) < 1e-4,
        "H1 normalized/force-input equivalence drifted",
    )

    inv = data["H1_config_invariance"]
    require(inv["predictor_A_B_constant"] is True, "H1 predictor is not marked constant")
    require(max(inv["lambda_t_diag_variation_pct"]) > 1000.0, "Lambda sweep no longer shows large configuration variation")

    h2_rep = data["H2_representation"]
    require(h2_rep["body_offset_reduction_x"] > 50.0, "H2 body representation offset reduction regressed")
    require(h2_rep["task_offset_reduction_x"] > 100.0, "H2 task representation offset reduction regressed")

    h2_full = data["H2_full_realizer"]
    body = h2_full["body_port"]
    task = h2_full["task_port"]
    require(body["fell_no_observer"] is False and body["fell_observer"] is False, "H2 body full-realizer fall")
    require(task["fell_no_observer"] is False and task["fell_observer"] is False, "H2 task full-realizer fall")
    require(body["observer_ss_com_mm"] < body["no_observer_ss_com_mm"], "H2 body observer does not improve CoM error")
    require(task["observer_ss_hand_mm"] < task["no_observer_ss_hand_mm"], "H2 task observer does not improve hand error")

    for figure in ("h1_equivalence.png", "h2_offset_free.png"):
        require_file(RESULTS / figure)

    return {
        "h1_equivalence_max_diff": eq["max_command_diff_normalized_vs_forceinput"],
        "lambda_variation_pct": inv["lambda_t_diag_variation_pct"],
        "h2_body_mm": [body["no_observer_ss_com_mm"], body["observer_ss_com_mm"]],
        "h2_task_mm": [task["no_observer_ss_hand_mm"], task["observer_ss_hand_mm"]],
    }


def verify_torque_aggregate(name: str, *, expected_trials: int, expected_passed: int, expected_fell: int,
                            max_residual: float = 0.5, max_tau_sat: float = 0.5):
    # Fixed-support gates stay well within actuator limits (residual/clipping
    # < 0.5). The dynamic DCM stepping gates saturate the torque limits during
    # single support/swing (tau utilization -> 1), and that clipping shows up as
    # an equal dynamics-equality residual; a looser bound is passed for them.
    path = RESULTS / f"g1_torque_{name}_aggregate.json"
    data = load_json(path)
    require(data["num_trials"] == expected_trials, f"{name}: trial count mismatch")
    require(data["num_passed"] == expected_passed, f"{name}: pass count mismatch")
    require(data["num_fell"] == expected_fell, f"{name}: fall count mismatch")
    require(len(data["trials"]) == expected_trials, f"{name}: trial list length mismatch")
    for trial in data["trials"]:
        require("RandomWalkDisturbanceObserver" in trial["claim_scope"], f"{name}: missing observer claim scope")
        require(trial["max_realizer_residual"] is not None, f"{name}: missing realizer residual")
        require(trial["max_realizer_residual"] < max_residual, f"{name}: realizer residual too large")
        require(trial["max_tau_saturation_norm"] < max_tau_sat, f"{name}: post-QP torque clipping residual too large")
        require(0.0 <= trial["max_tau_limit_utilization"] <= 1.0 + 1e-9, f"{name}: torque utilization out of range")
        require_file(RESULTS / f"g1_torque_{name}_seed{trial['seed']}_summary.json")
        require_file(RESULTS / f"g1_torque_{name}_seed{trial['seed']}_log.npz")
        require_file(RESULTS / f"g1_torque_{name}_seed{trial['seed']}.png")
    return {
        "success_rate": data["success_rate"],
        "median_completed_s": data["median_completed_s"],
        "median_max_roll_pitch_rad": data["median_max_roll_pitch_rad"],
    }


def verify_torque_smoke_results():
    stand = verify_torque_aggregate("stand", expected_trials=1, expected_passed=1, expected_fell=0)
    stand_push = verify_torque_aggregate("stand_push", expected_trials=3, expected_passed=3, expected_fell=0)
    contact_switch = verify_torque_aggregate("contact_switch", expected_trials=1, expected_passed=0, expected_fell=1,
                                              max_residual=1.5, max_tau_sat=1.5)
    walk = verify_torque_aggregate("walk", expected_trials=1, expected_passed=0, expected_fell=1,
                                   max_residual=1.5, max_tau_sat=1.5)

    push_data = load_json(RESULTS / "g1_torque_stand_push_aggregate.json")
    require(all(trial["detected_push"] for trial in push_data["trials"]), "not all stand-push trials detected the push")

    contact_data = load_json(RESULTS / "g1_torque_contact_switch_aggregate.json")
    walk_data = load_json(RESULTS / "g1_torque_walk_aggregate.json")
    # With the DCM (capture-point) reference unified into the smoke test, the
    # stepping gates no longer tip immediately in single support; they now carry
    # the faithful recovery through several contact-mode switches before the
    # single-support bandwidth limit is reached. They still fall (fell=1 above).
    require(contact_data["trials"][0]["duration_completed_s"] > 1.5, "contact-switch (DCM ref) fall time regressed; update paper text")
    require(contact_data["trials"][0]["contact_switches_detected"] >= 4, "contact-switch DCM stepping regressed")
    require(walk_data["trials"][0]["duration_completed_s"] > 1.5, "walk (DCM ref) fall time regressed; update paper text")
    require(walk_data["trials"][0]["contact_switches_detected"] >= 6, "walk DCM stepping regressed")

    return {
        "stand": stand,
        "stand_push": stand_push,
        "contact_switch": contact_switch,
        "walk": walk,
    }


def verify_gait_extension_results():
    faithful = load_json(RESULTS / "gait_faithful_summary.json")
    dcm = load_json(RESULTS / "gait_dcm_summary.json")
    require_file(RESULTS / "gait_faithful.png")
    require_file(RESULTS / "gait_dcm.png")

    require(faithful["fell"] is True, "faithful gait unexpectedly stopped falling; update paper text")
    require(faithful["contact_switches"] >= 4, "faithful gait no longer reaches four contact switches")
    require(faithful["min_pelvis_height_m"] > 0.70, "faithful gait fall mode changed from roll/pitch to height")

    require(dcm["fell"] is True, "DCM gait unexpectedly stopped falling; update paper text")
    require(dcm["contact_switches"] >= 6, "DCM gait contact switching regressed")
    require(dcm["min_pelvis_height_m"] < 0.46, "DCM gait fall mode changed from height threshold")

    # CoP/DCM stabilizer: implemented, isolates the single-support authority limit
    # (ankle CoP saturates on the wide stance); still does not sustain walking.
    stab = load_json(RESULTS / "gait_dcm_stab_summary.json")
    require_file(RESULTS / "gait_dcm_stab.png")
    require(stab["fell"] is True, "DCM stabilizer unexpectedly sustains walking; update paper text")
    require(stab["completed_full_plan"] is False, "DCM stabilizer now completes the plan; update paper text")

    # Full walker: hip strategy + capture-point step adaptation + initiation.
    # Implemented; still does not sustain walking (a few adapted steps, then falls).
    walk = load_json(RESULTS / "gait_walk_summary.json")
    require_file(RESULTS / "gait_walk.png")
    require(walk["fell"] is True, "full walker unexpectedly sustains walking; update paper text")
    require(walk["completed_full_plan"] is False, "full walker now completes the plan; update paper text")

    return {"faithful": faithful, "dcm": dcm, "dcm_stab": stab, "walk": walk}


def verify_h3_h4_results():
    # H3: coupled arm-reaction preview reduces the peak CoM excursion vs split.
    h3 = load_json(RESULTS / "h3_coupling_summary.json")
    require_file(RESULTS / "h3_coupling.png")
    require(h3["coupled"]["peak_com_mm"] < h3["split"]["peak_com_mm"],
            "H3 coupled preview no longer reduces the CoM transient")
    require(h3["peak_com_reduction_x"] >= 1.5, "H3 CoM-transient reduction regressed below 1.5x")

    # H4: all scripted contact events detected, no misses or false positives.
    h4 = load_json(RESULTS / "h4_detection_summary.json")
    require_file(RESULTS / "h4_detection.png")
    require(h4["missed"] == 0, "H4 missed a contact event")
    require(h4["false_positives"] == 0, "H4 produced a false positive")
    require(h4["detected"] == h4["true_events"], "H4 detection count != oracle event count")
    require(h4["mean_latency_ms"] is not None and h4["mean_latency_ms"] < 150.0,
            "H4 detection latency regressed")

    # H5: constraints live in recovery. Constrained recovery keeps the recovered
    # wrench/torque feasible and stands; unconstrained recovery violates both and
    # falls.
    h5 = load_json(RESULTS / "h5_constraints_summary.json")
    require_file(RESULTS / "h5_constraints.png")
    con, unc = h5["constrained"], h5["unconstrained"]
    require(con["fell"] is False, "H5 constrained recovery now falls; update paper text")
    require(unc["fell"] is True, "H5 unconstrained recovery no longer falls; update paper text")
    require(con["max_friction_violation_N"] < 5.0 and con["max_torque_violation_Nm"] < 5.0,
            "H5 constrained recovery now violates the constraints")
    require(unc["max_friction_violation_N"] > 100.0 and unc["max_torque_violation_Nm"] > 100.0,
            "H5 unconstrained recovery no longer produces large violations")

    # H5 statistics: over 50 randomized pushes, constrained recovery stays
    # feasible and stands every time; unconstrained recovery falls every time.
    h5s = load_json(RESULTS / "h5_constraints_stats.json")
    require(h5s["n_seeds"] >= 50, "H5 statistics ran fewer than 50 seeds")
    require(h5s["constrained"]["success_rate"] >= 0.98,
            "H5 constrained recovery no longer stands across randomized pushes")
    require(h5s["unconstrained"]["fall_rate"] >= 0.98,
            "H5 unconstrained recovery no longer falls across randomized pushes")
    require(h5s["constrained"]["friction_violation_N"]["max"] < 5.0
            and h5s["constrained"]["torque_violation_Nm"]["max"] < 5.0,
            "H5 constrained recovery now violates constraints across randomized pushes")

    # H3 also contrasts the internal-momentum form: the unified whole-body QP
    # compensates an equal-magnitude internal arm swing natively, so its
    # uncompensated transient is much smaller than the external load's.
    require("internal" in h3 and "external" in h3, "H3 no longer reports both preview forms")
    require(h3["internal"]["split"]["peak_com_mm"] < h3["external"]["split"]["peak_com_mm"],
            "H3 internal arm-momentum transient is no longer smaller than the external-load transient")

    # H6: interaction layer on a moving base. Previewing the planned lateral load
    # keeps the CoM on the base's lateral reference (large reduction), while the
    # base's own forward weight-shift is tracked equally with or without the layer.
    h6 = load_json(RESULTS / "h6_onbase_summary.json")
    require_file(RESULTS / "h6_onbase.png")
    require(h6["layer_off"]["fell"] is False and h6["layer_on"]["fell"] is False,
            "H6 fell; the moving-base demo must stay in double support")
    require(h6["lat_rms_reduction_x"] >= 1.5, "H6 lateral load-rejection benefit regressed below 1.5x")
    require(abs(h6["layer_on"]["rms_fwd_mm"] - h6["layer_off"]["rms_fwd_mm"]) < 3.0,
            "H6 layer now disturbs the base's own forward motion")
    return {"h3": h3, "h4": h4, "h5": h5, "h6": h6}


def verify_results():
    paper_alias = RESULTS / "g1_walk_10s_1p2ms.png"
    require_file(paper_alias)
    canonical = RESULTS / "g1_walk_10s.png"
    require_file(canonical)
    require(
        paper_alias.read_bytes() == canonical.read_bytes(),
        "paper walking figure alias differs from canonical regenerated plot",
    )
    return {
        "no_push": verify_result_prefix("g1_walk_10s", expected_push=False),
        "short_push": verify_result_prefix("g1_walk_10s_push", expected_push=True),
        "h1_h2": verify_h1_h2_results(),
        "h3_h4": verify_h3_h4_results(),
        "torque_smoke": verify_torque_smoke_results(),
        "gait_extension": verify_gait_extension_results(),
    }


def main():
    report = {
        "normalized_mpc": verify_normalized_mpc(),
        "model": verify_model(),
        "results": verify_results(),
    }
    print(json.dumps(report, indent=2))
    print("PASS: v3 implemented artifacts are present and internally consistent.")


if __name__ == "__main__":
    main()
