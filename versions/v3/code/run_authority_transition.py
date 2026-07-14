#!/usr/bin/env python3
"""Authority-gated double--single--double support development test.

The actual support mode changes only after the candidate single-support
authority set is nonempty for a measured-state dwell.  The canonical MPC pair
stays fixed; only its asymmetric residual-command bounds are updated.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from normalized_mpc import NormalizedMPC
from realization_authority import (
    AnalyticAuthorityMapper, ContinuationAuthorityEstimator,
    ExactResidualBisectionEstimator,
)
from run_authority_benchmarks import settle_model
from run_g1_torque_realizer_benchmark import (
    SIM_DT,
    TORQUE_STAND_CTRL,
    InverseDynamicsQPRealizer,
    com_velocity,
    hand_state,
    measured_foot_contacts,
    robot_com,
    roll_pitch_yaw_from_body,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
FIGURES = HERE.parent / "figures"
FIGURES.mkdir(exist_ok=True)


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def run_cycle(
    duration: float = 4.0,
    seed: int | None = None,
    authority: str = "analytic",
) -> tuple[dict, dict[str, np.ndarray]]:
    model, data, torso, hand_sid, height_ref = settle_model()
    realizer = InverseDynamicsQPRealizer(model, exact_realizer=True)
    estimator = ExactResidualBisectionEstimator()
    mapper = AnalyticAuthorityMapper()
    continuation = ContinuationAuthorityEstimator(max_regions=60)
    snapshot = None          # published by the 1 kHz realizer, consumed next cycle
    analytic = (authority == "analytic")
    contin = (authority == "continuation")
    rng = np.random.default_rng(seed)
    friction_scale = 1.0 if seed is None else float(rng.uniform(0.95, 1.05))
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if "ankle" in name or "foot" in name:
            model.geom_friction[gid, 0] *= friction_scale
    initial_lateral_velocity = (
        0.0 if seed is None else float(rng.normal(0.0, 1.5e-3))
    )
    data.qvel[1] += initial_lateral_velocity
    if seed is not None:
        data.qvel[realizer.dof] += rng.normal(0.0, 3.0e-4, size=realizer.nu)
    mujoco.mj_forward(model, data)
    lift_duration = 0.24 if seed is None else float(rng.uniform(0.23, 0.25))
    lift_height = 0.008 if seed is None else float(rng.uniform(0.0075, 0.0085))
    landing_preload = 0.012 if seed is None else float(rng.uniform(0.011, 0.013))
    mpc_dt = 0.04
    mpc = NormalizedMPC(
        dim=2, dt=mpc_dt, horizon=10,
        q_pos=70.0, q_vel=2.0, r=0.02,
    )
    A0, B0, H0 = mpc.A.copy(), mpc.B.copy(), mpc.H.copy()

    com0 = robot_com(model, data)
    vel0 = com_velocity(model, data, realizer.root_body)
    del vel0
    left_site = realizer.foot_site["left"]
    right_site = realizer.foot_site["right"]
    left0 = data.site_xpos[left_site].copy()
    right0 = data.site_xpos[right_site].copy()
    qd_ref = np.zeros(model.nu)

    stance: tuple[str, ...] = ("left", "right")
    contacts = realizer.contact_points(model, data, stance)
    stance_targets = {key: pos.copy() for key, (pos, _) in contacts.items()}
    candidate_contacts = realizer.contact_points(model, data, ("left",))
    candidate_targets = {key: pos.copy() for key, (pos, _) in candidate_contacts.items()}

    steps = int(round(duration / SIM_DT))
    stride = max(1, int(round(mpc_dt / SIM_DT)))
    log = {
        "t": np.zeros(steps),
        "phase": np.zeros(steps, dtype=int),
        "contact": np.zeros((steps, 2), dtype=int),
        "com": np.zeros((steps, 3)),
        "com_error": np.zeros((steps, 2)),
        "request": np.zeros((steps, 2)),
        "realized": np.zeros((steps, 2)),
        "residual": np.zeros((steps, 2)),
        "authority_lower": np.zeros((steps, 2)),
        "authority_upper": np.zeros((steps, 2)),
        "authority_valid": np.zeros(steps, dtype=int),
        "candidate_ss_valid": np.zeros(steps, dtype=int),
        "candidate_ss_area": np.zeros(steps),
        "candidate_ss_residual": np.zeros((steps, 2)),
        "rpy": np.zeros((steps, 3)),
        "foot_z": np.zeros((steps, 2)),
        "fallback": np.zeros(steps, dtype=int),
    }

    phase = 0  # 0 prepare DS, 1 lift/SS, 2 lower, 3 recovered DS
    phase_start = 0.0
    command = np.zeros(2)
    authority_lower = np.zeros(2)
    authority_upper = np.zeros(2)
    authority_valid = False
    candidate_valid = False
    candidate_area = 0.0
    candidate_residual = np.full(2, np.inf)
    q_ref = TORQUE_STAND_CTRL.copy()
    ready_updates = 0
    ready_required = 3
    lift_authorized_time: float | None = None
    touchdown_time: float | None = None
    measured_ss_start: float | None = None
    measured_ss_end: float | None = None
    touchdown_contact_run = 0
    touchdown_contact_start: float | None = None
    touchdown_dwell_samples = max(1, int(round(0.020 / SIM_DT)))
    active_query_times: list[float] = []
    total_update_times: list[float] = []
    active_solve_counts: list[int] = []
    candidate_solve_counts: list[int] = []
    authority_update_valid: list[bool] = []
    command_bound_violations: list[float] = []
    invalid_updates = 0
    fell = False
    end_index = steps

    for k in range(steps):
        t = k * SIM_DT
        com = robot_com(model, data)
        vel = com_velocity(model, data, realizer.root_body)
        rpy = roll_pitch_yaw_from_body(data, torso)
        measured_before = measured_foot_contacts(model, data)

        if phase == 0:
            alpha = smoothstep((t - 0.10) / 1.25)
            y_ref = (1.0 - alpha) * com0[1] + alpha * left0[1]
            swing = None
            knee_bias = 0.0
        elif phase == 1:
            elapsed = t - phase_start
            y_ref = left0[1]
            s = np.clip(elapsed / lift_duration, 0.0, 1.0)
            swing = {
                "sid": right_site,
                "pos_des": right0 + np.array([0.0, 0.0, lift_height * np.sin(np.pi * s)]),
                "vel_des": np.array([
                    0.0, 0.0,
                    lift_height * np.pi / lift_duration * np.cos(np.pi * s),
                ]),
                "kp": 240.0,
                "kd": 32.0,
                "weight": 55.0,
            }
            knee_bias = 0.05 * np.sin(np.pi * s)
            if elapsed >= lift_duration:
                phase = 2
                phase_start = t
        elif phase == 2:
            elapsed = t - phase_start
            y_ref = 0.88 * left0[1] + 0.12 * com0[1]
            target = right0.copy()
            target[2] -= 0.030
            swing = {
                "sid": right_site,
                "pos_des": target,
                "vel_des": np.zeros(3),
                "kp": 400.0,
                "kd": 50.0,
                "weight": 400.0,
            }
            knee_bias = -0.18
            if measured_before[1] and elapsed >= 0.08:
                if touchdown_contact_run == 0:
                    touchdown_contact_start = t
                touchdown_contact_run += 1
            else:
                touchdown_contact_run = 0
                touchdown_contact_start = None
            if touchdown_contact_run >= touchdown_dwell_samples:
                touchdown_time = touchdown_contact_start
                measured_ss_end = touchdown_time
                phase = 3
                phase_start = t
                stance = ("left", "right")
                contacts = realizer.contact_points(model, data, stance)
                stance_targets = {key: pos.copy() for key, (pos, _) in contacts.items()}
                for key in stance_targets:
                    if key.split("_", 1)[0] == "right":
                        stance_targets[key][2] -= landing_preload
                swing = None
                knee_bias = 0.0
        else:
            beta = smoothstep((t - phase_start) / 0.65)
            y_ref = (1.0 - beta) * left0[1] + beta * com0[1]
            swing = None
            knee_bias = 0.0

        # Contact targets remain fixed in the world while contact Jacobians
        # and current corner positions must be refreshed from the measured
        # configuration at every torque-level realization step.
        contacts = realizer.contact_points(model, data, stance)

        error = np.array([com[0] - com0[0], com[1] - y_ref])
        _, hand_vel, hand_jac = hand_state(model, data, hand_sid)
        task_acc = -4.0 * hand_vel

        if k % stride == 0:
            q_ref = TORQUE_STAND_CTRL.copy()
            q_ref[9] += knee_bias
            q_ref[10] -= 0.45 * knee_bias
            update_tic = time.perf_counter()
            active_tic = time.perf_counter()
            if contin:
                # PWA continuation on the realizer's CURRENT solve: walks adjacent
                # critical regions with KKT solves only -- zero extra whole-body QPs.
                cbox = continuation.estimate(realizer, model, data)
                if cbox.valid:
                    authority_lower, authority_upper = cbox.lower.copy(), cbox.upper.copy()
                    authority_valid = True
                else:
                    authority_lower = np.array([-1.5, -2.0])
                    authority_upper = np.array([1.5, 2.0])
                    authority_valid = False
                mpc.update_input_box(authority_lower, authority_upper)
                active_query_times.append(time.perf_counter() - active_tic)
                active_solve_counts.append(0)     # zero whole-body QP solves
            elif analytic:
                # The 1 kHz loop already published this; the predictor consumes the
                # latest snapshot and never blocks.  Zero extra whole-body QPs.
                if snapshot is not None and snapshot.valid and snapshot.contact_mode == stance:
                    authority_lower, authority_upper = snapshot.axis_extent()
                    authority_valid = True
                    mpc.update_input_polytope(snapshot.H_body, snapshot.h_body)
                else:
                    authority_lower = np.array([-1.5, -2.0])
                    authority_upper = np.array([1.5, 2.0])
                    authority_valid = False
                    mpc.update_input_box(authority_lower, authority_upper)
                active_query_times.append(time.perf_counter() - active_tic)
                active_solve_counts.append(0)
            else:
                box = estimator.estimate(
                    realizer, model, data,
                    q_ref=q_ref, qd_ref=qd_ref,
                    task_acc_des=task_acc, hand_jac=hand_jac,
                    stance_contacts=contacts, stance_targets=stance_targets,
                    base_height_ref=height_ref, rpy=rpy,
                    nominal_com_acc_des=np.zeros(3), swing_task=swing,
                    attitude_weight=60.0,
                    centroidal_moment_des=np.zeros(3),
                )
                active_query_times.append(time.perf_counter() - active_tic)
                active_solve_counts.append(box.solve_count)
                authority_lower, authority_upper = box.lower.copy(), box.upper.copy()
                authority_valid = box.valid
                mpc.update_input_box(authority_lower, authority_upper)
            authority_update_valid.append(authority_valid)
            invalid_updates += int(not authority_valid)
            command = mpc.solve(np.r_[error, vel[:2]])
            command_bound_violations.append(float(np.max(np.maximum(
                authority_lower - command,
                command - authority_upper,
            ))))

            if phase == 0:
                candidate_contacts = realizer.contact_points(model, data, ("left",))
                if analytic or contin:
                    # One extra whole-body QP in the CANDIDATE mode, then one KKT
                    # solve (or a KKT walk) -- not a 62-QP search.
                    realizer.command(
                        model, data, q_ref, qd_ref, np.zeros(2), task_acc, hand_jac,
                        candidate_contacts, candidate_targets, height_ref, rpy,
                        com_acc_des=np.zeros(3), swing_task=None,
                        attitude_weight=60.0, centroidal_moment_des=np.zeros(3),
                    )
                    cand_snap = mapper.snapshot(realizer, model, data,
                                                timestamp=t, contact_mode=("left",))
                    c_lo, c_hi = cand_snap.axis_extent()
                    candidate_valid = bool(
                        cand_snap.valid
                        and np.min(0.5 * (c_hi - c_lo)) >= 0.04
                        and np.max(np.abs(cand_snap.nominal_residual)) <= 0.35
                    )
                    candidate_area = float(np.prod(np.maximum(c_hi - c_lo, 0.0)))
                    candidate_residual = cand_snap.nominal_residual.copy()
                    candidate_solve_counts.append(1)
                else:
                    candidate_box = estimator.estimate(
                        realizer, model, data,
                        q_ref=q_ref, qd_ref=qd_ref,
                        task_acc_des=task_acc, hand_jac=hand_jac,
                        stance_contacts=candidate_contacts,
                        stance_targets=candidate_targets,
                        base_height_ref=height_ref, rpy=rpy,
                        nominal_com_acc_des=np.zeros(3), swing_task=None,
                        attitude_weight=60.0,
                        centroidal_moment_des=np.zeros(3),
                    )
                    candidate_valid = bool(
                        candidate_box.valid
                        and np.min(candidate_box.radius) >= 0.04
                        and np.max(np.abs(candidate_box.nominal_residual)) <= 0.35
                    )
                    candidate_area = candidate_box.area
                    candidate_residual = candidate_box.nominal_residual.copy()
                    candidate_solve_counts.append(candidate_box.solve_count)
                state_ready = bool(
                    abs(com[1] - left0[1]) < 0.035
                    and abs(vel[1]) < 0.16
                    and np.max(np.abs(rpy[:2])) < 0.22
                )
                ready_updates = ready_updates + 1 if candidate_valid and state_ready else 0
                if ready_updates >= ready_required:
                    lift_authorized_time = t
                    phase = 1
                    phase_start = t
                    stance = ("left",)
                    contacts = candidate_contacts
                    stance_targets = candidate_targets.copy()
                    # The phase decision occurs after the per-sample swing task
                    # was assembled.  Supply an explicit initial task so the
                    # released foot is never unconstrained for one MPC period.
                    swing = {
                        "sid": right_site,
                        "pos_des": right0.copy(),
                        "vel_des": np.zeros(3),
                        "kp": 280.0,
                        "kd": 38.0,
                        "weight": 90.0,
                    }
            else:
                candidate_valid = False
                candidate_area = np.nan
                candidate_residual = np.full(2, np.nan)
            # This is the complete queried update cost.  In preparation it
            # includes both the active DS set and candidate left-SS set.
            total_update_times.append(time.perf_counter() - update_tic)
            q_ref[1] += 0.035 * command[1]
            q_ref[7] += 0.035 * command[1]

        realizer.command(
            model, data, q_ref, qd_ref, command, task_acc, hand_jac,
            contacts, stance_targets, height_ref, rpy,
            com_acc_des=np.r_[command, 0.0], swing_task=swing,
            attitude_weight=60.0,
            centroidal_moment_des=np.zeros(3),
        )
        Jcom = np.zeros((3, realizer.nv))
        mujoco.mj_jacSubtreeCom(model, data, Jcom, realizer.root_body)
        realized = Jcom[:2] @ realizer.last_qdd
        if analytic:
            snapshot = mapper.snapshot(realizer, model, data,
                                       timestamp=t, contact_mode=stance)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        measured = measured_foot_contacts(model, data)
        if (
            lift_authorized_time is not None
            and measured.sum() == 1
            and measured_ss_start is None
        ):
            measured_ss_start = data.time
        rpy_after = roll_pitch_yaw_from_body(data, torso)
        fell = bool(data.qpos[2] < 0.45 or np.max(np.abs(rpy_after[:2])) > 0.85)

        log["t"][k] = data.time
        log["phase"][k] = phase
        log["contact"][k] = measured
        log["com"][k] = com
        log["com_error"][k] = error
        log["request"][k] = command
        log["realized"][k] = realized
        log["residual"][k] = realized - command
        log["authority_lower"][k] = authority_lower
        log["authority_upper"][k] = authority_upper
        log["authority_valid"][k] = int(authority_valid)
        log["candidate_ss_valid"][k] = int(candidate_valid)
        log["candidate_ss_area"][k] = candidate_area
        log["candidate_ss_residual"][k] = candidate_residual
        log["rpy"][k] = rpy_after
        log["foot_z"][k] = [data.site_xpos[left_site, 2], data.site_xpos[right_site, 2]]
        log["fallback"][k] = int(realizer.last_fallback)

        if fell:
            end_index = k + 1
            break
        if phase == 3 and t - phase_start >= 0.75:
            end_index = k + 1
            break

    for key in log:
        log[key] = log[key][:end_index]
    if measured_ss_start is not None:
        ss_end = measured_ss_end if measured_ss_end is not None else float(log["t"][-1])
        measured_ss_duration = max(0.0, ss_end - measured_ss_start)
    else:
        measured_ss_duration = 0.0
    post_ds_duration = 0.0
    continuous_post_ds_duration = 0.0
    longest_post_ds_duration = 0.0
    post_ds_duty_fraction = 0.0
    if touchdown_time is not None:
        after = log["t"] >= touchdown_time
        ds = np.sum(log["contact"], axis=1) == 2
        post_ds_duration = float(SIM_DT * np.sum(after & ds))
        post_ds_duty_fraction = float(np.mean(ds[after]))
        longest_samples = current_samples = 0
        for is_ds in ds[after]:
            current_samples = current_samples + 1 if is_ds else 0
            longest_samples = max(longest_samples, current_samples)
        longest_post_ds_duration = float(SIM_DT * longest_samples)
        suffix_samples = 0
        for is_ds, is_after in zip(ds[::-1], after[::-1]):
            if not is_after or not is_ds:
                break
            suffix_samples += 1
        continuous_post_ds_duration = float(SIM_DT * suffix_samples)
    residual_inf = np.max(np.abs(log["residual"]), axis=1)
    valid_samples = log["authority_valid"].astype(bool)
    summary = {
        "test": "authority_gated_DS_leftSS_DS",
        "authority_source": authority,
        "seed": seed,
        "friction_scale": friction_scale,
        "initial_lateral_velocity_mps": initial_lateral_velocity,
        "lift_duration_s": lift_duration,
        "lift_height_m": lift_height,
        "landing_preload_m": landing_preload,
        "duration_s": float(log["t"][-1]),
        "mpc_period_s": mpc_dt,
        "lift_authorized": lift_authorized_time is not None,
        "lift_authorized_time_s": lift_authorized_time,
        "touchdown_time_s": touchdown_time,
        "measured_single_support_start_s": measured_ss_start,
        "measured_single_support_end_s": measured_ss_end,
        "measured_single_support_duration_s": measured_ss_duration,
        "post_touchdown_double_support_s": post_ds_duration,
        "post_touchdown_double_support_duty_fraction": post_ds_duty_fraction,
        "longest_post_touchdown_double_support_s": longest_post_ds_duration,
        "continuous_final_double_support_s": continuous_post_ds_duration,
        "fell": fell,
        "qp_fallbacks": int(np.sum(log["fallback"])),
        "invalid_authority_updates": invalid_updates,
        "max_roll_pitch_rad": float(np.max(np.abs(log["rpy"][:, :2]))),
        "max_right_foot_lift_m": float(np.max(log["foot_z"][:, 1] - right0[2])),
        "rms_planar_error_mm": float(1000.0 * np.sqrt(np.mean(np.sum(log["com_error"] ** 2, axis=1)))),
        "median_realization_residual_inf_mps2": float(np.median(residual_inf)),
        "max_realization_residual_inf_mps2": float(np.max(residual_inf)),
        "residual_tolerance_violation_fraction": float(np.mean(residual_inf > 0.35)),
        "valid_authority_residual_violation_fraction": (
            0.0 if not np.any(valid_samples)
            else float(np.mean(residual_inf[valid_samples] > 0.35))
        ),
        "authority_valid_update_fraction": float(np.mean(authority_update_valid)),
        "max_mpc_bound_violation_mps2": float(max(command_bound_violations)),
        "all_mpc_commands_inside_queried_bounds": bool(
            max(command_bound_violations) <= 1e-6
        ),
        "active_query_time_ms_median": float(1000.0 * np.median(active_query_times)),
        "active_query_time_ms_max": float(1000.0 * np.max(active_query_times)),
        "total_constraint_update_time_ms_median": float(1000.0 * np.median(total_update_times)),
        "total_constraint_update_time_ms_max": float(1000.0 * np.max(total_update_times)),
        "active_query_solve_count_median": float(np.median(active_solve_counts)),
        "active_query_solve_count_max": int(np.max(active_solve_counts)),
        "candidate_query_solve_count_median": (
            None if not candidate_solve_counts
            else float(np.median(candidate_solve_counts))
        ),
        "canonical_matrices_bitwise_invariant": bool(
            np.array_equal(A0, mpc.A) and np.array_equal(B0, mpc.B)
            and np.array_equal(H0, mpc.H)
        ),
    }
    summary["passed"] = bool(
        summary["lift_authorized"]
        and summary["touchdown_time_s"] is not None
        and measured_ss_duration >= 0.20
        and post_ds_duty_fraction >= 0.95
        and longest_post_ds_duration >= 0.20
        and continuous_post_ds_duration >= 0.20
        and not fell
        and summary["qp_fallbacks"] == 0
        and summary["all_mpc_commands_inside_queried_bounds"]
        and summary["canonical_matrices_bitwise_invariant"]
    )
    return summary, log


def plot_result(summary: dict, log: dict[str, np.ndarray], path: Path) -> None:
    t = log["t"]
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 8.5), sharex=True)
    axes[0].plot(t, 1000.0 * log["com_error"][:, 1], label=r"$e_y$")
    axes[0].plot(t, 1000.0 * (log["com"][:, 1] - log["com"][0, 1]), label="CoM lateral shift")
    axes[0].set_ylabel("mm")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, log["request"][:, 1], label=r"requested $u_y$")
    axes[1].plot(t, log["realized"][:, 1], label=r"realized $u_y$")
    axes[1].fill_between(t, log["authority_lower"][:, 1], log["authority_upper"][:, 1], alpha=0.18, label="queried authority")
    axes[1].set_ylabel(r"m/s$^2$")
    axes[1].legend(fontsize=8)
    axes[2].plot(t, log["candidate_ss_area"], label="candidate SS box area")
    axes[2].step(t, log["candidate_ss_valid"], where="post", label="candidate valid")
    axes[2].step(t, log["authority_valid"], where="post", label="active-mode valid", alpha=0.8)
    axes[2].set_ylabel("authority")
    axes[2].legend(fontsize=8)
    axes[3].step(t, log["contact"][:, 0], where="post", label="left contact")
    axes[3].step(t, log["contact"][:, 1], where="post", label="right contact")
    axes[3].plot(t, 20.0 * (log["foot_z"][:, 1] - log["foot_z"][0, 1]), label="right lift x20")
    axes[3].set_ylabel("contact/lift")
    axes[3].set_xlabel("time [s]")
    axes[3].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("Measured authority-gated DS--left SS--DS transition")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--authority", choices=("analytic", "continuation", "exact"),
                        default="analytic")
    args = parser.parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be positive")
    if args.trials == 1:
        summary, log = run_cycle(args.duration, seed=args.seed, authority=args.authority)
        summary_path = RESULTS / "authority_transition_summary.json"
        log_path = RESULTS / "authority_transition_log.npz"
        plot_path = RESULTS / "authority_transition.png"
        summary_path.write_text(json.dumps(summary, indent=2))
        np.savez_compressed(log_path, **log)
        plot_result(summary, log, plot_path)
        plot_result(summary, log, FIGURES / "authority_transition.png")
        print(json.dumps(summary, indent=2))
        return

    first_seed = 100 if args.seed is None else args.seed
    rows = []
    for offset in range(args.trials):
        seed = first_seed + offset
        summary, log = run_cycle(args.duration, seed=seed, authority=args.authority)
        rows.append(summary)
        stem = f"authority_transition_seed{seed}"
        (RESULTS / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
        np.savez_compressed(RESULTS / f"{stem}.npz", **log)
        plot_result(summary, log, RESULTS / f"{stem}.png")
    aggregate = {
        "test": "perturbed_authority_gated_DS_leftSS_DS",
        "first_seed": first_seed,
        "trials": args.trials,
        "passed": int(sum(row["passed"] for row in rows)),
        "success_rate": float(np.mean([row["passed"] for row in rows])),
        "fall_rate": float(np.mean([row["fell"] for row in rows])),
        "fallback_total": int(sum(row["qp_fallbacks"] for row in rows)),
        "median_single_support_s": float(np.median([
            row["measured_single_support_duration_s"] for row in rows
        ])),
        "median_continuous_final_double_support_s": float(np.median([
            row["continuous_final_double_support_s"] for row in rows
        ])),
        "minimum_post_touchdown_double_support_duty_fraction": float(min(
            row["post_touchdown_double_support_duty_fraction"] for row in rows
        )),
        "minimum_longest_post_touchdown_double_support_s": float(min(
            row["longest_post_touchdown_double_support_s"] for row in rows
        )),
        "minimum_continuous_final_double_support_s": float(min(
            row["continuous_final_double_support_s"] for row in rows
        )),
        "median_rms_planar_error_mm": float(np.median([
            row["rms_planar_error_mm"] for row in rows
        ])),
        "max_roll_pitch_rad": float(max(row["max_roll_pitch_rad"] for row in rows)),
        "max_realization_residual_inf_mps2": float(max(
            row["max_realization_residual_inf_mps2"] for row in rows
        )),
        "median_active_query_time_ms": float(np.median([
            row["active_query_time_ms_median"] for row in rows
        ])),
        "median_authority_valid_update_fraction": float(np.median([
            row["authority_valid_update_fraction"] for row in rows
        ])),
        "median_valid_authority_residual_violation_fraction": float(np.median([
            row["valid_authority_residual_violation_fraction"] for row in rows
        ])),
        "maximum_command_bound_violation_mps2": float(max(
            row["max_mpc_bound_violation_mps2"] for row in rows
        )),
        "all_canonical_matrices_invariant": bool(all(
            row["canonical_matrices_bitwise_invariant"] for row in rows
        )),
        "trial_summaries": rows,
    }
    path = RESULTS / f"authority_transition_trials_seed{first_seed}_n{args.trials}.json"
    path.write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
