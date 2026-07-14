#!/usr/bin/env python3
"""Regenerate the realization-informed authority experiments for wbc_v3.

The benchmark intentionally evaluates the new claim only: the normalized
predictor keeps one constant exact-ZOH pair while its residual-command box is
recomputed from the present whole-body realization problem.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from normalized_mpc import NormalizedMPC
from realization_authority import AuthorityBox, PlanarBodyAuthorityEstimator
from run_g1_torque_realizer_benchmark import (
    ACTUATED_JOINT_NAMES,
    SIM_DT,
    TORQUE_STAND_CTRL,
    InverseDynamicsQPRealizer,
    body_id,
    com_velocity,
    generate_torque_model,
    hand_state,
    joint_id,
    robot_com,
    roll_pitch_yaw_from_body,
    site_id,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE.parent / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def settle_model(task_weight: float | None = None):
    model = mujoco.MjModel.from_xml_path(str(generate_torque_model()))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for value, name in zip(TORQUE_STAND_CTRL, ACTUATED_JOINT_NAMES):
        jid = joint_id(model, name)
        data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)

    torso = body_id(model, "torso_link")
    hand_sid = site_id(model, "right_hand_site")
    qd_ref = np.zeros(model.nu)
    com_ref = robot_com(model, data)
    height_ref = float(data.qpos[2])
    warm = InverseDynamicsQPRealizer(model, exact_realizer=False)
    # The warm-up must use the SAME controller configuration as the experiment
    # that follows.  Settling with the body-only task weight and then evaluating
    # a task port at a much higher weight is two different controllers in one
    # run, and it changes the reported offset reduction by ~10x.
    if task_weight is not None:
        warm.task_weight = float(task_weight)
    for _ in range(int(round(0.35 / SIM_DT))):
        contacts = warm.contact_points(model, data, ("left", "right"))
        targets = {key: pos.copy() for key, (pos, _) in contacts.items()}
        _, _, hand_jac = hand_state(model, data, hand_sid)
        rpy = roll_pitch_yaw_from_body(data, torso)
        acc = -25.0 * (robot_com(model, data) - com_ref)
        acc -= 8.0 * com_velocity(model, data, warm.root_body)
        warm.command(
            model, data, TORQUE_STAND_CTRL.copy(), qd_ref, np.zeros(2),
            np.zeros(3), hand_jac, contacts, targets, height_ref, rpy,
            com_acc_des=np.clip(acc, -3.0, 3.0), attitude_weight=60.0,
        )
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
    data.time = 0.0
    return model, data, torso, hand_sid, height_ref


def scenario_context(model, data, torso, hand_sid, height_ref, scenario: dict):
    realizer = InverseDynamicsQPRealizer(model, exact_realizer=True)
    stance = tuple(scenario["stance"])
    contacts = realizer.contact_points(model, data, stance)
    targets = {key: pos.copy() for key, (pos, _) in contacts.items()}
    _, _, hand_jac = hand_state(model, data, hand_sid)
    q_ref = TORQUE_STAND_CTRL.copy()
    if scenario.get("arm_reach", False):
        q_ref[22] += 0.45
        q_ref[25] -= 0.55
    mass = float(np.sum(model.body_mass))
    payload = float(scenario.get("payload_kg", 0.0))
    nominal = np.array(scenario.get("nominal_com_acc", [0.0, 0.0, 0.0]), dtype=float)
    nominal[2] += payload * 9.81 / mass
    return {
        "realizer": realizer,
        "q_ref": q_ref,
        "qd_ref": np.zeros(model.nu),
        "task_acc_des": np.zeros(3),
        "hand_jac": hand_jac,
        "stance_contacts": contacts,
        "stance_targets": targets,
        "base_height_ref": height_ref,
        "rpy": roll_pitch_yaw_from_body(data, torso),
        "nominal_com_acc_des": nominal,
        "attitude_weight": 60.0,
        "centroidal_moment_des": np.zeros(3),
    }


def estimate(estimator, model, data, context) -> AuthorityBox:
    return estimator.estimate(context["realizer"], model, data, **{
        key: value for key, value in context.items() if key != "realizer"
    })


def evaluate_request(model, data, context, offset: np.ndarray, tolerance: float = 0.35) -> dict:
    realizer = context["realizer"]
    request = context["nominal_com_acc_des"].copy()
    request[:2] += offset
    realizer.command(
        model, data, context["q_ref"], context["qd_ref"], request[:2],
        context["task_acc_des"], context["hand_jac"],
        context["stance_contacts"], context["stance_targets"],
        context["base_height_ref"], context["rpy"],
        com_acc_des=request, attitude_weight=context["attitude_weight"],
        centroidal_moment_des=context["centroidal_moment_des"],
    )
    Jcom = np.zeros((3, realizer.nv))
    mujoco.mj_jacSubtreeCom(model, data, Jcom, realizer.root_body)
    realized = Jcom[:2] @ realizer.last_qdd if not realizer.last_fallback else np.full(2, np.nan)
    residual = realized - request[:2]
    tau_margin = float(np.min(np.minimum(
        realizer.last_tau_qp - realizer.torque_min,
        realizer.torque_max - realizer.last_tau_qp,
    ))) if not realizer.last_fallback else -np.inf
    lam = realizer.last_contact_force.reshape(-1, 3)
    if lam.size:
        friction_margin = float(np.min(np.c_[
            realizer.mu * lam[:, 2] - np.abs(lam[:, 0]),
            realizer.mu * lam[:, 2] - np.abs(lam[:, 1]),
        ]))
        normal_margin = float(np.min(lam[:, 2]))
    else:
        friction_margin = normal_margin = -np.inf
    feasible = bool(
        not realizer.last_fallback
        and np.linalg.norm(residual, ord=np.inf) <= tolerance
    )
    return {
        "request": offset.tolist(),
        "realized": realized.tolist(),
        "residual": residual.tolist(),
        "residual_inf": float(np.linalg.norm(residual, ord=np.inf)),
        "tau_margin_nm": tau_margin,
        "friction_margin_n": friction_margin,
        "normal_margin_n": normal_margin,
        "fallback": bool(realizer.last_fallback),
        "feasible": feasible,
    }


def corners(lower, upper):
    return np.array(
        [[x, y] for x in (lower[0], upper[0]) for y in (lower[1], upper[1])],
        dtype=float,
    )


def run_closed_loop(variant: str, duration: float = 0.65) -> tuple[dict, dict]:
    model, data, torso, hand_sid, height_ref = settle_model()
    realizer = InverseDynamicsQPRealizer(model, exact_realizer=True)
    estimator = PlanarBodyAuthorityEstimator()
    mpc = NormalizedMPC(
        dim=2, dt=0.02, horizon=15, q_pos=60.0, q_vel=1.0, r=0.01,
        u_max=(np.array([4.0, 4.0]) if variant == "C1_fixed" else None),
    )
    com0 = robot_com(model, data)
    contacts = realizer.contact_points(model, data, ("left", "right"))
    targets = {key: pos.copy() for key, (pos, _) in contacts.items()}
    q_ref = TORQUE_STAND_CTRL.copy()
    qd_ref = np.zeros(model.nu)
    command_stride = max(1, int(round(0.02 / SIM_DT)))
    steps = int(round(duration / SIM_DT))
    command = np.zeros(2)
    authority_lower = np.full(2, np.nan)
    authority_upper = np.full(2, np.nan)
    box_valid = False
    update_times = []
    log = {
        "t": np.zeros(steps),
        "error": np.zeros((steps, 2)),
        "request": np.zeros((steps, 2)),
        "realized": np.zeros((steps, 2)),
        "residual": np.zeros((steps, 2)),
        "authority_lower": np.zeros((steps, 2)),
        "authority_upper": np.zeros((steps, 2)),
        "authority_valid": np.zeros(steps, dtype=int),
        "bound_active": np.zeros(steps, dtype=int),
        "fallback": np.zeros(steps, dtype=int),
    }
    for k in range(steps):
        t = k * SIM_DT
        com = robot_com(model, data)
        vel = com_velocity(model, data, realizer.root_body)
        y_ref = com0[1] + (0.075 if t >= 0.12 else 0.0)
        error = np.array([com[0] - com0[0], com[1] - y_ref])
        contacts = realizer.contact_points(model, data, ("left", "right"))
        hand_pos, hand_vel, hand_jac = hand_state(model, data, hand_sid)
        del hand_pos
        task_acc = -4.0 * hand_vel
        rpy = roll_pitch_yaw_from_body(data, torso)
        if k % command_stride == 0:
            q_ref = TORQUE_STAND_CTRL.copy()
            if variant == "C2_online":
                tic = time.perf_counter()
                box = estimator.estimate(
                    realizer, model, data, q_ref=q_ref, qd_ref=qd_ref,
                    task_acc_des=task_acc, hand_jac=hand_jac,
                    stance_contacts=contacts, stance_targets=targets,
                    base_height_ref=height_ref, rpy=rpy,
                    nominal_com_acc_des=np.zeros(3), attitude_weight=60.0,
                    centroidal_moment_des=np.zeros(3),
                )
                update_times.append(time.perf_counter() - tic)
                authority_lower, authority_upper = box.lower, box.upper
                box_valid = box.valid
                mpc.update_input_box(authority_lower, authority_upper)
            else:
                authority_lower = np.full(2, -4.0)
                authority_upper = np.full(2, 4.0)
                box_valid = True
            command = mpc.solve(np.r_[error, vel[:2]])
            q_ref[1] += 0.035 * command[1]
            q_ref[7] += 0.035 * command[1]

        realizer.command(
            model, data, q_ref, qd_ref, command, task_acc, hand_jac,
            contacts, targets, height_ref, rpy,
            com_acc_des=np.r_[command, 0.0], attitude_weight=60.0,
            centroidal_moment_des=np.zeros(3),
        )
        Jcom = np.zeros((3, realizer.nv))
        mujoco.mj_jacSubtreeCom(model, data, Jcom, realizer.root_body)
        realized = Jcom[:2] @ realizer.last_qdd
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        log["t"][k] = t
        log["error"][k] = error
        log["request"][k] = command
        log["realized"][k] = realized
        log["residual"][k] = realized - command
        log["authority_lower"][k] = authority_lower
        log["authority_upper"][k] = authority_upper
        log["authority_valid"][k] = int(box_valid)
        log["bound_active"][k] = int(mpc.last_bound_active)
        log["fallback"][k] = int(realizer.last_fallback)

    residual_inf = np.max(np.abs(log["residual"]), axis=1)
    summary = {
        "variant": variant,
        "duration_s": duration,
        "mpc_update_period_s": 0.02,
        "authority_recomputed_at_every_mpc_update": variant == "C2_online",
        "authority_valid_fraction": float(np.mean(log["authority_valid"])),
        "bound_active_fraction": float(np.mean(log["bound_active"])),
        "rms_planar_error_mm": float(1000.0 * np.sqrt(np.mean(np.sum(log["error"] ** 2, axis=1)))),
        "median_realization_residual_inf_mps2": float(np.median(residual_inf)),
        "max_realization_residual_inf_mps2": float(np.max(residual_inf)),
        "qp_fallbacks": int(np.sum(log["fallback"])),
        "authority_update_time_ms_median": (
            None if not update_times else float(1000.0 * np.median(update_times))
        ),
        "authority_update_time_ms_max": (
            None if not update_times else float(1000.0 * np.max(update_times))
        ),
    }
    return summary, log


def main():
    model, data, torso, hand_sid, height_ref = settle_model()
    estimator = PlanarBodyAuthorityEstimator()
    scenarios = [
        {"name": "double_support_nominal", "stance": ["left", "right"]},
        {"name": "double_support_payload", "stance": ["left", "right"], "payload_kg": 5.0},
        {"name": "double_support_arm_reach", "stance": ["left", "right"], "arm_reach": True},
        {"name": "single_support_left", "stance": ["left"]},
    ]
    records = {}
    contexts = {}
    boxes = {}
    for scenario in scenarios:
        context = scenario_context(model, data, torso, hand_sid, height_ref, scenario)
        box = estimate(estimator, model, data, context)
        contexts[scenario["name"]] = context
        boxes[scenario["name"]] = box
        variants = {
            "C0_loose": (np.full(2, -8.0), np.full(2, 8.0)),
            "C1_fixed": (np.full(2, -4.0), np.full(2, 4.0)),
            "C2_online": (box.lower, box.upper),
        }
        validations = {}
        for name, (lower, upper) in variants.items():
            trials = [evaluate_request(model, data, context, u) for u in corners(lower, upper)]
            validations[name] = {
                "corner_feasible_fraction": float(np.mean([v["feasible"] for v in trials])),
                "max_corner_residual_inf": float(max(v["residual_inf"] for v in trials)),
                "trials": trials,
            }
        records[scenario["name"]] = {
            "scenario": scenario,
            "authority": {
                "valid": box.valid,
                "status": box.status,
                "lower": box.lower.tolist(),
                "upper": box.upper.tolist(),
                "center": box.center.tolist(),
                "width": (box.upper - box.lower).tolist(),
                "area": box.area,
                "nominal_residual": box.nominal_residual.tolist(),
                "active_constraint": box.active_constraint,
                "exact_realizer_solve_count": box.solve_count,
                "corner_scale": box.corner_scale,
            },
            "corner_validation": validations,
        }

    ds = boxes["double_support_nominal"]
    ss = boxes["single_support_left"]
    intersection_lower = np.maximum(ds.lower, ss.lower)
    intersection_upper = np.minimum(ds.upper, ss.upper)
    intersection_valid = bool(np.all(intersection_lower <= intersection_upper))
    if not intersection_valid:
        intersection_lower = intersection_upper = np.zeros(2)

    # Regression probe for the failure mode that motivated the exact search:
    # a local hard-constraint sensitivity incorrectly pinned upper_x to zero
    # even though the realizer faithfully tracks positive forward requests.
    forward_axis_probe = []
    context = contexts["double_support_nominal"]
    for ux in (0.25, 0.75, 1.50, 2.00):
        result = evaluate_request(model, data, context, np.array([ux, 0.0]))
        result["inside_queried_box"] = bool(
            np.all(np.array([ux, 0.0]) >= ds.lower - 1e-9)
            and np.all(np.array([ux, 0.0]) <= ds.upper + 1e-9)
        )
        forward_axis_probe.append(result)

    # Conditional offset-free boundary: compare cancelling commands against
    # the online set and the exact realizer at the nominal double-support state.
    disturbance_sweep = np.linspace(-5.0, 5.0, 41)
    boundary_trials = []
    for disturbance in disturbance_sweep:
        cancellation = np.array([0.0, -disturbance])
        result = evaluate_request(model, data, context, cancellation)
        result["disturbance_y"] = float(disturbance)
        result["inside_online_box"] = bool(
            np.all(cancellation >= ds.lower - 1e-9)
            and np.all(cancellation <= ds.upper + 1e-9)
        )
        boundary_trials.append(result)

    mpc = NormalizedMPC(dim=2, dt=0.002, horizon=60, q_pos=60.0, q_vel=1.0, r=0.01)
    A0, B0, H0 = mpc.A.copy(), mpc.B.copy(), mpc.H.copy()
    for box in boxes.values():
        mpc.update_input_box(box.lower, box.upper)
        mpc.solve(np.array([0.02, -0.02, 0.0, 0.0]))
    invariant = bool(np.array_equal(A0, mpc.A) and np.array_equal(B0, mpc.B)
                     and np.array_equal(H0, mpc.H))

    closed_loop = {}
    closed_loop_logs = {}
    for variant in ("C1_fixed", "C2_online"):
        summary, log = run_closed_loop(variant)
        closed_loop[variant] = summary
        closed_loop_logs[variant] = log

    report = {
        "model": "Unitree G1 torque model in MuJoCo",
        "evaluation_scope": "frozen-state authority fidelity and support-mode scheduling",
        "realization_tolerance_mps2": estimator.realization_tolerance,
        "canonical_matrices_bitwise_invariant_during_bound_updates": invariant,
        "scenarios": records,
        "transition_schedule": {
            "double_support_lower": ds.lower.tolist(),
            "double_support_upper": ds.upper.tolist(),
            "single_support_lower": ss.lower.tolist(),
            "single_support_upper": ss.upper.tolist(),
            "intersection_lower": intersection_lower.tolist(),
            "intersection_upper": intersection_upper.tolist(),
            "intersection_valid": intersection_valid,
        },
        "forward_axis_regression_probe": forward_axis_probe,
        "offset_free_boundary": boundary_trials,
        "closed_loop_step": closed_loop,
    }
    with (RESULTS / "authority_benchmark.json").open("w") as f:
        json.dump(report, f, indent=2)

    colors = ["#2474B5", "#E17C05", "#4C956C", "#8D5A97"]
    linestyles = ["-", "-", "--", ":"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for color, linestyle, scenario in zip(colors, linestyles, scenarios):
        name = scenario["name"]
        box = boxes[name]
        rect = plt.Rectangle(box.lower, *(box.upper - box.lower), fill=False,
                             linewidth=2.2, linestyle=linestyle,
                             edgecolor=color, label=name.replace("_", " "))
        ax.add_patch(rect)
        ax.plot(box.center[0], box.center[1], "o", color=color, ms=4)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlabel(r"residual command $u_x$ [m/s$^2$]")
    ax.set_ylabel(r"residual command $u_y$ [m/s$^2$]")
    ax.set_title("Exact-query residual-authority boxes")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.autoscale()
    fig.tight_layout()
    fig.savefig(FIGURES / "authority_sets.png", dpi=220)
    plt.close(fig)

    scenario_names = [s["name"] for s in scenarios]
    x = np.arange(len(scenario_names))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    display_names = {"C0_loose": "C0 loose", "C1_fixed": "C1 fixed",
                     "C2_online": "C2 queried"}
    for j, variant in enumerate(("C0_loose", "C1_fixed", "C2_online")):
        vals = [records[name]["corner_validation"][variant]["corner_feasible_fraction"]
                for name in scenario_names]
        axes[0].bar(x + (j - 1) * 0.23, vals, width=0.22,
                    label=display_names[variant])
    axes[0].set_xticks(x, ["DS", "payload", "arm reach", "SS"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("feasible tested corners")
    axes[0].set_title("Tested-corner consistency")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    inside = np.array([r["inside_online_box"] for r in boundary_trials])
    feasible = np.array([r["feasible"] for r in boundary_trials])
    axes[1].step(disturbance_sweep, inside.astype(float), where="mid", label="cancelling request inside set")
    axes[1].plot(disturbance_sweep, feasible.astype(float), "o", ms=3, label="exact-realizer feasible")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel(r"constant disturbance $d_y$ [m/s$^2$]")
    axes[1].set_ylabel("condition satisfied")
    axes[1].set_title("Conditional offset-free boundary")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "authority_fidelity_boundary.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.4), sharex=True)
    for variant, color in (("C1_fixed", "#E17C05"), ("C2_online", "#2474B5")):
        log = closed_loop_logs[variant]
        axes[0].plot(log["t"], 1000.0 * log["error"][:, 1], color=color,
                     label=display_names[variant])
        axes[1].plot(log["t"], log["request"][:, 1], color=color,
                     label=f"{display_names[variant]} request")
        if variant == "C2_online":
            axes[1].fill_between(
                log["t"], log["authority_lower"][:, 1],
                log["authority_upper"][:, 1], color=color, alpha=0.18,
                label="queried authority",
            )
    axes[0].set_ylabel(r"lateral error $e_y$ [mm]")
    axes[0].set_title("Non-real-time per-update authority query")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel(r"$u_y$ [m/s$^2$]")
    axes[1].set_xlabel("time [s]")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "authority_closed_loop.png", dpi=220)
    plt.close(fig)

    print(json.dumps({
        "report": str(RESULTS / "authority_benchmark.json"),
        "figures": [str(FIGURES / "authority_sets.png"),
                    str(FIGURES / "authority_fidelity_boundary.png"),
                    str(FIGURES / "authority_closed_loop.png")],
        "canonical_invariant": invariant,
        "closed_loop": closed_loop,
        "authority": {name: records[name]["authority"] for name in records},
    }, indent=2))


if __name__ == "__main__":
    main()
