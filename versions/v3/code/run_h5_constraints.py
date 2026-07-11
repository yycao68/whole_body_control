#!/usr/bin/env python3
"""H5: physical constraints belong to recovery, not to the predictor.

Friction cones, unilateral contact, center-of-pressure, and joint-torque limits
are enforced in the whole-body interaction realizer (Eq. 22), not in the
normalized predictor. Under a strong lateral disturbance, we compare CONSTRAINED
recovery (friction pyramid + torque limits active in the QP) against UNCONSTRAINED
recovery (both dropped) on the standing torque-actuated G1, and measure the
constraint violations of the *recovered* (QP-commanded) wrench/torque and the
resulting tracking slack.

Constrained recovery keeps the recovered contact forces inside the friction cone
and the recovered torques inside the actuator limits (feasible), at the cost of a
larger CoM tracking slack; unconstrained recovery would command infeasible
tangential forces and over-limit torques -- exactly the quantities the recovery
constraints exist to bound.

Usage: MPLCONFIGDIR=/private/tmp/mplconfig python3 run_h5_constraints.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from normalized_mpc import NormalizedMPC, RandomWalkDisturbanceObserver
from run_g1_root_assist_demo import ACTUATED_JOINT_NAMES, robot_com, roll_pitch_yaw_from_body
from run_g1_torque_realizer_benchmark import (
    InverseDynamicsQPRealizer, TORQUE_STAND_CTRL,
    body_id, generate_torque_model, hand_state, joint_id, site_id,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)
SIM_DT = 0.001
COMMAND_DT = 0.002
DURATION = 3.0
PUSH_ON, PUSH_OFF = 1.0, 2.5
PUSH_PEAK = 45.0      # sustained lateral pelvis force [N]
MU = 0.5              # friction coefficient used by the recovery QP


def run(constraints_on: bool, push_vec=None, push_on: float = PUSH_ON, push_off: float = PUSH_OFF):
    if push_vec is None:
        push_vec = np.array([0.0, PUSH_PEAK, 0.0])
    push_vec = np.asarray(push_vec, dtype=float)
    model = mujoco.MjModel.from_xml_path(str(generate_torque_model()))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for v, n in zip(TORQUE_STAND_CTRL, ACTUATED_JOINT_NAMES):
        data.qpos[model.jnt_qposadr[joint_id(model, n)]] = v
    mujoco.mj_forward(model, data)

    realizer = InverseDynamicsQPRealizer(model)
    realizer.constraints_on = constraints_on
    realizer.mu = MU
    tau_max = np.maximum(np.abs(realizer.torque_min), np.abs(realizer.torque_max))
    body_mpc = NormalizedMPC(dim=2, dt=COMMAND_DT, horizon=35, q_pos=55.0, q_vel=12.0, r=0.08, u_max=np.array([3.5, 3.0]))
    body_obs = RandomWalkDisturbanceObserver(dim=2, dt=COMMAND_DT, q_d=0.05, r_y=1.5e-4)

    torso = body_id(model, "torso_link"); pelvis = body_id(model, "pelvis")
    hand_sid = site_id(model, "right_hand_site")
    com0 = robot_com(model, data); base_h = float(data.qpos[2])
    q_nom = TORQUE_STAND_CTRL.copy(); qd0 = np.zeros_like(q_nom)
    stance = ("left", "right")
    stance_contacts = realizer.contact_points(model, data, stance)
    stance_targets = {k: p.copy() for k, (p, _) in stance_contacts.items()}

    d_body = np.zeros(2); u_body = np.zeros(2); com_acc_des = np.zeros(3)
    N = int(DURATION / SIM_DT); period = max(1, int(COMMAND_DT / SIM_DT))
    t_log = np.zeros(N); fric_viol = np.zeros(N); tau_viol = np.zeros(N)
    tau_sat = np.zeros(N); com_err = np.zeros((N, 2)); fell = False

    for k in range(N):
        t = k * SIM_DT
        com = robot_com(model, data); rpy = roll_pitch_yaw_from_body(data, torso)
        _, _, handj = hand_state(model, data, hand_sid)
        stance_contacts = realizer.contact_points(model, data, stance)

        if k % period == 0:
            y = com[:2] - com0[:2]
            u_body = body_mpc.solve(np.r_[y, data.qvel[:2]], d_body)
            d_body, _ = body_obs.step(y, u_body)
            com_acc_des = np.array([u_body[0], u_body[1], 0.0])

        push = np.zeros(3)
        if push_on <= t < push_off:
            push = push_vec
        data.xfrc_applied[:] = 0.0
        data.xfrc_applied[pelvis, :3] = push

        tau, tau_unsat, saturation = realizer.command(
            model, data, q_nom, qd0, np.zeros(2), np.zeros(3), handj,
            stance_contacts, stance_targets, base_h, rpy, com_acc_des=com_acc_des)
        mujoco.mj_step(model, data); mujoco.mj_forward(model, data)

        # constraint violations of the RECOVERED (QP-commanded) quantities
        lam = realizer.last_contact_force
        fv = 0.0
        for i in range(len(lam) // 3):
            fx, fy, fz = lam[3 * i:3 * i + 3]
            mz = MU * max(fz, 0.0)
            fv = max(fv, float(abs(fx) - mz), float(abs(fy) - mz))     # >0 = outside pyramid (the QP's constraint)
        fric_viol[k] = fv
        tau_viol[k] = max(0.0, float(np.max(np.abs(tau_unsat) - tau_max)))  # true nonnegative violation
        tau_sat[k] = float(np.linalg.norm(saturation))
        t_log[k] = t; com_err[k] = (robot_com(model, data) - com0)[:2]
        if data.qpos[2] < 0.45 or np.max(np.abs(rpy[:2])) > 0.7:
            fell = True
            for arr in (fric_viol, tau_viol, tau_sat):
                arr[k + 1:] = arr[k]
            com_err[k + 1:] = com_err[k]; t_log[k + 1:] = t
            break

    m = (t_log >= push_on) & (t_log <= push_off)
    if not np.any(m):                      # fell before/at push onset
        m = np.arange(N) <= max(1, k)
    return dict(
        constraints_on=constraints_on, fell=fell, fell_t=float(t_log[max(0, k)]),
        max_friction_violation_N=round(float(np.max(fric_viol[m])), 2),
        max_torque_violation_Nm=round(float(np.max(tau_viol[m])), 2),
        max_torque_saturation_Nm=round(float(np.max(tau_sat[m])), 2),
        ss_com_error_mm=round(float(1000 * np.mean(np.linalg.norm(com_err[m], axis=1))), 1),
        t=t_log, fric_viol=fric_viol, tau_viol=tau_viol, com_err=com_err,
    )


def run_batch(n_seeds: int = 50):
    """Randomized-push statistics: over n_seeds pushes (magnitude, direction, and
    onset randomized), compare constrained vs unconstrained recovery. Reports
    success rate and the distribution of recovered-wrench/torque violations."""
    rng = np.random.default_rng(20260705)
    con_fell = np.zeros(n_seeds, bool); unc_fell = np.zeros(n_seeds, bool)
    con_fric = np.zeros(n_seeds); con_tau = np.zeros(n_seeds); con_com = np.zeros(n_seeds)
    unc_fric = np.zeros(n_seeds); unc_tau = np.zeros(n_seeds)
    mags = np.zeros(n_seeds)
    for s in range(n_seeds):
        mag = float(rng.uniform(30.0, 50.0))                       # push magnitude [N]
        side = 1.0 if rng.random() < 0.5 else -1.0                 # lateral sign
        dx = float(rng.uniform(-0.4, 0.4))                         # small sagittal component
        d = np.array([dx, side * 1.0, 0.0]); d /= np.linalg.norm(d[:2])
        push = mag * d
        t_on = float(rng.uniform(0.8, 1.2)); t_off = t_on + 1.5
        mags[s] = mag
        c = run(True, push, t_on, t_off); u = run(False, push, t_on, t_off)
        con_fell[s] = c["fell"]; con_fric[s] = c["max_friction_violation_N"]
        con_tau[s] = c["max_torque_violation_Nm"]; con_com[s] = c["ss_com_error_mm"]
        unc_fell[s] = u["fell"]; unc_fric[s] = u["max_friction_violation_N"]
        unc_tau[s] = u["max_torque_violation_Nm"]
    def stat(a):
        return dict(mean=round(float(np.mean(a)), 2), max=round(float(np.max(a)), 2), std=round(float(np.std(a)), 2))
    return dict(
        n_seeds=n_seeds,
        push="pelvis push, magnitude U(30,50) N, randomized lateral-dominant direction, onset U(0.8,1.2) s for 1.5 s; friction mu=%.2f" % MU,
        push_magnitude_N=stat(mags),
        constrained=dict(success_rate=round(float(np.mean(~con_fell)), 3), n_stood=int(np.sum(~con_fell)),
                         friction_violation_N=stat(con_fric), torque_violation_Nm=stat(con_tau),
                         ss_com_error_mm=stat(con_com)),
        unconstrained=dict(fall_rate=round(float(np.mean(unc_fell)), 3), n_fell=int(np.sum(unc_fell)),
                           friction_violation_N=stat(unc_fric), torque_violation_Nm=stat(unc_tau)),
    )


def main():
    con = run(constraints_on=True)
    unc = run(constraints_on=False)
    res = dict(
        disturbance="%.0f N lateral pelvis push, %.1f-%.1f s; friction mu=%.2f" % (PUSH_PEAK, PUSH_ON, PUSH_OFF, MU),
        constrained=dict(max_friction_violation_N=con["max_friction_violation_N"],
                         max_torque_violation_Nm=con["max_torque_violation_Nm"],
                         max_torque_saturation_Nm=con["max_torque_saturation_Nm"],
                         ss_com_error_mm=con["ss_com_error_mm"], fell=con["fell"]),
        unconstrained=dict(max_friction_violation_N=unc["max_friction_violation_N"],
                           max_torque_violation_Nm=unc["max_torque_violation_Nm"],
                           max_torque_saturation_Nm=unc["max_torque_saturation_Nm"],
                           ss_com_error_mm=unc["ss_com_error_mm"], fell=unc["fell"]),
    )
    print(json.dumps(res, indent=2))
    with (RESULTS / "h5_constraints_summary.json").open("w") as f:
        json.dump(res, f, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(con["t"], con["fric_viol"], label="constrained")
    ax[0].plot(unc["t"], unc["fric_viol"], label="unconstrained")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_title("H5: recovered friction-cone violation [N]"); ax[0].set_xlabel("t [s]"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(con["t"], con["tau_viol"], label="constrained")
    ax[1].plot(unc["t"], unc["tau_viol"], label="unconstrained")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_title("H5: recovered torque-limit violation [Nm]"); ax[1].set_xlabel("t [s]"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(RESULTS / "h5_constraints.png", dpi=160)
    print("saved: results/h5_constraints.png, results/h5_constraints_summary.json")

    stats = run_batch(50)
    print(json.dumps(stats, indent=2))
    with (RESULTS / "h5_constraints_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)
    print("saved: results/h5_constraints_stats.json")


if __name__ == "__main__":
    main()
