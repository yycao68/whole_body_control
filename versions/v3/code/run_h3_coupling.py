#!/usr/bin/env python3
"""H3: arm-reaction preview (coupled realization) reduces cross-port transients.

A fast right-arm interaction produces a reaction wrench on the body (Sec. VII,
Eq. 23): W_{b<-t} = -[F_t; (x_t-c) x F_t]. Because the task wrench F_t is PLANNED,
the coupled controller can preview its CoM reaction -F_t/m and feed it forward
into the body-port command, cancelling the transient before it develops; the
split controller instead rejects the same reaction reactively through the body
disturbance observer, which lags a fast-changing reaction. Both run on the
standing torque-actuated G1 with the faithful centroidal-wrench recovery.

The reach drives a fast oscillating task wrench on the body (the known arm
reaction). Metric: peak/RMS CoM excursion during the reach.

Usage: MPLCONFIGDIR=/private/tmp/mplconfig python3 run_h3_coupling.py
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
from run_h1_h2 import contact_consistent_task_inertia

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)
SIM_DT = 0.001
COMMAND_DT = 0.002
DURATION = 4.0
REACH_ON, REACH_OFF = 1.0, 3.0     # fast reach window
REACT_AMP, REACT_F = 45.0, 1.6     # arm-reaction wrench amplitude [N], frequency [Hz]


def run(coupled: bool):
    model = mujoco.MjModel.from_xml_path(str(generate_torque_model()))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for v, n in zip(TORQUE_STAND_CTRL, ACTUATED_JOINT_NAMES):
        data.qpos[model.jnt_qposadr[joint_id(model, n)]] = v
    mujoco.mj_forward(model, data)

    realizer = InverseDynamicsQPRealizer(model)
    realizer.task_weight = 25.0; realizer.task_acc_clip = 45.0
    body_mpc = NormalizedMPC(dim=2, dt=COMMAND_DT, horizon=35, q_pos=55.0, q_vel=12.0, r=0.08, u_max=np.array([3.5, 3.0]))
    task_mpc = NormalizedMPC(dim=3, dt=COMMAND_DT, horizon=18, q_pos=300.0, q_vel=24.0, r=0.04, u_max=np.array([20.0, 20.0, 20.0]))
    body_obs = RandomWalkDisturbanceObserver(dim=2, dt=COMMAND_DT, q_d=0.05, r_y=1.5e-4)
    task_obs = RandomWalkDisturbanceObserver(dim=3, dt=COMMAND_DT, q_d=0.3, r_y=1e-4)

    torso = body_id(model, "torso_link"); pelvis = body_id(model, "pelvis")
    hand_sid = site_id(model, "right_hand_site")
    com0 = robot_com(model, data); hand0, _, _ = hand_state(model, data, hand_sid)
    base_h = float(data.qpos[2]); q_nom = TORQUE_STAND_CTRL.copy(); qd0 = np.zeros_like(q_nom)
    mass = float(np.sum(model.body_mass))
    stance = ("left", "right")
    stance_contacts = realizer.contact_points(model, data, stance)
    stance_targets = {k: p.copy() for k, (p, _) in stance_contacts.items()}

    d_body = np.zeros(2); u_body = np.zeros(2); d_task = np.zeros(3); u_task = np.zeros(3)
    com_acc_des = np.zeros(3); task_acc_des = np.zeros(3)
    N = int(DURATION / SIM_DT); period = max(1, int(COMMAND_DT / SIM_DT))
    t_log = np.zeros(N); com_err = np.zeros((N, 2)); rpy_log = np.zeros((N, 2)); freact = np.zeros(N)

    for k in range(N):
        t = k * SIM_DT
        com = robot_com(model, data); rpy = roll_pitch_yaw_from_body(data, torso)
        hand, handv, handj = hand_state(model, data, hand_sid)
        stance_contacts = realizer.contact_points(model, data, stance)

        # planned arm-reaction wrench on the body (fast lateral oscillation)
        if REACH_ON <= t < REACH_OFF:
            F_react = np.array([0.0, REACT_AMP * np.sin(2 * np.pi * REACT_F * (t - REACH_ON)), 0.0])
        else:
            F_react = np.zeros(3)

        if k % period == 0:
            task_acc_des = np.zeros(3)                         # arm holds; reaction is the disturbance
            x_body = np.r_[com[:2] - com0[:2], data.qvel[:2]]
            u_body = body_mpc.solve(x_body, d_body)
            d_body, _ = body_obs.step(com[:2] - com0[:2], u_body)
            com_acc_xy = u_body.copy()
            if coupled:
                # preview the planned reaction and cancel its CoM effect (+F/m
                # external -> command -F/m); split relies on the observer only
                com_acc_xy = com_acc_xy - F_react[:2] / mass
            com_acc_des = np.array([com_acc_xy[0], com_acc_xy[1], 0.0])
            freact_k = float(np.linalg.norm(F_react[:2]))

        data.xfrc_applied[:] = 0.0
        data.xfrc_applied[pelvis, :3] = F_react               # the arm reaction on the body
        realizer.command(model, data, q_nom, qd0, np.zeros(2), task_acc_des, handj,
                          stance_contacts, stance_targets, base_h, rpy, com_acc_des=com_acc_des)
        mujoco.mj_step(model, data); mujoco.mj_forward(model, data)

        t_log[k] = t; com_err[k] = (robot_com(model, data) - com0)[:2]
        rpy_log[k] = roll_pitch_yaw_from_body(data, torso)[:2]; freact[k] = freact_k

    m = (t_log >= REACH_ON) & (t_log <= REACH_OFF)
    return dict(
        peak_com_mm=float(1000 * np.max(np.linalg.norm(com_err[m], axis=1))),
        rms_com_mm=float(1000 * np.sqrt(np.mean(np.sum(com_err[m] ** 2, axis=1)))),
        peak_rp_rad=float(np.max(np.abs(rpy_log[m]))),
        max_react_N=float(np.max(freact[m])),
        t=t_log, com_err=com_err, rpy=rpy_log,
    )


def main():
    split = run(coupled=False)
    coupled = run(coupled=True)
    res = dict(
        reach="oscillating arm reaction %.0f N at %.1f Hz, %.1f-%.1f s" % (REACT_AMP, REACT_F, REACH_ON, REACH_OFF),
        max_arm_reaction_N=round(split["max_react_N"], 1),
        split=dict(peak_com_mm=round(split["peak_com_mm"], 2), rms_com_mm=round(split["rms_com_mm"], 2), peak_rp_rad=round(split["peak_rp_rad"], 4)),
        coupled=dict(peak_com_mm=round(coupled["peak_com_mm"], 2), rms_com_mm=round(coupled["rms_com_mm"], 2), peak_rp_rad=round(coupled["peak_rp_rad"], 4)),
        peak_com_reduction_x=round(split["peak_com_mm"] / max(coupled["peak_com_mm"], 1e-6), 2),
        peak_rp_reduction_x=round(split["peak_rp_rad"] / max(coupled["peak_rp_rad"], 1e-9), 2),
    )
    print(json.dumps(res, indent=2))
    with (RESULTS / "h3_coupling_summary.json").open("w") as f:
        json.dump(res, f, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(split["t"], 1000 * np.linalg.norm(split["com_err"], axis=1), label="split")
    ax[0].plot(coupled["t"], 1000 * np.linalg.norm(coupled["com_err"], axis=1), label="coupled (preview)")
    ax[0].set_title("H3: CoM excursion during fast reach [mm]"); ax[0].set_xlabel("t [s]"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(split["t"], np.max(np.abs(split["rpy"]), axis=1), label="split")
    ax[1].plot(coupled["t"], np.max(np.abs(coupled["rpy"]), axis=1), label="coupled (preview)")
    ax[1].set_title("H3: torso |roll/pitch| [rad]"); ax[1].set_xlabel("t [s]"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(RESULTS / "h3_coupling.png", dpi=160)
    print("saved: results/h3_coupling.png, results/h3_coupling_summary.json")


if __name__ == "__main__":
    main()
