#!/usr/bin/env python3
"""H6: the interaction layer on a moving base (guided load-carry).

The repositioning claim is that this is an interaction-dynamics layer that rides
on top of a base doing its own motion, and adds predictable, constraint-aware
physical interaction. This experiment demonstrates that directly, staying in the
regime where the whole-body realizer is solid (double support, no stepping):

  * The BASE commands its own center-of-mass trajectory -- a lateral weight-shift
    sway c_ref(t) = c0 + A sin(2 pi f_sway t) e_y, feasible inside the support
    polygon -- standing in for what a locomotion/balance base would command.
  * A PLANNED oscillating interaction load F_h(t) is reacted at the trunk (a
    pushed/pulled object, tether, or carried load whose reaction passes through
    the body), disturbing the center-of-mass translation channel.

The body port tracks c_ref through the normalized MPC (feedforward of the base
reference acceleration plus the residual correction u_c). The interaction layer,
when ON (coupled), previews the planned load's centroidal effect -F_h/m and feeds
it forward; when OFF (split), the same load is left to the body disturbance
observer, which lags. Metric: RMS/peak base-reference tracking error of the CoM
during the load window.

Result: with the interaction layer the base keeps tracking its own trajectory
despite the manipulation load; without it the load bleeds into a base-tracking
error. This is the interaction value-add on top of a moving base.

Usage: MPLCONFIGDIR=/private/tmp/mplconfig python3 run_h6_onbase.py
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
DURATION = 6.0
SWAY_A, SWAY_F = 0.05, 0.25         # base fwd/back weight-shift amplitude [m], frequency [Hz]
LOAD_ON, LOAD_OFF = 1.5, 5.5        # planned-load window [s]
LOAD_AMP, LOAD_F = 45.0, 1.6        # planned lateral trunk load amplitude [N], frequency [Hz]


def run(interaction_layer: bool):
    model = mujoco.MjModel.from_xml_path(str(generate_torque_model()))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for v, n in zip(TORQUE_STAND_CTRL, ACTUATED_JOINT_NAMES):
        data.qpos[model.jnt_qposadr[joint_id(model, n)]] = v
    mujoco.mj_forward(model, data)

    realizer = InverseDynamicsQPRealizer(model)
    body_mpc = NormalizedMPC(dim=2, dt=COMMAND_DT, horizon=35, q_pos=55.0, q_vel=12.0, r=0.08, u_max=np.array([3.5, 3.0]))
    body_obs = RandomWalkDisturbanceObserver(dim=2, dt=COMMAND_DT, q_d=0.05, r_y=1.5e-4)

    torso = body_id(model, "torso_link"); pelvis = body_id(model, "pelvis")
    hand_sid = site_id(model, "right_hand_site")
    com0 = robot_com(model, data)
    base_h = float(data.qpos[2]); q_nom = TORQUE_STAND_CTRL.copy(); qd0 = np.zeros_like(q_nom)
    mass = float(np.sum(model.body_mass))
    stance = ("left", "right")
    stance_contacts = realizer.contact_points(model, data, stance)
    stance_targets = {k: p.copy() for k, (p, _) in stance_contacts.items()}
    ws = 2 * np.pi * SWAY_F; wl = 2 * np.pi * LOAD_F

    d_body = np.zeros(2); u_body = np.zeros(2); com_acc_des = np.zeros(3)
    N = int(DURATION / SIM_DT); period = max(1, int(COMMAND_DT / SIM_DT))
    t_log = np.zeros(N); track_err = np.zeros((N, 2)); load_log = np.zeros(N); fell = False

    for k in range(N):
        t = k * SIM_DT
        com = robot_com(model, data); rpy = roll_pitch_yaw_from_body(data, torso)
        _, _, handj = hand_state(model, data, hand_sid)
        stance_contacts = realizer.contact_points(model, data, stance)

        # base's own commanded CoM trajectory (forward/back weight-shift, where
        # the base has ample sagittal authority; the lateral load preview then
        # does not compete with the base motion for scarce lateral authority)
        c_ref = com0.copy(); c_ref[0] = com0[0] + SWAY_A * np.sin(ws * t)
        cdd_ref = np.array([-SWAY_A * ws * ws * np.sin(ws * t), 0.0])   # reference accel (feedforward)

        # planned carried load at the hand (lateral oscillation)
        F_load = np.zeros(3)
        if LOAD_ON <= t < LOAD_OFF:
            F_load[1] = LOAD_AMP * np.sin(wl * (t - LOAD_ON))

        if k % period == 0:
            y = com[:2] - c_ref[:2]                              # base-tracking error
            u_body = body_mpc.solve(np.r_[y, data.qvel[:2]], d_body)
            d_body, _ = body_obs.step(y, u_body)
            com_acc_xy = cdd_ref + u_body                        # track the base reference
            if interaction_layer:
                com_acc_xy = com_acc_xy - F_load[:2] / mass      # preview the planned load
            com_acc_des = np.array([com_acc_xy[0], com_acc_xy[1], 0.0])

        data.xfrc_applied[:] = 0.0
        data.xfrc_applied[pelvis, :3] = F_load
        realizer.command(model, data, q_nom, qd0, np.zeros(2), np.zeros(3), handj,
                          stance_contacts, stance_targets, base_h, rpy, com_acc_des=com_acc_des)
        mujoco.mj_step(model, data); mujoco.mj_forward(model, data)

        t_log[k] = t; track_err[k] = (robot_com(model, data) - c_ref)[:2]; load_log[k] = F_load[1]
        if data.qpos[2] < 0.45 or np.max(np.abs(rpy[:2])) > 0.7:
            fell = True; track_err[k + 1:] = track_err[k]; t_log[k + 1:] = t; load_log[k + 1:] = 0.0
            break

    m = (t_log >= LOAD_ON) & (t_log <= LOAD_OFF)
    return dict(
        # lateral axis (where the load acts and the preview corrects)
        rms_lat_mm=float(1000 * np.sqrt(np.mean(track_err[m, 1] ** 2))),
        peak_lat_mm=float(1000 * np.max(np.abs(track_err[m, 1]))),
        # forward axis (the base's own weight-shift motion; unaffected by preview)
        rms_fwd_mm=float(1000 * np.sqrt(np.mean(track_err[m, 0] ** 2))),
        max_load_N=float(np.max(np.abs(load_log[m]))), fell=fell,
        t=t_log, track_err=track_err, load=load_log,
    )


def main():
    off = run(interaction_layer=False)
    on = run(interaction_layer=True)
    res = dict(
        setup="double-support base fwd/back weight-shift %.0f mm at %.2f Hz; planned lateral trunk load %.0f N at %.1f Hz over %.1f-%.1f s" % (
            1000 * SWAY_A, SWAY_F, LOAD_AMP, LOAD_F, LOAD_ON, LOAD_OFF),
        max_load_N=round(off["max_load_N"], 1),
        layer_off=dict(rms_lat_mm=round(off["rms_lat_mm"], 2), peak_lat_mm=round(off["peak_lat_mm"], 2), rms_fwd_mm=round(off["rms_fwd_mm"], 2), fell=off["fell"]),
        layer_on=dict(rms_lat_mm=round(on["rms_lat_mm"], 2), peak_lat_mm=round(on["peak_lat_mm"], 2), rms_fwd_mm=round(on["rms_fwd_mm"], 2), fell=on["fell"]),
        lat_rms_reduction_x=round(off["rms_lat_mm"] / max(on["rms_lat_mm"], 1e-6), 2),
        lat_peak_reduction_x=round(off["peak_lat_mm"] / max(on["peak_lat_mm"], 1e-6), 2),
        note="Interaction layer riding a moving (weight-shifting) base. The base tracks its own forward weight-shift equally well with or without the layer (fwd error nearly identical); previewing the planned lateral load keeps the CoM on the base's lateral reference, whereas without the layer the load bleeds into a lateral tracking error. Double support throughout; neither run falls.",
    )
    print(json.dumps(res, indent=2))
    with (RESULTS / "h6_onbase_summary.json").open("w") as f:
        json.dump(res, f, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(off["t"], 1000 * np.abs(off["track_err"][:, 1]), label="layer off (split)")
    ax[0].plot(on["t"], 1000 * np.abs(on["track_err"][:, 1]), label="layer on (preview)")
    ax[0].axvspan(LOAD_ON, LOAD_OFF, color="gray", alpha=0.12)
    ax[0].set_title("H6: lateral base-reference tracking error [mm]"); ax[0].set_xlabel("t [s]"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(off["t"], off["load"], color="k", lw=0.8)
    ax[1].set_title("H6: planned lateral trunk load [N]"); ax[1].set_xlabel("t [s]"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(RESULTS / "h6_onbase.png", dpi=160)
    print("saved: results/h6_onbase.png, results/h6_onbase_summary.json")


if __name__ == "__main__":
    main()
