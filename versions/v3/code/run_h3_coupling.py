#!/usr/bin/env python3
"""H3: which cross-port reactions need preview, and which the realizer handles.

Section VII gives the body port two ways to anticipate a planned task/arm action:
 (a) EXTERNAL-wrench form (Eq. 23): a planned external interaction load F_h acts on
     the body; the whole-body QP does not model it, so it is a genuine disturbance.
     The coupled controller previews its CoM effect -F_h/m and feeds it forward;
     the split controller rejects it reactively through the body observer, which
     lags a fast-changing load.
 (b) INTERNAL-momentum form (Eq. 23b): a planned arm swing redistributes centroidal
     momentum internally (no external force). Its CoM reaction is
     l_dot_{G,arm}/m = J_com[:,arm] q_ddot_arm.

The experiment runs both on the standing torque-actuated G1 with faithful
centroidal-wrench recovery, at comparable reaction magnitude (~45 N), and reports
the peak/RMS CoM excursion with and without preview. The contrast is the point:
the external load needs the preview (large reduction), whereas the internal arm
momentum is already compensated natively by the unified whole-body QP -- it shares
the CoM Jacobian across arm and legs -- so its uncompensated transient is already
small and a naive external-style preview is unnecessary (indeed counterproductive,
because it perturbs a CoM command the QP is already satisfying). Preview therefore
belongs to what the realizer does not model: external contact loads.

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

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)
SIM_DT = 0.001
COMMAND_DT = 0.002
DURATION = 4.0
ON, OFF = 1.0, 3.0
EXT_AMP, EXT_F = 45.0, 1.6          # external load amplitude [N], frequency [Hz]
SWING_F = 1.2                       # arm-swing frequency [Hz]
# right-arm joints to swing (local index in ACTUATED_JOINT_NAMES) and amplitudes [rad]
SWING_JOINTS = {
    "right_shoulder_pitch_joint": 0.75,
    "right_shoulder_roll_joint": 0.55,
    "right_elbow_joint": 0.65,
}


def _setup():
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
    return model, data, realizer, body_mpc, body_obs


def run(mode: str, coupled: bool):
    """mode='external' -> Eq. 23 external load; mode='internal' -> Eq. 23b arm swing."""
    model, data, realizer, body_mpc, body_obs = _setup()
    torso = body_id(model, "torso_link"); pelvis = body_id(model, "pelvis")
    hand_sid = site_id(model, "right_hand_site")
    com0 = robot_com(model, data)
    base_h = float(data.qpos[2]); q_nom = TORQUE_STAND_CTRL.copy(); qd0 = np.zeros_like(q_nom)
    mass = float(np.sum(model.body_mass))
    stance = ("left", "right")
    stance_contacts = realizer.contact_points(model, data, stance)
    stance_targets = {k: p.copy() for k, (p, _) in stance_contacts.items()}

    names = list(SWING_JOINTS)
    loc = [ACTUATED_JOINT_NAMES.index(n) for n in names]
    arm_dof = realizer.dof[loc]; amps = np.array([SWING_JOINTS[n] for n in names])
    q_arm0 = q_nom[loc].copy(); w = 2 * np.pi * SWING_F

    d_body = np.zeros(2); u_body = np.zeros(2); com_acc_des = np.zeros(3); react_acc = np.zeros(2)
    N = int(DURATION / SIM_DT); period = max(1, int(COMMAND_DT / SIM_DT))
    t_log = np.zeros(N); com_err = np.zeros((N, 2)); rpy_log = np.zeros((N, 2)); freact = np.zeros(N)
    freact_k = 0.0; F_ext = np.zeros(3)

    for k in range(N):
        t = k * SIM_DT
        com = robot_com(model, data); rpy = roll_pitch_yaw_from_body(data, torso)
        _, _, handj = hand_state(model, data, hand_sid)
        stance_contacts = realizer.contact_points(model, data, stance)

        q_ref = q_nom.copy()
        F_ext = np.zeros(3)
        active = ON <= t < OFF
        if mode == "external":
            if active:
                F_ext = np.array([0.0, EXT_AMP * np.sin(2 * np.pi * EXT_F * (t - ON)), 0.0])
        else:  # internal arm swing
            if active:
                ph = w * (t - ON)
                q_ref[loc] = q_arm0 + amps * np.sin(ph)
                qdd_arm = -amps * w * w * np.sin(ph)
            else:
                qdd_arm = np.zeros(len(loc))

        if k % period == 0:
            if mode == "external":
                react_acc = F_ext[:2] / mass                 # CoM effect of the external load
                freact_k = float(np.linalg.norm(F_ext[:2]))
            else:
                Jcom = np.zeros((3, realizer.nv))
                mujoco.mj_jacSubtreeCom(model, data, Jcom, realizer.root_body)
                react_acc = Jcom[:2, arm_dof] @ qdd_arm      # l_dot_{G,arm}/m (Eq. 23b)
                freact_k = float(mass * np.linalg.norm(react_acc))

            y = com[:2] - com0[:2]
            u_body = body_mpc.solve(np.r_[y, data.qvel[:2]], d_body)
            d_body, _ = body_obs.step(y, u_body)
            com_acc_xy = u_body.copy()
            if coupled:
                com_acc_xy = com_acc_xy - react_acc          # preview the planned reaction
            com_acc_des = np.array([com_acc_xy[0], com_acc_xy[1], 0.0])

        data.xfrc_applied[:] = 0.0
        if mode == "external":
            data.xfrc_applied[pelvis, :3] = F_ext
        realizer.command(model, data, q_ref, qd0, np.zeros(2), np.zeros(3), handj,
                          stance_contacts, stance_targets, base_h, rpy, com_acc_des=com_acc_des)
        mujoco.mj_step(model, data); mujoco.mj_forward(model, data)

        t_log[k] = t; com_err[k] = (robot_com(model, data) - com0)[:2]
        rpy_log[k] = roll_pitch_yaw_from_body(data, torso)[:2]; freact[k] = freact_k

    m = (t_log >= ON) & (t_log <= OFF)
    return dict(
        peak_com_mm=float(1000 * np.max(np.linalg.norm(com_err[m], axis=1))),
        rms_com_mm=float(1000 * np.sqrt(np.mean(np.sum(com_err[m] ** 2, axis=1)))),
        peak_rp_rad=float(np.max(np.abs(rpy_log[m]))),
        max_react_N=float(np.max(freact[m])),
        t=t_log, com_err=com_err, rpy=rpy_log,
    )


def _pack(split, coupled):
    return dict(
        max_reaction_N=round(split["max_react_N"], 1),
        split=dict(peak_com_mm=round(split["peak_com_mm"], 2), rms_com_mm=round(split["rms_com_mm"], 2), peak_rp_rad=round(split["peak_rp_rad"], 4)),
        coupled=dict(peak_com_mm=round(coupled["peak_com_mm"], 2), rms_com_mm=round(coupled["rms_com_mm"], 2), peak_rp_rad=round(coupled["peak_rp_rad"], 4)),
        peak_com_reduction_x=round(split["peak_com_mm"] / max(coupled["peak_com_mm"], 1e-6), 2),
        rms_com_reduction_x=round(split["rms_com_mm"] / max(coupled["rms_com_mm"], 1e-6), 2),
    )


def main():
    ext_s, ext_c = run("external", False), run("external", True)
    int_s, int_c = run("internal", False), run("internal", True)
    external = _pack(ext_s, ext_c); internal = _pack(int_s, int_c)
    res = dict(
        external=dict(load="planned external load %.0f N at %.1f Hz on the body (Eq. 23)" % (EXT_AMP, EXT_F), **external),
        internal=dict(load="planned right-arm swing (shoulder pitch/roll + elbow) at %.1f Hz, no external force (Eq. 23b)" % SWING_F, **internal),
        note="External load needs the preview (large reduction); internal arm momentum is compensated natively by the unified whole-body QP, so its uncompensated transient is already small and preview does not help. Preview belongs to what the realizer does not model: external contact loads.",
        # top-level headline keys (external case) for downstream checks
        split=external["split"], coupled=external["coupled"],
        peak_com_reduction_x=external["peak_com_reduction_x"],
    )
    print(json.dumps(res, indent=2))
    with (RESULTS / "h3_coupling_summary.json").open("w") as f:
        json.dump(res, f, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ext_s["t"], 1000 * np.linalg.norm(ext_s["com_err"], axis=1), label="split")
    ax[0].plot(ext_c["t"], 1000 * np.linalg.norm(ext_c["com_err"], axis=1), label="coupled (preview)")
    ax[0].set_title("H3(a) external load: CoM excursion [mm]"); ax[0].set_xlabel("t [s]"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(int_s["t"], 1000 * np.linalg.norm(int_s["com_err"], axis=1), label="split")
    ax[1].plot(int_c["t"], 1000 * np.linalg.norm(int_c["com_err"], axis=1), label="coupled (preview)")
    ax[1].set_title("H3(b) internal arm momentum: CoM excursion [mm]"); ax[1].set_xlabel("t [s]"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(RESULTS / "h3_coupling.png", dpi=160)
    print("saved: results/h3_coupling.png, results/h3_coupling_summary.json")


if __name__ == "__main__":
    main()
