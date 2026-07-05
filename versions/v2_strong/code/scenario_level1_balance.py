"""
Scenario G — experimental Level-1 centroidal balance controller.

This file tests whether a small centroidal GRF QP can replace the hand-tuned
quasi-static balance stand-in used by Scenario F.  It is intentionally separate
from the paper benchmark until it passes the contact audit.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from impedance_mpc import ImpedanceMPC
from kalman import KalmanDisturbanceEstimator
from level1_centroidal import Level1CentroidalBalance
import scenario_qstatic as qstatic
from wbc_core import (
    get_body_jacobian,
    get_contact_consistent_inverse,
    get_mass_matrix,
    get_site_jacobian,
    get_task_inertia,
)


MODEL = Path(__file__).with_name("biped_qstatic.xml")
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CTRL_DT = 0.0005
T_SHIFT = 2.2
T_PRELIFT = T_SHIFT + 0.8
T_LIFT = T_PRELIFT + 0.8
T_HOLD = T_LIFT + 1.2
T_PLACE = T_HOLD + 0.9
T_DWELL = T_PLACE + 0.5
T_RECTR = T_DWELL + 2.2
T_END = T_RECTR + 0.8
N_RUN = int(T_END / CTRL_DT)

Y_TARGET = -0.090
L_STANCE = np.array([-0.05, 0.10, -0.05])
L_LIFT = np.array([-1.00, 1.60, 0.70])
F_DIST = np.array([8.0, 0.0, 0.0])
T_DIST = 0.5
Q_MPC = np.diag([6e4, 6e4, 6e4, 60.0, 60.0, 60.0])
R_MPC = 0.01 * np.eye(3)
F_MAX = 80.0

ARM_JOINTS = ["right_shoulder_x", "right_shoulder_y", "right_elbow_y"]
ARM_REF = {"right_shoulder_x": 0.0, "right_shoulder_y": 0.5, "right_elbow_y": -1.0}
LEG_JOINTS = [
    "left_hip_x", "left_hip_y", "left_knee_y", "left_ankle_y", "left_ankle_x",
    "right_hip_x", "right_hip_y", "right_knee_y", "right_ankle_y", "right_ankle_x",
]


def _smooth(a, b, x):
    x = float(np.clip(x, 0.0, 1.0))
    s = x * x * x * (x * (x * 6 - 15) + 10)
    return a + (b - a) * s


def _phase(t):
    if t < T_SHIFT:
        return _smooth(0.0, Y_TARGET, t / T_SHIFT), L_STANCE, "SHIFT", False
    if t < T_PRELIFT:
        return Y_TARGET, L_STANCE, "PRELIFT", False
    if t < T_LIFT:
        s = (t - T_PRELIFT) / (T_LIFT - T_PRELIFT)
        return Y_TARGET, _smooth(0, 1, s) * (L_LIFT - L_STANCE) + L_STANCE, "LIFT", s > 0.45
    if t < T_HOLD:
        return Y_TARGET, L_LIFT, "HOLD", True
    if t < T_PLACE:
        s = (t - T_HOLD) / (T_PLACE - T_HOLD)
        return Y_TARGET, _smooth(0, 1, s) * (L_STANCE - L_LIFT) + L_LIFT, "PLACE", s < 0.55
    if t < T_DWELL:
        return Y_TARGET, L_STANCE, "DWELL", False
    if t < T_RECTR:
        return _smooth(Y_TARGET, 0.0, (t - T_DWELL) / (T_RECTR - T_DWELL)), L_STANCE, "RECENTER", False
    return 0.0, L_STANCE, "DONE", False


def _make():
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d


def _setup(m):
    aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    joints = LEG_JOINTS + ARM_JOINTS
    return {
        "A": {n: aid(n) for n in LEG_JOINTS + ARM_JOINTS},
        "QAD": {n: m.jnt_qposadr[jid(n)] for n in joints},
        "DAD": {n: m.jnt_dofadr[jid(n)] for n in joints},
        "hand": sid("right_hand_site"),
        "lfoot": sid("left_foot_contact"),
        "rfoot": sid("right_foot_contact"),
        "hand_body": bid("right_hand"),
        "torso_body": bid("torso"),
        "floor_geom": gid("floor"),
        "lfoot_geom": gid("left_foot_geom"),
        "rfoot_geom": gid("right_foot_geom"),
    }


def _foot_contacts(d, ids):
    left = False
    right = False
    for i in range(d.ncon):
        pair = {d.contact[i].geom1, d.contact[i].geom2}
        if pair == {ids["floor_geom"], ids["lfoot_geom"]}:
            left = True
        elif pair == {ids["floor_geom"], ids["rfoot_geom"]}:
            right = True
    return left, right


def _joint_pd(d, ids, joint, ref, kp, kd, lim):
    q = d.qpos[ids["QAD"][joint]]
    dq = d.qvel[ids["DAD"][joint]]
    return float(np.clip(kp * (ref - q) - kd * dq, -lim, lim))


def _apply_arm_hold(d, ids, tau_task=None):
    for i, jn in enumerate(ARM_JOINTS):
        g = d.qfrc_bias[ids["DAD"][jn]]
        tau = _joint_pd(d, ids, jn, ARM_REF[jn], 24.0, 2.4, 80.0) + g
        if tau_task is not None:
            tau += tau_task[i]
        d.ctrl[ids["A"][jn]] = float(np.clip(tau, -80.0 if i < 2 else -60.0, 80.0 if i < 2 else 60.0))


def _apply_level1(m, d, ids, level1, t, with_arm=True, tau_arm=None):
    y_ref, l_ref, tag, sched_single = _phase(t)
    active = (not sched_single, True)
    com_ref = np.array([0.0, y_ref, 0.88])
    tau_gen, info = level1.torques(d, active=active, com_ref=com_ref)
    sign = -1.0
    # Leg posture terms shape the redundant joints; Level-1 GRF torques carry
    # the body wrench.  The swing leg tracks a large-clearance pose.
    for jn in LEG_JOINTS:
        base_tau = sign * tau_gen[ids["DAD"][jn]]
        if jn.startswith("left_"):
            if sched_single:
                refs = {
                    "left_hip_y": l_ref[0], "left_knee_y": l_ref[1],
                    "left_ankle_y": l_ref[2], "left_hip_x": 0.10,
                    "left_ankle_x": 0.0,
                }
                kp, kd = 700.0, 70.0
            else:
                refs = {
                    "left_hip_y": L_STANCE[0], "left_knee_y": L_STANCE[1],
                    "left_ankle_y": L_STANCE[2], "left_hip_x": 0.0,
                    "left_ankle_x": 0.0,
                }
                kp, kd = 140.0, 20.0
        else:
            refs = {
                "right_hip_y": -0.05, "right_knee_y": 0.10,
                "right_ankle_y": -0.05, "right_hip_x": 0.0,
                "right_ankle_x": 0.0,
            }
            kp, kd = 140.0, 20.0
        ref = refs.get(jn, 0.0)
        post_tau = _joint_pd(d, ids, jn, ref, kp, kd, 120.0)
        d.ctrl[ids["A"][jn]] = float(np.clip(base_tau + post_tau, -200.0, 200.0))
    if with_arm:
        _apply_arm_hold(d, ids, tau_arm)
    return active, tag, info


def _settle(m, d, ids, level1, n=3000):
    # Start from the known stable stance controller, then hand over to Level 1.
    qstatic._settle(m, d, ids, n=n)
    mujoco.mj_forward(m, d)


def run_balance_only(verbose=True):
    m, d = _make()
    ids = _setup(m)
    level1 = Level1CentroidalBalance(m, ids)
    _settle(m, d, ids, level1)
    min_z = 10.0
    fell = False
    left_contact = []
    right_contact = []
    single = []
    for step in range(N_RUN):
        t = step * CTRL_DT
        active, tag, _ = _apply_level1(m, d, ids, level1, t, with_arm=True)
        mujoco.mj_step(m, d)
        lc, rc = _foot_contacts(d, ids)
        left_contact.append(lc)
        right_contact.append(rc)
        single.append(not active[0])
        min_z = min(min_z, float(d.qpos[2]))
        if d.qpos[2] < 0.55:
            fell = True
            break
    single = np.asarray(single, dtype=bool)
    left_contact = np.asarray(left_contact, dtype=bool)
    right_contact = np.asarray(right_contact, dtype=bool)
    audit = {
        "left_single": float(np.mean(left_contact[single])) if np.any(single) else float("nan"),
        "right_single": float(np.mean(right_contact[single])) if np.any(single) else float("nan"),
        "left_double": float(np.mean(left_contact[~single])) if np.any(~single) else float("nan"),
        "right_double": float(np.mean(right_contact[~single])) if np.any(~single) else float("nan"),
    }
    if verbose:
        print(
            f"Level-1 balance-only: {'FELL' if fell else 'STOOD'} min_z={min_z:.3f} "
            f"left-on-floor single={audit['left_single']:.3f} right={audit['right_single']:.3f}"
        )
    return (not fell), min_z, audit


def run_interaction(cfg):
    m, d = _make()
    ids = _setup(m)
    level1 = Level1CentroidalBalance(m, ids)
    _settle(m, d, ids, level1)
    arm_dofs = [ids["DAD"][j] for j in ARM_JOINTS]
    p0 = (d.site_xpos[ids["hand"]] - d.xpos[ids["torso_body"]]).copy()
    mpc = ImpedanceMPC(N=20, dt=CTRL_DT, Q=Q_MPC, R=R_MPC, F_max=F_MAX)
    mpc.precompute_mode("double", 0.20 * np.eye(3))
    kal = None
    if cfg.get("use_kalman"):
        kal = KalmanDisturbanceEstimator(dt=CTRL_DT)
        kal.set_mode(mpc.A_d, mpc._mode_library["double"]["B_d"])
    prev_mode = "double"
    F_prev = np.zeros(3)
    t_log = np.zeros(N_RUN)
    e_log = np.zeros((N_RUN, 3))
    switch_times = []
    left_contact = np.zeros(N_RUN, dtype=bool)
    right_contact = np.zeros(N_RUN, dtype=bool)
    single_mode = np.zeros(N_RUN, dtype=bool)
    fell = False
    la_diag = {"double": [], "single": []}

    for step in range(N_RUN):
        t = step * CTRL_DT
        d.xfrc_applied[ids["hand_body"], :3] = F_DIST if t >= T_DIST else np.zeros(3)
        y_ref, _, _, sched_single = _phase(t)

        Jh = get_site_jacobian(m, d, ids["hand"])
        Jtorso, _ = get_body_jacobian(m, d, ids["torso_body"])
        Jrel = Jh - Jtorso
        p_rel = d.site_xpos[ids["hand"]] - d.xpos[ids["torso_body"]]
        e_pos = p_rel - p0
        e_vel = Jrel @ d.qvel
        t_log[step] = t
        e_log[step] = e_pos

        M = get_mass_matrix(m, d)
        rows = [get_site_jacobian(m, d, ids["rfoot"])]
        if not sched_single:
            rows.insert(0, get_site_jacobian(m, d, ids["lfoot"]))
        Jc = np.vstack(rows)
        Mbar = get_contact_consistent_inverse(M, Jc)
        La = get_task_inertia(Jrel, Mbar)
        mode_key = "single" if sched_single else "double"
        single_mode[step] = sched_single
        la_diag[mode_key].append(np.diag(La))
        switched = mode_key != prev_mode
        if switched and step > 0:
            switch_times.append(t)
        mode = mpc.get_or_update_mode(mode_key, La)
        d_hat = None
        if kal is not None:
            if switched and cfg.get("inflate_alpha", 1.0) > 1.0:
                kal.inflate_covariance(cfg["inflate_alpha"])
            kal.set_mode(mpc.A_d, mode["B_d"])
            kal.predict(F_prev)
            _, d_hat = kal.update(e_pos)
        F_mpc = mpc.solve(np.concatenate([e_pos, e_vel]), La, mode_key, d_hat, use_osqp=False)
        F_arm = -F_mpc
        F_prev = F_mpc
        prev_mode = mode_key
        tau_task = Jrel[:, arm_dofs].T @ F_arm

        _apply_level1(m, d, ids, level1, t, with_arm=True, tau_arm=tau_task)
        mujoco.mj_step(m, d)
        lc, rc = _foot_contacts(d, ids)
        left_contact[step] = lc
        right_contact[step] = rc
        if d.qpos[2] < 0.55:
            fell = True
            t_log = t_log[:step + 1]
            e_log = e_log[:step + 1]
            left_contact = left_contact[:step + 1]
            right_contact = right_contact[:step + 1]
            single_mode = single_mode[:step + 1]
            break

    rms = np.sqrt(np.mean(np.sum(e_log ** 2, axis=1))) * 1000
    peak = _switch_peak(t_log, e_log, switch_times)
    audit = {
        "left_single": float(np.mean(left_contact[single_mode])) if np.any(single_mode) else float("nan"),
        "right_single": float(np.mean(right_contact[single_mode])) if np.any(single_mode) else float("nan"),
        "left_double": float(np.mean(left_contact[~single_mode])) if np.any(~single_mode) else float("nan"),
        "right_double": float(np.mean(right_contact[~single_mode])) if np.any(~single_mode) else float("nan"),
    }
    return {
        "rms": rms,
        "peak": peak,
        "fell": fell,
        "switches": len(switch_times),
        "audit": audit,
        "t": t_log,
        "e": e_log,
        "sw": switch_times,
        "la_double": np.mean(la_diag["double"], axis=0) if la_diag["double"] else np.zeros(3),
        "la_single": np.mean(la_diag["single"], axis=0) if la_diag["single"] else np.zeros(3),
    }


def _switch_peak(t_log, e_log, switch_times, window=0.25):
    peaks = [
        np.max(np.linalg.norm(e_log[np.abs(t_log - ts) < window], axis=1)) * 1000
        for ts in switch_times
        if (np.abs(t_log - ts) < window).any()
    ]
    return float(np.mean(peaks)) if peaks else float("nan")


CONTROLLERS = {
    "D5 Level1 noKalman": dict(use_kalman=False),
    "D6 Level1 Kalman": dict(use_kalman=True, inflate_alpha=1.0),
    "D7 Level1 Kalman+Infl": dict(use_kalman=True, inflate_alpha=4.0),
}


if __name__ == "__main__":
    run_balance_only(verbose=True)
    print(f"\n{'Controller':<24}{'RMS [mm]':>10}{'Peak@switch [mm]':>18}{'switches':>10}")
    print("-" * 62)
    results = {}
    last = None
    for name, cfg in CONTROLLERS.items():
        r = run_interaction(cfg)
        results[name] = r
        last = r
        tag = " (FELL)" if r["fell"] else ""
        print(f"{name:<24}{r['rms']:>10.3f}{r['peak']:>18.3f}{r['switches']:>10}{tag}")
    print(
        "Contact audit:"
        f" left-on-floor single={last['audit']['left_single']:.3f},"
        f" right-on-floor single={last['audit']['right_single']:.3f},"
        f" left/right double={last['audit']['left_double']:.3f}/{last['audit']['right_double']:.3f}"
    )
    print(
        f"Lambda_arm diag [kg] double={np.round(last['la_double'], 2)} "
        f"single={np.round(last['la_single'], 2)}"
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for name, r in results.items():
        ax.plot(r["t"], np.linalg.norm(r["e"], axis=1) * 1000, label=name, lw=1.3)
    ax.axvspan(T_LIFT, T_PLACE, color="gray", alpha=0.18, label="right-foot model")
    for ts in last["sw"]:
        ax.axvline(ts, color="k", ls=":", lw=0.8)
    ax.set_title("Scenario G — experimental Level-1 centroidal balance")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("||e|| [mm]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "scenario_g_level1_results.png"
    fig.savefig(out, dpi=150)
    print(f"Figure saved -> {out}")
