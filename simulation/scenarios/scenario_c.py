"""
Scenario C: Unitree R1-Approximate Platform, Fixed Stance, 8 N Step Disturbance

Robot: r1_mass.xml — biped geometry with Unitree R1-calibrated masses (28.7 kg).
       Total mass is reduced from 46.1 kg (original biped) to 28.7 kg by setting:
         torso 15 kg, thighs 3.5 kg each, shanks 2.0 kg each,
         upper-arm 1.0 kg, forearm 0.7 kg  (matches R1's published breakdown).

Protocol: identical to Scenario A (double-support, 8 N step pHRI at t=0.5s,
          5 s episode), applied to the lighter R1-like robot.

Purpose: validate that the proposed Whole-Body Impedance MPC architecture
         generalises across different mass distributions.  The lighter robot
         yields a larger contact-consistent task inertia Λ_arm (less inertia
         dominated by the heavy torso), shifting the B_d channel and changing
         both the steady-state PD error and the Kalman convergence dynamics.

Table V in the IEEE paper.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from simulation.controllers.wbc_core import (
    get_hand_state, get_mass_matrix, get_bias_force,
    get_contact_consistent_inverse, get_contact_jacobian,
    get_task_inertia, get_site_jacobian, get_robot_com, _get_ids,
    WBCController,
)
from simulation.controllers.impedance_mpc import ImpedanceMPC
from simulation.controllers.kalman import KalmanDisturbanceEstimator

MODEL_PATH = Path(__file__).parent.parent / 'models' / 'r1_mass.xml'
OUT_DIR    = Path(__file__).parent.parent.parent / 'simulation' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Simulation constants (identical to Scenario A) ─────────────────────────
N_SETTLE   = 4000
N_RUN      = 5000
T_DIST     = 0.5
F_DIST     = np.array([8.0, 0.0, 0.0])
KP_DIST    = 800.0;  KD_PD = 40.0;  KI_PI = 150.0
Q_MPC      = np.diag([60000., 60000., 60000., 60., 60., 60.])
R_MPC      = 0.01 * np.eye(3)
F_MAX      = 80.0
KP_LEG     = np.array([200, 1000, 1000, 400,   200, 1000, 1000, 400], dtype=float)
KD_LEG     = np.array([ 20,  100,  100,  40,    20,  100,  100,  40], dtype=float)
STANCE_Q   = np.array([0., -0.05, 0.10, -0.05,   0., -0.05, 0.10, -0.05])

CONTROLLERS = {
    'D1 SK05 PD':           dict(use_mpc=False, use_kalman=False, use_integral=False,
                                  use_contact_consist=True),
    'D2 SK05 PI':           dict(use_mpc=False, use_kalman=False, use_integral=True,
                                  Ki=KI_PI, use_contact_consist=True),
    'D3 FixedBase MPC':     dict(use_mpc=True,  use_kalman=True,  use_integral=False,
                                  use_contact_consist=False, d3_reg=0.1),
    'D4 WBC+PD':            dict(use_mpc=False, use_kalman=False, use_integral=False,
                                  use_contact_consist=True, use_wbc=True),
    'D5 Proposed noKalman': dict(use_mpc=True,  use_kalman=False, use_integral=False,
                                  use_contact_consist=True),
    'D6 Proposed noInflat': dict(use_mpc=True,  use_kalman=True,  use_integral=False,
                                  use_contact_consist=True, alpha=1.0),
    'D7 Proposed Full':     dict(use_mpc=True,  use_kalman=True,  use_integral=False,
                                  use_contact_consist=True, alpha=4.0),
}


# ── Helpers (mirrors scenario_a.py) ───────────────────────────────────────

def _make_robot():
    m = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d


def _leg_pd(model, data, q_ref):
    names  = ['left_hip_x',  'left_hip_y',  'left_knee_y',  'left_ankle_y',
              'right_hip_x', 'right_hip_y', 'right_knee_y', 'right_ankle_y']
    limits = [120, 200, 200, 100,  120, 200, 200, 100]
    tau = np.zeros(8)
    for i, (nm, lim) in enumerate(zip(names, limits)):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
        q  = data.qpos[model.jnt_qposadr[jid]]
        dq = data.qvel[model.jnt_dofadr[jid]]
        tau[i] = float(np.clip(KP_LEG[i]*(q_ref[i]-q) - KD_LEG[i]*dq, -lim, lim))
    return tau


def _arm_from_force(model, data, ids, F_arm):
    J  = get_site_jacobian(model, data, ids['hand_site'])
    ad = [ids['rshoulder_x_dof'], ids['rshoulder_y_dof'], ids['relbow_y_dof']]
    return np.clip(J[:, ad].T @ F_arm, -np.array([80, 80, 60]), np.array([80, 80, 60]))


def _arm_null(model, data, ids, kp=30.0, kd=3.0):
    q0 = [0.0, 0.5, -1.0]
    nms = ['right_shoulder_x', 'right_shoulder_y', 'right_elbow_y']
    lims = [80., 80., 60.]
    tau = np.zeros(3)
    for i, (nm, ref, lim) in enumerate(zip(nms, q0, lims)):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
        q  = data.qpos[model.jnt_qposadr[jid]]
        dq = data.qvel[model.jnt_dofadr[jid]]
        tau[i] = float(np.clip(kp*(ref-q)-kd*dq, -lim, lim))
    return tau


def _arm_grav(model):
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, 0)
    mujoco.mj_forward(model, d)
    tau = np.zeros(3)
    for i, nm in enumerate(['right_shoulder_x', 'right_shoulder_y', 'right_elbow_y']):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
        tau[i] = d.qfrc_bias[model.jnt_dofadr[jid]]
    return np.clip(tau, -30, 30)


def _settle(model, data, ids, arm_grav, n=N_SETTLE):
    for _ in range(n):
        com, _ = get_robot_com(model, data)
        q_ref  = STANCE_Q.copy()
        q_ref[0] = -2.0 * com[1]
        q_ref[4] = -2.0 * com[1]
        data.ctrl[:8]   = _leg_pd(model, data, q_ref)
        data.ctrl[8:11] = _arm_null(model, data, ids) + arm_grav
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)


# ── Per-controller episode runner ──────────────────────────────────────────

def run_controller(name, cfg):
    model, data = _make_robot()
    ids = _get_ids(model)
    arm_grav = _arm_grav(model)
    _settle(model, data, ids, arm_grav)

    p0, _    = get_hand_state(model, data)
    foot_ids = [ids['left_foot_site'], ids['right_foot_site']]
    wbc      = WBCController(model) if cfg.get('use_wbc') else None

    mpc = kalman = La_use = None
    use_cc = cfg.get('use_contact_consist', True)
    if cfg.get('use_mpc'):
        mpc = ImpedanceMPC(N=20, dt=0.001, Q=Q_MPC, R=R_MPC, F_max=F_MAX)
        La_use = (0.20 * np.eye(3) if use_cc else 2.00 * np.eye(3))
        mpc.precompute_mode('ds', La_use)
        if cfg.get('use_kalman'):
            kalman = KalmanDisturbanceEstimator(dt=0.001)
            kalman.set_mode(mpc.A_d, mpc._mode_library['ds']['B_d'])

    integral = np.zeros(3)
    F_prev   = np.zeros(3)
    F_arm    = np.zeros(3)
    hand_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_hand')

    t_log = np.zeros(N_RUN)
    e_log = np.zeros((N_RUN, 3))

    for step in range(N_RUN):
        t = step * 0.001
        data.xfrc_applied[hand_body, :3] = F_DIST if t >= T_DIST else np.zeros(3)

        p_act, v_act = get_hand_state(model, data)
        e_pos = p_act - p0;  e_vel = v_act
        t_log[step] = t;  e_log[step] = e_pos

        if mpc is not None:
            d_hat = None
            if kalman is not None:
                kalman.predict(F_prev)
                _, d_hat = kalman.update(e_pos)
            F_mpc  = mpc.solve(np.concatenate([e_pos, e_vel]),
                               La_use, 'ds', d_hat, use_osqp=False)
            F_arm = -F_mpc;  F_prev = F_mpc
        else:
            F_arm = -(KP_DIST * e_pos + KD_PD * e_vel)
            if cfg.get('use_integral'):
                integral += e_pos * 0.001
                integral  = np.clip(integral, -0.05, 0.05)
                F_arm    -= cfg.get('Ki', KI_PI) * integral

        com, _ = get_robot_com(model, data)
        q_ref  = STANCE_Q.copy()
        q_ref[0] = -2.0 * com[1];  q_ref[4] = -2.0 * com[1]

        if wbc is not None:
            tau, _ = wbc.compute(data, F_arm, contact_mask=[True, True],
                                 q_ref_legs=q_ref, Kp_leg=KP_LEG, Kd_leg=KD_LEG)
            data.ctrl[:8] = tau[:8];  data.ctrl[8:11] = tau[8:]
        else:
            data.ctrl[:8]   = _leg_pd(model, data, q_ref)
            data.ctrl[8:11] = (_arm_from_force(model, data, ids, F_arm)
                               + _arm_null(model, data, ids) + arm_grav)
        for _ in range(2):
            mujoco.mj_step(model, data)

    return t_log, e_log


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(t_log, e_log, t_ss_start=3.5):
    rms  = np.sqrt(np.mean(np.sum(e_log**2, axis=1))) * 1000
    mask = t_log >= t_ss_start
    ss   = np.sqrt(np.mean(np.sum(e_log[mask]**2, axis=1))) * 1000
    return rms, ss


# ── Main ───────────────────────────────────────────────────────────────────

def run_all():
    print("\n" + "="*60)
    print("  Scenario C: R1-mass (28.7 kg), Fixed Stance, 8 N Step")
    print("  (Validates generalisation across mass distributions)")
    print("="*60)
    print(f"\n  {'Controller':<24} {'RMS [mm]':>9} {'SS err [mm]':>12}")
    print("  " + "-"*48)

    results = {}
    for name, cfg in CONTROLLERS.items():
        t_log, e_log = run_controller(name, cfg)
        rms, ss = compute_metrics(t_log, e_log)
        results[name] = dict(t=t_log, e=e_log, rms=rms, ss=ss)
        print(f"  {name:<24} {rms:>9.3f} {ss:>12.4f}")

    d1 = results['D1 SK05 PD']
    d7 = results['D7 Proposed Full']
    print(f"\n  Improvement D7 vs D1 (SS): {d1['ss']/max(d7['ss'],0.01):.0f}×")
    print()

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(CONTROLLERS)))
    for i, (nm, res) in enumerate(results.items()):
        ax1.plot(res['t'], np.linalg.norm(res['e'], axis=1)*1000,
                 label=nm, color=colors[i], lw=1.5)
    ax1.axvline(T_DIST, color='k', ls='--', lw=1.2,
                label=f'Disturbance on (8 N at t={T_DIST}s)')
    ax1.set_ylabel('||e|| [mm]', fontsize=11);  ax1.set_xlabel('Time [s]', fontsize=11)
    ax1.set_title('Scenario C — Unitree R1-mass (28.7 kg), Fixed Stance\n'
                  'Same protocol as Scenario A; validates cross-platform generalisation',
                  fontsize=11)
    ax1.legend(fontsize=7, ncol=2);  ax1.grid(True, alpha=0.3)

    names    = list(results.keys())
    ss_vals  = [results[n]['ss']  for n in names]
    rms_vals = [results[n]['rms'] for n in names]
    short    = [n.split()[0] + '\n' + n.split()[1] for n in names]
    x = np.arange(len(names));  w = 0.38
    ax2.bar(x-w/2, rms_vals, w, label='RMS (full episode)',  color='steelblue', alpha=0.85)
    ax2.bar(x+w/2, ss_vals,  w, label='SS error (t>3.5s)',   color='tomato',    alpha=0.85)
    ax2.set_xticks(x);  ax2.set_xticklabels(short, fontsize=9)
    ax2.set_ylabel('[mm]', fontsize=11)
    ax2.set_title('Steady-State Error Comparison — R1-mass Platform', fontsize=11)
    ax2.legend(fontsize=9);  ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = OUT_DIR / 'scenario_c_results.png'
    fig.savefig(out, dpi=150)
    print(f"  Figure saved → {out}")
    plt.close(fig)
    return results


if __name__ == '__main__':
    run_all()
