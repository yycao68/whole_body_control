"""
Scenario B: Double-Support Stance with Periodic Contact-Transition Shocks
 - Robot in stable double-support throughout (no walking instability)
 - Constant 8 N pHRI force applied from t=0
 - Every T_SWITCH seconds, a brief additional 6 N force spike (0.10 s)
   simulates the ground-impact disturbance that accompanies a foot touchdown
 - MPC always uses the physical double-support Lambda — no artificial mismatch

Why this models the reviewer's concern:
  Physical contact transitions change both Lambda_arm AND inject a brief
  mechanical shock through the kinematic chain.  The spike here isolates
  the shock component, which is directly testable in a stable stance. Since
  the contact set itself does not change, covariance inflation is intentionally
  not triggered here; Scenario E tests the real mode-switch protocol. D2 (PI)
  must show inferior transient recovery because integral action cannot
  pre-compensate for a
  sign-invariant step change in disturbance magnitude.

Table IV of the IEEE paper.
"""

import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from wbc_core import (
    WBCController, get_hand_state, get_mass_matrix,
    get_contact_jacobian, get_contact_consistent_inverse,
    get_task_inertia, get_site_jacobian, get_site_jacobian_dot,
    get_arm_bias_force, get_robot_com,
    _get_ids,
)
from impedance_mpc import ImpedanceMPC
from kalman import KalmanDisturbanceEstimator
from gait import StanceBalance

MODEL_PATH = Path(__file__).with_name('biped.xml')
OUT_DIR    = Path(__file__).parent / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIM_DT      = 0.0005
CTRL_DT     = 0.001
T_TOTAL     = 10.0
N_SETTLE    = 4000

F_DIST_BASE = np.array([8.0, 0.0, 0.0])   # constant pHRI
F_SPIKE     = np.array([6.0, 0.0, 0.0])   # additional shock at mode switch
T_SPIKE     = 0.10                          # shock duration [s]
T_SWITCH    = 1.00                          # mode-switch period [s]

KP_LEG = np.array([200, 1000, 1000, 400,  200, 1000, 1000, 400], dtype=float)
KD_LEG = np.array([ 20,  100,  100,  40,   20,  100,  100,  40], dtype=float)

CONTROLLERS = {
    'D1_SK05_PD':          dict(use_mpc=False, use_kalman=False,
                                 Kp=800.0, Kd=40.0),
    'D2_SK05_PI':          dict(use_mpc=False, use_kalman=False, use_integral=True,
                                 Kp=800.0, Kd=40.0, Ki=150.0),
    'D3_FreeSpace_Recovery': dict(use_mpc=True, use_kalman=True,
                                 use_contact_consist=False),
    'D4_WBC_PD':           dict(use_mpc=False, use_kalman=False,
                                 Kp=800.0, Kd=40.0, use_wbc=True),
    'D5_Proposed_noKalman':dict(use_mpc=True,  use_kalman=False),
    'D6_Proposed_noInflat':dict(use_mpc=True,  use_kalman=True,  inflate_alpha=1.0),
    'D7_Proposed_Full':    dict(use_mpc=True,  use_kalman=True,  inflate_alpha=4.0),
}


def run_controller(ctrl_name, ctrl_cfg):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data  = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    ids      = _get_ids(model)
    stance   = StanceBalance()
    wbc_ctrl = WBCController(model)
    foot_ids = [ids['left_foot_site'], ids['right_foot_site']]

    # Active stance settle
    for _ in range(N_SETTLE):
        com_pos, _ = get_robot_com(model, data)
        q_ref_s, _ = stance.get_refs(0.0, com_pos[1])
        tau_s, _ = wbc_ctrl.compute(data, np.zeros(3),
                                    contact_mask=[True, True],
                                    q_ref_legs=q_ref_s,
                                    Kp_leg=KP_LEG, Kd_leg=KD_LEG)
        data.ctrl[:] = tau_s
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    p0, _ = get_hand_state(model, data)
    p0    = p0.copy()

    use_cc = ctrl_cfg.get('use_contact_consist', True)
    mpc    = None
    kalman = None
    if ctrl_cfg.get('use_mpc'):
        mpc = ImpedanceMPC(N=20, dt=CTRL_DT,
                           Q=np.diag([6e4, 6e4, 6e4, 60., 60., 60.]),
                           R=0.01 * np.eye(3), F_max=80.0)
        mpc.precompute_mode('ds', 0.20 * np.eye(3))   # nominal; real Lambda passed each step
        if ctrl_cfg.get('use_kalman'):
            kalman = KalmanDisturbanceEstimator(dt=CTRL_DT)
            kalman.set_mode(mpc.A_d, mpc._mode_library['ds']['B_d'])

    Kp = ctrl_cfg.get('Kp', 800.0)
    Kd = ctrl_cfg.get('Kd', 40.0)
    Ki = ctrl_cfg.get('Ki', 0.0)
    integral_err = np.zeros(3)
    u_prev       = np.zeros(3)

    prev_switch_idx = -1

    n_steps = int(T_TOTAL / CTRL_DT) + 1
    t_log   = np.zeros(n_steps)
    e_log   = np.zeros((n_steps, 3))
    mode_switch_times = []
    step = 0
    t    = 0.0

    hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_hand')

    while t < T_TOTAL and step < n_steps:
        p_act, v_act = get_hand_state(model, data)
        e_pos = p_act - p0
        e_vel = v_act

        # Periodic disturbance: constant base + brief spike at each mode switch
        switch_idx = int(t / T_SWITCH)
        t_in_switch = t - switch_idx * T_SWITCH
        in_spike    = (t_in_switch < T_SPIKE)
        F_applied   = F_DIST_BASE + (F_SPIKE if in_spike else np.zeros(3))
        data.xfrc_applied[hand_body_id, :3] = F_applied

        # Log shock onsets. This scenario does not change the contact set, so
        # Kalman covariance inflation is intentionally not triggered here.
        if switch_idx != prev_switch_idx and switch_idx > 0:
            mode_switch_times.append(t)
            prev_switch_idx = switch_idx

        # Compute arm force — real contact-consistent Lambda_arm(q)
        if mpc is not None:
            M_  = get_mass_matrix(model, data)
            Jc_ = get_contact_jacobian(model, data, foot_ids, [True, True])
            if use_cc:
                Mbar_ = get_contact_consistent_inverse(M_, Jc_)
            else:
                Mbar_ = np.linalg.inv(M_ + 1e-4 * np.eye(model.nv))   # D3: free-space
            Jarm_  = get_site_jacobian(model, data, ids['hand_site'])
            La_cur = get_task_inertia(Jarm_, Mbar_)
            # Task-space Coriolis/gravity bias -- see scenario_a.py's
            # comment at the same point; p_ddot_d left at zero (fixed
            # target, not a moving reference, in this scenario too).
            Jarm_dot_ = get_site_jacobian_dot(model, data, ids['hand_site'])
            mu_arm_ = get_arm_bias_force(model, data, Jarm_, Jarm_dot_, Mbar_, La_cur)
            mode = mpc.get_or_update_mode('ds', La_cur)   # constant predictor, updated recovery inertia
            d_hat = None
            if kalman:
                kalman.set_mode(mpc.A_d, mode['B_d'])     # keep Kalman model in sync
                kalman.predict(u_prev)
                _, d_hat = kalman.update(e_pos)
            x_e_vec = np.concatenate([e_pos, e_vel])
            F_mpc   = mpc.solve(x_e_vec, La_cur, mode_key='ds',
                                d_hat=d_hat, use_osqp=False, mu_arm=mu_arm_)
            F_arm      = F_mpc
            u_prev     = mpc.last_u
        else:
            F_arm = -(Kp * e_pos + Kd * e_vel)
            if Ki > 0:
                integral_err += e_pos * CTRL_DT
                integral_err  = np.clip(integral_err, -0.05, 0.05)
                F_arm        -= Ki * integral_err

        # Stance balance (always double support)
        com_pos, _ = get_robot_com(model, data)
        q_ref_legs, dq_ref_legs = stance.get_refs(0.0, com_pos[1])

        if ctrl_cfg.get('use_wbc', False) or mpc is not None:
            tau, _ = wbc_ctrl.compute(data, F_arm,
                                       contact_mask=[True, True],
                                       q_ref_legs=q_ref_legs,
                                       dq_ref_legs=dq_ref_legs,
                                       Kp_leg=KP_LEG, Kd_leg=KD_LEG)
            data.ctrl[:] = tau
        else:
            # D1/D2: J^T arm mapping (SK05 law), WBC for legs
            J_arm_full  = get_site_jacobian(model, data, ids['hand_site'])
            arm_dofs    = [ids['rshoulder_x_dof'], ids['rshoulder_y_dof'],
                           ids['relbow_y_dof']]
            tau_arm_task = J_arm_full[:, arm_dofs].T @ F_arm
            tau_arm_null = wbc_ctrl._gravity_ff.copy()
            for i, dadr in enumerate(arm_dofs):
                jname = ['right_shoulder_x','right_shoulder_y','right_elbow_y'][i]
                jid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                q  = data.qpos[model.jnt_qposadr[jid]]
                dq = data.qvel[dadr]
                tau_arm_null[i] += (wbc_ctrl.Kp_null*(wbc_ctrl._q0_arm[i]-q)
                                    - wbc_ctrl.Kd_null*dq)
            tau_arm = np.clip(tau_arm_task + tau_arm_null,
                              -wbc_ctrl._tau_max[8:], wbc_ctrl._tau_max[8:])
            tau_leg_wbc, _ = wbc_ctrl.compute(data, np.zeros(3),
                                               contact_mask=[True, True],
                                               q_ref_legs=q_ref_legs,
                                               dq_ref_legs=dq_ref_legs,
                                               Kp_leg=KP_LEG, Kd_leg=KD_LEG)
            data.ctrl[:8]   = tau_leg_wbc[:8]
            data.ctrl[8:11] = tau_arm

        for _ in range(max(1, int(CTRL_DT / SIM_DT))):
            mujoco.mj_step(model, data)

        t_log[step] = t
        e_log[step] = e_pos

        t    += CTRL_DT
        step += 1

    return t_log[:step], e_log[:step], mode_switch_times


def compute_metrics(t_log, e_log, switch_times, window=0.15):
    rms = np.sqrt(np.mean(np.sum(e_log**2, axis=1))) * 1000
    peaks = []
    e_norm = np.linalg.norm(e_log, axis=1)
    for ts in switch_times:
        mask = np.abs(t_log - ts) < window
        if mask.any():
            peaks.append(np.max(e_norm[mask]))
    peak_trans = np.mean(peaks) * 1000 if peaks else 0.0
    return rms, peak_trans


def run_all():
    print("\n=== Scenario B: Stance + Periodic 6 N Disturbance Spikes at 1 Hz ===\n")
    print(f"{'Controller':<28} {'RMS [mm]':>10} {'Peak@trans [mm]':>18}")
    print("-" * 60)

    results = {}
    for name, cfg in CONTROLLERS.items():
        t_log, e_log, sw_times = run_controller(name, cfg)
        rms, peak = compute_metrics(t_log, e_log, sw_times)
        results[name] = dict(t=t_log, e=e_log, sw=sw_times, rms=rms, peak=peak)
        print(f"{name:<28} {rms:>10.2f} {peak:>18.2f}")

    # Readability subset: classical baselines vs full proposed controller.
    # D4–D6 are reported in the table only.
    PLOT_SET = ['D1_SK05_PD', 'D2_SK05_PI', 'D3_FreeSpace_Recovery', 'D7_Proposed_Full']
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(CONTROLLERS)))
    names  = PLOT_SET

    for i, (name, res) in enumerate(results.items()):
        if name not in PLOT_SET:
            continue
        e_norm = np.linalg.norm(res['e'], axis=1) * 1000
        axes[0].plot(res['t'], e_norm, label=name, color=colors[i], lw=1.4)
    for ts in results[names[-1]]['sw']:
        axes[0].axvline(ts, color='gray', ls=':', lw=0.8, alpha=0.6)
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('||e|| [mm]')
    axes[0].set_title('Scenario B — Stance + Periodic Contact-Transition Shocks (1 Hz)\n'
                      'Sustained 8 N pHRI + 6 N spike every 1 s for 0.10 s')
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)

    short  = [n.split('_')[0]+'_'+n.split('_')[1] for n in names]
    rms_v  = [results[n]['rms']  for n in names]
    pk_v   = [results[n]['peak'] for n in names]
    x      = np.arange(len(names))
    w      = 0.38
    axes[1].bar(x - w/2, rms_v, w, label='RMS error',       color='steelblue', alpha=0.85)
    axes[1].bar(x + w/2, pk_v,  w, label='Peak@transition',  color='tomato',    alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short, rotation=20, fontsize=8)
    axes[1].set_ylabel('[mm]')
    axes[1].set_title('RMS Error and Peak Transition Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'scenario_b_results.png', dpi=150)
    print(f"\nFigure saved to {OUT_DIR}/scenario_b_results.png")
    plt.close(fig)

    return results


if __name__ == '__main__':
    run_all()
