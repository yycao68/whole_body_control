"""
Scenario D — Genuine support-transition benchmark.

Unlike Scenario B (force spikes with both feet planted throughout), this
scenario periodically lifts the LEFT foot so the robot actually cycles
double-support -> single-support -> double-support.  The contact set therefore
changes, which:
  * changes the contact Jacobian J_c (6 rows -> 3 rows), hence M-bar,
    Lambda_arm(q), and the input matrix B_d — genuinely exercising the
    contact-mode-indexed library keyed by the active-foot set;
  * injects a real touchdown/lift-off transient at each switch.

This is the scenario that actually tests the covariance-inflation protocol:
at each detected contact-mode switch the Kalman B_d jumps, and D7 (alpha=4)
inflates the covariance to re-estimate d_hat faster than D6 (alpha=1).

A sustained 8 N pHRI force acts on the right arm throughout.
"""
import numpy as np
import mujoco

from simulation.controllers.wbc_core import (
    WBCController, get_hand_state, get_mass_matrix, get_contact_jacobian,
    get_contact_consistent_inverse, get_task_inertia, get_site_jacobian,
    get_robot_com, get_foot_contact_flags, _get_ids,
)
from simulation.controllers.impedance_mpc import ImpedanceMPC
from simulation.controllers.kalman import KalmanDisturbanceEstimator
import simulation.scenarios.scenario_a as A   # reuse robot, settle, constants

CTRL_DT = 0.001
T_DIST  = 0.5
N_RUN   = 6000                      # 6 s
F_DIST  = np.array([8.0, 0.0, 0.0])
Q_MPC   = np.diag([3e6, 3e6, 3e6, 60., 60., 60.])
R_MPC   = 0.01 * np.eye(3)
F_MAX   = 80.0

T_CYCLE = 1.5                        # lift period [s]
T_LIFT  = 0.28                       # single-support duration per cycle [s]

KP_LEG  = A.KP_LEG
KD_LEG  = A.KD_LEG
STANCE_Q = A.STANCE_Q
# Left-leg pose that lifts the foot clear of the ground (hip_y, knee_y, ankle_y)
LIFT_Q  = STANCE_Q.copy()
LIFT_Q[1], LIFT_Q[2], LIFT_Q[3] = -0.9, 1.4, 0.5

KP_DIST, KD_PD = 800.0, 40.0


def _mode_key(mask):
    if mask[0] and mask[1]:  return 'ds'      # double support
    if mask[1]:              return 'ss_R'    # left off, right stance
    if mask[0]:              return 'ss_L'
    return 'flight'


def run_controller(name, cfg):
    model, data = A._make_robot()
    ids  = _get_ids(model)
    arm_grav = A._precompute_arm_gravity(model)
    A._settle(model, data, ids, arm_grav)

    p0, _  = get_hand_state(model, data)
    foot   = [ids['left_foot_site'], ids['right_foot_site']]
    hand_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_hand')
    use_cc = cfg.get('use_contact_consist', True)

    wbc = WBCController(model, contact_consistent=use_cc)
    mpc = kalman = None
    if cfg.get('use_mpc'):
        mpc = ImpedanceMPC(N=20, dt=CTRL_DT, Q=Q_MPC, R=R_MPC, F_max=F_MAX)
        mpc.precompute_mode('ds', 0.20 * np.eye(3))
        if cfg.get('use_kalman'):
            kalman = KalmanDisturbanceEstimator(dt=CTRL_DT)
            kalman.set_mode(mpc.A_d, mpc._mode_library['ds']['B_d'])

    F_prev = np.zeros(3)
    prev_mask = (True, True)
    t_log = np.zeros(N_RUN); e_log = np.zeros((N_RUN, 3))
    switch_times = []

    for step in range(N_RUN):
        t = step * CTRL_DT
        data.xfrc_applied[hand_body, :3] = F_DIST if t >= T_DIST else np.zeros(3)

        p_act, v_act = get_hand_state(model, data)
        e_pos = p_act - p0; e_vel = v_act
        t_log[step] = t; e_log[step] = e_pos

        # Detected contact set drives the mode key (physics-based, not scripted)
        mask     = get_foot_contact_flags(model, data, foot)
        mode_key = _mode_key(mask)
        switched = (tuple(mask) != prev_mask)
        if switched and step > 0:
            switch_times.append(t)
        prev_mask = tuple(mask)

        # Contact-consistent Lambda_arm for the CURRENT contact set
        if mpc is not None:
            M_  = get_mass_matrix(model, data)
            Jc_ = get_contact_jacobian(model, data, foot, mask)
            if use_cc and Jc_.shape[0] > 0:
                Mbar_ = get_contact_consistent_inverse(M_, Jc_)
            else:
                Mbar_ = np.linalg.inv(M_ + 1e-4 * np.eye(model.nv))
            Jarm_ = get_site_jacobian(model, data, ids['hand_site'])
            La    = get_task_inertia(Jarm_, Mbar_)
            mode  = mpc.get_or_update_mode(mode_key, La)   # contact-mode-indexed B_d

            d_hat = None
            if kalman is not None:
                if switched and cfg.get('inflate_alpha', 1.0) > 1.0:
                    kalman.inflate_covariance(cfg['inflate_alpha'])
                kalman.set_mode(mpc.A_d, mode['B_d'])
                kalman.predict(F_prev)
                _, d_hat = kalman.update(e_pos)
            F_mpc = mpc.solve(np.concatenate([e_pos, e_vel]), La,
                              mode_key, d_hat, use_osqp=False)
            F_arm = -F_mpc; F_prev = F_mpc
        else:
            F_arm = -(KP_DIST * e_pos + KD_PD * e_vel)

        # Leg reference: lift the left foot for the first T_LIFT of each cycle
        t_in    = t % T_CYCLE
        lifting = (t >= T_DIST + 0.3) and (t_in < T_LIFT)
        com, _  = get_robot_com(model, data)
        qr      = LIFT_Q.copy() if lifting else STANCE_Q.copy()
        lean    = -4.0 * com[1] - (0.08 if lifting else 0.0)
        qr[0]   = lean; qr[4] = lean

        tau, _ = wbc.compute(data, F_arm, contact_mask=list(mask),
                             q_ref_legs=qr, Kp_leg=KP_LEG, Kd_leg=KD_LEG)
        data.ctrl[:8]   = tau[:8]
        data.ctrl[8:11] = tau[8:]
        for _ in range(2):
            mujoco.mj_step(model, data)

    return t_log, e_log, switch_times


def switch_peak(t_log, e_log, switch_times, window=0.20):
    """Mean peak Cartesian error in a +/-window around each detected switch."""
    peaks = []
    for ts in switch_times:
        m = (t_log >= ts) & (t_log <= ts + window)
        if m.any():
            peaks.append(np.max(np.linalg.norm(e_log[m], axis=1)) * 1000)
    return float(np.mean(peaks)) if peaks else float('nan')


CONTROLLERS = {
    'D3 FixedBase MPC':    dict(use_mpc=True, use_kalman=True, inflate_alpha=4.0,
                                use_contact_consist=False),
    'D5 Proposed noKalman':dict(use_mpc=True, use_kalman=False),
    'D6 Kalman noInflate': dict(use_mpc=True, use_kalman=True, inflate_alpha=1.0),
    'D7 Kalman + Inflate': dict(use_mpc=True, use_kalman=True, inflate_alpha=4.0),
}


if __name__ == '__main__':
    for nm, cfg in CONTROLLERS.items():
        t, e, sw = run_controller(nm, cfg)
        rms  = np.sqrt(np.mean(np.sum(e**2, axis=1))) * 1000
        peak = switch_peak(t, e, sw)
        print(f'{nm:24s} RMS={rms:8.3f} mm  peak@switch={peak:7.3f} mm  switches={len(sw)}')
