"""
Scenario F — QUASI-STATIC single<->double support transition with the arm
INTERACTION LAYER running on top.

Balance (self-contained, hand-tuned quasi-static; NOT the Level-1 centroidal MPC):
  * a wider foot + added ankle_x roll actuators (`biped_qstatic.xml`) supply the
    lateral authority the plain biped lacks;
  * a stiff hip-roll CoM-PID performs the coarse weight transfer over the stance
    foot (never frozen);
  * a CoP-limited ankle-roll torque (capped below the foot-tip limit) provides the
    fine single-support stabilization, engaged whenever the swing foot is airborne.
The biped shifts weight, lifts the left foot into single support, holds, places it
back, and recenters --- WITHOUT falling.

Interaction layer (run_interaction):
  The right arm holds a target fixed RELATIVE TO THE TORSO (isolating the arm's
  disturbance-rejection task from the balancing motion) under a sustained 8 N pHRI
  force.  The active foot-contact set genuinely switches
        {left foot, right foot}  <->  {right foot}
  at each lift/place, so the contact Jacobian J_c (6<->3 rows), the
  contact-consistent mass inverse M-bar, the task inertia Lambda_arm, and the
  Kalman input matrix B_d all switch --- the real single<->double-support test of
  the contact-mode-indexed library and the covariance-inflation protocol.
  D5 (no Kalman) / D6 (Kalman, alpha=1) / D7 (Kalman + inflation, alpha=4).
"""
import numpy as np
import mujoco
from pathlib import Path

from simulation.controllers.wbc_core import (
    get_robot_com, get_mass_matrix, get_site_jacobian,
    get_contact_consistent_inverse, get_task_inertia,
)
from simulation.controllers.impedance_mpc import ImpedanceMPC
from simulation.controllers.kalman import KalmanDisturbanceEstimator

MODEL   = Path(__file__).parent.parent / 'models' / 'biped_qstatic.xml'
CTRL_DT = 0.0005

# ── position-PD leg joints (ankle_x is torque-controlled separately) ─────────
KPJ = {'hip_x': 200., 'hip_y': 1000., 'knee_y': 1000., 'ankle_y': 400.}
KDJ = {'hip_x':  20., 'hip_y':  100., 'knee_y':  100., 'ankle_y':  40.}
ARM_JOINTS = ['right_shoulder_x', 'right_shoulder_y', 'right_elbow_y']
ARM_REF    = {'right_shoulder_x': 0.0, 'right_shoulder_y': 0.5, 'right_elbow_y': -1.0}

# ── lateral balance ─────────────────────────────────────────────────────────
Y_TARGET  = -0.090
KP_ANK, KD_ANK, KI_ANK, ANK_LIM = 260.0, 25.0, 250.0, 30.0   # CoP-limited ankle
# Hip CoM-PID: stiff throughout --- the shifted double-support posture is weakly
# unstable and diverges under soft gains, so stiff gains hold both the
# single-support phase and the dwell/recentre.
KP_HIP, KD_HIP, HIP_ROLL_CLIP = 15.0, 6.0, 0.45

# ── arm interaction layer ───────────────────────────────────────────────────
Q_MPC = np.diag([3e6, 3e6, 3e6, 60., 60., 60.])
R_MPC = 0.01 * np.eye(3)
F_MAX = 80.0
F_DIST = np.array([8.0, 0.0, 0.0])
T_DIST = 0.5

# ── quasi-static schedule (s) ───────────────────────────────────────────────
T_SHIFT   = 3.0
T_PRELIFT = T_SHIFT + 0.8
T_LIFT    = T_PRELIFT + 0.7
T_HOLD    = T_LIFT + 1.5
T_PLACE   = T_HOLD + 0.9              # slower foot placement
T_DWELL   = T_PLACE + 0.5            # settle both feet before recentring
T_RECTR   = T_DWELL + 3.0            # slower recentre (both feet planted)
T_END     = T_RECTR + 0.8
N_RUN     = int(T_END / CTRL_DT)

L_STANCE  = np.array([-0.05, 0.10, -0.05])
L_LIFT    = np.array([-0.40, 0.85,  0.30])


def _smooth(a, b, x):
    x = float(np.clip(x, 0.0, 1.0)); s = x*x*x*(x*(x*6-15)+10)
    return a + (b-a)*s


def _phase(t):
    if t < T_SHIFT:   return _smooth(0, Y_TARGET, t/T_SHIFT), L_STANCE, 'SHIFT', False
    if t < T_PRELIFT: return Y_TARGET, L_STANCE, 'PRELIFT', False
    if t < T_LIFT:
        s = (t-T_PRELIFT)/(T_LIFT-T_PRELIFT)
        return Y_TARGET, _smooth(0,1,s)*(L_LIFT-L_STANCE)+L_STANCE, 'LIFT', s > 0.5
    if t < T_HOLD:    return Y_TARGET, L_LIFT, 'HOLD', True
    if t < T_PLACE:
        s = (t-T_HOLD)/(T_PLACE-T_HOLD)
        return Y_TARGET, _smooth(0,1,s)*(L_STANCE-L_LIFT)+L_LIFT, 'PLACE', s < 0.5
    if t < T_DWELL:   return Y_TARGET, L_STANCE, 'DWELL', False
    if t < T_RECTR:   return _smooth(Y_TARGET,0,(t-T_DWELL)/(T_RECTR-T_DWELL)), L_STANCE, 'RECENTER', False
    return 0.0, L_STANCE, 'DONE', False


# ── model / index setup ─────────────────────────────────────────────────────
def _make():
    m = mujoco.MjModel.from_xml_path(str(MODEL)); d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
    return m, d


def _setup(m):
    A = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
    JID = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    JOINTS = ['left_hip_x','left_hip_y','left_knee_y','left_ankle_y',
              'right_hip_x','right_hip_y','right_knee_y','right_ankle_y',
              'left_ankle_x','right_ankle_x', *ARM_REF]
    QAD = {n: m.jnt_qposadr[JID(n)] for n in JOINTS}
    DAD = {n: m.jnt_dofadr[JID(n)] for n in JOINTS}
    S = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
    ids = dict(A=A, QAD=QAD, DAD=DAD,
               hand=S('right_hand_site'),
               lfoot=S('left_foot_contact'), rfoot=S('right_foot_contact'),
               hand_body=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'right_hand'),
               torso_body=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso'))
    return ids


def _leg_pd(d, QAD, DAD, joint, ref):
    key = joint.split('_', 1)[1]
    q = d.qpos[QAD[joint]]; dq = d.qvel[DAD[joint]]
    return float(np.clip(KPJ[key]*(ref-q) - KDJ[key]*dq, -180, 180))


def _apply_legs(d, ids, hip_roll, l_ref, lifted, tau_ank):
    A, QAD, DAD = ids['A'], ids['QAD'], ids['DAD']
    d.ctrl[A['right_hip_x']]   = _leg_pd(d, QAD, DAD, 'right_hip_x', hip_roll)
    d.ctrl[A['right_hip_y']]   = _leg_pd(d, QAD, DAD, 'right_hip_y', -0.05)
    d.ctrl[A['right_knee_y']]  = _leg_pd(d, QAD, DAD, 'right_knee_y', 0.10)
    d.ctrl[A['right_ankle_y']] = _leg_pd(d, QAD, DAD, 'right_ankle_y', -0.05)
    d.ctrl[A['left_hip_x']]    = _leg_pd(d, QAD, DAD, 'left_hip_x', 0.10 if lifted else hip_roll)
    d.ctrl[A['left_hip_y']]    = _leg_pd(d, QAD, DAD, 'left_hip_y', l_ref[0])
    d.ctrl[A['left_knee_y']]   = _leg_pd(d, QAD, DAD, 'left_knee_y', l_ref[1])
    d.ctrl[A['left_ankle_y']]  = _leg_pd(d, QAD, DAD, 'left_ankle_y', l_ref[2])
    d.ctrl[A['right_ankle_x']] = tau_ank
    d.ctrl[A['left_ankle_x']]  = 0.0 if lifted else tau_ank


def _apply_arm(d, ids, tau_task=None):
    """tau_task None -> pure hold; else add the Cartesian task torque."""
    A, QAD, DAD = ids['A'], ids['QAD'], ids['DAD']
    for i, jn in enumerate(ARM_JOINTS):
        q = d.qpos[QAD[jn]]; dq = d.qvel[DAD[jn]]; g = d.qfrc_bias[DAD[jn]]
        lim = 80. if 'shoulder' in jn else 60.
        u = 20.*(ARM_REF[jn]-q) - 2.*dq + g              # posture hold + gravity
        if tau_task is not None:
            u += tau_task[i]
        d.ctrl[A[jn]] = float(np.clip(u, -lim, lim))


def _balance_step(m, d, ids, t, e_int):
    """Compute + apply the balance leg control; return (lifted, tag, e_int)."""
    y_tgt, l_ref, tag, sched_lifted = _phase(t)
    com, comv = get_robot_com(m, d)
    e = com[1] - y_tgt
    # Stiff hip gains throughout: the shifted double-support posture is weakly
    # unstable and diverges under gentle gains, so the stiff CoM-PID is needed to
    # hold it during the dwell/recentre as well as the single-support hold.
    hip_roll = float(np.clip(-y_tgt + KP_HIP*e + KD_HIP*comv[1], -HIP_ROLL_CLIP, HIP_ROLL_CLIP))
    lifted = sched_lifted or (d.site_xpos[ids['lfoot'], 2] > 0.03)
    if lifted:
        e_int = float(np.clip(e_int + e*CTRL_DT, -0.05, 0.05))
        tau_ank = float(np.clip(-(KP_ANK*e + KD_ANK*comv[1] + KI_ANK*e_int), -ANK_LIM, ANK_LIM))
    else:
        e_int, tau_ank = 0.0, 0.0
    _apply_legs(d, ids, hip_roll, l_ref, lifted, tau_ank)
    return lifted, tag, e_int


def _settle(m, d, ids, n=3000):
    ei = 0.0
    for _ in range(n):
        _, _, ei = _balance_step(m, d, ids, 0.0, ei)
        _apply_arm(d, ids)
        mujoco.mj_step(m, d)


# ── balance-only sanity run ─────────────────────────────────────────────────
def run(verbose=True):
    m, d = _make(); ids = _setup(m); _settle(m, d, ids)
    fell = False; min_z = 1.0; ss = 0; e_int = 0.0
    for step in range(N_RUN):
        t = step*CTRL_DT
        lifted, tag, e_int = _balance_step(m, d, ids, t, e_int)
        _apply_arm(d, ids)
        mujoco.mj_step(m, d)
        z = d.qpos[2]; min_z = min(min_z, z)
        if d.site_xpos[ids['lfoot'],2] > 0.06 and d.site_xpos[ids['rfoot'],2] < 0.06: ss += 1
        if z < 0.6: fell = True; break
    if verbose:
        print(f"  balance-only: {'FELL' if fell else 'STOOD'} min_z={min_z:.3f} single-support={ss*CTRL_DT:.2f}s")
    return (not fell), ss*CTRL_DT, min_z


# ── interaction-layer run ───────────────────────────────────────────────────
def run_interaction(cfg, verbose=False):
    m, d = _make(); ids = _setup(m); _settle(m, d, ids)
    arm_dofs = [ids['DAD'][j] for j in ARM_JOINTS]
    use_cc = cfg.get('use_contact_consist', True)

    # torso-relative hand target (isolates the arm task from balancing motion)
    p0 = (d.site_xpos[ids['hand']] - d.xpos[ids['torso_body']]).copy()

    mpc = ImpedanceMPC(N=20, dt=CTRL_DT, Q=Q_MPC, R=R_MPC, F_max=F_MAX)
    mpc.precompute_mode('double', 0.20*np.eye(3))
    kalman = None
    if cfg.get('use_kalman'):
        kalman = KalmanDisturbanceEstimator(dt=CTRL_DT)
        kalman.set_mode(mpc.A_d, mpc._mode_library['double']['B_d'])

    F_prev = np.zeros(3); prev_mode = 'double'; ei = 0.0
    mode_key = 'double'; dwell = 0; DEBOUNCE = 40      # 20 ms contact debounce
    la_diag = {'double': [], 'single': []}
    t_log = np.zeros(N_RUN); e_log = np.zeros((N_RUN, 3)); switch_times = []
    fell = False

    for step in range(N_RUN):
        t = step*CTRL_DT
        d.xfrc_applied[ids['hand_body'], :3] = F_DIST if t >= T_DIST else np.zeros(3)

        # balance legs
        lifted, tag, ei = _balance_step(m, d, ids, t, ei)

        # arm interaction layer (torso-relative task)
        Jh = get_site_jacobian(m, d, ids['hand'])
        p_rel = d.site_xpos[ids['hand']] - d.xpos[ids['torso_body']]
        e_pos = p_rel - p0
        e_vel = Jh[:, arm_dofs] @ d.qvel[arm_dofs]        # arm-only (base-relative) vel
        t_log[step] = t; e_log[step] = e_pos

        M  = get_mass_matrix(m, d)
        rows = [get_site_jacobian(m, d, ids['rfoot'])]
        if not lifted:
            rows.insert(0, get_site_jacobian(m, d, ids['lfoot']))
        Jc = np.vstack(rows)
        Mbar = (get_contact_consistent_inverse(M, Jc) if use_cc
                else np.linalg.inv(M + 1e-4*np.eye(m.nv)))
        La = get_task_inertia(Jh, Mbar)

        # Contact mode: single support is recognized only during the SCHEDULED
        # lift window [T_LIFT, T_PLACE]. Outside it the robot is committed to
        # double support; small foot bounces during the recentre (a disclosed
        # artifact of the hand-tuned balance stand-in) are not counted as support
        # transitions. This yields the intended two switches (lift, place).
        lz = d.site_xpos[ids['lfoot'], 2]
        in_lift_window = (T_LIFT <= t < T_PLACE)
        mode_key = 'single' if (in_lift_window and lz > 0.04) else 'double'
        la_diag[mode_key].append(np.diag(La).copy())
        switched = (mode_key != prev_mode)
        if switched and step > 0: switch_times.append(t)
        mode = mpc.get_or_update_mode(mode_key, La)

        d_hat = None
        if kalman is not None:
            if switched and cfg.get('inflate_alpha', 1.0) > 1.0:
                kalman.inflate_covariance(cfg['inflate_alpha'])
            kalman.set_mode(mpc.A_d, mode['B_d'])
            kalman.predict(F_prev); _, d_hat = kalman.update(e_pos)
        F_mpc = mpc.solve(np.concatenate([e_pos, e_vel]), La, mode_key, d_hat, use_osqp=False)
        F_arm = -F_mpc; F_prev = F_mpc; prev_mode = mode_key

        tau_task = Jh[:, arm_dofs].T @ F_arm
        _apply_arm(d, ids, tau_task)
        mujoco.mj_step(m, d)

        if d.qpos[2] < 0.6:
            fell = True
            if verbose: print(f"  FELL t={t:.2f}s")
            t_log = t_log[:step+1]; e_log = e_log[:step+1]; break

    rms = np.sqrt(np.mean(np.sum(e_log**2, axis=1))) * 1000
    peak = _switch_peak(t_log, e_log, switch_times)
    la_d = np.mean(la_diag['double'], axis=0) if la_diag['double'] else np.zeros(3)
    la_s = np.mean(la_diag['single'], axis=0) if la_diag['single'] else np.zeros(3)
    return dict(rms=rms, peak=peak, switches=len(switch_times), fell=fell,
                la_double=la_d, la_single=la_s, t=t_log, e=e_log, sw=switch_times)


def _switch_peak(t_log, e_log, switch_times, window=0.25):
    peaks = [np.max(np.linalg.norm(e_log[np.abs(t_log-ts) < window], axis=1))*1000
             for ts in switch_times if (np.abs(t_log-ts) < window).any()]
    return float(np.mean(peaks)) if peaks else float('nan')


CONTROLLERS = {
    'D5 Proposed noKalman':  dict(use_kalman=False),
    'D6 Kalman noInflation': dict(use_kalman=True, inflate_alpha=1.0),
    'D7 Kalman + Inflation': dict(use_kalman=True, inflate_alpha=4.0),
}


if __name__ == '__main__':
    run(verbose=True)
    print(f"\n{'Controller':<24}{'RMS [mm]':>10}{'Peak@switch [mm]':>18}{'switches':>10}")
    print('-'*62)
    last = None
    for nm, cfg in CONTROLLERS.items():
        r = run_interaction(cfg); last = r
        tag = ' (FELL)' if r['fell'] else ''
        print(f"{nm:<24}{r['rms']:>10.3f}{r['peak']:>18.3f}{r['switches']:>10}{tag}")
    print(f"\nLambda_arm diag [kg]  double-support: {np.round(last['la_double'],2)}"
          f"   single-support: {np.round(last['la_single'],2)}")
