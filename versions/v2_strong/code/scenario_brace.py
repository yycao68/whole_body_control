"""
Scenario E — Bracing-hand support-transition benchmark.

A genuine contact-mode switch **without** single-support instability: the robot
keeps both feet planted while its LEFT hand periodically braces against / releases
a fixed rail.  The active contact set therefore alternates

    {left foot, right foot}   <->   {left foot, right foot, left hand}

which changes the contact Jacobian J_c (6 <-> 9 rows), the contact-consistent
mass inverse M-bar, the RIGHT-arm task inertia Lambda_arm, and hence the
input matrix B_d -- genuinely exercising the contact-mode-indexed library and
the Kalman covariance-inflation protocol (D6: alpha=1 vs D7: alpha=4).

A sustained 8 N pHRI force acts on the RIGHT (task) arm throughout.
"""
import numpy as np
import mujoco
from pathlib import Path

from wbc_core import (
    get_mass_matrix, get_site_jacobian, get_contact_consistent_inverse,
    get_task_inertia, get_robot_com,
)
from impedance_mpc import ImpedanceMPC
from kalman import KalmanDisturbanceEstimator

MODEL = Path(__file__).with_name('biped_brace.xml')
CTRL_DT = 0.001
N_RUN   = 8000                       # 8 s
T_DIST  = 0.5
F_DIST  = np.array([8.0, 0.0, 0.0])
Q_MPC   = np.diag([6e4, 6e4, 6e4, 60., 60., 60.])
R_MPC   = 0.01 * np.eye(3)
F_MAX   = 80.0

T_CYCLE = 2.0                         # brace period [s]
T_BRACE = 0.9                         # braced fraction of each cycle [s]

LEG   = ['left_hip_x','left_hip_y','left_knee_y','left_ankle_y',
         'right_hip_x','right_hip_y','right_knee_y','right_ankle_y']
RARM  = ['right_shoulder_x','right_shoulder_y','right_elbow_y']
LARM  = ['left_shoulder_x','left_shoulder_y','left_elbow_y']
LEG_REF = np.array([0,-0.05,0.10,-0.05, 0,-0.05,0.10,-0.05])
KPL = np.array([200,1000,1000,400,200,1000,1000,400.]); KDL = KPL*0.1
RARM_REF = np.array([0,0.5,-1.0])
POSE_RELEASE = np.array([0.0,  0.30, -0.8])
POSE_BRACE   = np.array([0.0, -1.15, -0.75])


def _ids(m):
    S = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
    A = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    G = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    jq = lambda n: (m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)],
                    m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
    d = dict(hand=S('right_hand_site'), lhand=S('left_hand_site'),
             lfoot=S('left_foot_contact'), rfoot=S('right_foot_contact'),
             lhand_geom=G('left_hand_geom'), rail_geom=G('rail'),
             hand_body=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'right_hand'))
    d['leg_q']  = [jq(n) for n in LEG];  d['leg_a']  = [A(n) for n in LEG]
    d['rarm_q'] = [jq(n) for n in RARM]; d['rarm_a'] = [A(n) for n in RARM]
    d['larm_q'] = [jq(n) for n in LARM]; d['larm_a'] = [A(n) for n in LARM]
    d['rarm_dof'] = [q[1] for q in d['rarm_q']]
    return d


def _braced(m, data, ids):
    return any({data.contact[i].geom1, data.contact[i].geom2} ==
               {ids['lhand_geom'], ids['rail_geom']} for i in range(data.ncon))


def _contact_jac(m, data, ids, braced):
    rows = [get_site_jacobian(m, data, ids['lfoot']),
            get_site_jacobian(m, data, ids['rfoot'])]
    if braced:
        rows.append(get_site_jacobian(m, data, ids['lhand']))   # +3 rows
    return np.vstack(rows)


def run_controller(name, cfg):
    m = mujoco.MjModel.from_xml_path(str(MODEL)); data = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, data, 0); mujoco.mj_forward(m, data)
    ids = _ids(m)
    use_cc = cfg.get('use_contact_consist', True)

    # settle standing with left arm released
    for _ in range(3000):
        _apply(m, data, ids, np.zeros(3), POSE_RELEASE)
        mujoco.mj_step(m, data)
    p0 = data.site_xpos[ids['hand']].copy()

    mpc = kalman = None
    if cfg.get('use_mpc'):
        mpc = ImpedanceMPC(N=20, dt=CTRL_DT, Q=Q_MPC, R=R_MPC, F_max=F_MAX)
        mpc.precompute_mode('free', 0.20*np.eye(3))
        if cfg.get('use_kalman'):
            kalman = KalmanDisturbanceEstimator(dt=CTRL_DT)
            kalman.set_mode(mpc.A_d, mpc._mode_library['free']['B_d'])

    F_prev = np.zeros(3); prev_mode = 'free'
    braced = False; dwell = 0                       # debounced contact state
    t_log = np.zeros(N_RUN); e_log = np.zeros((N_RUN, 3)); switch_times = []

    for step in range(N_RUN):
        t = step*CTRL_DT
        data.xfrc_applied[ids['hand_body'], :3] = F_DIST if t >= T_DIST else np.zeros(3)

        p_act = data.site_xpos[ids['hand']].copy()
        Jr = get_site_jacobian(m, data, ids['hand'])
        v_act = Jr @ data.qvel
        e_pos = p_act - p0; e_vel = v_act
        t_log[step] = t; e_log[step] = e_pos

        # Debounce the rail contact (require 15 ms of persistence to flip state)
        raw = _braced(m, data, ids)
        if raw != braced:
            dwell += 1
            if dwell >= 15:
                braced = raw; dwell = 0
        else:
            dwell = 0
        mode_key = 'braced' if braced else 'free'
        if mode_key != prev_mode and step > 0:
            switch_times.append(t)
        switched = (mode_key != prev_mode); prev_mode = mode_key

        # RIGHT-arm force command with contact-set-dependent Lambda_arm
        if mpc is not None:
            M_  = get_mass_matrix(m, data)
            Jc_ = _contact_jac(m, data, ids, braced)
            Mbar_ = (get_contact_consistent_inverse(M_, Jc_) if use_cc
                     else np.linalg.inv(M_ + 1e-4*np.eye(m.nv)))
            La = get_task_inertia(Jr, Mbar_)
            mode = mpc.get_or_update_mode(mode_key, La)
            d_hat = None
            if kalman is not None:
                if switched and cfg.get('inflate_alpha', 1.0) > 1.0:
                    kalman.inflate_covariance(cfg['inflate_alpha'])
                kalman.set_mode(mpc.A_d, mode['B_d'])
                kalman.predict(F_prev); _, d_hat = kalman.update(e_pos)
            F_mpc = mpc.solve(np.concatenate([e_pos, e_vel]), La, mode_key, d_hat, use_osqp=False)
            F_arm = -F_mpc; F_prev = F_mpc
        else:
            F_arm = -(800.0*e_pos + 40.0*e_vel)

        # left-arm brace schedule
        t_in  = t % T_CYCLE
        pose  = POSE_BRACE if (t >= T_DIST+0.3 and t_in < T_BRACE) else POSE_RELEASE
        _apply(m, data, ids, F_arm, pose)
        for _ in range(2):
            mujoco.mj_step(m, data)

    return t_log, e_log, switch_times


def _apply(m, data, ids, F_arm, left_pose):
    """Assemble ctrl: leg stance PD, right-arm task force, left-arm brace PD."""
    c = np.zeros(m.nu)
    for i, (qa, da) in enumerate(ids['leg_q']):
        c[ids['leg_a'][i]] = KPL[i]*(LEG_REF[i]-data.qpos[qa]) - KDL[i]*data.qvel[da]
    Jr = get_site_jacobian(m, data, ids['hand'])
    tau_task = Jr[:, ids['rarm_dof']].T @ F_arm
    for i, (qa, da) in enumerate(ids['rarm_q']):
        null = 3.0*(RARM_REF[i]-data.qpos[qa]) - 0.5*data.qvel[da]
        c[ids['rarm_a'][i]] = tau_task[i] + null
    for i, (qa, da) in enumerate(ids['larm_q']):
        c[ids['larm_a'][i]] = 60.0*(left_pose[i]-data.qpos[qa]) - 5.0*data.qvel[da]
    data.ctrl[:] = c


def switch_peak(t_log, e_log, switch_times, window=0.25):
    peaks = [np.max(np.linalg.norm(e_log[(t_log >= ts) & (t_log <= ts+window)], axis=1))*1000
             for ts in switch_times if ((t_log >= ts) & (t_log <= ts+window)).any()]
    return float(np.mean(peaks)) if peaks else float('nan')


CONTROLLERS = {
    'D5 noKalman':      dict(use_mpc=True, use_kalman=False),
    'D6 Kalman noInfl': dict(use_mpc=True, use_kalman=True, inflate_alpha=1.0),
    'D7 Kalman+Infl':   dict(use_mpc=True, use_kalman=True, inflate_alpha=4.0),
}

if __name__ == '__main__':
    for nm, cfg in CONTROLLERS.items():
        t, e, sw = run_controller(nm, cfg)
        rms  = np.sqrt(np.mean(np.sum(e**2, axis=1)))*1000
        peak = switch_peak(t, e, sw)
        print(f'{nm:20s} RMS={rms:7.3f} mm  peak@brace/release={peak:7.3f} mm  switches={len(sw)}')
