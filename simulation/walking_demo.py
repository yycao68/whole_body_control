"""
Whole-Body Impedance MPC Demo — Video Recording
=================================================
Demonstrates D7 (Kalman-augmented Impedance MPC) on a standing biped:
  Phase 1 (0-3s):   Robot stands; arm tracks sinusoidal reference
  Phase 2 (3-5s):   In-place weight-shift stepping (contact transitions)
  Phase 3 (5-9s):   8 N pHRI disturbance applied → Kalman rejects it
  Phase 4 (9-12s):  Disturbance removed; arm returns to reference

The key result: D7 achieves ~11× lower SS error than PD under pHRI,
and the covariance-inflation protocol handles contact transitions cleanly.

Run:  cd whole_body_control && python3 simulation/walking_demo.py
Output: simulation/walking_demo.mp4
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import mujoco
import imageio
from pathlib import Path

from simulation.controllers.wbc_core import (
    get_hand_state, get_mass_matrix,
    get_contact_consistent_inverse, get_contact_jacobian,
    get_task_inertia, get_site_jacobian, get_robot_com,
    get_foot_contact_flags, get_bias_force, _get_ids,
)
from simulation.controllers.impedance_mpc import ImpedanceMPC
from simulation.controllers.kalman import KalmanDisturbanceEstimator

MODEL_PATH = Path(__file__).parent / 'models' / 'biped.xml'
OUT_PATH   = Path(__file__).parent.parent / 'simulation' / 'walking_demo.mp4'

SIM_DT    = 0.0005
CTRL_DT   = 0.001
VIDEO_FPS = 30
VIDEO_H, VIDEO_W = 480, 640

T_SETTLE   = 3.0     # initial settling [s]
T_TOTAL    = 12.0    # total demo [s]
T_DIST_ON  = 5.0
T_DIST_OFF = 9.0
F_DIST = np.array([8.0, 0.0, 0.0])

# Arm reference: sinusoidal z oscillation, amplitude 30mm, 0.4 Hz
ARM_AMP  = 0.030
ARM_FREQ = 0.4

# Weight-shift stepping (gentle in-place stepping, contact-transition demo)
STEP_AMP   = 0.08   # hip roll amplitude [rad] for weight shift
STEP_FREQ  = 0.8    # stepping frequency [Hz]

# Leg PD gains — needs to support ~225N per leg at ~0.4m lever → 90Nm
# KP * 0.1rad_error ≥ 90Nm → KP ≥ 900 Nm/rad for hip
KP_LEG = np.array([200, 1000, 1000, 400,   200, 1000, 1000, 400], dtype=float)
KD_LEG = np.array([ 20,  100,  100,  40,    20,  100,  100,  40], dtype=float)

Q_MPC = np.diag([5000., 5000., 5000., 5., 5., 5.])
R_MPC = 0.01 * np.eye(3)


def _joint_pd(model, data, jname, q_ref, kp, kd, lim):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    q  = data.qpos[model.jnt_qposadr[jid]]
    dq = data.qvel[model.jnt_dofadr[jid]]
    return float(np.clip(kp*(q_ref-q)-kd*dq, -lim, lim))


def compute_leg_ctrl(model, data, q_ref8):
    """8-DOF leg PD control."""
    names  = ['left_hip_x','left_hip_y','left_knee_y','left_ankle_y',
              'right_hip_x','right_hip_y','right_knee_y','right_ankle_y']
    limits = [120, 200, 200, 100, 120, 200, 200, 100]
    return np.array([_joint_pd(model, data, nm, q_ref8[i], KP_LEG[i], KD_LEG[i], lim)
                     for i,(nm,lim) in enumerate(zip(names,limits))])


def compute_arm_ctrl(model, data, ids, F_arm_task):
    """Arm torques: task force + null-space centering (no gravity comp for stability)."""
    J_arm  = get_site_jacobian(model, data, ids['hand_site'])
    adofs  = [ids['rshoulder_x_dof'], ids['rshoulder_y_dof'], ids['relbow_y_dof']]
    aqadrs = [ids['rshoulder_x_qadr'], ids['rshoulder_y_qadr'], ids['relbow_y_qadr']]
    J_cols = J_arm[:, adofs]
    tau_task = np.clip(J_cols.T @ F_arm_task, -np.array([80,80,60]), np.array([80,80,60]))
    tau_null = np.zeros(3)
    q0 = [0.0, 0.5, -1.0]
    for i, (dadr, qadr) in enumerate(zip(adofs, aqadrs)):
        q  = data.qpos[qadr]
        dq = data.qvel[dadr]
        tau_null[i] = np.clip(3.0*(q0[i]-q) - 0.5*dq, -20, 20)
    return tau_task + tau_null


def run_walking_demo():
    print("Loading model...")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data  = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    ids   = _get_ids(model)

    mpc = ImpedanceMPC(N=20, dt=CTRL_DT, Q=Q_MPC, R=R_MPC, F_max=80.0)
    kal = KalmanDisturbanceEstimator(dt=CTRL_DT)

    # ── Settle 3s in stance ───────────────────────────────────────────────
    print(f"Settling {T_SETTLE}s...")
    stance = np.array([0., -0.05, 0.10, -0.05, 0., -0.05, 0.10, -0.05])
    for step in range(int(T_SETTLE / SIM_DT)):
        com, _ = get_robot_com(model, data)
        q_ref  = stance.copy()
        q_ref[0] = -2.0 * com[1]
        q_ref[4] = -2.0 * com[1]
        tau_l = compute_leg_ctrl(model, data, q_ref)
        tau_a = compute_arm_ctrl(model, data, ids, np.zeros(3))
        data.ctrl[:8]  = tau_l
        data.ctrl[8:11] = tau_a
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    p0, _ = get_hand_state(model, data)
    p0 = p0.copy()
    print(f"  torso_z={data.xpos[1,2]:.3f}  hand={p0.round(3)}")

    # Init MPC
    foot_ids  = [ids['left_foot_site'], ids['right_foot_site']]
    init_mode = frozenset(['left','right'])
    M   = get_mass_matrix(model, data)
    Jc  = get_contact_jacobian(model, data, foot_ids, [True,True])
    Mb  = get_contact_consistent_inverse(M, Jc)
    Ja  = get_site_jacobian(model, data, ids['hand_site'])
    La  = get_task_inertia(Ja, Mb)
    mpc.precompute_mode(init_mode, La)
    kal.set_mode(mpc.A_d, mpc._mode_library[init_mode]['B_d'])
    prev_mode = init_mode
    F_prev    = np.zeros(3)
    F_arm     = np.zeros(3)

    # ── Video renderer ────────────────────────────────────────────────────
    renderer  = mujoco.Renderer(model, height=VIDEO_H, width=VIDEO_W)
    cam       = mujoco.MjvCamera()
    cam.lookat[:] = [0., 0., 0.75]
    cam.distance  = 3.2
    cam.azimuth   = 80.0
    cam.elevation = -10.0

    frames     = []
    vid_period = max(1, int(1.0 / (VIDEO_FPS * SIM_DT)))
    frame_ctr  = 0
    mpc_ctr    = 0
    mpc_period = max(1, int(CTRL_DT / SIM_DT))
    hand_body  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_hand')
    n_total    = int(T_TOTAL / SIM_DT)

    print(f"Recording {T_TOTAL}s at {VIDEO_FPS} fps...")

    for step in range(n_total):
        t  = step * SIM_DT
        tz = data.xpos[1, 2]

        # ── pHRI disturbance ─────────────────────────────────────────
        data.xfrc_applied[hand_body, :3] = (
            F_DIST if T_DIST_ON <= t < T_DIST_OFF else np.zeros(3))

        # ── Leg reference (stance + gentle weight-shift) ──────────────
        com, _ = get_robot_com(model, data)
        q_leg  = stance.copy()
        q_leg[0] = -2.0 * com[1]
        q_leg[4] = -2.0 * com[1]
        # Weight-shift phase (t = 3..12s): rock hips sideways
        if t >= 3.0:
            shift = STEP_AMP * np.sin(2*np.pi*STEP_FREQ*(t-3.0))
            q_leg[0] += shift    # left hip x
            q_leg[4] -= shift    # right hip x (opposite phase)
        tau_l = compute_leg_ctrl(model, data, q_leg)

        # ── Arm MPC (CTRL_DT updates) ─────────────────────────────────
        mpc_ctr += 1
        if mpc_ctr >= mpc_period:
            mpc_ctr = 0

            contact_mask = get_foot_contact_flags(model, data, foot_ids)
            cur_mode = frozenset(
                (['left']  if contact_mask[0] else []) +
                (['right'] if contact_mask[1] else [])
            )
            active = cur_mode if len(cur_mode) > 0 else prev_mode

            if cur_mode != prev_mode and len(cur_mode) > 0:
                M  = get_mass_matrix(model, data)
                Jc = get_contact_jacobian(model, data, foot_ids, contact_mask)
                Mb = get_contact_consistent_inverse(M, Jc)
                Ja = get_site_jacobian(model, data, ids['hand_site'])
                La = get_task_inertia(Ja, Mb)
                mpc.precompute_mode(cur_mode, La)
                kal.set_mode(mpc.A_d, mpc._mode_library[cur_mode]['B_d'])
                kal.inflate_covariance(4.0)
                prev_mode = cur_mode

            # Arm target in torso-local frame → world frame
            R_t  = data.xmat[1].reshape(3,3)
            off  = np.array([0.05, -0.18,
                             -0.35 + ARM_AMP*np.sin(2*np.pi*ARM_FREQ*t)])
            p_des = data.xpos[1] + R_t @ off
            vz_l  = np.array([0.,0., ARM_AMP*2*np.pi*ARM_FREQ*np.cos(2*np.pi*ARM_FREQ*t)])
            v_des = R_t @ vz_l

            p_act, v_act = get_hand_state(model, data)
            e_pos = p_act - p_des
            e_vel = v_act - v_des

            M  = get_mass_matrix(model, data)
            Jc = get_contact_jacobian(model, data, foot_ids, contact_mask)
            Mb = get_contact_consistent_inverse(M, Jc)
            Ja = get_site_jacobian(model, data, ids['hand_site'])
            La = get_task_inertia(Ja, Mb)

            kal.predict(F_prev)
            _, d_hat = kal.update(e_pos)
            F_mpc = mpc.solve(np.concatenate([e_pos,e_vel]), La,
                              active, d_hat, use_osqp=False)
            F_arm  = -F_mpc
            F_prev = F_mpc

        tau_a = compute_arm_ctrl(model, data, ids, F_arm)
        data.ctrl[:8]  = tau_l
        data.ctrl[8:11] = tau_a
        mujoco.mj_step(model, data)

        # ── Capture frame ─────────────────────────────────────────────
        frame_ctr += 1
        if frame_ctr >= vid_period:
            frame_ctr = 0
            cam.lookat[0] = data.xpos[1,0]
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

        if step % 4000 == 0:
            p_a, _ = get_hand_state(model, data)
            R_t = data.xmat[1].reshape(3,3)
            pd  = data.xpos[1] + R_t @ np.array([0.05,-0.18,-0.35+ARM_AMP*np.sin(2*np.pi*ARM_FREQ*t)])
            e_mm = np.linalg.norm(p_a - pd)*1000
            print(f"  t={t:5.1f}s  z={tz:.3f}  |e|={e_mm:.1f}mm  "
                  f"d̂={kal.d_hat.round(2)}  "
                  f"{'[DIST]' if T_DIST_ON<=t<T_DIST_OFF else ''}")

    renderer.close()
    print(f"  Final: z={data.xpos[1,2]:.3f}  "
          f"{'STANDING' if data.xpos[1,2]>0.4 else 'FELL'}")

    print(f"\nSaving {len(frames)} frames → {OUT_PATH}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(OUT_PATH), fps=VIDEO_FPS,
                             codec='libx264', quality=8,
                             macro_block_size=1) as writer:
        for fr in frames:
            writer.append_data(fr)
    kb = OUT_PATH.stat().st_size / 1024
    print(f"Saved: {OUT_PATH}  ({len(frames)/VIDEO_FPS:.1f}s  {kb:.0f} KB)")
    return frames


if __name__ == '__main__':
    run_walking_demo()
