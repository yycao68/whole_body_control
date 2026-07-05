"""
Record comparison videos for Scenarios A, B, and C.

Each video shows D1 (SK05 PD baseline) on the left vs D7 (full proposed:
WBC + Impedance MPC + Kalman + covariance inflation) on the right, rendered
side-by-side at 30 fps so the viewer can directly compare tracking quality.

A coloured label strip (red = D1 baseline, green = D7 proposed) is burned
into the top of each panel so the panels are self-identifying.

Output
------
  simulation/results/scenario_a_video.mp4   — Fixed stance, 8 N step pHRI
  simulation/results/scenario_b_video.mp4   — Stance + 1 Hz shocks, 8 N pHRI
  simulation/results/scenario_c_video.mp4   — R1-mass platform, 8 N step pHRI

Usage
-----
  cd whole_body_control
  python3 simulation/record_videos.py [--scenario a|b|c|all]
"""

import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import mujoco
import imageio
from pathlib import Path

from simulation.controllers.wbc_core import (
    get_hand_state, get_mass_matrix, get_bias_force,
    get_contact_consistent_inverse, get_contact_jacobian,
    get_task_inertia, get_site_jacobian, get_robot_com,
    _get_ids,
)
from simulation.controllers.impedance_mpc import ImpedanceMPC
from simulation.controllers.kalman import KalmanDisturbanceEstimator

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_BIPED  = Path(__file__).parent / 'models' / 'biped.xml'
MODEL_R1     = Path(__file__).parent / 'models' / 'r1_mass.xml'
MODEL_G1     = Path(__file__).parent / 'models' / 'g1_wbc.xml'
OUT_DIR      = Path(__file__).parent.parent / 'simulation' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Video settings ─────────────────────────────────────────────────────────
VID_H   = 480
VID_W   = 640
VID_FPS = 30
SIM_DT  = 0.0005
CTRL_DT = 0.001
VID_PERIOD = max(1, int(round(1.0 / (VID_FPS * SIM_DT))))   # physics steps per frame

# ── Controller parameters (identical to benchmark scenarios) ───────────────
KP_DIST = 800.0;  KD_PD = 40.0
Q_MPC   = np.diag([60000., 60000., 60000., 60., 60., 60.])
R_MPC   = 0.01 * np.eye(3)
F_MAX   = 80.0
KP_LEG  = np.array([200, 1000, 1000, 400,   200, 1000, 1000, 400], dtype=float)
KD_LEG  = np.array([ 20,  100,  100,  40,    20,  100,  100,  40], dtype=float)
STANCE_Q = np.array([0., -0.05, 0.10, -0.05,   0., -0.05, 0.10, -0.05])

# Label colours burned into the top strip of each panel
LABEL_H     = 16          # pixels
LABEL_D1    = (220,  60,  60)    # red  — D1 baseline
LABEL_D7    = ( 40, 180,  60)    # green — D7 proposed


# ── Helpers ────────────────────────────────────────────────────────────────

def _make(model_path):
    m = mujoco.MjModel.from_xml_path(str(model_path))
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


def _settle(model, data, ids, ag, n_settle=4000):
    for _ in range(n_settle):
        com, _ = get_robot_com(model, data)
        q_ref  = STANCE_Q.copy()
        q_ref[0] = -2.0 * com[1]
        q_ref[4] = -2.0 * com[1]
        data.ctrl[:8]   = _leg_pd(model, data, q_ref)
        data.ctrl[8:11] = _arm_null(model, data, ids) + ag
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)


def _make_renderer(model):
    renderer = mujoco.Renderer(model, height=VID_H, width=VID_W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0., 0., 0.75]
    cam.distance  = 3.0
    cam.azimuth   = 75.0
    cam.elevation = -12.0
    return renderer, cam


def _render_frame(renderer, cam, data, label_colour):
    """Render one frame and burn a coloured label strip into the top."""
    cam.lookat[0] = data.xpos[1, 0]   # track robot x position
    renderer.update_scene(data, camera=cam)
    frame = renderer.render().copy()   # (H, W, 3) uint8
    frame[:LABEL_H, :] = label_colour
    return frame


def _save_video(frames, path):
    print(f"  Saving {len(frames)} frames → {path}")
    with imageio.get_writer(str(path), fps=VID_FPS,
                             codec='libx264', quality=8,
                             macro_block_size=1) as writer:
        for fr in frames:
            writer.append_data(fr)


def _init_d7(model, data, ids, foot_ids):
    """Initialise D7 MPC+Kalman at the settled pose."""
    mpc    = ImpedanceMPC(N=20, dt=CTRL_DT, Q=Q_MPC, R=R_MPC, F_max=F_MAX)
    La     = 0.20 * np.eye(3)
    mpc.precompute_mode('ds', La)
    kalman = KalmanDisturbanceEstimator(dt=CTRL_DT)
    kalman.set_mode(mpc.A_d, mpc._mode_library['ds']['B_d'])
    return mpc, kalman, La


# ── Scenario A / C  (step disturbance) ────────────────────────────────────

def _run_step_disturbance(model_path, n_settle, n_run, t_dist, f_dist, out_path):
    """
    Record D1 vs D7 for a step-disturbance scenario.
    Left panel = D1 (red strip), right panel = D7 (green strip).
    """
    print(f"\n--- Recording {out_path.name} ---")

    frames_d1, frames_d7 = [], []
    foot_ids_key = ['left_foot_site', 'right_foot_site']

    for label, use_mpc in [('D1 PD', False), ('D7 Full', True)]:
        print(f"  Simulating {label}...")
        model, data = _make(model_path)
        ids  = _get_ids(model)
        ag   = _arm_grav(model)
        _settle(model, data, ids, ag, n_settle)

        p0, _     = get_hand_state(model, data)
        foot_ids  = [ids[k] for k in foot_ids_key]
        hand_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_hand')

        mpc = kalman = La = None
        if use_mpc:
            mpc, kalman, La = _init_d7(model, data, ids, foot_ids)

        renderer, cam = _make_renderer(model)
        F_prev = np.zeros(3);  F_arm = np.zeros(3)
        frame_ctr = 0;  ctrl_ctr = 0
        ctrl_period = max(1, int(round(CTRL_DT / SIM_DT)))
        colour = LABEL_D7 if use_mpc else LABEL_D1

        for step in range(n_run):
            t = step * SIM_DT
            data.xfrc_applied[hand_body, :3] = f_dist if t >= t_dist else np.zeros(3)

            ctrl_ctr += 1
            if ctrl_ctr >= ctrl_period:
                ctrl_ctr = 0
                p_act, v_act = get_hand_state(model, data)
                e_pos = p_act - p0;  e_vel = v_act

                if mpc is not None:
                    kalman.predict(F_prev)
                    _, d_hat = kalman.update(e_pos)
                    F_mpc  = mpc.solve(np.concatenate([e_pos, e_vel]),
                                       La, 'ds', d_hat, use_osqp=False)
                    F_arm  = -F_mpc;  F_prev = F_mpc
                else:
                    F_arm = -(KP_DIST * e_pos + KD_PD * e_vel)

            com, _ = get_robot_com(model, data)
            q_ref  = STANCE_Q.copy()
            q_ref[0] = -2.0 * com[1];  q_ref[4] = -2.0 * com[1]
            data.ctrl[:8]   = _leg_pd(model, data, q_ref)
            data.ctrl[8:11] = _arm_from_force(model, data, ids, F_arm) + \
                              _arm_null(model, data, ids) + ag
            mujoco.mj_step(model, data)

            frame_ctr += 1
            if frame_ctr >= VID_PERIOD:
                frame_ctr = 0
                fr = _render_frame(renderer, cam, data, colour)
                if use_mpc:
                    frames_d7.append(fr)
                else:
                    frames_d1.append(fr)

        renderer.close()

    n = min(len(frames_d1), len(frames_d7))
    combined = [np.hstack([frames_d1[i], frames_d7[i]]) for i in range(n)]
    _save_video(combined, out_path)


# ── Scenario B  (periodic shocks) ─────────────────────────────────────────

def _run_shock_scenario(out_path):
    """
    Record D1 vs D7 for Scenario B (sustained pHRI + 1 Hz shocks).
    """
    print(f"\n--- Recording {out_path.name} ---")
    T_TOTAL    = 10.0
    F_BASE     = np.array([8.0, 0.0, 0.0])
    F_SPIKE    = np.array([6.0, 0.0, 0.0])
    T_SPIKE    = 0.10
    T_SWITCH   = 1.00
    n_run      = int(T_TOTAL / SIM_DT)
    foot_ids_key = ['left_foot_site', 'right_foot_site']

    frames_d1, frames_d7 = [], []

    for label, use_mpc in [('D1 PD', False), ('D7 Full', True)]:
        print(f"  Simulating {label}...")
        model, data = _make(MODEL_BIPED)
        ids       = _get_ids(model)
        ag        = _arm_grav(model)
        _settle(model, data, ids, ag)

        p0, _     = get_hand_state(model, data)
        foot_ids  = [ids[k] for k in foot_ids_key]
        hand_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_hand')

        mpc = kalman = La = None
        mode_key = 0
        if use_mpc:
            mpc, kalman, La = _init_d7(model, data, ids, foot_ids)

        renderer, cam = _make_renderer(model)
        F_prev = np.zeros(3);  F_arm = np.zeros(3)
        frame_ctr = 0;  ctrl_ctr = 0
        ctrl_period = max(1, int(round(CTRL_DT / SIM_DT)))
        colour = LABEL_D7 if use_mpc else LABEL_D1
        prev_spike = False

        for step in range(n_run):
            t = step * SIM_DT
            t_mod  = t % T_SWITCH
            spiking = (t_mod < T_SPIKE)
            f_now  = F_BASE + (F_SPIKE if spiking else np.zeros(3))
            data.xfrc_applied[hand_body, :3] = f_now

            # signal mode key change at spike onset so Kalman can inflate
            if use_mpc and spiking and not prev_spike:
                mode_key += 1
                if kalman is not None:
                    kalman.inflate_covariance(4.0)
            prev_spike = spiking

            ctrl_ctr += 1
            if ctrl_ctr >= ctrl_period:
                ctrl_ctr = 0
                p_act, v_act = get_hand_state(model, data)
                e_pos = p_act - p0;  e_vel = v_act

                if mpc is not None:
                    kalman.predict(F_prev)
                    _, d_hat = kalman.update(e_pos)
                    F_mpc  = mpc.solve(np.concatenate([e_pos, e_vel]),
                                       La, 'ds', d_hat, use_osqp=False)
                    F_arm  = -F_mpc;  F_prev = F_mpc
                else:
                    F_arm = -(KP_DIST * e_pos + KD_PD * e_vel)

            com, _ = get_robot_com(model, data)
            q_ref  = STANCE_Q.copy()
            q_ref[0] = -2.0 * com[1];  q_ref[4] = -2.0 * com[1]
            data.ctrl[:8]   = _leg_pd(model, data, q_ref)
            data.ctrl[8:11] = _arm_from_force(model, data, ids, F_arm) + \
                              _arm_null(model, data, ids) + ag
            mujoco.mj_step(model, data)

            frame_ctr += 1
            if frame_ctr >= VID_PERIOD:
                frame_ctr = 0
                fr = _render_frame(renderer, cam, data, colour)
                if use_mpc:
                    frames_d7.append(fr)
                else:
                    frames_d1.append(fr)

        renderer.close()

    n = min(len(frames_d1), len(frames_d7))
    combined = [np.hstack([frames_d1[i], frames_d7[i]]) for i in range(n)]
    _save_video(combined, out_path)


# ── Main ───────────────────────────────────────────────────────────────────

def _run_g1_video(out_path):
    """D1 vs D7 on the official Unitree G1 model using position-as-torque arm ctrl."""
    print(f"\n--- Recording {out_path.name} (Unitree G1, 33.3 kg) ---")

    from simulation.scenarios.scenario_c_g1 import (
        G1Ids, G1_CTRL_STAND, _settle as g1_settle,
        _get_hand_state as g1_hand, _get_robot_com as g1_com,
        _arm_force_to_ctrl, N_SETTLE as G1_SETTLE,
    )
    from simulation.controllers.impedance_mpc import ImpedanceMPC
    from simulation.controllers.kalman import KalmanDisturbanceEstimator

    frames_d1, frames_d7 = [], []
    F8 = np.array([8.0, 0.0, 0.0])

    for label, use_mpc in [('D1 PD', False), ('D7 Full', True)]:
        print(f"  Simulating G1 {label}...")
        model = mujoco.MjModel.from_xml_path(str(MODEL_G1))
        data  = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        ids = G1Ids.get(model)
        g1_settle(model, data)

        p0, _ = g1_hand(model, data, ids)
        hand_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_wrist_yaw_link')

        mpc = kalman = La = None
        if use_mpc:
            La  = 0.20 * np.eye(3)
            mpc = ImpedanceMPC(N=20, dt=CTRL_DT, Q=Q_MPC, R=R_MPC, F_max=F_MAX)
            mpc.precompute_mode('ds', La)
            kalman = KalmanDisturbanceEstimator(dt=CTRL_DT)
            kalman.set_mode(mpc.A_d, mpc._mode_library['ds']['B_d'])

        renderer, cam = _make_renderer(model)
        cam.lookat[:] = [0., 0., 0.90]
        cam.distance   = 3.5
        F_prev = np.zeros(3);  F_arm = np.zeros(3)
        frame_ctr = 0;  ctrl_ctr = 0
        ctrl_period = max(1, int(round(CTRL_DT / SIM_DT)))
        colour = LABEL_D7 if use_mpc else LABEL_D1
        n_run = int(5.0 / SIM_DT)

        for step in range(n_run):
            t = step * SIM_DT
            data.xfrc_applied[hand_body, :3] = F8 if t >= 0.5 else np.zeros(3)

            ctrl_ctr += 1
            if ctrl_ctr >= ctrl_period:
                ctrl_ctr = 0
                p_act, v_act = g1_hand(model, data, ids)
                e_pos = p_act - p0;  e_vel = v_act
                if mpc is not None:
                    kalman.predict(F_prev)
                    _, d_hat = kalman.update(e_pos)
                    F_mpc = mpc.solve(np.concatenate([e_pos, e_vel]),
                                      La, 'ds', d_hat, use_osqp=False)
                    F_arm = -F_mpc;  F_prev = F_mpc
                else:
                    F_arm = -(KP_DIST * e_pos + KD_PD * e_vel)

            com, _ = g1_com(model, data)
            ctrl   = G1_CTRL_STAND.copy()
            ctrl[1] -= 1.5 * com[1];  ctrl[7] -= 1.5 * com[1]
            ctrl[22:29] = _arm_force_to_ctrl(model, data, ids, F_arm)
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)

            frame_ctr += 1
            if frame_ctr >= VID_PERIOD:
                frame_ctr = 0
                fr = _render_frame(renderer, cam, data, colour)
                if use_mpc: frames_d7.append(fr)
                else:        frames_d1.append(fr)

        renderer.close()

    n = min(len(frames_d1), len(frames_d7))
    combined = [np.hstack([frames_d1[i], frames_d7[i]]) for i in range(n)]
    _save_video(combined, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Record scenario comparison videos.')
    parser.add_argument('--scenario', choices=['a', 'b', 'c', 'g1', 'all'], default='all')
    args = parser.parse_args()

    F8 = np.array([8.0, 0.0, 0.0])

    if args.scenario in ('a', 'all'):
        _run_step_disturbance(
            model_path = MODEL_BIPED,
            n_settle   = 4000,
            n_run      = int(5.0 / SIM_DT),
            t_dist     = 0.5,
            f_dist     = F8,
            out_path   = OUT_DIR / 'scenario_a_video.mp4',
        )

    if args.scenario in ('b', 'all'):
        _run_shock_scenario(
            out_path = OUT_DIR / 'scenario_b_video.mp4',
        )

    if args.scenario in ('c', 'all'):
        _run_step_disturbance(
            model_path = MODEL_R1,
            n_settle   = 4000,
            n_run      = int(5.0 / SIM_DT),
            t_dist     = 0.5,
            f_dist     = F8,
            out_path   = OUT_DIR / 'scenario_c_video.mp4',
        )

    if args.scenario in ('g1', 'all'):
        _run_g1_video(out_path = OUT_DIR / 'scenario_g1_video.mp4')

    print("\nAll videos saved to", OUT_DIR)
