#!/usr/bin/env python3
"""Root-assisted Unitree G1 walking visualization for the v3 paper.

This script intentionally mirrors the visual walking scaffold used by
``whole_body_control/g1_ab_simulation/run_g1_ab.py``.  The floating base is
guided by a normalized body interaction MPC along a 10 s walking reference that
ramps to 1.2 m/s at 1 s, cruises until 9 s, and stops at 10 s while MuJoCo
renders the G1 model and the commanded joint gait lifts one foot at a time.  A
second normalized task interaction MPC regulates the right-hand trajectory
through a damped Jacobian update.

This is a visualization/architecture demo, not a torque-level validation of
dynamic humanoid walking.  The physically stronger benchmark remains the
planned torque-actuated G1 inverse-dynamics test.
"""

from __future__ import annotations

import json
import math
import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from normalized_mpc import NormalizedMPC, RandomWalkDisturbanceObserver


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "g1_wbc.xml"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

SIM_DT = 0.001
COMMAND_DT = 0.002
DURATION = 10.0
CRUISE_SPEED = 1.2
RAMP_TIME = 1.0
DISTANCE = CRUISE_SPEED * (DURATION - RAMP_TIME)

ACTUATED_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

RIGHT_ARM_ACT = slice(22, 29)
RIGHT_ARM_JOINT_NAMES = ACTUATED_JOINT_NAMES[22:29]

G1_STAND_CTRL = np.array([
    0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0.,
    0., 0., 0.,
    0.2,  0.2, 0., 1.28, 0., 0., 0.,
    0.2, -0.2, 0., 1.28, 0., 0., 0.,
], dtype=float)


@dataclass
class LocalTrajectory:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    heading: float
    phase_hint: str = "walk"


class G1CommandLayer:
    """500 Hz position command layer with explicit one-foot swing phases."""

    def __init__(self):
        self.ctrl = G1_STAND_CTRL.copy()

    def step(self, t: float, traj: LocalTrajectory) -> np.ndarray:
        ctrl = G1_STAND_CTRL.copy()
        speed = float(np.linalg.norm(traj.velocity))
        gait_frequency = 0.78
        cycle = (gait_frequency * t) % 1.0
        left_swing = cycle < 0.5
        phase = cycle * 2.0 if left_swing else (cycle - 0.5) * 2.0
        phase = float(np.clip(phase, 0.0, 1.0))
        swing_lift = math.sin(math.pi * phase)
        swing_travel = np.clip(0.08 + 0.92 * phase, 0.0, 1.0)
        lift = np.clip(0.42 + 0.95 * speed, 0.42, 0.78)
        stride = np.clip(0.30 + 0.75 * speed, 0.30, 0.54)

        def stance_leg(progress: float, side_sign: float) -> np.ndarray:
            hip_pitch = -0.28 * stride + 1.10 * stride * progress
            knee = 0.10 + 0.04 * math.sin(math.pi * progress)
            return np.array([
                hip_pitch,
                side_sign * 0.035,
                0.0,
                knee,
                -0.10 - 0.22 * hip_pitch,
                -side_sign * 0.035,
            ])

        def swing_leg(progress: float, side_sign: float) -> np.ndarray:
            knee_lift = 0.38 * lift * swing_lift
            hip_pitch = 0.62 * stride - 1.28 * stride * swing_travel - knee_lift
            return np.array([
                hip_pitch,
                side_sign * (0.04 + 0.08 * swing_lift),
                side_sign * 0.025 * math.sin(2.0 * math.pi * progress),
                0.12 + lift * swing_lift,
                -0.12 - 0.74 * lift * swing_lift - 0.20 * hip_pitch,
                -side_sign * (0.04 + 0.07 * swing_lift),
            ])

        left = slice(0, 6)
        right = slice(6, 12)
        if left_swing:
            ctrl[left] = swing_leg(phase, 1.0)
            ctrl[right] = stance_leg(phase, -1.0)
        else:
            ctrl[left] = stance_leg(phase, 1.0)
            ctrl[right] = swing_leg(phase, -1.0)

        arm_phase = math.sin(2.0 * math.pi * cycle)
        ctrl[12:15] = np.array([0.0, 0.0, 0.03 * arm_phase])
        ctrl[15:22] = np.array([0.12 + 0.18 * arm_phase, 0.20, 0.0, 1.18, 0.0, 0.0, 0.0])
        ctrl[22:29] = np.array([0.12 - 0.18 * arm_phase, -0.20, 0.0, 1.18, 0.0, 0.0, 0.0])
        self.ctrl = ctrl
        return ctrl


def body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def site_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def joint_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def site_jac(model, data, sid):
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, sid)
    return jp


def hand_state(model, data, sid):
    jp = site_jac(model, data, sid)
    return data.site_xpos[sid].copy(), jp @ data.qvel


def arm_q_and_jac(model, data, hand_sid):
    dofs = [model.jnt_dofadr[joint_id(model, n)] for n in RIGHT_ARM_JOINT_NAMES]
    qadrs = [model.jnt_qposadr[joint_id(model, n)] for n in RIGHT_ARM_JOINT_NAMES]
    q = np.array([data.qpos[a] for a in qadrs])
    j = site_jac(model, data, hand_sid)[:, dofs]
    return q, j


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])


def smoothstep(tau: float):
    tau = float(np.clip(tau, 0.0, 1.0))
    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    sd = 30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4
    sdd = 60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3
    return s, sd, sdd


def trapezoid_profile(
    t: float,
    duration: float = DURATION,
    distance: float = DISTANCE,
    ramp_time: float = RAMP_TIME,
):
    """Position, velocity, and acceleration for a symmetric velocity ramp.

    With the default 10 s duration and 1 s ramps, the total distance is
    ``v_cruise * 9``.  Thus ``DISTANCE = 10.8`` gives the requested
    ``1.2 m/s`` cruise speed from 1 s to 9 s.
    """
    duration = float(max(duration, 1e-9))
    ramp = float(np.clip(ramp_time, 0.0, 0.5 * duration))
    t = float(np.clip(t, 0.0, duration))
    effective_time = max(duration - ramp, 1e-9)
    cruise_speed = float(distance) / effective_time

    if ramp <= 1e-9:
        return cruise_speed * t, cruise_speed, 0.0

    accel = cruise_speed / ramp
    if t < ramp:
        x = 0.5 * accel * t**2
        xd = accel * t
        xdd = accel
    elif t <= duration - ramp:
        x = 0.5 * cruise_speed * ramp + cruise_speed * (t - ramp)
        xd = cruise_speed
        xdd = 0.0
    else:
        tau = duration - t
        x = float(distance) - 0.5 * accel * tau**2
        xd = accel * tau
        xdd = -accel
    return x, xd, xdd


def trajectory(t: float, distance: float = DISTANCE, duration: float = DURATION) -> LocalTrajectory:
    x, xd, xdd = trapezoid_profile(t, duration=duration, distance=distance)
    return LocalTrajectory(
        position=np.array([x, 0.0]),
        velocity=np.array([xd, 0.0]),
        acceleration=np.array([xdd, 0.0]),
        heading=0.0,
    )


def apply_root_assist(data: mujoco.MjData, traj: LocalTrajectory, t: float, height: float = 0.82):
    bob = 0.008 * max(0.0, math.sin(2.0 * math.pi * 2.2 * t))
    data.qpos[0] = traj.position[0]
    data.qpos[1] = traj.position[1]
    data.qpos[2] = height + bob
    data.qpos[3:7] = yaw_quat(traj.heading + math.pi)
    data.qvel[0] = traj.velocity[0]
    data.qvel[1] = traj.velocity[1]
    data.qvel[2] = 0.0
    data.qvel[3:6] = 0.0


def apply_commanded_pose(model: mujoco.MjModel, data: mujoco.MjData, ctrl: np.ndarray):
    for value, joint_name in zip(ctrl, ACTUATED_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[jid]] = value
        data.qvel[model.jnt_dofadr[jid]] = 0.0


def support_phase(t: float) -> str:
    cycle = (0.78 * t) % 1.0
    return "right_stance" if cycle < 0.5 else "left_stance"


def support_site_name(support: str) -> str:
    return "right_foot" if support == "right_stance" else "left_foot"


def pin_support_foot(model: mujoco.MjModel, data: mujoco.MjData, support: str, plant_xy: np.ndarray,
                     alpha: float = 0.12):
    foot_id = site_id(model, support_site_name(support))
    foot_xy = data.site_xpos[foot_id, :2].copy()
    data.qpos[:2] += alpha * (plant_xy - foot_xy)
    data.qvel[:2] = 0.0
    mujoco.mj_forward(model, data)


def robot_com(model, data):
    total = float(np.sum(model.body_mass[1:]))
    c = np.zeros(3)
    for bid in range(1, model.nbody):
        c += model.body_mass[bid] * data.xipos[bid]
    return c / total


def roll_pitch_yaw_from_body(data, bid):
    R = data.xmat[bid].reshape(3, 3)
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def _push_window(summary):
    if not summary.get("push_enabled", False):
        return None
    start = float(summary["push_start_s"])
    return start, start + float(summary["push_duration_s"])


def save_plot(log, summary, out_png):
    t = log["t"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(t, log["com"][:, 0], label="CoM x")
    axes[0].plot(t, log["com_ref"][:, 0], "--", label="reference")
    axes[0].set_ylabel("x [m]")
    axes[0].legend(loc="best")

    axes[1].plot(t, log["foot_z"][:, 0], label="left foot")
    axes[1].plot(t, log["foot_z"][:, 1], label="right foot")
    axes[1].set_ylabel("foot z [m]")
    axes[1].legend(loc="best")

    axes[2].plot(t, log["rpy"][:, 0], label="roll")
    axes[2].plot(t, log["rpy"][:, 1], label="pitch")
    axes[2].set_ylabel("rad")
    axes[2].legend(loc="best")

    axes[3].step(t, log["contact"][:, 0], where="post", label="left stance")
    axes[3].step(t, log["contact"][:, 1] + 1.2, where="post", label="right stance")
    axes[3].set_yticks([0, 1, 1.2, 2.2])
    axes[3].set_ylabel("support")
    axes[3].set_xlabel("time [s]")
    axes[3].legend(loc="best")

    window = _push_window(summary)
    if window is not None:
        for ax in axes:
            ax.axvspan(window[0], window[1], color="tab:red", alpha=0.15, label="_push")
    fig.suptitle(
        f"G1 root-assisted walking demo: distance={summary['distance_m']:.2f} m, "
        f"one-foot swing={summary['one_foot_swing_visible']}"
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def run_demo(
    duration: float = DURATION,
    distance: float = DISTANCE,
    push: bool = False,
    push_start: float = 4.5,
    push_duration: float = 0.18,
    push_accel: float = 1.6,
):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    command = G1CommandLayer()
    torso = body_id(model, "torso_link")
    left_sid = site_id(model, "left_foot")
    right_sid = site_id(model, "right_foot")
    hand_sid = site_id(model, "right_hand_site")

    body_mpc = NormalizedMPC(
        dim=2,
        dt=SIM_DT,
        horizon=40,
        q_pos=85.0,
        q_vel=18.0,
        qf_pos=120.0,
        qf_vel=25.0,
        r=0.04,
        u_max=np.array([3.0, 1.2]),
    )
    task_mpc = NormalizedMPC(
        dim=3,
        dt=COMMAND_DT,
        horizon=20,
        q_pos=160.0,
        q_vel=12.0,
        qf_pos=240.0,
        qf_vel=18.0,
        r=0.10,
        u_max=np.array([8.0, 8.0, 8.0]),
    )
    body_observer = RandomWalkDisturbanceObserver(
        dim=2,
        dt=SIM_DT,
        q_d=0.06,
        r_y=8e-5,
    )

    steps = int(round(duration / SIM_DT))
    command_period = max(1, int(round(COMMAND_DT / SIM_DT)))
    log = {
        "t": np.zeros(steps),
        "com": np.zeros((steps, 3)),
        "com_ref": np.zeros((steps, 3)),
        "u_body": np.zeros((steps, 3)),
        "d_body": np.zeros((steps, 3)),
        "kalman_innovation": np.zeros((steps, 3)),
        "push_detected": np.zeros(steps, dtype=int),
        "rpy": np.zeros((steps, 3)),
        "hand_err": np.zeros((steps, 3)),
        "contact": np.zeros((steps, 2), dtype=int),
        "raw_contact": np.zeros((steps, 2), dtype=int),
        "foot_z": np.zeros((steps, 2)),
        "force": np.zeros((steps, 3)),
        "push_accel": np.zeros((steps, 3)),
        "u_task": np.zeros((steps, 3)),
        "qpos": np.zeros((steps, model.nq)),
    }

    locked_support = None
    plant_xy = None
    ctrl = G1_STAND_CTRL.copy()
    c0 = robot_com(model, data)
    root_p = data.qpos[:2].copy()
    root_v = np.zeros(2)
    apply_root_assist(data, trajectory(0.0, distance=distance, duration=duration), 0.0)
    apply_commanded_pose(model, data, ctrl)
    mujoco.mj_forward(model, data)
    h0, _ = hand_state(model, data, hand_sid)
    u_body = np.zeros(2)
    u_task = np.zeros(3)
    hand_err = np.zeros(3)
    d_body_hat = np.zeros(2)
    innovation = np.zeros(2)
    detection_threshold = 0.20

    for k in range(steps):
        t = k * SIM_DT
        ref = trajectory(t, distance=distance, duration=duration)
        x_body = np.concatenate([root_p - ref.position, root_v - ref.velocity])
        u_body = body_mpc.solve(x_body, d_hat=d_body_hat)
        push_xy = np.zeros(2)
        if push and push_start <= t < push_start + push_duration:
            push_xy[1] = push_accel
        root_acc = ref.acceleration + u_body + push_xy
        root_p = root_p + root_v * SIM_DT + 0.5 * root_acc * SIM_DT**2
        root_v = root_v + root_acc * SIM_DT
        ref_next = trajectory(min(t + SIM_DT, duration), distance=distance, duration=duration)
        y_body = root_p - ref_next.position
        d_body_hat, innovation = body_observer.step(y_body, u_body)
        traj = LocalTrajectory(root_p.copy(), root_v.copy(), root_acc.copy(), heading=0.0)
        if k % command_period == 0:
            ctrl = command.step(t, traj)
            apply_root_assist(data, traj, t)
            apply_commanded_pose(model, data, ctrl)
            mujoco.mj_forward(model, data)

            hand, hand_vel = hand_state(model, data, hand_sid)
            hand_ref = h0 + np.array([
                ref.position[0],
                0.025 * math.sin(2.0 * math.pi * 0.35 * t),
                0.0,
            ])
            hand_v_ref = np.array([
                ref.velocity[0],
                0.025 * 2.0 * math.pi * 0.35 * math.cos(2.0 * math.pi * 0.35 * t),
                0.0,
            ])
            hand_err = hand - hand_ref
            hand_ed = hand_vel - hand_v_ref
            u_task = task_mpc.solve(np.concatenate([hand_err, hand_ed]))

            q_arm, j_arm = arm_q_and_jac(model, data, hand_sid)
            desired_dx = np.clip(0.004 * u_task, -0.035, 0.035)
            damping = 2e-3
            dq = j_arm.T @ np.linalg.solve(j_arm @ j_arm.T + damping * np.eye(3), desired_dx)
            ctrl[RIGHT_ARM_ACT] = np.clip(0.85 * ctrl[RIGHT_ARM_ACT] + 0.15 * (q_arm + dq), -2.4, 2.4)
            data.ctrl[:] = ctrl

        apply_root_assist(data, traj, t)
        apply_commanded_pose(model, data, ctrl)
        mujoco.mj_forward(model, data)

        support = support_phase(t)
        if support != locked_support:
            locked_support = support
            plant_xy = data.site_xpos[site_id(model, support_site_name(support)), :2].copy()
        else:
            pin_support_foot(model, data, support, plant_xy)

        mujoco.mj_step(model, data)
        apply_root_assist(data, traj, t)
        apply_commanded_pose(model, data, ctrl)
        mujoco.mj_forward(model, data)

        left_stance = support == "left_stance"
        right_stance = support == "right_stance"
        com_ref = c0 + np.array([ref.position[0], ref.position[1], 0.0])

        log["t"][k] = t
        log["com"][k] = robot_com(model, data)
        log["com_ref"][k] = com_ref
        log["u_body"][k] = [u_body[0], u_body[1], 0.0]
        log["d_body"][k] = [d_body_hat[0], d_body_hat[1], 0.0]
        log["kalman_innovation"][k] = [innovation[0], innovation[1], 0.0]
        log["push_detected"][k] = int(np.linalg.norm(d_body_hat) >= detection_threshold)
        log["rpy"][k] = roll_pitch_yaw_from_body(data, torso)
        log["hand_err"][k] = hand_err
        log["contact"][k] = [int(left_stance), int(right_stance)]
        log["raw_contact"][k] = log["contact"][k]
        log["foot_z"][k] = [data.site_xpos[left_sid, 2], data.site_xpos[right_sid, 2]]
        log["force"][k] = [u_body[0], u_body[1], 0.0]
        log["push_accel"][k] = [push_xy[0], push_xy[1], 0.0]
        log["u_task"][k] = u_task
        log["qpos"][k] = data.qpos

    return log


def summarize(
    log,
    distance: float = DISTANCE,
    push: bool = False,
    push_start: float = 4.5,
    push_duration: float = 0.18,
    push_accel: float = 1.6,
):
    t = log["t"]
    duration_s = float(t[-1] + SIM_DT)
    ramp = float(np.clip(RAMP_TIME, 0.0, 0.5 * duration_s))
    effective_time = max(duration_s - ramp, 1e-9)
    cruise_speed = float(distance) / effective_time
    traveled = float(log["qpos"][-1, 0] - log["qpos"][0, 0])
    foot_z = log["foot_z"]
    left_lift = float(np.max(foot_z[:, 0]) - np.min(foot_z[:, 0]))
    right_lift = float(np.max(foot_z[:, 1]) - np.min(foot_z[:, 1]))
    support_switches = int(np.sum(np.any(np.diff(log["contact"], axis=0) != 0, axis=1)))
    max_abs_roll_pitch = float(np.max(np.abs(log["rpy"][:, :2])))
    min_com_height = float(np.min(log["com"][:, 2]))
    lateral_error = log["com"][:, 1] - log["com_ref"][:, 1]
    max_lateral_deviation = float(np.max(np.abs(lateral_error)))
    detected = log["push_detected"].astype(bool)
    detection_time = None
    detection_latency = None
    if push and np.any(detected):
        detected_after_start = np.flatnonzero(detected & (t >= push_start))
        if len(detected_after_start) > 0:
            detection_time = float(t[detected_after_start[0]])
            detection_latency = float(detection_time - push_start)
    false_positive_samples = int(np.sum(detected & (t < push_start))) if push else int(np.sum(detected))
    push_end = push_start + push_duration
    after = log["t"] >= push_end
    recovery_time = None
    if push and np.any(after):
        within = np.abs(lateral_error) < 0.02
        for idx in np.flatnonzero(after):
            if np.all(within[idx:]):
                recovery_time = float(log["t"][idx] - push_end)
                break
    return {
        "duration_s": duration_s,
        "commanded_distance_m": float(distance),
        "velocity_profile": (
            f"0-{ramp:.1f} s ramp to {cruise_speed:.3f} m/s, "
            f"{ramp:.1f}-{duration_s - ramp:.1f} s cruise, "
            f"{duration_s - ramp:.1f}-{duration_s:.1f} s ramp to 0 m/s"
        ),
        "cruise_speed_mps": cruise_speed,
        "accel_ramp_time_s": ramp,
        "push_enabled": bool(push),
        "push_start_s": float(push_start) if push else None,
        "push_duration_s": float(push_duration) if push else None,
        "push_accel_mps2": float(push_accel) if push else 0.0,
        "root_assist_enabled": True,
        "body_mpc_enabled": True,
        "task_mpc_enabled": True,
        "kalman_detection_enabled": True,
        "kalman_detection_threshold_mps2": 0.20,
        "kalman_detected_push": bool(push and detection_time is not None),
        "kalman_detection_time_s": detection_time,
        "kalman_detection_latency_s": detection_latency,
        "kalman_false_positive_samples": false_positive_samples,
        "distance_m": traveled,
        "distance_error_m": traveled - distance,
        "left_foot_lift_m": left_lift,
        "right_foot_lift_m": right_lift,
        "min_com_height_m": min_com_height,
        "max_abs_roll_pitch_rad": max_abs_roll_pitch,
        "max_body_mpc_accel_mps2": float(np.max(np.linalg.norm(log["u_body"][:, :2], axis=1))),
        "max_estimated_body_disturbance_mps2": float(np.max(np.linalg.norm(log["d_body"][:, :2], axis=1))),
        "max_task_mpc_accel_mps2": float(np.max(np.linalg.norm(log["u_task"], axis=1))),
        "max_lateral_deviation_m": max_lateral_deviation,
        "post_push_recovery_time_s": recovery_time,
        "hand_rms_error_mm": float(1000.0 * np.sqrt(np.mean(np.sum(log["hand_err"]**2, axis=1)))),
        "support_switches": support_switches,
        "one_foot_swing_visible": bool(min(left_lift, right_lift) > 0.03 and support_switches >= 6),
        "fell": False,
        "fall_assessment_valid": False,
        "passes_visual_demo": bool(traveled > 2.5 and min(left_lift, right_lift) > 0.03),
        "interface_note": (
            "Dual-MPC root-assisted G1 MuJoCo visualization based on the "
            "g1_ab_simulation walking scaffold. Body and task commands come "
            "from normalized interaction MPCs, and the body MPC uses the "
            "Kalman-style random-walk disturbance observer in normalized_mpc.py. "
            "The free root is "
            "kinematically assisted and the position-actuated G1 executes "
            "alternating one-foot swing commands. This should not be reported "
            "as torque-level dynamic walking validation or a valid fall test."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=DURATION)
    parser.add_argument("--distance", type=float, default=DISTANCE)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--push-start", type=float, default=4.5)
    parser.add_argument("--push-duration", type=float, default=0.18)
    parser.add_argument("--push-accel", type=float, default=1.6)
    args = parser.parse_args()

    log = run_demo(
        duration=args.duration,
        distance=args.distance,
        push=args.push,
        push_start=args.push_start,
        push_duration=args.push_duration,
        push_accel=args.push_accel,
    )
    summary = summarize(
        log,
        distance=args.distance,
        push=args.push,
        push_start=args.push_start,
        push_duration=args.push_duration,
        push_accel=args.push_accel,
    )
    duration_tag = f"{int(round(args.duration))}s"
    prefix = f"g1_walk_{duration_tag}_push" if args.push else f"g1_walk_{duration_tag}"
    with (RESULTS / f"{prefix}_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(RESULTS / f"{prefix}_log.npz", **log)
    save_plot(log, summary, RESULTS / f"{prefix}.png")
    print(json.dumps(summary, indent=2))
    print(f"saved: {RESULTS / f'{prefix}_summary.json'}")
    print(f"saved: {RESULTS / f'{prefix}_log.npz'}")
    print(f"saved: {RESULTS / f'{prefix}.png'}")


if __name__ == "__main__":
    main()
