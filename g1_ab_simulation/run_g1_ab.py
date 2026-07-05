#!/usr/bin/env python3
"""
G1 A-to-B MuJoCo scaffold.

This is an integration demo for the architecture:
  global waypoints at 10 Hz,
  local configuration-independent double-integrator planner at 100 Hz,
  G1 command layer at 500 Hz,
  MuJoCo physics at 1000 Hz.

By default the script uses a kinematic root-assist mode. That keeps the G1
upright while validating the A-to-B planning and command interfaces. Disable it
with --no-root-assist when replacing the command layer with a true gait/WBC.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from whole_body_control.simulation.controllers.impedance_mpc import ImpedanceMPC
from whole_body_control.simulation.controllers.kalman import KalmanDisturbanceEstimator


MODEL_PATH = HERE.parent / "simulation" / "models" / "g1_wbc.xml"
OUT_DIR = HERE / "results"


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

LEFT_ARM_JOINT_NAMES = ACTUATED_JOINT_NAMES[15:22]
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
    phase_hint: str


@dataclass
class AppleCatchScenario:
    detect_time: float = 1.5
    drop_time: float = 7.0
    start_pos: np.ndarray = None
    catch_pos: np.ndarray = None
    gravity: float = 0.22
    caught: bool = False

    def __post_init__(self):
        if self.start_pos is None:
            self.start_pos = np.array([2.35, -0.18, 2.15], dtype=float)
        if self.catch_pos is None:
            self.catch_pos = np.array([2.35, -0.18, 1.00], dtype=float)

    @property
    def target_xy(self) -> np.ndarray:
        return self.catch_pos[:2].copy()

    def apple_position(self, t: float, hand_pos: np.ndarray | None = None) -> np.ndarray:
        if self.caught and hand_pos is not None:
            return hand_pos.copy()
        if t < self.drop_time:
            return self.start_pos.copy()
        fall_t = t - self.drop_time
        z = max(self.catch_pos[2], self.start_pos[2] - 0.5 * self.gravity * fall_t * fall_t)
        return np.array([self.start_pos[0], self.start_pos[1], z], dtype=float)

    def update_catch(self, t: float, hand_pos: np.ndarray, root_xy: np.ndarray):
        apple_pos = self.apple_position(t)
        near_robot = np.linalg.norm(root_xy - self.target_xy) < 0.46
        near_hand = np.linalg.norm(hand_pos - apple_pos) < 0.45
        low_enough = apple_pos[2] <= self.catch_pos[2] + 0.18
        if (near_robot and low_enough) or near_hand:
            self.caught = True


class WaypointGlobalPlanner:
    """Simple RRT-like waypoint source for the A-to-B scaffold."""

    def __init__(self, start: np.ndarray, goal: np.ndarray):
        mid = np.array([(start[0] + goal[0]) * 0.5, (start[1] + goal[1]) * 0.5 + 0.15])
        self.waypoints = [start.copy(), mid, goal.copy()]
        self.index = 1

    def current_target(self, p_xy: np.ndarray) -> np.ndarray:
        if self.index < len(self.waypoints) - 1:
            if np.linalg.norm(self.waypoints[self.index] - p_xy) < 0.18:
                self.index += 1
        return self.waypoints[self.index].copy()


class DoubleIntegratorLocalPlanner:
    """Closed-form 100 Hz local planner using a virtual double integrator."""

    def __init__(self, dt: float, vmax: float = 0.35, amax: float = 0.60, kp: float = 1.2, kd: float = 1.4):
        self.dt = dt
        self.vmax = vmax
        self.amax = amax
        self.kp = kp
        self.kd = kd
        self.p = np.zeros(2)
        self.v = np.zeros(2)

    def reset(self, p_xy: np.ndarray):
        self.p = p_xy.astype(float).copy()
        self.v[:] = 0.0

    def step(self, target_xy: np.ndarray) -> LocalTrajectory:
        err = target_xy - self.p
        acc = self.kp * err - self.kd * self.v
        norm = np.linalg.norm(acc)
        if norm > self.amax:
            acc *= self.amax / norm
        self.v += acc * self.dt
        speed = np.linalg.norm(self.v)
        if speed > self.vmax:
            self.v *= self.vmax / speed
        self.p += self.v * self.dt + 0.5 * acc * self.dt * self.dt
        heading = math.atan2(self.v[1], self.v[0]) if np.linalg.norm(self.v) > 1e-4 else 0.0
        return LocalTrajectory(self.p.copy(), self.v.copy(), acc.copy(), heading, "walk")


class G1CommandLayer:
    """500 Hz command layer for G1 position actuators."""

    def __init__(self, scripted_catch: bool = True):
        self.ctrl = G1_STAND_CTRL.copy()
        self.scripted_catch = scripted_catch

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

        # G1 actuator order:
        # left leg:  hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
        # right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
        left = slice(0, 6)
        right = slice(6, 12)
        if left_swing:
            ctrl[left] = swing_leg(phase, 1.0)
            ctrl[right] = stance_leg(phase, -1.0)
        else:
            ctrl[left] = stance_leg(phase, 1.0)
            ctrl[right] = swing_leg(phase, -1.0)

        # Keep torso nearly upright. Arms swing contralaterally to the legs:
        # left arm back when left leg swings forward, and vice versa.
        arm_phase = math.sin(2.0 * math.pi * cycle)
        ctrl[12:15] = np.array([0.0, 0.0, 0.03 * arm_phase])
        ctrl[15:22] = np.array([0.12 + 0.18 * arm_phase, 0.20, 0.0, 1.18, 0.0, 0.0, 0.0])
        ctrl[22:29] = np.array([0.12 - 0.18 * arm_phase, -0.20, 0.0, 1.18, 0.0, 0.0, 0.0])
        if self.scripted_catch and traj.phase_hint == "catch":
            ctrl[12:15] = np.array([0.0, 0.0, -0.18])
            ctrl[15:22] = np.array([-0.70, 0.65, -0.05, 0.95, 0.0, 0.0, 0.0])
            ctrl[22:29] = np.array([-0.95, -0.78, 0.10, 1.00, 0.0, 0.0, 0.0])
        self.ctrl = ctrl
        return ctrl


class ArmImpedanceMPCAdapter:
    """Impedance-MPC end-effector task mapped onto G1 position actuators."""

    def __init__(self, model: mujoco.MjModel, side: str = "left", dt: float = 0.002):
        self.side = side
        self.dt = dt
        self.site_name = f"{side}_hand_site"
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.site_name)
        if self.site_id < 0:
            raise ValueError(f"Missing site {self.site_name} in model")
        self.joint_names = LEFT_ARM_JOINT_NAMES if side == "left" else RIGHT_ARM_JOINT_NAMES
        self.ctrl_slice = slice(15, 22) if side == "left" else slice(22, 29)
        self.dof_ids = []
        self.qpos_ids = []
        for joint_name in self.joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.dof_ids.append(model.jnt_dofadr[jid])
            self.qpos_ids.append(model.jnt_qposadr[jid])

        # Cost weights mirror v2_strong scenario_c_g1 (the canonical G1 arm task):
        # position weight softened to 6e4 (was 3e6 there / 9000 here), R=0.01.
        # N/dt stay tied to this scaffold's 500 Hz command layer. F_max stays at 35
        # because the MPC force is bridged to arm position through a damped Jacobian
        # (desired_dx clipped at +-0.055), which saturates well below 80 N — so a
        # larger F_max only inflates the logged force without changing behavior.
        q_mpc = np.diag([6e4, 6e4, 6e4, 60.0, 60.0, 60.0])
        r_mpc = 0.01 * np.eye(3)
        self.mpc = ImpedanceMPC(N=18, dt=dt, Q=q_mpc, R=r_mpc, F_max=35.0)
        self.lambda_arm = 0.30 * np.eye(3)
        mode = self.mpc.precompute_mode("g1_double_support", self.lambda_arm)
        self.kalman = KalmanDisturbanceEstimator(dt=dt, Q_w_d=2e-2)
        self.kalman.set_mode(self.mpc.A_d, mode["B_d"])
        self.force_prev = np.zeros(3)
        self.force_cmd = np.zeros(3)
        self.force_to_step = 0.0022

    def hand_state(self, model: mujoco.MjModel, data: mujoco.MjData):
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, self.site_id)
        pos = data.site_xpos[self.site_id].copy()
        vel = jacp @ data.qvel
        return pos, vel, jacp[:, self.dof_ids]

    def update(self, model: mujoco.MjModel, data: mujoco.MjData,
               ctrl: np.ndarray, target_pos: np.ndarray):
        hand_pos, hand_vel, jac_arm = self.hand_state(model, data)
        e_pos = hand_pos - target_pos
        e_vel = hand_vel

        self.kalman.predict(self.force_prev)
        _, d_hat = self.kalman.update(e_pos, e_vel)
        force_mpc = self.mpc.solve(np.concatenate([e_pos, e_vel]),
                                   self.lambda_arm,
                                   "g1_double_support",
                                   d_hat,
                                   use_osqp=False)
        self.force_prev = force_mpc
        self.force_cmd = -force_mpc

        desired_dx = np.clip(-self.force_to_step * force_mpc, -0.055, 0.055)
        damping = 2e-3
        dq = jac_arm.T @ np.linalg.solve(jac_arm @ jac_arm.T + damping * np.eye(3),
                                         desired_dx)
        q_now = np.array([data.qpos[qadr] for qadr in self.qpos_ids])
        q_nom = ctrl[self.ctrl_slice].copy()
        q_cmd = 0.72 * q_nom + 0.28 * (q_now + dq)
        ctrl[self.ctrl_slice] = np.clip(q_cmd, -2.4, 2.4)
        return e_pos, e_vel, self.force_cmd


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])


def apply_root_assist(data: mujoco.MjData, traj: LocalTrajectory, t: float, height: float = 0.82):
    """Kinematically guide the free root for architecture validation."""
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
    """Mirror the command into qpos for a clear root-assisted walking demo."""
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
    """Keep the stance foot near its plant point without hard-snapping the base.

    A full per-step correction (alpha=1) yanks the whole base fore/aft every
    step to cancel the stance-foot slide from the swinging gait pose, which
    injects a step-rate (~1.55 Hz) wobble into the base and, through it, the
    hands. Low-passing the correction (alpha~0.1) lets the base follow the
    smooth root-assist planner while the foot still stays close to planted.
    """
    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, support_site_name(support))
    foot_xy = data.site_xpos[foot_id, :2].copy()
    data.qpos[:2] += alpha * (plant_xy - foot_xy)
    data.qvel[:2] = 0.0
    mujoco.mj_forward(model, data)


def render_frame(model, data, renderer, cam):
    cam.lookat[:] = [data.qpos[0] + 0.25, data.qpos[1] - 0.15, 0.95]
    renderer.update_scene(data, camera=cam)
    return renderer.render().copy()


def save_path_plot(rows, waypoints, goal, out_path):
    arr = np.array([[r["t"], r["x"], r["y"], r["x_ref"], r["y_ref"]] for r in rows], dtype=float)
    plt.figure(figsize=(7.0, 5.0))
    plt.plot(arr[:, 1], arr[:, 2], label="G1 base")
    plt.plot(arr[:, 3], arr[:, 4], "--", label="local plan")
    wp = np.array(waypoints)
    plt.plot(wp[:, 0], wp[:, 1], "ko-", label="global waypoints")
    plt.plot(goal[0], goal[1], "rx", markersize=10, label="goal")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("G1 A-to-B Planner Scaffold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_fallback_video(rows, waypoints, goal, out_path, fps):
    """Create an MP4 from the simulated trajectory when MuJoCo GL is unavailable."""
    arr = np.array([[r["t"], r["x"], r["y"], r["x_ref"], r["y_ref"], r["vx_ref"], r["vy_ref"]] for r in rows], dtype=float)
    wp = np.array(waypoints)
    xmin = min(arr[:, 1].min(), arr[:, 3].min(), wp[:, 0].min(), goal[0]) - 0.25
    xmax = max(arr[:, 1].max(), arr[:, 3].max(), wp[:, 0].max(), goal[0]) + 0.25
    ymin = min(arr[:, 2].min(), arr[:, 4].min(), wp[:, 1].min(), goal[1]) - 0.25
    ymax = max(arr[:, 2].max(), arr[:, 4].max(), wp[:, 1].max(), goal[1]) + 0.25

    step = max(1, int(round((1.0 / fps) / 0.01)))  # rows are logged every 10 ms
    frame_ids = list(range(0, len(rows), step))
    if frame_ids[-1] != len(rows) - 1:
        frame_ids.append(len(rows) - 1)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=120)
    with imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for idx in frame_ids:
            ax.clear()
            ax.plot(wp[:, 0], wp[:, 1], "ko-", lw=1.8, label="global waypoints")
            ax.plot(arr[: idx + 1, 3], arr[: idx + 1, 4], "--", color="tab:orange", lw=2.0, label="local plan")
            ax.plot(arr[: idx + 1, 1], arr[: idx + 1, 2], color="tab:blue", lw=2.4, label="G1 base")
            ax.plot(goal[0], goal[1], "rx", markersize=10, mew=2, label="goal")

            x, y = arr[idx, 1], arr[idx, 2]
            vx, vy = arr[idx, 5], arr[idx, 6]
            theta = math.atan2(vy, vx) if math.hypot(vx, vy) > 1e-4 else 0.0
            body = plt.Circle((x, y), 0.045, fill=False, lw=2.0, color="black")
            ax.add_patch(body)
            ax.arrow(x, y, 0.10 * math.cos(theta), 0.10 * math.sin(theta),
                     width=0.006, head_width=0.035, head_length=0.035,
                     length_includes_head=True, color="black")

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_title(f"G1 A-to-B Scaffold  t={arr[idx, 0]:.2f}s")
            ax.legend(loc="upper left", fontsize=8)
            fig.tight_layout()
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            writer.append_data(frame)
    plt.close(fig)


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    sim_dt = 0.001
    global_dt = 0.10
    local_dt = 0.01
    command_dt = 0.002
    model.opt.timestep = sim_dt

    start = np.array([data.qpos[0], data.qpos[1]], dtype=float)
    goal = np.array(args.goal, dtype=float)
    apple = AppleCatchScenario() if args.apple_catch else None
    global_planner = WaypointGlobalPlanner(start, goal)
    local_planner = DoubleIntegratorLocalPlanner(dt=local_dt)
    local_planner.reset(start)
    use_impedance_mpc = args.wbc_control == "impedance-mpc"
    command = G1CommandLayer(scripted_catch=not use_impedance_mpc)
    arm_mpc = ArmImpedanceMPCAdapter(model, side="left", dt=command_dt) if use_impedance_mpc else None
    traj = LocalTrajectory(start.copy(), np.zeros(2), np.zeros(2), 0.0, "stand")
    target = goal.copy()
    arm_error = np.full(3, np.nan)
    arm_force = np.full(3, np.nan)
    apple_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "apple_body")
    apple_mocap_id = model.body_mocapid[apple_body_id] if apple_body_id >= 0 else -1
    left_hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_hand_site")
    right_hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
    if apple is not None and apple_mocap_id >= 0:
        data.mocap_pos[apple_mocap_id] = apple.start_pos

    requested_video = bool(args.video)
    fallback_video = False
    renderer = None
    cam = None
    frames = []
    if args.video:
        try:
            renderer = mujoco.Renderer(model, height=480, width=640)
            cam = mujoco.MjvCamera()
            cam.distance = 3.0 if args.apple_catch else 2.45
            cam.azimuth = -58.0 if args.apple_catch else 74.0
            cam.elevation = -11.0 if args.apple_catch else -9.0
        except Exception as exc:
            print(f"MuJoCo renderer unavailable ({exc}); using trajectory-video fallback.")
            args.video = False
            fallback_video = True

    rows = []
    locked_support = None
    plant_xy = None
    n_steps = int(args.duration / sim_dt)
    global_period = max(1, int(global_dt / sim_dt))
    local_period = max(1, int(local_dt / sim_dt))
    command_period = max(1, int(command_dt / sim_dt))
    video_period = max(1, int(1.0 / (args.fps * sim_dt)))

    for k in range(n_steps):
        t = k * sim_dt
        p_xy = data.qpos[:2].copy()

        if k % global_period == 0:
            target = global_planner.current_target(p_xy)
            if apple is not None and t >= apple.detect_time and not apple.caught:
                target = apple.target_xy

        if k % local_period == 0:
            traj = local_planner.step(target)
            if apple is not None and t >= apple.detect_time and not apple.caught:
                traj.phase_hint = "catch"

        if k % command_period == 0:
            data.ctrl[:] = command.step(t, traj)

        if args.root_assist:
            apply_root_assist(data, traj, t)
            apply_commanded_pose(model, data, data.ctrl)
            mujoco.mj_forward(model, data)
            if arm_mpc is not None and apple is not None and t >= apple.detect_time and not apple.caught:
                arm_target = apple.apple_position(t)
                arm_error, _, arm_force = arm_mpc.update(model, data, data.ctrl, arm_target)
                apply_commanded_pose(model, data, data.ctrl)
                mujoco.mj_forward(model, data)
            support = support_phase(t)
            if support != locked_support:
                locked_support = support
                foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, support_site_name(support))
                plant_xy = data.site_xpos[foot_id, :2].copy()
            else:
                pin_support_foot(model, data, support, plant_xy)

        mujoco.mj_step(model, data)

        if args.root_assist:
            apply_root_assist(data, traj, t)
            apply_commanded_pose(model, data, data.ctrl)
            mujoco.mj_forward(model, data)
            if arm_mpc is not None and apple is not None and t >= apple.detect_time and not apple.caught:
                arm_target = apple.apple_position(t)
                arm_error, _, arm_force = arm_mpc.update(model, data, data.ctrl, arm_target)
                apply_commanded_pose(model, data, data.ctrl)
                mujoco.mj_forward(model, data)
            support = support_phase(t)
            if support != locked_support:
                locked_support = support
                foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, support_site_name(support))
                plant_xy = data.site_xpos[foot_id, :2].copy()
            else:
                pin_support_foot(model, data, support, plant_xy)

        if apple is not None and apple_mocap_id >= 0:
            left_hand = data.site_xpos[left_hand_id].copy() if left_hand_id >= 0 else None
            right_hand = data.site_xpos[right_hand_id].copy() if right_hand_id >= 0 else None
            catch_hand = left_hand if left_hand is not None else (right_hand if right_hand is not None else None)
            if catch_hand is not None:
                apple.update_catch(t, catch_hand, data.qpos[:2].copy())
            data.mocap_pos[apple_mocap_id] = apple.apple_position(t, catch_hand)
            mujoco.mj_forward(model, data)

        if k % 10 == 0:
            left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
            right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
            left_knee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_knee_link")
            right_knee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_knee_link")
            left_foot = data.site_xpos[left_foot_id].copy()
            right_foot = data.site_xpos[right_foot_id].copy()
            left_knee = data.xpos[left_knee_id].copy()
            right_knee = data.xpos[right_knee_id].copy()
            apple_pos = data.mocap_pos[apple_mocap_id].copy() if apple is not None and apple_mocap_id >= 0 else np.full(3, np.nan)
            left_hand_pos = data.site_xpos[left_hand_id].copy() if left_hand_id >= 0 else np.full(3, np.nan)
            hand_pos = data.site_xpos[right_hand_id].copy() if right_hand_id >= 0 else np.full(3, np.nan)
            rows.append({
                "t": t,
                "x": float(data.qpos[0]),
                "y": float(data.qpos[1]),
                "x_ref": float(traj.position[0]),
                "y_ref": float(traj.position[1]),
                "target_x": float(target[0]),
                "target_y": float(target[1]),
                "vx_ref": float(traj.velocity[0]),
                "vy_ref": float(traj.velocity[1]),
                "support": support_phase(t),
                "left_foot_x": float(left_foot[0]),
                "left_foot_y": float(left_foot[1]),
                "left_foot_z": float(left_foot[2]),
                "left_knee_x": float(left_knee[0]),
                "left_knee_y": float(left_knee[1]),
                "left_knee_z": float(left_knee[2]),
                "right_foot_x": float(right_foot[0]),
                "right_foot_y": float(right_foot[1]),
                "right_foot_z": float(right_foot[2]),
                "right_knee_x": float(right_knee[0]),
                "right_knee_y": float(right_knee[1]),
                "right_knee_z": float(right_knee[2]),
                "apple_x": float(apple_pos[0]),
                "apple_y": float(apple_pos[1]),
                "apple_z": float(apple_pos[2]),
                "left_hand_x": float(left_hand_pos[0]),
                "left_hand_y": float(left_hand_pos[1]),
                "left_hand_z": float(left_hand_pos[2]),
                "right_hand_x": float(hand_pos[0]),
                "right_hand_y": float(hand_pos[1]),
                "right_hand_z": float(hand_pos[2]),
                "apple_caught": bool(apple.caught) if apple is not None else False,
                "wbc_control": args.wbc_control,
                "arm_error_x": float(arm_error[0]),
                "arm_error_y": float(arm_error[1]),
                "arm_error_z": float(arm_error[2]),
                "arm_force_x": float(arm_force[0]),
                "arm_force_y": float(arm_force[1]),
                "arm_force_z": float(arm_force[2]),
            })

        if args.video and k % video_period == 0:
            frames.append(render_frame(model, data, renderer, cam))

    if renderer is not None:
        renderer.close()

    csv_path = OUT_DIR / "g1_ab_log.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    save_path_plot(rows, global_planner.waypoints, goal, OUT_DIR / "g1_ab_path.png")

    if args.video:
        video_path = OUT_DIR / "g1_ab_demo.mp4"
        with imageio.get_writer(str(video_path), fps=args.fps, codec="libx264", quality=8, macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(frame)
    elif requested_video and fallback_video:
        video_path = OUT_DIR / "g1_ab_demo.mp4"
        save_fallback_video(rows, global_planner.waypoints, goal, video_path, args.fps)

    final = np.array([rows[-1]["x"], rows[-1]["y"]])
    err = np.linalg.norm(final - goal)
    print(f"Final position: {final.round(3)}  goal: {goal.round(3)}  error: {err:.3f} m")
    print(f"Wrote {csv_path}")
    print(f"Wrote {OUT_DIR / 'g1_ab_path.png'}")
    if args.video or (requested_video and fallback_video):
        print(f"Wrote {OUT_DIR / 'g1_ab_demo.mp4'}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", nargs=2, type=float, default=[1.2, 0.35], metavar=("X", "Y"))
    parser.add_argument("--duration", type=float, default=7.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--apple-catch", action="store_true")
    parser.add_argument("--wbc-control", choices=["impedance-mpc", "scripted"], default="impedance-mpc")
    parser.add_argument("--root-assist", dest="root_assist", action="store_true", default=True)
    parser.add_argument("--no-root-assist", dest="root_assist", action="store_false")
    parser.add_argument("--video", dest="video", action="store_true", default=True)
    parser.add_argument("--no-video", dest="video", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
