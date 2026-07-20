#!/usr/bin/env python3
"""Paired torque-level G1 uneven-ground interaction benchmark.

Every controller uses the same external DCM reference, physical terrain,
measured-contact touchdown reflex, and inverse-dynamics/contact QP.  Only the
body-acceleration correction law changes.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from interaction_estimator import FilteredAccelerationResidualEstimator
from normalized_mpc import NormalizedMPC
from reference_provider import DCMReferenceProvider
from capture_point import CapturePointStabilizer, StabilizerParams
from run_g1_root_assist_demo import (
    ACTUATED_JOINT_NAMES,
    robot_com,
    roll_pitch_yaw_from_body,
    site_jac,
)
from run_g1_torque_realizer_benchmark import (
    InverseDynamicsQPRealizer,
    TORQUE_STAND_CTRL,
    body_id,
    com_velocity,
    friction_margin,
    generate_torque_model,
    hand_state,
    joint_id,
    measured_foot_contacts,
    site_id,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
SIM_DT = 0.001
WBC_DT = 0.002
MPC_DT = 0.010
# Publication controllers.  Development-only ablations must be requested
# explicitly so the authoritative artifacts cannot silently contain rows that
# are absent from the manuscript.
CONTROLLERS = ("impedance", "nominal_mpc", "interaction_mpc")
DIAGNOSTIC_CONTROLLERS = ("no_realization_feedback",)
ALL_CONTROLLERS = CONTROLLERS + DIAGNOSTIC_CONTROLLERS
TERRAINS = ("flat", "depression", "obstacle", "rough")
CHALLENGE_TERRAINS = ("platform",)

# Frozen nonlinear conditioning used by the evaluated ID-MPC.  These constants
# are part of the controller, not incidental numerical guards, and are exported
# in every publication artifact.
RESIDUAL_DEADBAND_TASK_ACC = 0.30
RESIDUAL_CAP_TASK_ACC = 0.50
COMMAND_SLEW_TASK_ACC_PER_UPDATE = 0.70
PUBLICATION_GAIT = {
    "n_steps": 12,
    "step_length": 0.03,
    "step_time": 1.40,
    "double_support_time": 1.00,
    "settle_time": 1.00,
    "lateral_zmp_scale": 1.00,
    "smooth_double_support": True,
    "zmp_transfer_time": 0.05,
    "smooth_lateral_only": False,
}


@dataclass(frozen=True)
class PushSpec:
    """A phase-locked, finite-duration torso wrench applied to the plant only.

    The wrench perturbs the simulated robot and is logged at 1 kHz for ground
    truth, but is never supplied to the estimator or any controller in the
    primary comparison (Stage 3B of the change plan).
    """

    direction: str          # "lateral" (+y) or "forward" (+x)
    phase: str              # "double_support" or "single_support"
    magnitude_n: float      # peak force of the half-sine profile
    duration_s: float = 0.15
    earliest_onset_s: float = 1.6
    start_time_s: float | None = None
    profile: str = "half_sine"  # "half_sine" or smooth "flat_top"
    ramp_s: float = 0.10
    dwell_s: float = 0.06   # measured contact phase must hold this long before onset

    def axis(self) -> int:
        return 1 if self.direction == "lateral" else 0

    def force_at(self, since: float) -> float:
        if since < 0.0 or since > self.duration_s:
            return 0.0
        if self.profile == "flat_top":
            ramp = min(max(self.ramp_s, 0.0), 0.5 * self.duration_s)
            if ramp <= 0.0:
                return self.magnitude_n
            if since < ramp:
                return self.magnitude_n * 0.5 * (1.0 - math.cos(math.pi * since / ramp))
            if since > self.duration_s - ramp:
                tail = self.duration_s - since
                return self.magnitude_n * 0.5 * (1.0 - math.cos(math.pi * tail / ramp))
            return self.magnitude_n
        if self.profile != "half_sine":
            raise ValueError(f"unknown push profile {self.profile!r}")
        return self.magnitude_n * math.sin(math.pi * since / self.duration_s)


PUSH_CONDITIONS = (
    ("lateral", "double_support"),
    ("lateral", "single_support"),
    ("forward", "double_support"),
    ("forward", "single_support"),
)


def _add_box(world, name, center, half_size, rgba="0.35 0.30 0.22 1"):
    ET.SubElement(world, "geom", {
        "name": name, "type": "box",
        "pos": " ".join(f"{x:.6g}" for x in center),
        "size": " ".join(f"{x:.6g}" for x in half_size),
        "rgba": rgba, "friction": "0.9 0.02 0.001", "condim": "6",
        "contype": "1", "conaffinity": "1",
    })


def generate_terrain_model(terrain: str, height_mm: float = 20.0,
                           platform_start_x: float = 0.025,
                           platform_end_x: float = 0.065) -> Path:
    """Create deterministic terrain while preserving a flat nominal plan.

    ``height_mm`` sets the depression/obstacle amplitude (default 20 mm keeps the
    frozen benchmark unchanged); the platform vignette sweeps 20/30/40 mm.
    """
    if terrain not in TERRAINS + CHALLENGE_TERRAINS:
        raise ValueError(terrain)
    h = float(height_mm) / 1000.0
    base = generate_torque_model()
    tree = ET.parse(base)
    root = tree.getroot()
    root.find("compiler").set("meshdir", str((HERE / "models" / "assets").resolve()))
    world = root.find("worldbody")
    floor = next(g for g in world.findall("geom") if g.attrib.get("name") == "floor")
    if terrain in ("depression", "rough"):
        floor.set("pos", f"0 0 {-h:.4f}")
        # Common initial platform: both initial feet see exactly the flat model.
        _add_box(world, "terrain_start", (-0.25, 0.0, -h / 2), (0.29, 0.34, h / 2))
    if terrain == "depression":
        # Left lane stays nominal; the right lane drops to the lower base plane.
        _add_box(world, "terrain_left_nominal", (0.28, 0.135, -h / 2), (0.24, 0.105, h / 2))
    elif terrain == "obstacle":
        # A finite future patch, separated from the initial feet.  The previous
        # implementation used a long raised lane beginning near the initial
        # stance; settling on that lane absorbed the height into the initial
        # condition and made the 20/30/40-mm sweep nearly invariant.  This patch
        # starts at x=0.22 m and therefore requires a longer dedicated obstacle
        # protocol than the legacy 4-s window before it may support a paper
        # claim.  The larger separation also clears the initial foot collision
        # geometry, not only the nominal foot-site origin.
        _add_box(world, "terrain_right_obstacle", (0.28, -0.135, h / 2),
                 (0.06, 0.105, h / 2))
    elif terrain == "rough":
        # Frozen two-lane sequence: left +15 mm, right -h after x=0.04 m.
        _add_box(world, "terrain_left_high", (0.28, 0.135, (0.015 - h) / 2),
                 (0.24, 0.105, (0.015 + h) / 2))
    elif terrain == "platform":
        if platform_end_x <= platform_start_x:
            raise ValueError("platform_end_x must exceed platform_start_x")
        center_x = 0.5 * (platform_start_x + platform_end_x)
        half_x = 0.5 * (platform_end_x - platform_start_x)
        _add_box(world, "challenge_platform", (center_x, 0.0, h / 2),
                 (half_x, 0.34, h / 2), rgba="0.22 0.42 0.62 1")
    out = RESULTS / f"g1_wbc_torque_{terrain}_{int(round(height_mm))}.xml"
    tree.write(out, encoding="unicode")
    return out


def _timing(values: list[float], deadline_ms: float) -> dict:
    a = np.asarray(values, float)
    if not len(a):
        return {"n": 0, "median_ms": None, "p99_ms": None, "max_ms": None,
                "deadline_ms": deadline_ms, "deadline_miss_fraction": None}
    return {
        "n": int(a.size), "median_ms": float(np.median(a)),
        "p99_ms": float(np.percentile(a, 99)), "max_ms": float(np.max(a)),
        "deadline_ms": float(deadline_ms),
        "deadline_miss_fraction": float(np.mean(a > deadline_ms)),
    }


def _condition_residual(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, float)
    return np.sign(value) * np.clip(
        np.abs(value) - RESIDUAL_DEADBAND_TASK_ACC,
        0.0,
        RESIDUAL_CAP_TASK_ACC,
    )


def _prediction_rmse(log: dict, dt: float) -> dict:
    """Offline model audit using recorded future inputs but no future outputs."""
    e = np.asarray(log["task_error"])
    ed = np.asarray(log["task_velocity_error"])
    u = np.asarray(log["correction"])
    d = np.asarray(log["conditioned_effective_estimate"])
    valid = np.asarray(log["mpc_sample"], bool) & ~np.asarray(log["fell"], bool)
    idx = np.flatnonzero(valid)
    horizons = (1, 5, 10)
    out = {}
    for h in horizons:
        nominal, augmented = [], []
        for k in idx:
            if k + h >= len(e):
                continue
            en = e[k].copy(); vn = ed[k].copy()
            ea = en.copy(); va = vn.copy(); dh = d[k]
            for j in range(h):
                uj = u[k + j]
                en = en + dt * vn + 0.5 * dt * dt * uj
                vn = vn + dt * uj
                aa = uj + dh
                ea = ea + dt * va + 0.5 * dt * dt * aa
                va = va + dt * aa
            nominal.append(en - e[k + h])
            augmented.append(ea - e[k + h])
        nn = np.asarray(nominal); aa = np.asarray(augmented)
        out[str(h)] = {
            "horizon_ms": float(1000.0 * h * dt), "n": int(len(nn)),
            "nominal_com_rmse_mm": None if not len(nn) else float(
                1000 * np.sqrt(np.mean(nn[:, :3] ** 2))
            ),
            "augmented_com_rmse_mm": None if not len(aa) else float(
                1000 * np.sqrt(np.mean(aa[:, :3] ** 2))
            ),
            "nominal_roll_pitch_rmse_mrad": None if not len(nn) else float(
                1000 * np.sqrt(np.mean(nn[:, 3:5] ** 2))
            ),
            "augmented_roll_pitch_rmse_mrad": None if not len(aa) else float(
                1000 * np.sqrt(np.mean(aa[:, 3:5] ** 2))
            ),
        }
    return out


def _geom_contact_active(model: mujoco.MjModel, data: mujoco.MjData,
                         geom_name: str) -> bool:
    """Whether a named terrain geom participates in a current contact."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        return False
    return any(data.contact[i].geom1 == gid or data.contact[i].geom2 == gid
               for i in range(data.ncon))


def run_trial(controller: str, terrain: str, seed: int, duration: float = 4.0,
              push: PushSpec | None = None,
              mpc_solve_dt: float = MPC_DT,
              video: dict | None = None,
              gait: dict | None = None,
              terrain_height_mm: float = 20.0,
              platform_start_x: float = 0.025,
              platform_end_x: float = 0.065,
              capture_gain: tuple[float, float] | None = None,
              stabilizer: StabilizerParams | None = None,
              phase_sync: bool = False) -> tuple[dict, dict]:
    # ``stabilizer`` enables the shared, controller-independent capture-point gait
    # stabilizer (capture_point.py): discrete touchdown anchoring + one-step
    # predicted-touchdown DCM foot placement.  It reads only the physical CoM
    # state and footholds, so every controller uses the same module and params.
    # ``capture_gain`` is the earlier diagnostic prototype, kept for reference.
    # Both default to off.  Publication results must be regenerated whenever
    # this shared gait or realizer changes.
    # ``mpc_solve_dt`` sets only how often the interaction MPC is re-solved; the
    # horizon model keeps dt=MPC_DT (0.25 s preview) so a faster solve rate is a
    # clean update-rate change, not a shorter horizon.
    if controller not in ALL_CONTROLLERS or terrain not in TERRAINS + CHALLENGE_TERRAINS:
        raise ValueError((controller, terrain))
    rng = np.random.default_rng(seed)
    model_path = generate_terrain_model(
        terrain, height_mm=terrain_height_mm,
        platform_start_x=platform_start_x, platform_end_x=platform_end_x,
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for value, name in zip(TORQUE_STAND_CTRL, ACTUATED_JOINT_NAMES):
        data.qpos[model.jnt_qposadr[joint_id(model, name)]] = value
    # Tiny paired perturbation makes seeds meaningful without changing terrain.
    data.qvel[6:] += rng.normal(0.0, 2e-4, size=model.nv - 6)
    mujoco.mj_forward(model, data)

    torso = body_id(model, "torso_link")
    hand_sid = site_id(model, "right_hand_site")
    left_sid = site_id(model, "left_foot")
    right_sid = site_id(model, "right_foot")
    realizer = InverseDynamicsQPRealizer(model, exact_realizer=False)
    realizer.task_weight = 0.0
    realizer.com_task_weight = 400.0

    q_nom = TORQUE_STAND_CTRL.copy()
    qd_nom = np.zeros_like(q_nom)
    # Establish physical contact before recording or freezing stance targets.
    warm_com = robot_com(model, data)
    warm_height = float(data.qpos[2])
    for _ in range(round(0.35 / SIM_DT)):
        contacts = realizer.contact_points(model, data, ("left", "right"))
        targets = {key: pos.copy() for key, (pos, _) in contacts.items()}
        _, _, hand_jac = hand_state(model, data, hand_sid)
        rpy = roll_pitch_yaw_from_body(data, torso)
        acc = -25.0 * (robot_com(model, data) - warm_com) - 8.0 * com_velocity(
            model, data, realizer.root_body
        )
        realizer.command(
            model, data, q_nom, qd_nom, np.zeros(2), np.zeros(3), hand_jac,
            contacts, targets, warm_height, rpy,
            com_acc_des=np.clip(acc, -3.0, 3.0), attitude_weight=120.0,
        )
        mujoco.mj_step(model, data); mujoco.mj_forward(model, data)
    initial_obstacle_contact = _geom_contact_active(
        model, data, "terrain_right_obstacle"
    )
    if terrain == "obstacle" and initial_obstacle_contact:
        raise RuntimeError(
            "invalid obstacle protocol: raised patch contacts the robot during settling"
        )
    data.time = 0.0

    com0 = robot_com(model, data)
    base_height_reference = float(data.qpos[2])
    ground_z_nominal = float(min(data.site_xpos[left_sid, 2], data.site_xpos[right_sid, 2]))
    left0 = data.site_xpos[left_sid, :2].copy()
    right0 = data.site_xpos[right_sid, :2].copy()
    g = PUBLICATION_GAIT.copy()
    if gait:
        g.update(gait)
    plan = DCMReferenceProvider(
        left0, right0, step_length=g["step_length"], n_steps=g["n_steps"],
        com_height=float(com0[2] - ground_z_nominal), step_time=g["step_time"],
        double_support_time=g["double_support_time"], settle_time=g["settle_time"],
        lateral_zmp_scale=g["lateral_zmp_scale"],
        smooth_double_support=g["smooth_double_support"],
        zmp_transfer_time=g["zmp_transfer_time"],
        smooth_lateral_only=g["smooth_lateral_only"],
    )
    task_dim = 5  # CoM x/y/z and body roll/pitch
    mpc = NormalizedMPC(
        dim=task_dim, dt=MPC_DT, horizon=25, q_pos=90.0, q_vel=16.0,
        r=0.05, u_max=np.array([6.0, 6.0, 4.0, 15.0, 15.0]),
    )
    estimator = FilteredAccelerationResidualEstimator(task_dim, WBC_DT, bandwidth_hz=3.0)

    n = int(round(duration / SIM_DT))
    log = {
        "t": np.zeros(n), "com": np.zeros((n, 3)), "com_ref": np.zeros((n, 3)),
        "task_error": np.zeros((n, task_dim)),
        "task_velocity_error": np.zeros((n, task_dim)),
        "rpy": np.zeros((n, 3)), "contact": np.zeros((n, 2), int),
        "foot_position": np.zeros((n, 2, 3)), "phase_index": np.zeros(n, int),
        "planned_stance": np.zeros((n, 2), int),
        "correction": np.zeros((n, task_dim)), "command_acc": np.zeros((n, task_dim)),
        "realized_acc": np.zeros((n, task_dim)),
        "qp_predicted_acc": np.zeros((n, task_dim)),
        "interaction_estimate": np.zeros((n, task_dim)),
        "realization_estimate": np.zeros((n, task_dim)),
        "effective_estimate": np.zeros((n, task_dim)),
        "conditioned_effective_estimate": np.zeros((n, task_dim)),
        "realization_residual": np.zeros((n, task_dim)), "contact_force_n": np.zeros(n),
        "friction_margin": np.zeros(n), "torque_utilization": np.zeros(n),
        "qp_fallback": np.zeros(n, int), "fell": np.zeros(n, int),
        "mpc_sample": np.zeros(n, int), "wbc_sample": np.zeros(n, int),
        "obstacle_contact": np.zeros(n, int),
    }
    com_ref_xy = com0[:2].copy()
    com_ref_velocity = np.zeros(2)
    com_ref_acceleration = np.zeros(2)
    correction = np.zeros(task_dim)
    estimate_interaction = np.zeros(task_dim)
    estimate_realization = np.zeros(task_dim)
    estimate_effective = np.zeros(task_dim)
    conditioned_effective = np.zeros(task_dim)
    realized_error_acceleration = np.zeros(task_dim)
    measured_error_acceleration = np.zeros(task_dim)
    stance_previous: tuple[str, ...] = ()
    stance_targets: dict[str, np.ndarray] = {}
    pending_foot = None
    pending_xy = None
    pending_since = None
    last_swing = None
    last_swing_xy = None
    last_swing_nominal_xy = None
    pending_nominal_xy = None
    synced_swing = None
    if phase_sync:
        plan.reset_sync()
    stab = (CapturePointStabilizer(omega=plan.omega, params=stabilizer)
            if stabilizer is not None else None)
    mpc_ms, wbc_ms = [], []
    last_wbc_k = -999
    last_mpc_k = -999
    tau_hold = np.clip(data.qfrc_bias[realizer.dof], realizer.torque_min, realizer.torque_max)
    q_servo_reference, qd_servo_reference = realizer.joint_state(data)
    servo_kp = 20.0
    servo_kd = 1.2

    # Phase-locked torso-wrench state (plant-only; hidden from the estimator).
    push_onset: float | None = None
    push_onset_contact: tuple[int, int] | None = None
    push_match_since: float | None = None  # start of the current measured-phase run
    if push is not None:
        log["applied_force"] = np.zeros((n, 3))

    # Optional offscreen video capture (only used for illustration runs).
    renderer = None
    frames: list = []
    cam = None
    v_stride = 1
    if video is not None:
        renderer = mujoco.Renderer(model, height=video.get("height", 480),
                                   width=video.get("width", 640))
        v_stride = max(1, round(1.0 / (video.get("fps", 30) * SIM_DT)))
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = torso
        cam.distance = video.get("distance", 3.2)
        cam.azimuth = video.get("azimuth", 135.0)
        cam.elevation = video.get("elevation", -12.0)

    for k in range(n):
        t = k * SIM_DT
        measured = measured_foot_contacts(model, data)
        if phase_sync:
            sw_idx = 1 if synced_swing == "right" else 0
            swing_down = bool(measured[sw_idx]) if synced_swing is not None else False
            sample = plan.sample_synced(SIM_DT, swing_down)
            synced_swing = sample.swing
        else:
            sample = plan.sample(t)
        if last_swing is not None and sample.swing is None and pending_foot is None:
            pending_foot, pending_xy, pending_since = last_swing, last_swing_xy.copy(), t
            pending_nominal_xy = (last_swing_nominal_xy.copy()
                                  if last_swing_nominal_xy is not None else pending_xy.copy())
        if sample.swing is not None:
            last_swing, last_swing_xy = sample.swing, sample.swing_target_xy.copy()
            last_swing_nominal_xy = sample.swing_target_xy.copy()
            if stab is not None and sample.swing != stab._swing:
                stab.on_new_swing(sample.swing, sample.swing_target_xy)

        stance = sample.stance
        swing = sample.swing
        swing_xy = sample.swing_target_xy
        swing_progress = sample.swing_progress
        touchdown_search = 0.0
        if pending_foot is not None:
            pi = 0 if pending_foot == "left" else 1
            if measured[pi]:
                fsid = left_sid if pending_foot == "left" else right_sid
                actual_xy = data.site_xpos[fsid, :2].copy()
                if stab is not None:
                    old_off = stab.gait_offset.copy()
                    plan.commit_plant(pending_foot, actual_xy)
                    stab.on_touchdown(actual_xy, pending_nominal_xy)
                    # Translate the running CoM reference by the same offset delta
                    # so the discrete anchoring update is bump-free.
                    com_ref_xy = com_ref_xy + (stab.gait_offset - old_off)
                elif capture_gain is not None:
                    plan.commit_plant(pending_foot, actual_xy)
                else:
                    plan.commit_plant(pending_foot, pending_xy)
                pending_foot = pending_xy = pending_since = pending_nominal_xy = None
                last_swing = None
            else:
                swing = pending_foot
                swing_xy = pending_xy
                stance = ("right",) if pending_foot == "left" else ("left",)
                touchdown_search = min(0.04, 0.05 * (t - pending_since))
                swing_progress = 1.0

        com = robot_com(model, data)
        com_vel = com_velocity(model, data, realizer.root_body)
        rpy = roll_pitch_yaw_from_body(data, torso)
        _, _, hand_jac = hand_state(model, data, hand_sid)
        dcm = sample.dcm_xy; zmp = sample.zmp_xy
        if stab is not None:
            # Module A: anchor the body reference to the actual stepped feet.
            dcm = dcm + stab.gait_offset
            zmp = zmp + stab.gait_offset
            # Module B: one-step predicted-touchdown DCM foot placement.
            if swing is not None and pending_foot is None and len(stance) == 1:
                omega = plan.omega
                t_rem = max((1.0 - swing_progress)
                            * (sample.seg_duration - plan.double_support_time), 0.0)
                ssid = left_sid if stance[0] == "left" else right_sid
                p_st = data.site_xpos[ssid, :2].copy()
                swing_xy = stab.foot_placement(
                    measured_dcm=com[:2] + com_vel[:2] / omega,
                    desired_dcm=sample.dcm_xy,
                    measured_stance_xy=p_st, nominal_stance_xy=sample.zmp_xy,
                    nominal_next_foot_xy=sample.swing_target_xy,
                    remaining_time=t_rem, next_is_left=(swing == "left"))
                last_swing_xy = swing_xy.copy()
        elif capture_gain is not None:
            if len(stance) == 1:
                ssid = left_sid if stance[0] == "left" else right_sid
                foot_y = float(data.site_xpos[ssid, 1])
                zmp = zmp.copy(); zmp[1] = 0.5 * zmp[1] + 0.5 * foot_y
                dcm = dcm.copy(); dcm[1] = 0.5 * dcm[1] + 0.5 * foot_y
            if swing is not None and pending_foot is None:
                xi_err = (com[:2] + com_vel[:2] / plan.omega) - dcm
                kx, ky = capture_gain
                swing_xy = swing_xy + np.array([
                    float(np.clip(kx * xi_err[0], -0.08, 0.08)),
                    float(np.clip(ky * xi_err[1], -0.06, 0.06))])
                last_swing_xy = swing_xy.copy()
        dcm_dot = plan.omega * (dcm - zmp)
        com_ref_velocity = plan.omega * (dcm - com_ref_xy)
        com_ref_acceleration = plan.omega * (dcm_dot - com_ref_velocity)
        com_ref_xy = com_ref_xy + SIM_DT * com_ref_velocity
        com_reference = np.r_[com_ref_xy, com0[2]]
        com_reference_velocity = np.r_[com_ref_velocity, 0.0]
        com_reference_acceleration = np.r_[com_ref_acceleration, 0.0]
        error = np.r_[com - com_reference, rpy[:2]]
        error_velocity = np.r_[
            com_vel - com_reference_velocity, data.qvel[3:5]
        ]

        if k - last_wbc_k >= round(WBC_DT / SIM_DT):
            last_wbc_k = k
            log["wbc_sample"][k] = 1
            est = estimator.step(error_velocity, correction, realized_error_acceleration)
            measured_error_acceleration = est.measured_acceleration
            estimate_interaction = est.interaction
            estimate_realization = est.realization
            # Same-estimator realization-feedback ablation: the full controller
            # uses the combined residual (interaction + realization); the
            # no-feedback controller uses the interaction component alone.  One
            # estimator, one component removed, so the contrast isolates the
            # realization-feedback term rather than an estimator change.
            estimate_effective = est.effective
            conditioned_effective = _condition_residual(estimate_effective)

            if k - last_mpc_k >= round(mpc_solve_dt / SIM_DT):
                last_mpc_k = k
                log["mpc_sample"][k] = 1
                tm = time.perf_counter()
                state = np.r_[error, error_velocity]
                if controller == "impedance":
                    kp = np.array([16.72048113, 16.72048113, 20.0, 24.0, 24.0])
                    kd = np.array([17.17302876, 17.17302876, 12.0, 10.0, 10.0])
                    candidate = np.clip(
                        -kp * error - kd * error_velocity,
                        np.array([-6.0, -6.0, -4.0, -15.0, -15.0]),
                        np.array([6.0, 6.0, 4.0, 15.0, 15.0]),
                    )
                elif controller == "nominal_mpc":
                    candidate = mpc.solve(state, np.zeros(task_dim))
                elif controller == "interaction_mpc":
                    candidate = mpc.solve(state, conditioned_effective)
                else:
                    candidate = mpc.solve(state, _condition_residual(estimate_interaction))
                # Shared transparent command-slew constraint.  This is the
                # input-rate penalty's hard safety counterpart and prevents a
                # noisy touchdown sample from becoming an acceleration step.
                correction = correction + np.clip(
                    candidate - correction,
                    -COMMAND_SLEW_TASK_ACC_PER_UPDATE,
                    COMMAND_SLEW_TASK_ACC_PER_UPDATE,
                )
                mpc_ms.append(1000.0 * (time.perf_counter() - tm))

            # Established stance contacts are latched through individual
            # MuJoCo contact-sample flicker.  A *new* landing foot is activated
            # only by the pending-touchdown logic above, so a depression cannot
            # create fictitious support while a one-sample miss cannot delete
            # an existing stance.
            stance_contacts = realizer.contact_points(model, data, stance)
            if stance != stance_previous:
                valid = set(stance_contacts)
                for key, (pos, _) in stance_contacts.items():
                    if key not in stance_targets:
                        stance_targets[key] = pos.copy()
                for key in list(stance_targets):
                    if key not in valid:
                        del stance_targets[key]
                stance_previous = stance

            swing_task = None
            if swing is not None and swing_xy is not None:
                sid = left_sid if swing == "left" else right_sid
                if touchdown_search > 0.0:
                    z_des = ground_z_nominal - touchdown_search
                else:
                    z_des = ground_z_nominal + 0.05 * math.sin(math.pi * swing_progress)
                xy0 = sample.swing_start_xy if sample.swing_start_xy is not None else swing_xy
                xy_des = xy0 + (swing_xy - xy0) * (0.5 - 0.5 * math.cos(math.pi * swing_progress))
                swing_task = dict(
                    sid=sid, pos_des=np.r_[xy_des, z_des], vel_des=np.zeros(3),
                    kp=280.0, kd=32.0, weight=14.0,
                )

            tw = time.perf_counter()
            tau_hold, _, _ = realizer.command(
                model, data, q_nom, qd_nom, correction[:2], np.zeros(3), hand_jac,
                stance_contacts, stance_targets, base_height_reference, rpy,
                com_acc_des=com_reference_acceleration + correction[:3],
                swing_task=swing_task, attitude_weight=120.0,
                attitude_acc_correction=correction[3:5],
            )
            wbc_ms.append(1000.0 * (time.perf_counter() - tw))
            Jcom = np.zeros((3, model.nv))
            mujoco.mj_jacSubtreeCom(model, data, Jcom, realizer.root_body)
            if np.all(np.isfinite(realizer.last_qdd)):
                realized_com_acc = Jcom @ realizer.last_qdd + realizer.last_com_bias_acc
                realized_error_acceleration = np.r_[
                    realized_com_acc - com_reference_acceleration,
                    realizer.last_qdd[3:5],
                ]
                q_now, qd_now = realizer.joint_state(data)
                qdd_joint = realizer.last_qdd[realizer.dof]
                q_servo_reference = q_now + WBC_DT * qd_now + 0.5 * WBC_DT**2 * qdd_joint
                qd_servo_reference = qd_now + WBC_DT * qdd_joint

        # 1 kHz bounded joint servo around the latest 500 Hz QP torque.
        q_act, qd_act = realizer.joint_state(data)
        tau_servo = tau_hold + servo_kp * (q_servo_reference - q_act) \
            + servo_kd * (qd_servo_reference - qd_act)
        data.ctrl[realizer.ctrl_id] = np.clip(
            tau_servo, realizer.torque_min, realizer.torque_max
        )

        # Phase-locked torso wrench: applied to the plant only, never fed back
        # to the estimator or controller.  Onset is the first planned occurrence
        # of the target gait phase after the settle time.
        data.xfrc_applied[torso, :] = 0.0
        if push is not None:
            if push_onset is None:
                if push.start_time_s is not None and t >= push.start_time_s:
                    push_onset = t
                    push_onset_contact = (int(measured[0]), int(measured[1]))
                elif push.start_time_s is None and t >= push.earliest_onset_s:
                    # Gate on MEASURED foot contact, not the plan: single support
                    # means exactly one foot in measured contact, double support
                    # means both.  Require the measured phase to hold for dwell_s
                    # so the wrench lands inside the phase, not on a transition.
                    n_contact = int(measured[0]) + int(measured[1])
                    want = 2 if push.phase == "double_support" else 1
                    if n_contact == want:
                        if push_match_since is None:
                            push_match_since = t
                        if t - push_match_since >= push.dwell_s:
                            push_onset = t
                            push_onset_contact = (int(measured[0]), int(measured[1]))
                    else:
                        push_match_since = None
            if push_onset is not None:
                data.xfrc_applied[torso, push.axis()] = push.force_at(t - push_onset)
            log["applied_force"][k] = data.xfrc_applied[torso, :3]

        mujoco.mj_step(model, data); mujoco.mj_forward(model, data)
        if renderer is not None and k % v_stride == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
        contacts_now = measured_foot_contacts(model, data)
        obstacle_contact = _geom_contact_active(
            model, data, "terrain_right_obstacle"
        )
        contact_force = realizer.last_contact_force.reshape(-1, 3) if realizer.last_contact_force.size else np.zeros((0, 3))
        tau_lim = np.maximum(np.abs(realizer.torque_min), np.abs(realizer.torque_max))
        fall = bool(data.qpos[2] < 0.45 or np.max(np.abs(rpy[:2])) > 0.85)
        log["t"][k] = data.time; log["com"][k] = robot_com(model, data)
        log["com_ref"][k] = com_reference
        log["task_error"][k] = error
        log["task_velocity_error"][k] = error_velocity
        log["rpy"][k] = rpy; log["contact"][k] = contacts_now
        log["obstacle_contact"][k] = int(obstacle_contact)
        log["foot_position"][k, 0] = data.site_xpos[left_sid]
        log["foot_position"][k, 1] = data.site_xpos[right_sid]
        log["phase_index"][k] = sample.phase_index
        log["planned_stance"][k] = (
            int("left" in stance), int("right" in stance)
        )
        log["correction"][k] = correction
        log["command_acc"][k] = np.r_[
            com_reference_acceleration + correction[:3], correction[3:5]
        ]
        # Physical acceleration estimate from finite-differenced measured task
        # velocity.  Keep the QP-predicted acceleration separately: the 1-kHz
        # joint servo changes the applied torque after the QP solve, so J*qdd_QP
        # is not a plant measurement.
        log["realized_acc"][k] = np.r_[
            measured_error_acceleration[:3] + com_reference_acceleration,
            measured_error_acceleration[3:5],
        ]
        log["qp_predicted_acc"][k] = np.r_[
            realized_error_acceleration[:3] + com_reference_acceleration,
            realized_error_acceleration[3:5],
        ]
        log["interaction_estimate"][k] = estimate_interaction
        log["realization_estimate"][k] = estimate_realization
        log["effective_estimate"][k] = estimate_effective
        log["conditioned_effective_estimate"][k] = conditioned_effective
        # Requested-model residual seen by the controller (measured minus
        # commanded).  It includes interaction, realization, and model effects.
        log["realization_residual"][k] = measured_error_acceleration - correction
        log["contact_force_n"][k] = float(np.sum(contact_force[:, 2])) if len(contact_force) else 0.0
        log["friction_margin"][k] = friction_margin(model, data)
        log["torque_utilization"][k] = float(np.max(np.abs(data.ctrl[realizer.ctrl_id]) / np.maximum(tau_lim, 1e-9)))
        log["qp_fallback"][k] = int(realizer.last_fallback); log["fell"][k] = int(fall)
        if fall:
            for value in log.values():
                value[k + 1:] = value[k]
            break

    if renderer is not None:
        renderer.close()
        log["frames"] = frames
        log["video_fps"] = int(video.get("fps", 30))

    active = (log["t"] >= 1.0) & (log["fell"] == 0)
    if not np.any(active):
        active = log["t"] >= 0.0
    e = log["task_error"][active]; rr = log["realization_residual"][active]
    tail = active & (log["t"] >= max(1.0, float(log["t"][k]) - 1.0))
    e_y_mm = 1000.0 * e[:, 1]
    e_y_tail_mm = 1000.0 * log["task_error"][tail, 1]
    summary = {
        "controller": controller, "terrain": terrain, "seed": int(seed),
        "duration_requested_s": float(duration), "duration_completed_s": float(log["t"][k]),
        "fell": bool(np.any(log["fell"])), "qp_fallbacks": int(np.sum(log["qp_fallback"])),
        "forward_travel_m": float(log["com"][k, 0] - log["com"][0, 0]),
        "com_xyz_rms_mm": float(1000 * np.sqrt(np.mean(e[:, :3]**2))),
        "com_xyz_peak_mm": float(1000 * np.max(np.abs(e[:, :3]))),
        "lateral_error_rms_mm": float(np.sqrt(np.mean(e_y_mm**2))),
        "lateral_error_peak_mm": float(np.max(np.abs(e_y_mm))),
        "lateral_error_mean_mm": float(np.mean(e_y_mm)),
        "lateral_error_final_second_rms_mm": float(
            np.sqrt(np.mean(e_y_tail_mm**2))
        ),
        "lateral_error_final_second_mean_mm": float(np.mean(e_y_tail_mm)),
        "roll_pitch_rms_rad": float(np.sqrt(np.mean(e[:, 3:5]**2))),
        "max_abs_roll_pitch_rad": float(np.max(np.abs(log["rpy"][:k + 1, :2]))),
        "realization_residual_rms_mps2": float(np.sqrt(np.mean(rr**2))),
        "realization_residual_peak_mps2": float(np.max(np.abs(rr))),
        "peak_contact_force_n": float(np.max(log["contact_force_n"][:k + 1])),
        "contact_impulse_ns": float(np.sum(np.maximum(log["contact_force_n"][:k + 1], 0.0)) * SIM_DT),
        "max_torque_utilization": float(np.max(log["torque_utilization"][:k + 1])),
        "min_friction_margin": float(np.min(log["friction_margin"][:k + 1])),
        "obstacle_contacted": bool(np.any(log["obstacle_contact"][:k + 1])),
        "obstacle_contact_during_settling": bool(initial_obstacle_contact),
        "obstacle_first_contact_s": (
            float(log["t"][np.flatnonzero(log["obstacle_contact"][:k + 1])[0]])
            if np.any(log["obstacle_contact"][:k + 1]) else None
        ),
        "prediction": _prediction_rmse(log, SIM_DT),
        "timing": {"wbc": _timing(wbc_ms[20:], 2.0), "mpc": _timing(mpc_ms[5:], 10.0)},
    }
    if push is not None:
        impulse = float(np.sum(np.abs(log["applied_force"][:k + 1, push.axis()])) * SIM_DT)
        summary["push"] = {
            "direction": push.direction, "phase": push.phase,
            "magnitude_n": push.magnitude_n, "duration_s": push.duration_s,
            "profile": push.profile,
            "onset_s": push_onset, "onset_contact": push_onset_contact,
            "impulse_ns": impulse,
        }
    return log, summary


def aggregate(trials: list[dict]) -> dict:
    cells = {}
    for terrain in TERRAINS:
        for controller in CONTROLLERS:
            selected = [x for x in trials if x["terrain"] == terrain and x["controller"] == controller]
            if not selected:
                continue
            key = f"{terrain}/{controller}"
            cells[key] = {
                "n": len(selected), "falls": int(sum(x["fell"] for x in selected)),
                "qp_fallbacks": int(sum(x["qp_fallbacks"] for x in selected)),
            }
            for metric in ("com_xyz_rms_mm", "com_xyz_peak_mm",
                           "lateral_error_rms_mm", "lateral_error_peak_mm",
                           "lateral_error_mean_mm",
                           "lateral_error_final_second_rms_mm",
                           "lateral_error_final_second_mean_mm",
                           "roll_pitch_rms_rad",
                           "max_abs_roll_pitch_rad",
                           "realization_residual_rms_mps2", "peak_contact_force_n",
                           "forward_travel_m"):
                a = np.asarray([x[metric] for x in selected], float)
                cells[key][metric] = {"median": float(np.median(a)), "mean": float(np.mean(a)),
                                      "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0}
    return {"schema_version": 2, "trials": trials, "cells": cells}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=4200)
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--controllers", nargs="+", choices=ALL_CONTROLLERS,
                    default=list(CONTROLLERS))
    ap.add_argument("--terrains", nargs="+", choices=TERRAINS, default=list(TERRAINS))
    ap.add_argument("--no-logs", action="store_true")
    ap.add_argument("--artifact", default="uneven_ground_benchmark.json")
    ap.add_argument("--allow-invalid-obstacle", action="store_true",
                    help="diagnostic only: permit an obstacle trial with no later patch contact")
    args = ap.parse_args()
    trials = []
    for terrain in args.terrains:
        for i in range(args.trials):
            seed = args.seed + i
            for controller in args.controllers:
                print(f"RUN terrain={terrain} seed={seed} controller={controller}", flush=True)
                log, summary = run_trial(controller, terrain, seed, args.duration)
                if terrain == "obstacle" and not args.allow_invalid_obstacle:
                    if (summary["obstacle_contact_during_settling"]
                            or not summary["obstacle_contacted"]
                            or summary["obstacle_first_contact_s"] < 1.0):
                        raise RuntimeError(
                            "invalid publication obstacle trial: require no settling contact "
                            "and a later measured patch contact"
                        )
                trials.append(summary)
                if not args.no_logs:
                    np.savez_compressed(
                        RESULTS / f"uneven_{terrain}_{controller}_seed{seed}.npz", **log
                    )
                print(json.dumps(summary, sort_keys=True), flush=True)
    artifact = aggregate(trials)
    artifact["metadata"] = {
        "model": "Unitree G1 MuJoCo torque model", "mujoco": mujoco.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "servo_dt_s": SIM_DT, "wbc_dt_s": WBC_DT, "mpc_dt_s": MPC_DT,
        "duration_s": args.duration, "seed_start": args.seed,
        "gait": PUBLICATION_GAIT,
        "controllers": list(args.controllers),
        "residual_conditioning": {
            "deadband_task_acceleration": RESIDUAL_DEADBAND_TASK_ACC,
            "cap_task_acceleration": RESIDUAL_CAP_TASK_ACC,
            "command_slew_task_acceleration_per_update": COMMAND_SLEW_TASK_ACC_PER_UPDATE,
        },
        "terrain_definitions_m": {"depression": -0.020, "obstacle": 0.020,
                                  "rough_left": 0.015, "rough_right": -0.020},
        "obstacle_patch_x_m": [0.22, 0.34],
        "obstacle_validity": (
            "no contact during settling; later physical contact required for evidence"
        ),
    }
    out = RESULTS / args.artifact
    out.write_text(json.dumps(artifact, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
