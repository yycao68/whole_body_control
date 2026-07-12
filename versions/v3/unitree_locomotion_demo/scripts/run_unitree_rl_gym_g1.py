#!/usr/bin/env python3
"""Run Unitree's open-source G1 MuJoCo policy headlessly.

This repeats the Sim2Sim logic from Unitree's `unitree_rl_gym` repository:
load the G1 MuJoCo scene, load `deploy/pre_train/g1/motion.pt`, evaluate the
policy at 50 Hz, convert its actions to target joint positions, and apply
motor torques through the same PD law used by Unitree's deployment script.

The wrapper adds headless rendering, scripted external loads, and simple
high-level correction hooks so every demo video can be generated locally from
Unitree's open-source simulation code. It does not claim that our dual-MPC
controller is generating the gait.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
DEMO_ROOT = HERE.parent
V3_ROOT = DEMO_ROOT.parent
UNITREE_ROOT = V3_ROOT / "external_deps" / "unitree_rl_gym"
DEFAULT_CONFIG = UNITREE_ROOT / "deploy" / "deploy_mujoco" / "configs" / "g1.yaml"

SCENE_LABELS = {
    "baseline": "D0 Baseline: Unitree G1 walking",
    "push_off": "D1 Push Test: Unitree policy only",
    "push_on": "D2 Push Test: Interaction Dynamics MPC correction",
    "load_preview": "D3 Planned Load: preview correction",
}


def font(size: int):
    for p in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion
    return np.array(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def quat_to_yaw(quaternion: np.ndarray) -> float:
    qw, qx, qy, qz = quaternion
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    return (target_q - q) * kp + (target_dq - dq) * kd


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    for key in ("policy_path", "xml_path"):
        config[key] = config[key].replace("{LEGGED_GYM_ROOT_DIR}", str(UNITREE_ROOT))
    return config


def draw_overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    pad = 16
    line_h = 25
    box_h = pad * 2 + line_h * len(lines)
    draw.rectangle([0, 0, img.width, box_h], fill=(4, 8, 14, 205))
    for i, line in enumerate(lines):
        color = (255, 255, 255, 255) if i == 0 else (215, 226, 242, 255)
        draw.text((18, pad + i * line_h), line, font=font(17 if i else 19), fill=color)
    return np.asarray(img)


def add_reference_path(scene, x_center: float, y_ref: float, z: float = 0.018) -> None:
    """Draw the nominal walking line as a thin ground strip in the MuJoCo scene."""
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    size = np.array([12.0, 0.025, 0.006], dtype=np.float64)
    pos = np.array([x_center, y_ref, z], dtype=np.float64)
    mat = np.eye(3, dtype=np.float64).reshape(-1)
    rgba = np.array([0.0, 0.85, 1.0, 0.78], dtype=np.float32)
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_BOX,
        size,
        pos,
        mat,
        rgba,
    )
    scene.ngeom += 1


def run(args: argparse.Namespace) -> dict:
    if str(UNITREE_ROOT) not in sys.path:
        sys.path.insert(0, str(UNITREE_ROOT))
    config = load_config(args.config)
    config["simulation_duration"] = float(args.duration)
    if args.vx is not None:
        config["cmd_init"][0] = float(args.vx)
    if args.vy is not None:
        config["cmd_init"][1] = float(args.vy)
    if args.yaw_rate is not None:
        config["cmd_init"][2] = float(args.yaw_rate)
    scenario_defaults = {
        "baseline": {"push_y": 0.0, "layer": "off", "preview": False},
        "push_off": {"push_y": 90.0, "layer": "off", "preview": False},
        "push_on": {"push_y": 90.0, "layer": "on", "preview": False},
        "load_preview": {"push_y": 60.0, "layer": "on", "preview": True},
    }[args.scenario]
    if args.push_y is None:
        args.push_y = scenario_defaults["push_y"]
    if args.layer == "auto":
        args.layer = scenario_defaults["layer"]
    if args.preview == "auto":
        args.preview = "on" if scenario_defaults["preview"] else "off"

    simulation_dt = float(config["simulation_dt"])
    control_decimation = int(config["control_decimation"])
    duration = float(config["simulation_duration"])
    render_every = max(1, int(round(1.0 / (args.fps * simulation_dt))))
    total_steps = int(round(duration / simulation_dt))

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    cmd = np.array(config["cmd_init"], dtype=np.float32)
    nominal_cmd = cmd.copy()
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    model = mujoco.MjModel.from_xml_path(config["xml_path"])
    data = mujoco.MjData(model)
    model.opt.timestep = simulation_dt
    if args.video:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, int(args.width))
        model.vis.global_.offheight = max(model.vis.global_.offheight, int(args.height))
    policy = torch.jit.load(config["policy_path"])

    num_actions = int(config["num_actions"])
    num_obs = int(config["num_obs"])
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    writer = None
    renderer = None
    render_camera = None
    frames = 0
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        if not args.camera:
            render_camera = mujoco.MjvCamera()
            render_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            render_camera.distance = args.camera_distance
            render_camera.azimuth = args.camera_azimuth
            render_camera.elevation = args.camera_elevation
        writer = imageio.get_writer(str(args.video), fps=args.fps, codec="libx264", quality=8)

    q0 = data.qpos.copy()
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.push_body)
    if pelvis_id < 0:
        raise ValueError(f'Body "{args.push_body}" does not exist in the Unitree G1 XML.')
    y0 = float(data.qpos[1])
    min_base_z = float("inf")
    max_tilt = 0.0
    max_tau = 0.0
    max_abs_y_error = 0.0
    max_abs_vy_correction = 0.0
    push_impulse = 0.0
    fall_step = None
    log = {
        "time": [],
        "x": [],
        "y_error": [],
        "z": [],
        "vy": [],
        "vy_correction": [],
        "push_y": [],
        "tilt_xy": [],
        "yaw": [],
    }

    try:
        for step in range(total_steps):
            t = step * simulation_dt
            data.xfrc_applied[:] = 0.0
            push_active = args.push_start <= t <= args.push_start + args.push_duration
            external_fy = float(args.push_y) if push_active else 0.0
            if external_fy:
                data.xfrc_applied[pelvis_id, 1] = external_fy
                push_impulse += abs(external_fy) * simulation_dt

            y_error = float(data.qpos[1] - y0)
            vy = float(data.qvel[1])
            yaw = quat_to_yaw(data.qpos[3:7])
            cmd[:] = nominal_cmd
            if args.layer == "on":
                global_vy_feedback = -args.y_kp * y_error - args.y_kd * vy
                preview = 0.0
                if args.preview == "on":
                    preview_window = args.push_start - args.preview_lead <= t <= args.push_start + args.push_duration
                    if preview_window:
                        preview = -args.preview_gain * float(args.push_y)
                desired_global_vx = float(nominal_cmd[0])
                desired_global_vy = float(nominal_cmd[1]) + np.clip(
                    global_vy_feedback + preview,
                    -args.max_vy_correction,
                    args.max_vy_correction,
                )
                c = float(np.cos(yaw))
                s = float(np.sin(yaw))
                cmd[0] = c * desired_global_vx + s * desired_global_vy
                cmd[1] = -s * desired_global_vx + c * desired_global_vy
                cmd[2] = float(nominal_cmd[2]) - args.yaw_kp * yaw
                max_abs_vy_correction = max(max_abs_vy_correction, abs(float(cmd[1] - nominal_cmd[1])))
            vy_correction = float(cmd[1] - nominal_cmd[1])

            tau = pd_control(
                target_dof_pos,
                data.qpos[7:],
                kps,
                np.zeros_like(kds),
                data.qvel[6:],
                kds,
            )
            data.ctrl[:] = tau
            max_tau = max(max_tau, float(np.max(np.abs(tau))))
            mujoco.mj_step(model, data)

            if step % control_decimation == 0:
                qj = (data.qpos[7:] - default_angles) * float(config["dof_pos_scale"])
                dqj = data.qvel[6:] * float(config["dof_vel_scale"])
                gravity_orientation = get_gravity_orientation(data.qpos[3:7])
                omega = data.qvel[3:6] * float(config["ang_vel_scale"])

                period = 0.8
                count = step * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2.0 * np.pi * phase)
                cos_phase = np.cos(2.0 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array(
                    [sin_phase, cos_phase], dtype=np.float32
                )

                with torch.no_grad():
                    action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
                target_dof_pos = action * float(config["action_scale"]) + default_angles

            gravity = get_gravity_orientation(data.qpos[3:7])
            tilt = float(np.linalg.norm(gravity[:2]))
            log["time"].append(t)
            log["x"].append(float(data.qpos[0] - q0[0]))
            log["y_error"].append(float(data.qpos[1] - y0))
            log["z"].append(float(data.qpos[2]))
            log["vy"].append(float(data.qvel[1]))
            log["vy_correction"].append(vy_correction)
            log["push_y"].append(external_fy)
            log["tilt_xy"].append(tilt)
            log["yaw"].append(yaw)
            max_tilt = max(max_tilt, tilt)
            min_base_z = min(min_base_z, float(data.qpos[2]))
            max_abs_y_error = max(max_abs_y_error, abs(float(data.qpos[1] - y0)))
            if fall_step is None and (data.qpos[2] < args.fall_height or tilt > args.fall_tilt):
                fall_step = step

            if renderer is not None and writer is not None and step % render_every == 0:
                if args.camera:
                    renderer.update_scene(data, camera=args.camera)
                else:
                    render_camera.lookat[:] = np.array(
                        [
                            data.qpos[0] + args.camera_x_offset,
                            data.qpos[1] + args.camera_y_offset,
                            args.camera_z_lookat,
                        ]
                    )
                    renderer.update_scene(data, camera=render_camera)
                if args.reference_line:
                    add_reference_path(renderer.scene, float(data.qpos[0] + 2.5), y0)
                frame = renderer.render()
                scene_label = SCENE_LABELS.get(args.scenario, args.scenario)
                correction_label = "on" if args.layer == "on" else "off"
                preview_label = "on" if args.preview else "off"
                lines = [
                    scene_label,
                    f"Unitree RL Gym policy | vx={nominal_cmd[0]:.2f} m/s | push_y={args.push_y:.0f} N",
                    f"correction={correction_label}, preview={preview_label}, y_err={data.qpos[1]-y0:+.3f} m, vy_corr={cmd[1]-nominal_cmd[1]:+.3f} m/s",
                    "Walking comes from Unitree; this demo only changes high-level correction commands.",
                ]
                writer.append_data(draw_overlay(frame, lines))
                frames += 1
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    elapsed = total_steps * simulation_dt
    summary = {
        "source": "unitreerobotics/unitree_rl_gym",
        "config": str(args.config),
        "policy_path": config["policy_path"],
        "xml_path": config["xml_path"],
        "duration_s": elapsed,
        "simulation_dt": simulation_dt,
        "control_rate_hz": 1.0 / (simulation_dt * control_decimation),
        "command": {"vx": float(cmd[0]), "vy": float(cmd[1]), "yaw_rate": float(cmd[2])},
        "nominal_command": {
            "vx": float(nominal_cmd[0]),
            "vy": float(nominal_cmd[1]),
            "yaw_rate": float(nominal_cmd[2]),
        },
        "scenario": args.scenario,
        "interaction_layer": args.layer,
        "preview": args.preview,
        "push": {
            "body": args.push_body,
            "start_s": float(args.push_start),
            "duration_s": float(args.push_duration),
            "force_y_N": float(args.push_y),
            "absolute_impulse_Ns": push_impulse,
        },
        "base_displacement_m": {
            "x": float(data.qpos[0] - q0[0]),
            "y": float(data.qpos[1] - q0[1]),
            "z_final": float(data.qpos[2]),
        },
        "max_abs_lateral_error_m": max_abs_y_error,
        "max_abs_vy_correction_mps": max_abs_vy_correction,
        "min_base_z_m": min_base_z,
        "max_projected_gravity_xy": max_tilt,
        "max_abs_motor_torque": max_tau,
        "fall_detected": fall_step is not None,
        "fall_time_s": None if fall_step is None else float(fall_step * simulation_dt),
        "video": None if args.video is None else str(args.video),
        "log": None if args.log is None else str(args.log),
        "video_frames": frames,
        "claim_boundary": (
            "Unitree RL policy supplies locomotion. This run is demo-only and is "
            "not evidence that the v3 dual-MPC controller generates walking."
        ),
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.log, **{k: np.asarray(v, dtype=float) for k, v in log.items()})
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument(
        "--scenario",
        choices=("baseline", "push_off", "push_on", "load_preview"),
        default="baseline",
    )
    ap.add_argument("--vx", type=float, default=None)
    ap.add_argument("--vy", type=float, default=None)
    ap.add_argument("--yaw-rate", type=float, default=None)
    ap.add_argument("--video", type=Path, default=DEMO_ROOT / "results" / "unitree_base_only.mp4")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--summary", type=Path, default=DEMO_ROOT / "results" / "unitree_base_only_summary.json")
    ap.add_argument("--log", type=Path, default=DEMO_ROOT / "results" / "unitree_base_only_log.npz")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--camera", type=str, default="")
    ap.add_argument("--camera-distance", type=float, default=4.0)
    ap.add_argument("--camera-azimuth", type=float, default=135.0)
    ap.add_argument("--camera-elevation", type=float, default=-18.0)
    ap.add_argument("--camera-x-offset", type=float, default=0.25)
    ap.add_argument("--camera-y-offset", type=float, default=0.0)
    ap.add_argument("--camera-z-lookat", type=float, default=0.8)
    ap.add_argument("--fall-height", type=float, default=0.45)
    ap.add_argument("--fall-tilt", type=float, default=0.75)
    ap.add_argument("--push-body", type=str, default="pelvis")
    ap.add_argument("--push-start", type=float, default=3.0)
    ap.add_argument("--push-duration", type=float, default=0.35)
    ap.add_argument("--push-y", type=float, default=None)
    ap.add_argument("--layer", choices=("auto", "off", "on"), default="auto")
    ap.add_argument("--preview", choices=("auto", "off", "on"), default="auto")
    ap.add_argument("--preview-lead", type=float, default=0.35)
    ap.add_argument("--preview-gain", type=float, default=0.003)
    ap.add_argument("--y-kp", type=float, default=1.6)
    ap.add_argument("--y-kd", type=float, default=0.45)
    ap.add_argument("--yaw-kp", type=float, default=1.4)
    ap.add_argument("--max-vy-correction", type=float, default=0.45)
    ap.add_argument(
        "--no-reference-line",
        dest="reference_line",
        action="store_false",
        help="Disable the cyan ground reference line at the nominal walking path.",
    )
    ap.set_defaults(reference_line=True)
    args = ap.parse_args()
    if args.no_video:
        args.video = None
    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
