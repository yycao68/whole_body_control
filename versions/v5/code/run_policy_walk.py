"""Headless Unitree G1 pretrained-policy walk (official sim2sim interface).

Reuses the exact observation/action interface of unitree_rl_gym's
deploy_mujoco.py (47-dim obs, 12 leg actions, action->PD->torque, 50 Hz control
on a 500 Hz sim), but runs headless, records the full trajectory, and reports
stability so we can (a) confirm a stable >=20 s walk and (b) record a frozen
nominal reference for the Interaction-Dynamics stack.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import mujoco
import torch

HERE = Path(__file__).resolve().parent
SCENE = HERE / "g1_description" / "scene.xml"
POLICY = HERE / "motion.pt"

# --- config mirrored from configs/g1.yaml ---------------------------------
SIM_DT = 0.002
CONTROL_DECIMATION = 10          # 50 Hz control
KPS = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], float)
KDS = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], float)
DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                           -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], float)
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ACTION_SCALE = 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25], float)
NUM_ACTIONS = 12
NUM_OBS = 47
GAIT_PERIOD = 0.8
BASE_HEIGHT0 = 0.793


def gravity_orientation(quat):
    qw, qx, qy, qz = quat
    return np.array([
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz),
    ])


def pd_control(target_q, q, kp, dq, kd):
    return (target_q - q) * kp - dq * kd


def run(duration=20.0, cmd=(0.5, 0.0, 0.0), seed=0, settle=0.5):
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)

    # Standing init (no keyframe ships with the 12-DoF model).
    data.qpos[:3] = [0.0, 0.0, BASE_HEIGHT0]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:7 + NUM_ACTIONS] = DEFAULT_ANGLES
    rng = np.random.default_rng(seed)
    data.qvel[6:] += rng.normal(0.0, 2e-4, size=model.nv - 6)
    mujoco.mj_forward(model, data)

    policy = torch.jit.load(str(POLICY))
    cmd = np.array(cmd, float)

    lfoot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    rfoot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    total_mass = float(np.sum(model.body_mass[1:]))

    n = int(round(duration / SIM_DT))
    action = np.zeros(NUM_ACTIONS, np.float32)
    target_dof_pos = DEFAULT_ANGLES.copy()
    obs = np.zeros(NUM_OBS, np.float32)

    log = {k: np.zeros((n, d)) for k, d in {
        "base_pos": 3, "base_quat": 4, "base_linvel": 3, "base_angvel": 3,
        "com": 3, "qj": NUM_ACTIONS, "dqj": NUM_ACTIONS,
        "lfoot": 3, "rfoot": 3, "contact": 2, "action": NUM_ACTIONS,
    }.items()}
    log["t"] = np.zeros(n)
    fell_at = None

    counter = 0
    for k in range(n):
        t = k * SIM_DT
        tau = pd_control(target_dof_pos, data.qpos[7:], KPS, data.qvel[6:], KDS)
        # brief settle: hold default pose before releasing the policy
        if t < settle:
            tau = pd_control(DEFAULT_ANGLES, data.qpos[7:], KPS, data.qvel[6:], KDS)
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        counter += 1

        if t >= settle and counter % CONTROL_DECIMATION == 0:
            qj = (data.qpos[7:] - DEFAULT_ANGLES) * DOF_POS_SCALE
            dqj = data.qvel[6:] * DOF_VEL_SCALE
            grav = gravity_orientation(data.qpos[3:7])
            omega = data.qvel[3:6] * ANG_VEL_SCALE
            phase = (counter * SIM_DT) % GAIT_PERIOD / GAIT_PERIOD
            obs[:3] = omega
            obs[3:6] = grav
            obs[6:9] = cmd * CMD_SCALE
            obs[9:9 + NUM_ACTIONS] = qj
            obs[9 + NUM_ACTIONS:9 + 2 * NUM_ACTIONS] = dqj
            obs[9 + 2 * NUM_ACTIONS:9 + 3 * NUM_ACTIONS] = action
            obs[9 + 3 * NUM_ACTIONS:] = [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
            action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
            target_dof_pos = action * ACTION_SCALE + DEFAULT_ANGLES

        com = (model.body_mass[1:, None] * data.xipos[1:]).sum(0) / total_mass
        # foot contact: geom-level contact involving a foot body's geoms
        c = [0, 0]
        for ci in range(data.ncon):
            con = data.contact[ci]
            for gi in (con.geom1, con.geom2):
                b = model.geom_bodyid[gi]
                if b == lfoot or model.body_parentid[b] == lfoot:
                    c[0] = 1
                if b == rfoot or model.body_parentid[b] == rfoot:
                    c[1] = 1
        log["t"][k] = t
        log["base_pos"][k] = data.qpos[:3]
        log["base_quat"][k] = data.qpos[3:7]
        log["base_linvel"][k] = data.qvel[:3]
        log["base_angvel"][k] = data.qvel[3:6]
        log["com"][k] = com
        log["qj"][k] = data.qpos[7:]
        log["dqj"][k] = data.qvel[6:]
        log["lfoot"][k] = data.xpos[lfoot]
        log["rfoot"][k] = data.xpos[rfoot]
        log["contact"][k] = c
        log["action"][k] = action

        # Projected-gravity z is ~-1 when upright and rises toward 0 as the base
        # tilts; a fall is a low base or a tilt past ~60 deg (proj-grav z > -0.5).
        up = gravity_orientation(data.qpos[3:7])[2]
        if fell_at is None and t >= settle and (data.qpos[2] < 0.45 or up > -0.5):
            fell_at = t
    return log, fell_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--save", type=str, default="")
    args = ap.parse_args()

    for seed in args.seeds:
        log, fell = run(args.duration, cmd=(args.vx, 0.0, 0.0), seed=seed)
        survived = args.duration if fell is None else fell
        dist = float(log["base_pos"][-1, 0] - log["base_pos"][0, 0])
        lift = float(np.ptp(log["lfoot"][:, 2])), float(np.ptp(log["rfoot"][:, 2]))
        sw = int(np.sum(np.any(np.diff(log["contact"], axis=0) != 0, axis=1)))
        print(f"seed {seed}: fell={fell is not None} survived={survived:5.2f}/{args.duration:.0f}s "
              f"dist={dist:5.2f}m footlift=({lift[0]:.3f},{lift[1]:.3f})m switches={sw}", flush=True)
        if args.save and seed == args.seeds[0]:
            np.savez_compressed(args.save, **log)
            print("saved", args.save)


if __name__ == "__main__":
    main()
