#!/usr/bin/env python3
"""Extend the faithful centroidal-wrench recovery to a stepping gait.

Fixed-support H2 works because the body-port residual acceleration is realized
as a centroidal wrench (CoM-acceleration objective), so e_ddot = u + d holds.
This script carries that recovery through CONTACT-MODE SWITCHES: a stepping gait
alternates double support (DS) and single support (SS), the normalized centroidal
MPC regulates the CoM to a support-consistent reference, the realizer allocates
the centroidal wrench to the *current* stance feet only, and the swing foot
tracks a lift-and-place Cartesian trajectory. The body Kalman observer keeps CoM
tracking offset-free across the switches, optionally under a mid-gait push.

Metrics: steps completed, contact switches, per-foot lift, CoM-tracking RMS,
fall status. Usage:
  MPLCONFIGDIR=/private/tmp/mplconfig python3 run_gait_faithful.py --step-len 0.0
  MPLCONFIGDIR=/private/tmp/mplconfig python3 run_gait_faithful.py --step-len 0.06
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from normalized_mpc import NormalizedMPC, RandomWalkDisturbanceObserver
from run_g1_root_assist_demo import ACTUATED_JOINT_NAMES, robot_com, roll_pitch_yaw_from_body, site_jac
from run_g1_torque_realizer_benchmark import (
    InverseDynamicsQPRealizer, TORQUE_STAND_CTRL,
    body_id, generate_torque_model, hand_state, joint_id, site_id,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)
SIM_DT = 0.001
COMMAND_DT = 0.002
G = np.array([0.0, 0.0, 9.81])

# Quasi-static stepping: the weight shift must settle in double support, and the
# single-support window is kept short because the realizer has no CoP/capture-point
# regulation, so single support is only marginally stable (a slow tip over ~3-4 s).
T_DS = 1.40       # double-support duration [s] (weight shift must settle)
T_SS = 0.30       # single-support duration [s] (kept short for stability)
LIFT_H = 0.06     # swing-foot apex height [m]


class Gait:
    """Phase clock over [DS_L, SS_L, DS_R, SS_R]. SS_L = stance left, swing right.
    Tracks each foot's planted xy so the swing target can step forward."""

    def __init__(self, left_xy, right_xy, step_len):
        self.plant = {"left": np.array(left_xy, float), "right": np.array(right_xy, float)}
        self.step_len = step_len
        self.T = [T_DS, T_SS, T_DS, T_SS]
        self.cycle = sum(self.T)
        self.swing_start = None
        self.swing_target = None
        self.prev_idx = -1
        self.steps = 0

    def _phase(self, t):
        tt = t % self.cycle
        acc = 0.0
        for i, dur in enumerate(self.T):
            if tt < acc + dur:
                return i, (tt - acc) / dur
            acc += dur
        return 3, 1.0

    def update(self, t):
        idx, s = self._phase(t)
        if idx != self.prev_idx:
            # entering a new phase
            if idx == 1:   # SS_L begins -> right foot swings
                self.swing_start = self.plant["right"].copy()
                self.swing_target = self.plant["right"] + np.array([self.step_len, 0.0])
            elif idx == 3: # SS_R begins -> left foot swings
                self.swing_start = self.plant["left"].copy()
                self.swing_target = self.plant["left"] + np.array([self.step_len, 0.0])
            elif idx in (0, 2) and self.prev_idx in (1, 3):
                # touchdown: commit the swing foot's new plant
                foot = "right" if self.prev_idx == 1 else "left"
                if self.swing_target is not None:
                    self.plant[foot] = self.swing_target.copy()
                self.steps += 1
            self.prev_idx = idx

        if idx == 0:      # DS, pre-shift toward left
            return dict(stance=("left", "right"), swing=None,
                        com_to=self._blend("center", "left", s))
        if idx == 1:      # SS left
            return dict(stance=("left",), swing="right", swing_s=s,
                        com_to=self.plant["left"].copy())
        if idx == 2:      # DS, pre-shift toward right
            return dict(stance=("left", "right"), swing=None,
                        com_to=self._blend("left", "right", s))
        return dict(stance=("right",), swing="left", swing_s=s,   # SS right
                    com_to=self.plant["right"].copy())

    def _center(self):
        return 0.5 * (self.plant["left"] + self.plant["right"])

    def _blend(self, a, b, s):
        pa = self._center() if a == "center" else self.plant[a]
        pb = self._center() if b == "center" else self.plant[b]
        s = 0.5 - 0.5 * np.cos(np.pi * np.clip(s, 0, 1))   # smoothstep
        return pa + (pb - pa) * s

    def swing_pos(self, swing_start, swing_target, s, ground_z):
        s = np.clip(s, 0, 1)
        xy = swing_start + (swing_target - swing_start) * (0.5 - 0.5 * np.cos(np.pi * s))
        z = ground_z + LIFT_H * np.sin(np.pi * s)
        return np.array([xy[0], xy[1], z])


def run(step_len=0.0, duration=6.0, push=False, seed=0):
    model = mujoco.MjModel.from_xml_path(str(generate_torque_model()))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for v, n in zip(TORQUE_STAND_CTRL, ACTUATED_JOINT_NAMES):
        data.qpos[model.jnt_qposadr[joint_id(model, n)]] = v
    mujoco.mj_forward(model, data)

    realizer = InverseDynamicsQPRealizer(model)
    body_mpc = NormalizedMPC(dim=2, dt=COMMAND_DT, horizon=35, q_pos=90.0, q_vel=16.0, r=0.05, u_max=np.array([6.0, 6.0]))
    body_obs = RandomWalkDisturbanceObserver(dim=2, dt=COMMAND_DT, q_d=0.1, r_y=1.5e-4)
    task_mpc = NormalizedMPC(dim=3, dt=COMMAND_DT, horizon=18, q_pos=120.0, q_vel=12.0, r=0.1, u_max=np.array([8.0, 8.0, 8.0]))
    task_obs = RandomWalkDisturbanceObserver(dim=3, dt=COMMAND_DT, q_d=0.3, r_y=1e-4)

    torso = body_id(model, "torso_link"); pelvis = body_id(model, "pelvis")
    hand_sid = site_id(model, "right_hand_site")
    lsid = site_id(model, "left_foot"); rsid = site_id(model, "right_foot")
    left0 = data.site_xpos[lsid][:2].copy(); right0 = data.site_xpos[rsid][:2].copy()
    ground_z = float(min(data.site_xpos[lsid][2], data.site_xpos[rsid][2]))
    gait = Gait(left0, right0, step_len)

    com0 = robot_com(model, data); hand0, _, _ = hand_state(model, data, hand_sid)
    base_h = float(data.qpos[2]); q_nom = TORQUE_STAND_CTRL.copy(); qd0 = np.zeros_like(q_nom)

    d_body = np.zeros(2); u_body = np.zeros(2); d_task = np.zeros(3); u_task = np.zeros(3)
    com_ref_prev = com0[:2].copy(); com_ref_vel = np.zeros(2)
    com_acc_des = np.zeros(3); task_acc_des = np.zeros(3); swing_task = None
    stance_prev = (); stance_targets = {}

    N = int(duration / SIM_DT); period = max(1, int(COMMAND_DT / SIM_DT))
    log = {k: np.zeros((N, d)) for k, d in
           [("t", 1), ("com", 2), ("com_ref", 2), ("footz", 2), ("rpy", 2), ("hand", 3)]}
    log["nstance"] = np.zeros(N); log["height"] = np.zeros(N); fell = False; switches = 0

    for k in range(N):
        t = k * SIM_DT
        com = robot_com(model, data); comv = (data.cvel[pelvis][3:6] if False else None)
        rpy = roll_pitch_yaw_from_body(data, torso)
        hand, handv, handj = hand_state(model, data, hand_sid)
        g = gait.update(t)
        stance = g["stance"]

        # stance target bookkeeping across contact-mode switches
        if stance != stance_prev:
            switches += 1 if stance_prev != () else 0
            cur = realizer.contact_points(model, data, stance)
            for key, (pos, _) in cur.items():
                foot = key.split("_", 1)[0]
                if foot not in stance_prev or key not in stance_targets:
                    stance_targets[key] = pos.copy()
            for key in list(stance_targets):
                if key not in cur:
                    del stance_targets[key]
            stance_prev = stance
        stance_contacts = realizer.contact_points(model, data, stance)

        if k % period == 0:
            com_ref = g["com_to"]
            com_ref_vel = (com_ref - com_ref_prev) / COMMAND_DT
            com_ref_acc = np.zeros(2)  # slow, smooth reference
            com_ref_prev = com_ref.copy()
            # pelvis planar velocity (not full-body CoM velocity, which spikes
            # with the swing leg and would inject a disturbance into balance)
            com_v = data.qvel[:2].copy()
            x_body = np.r_[com[:2] - com_ref, com_v - com_ref_vel]
            u_body = body_mpc.solve(x_body, d_body)
            d_body, _ = body_obs.step(com[:2] - com_ref, u_body)
            com_acc_des = np.array([com_ref_acc[0] + u_body[0], com_ref_acc[1] + u_body[1], 0.0])

            x_task = np.r_[hand - hand0, handv]
            u_task = task_mpc.solve(x_task, d_task)
            d_task, _ = task_obs.step(hand - hand0, u_task)
            task_acc_des = u_task

            swing_task = None
            if g["swing"] is not None:
                sid = rsid if g["swing"] == "right" else lsid
                pos_des = gait.swing_pos(gait.swing_start, gait.swing_target, g["swing_s"], ground_z)
                swing_task = dict(sid=sid, pos_des=pos_des, vel_des=np.zeros(3),
                                  kp=280.0, kd=32.0, weight=14.0)

        data.xfrc_applied[:] = 0.0
        if push and 2.5 <= t < 2.6:
            data.xfrc_applied[pelvis, :3] = np.array([0.0, 35.0, 0.0])

        realizer.command(model, data, q_nom, qd0, np.zeros(2), task_acc_des, handj,
                          stance_contacts, stance_targets, base_h, rpy,
                          com_acc_des=com_acc_des, swing_task=swing_task)
        mujoco.mj_step(model, data); mujoco.mj_forward(model, data)

        t_log = float(data.time)
        com_log = robot_com(model, data)
        rpy_log = roll_pitch_yaw_from_body(data, torso)
        hand_log, _, _ = hand_state(model, data, hand_sid)
        log["t"][k] = t_log; log["com"][k] = com_log[:2]; log["com_ref"][k] = g["com_to"]
        log["footz"][k] = [data.site_xpos[lsid][2], data.site_xpos[rsid][2]]
        log["rpy"][k] = rpy_log[:2]; log["hand"][k] = hand_log; log["nstance"][k] = len(stance)
        log["height"][k] = data.qpos[2]
        if data.qpos[2] < 0.45 or np.max(np.abs(rpy_log[:2])) > 0.7:
            fell = True
            for key, arr in log.items():
                arr[k + 1:] = arr[k]
            break

    end = k if fell else N - 1
    ss = slice(int(1.0 / SIM_DT), end)   # skip initial settling
    com_rms = float(1000 * np.sqrt(np.mean(np.sum((log["com"][ss] - log["com_ref"][ss]) ** 2, axis=1))))
    summary = dict(
        step_len=step_len, duration=duration, push=push, fell=fell,
        completed_s=float(log["t"][end, 0]), steps=int(gait.steps),
        contact_switches=int(switches),
        left_max_lift_mm=float(1000 * (log["footz"][:end + 1, 0].max() - ground_z)),
        right_max_lift_mm=float(1000 * (log["footz"][:end + 1, 1].max() - ground_z)),
        com_tracking_rms_mm=com_rms,
        max_roll_pitch_rad=float(np.max(np.abs(log["rpy"][:end + 1]))),
        min_pelvis_height_m=float(np.min(log["height"][:end + 1])),
        forward_travel_m=float(log["com"][end, 0] - log["com"][0, 0]),
    )
    return log, summary, ground_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-len", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    log, summ, gz = run(step_len=args.step_len, duration=args.duration, push=args.push)
    print(json.dumps(summ, indent=2))

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(log["t"], log["com"][:, 1], label="CoM y")
    ax[0].plot(log["t"], log["com_ref"][:, 1], "--", label="CoM y ref")
    ax[0].set_ylabel("lateral y [m]"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(log["t"], 1000 * (log["footz"][:, 0] - gz), label="left foot")
    ax[1].plot(log["t"], 1000 * (log["footz"][:, 1] - gz), label="right foot")
    ax[1].set_ylabel("foot lift [mm]"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].step(log["t"], log["nstance"], where="post"); ax[2].set_ylabel("# stance feet")
    ax[2].set_xlabel("t [s]"); ax[2].grid(alpha=.3)
    fig.suptitle(f"Faithful-recovery gait: steps={summ['steps']} fell={summ['fell']} CoM-RMS={summ['com_tracking_rms_mm']:.1f}mm")
    fig.tight_layout(); fig.savefig(RESULTS / "gait_faithful.png", dpi=150)
    with (RESULTS / "gait_faithful_summary.json").open("w") as f:
        json.dump(summ, f, indent=2)
    print("saved: results/gait_faithful.png, results/gait_faithful_summary.json")


if __name__ == "__main__":
    main()
