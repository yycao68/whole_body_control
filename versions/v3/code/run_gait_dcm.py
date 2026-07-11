#!/usr/bin/env python3
"""DCM (capture-point) walking layered on the faithful centroidal recovery.

The marginal single-support balance of run_gait_faithful.py comes from holding
the CoM over one small foot. A divergent-component-of-motion (DCM) planner
instead produces a *dynamically feasible* CoM trajectory whose ZMP stays inside
the support foot: the CoM sways with the correct timing so the robot never has
to statically balance on a point. The normalized centroidal MPC (body port) then
tracks this reference with offset-free disturbance rejection, and the faithful
centroidal-wrench recovery (CoM-acceleration objective) realizes it on the feet.
The interaction-dynamics body port is unchanged; only the reference becomes
walk-feasible.

DCM:  xi = c + c_dot/omega,  omega = sqrt(g/z_c),  LIPM: c_ddot = omega^2 (c - p_zmp).
Backward recursion over footsteps gives xi(t); CoM follows c_dot = omega (xi - c).

Usage:
  MPLCONFIGDIR=/private/tmp/mplconfig python3 run_gait_dcm.py --step-len 0.06 --n-steps 8
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
from run_g1_root_assist_demo import ACTUATED_JOINT_NAMES, robot_com, roll_pitch_yaw_from_body
from run_g1_torque_realizer_benchmark import (
    InverseDynamicsQPRealizer, TORQUE_STAND_CTRL,
    body_id, generate_torque_model, hand_state, joint_id, site_id,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)
SIM_DT = 0.001
COMMAND_DT = 0.002
G = 9.81


class DCMWalk:
    """Footstep + DCM planner. Support feet alternate left/right, advancing x by
    step_len each step; a settle window is prepended and appended."""

    def __init__(self, left0, right0, step_len, n_steps, z_c, t_step, t_ds, t_settle,
                 zmp_y_scale=1.0):
        self.w = float(np.sqrt(G / z_c))
        self.T = t_step
        self.t_ds = t_ds
        self.t_settle = t_settle
        self.step_len = step_len
        # Pull the lateral ZMP toward the center: the CoM then sways less than the
        # full foot separation, so the single-support CoP stays inside the foot
        # (ankle authority) instead of saturating on a wide stance.
        w_l, w_r = left0[1] * zmp_y_scale, right0[1] * zmp_y_scale
        x0 = 0.5 * (left0[0] + right0[0])
        # support positions (ZMP targets) and stance side per step
        self.zmp = [np.array([x0, 0.0])]                 # initial centered settle
        self.side = ["both"]
        for k in range(n_steps):
            side = "left" if k % 2 == 0 else "right"
            x = x0 + (k // 1) * 0.0                        # placeholder; set below
            self.side.append(side)
        # advance x every step; feet stay on their y side
        x = x0
        for k in range(n_steps):
            side = "left" if k % 2 == 0 else "right"
            y = w_l if side == "left" else w_r
            x = x0 + k * step_len
            self.zmp.append(np.array([x, y]))
        self.zmp.append(self.zmp[-1].copy())              # final settle over last foot
        self.side.append("both")
        self.n = len(self.zmp)
        # step start times: settle, then n_steps of T, then settle
        self.tk = [0.0, self.t_settle]
        for k in range(1, self.n - 1):
            self.tk.append(self.tk[-1] + self.T)
        self.total = self.tk[-1] + self.t_settle
        # DCM backward recursion (per-segment duration)
        self.xi_ini = [None] * self.n
        self.xi_ini[-1] = self.zmp[-1].copy()
        for k in range(self.n - 2, -1, -1):
            dt = self.tk[k + 1] - self.tk[k]
            self.xi_ini[k] = self.zmp[k] + (self.xi_ini[k + 1] - self.zmp[k]) * np.exp(-self.w * dt)
        # foot plant positions over time: each foot's planted xy per step index
        self.left_plant = left0.copy()
        self.right_plant = right0.copy()

    def seg(self, t):
        for k in range(self.n - 1):
            if self.tk[k] <= t < self.tk[k + 1]:
                return k, t - self.tk[k], self.tk[k + 1] - self.tk[k]
        return self.n - 1, 0.0, 1.0

    def xi_and_zmp(self, t):
        k, tau, _ = self.seg(t)
        xi = self.zmp[k] + (self.xi_ini[k] - self.zmp[k]) * np.exp(self.w * tau)
        return xi, self.zmp[k]

    def schedule(self, t):
        """Return (stance tuple, swing foot or None, swing progress s in [0,1],
        swing_start xy, swing_target xy)."""
        k, tau, dur = self.seg(t)
        side = self.side[k]
        if side == "both":
            return ("left", "right"), None, 0.0, None, None
        # single support on `side`; the OTHER foot swings to its next plant
        swing = "right" if side == "left" else "left"
        # next plant of the swing foot = its next same-side support position
        nxt = None
        for j in range(k + 1, self.n):
            if self.side[j] == swing:
                nxt = self.zmp[j].copy(); break
        cur_plant = self.right_plant if swing == "right" else self.left_plant
        if nxt is None:
            nxt = cur_plant.copy()
        # brief double-support at the start of each single-support segment
        if tau < self.t_ds:
            return ("left", "right"), None, 0.0, None, None
        s = np.clip((tau - self.t_ds) / max(dur - self.t_ds, 1e-3), 0.0, 1.0)
        return (side,), swing, s, cur_plant.copy(), nxt

    def commit_plant(self, foot, xy):
        if foot == "left":
            self.left_plant = xy.copy()
        else:
            self.right_plant = xy.copy()


def run(step_len=0.06, n_steps=8, t_step=0.62, t_ds=0.12, lift_h=0.05,
        push=False, duration=None, seed=0, body_qpos=90.0, body_qvel=16.0):
    model = mujoco.MjModel.from_xml_path(str(generate_torque_model()))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for v, n in zip(TORQUE_STAND_CTRL, ACTUATED_JOINT_NAMES):
        data.qpos[model.jnt_qposadr[joint_id(model, n)]] = v
    mujoco.mj_forward(model, data)

    realizer = InverseDynamicsQPRealizer(model)
    body_mpc = NormalizedMPC(dim=2, dt=COMMAND_DT, horizon=35, q_pos=body_qpos, q_vel=body_qvel, r=0.05, u_max=np.array([6.0, 6.0]))
    body_obs = RandomWalkDisturbanceObserver(dim=2, dt=COMMAND_DT, q_d=0.1, r_y=1.5e-4)
    task_mpc = NormalizedMPC(dim=3, dt=COMMAND_DT, horizon=18, q_pos=120.0, q_vel=12.0, r=0.1, u_max=np.array([8.0, 8.0, 8.0]))
    task_obs = RandomWalkDisturbanceObserver(dim=3, dt=COMMAND_DT, q_d=0.3, r_y=1e-4)

    torso = body_id(model, "torso_link"); pelvis = body_id(model, "pelvis")
    hand_sid = site_id(model, "right_hand_site")
    lsid = site_id(model, "left_foot"); rsid = site_id(model, "right_foot")
    left0 = data.site_xpos[lsid][:2].copy(); right0 = data.site_xpos[rsid][:2].copy()
    ground_z = float(min(data.site_xpos[lsid][2], data.site_xpos[rsid][2]))
    com0 = robot_com(model, data); z_c = float(com0[2] - ground_z)
    hand0, _, _ = hand_state(model, data, hand_sid)
    base_h = float(data.qpos[2]); q_nom = TORQUE_STAND_CTRL.copy(); qd0 = np.zeros_like(q_nom)

    plan = DCMWalk(left0, right0, step_len, n_steps, z_c, t_step, t_ds, t_settle=0.8)
    if duration is None:
        duration = plan.total + 0.6

    d_body = np.zeros(2); u_body = np.zeros(2); d_task = np.zeros(3); u_task = np.zeros(3)
    com_ref = com0[:2].copy(); com_ref_vel = np.zeros(2)
    com_acc_des = np.zeros(3); task_acc_des = np.zeros(3); swing_task = None
    stance_prev = (); stance_targets = {}; swing_prev = None

    N = int(duration / SIM_DT); period = max(1, int(COMMAND_DT / SIM_DT))
    log = {k: np.zeros((N, d)) for k, d in
           [("t", 1), ("com", 2), ("com_ref", 2), ("xi", 2), ("zmp", 2),
            ("footz", 2), ("rpy", 2)]}
    log["nstance"] = np.zeros(N); log["height"] = np.zeros(N); fell = False; switches = 0

    for k in range(N):
        t = k * SIM_DT
        com = robot_com(model, data)
        rpy = roll_pitch_yaw_from_body(data, torso)
        hand, handv, handj = hand_state(model, data, hand_sid)
        stance, swing, s, sw_start, sw_target = plan.schedule(t)

        if swing_prev is not None and swing is None:
            # swing foot just landed -> commit its new plant
            plan.commit_plant(swing_prev, last_sw_target)
        swing_prev = swing
        if swing is not None:
            last_sw_target = sw_target

        if stance != stance_prev:
            if stance_prev != ():
                switches += 1
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

        xi, zmp = plan.xi_and_zmp(t)
        if k % period == 0:
            # integrate the DCM CoM reference (feasible LIPM trajectory)
            xi_dot = plan.w * (xi - zmp)
            com_ref_vel = plan.w * (xi - com_ref)
            com_ref_acc = plan.w * (xi_dot - com_ref_vel)
            com_ref = com_ref + com_ref_vel * COMMAND_DT
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
            if swing is not None:
                sid = rsid if swing == "right" else lsid
                xy = sw_start + (sw_target - sw_start) * (0.5 - 0.5 * np.cos(np.pi * s))
                z = ground_z + lift_h * np.sin(np.pi * s)
                swing_task = dict(sid=sid, pos_des=np.array([xy[0], xy[1], z]),
                                  vel_des=np.zeros(3), kp=280.0, kd=32.0, weight=14.0)

        data.xfrc_applied[:] = 0.0
        if push and (0.5 * duration) <= t < (0.5 * duration + 0.1):
            data.xfrc_applied[pelvis, :3] = np.array([0.0, 30.0, 0.0])

        realizer.command(model, data, q_nom, qd0, np.zeros(2), task_acc_des, handj,
                          stance_contacts, stance_targets, base_h, rpy,
                          com_acc_des=com_acc_des, swing_task=swing_task)
        mujoco.mj_step(model, data); mujoco.mj_forward(model, data)

        t_log = float(data.time)
        com_log = robot_com(model, data)
        rpy_log = roll_pitch_yaw_from_body(data, torso)
        log["t"][k] = t_log; log["com"][k] = com_log[:2]; log["com_ref"][k] = com_ref
        log["xi"][k] = xi; log["zmp"][k] = zmp
        log["footz"][k] = [data.site_xpos[lsid][2], data.site_xpos[rsid][2]]
        log["rpy"][k] = rpy_log[:2]; log["nstance"][k] = len(stance)
        log["height"][k] = data.qpos[2]
        if data.qpos[2] < 0.45 or np.max(np.abs(rpy_log[:2])) > 0.7:
            fell = True
            for arr in log.values():
                arr[k + 1:] = arr[k]
            break

    end = k
    ss = slice(int(0.8 / SIM_DT), end)
    com_rms = float(1000 * np.sqrt(np.mean(np.sum((log["com"][ss] - log["com_ref"][ss]) ** 2, axis=1)))) if end > int(0.8 / SIM_DT) else float("nan")
    summary = dict(
        step_len=step_len, n_steps=n_steps, t_step=t_step, push=push, fell=fell,
        completed_s=float(log["t"][end, 0]), planned_s=float(plan.total),
        contact_switches=int(switches),
        forward_travel_m=float(log["com"][end, 0] - log["com"][0, 0]),
        com_tracking_rms_mm=com_rms,
        left_max_lift_mm=float(1000 * (log["footz"][:end + 1, 0].max() - ground_z)) if end > 0 else 0.0,
        right_max_lift_mm=float(1000 * (log["footz"][:end + 1, 1].max() - ground_z)) if end > 0 else 0.0,
        max_roll_pitch_rad=float(np.max(np.abs(log["rpy"][:end + 1]))) if end > 0 else 0.0,
        min_pelvis_height_m=float(np.min(log["height"][:end + 1])) if end > 0 else 0.0,
        z_c=z_c, omega=plan.w,
    )
    return log, summary, ground_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-len", type=float, default=0.06)
    ap.add_argument("--n-steps", type=int, default=8)
    ap.add_argument("--t-step", type=float, default=0.62)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    log, summ, gz = run(step_len=args.step_len, n_steps=args.n_steps, t_step=args.t_step, push=args.push)
    print(json.dumps(summ, indent=2))

    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(log["t"], log["com"][:, 1], label="CoM y")
    ax[0].plot(log["t"], log["com_ref"][:, 1], "--", label="CoM y ref")
    ax[0].plot(log["t"], log["xi"][:, 1], ":", label="DCM y", alpha=.6)
    ax[0].set_ylabel("lateral y [m]"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(log["t"], log["com"][:, 0], label="CoM x")
    ax[1].plot(log["t"], log["com_ref"][:, 0], "--", label="CoM x ref")
    ax[1].set_ylabel("forward x [m]"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[2].plot(log["t"], 1000 * (log["footz"][:, 0] - gz), label="left foot")
    ax[2].plot(log["t"], 1000 * (log["footz"][:, 1] - gz), label="right foot")
    ax[2].set_ylabel("foot lift [mm]"); ax[2].set_xlabel("t [s]"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    fig.suptitle(f"DCM walk: switches={summ['contact_switches']} travel={summ['forward_travel_m']:.2f}m fell={summ['fell']} CoM-RMS={summ['com_tracking_rms_mm']:.0f}mm")
    fig.tight_layout(); fig.savefig(RESULTS / "gait_dcm.png", dpi=150)
    with (RESULTS / "gait_dcm_summary.json").open("w") as f:
        json.dump(summ, f, indent=2)
    print("saved: results/gait_dcm.png, results/gait_dcm_summary.json")


if __name__ == "__main__":
    main()
