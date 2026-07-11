#!/usr/bin/env python3
"""Sustained walking: DCM tracking + capture-point step adaptation + hip strategy,
on the faithful centroidal-wrench recovery.

The CoP/DCM stabilizer (run_gait_dcm_stab.py) showed the binding limit is
single-support actuation authority: the ankle CoP saturates on the wide stance.
The two standard fixes, both added here:

  * Step adaptation (capture point): the next footstep is placed at the predicted
    end-of-step DCM minus the nominal DCM offset, u_next = xi_eos - b_nom, so the
    robot steps UNDER its falling CoM instead of trying to arrest it with the CoP.
    This is the dominant robustness mechanism (Khadiv / Englsberger).
  * Hip strategy: the torso-attitude objective in the realizer is relaxed so the
    QP can use centroidal angular momentum for balance beyond the ankle CoP.

LIPM/DCM:  xi = c + c_dot/omega,  xi_dot = omega (xi - p_zmp),
nominal DCM offsets  b_x = L/(e^{wT}-1),  b_y = 2w/(e^{wT}+1) (alternating side).

Usage:
  MPLCONFIGDIR=/private/tmp/mplconfig python3 run_gait_walk.py --step-len 0.06 --n-steps 14
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

from run_g1_root_assist_demo import ACTUATED_JOINT_NAMES, robot_com, roll_pitch_yaw_from_body
from run_g1_torque_realizer_benchmark import (
    InverseDynamicsQPRealizer, TORQUE_STAND_CTRL,
    body_id, generate_torque_model, hand_state, joint_id, site_id,
)
from run_gait_dcm_stab import com_velocity

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)
SIM_DT = 0.001
COMMAND_DT = 0.002
G = 9.81

COP_HALF = np.array([0.06, 0.025])       # stance-foot CoP half-extents
FOOT_HALF_Z = 0.0
W_NOM = 0.11                             # nominal half stance width (feet migrate to +-W_NOM)
MIN_SEP, MAX_SEP = 0.09, 0.32            # foot lateral separation limits (no crossing)
MAX_DX = 0.35                            # max forward step reach


def clamp_cop(p, stance_xy, stance):
    if len(stance) == 2:
        xs = [stance_xy[f][0] for f in stance]; ys = [stance_xy[f][1] for f in stance]
        lo = np.array([min(xs) - COP_HALF[0], min(ys) - COP_HALF[1]])
        hi = np.array([max(xs) + COP_HALF[0], max(ys) + COP_HALF[1]])
    else:
        c = stance_xy[stance[0]]; lo = c - COP_HALF; hi = c + COP_HALF
    return np.clip(p, lo, hi)


def run(step_len=0.06, n_steps=14, t_ss=0.40, t_ds=0.10, lift_h=0.05,
        attitude_weight=2.5, push=False, duration=None, seed=0):
    model = mujoco.MjModel.from_xml_path(str(generate_torque_model()))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for v, n in zip(TORQUE_STAND_CTRL, ACTUATED_JOINT_NAMES):
        data.qpos[model.jnt_qposadr[joint_id(model, n)]] = v
    mujoco.mj_forward(model, data)

    realizer = InverseDynamicsQPRealizer(model)
    torso = body_id(model, "torso_link"); pelvis = body_id(model, "pelvis")
    hand_sid = site_id(model, "right_hand_site")
    lsid = site_id(model, "left_foot"); rsid = site_id(model, "right_foot")
    foot_sid = {"left": lsid, "right": rsid}
    ground_z = float(min(data.site_xpos[lsid][2], data.site_xpos[rsid][2]))
    com0 = robot_com(model, data); z_c = float(com0[2] - ground_z)
    hand0, _, _ = hand_state(model, data, hand_sid)
    base_h = float(data.qpos[2]); q_nom = TORQUE_STAND_CTRL.copy(); qd0 = np.zeros_like(q_nom)
    omega = float(np.sqrt(G / z_c))
    eT = np.exp(omega * t_ss)
    b_x = step_len / (eT - 1.0)
    b_y_mag = 2.0 * W_NOM / (eT + 1.0)

    foot_pos = {"left": data.site_xpos[lsid][:2].copy(), "right": data.site_xpos[rsid][:2].copy()}
    # gait state machine
    T = t_ss + t_ds
    n_total = n_steps
    stance = ("left", "right")           # start in double support (settle)
    swing = None; step_i = -1; step_t0 = 0.0
    settle = 1.1
    next_target = None; lift_from = None

    task_acc_des = np.zeros(3); swing_cmd = None
    stance_prev = (); stance_targets = {}

    if duration is None:
        duration = settle + n_total * T + 0.8
    N = int(duration / SIM_DT); period = max(1, int(COMMAND_DT / SIM_DT))
    log = {k: np.zeros((N, d)) for k, d in
           [("t", 1), ("com", 2), ("xi", 2), ("pcmd", 2), ("footz", 2), ("rpy", 2),
            ("lfoot", 2), ("rfoot", 2)]}
    log["nstance"] = np.zeros(N); log["height"] = np.zeros(N); fell = False; switches = 0
    com_acc_des = np.zeros(3)

    for k in range(N):
        t = k * SIM_DT
        com = robot_com(model, data)
        rpy = roll_pitch_yaw_from_body(data, torso)
        hand, handv, handj = hand_state(model, data, hand_sid)
        cv = com_velocity(model, data, pelvis)[:2]
        xi = com[:2] + cv / omega
        stance_xy = {"left": data.site_xpos[lsid][:2].copy(), "right": data.site_xpos[rsid][:2].copy()}

        # ---- gait state machine ----
        if t < settle:
            stance = ("left", "right"); swing = None
        else:
            if step_i < 0 or (t - step_t0) >= T:
                # start a new step: previous swing (if any) has landed -> new stance
                if swing is not None:
                    foot_pos[swing] = next_target.copy()
                step_i += 1
                step_t0 = t
                if step_i >= n_total:
                    stance = ("left", "right"); swing = None
                else:
                    swing = "left" if (step_i % 2 == 0) else "right"   # which foot swings
                    stance = ("right",) if swing == "left" else ("left",)
                    lift_from = foot_pos[swing].copy()
            if swing is not None:
                tau = t - step_t0
                # double-support micro-phase at step start: both feet down
                if tau < t_ds:
                    stance_cur = ("left", "right")
                else:
                    stance_cur = ("right",) if swing == "left" else ("left",)
                stance = stance_cur

        # ---- contact-target bookkeeping ----
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

        if k % period == 0:
            if swing is not None and (t - step_t0) >= t_ds:
                # ---- capture-point step adaptation ----
                p_st = foot_pos[stance[0]] if len(stance) == 1 else 0.5 * (foot_pos["left"] + foot_pos["right"])
                tau = t - step_t0
                tau_rem = max(T - tau, 1e-3)
                xi_eos = p_st + (xi - p_st) * np.exp(omega * tau_rem)
                s = +1.0 if swing == "left" else -1.0
                b_next = np.array([b_x, -s * b_y_mag])
                u_next = xi_eos - b_next
                # kinematic clamp: correct side + separation + forward reach
                stance_y = foot_pos[stance[0]][1] if len(stance) == 1 else 0.5 * (foot_pos["left"][1] + foot_pos["right"][1])
                stance_x = foot_pos[stance[0]][0] if len(stance) == 1 else 0.5 * (foot_pos["left"][0] + foot_pos["right"][0])
                if s > 0:   # left foot -> left of stance
                    u_next[1] = np.clip(u_next[1], stance_y + MIN_SEP, stance_y + MAX_SEP)
                else:
                    u_next[1] = np.clip(u_next[1], stance_y - MAX_SEP, stance_y - MIN_SEP)
                u_next[0] = np.clip(u_next[0], stance_x - 0.15, stance_x + MAX_DX)
                # freeze near touchdown for a clean landing
                if tau_rem < 0.10:
                    u_next = next_target if next_target is not None else u_next
                next_target = u_next

                # swing trajectory toward the (adapting) target
                sig = np.clip((tau - t_ds) / max(T - t_ds, 1e-3), 0.0, 1.0)
                xy = lift_from + (u_next - lift_from) * (0.5 - 0.5 * np.cos(np.pi * sig))
                zt = ground_z + lift_h * np.sin(np.pi * sig)
                swing_cmd = dict(sid=foot_sid[swing], pos_des=np.array([xy[0], xy[1], zt]),
                                 vel_des=np.zeros(3), kp=300.0, kd=34.0, weight=16.0)
                p_cmd = clamp_cop(p_st, stance_xy, stance)
            elif t < settle:
                # Walking initiation: shift the CoM/DCM over the FIRST stance foot
                # (step 0 swings the left foot, so stance is the right foot) using
                # the DCM control law, so single support starts with the CoM over
                # the support instead of running away from it.
                swing_cmd = None
                first_stance = foot_pos["right"] * np.array([1.0, 0.85])   # slightly inside the foot
                a = 0.5 - 0.5 * np.cos(np.pi * np.clip(t / settle, 0.0, 1.0))
                xi_ref_s = com0[:2] + (first_stance - com0[:2]) * a
                p_cmd = xi_ref_s + (1.0 + 2.5 / omega) * (xi - xi_ref_s)
                p_cmd = clamp_cop(p_cmd, stance_xy, stance)
            else:
                swing_cmd = None
                p_st = 0.5 * (foot_pos["left"] + foot_pos["right"])
                p_cmd = clamp_cop(0.5 * (xi + p_st), stance_xy, stance)

            com_acc_xy = omega ** 2 * (com[:2] - p_cmd)
            com_acc_des = np.array([com_acc_xy[0], com_acc_xy[1], 0.0])
            # arm holds gently
            task_acc_des = -6.0 * (hand - hand0) - 2.0 * handv
            p_cmd_log = p_cmd

        data.xfrc_applied[:] = 0.0
        if push and (0.5 * duration) <= t < (0.5 * duration + 0.1):
            data.xfrc_applied[pelvis, :3] = np.array([0.0, 45.0, 0.0])

        realizer.command(model, data, q_nom, qd0, np.zeros(2), task_acc_des, handj,
                          stance_contacts, stance_targets, base_h, rpy,
                          com_acc_des=com_acc_des, swing_task=swing_cmd,
                          attitude_weight=attitude_weight)
        mujoco.mj_step(model, data); mujoco.mj_forward(model, data)

        log["t"][k] = float(data.time); log["com"][k] = robot_com(model, data)[:2]
        log["xi"][k] = xi; log["pcmd"][k] = p_cmd_log
        log["footz"][k] = [data.site_xpos[lsid][2], data.site_xpos[rsid][2]]
        log["lfoot"][k] = data.site_xpos[lsid][:2]; log["rfoot"][k] = data.site_xpos[rsid][:2]
        log["rpy"][k] = roll_pitch_yaw_from_body(data, torso)[:2]
        log["nstance"][k] = len(stance); log["height"][k] = data.qpos[2]
        if data.qpos[2] < 0.45 or np.max(np.abs(log["rpy"][k])) > 0.7:
            fell = True
            for arr in log.values():
                arr[k + 1:] = arr[k]
            break

    end = k
    planned = settle + n_total * T
    summary = dict(
        step_len=step_len, n_steps=n_steps, t_ss=t_ss, attitude_weight=attitude_weight, push=push,
        fell=fell, completed_s=float(log["t"][end, 0]), planned_s=float(planned),
        completed_full_plan=bool((not fell) and log["t"][end, 0] >= planned - 0.05),
        steps_taken=int(max(step_i, 0)), contact_switches=int(switches),
        forward_travel_m=float(log["com"][end, 0] - log["com"][0, 0]),
        left_max_lift_mm=float(1000 * (log["footz"][:end + 1, 0].max() - ground_z)) if end > 0 else 0.0,
        right_max_lift_mm=float(1000 * (log["footz"][:end + 1, 1].max() - ground_z)) if end > 0 else 0.0,
        max_roll_pitch_rad=float(np.max(np.abs(log["rpy"][:end + 1]))) if end > 0 else 0.0,
        min_pelvis_height_m=float(np.min(log["height"][:end + 1])) if end > 0 else 0.0,
        z_c=z_c, omega=omega,
    )
    return log, summary, ground_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-len", type=float, default=0.06)
    ap.add_argument("--n-steps", type=int, default=14)
    ap.add_argument("--t-ss", type=float, default=0.40)
    ap.add_argument("--attitude-weight", type=float, default=2.5)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    log, summ, gz = run(step_len=args.step_len, n_steps=args.n_steps, t_ss=args.t_ss,
                        attitude_weight=args.attitude_weight, push=args.push)
    print(json.dumps(summ, indent=2))

    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(log["t"], log["com"][:, 1], label="CoM y"); ax[0].plot(log["t"], log["xi"][:, 1], ":", label="DCM y", alpha=.6)
    ax[0].plot(log["t"], log["lfoot"][:, 1], lw=.6, label="L foot y"); ax[0].plot(log["t"], log["rfoot"][:, 1], lw=.6, label="R foot y")
    ax[0].set_ylabel("lateral y [m]"); ax[0].legend(fontsize=7, ncol=2); ax[0].grid(alpha=.3)
    ax[1].plot(log["t"], log["com"][:, 0], label="CoM x"); ax[1].set_ylabel("forward x [m]"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[2].plot(log["t"], 1000 * (log["footz"][:, 0] - gz), label="left foot")
    ax[2].plot(log["t"], 1000 * (log["footz"][:, 1] - gz), label="right foot")
    ax[2].set_ylabel("foot lift [mm]"); ax[2].set_xlabel("t [s]"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    fig.suptitle(f"Walk (step adaptation + hip strategy): steps={summ['steps_taken']} switches={summ['contact_switches']} "
                 f"travel={summ['forward_travel_m']:.2f}m full_plan={summ['completed_full_plan']} fell={summ['fell']}")
    fig.tight_layout(); fig.savefig(RESULTS / "gait_walk.png", dpi=150)
    with (RESULTS / "gait_walk_summary.json").open("w") as f:
        json.dump(summ, f, indent=2)
    print("saved: results/gait_walk.png, results/gait_walk_summary.json")


if __name__ == "__main__":
    main()
