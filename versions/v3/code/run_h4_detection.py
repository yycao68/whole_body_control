#!/usr/bin/env python3
"""H4: contact events can be detected without an oracle.

A contact event (a hand brace touching/leaving a surface, or a foot touchdown/
liftoff) creates an unmodeled wrench, so the disturbance observer's innovation
spikes (Sec. VIII, Eq. 25: eta = nu^T S^-1 nu). A change detector on the
normalized innovation declares an event when eta exceeds a calibrated threshold
for n_d consecutive samples, with a refractory window. The detector never reads
the scripted event schedule; the schedule is used only as the oracle to score
latency, false positives, and missed events.

Setup: standing torque-actuated G1, faithful centroidal-wrench recovery, with a
sequence of lateral brace-contact onsets/offsets. Detection runs on the body CoM
disturbance observer.

Usage: MPLCONFIGDIR=/private/tmp/mplconfig python3 run_h4_detection.py
"""
from __future__ import annotations

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
DURATION = 6.0
# scripted brace contact: (onset, offset) pairs -> events are all onsets+offsets
BRACE_INTERVALS = [(1.0, 2.0), (3.0, 3.8), (4.6, 5.4)]
BRACE_FORCE = np.array([0.0, 18.0, 0.0])   # brace contact wrench on the body [N]

N_SIGMA = 6.0        # detection threshold in innovation sigmas
N_D = 3              # consecutive samples above threshold
REFRACTORY = 0.15    # s, min spacing between declared events
CALIB = (0.3, 0.9)   # quiet window for threshold calibration
MATCH_WIN = 0.30     # s, oracle matching window after an event


def run():
    model = mujoco.MjModel.from_xml_path(str(generate_torque_model()))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for v, n in zip(TORQUE_STAND_CTRL, ACTUATED_JOINT_NAMES):
        data.qpos[model.jnt_qposadr[joint_id(model, n)]] = v
    mujoco.mj_forward(model, data)

    realizer = InverseDynamicsQPRealizer(model)
    body_mpc = NormalizedMPC(dim=2, dt=COMMAND_DT, horizon=35, q_pos=55.0, q_vel=12.0, r=0.08, u_max=np.array([3.5, 3.0]))
    body_obs = RandomWalkDisturbanceObserver(dim=2, dt=COMMAND_DT, q_d=0.05, r_y=1.5e-4)

    torso = body_id(model, "torso_link"); pelvis = body_id(model, "pelvis")
    hand_sid = site_id(model, "right_hand_site")
    com0 = robot_com(model, data); base_h = float(data.qpos[2])
    q_nom = TORQUE_STAND_CTRL.copy(); qd0 = np.zeros_like(q_nom)
    stance = ("left", "right")
    stance_contacts = realizer.contact_points(model, data, stance)
    stance_targets = {k: p.copy() for k, (p, _) in stance_contacts.items()}

    d_body = np.zeros(2); u_body = np.zeros(2); com_acc_des = np.zeros(3)
    N = int(DURATION / SIM_DT); period = max(1, int(COMMAND_DT / SIM_DT))
    t_log = np.zeros(N); nis_log = np.zeros(N); brace_log = np.zeros(N)

    # oracle event times (each onset and offset is a contact event)
    true_events = sorted([e for iv in BRACE_INTERVALS for e in iv])

    for k in range(N):
        t = k * SIM_DT
        com = robot_com(model, data); rpy = roll_pitch_yaw_from_body(data, torso)
        _, _, handj = hand_state(model, data, hand_sid)
        stance_contacts = realizer.contact_points(model, data, stance)

        braced = any(a <= t < b for (a, b) in BRACE_INTERVALS)
        F_brace = BRACE_FORCE if braced else np.zeros(3)

        if k % period == 0:
            y = com[:2] - com0[:2]
            x_body = np.r_[y, data.qvel[:2]]
            u_body = body_mpc.solve(x_body, d_body)
            # innovation-based NIS (prior innovation covariance)
            S = body_obs.C @ (body_obs.Aa @ body_obs.P @ body_obs.Aa.T + body_obs.Q) @ body_obs.C.T + body_obs.R
            d_body, innov = body_obs.step(y, u_body)
            nis = float(innov @ np.linalg.solve(S, innov))
            com_acc_des = np.array([u_body[0], u_body[1], 0.0])
        nis_log[k] = nis
        brace_log[k] = 1.0 if braced else 0.0

        data.xfrc_applied[:] = 0.0
        data.xfrc_applied[pelvis, :3] = F_brace
        realizer.command(model, data, q_nom, qd0, np.zeros(2), np.zeros(3), handj,
                          stance_contacts, stance_targets, base_h, rpy, com_acc_des=com_acc_des)
        mujoco.mj_step(model, data); mujoco.mj_forward(model, data)
        t_log[k] = t

    # ---- change detector on the normalized innovation (no oracle access) ----
    cm = (t_log >= CALIB[0]) & (t_log <= CALIB[1])
    mu, sd = float(np.mean(nis_log[cm])), float(np.std(nis_log[cm]) + 1e-9)
    thr = mu + N_SIGMA * sd
    detections = []
    run_cnt = 0; last_det = -1e9
    for k in range(N):
        if t_log[k] < CALIB[1]:
            continue
        run_cnt = run_cnt + 1 if nis_log[k] > thr else 0
        if run_cnt >= N_D and (t_log[k] - last_det) >= REFRACTORY:
            detections.append(float(t_log[k])); last_det = t_log[k]

    # ---- score against oracle ----
    matched = []; used = set()
    for te in true_events:
        cand = [d for d in detections if 0.0 <= d - te <= MATCH_WIN and d not in used]
        if cand:
            d = min(cand); used.add(d); matched.append((te, d, 1000 * (d - te)))
    lat = [m[2] for m in matched]
    fp = [d for d in detections if d not in used]
    missed = [te for te in true_events if te not in [m[0] for m in matched]]

    summary = dict(
        true_events=len(true_events), detected=len(detections), matched=len(matched),
        missed=len(missed), false_positives=len(fp),
        mean_latency_ms=round(float(np.mean(lat)), 1) if lat else None,
        max_latency_ms=round(float(np.max(lat)), 1) if lat else None,
        per_event_latency_ms=[round(x, 1) for x in lat],
        nis_threshold=round(thr, 2), calib_mean=round(mu, 3), calib_std=round(sd, 4),
        brace_force_N=BRACE_FORCE.tolist(),
    )
    return summary, dict(t=t_log, nis=nis_log, brace=brace_log, thr=thr,
                         true_events=true_events, detections=detections)


def main():
    summary, log = run()
    print(json.dumps(summary, indent=2))
    with (RESULTS / "h4_detection_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(log["t"], log["nis"], lw=0.8, label="normalized innovation (NIS)")
    ax.axhline(log["thr"], color="k", ls="--", lw=0.8, label="threshold")
    ax.plot(log["t"], log["brace"] * log["thr"], color="gray", alpha=0.3, label="brace contact (oracle)")
    for te in log["true_events"]:
        ax.axvline(te, color="green", ls=":", lw=0.8)
    for d in log["detections"]:
        ax.axvline(d, color="orange", ls="-", lw=1.0, alpha=0.7)
    ax.set_yscale("log"); ax.set_xlabel("t [s]"); ax.set_ylabel("NIS")
    ax.set_title("H4: contact-event detection (green=oracle event, orange=detected)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(RESULTS / "h4_detection.png", dpi=160)
    print("saved: results/h4_detection.png, results/h4_detection_summary.json")


if __name__ == "__main__":
    main()
