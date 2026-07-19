#!/usr/bin/env python3
"""Sustained-push campaign on the root-assist body model (kinematic substrate).

The physics walker cannot sustain 15 s of walking (it falls at ~7 s), so the
sustained-push study runs on the 2-D body interaction model that drives the
root-assist walking visualization: the floating-base CoM tracks a 1.2 m/s
forward reference while a 1 s constant lateral force is applied.  This isolates
exactly the offset-free property: with the augmented disturbance observer
(Interaction MPC) a constant matched force is rejected without steady-state
error until the command authority saturates, whereas the nominal MPC
(d_hat = 0) holds a droop proportional to the force.

Four steps of the plan map here: 15 s flat walk, 1 s sustained push at
30/50/70 N, Nominal vs Interaction MPC, and ten paired seeds with recovery
metrics.  The platform (step up/down) study is a separate short physics
vignette.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from normalized_mpc import NormalizedMPC, RandomWalkDisturbanceObserver
from run_g1_root_assist_demo import SIM_DT, trajectory

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

MASS_KG = 34.04
DURATION = 15.0
DISTANCE = 1.2 * (DURATION - 1.0)      # 1 s ramp to 1.2 m/s cruise
PUSH_START = 5.0
PUSH_DUR = 1.0
FORCES_N = (30.0, 50.0, 70.0)
SEEDS = range(4200, 4210)
PROC_NOISE_SIGMA = 0.08                # per-seed lateral process noise [m/s^2]
RECOVERY_BAND_MM = 15.0
RECOVERY_DWELL_S = 0.20


def simulate(controller: str, force_n: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    body_mpc = NormalizedMPC(dim=2, dt=SIM_DT, horizon=40, q_pos=85.0, q_vel=18.0,
                             qf_pos=120.0, qf_vel=25.0, r=0.04,
                             u_max=np.array([3.0, 1.4]))
    observer = RandomWalkDisturbanceObserver(dim=2, dt=SIM_DT, q_d=0.06, r_y=8e-5)
    n = int(round(DURATION / SIM_DT))
    root_p, root_v, d_hat = np.zeros(2), np.zeros(2), np.zeros(2)
    accel = force_n / MASS_KG
    t = np.zeros(n)
    lat_err = np.zeros(n)
    for k in range(n):
        tk = k * SIM_DT
        ref = trajectory(tk, distance=DISTANCE, duration=DURATION)
        x_body = np.concatenate([root_p - ref.position, root_v - ref.velocity])
        d_ff = d_hat if controller == "interaction" else np.zeros(2)
        u = body_mpc.solve(x_body, d_hat=d_ff)
        push = np.zeros(2)
        if PUSH_START <= tk < PUSH_START + PUSH_DUR:
            push[1] = accel
        noise = np.array([0.0, rng.normal(0.0, PROC_NOISE_SIGMA)])
        root_acc = ref.acceleration + u + push + noise
        root_p = root_p + root_v * SIM_DT + 0.5 * root_acc * SIM_DT**2
        root_v = root_v + root_acc * SIM_DT
        ref_next = trajectory(min(tk + SIM_DT, DURATION), distance=DISTANCE, duration=DURATION)
        d_hat, _ = observer.step(root_p - ref_next.position, u)
        t[k] = tk
        lat_err[k] = root_p[1] - ref.position[1]

    lat_mm = 1000.0 * np.abs(lat_err)
    during = (t >= PUSH_START) & (t < PUSH_START + PUSH_DUR)
    steady_win = (t >= PUSH_START + 0.6) & (t < PUSH_START + PUSH_DUR)   # last 0.4 s
    peak_window = (t >= PUSH_START) & (t < PUSH_START + PUSH_DUR + 0.5)
    # recovery: after the push ends, time until |lat| stays under the band
    dwell = int(round(RECOVERY_DWELL_S / SIM_DT))
    recovery = None
    after = np.flatnonzero(t >= PUSH_START + PUSH_DUR)
    for k in after:
        if k + dwell >= n:
            break
        if np.all(lat_mm[k:k + dwell] < RECOVERY_BAND_MM):
            recovery = float(t[k] - (PUSH_START + PUSH_DUR))
            break
    return {
        "controller": controller, "force_n": force_n, "seed": int(seed),
        "peak_lateral_mm": float(np.max(lat_mm[peak_window])),
        "steady_offset_mm": float(np.mean(lat_mm[steady_win])),
        "mean_during_push_mm": float(np.mean(lat_mm[during])),
        "recovery_time_s": recovery,
        "recovered": recovery is not None,
    }


def _median(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def main() -> None:
    trials = []
    for controller in ("nominal", "interaction"):
        for force in FORCES_N:
            for seed in SEEDS:
                trials.append(simulate(controller, force, seed))
    cells = {}
    print(f"{'controller':12s} {'force':>6s} {'peak_mm':>8s} {'steady_mm':>10s} "
          f"{'recov_s':>8s} {'rec/10':>7s}")
    for controller in ("nominal", "interaction"):
        for force in FORCES_N:
            rows = [r for r in trials if r["controller"] == controller and r["force_n"] == force]
            cell = {
                "controller": controller, "force_n": force, "n": len(rows),
                "peak_lateral_mm": _median([r["peak_lateral_mm"] for r in rows]),
                "steady_offset_mm": _median([r["steady_offset_mm"] for r in rows]),
                "recovery_time_s": _median([r["recovery_time_s"] for r in rows]),
                "recovered_fraction": float(np.mean([r["recovered"] for r in rows])),
            }
            cells[f"{controller}|{int(force)}"] = cell
            rec = f"{cell['recovery_time_s']:.2f}" if cell["recovery_time_s"] is not None else "  -"
            print(f"{controller:12s} {force:6.0f} {cell['peak_lateral_mm']:8.1f} "
                  f"{cell['steady_offset_mm']:10.1f} {rec:>8s} "
                  f"{int(round(cell['recovered_fraction']*10)):>6d}/10")
    out = {
        "benchmark": "sustained_push_root_assist",
        "substrate": "kinematic root-assist 2-D body model",
        "mass_kg": MASS_KG, "duration_s": DURATION,
        "push_start_s": PUSH_START, "push_duration_s": PUSH_DUR,
        "forces_n": list(FORCES_N), "seeds": list(SEEDS),
        "process_noise_sigma_mps2": PROC_NOISE_SIGMA,
        "recovery_band_mm": RECOVERY_BAND_MM, "recovery_dwell_s": RECOVERY_DWELL_S,
        "body_mpc_u_max_lateral_mps2": 1.4,
        "cells": cells, "trials": trials,
    }
    path = RESULTS / "sustained_push_benchmark.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {path}  ({len(trials)} trials)")


if __name__ == "__main__":
    main()
