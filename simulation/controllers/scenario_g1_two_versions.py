#!/usr/bin/env python3
"""Unitree G1 (MuJoCo): layered vs combined interaction control — full-body realization.

Reuses the G1 stance + arm-MPC infrastructure of scenario_c_g1. The right arm
performs a fast lateral reach, whose reaction perturbs the CoM laterally. Two
balance versions:

  LAYERED  : reactive hip-roll CoM regulator (corrects com[y] after it moves) —
             the standard three-layer stack; balance does not know the arm command.
  COMBINED : the same regulator PLUS a feedforward of the KNOWN arm force
             (the Γ_bt anticipation), pre-compensating the base for the reach.

Both use the same offset-free arm MPC. We log the lateral CoM excursion during
the reach. Run:  python3 scenario_g1_two_versions.py
"""
import sys
from pathlib import Path

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulation.controllers.scenario_c_g1 import (
    _make_robot, G1Ids, _get_hand_state, _get_robot_com, _arm_force_to_ctrl,
    _get_foot_site_ids, _settle, G1_CTRL_STAND, CTRL_DT,
    Q_MPC, R_MPC, F_MAX,
)
from simulation.controllers.wbc_core import (
    get_mass_matrix, get_contact_jacobian, get_contact_consistent_inverse,
    get_task_inertia, get_site_jacobian,
)
from simulation.controllers.impedance_mpc import ImpedanceMPC
from simulation.controllers.kalman import KalmanDisturbanceEstimator

N_RUN = 5000                      # 5 s
T_REACH = 2.0                     # arm reach start
DUR_REACH = 0.4                   # reach duration
REACH_Y = 0.15                    # lateral reach [m]
K_FF = 0.045                      # anticipation gain (arm lateral force -> hip roll)


def _reach(t):
    """Smooth min-jerk lateral arm-reference offset (y) and its rate."""
    if t < T_REACH:
        return 0.0
    if t > T_REACH + DUR_REACH:
        return REACH_Y
    s = (t - T_REACH) / DUR_REACH
    return REACH_Y * (10 * s**3 - 15 * s**4 + 6 * s**5)


def run(mode, k_ff=0.0):
    model, data = _make_robot()
    ids = G1Ids.get(model)
    foot_ids = _get_foot_site_ids(ids)
    _settle(model, data)
    p0, _ = _get_hand_state(model, data, ids)

    mpc = ImpedanceMPC(N=20, dt=CTRL_DT, Q=Q_MPC, R=R_MPC, F_max=F_MAX)
    mpc.precompute_mode('ds', 0.20 * np.eye(3))
    kalman = KalmanDisturbanceEstimator(dt=CTRL_DT)
    kalman.set_mode(mpc.A_d, mpc._mode_library['ds']['B_d'])

    F_prev = np.zeros(3)
    com_y = np.zeros(N_RUN)
    t_log = np.zeros(N_RUN)

    for step in range(N_RUN):
        t = step * CTRL_DT
        p_ref = p0 + np.array([0.0, _reach(t), 0.0])

        p_act, v_act = _get_hand_state(model, data, ids)
        e_pos = p_act - p_ref
        e_vel = v_act

        # offset-free arm MPC (contact-consistent Lambda_arm on the real G1)
        M_ = get_mass_matrix(model, data)
        Jc_ = get_contact_jacobian(model, data, foot_ids, [True, True])
        Mbar_ = get_contact_consistent_inverse(M_, Jc_)
        Jarm_ = get_site_jacobian(model, data, ids['hand_site'])
        La_use = get_task_inertia(Jarm_, Mbar_)
        mode_m = mpc.get_or_update_mode('ds', La_use)
        kalman.set_mode(mpc.A_d, mode_m['B_d'])
        kalman.predict(mpc.last_u)
        _, d_hat = kalman.update(e_pos)
        F_mpc = mpc.solve(np.concatenate([e_pos, e_vel]), La_use, 'ds', d_hat, use_osqp=False)
        F_arm = -F_mpc
        F_prev = F_mpc

        com, _ = _get_robot_com(model, data)
        ctrl = G1_CTRL_STAND.copy()
        # balance: reactive hip-roll CoM regulator (both versions)
        bal = 1.5 * com[1]
        if mode == "combined":
            # anticipation: pre-compensate the hips for the KNOWN arm lateral reaction
            bal -= k_ff * F_arm[1]
        ctrl[1] -= bal
        ctrl[7] -= bal
        ctrl[22:29] = _arm_force_to_ctrl(model, data, ids, F_arm)
        data.ctrl[:] = ctrl

        for _ in range(2):
            mujoco.mj_step(model, data)

        com_y[step] = com[1]
        t_log[step] = t

    return t_log, com_y


def _excursion(cy):
    reach = slice(int((T_REACH - 0.05) / CTRL_DT), int((T_REACH + DUR_REACH + 0.6) / CTRL_DT))
    c0 = np.mean(cy[:int(1.5 / CTRL_DT)])
    return np.max(np.abs(cy[reach] - c0)) * 1e3, c0


def main():
    tl, cy_lay = run("layered")
    peak_lay, c0_lay = _excursion(cy_lay)
    print(f"LAYERED  lateral CoM excursion = {peak_lay:.2f} mm")

    best = None
    for k in (0.001, 0.002, 0.004, 0.006, 0.010):
        _, cy = run("combined", k_ff=k)
        pk, c0 = _excursion(cy)
        print(f"COMBINED k_ff={k:.3f}  excursion = {pk:.2f} mm")
        if best is None or pk < best[1]:
            best = (k, pk, cy, c0)
    k_best, peak_com, cy_com, c0_com = best
    print(f"\nbest k_ff={k_best:.3f}:  LAYERED {peak_lay:.2f} -> COMBINED {peak_com:.2f} mm "
          f"({peak_lay / max(peak_com, 1e-6):.2f}x lower)")

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(tl, (cy_lay - c0_lay) * 1e3, color="#d62728", label="Layered (reactive balance)")
    ax.plot(tl, (cy_com - c0_com) * 1e3, color="#2ca02c",
            label=f"Combined (balance + arm-reaction FF, k={k_best:.3f})")
    ax.axvspan(T_REACH, T_REACH + DUR_REACH, alpha=0.08, color="k")
    ax.set_xlabel("time [s]"); ax.set_ylabel("lateral CoM excursion [mm]")
    ax.set_title("Unitree G1 — CoM under a fast lateral arm reach (shaded)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    out = "g1_two_versions.png"
    fig.savefig(out, dpi=150)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
