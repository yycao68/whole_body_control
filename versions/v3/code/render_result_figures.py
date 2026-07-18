#!/usr/bin/env python3
"""Render the two result figures the paper references (Fig. 3 and Fig. 4).

These reuse the exact code paths of ``run_multirate_benchmarks.py`` (E5 mapping
fidelity and the E6 task-port continuation stress test), so the figures are
consistent with the committed ``multirate_benchmark.json`` without regenerating
it.  Two outputs:

  figures/e5_admissible_set.png       -- nominal-double-support residual-command
                                          classification: repeated-QP reference,
                                          single-cell map, continuation box.
  figures/e6_task_port_timeseries.png -- hand-tracking error over the 5 s
                                          observer-on window, with the [2,3] s
                                          transient comparison window marked.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mujoco
import numpy as np

from run_multirate_benchmarks import (
    TOL, TASK_WEIGHT, RateConfig, FIGURES,
    AnalyticAuthorityMapper, ContinuationAuthorityEstimator,
    ExactResidualBisectionEstimator, InverseDynamicsQPRealizer,
    MultirateInteractionController, NormalizedMPC, RandomWalkDisturbanceObserver,
    settle_model, scenario_context,
    robot_com, hand_state, com_velocity, roll_pitch_yaw_from_body,
    TORQUE_STAND_CTRL,
)

INK = "#171717"
C_REF = "#4c9f70"      # reference-feasible
C_INFEAS = "#d6d6d6"   # reference-infeasible
C_CELL = "#c05f28"     # single-cell map
C_CONT = "#2f6fb0"     # continuation
C_ORACLE = "#7a7a7a"   # oracle box


# ---------------------------------------------------------------------------
# Fig. 3 -- admissible residual-command set (nominal double support)
# ---------------------------------------------------------------------------

def render_admissible_set(n_grid: int = 61, span: float = 3.0) -> None:
    model, data, torso, hand_sid, hr = settle_model()
    exact = ExactResidualBisectionEstimator(realization_tolerance=TOL)
    mapper = AnalyticAuthorityMapper()
    sc = {"name": "double_support_nominal", "stance": ["left", "right"]}
    ctx = scenario_context(model, data, torso, hand_sid, hr, sc)
    R = ctx["realizer"]
    req = ctx["nominal_com_acc_des"].copy()

    def realize(u):
        rq = req.copy(); rq[:2] += u
        R.command(model, data, ctx["q_ref"], ctx["qd_ref"], rq[:2], ctx["task_acc_des"],
                  ctx["hand_jac"], ctx["stance_contacts"], ctx["stance_targets"],
                  ctx["base_height_ref"], ctx["rpy"], com_acc_des=rq,
                  attitude_weight=ctx["attitude_weight"],
                  centroidal_moment_des=ctx["centroidal_moment_des"])
        Jcom = np.zeros((3, R.nv))
        mujoco.mj_jacSubtreeCom(model, data, Jcom, R.root_body)
        return float(np.max(np.abs(Jcom[:2] @ R.last_qdd - rq[:2])))

    realize(np.zeros(2))
    snap = mapper.snapshot(R, model, data, timestamp=0.0,
                           contact_mode=("left", "right"))
    realize(np.zeros(2))
    cont = ContinuationAuthorityEstimator(realization_tolerance=TOL, max_regions=60)
    cbox = cont.estimate(R, model, data)
    box = exact.estimate(R, model, data,
                         **{k: v for k, v in ctx.items() if k != "realizer"})

    grid = np.linspace(-span, span, n_grid)
    ref_feasible = np.zeros((n_grid, n_grid), dtype=bool)
    cell_feasible = np.zeros((n_grid, n_grid), dtype=bool)
    for i, uy in enumerate(grid):
        for j, ux in enumerate(grid):
            u = np.array([ux, uy])
            ref_feasible[i, j] = realize(u) <= TOL
            cell_feasible[i, j] = bool(snap.contains(u))

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    extent = [-span, span, -span, span]
    # reference-feasible region as a filled field
    ax.imshow(np.where(ref_feasible, 1.0, np.nan), origin="lower", extent=extent,
              cmap=matplotlib.colors.ListedColormap([C_REF]), alpha=0.5,
              interpolation="nearest", zorder=1)
    ax.imshow(np.where(~ref_feasible, 1.0, np.nan), origin="lower", extent=extent,
              cmap=matplotlib.colors.ListedColormap([C_INFEAS]), alpha=0.6,
              interpolation="nearest", zorder=0)
    # single-cell map outline (the critical-region "diamond")
    ax.contour(grid, grid, cell_feasible.astype(float), levels=[0.5],
               colors=[C_CELL], linewidths=2.0, zorder=4)
    # continuation and oracle boxes
    ax.add_patch(Rectangle((cbox.lower[0], cbox.lower[1]),
                           cbox.upper[0] - cbox.lower[0], cbox.upper[1] - cbox.lower[1],
                           fill=False, edgecolor=C_CONT, lw=2.2, ls="-", zorder=5))
    ax.add_patch(Rectangle((box.lower[0], box.lower[1]),
                           box.upper[0] - box.lower[0], box.upper[1] - box.lower[1],
                           fill=False, edgecolor=C_ORACLE, lw=1.6, ls="--", zorder=3))
    ax.plot(0, 0, "k+", ms=9, mew=1.6, zorder=6)

    handles = [
        plt.Line2D([0], [0], marker="s", ls="", mfc=C_REF, mec="none", ms=11,
                   alpha=0.6, label="repeated-QP reference feasible"),
        plt.Line2D([0], [0], color=C_CELL, lw=2.0, label="single-cell map"),
        plt.Line2D([0], [0], color=C_CONT, lw=2.2, label="continuation box"),
        plt.Line2D([0], [0], color=C_ORACLE, lw=1.6, ls="--", label="oracle box"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=0.92)
    ax.set_xlabel(r"$u_x$  [m/s$^2$]"); ax.set_ylabel(r"$u_y$  [m/s$^2$]")
    ax.set_xlim(-span, span); ax.set_ylim(-span, span)
    ax.set_aspect("equal")
    ax.set_title("Nominal double-support admissible residual commands",
                 fontsize=9.5, color=INK)
    fig.tight_layout()
    out = FIGURES / "e5_admissible_set.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
# Fig. 4 -- task-port hand-tracking time series (observer on, continuation)
# ---------------------------------------------------------------------------

def render_task_port_timeseries(hand_force_n: float = 5.0, fallback_box: float = 10.0,
                                duration: float = 5.0) -> None:
    rates = RateConfig(servo_dt=0.001, node_dt=0.005, authority_dt=0.020)
    model, data, torso, hand_sid, hr = settle_model(task_weight=TASK_WEIGHT)
    realizer = InverseDynamicsQPRealizer(model, exact_realizer=True)
    realizer.task_weight = TASK_WEIGHT
    ctrl = MultirateInteractionController(
        realizer, rates=rates,
        body_mpc=NormalizedMPC(dim=2, dt=rates.body_dt, horizon=25,
                               q_pos=55.0, q_vel=12.0, r=0.08),
        task_mpc=NormalizedMPC(dim=3, dt=rates.task_dt, horizon=30,
                               q_pos=800.0, q_vel=40.0, r=0.05),
        body_obs=RandomWalkDisturbanceObserver(dim=2, dt=rates.body_dt,
                                               q_d=0.05, r_y=1.5e-4),
        task_obs=RandomWalkDisturbanceObserver(dim=3, dt=rates.task_dt,
                                               q_d=0.04, r_y=2.0e-4),
        mapper=AnalyticAuthorityMapper(), use_authority=True,
        task_continuation=True,
    )
    ctrl.task_fallback_box = np.array([fallback_box] * 3)
    R = ctrl.realizer
    stance = ("left", "right")
    com0 = robot_com(model, data)
    targets = {k: p.copy() for k, (p, _) in
               R.contact_points(model, data, stance).items()}
    hand0, _, _ = hand_state(model, data, hand_sid)
    hb = int(model.site_bodyid[hand_sid])
    qd_ref = np.zeros(model.nu)
    N = int(round(duration / rates.sim_dt))
    ts, hand_err, com_disp = [], [], []
    for k in range(N):
        t = k * rates.sim_dt
        com = robot_com(model, data)
        vel = com_velocity(model, data, R.root_body)
        rpy = roll_pitch_yaw_from_body(data, torso)
        hp, hv, hj = hand_state(model, data, hand_sid)
        data.xfrc_applied[:] = 0.0
        if t >= 1.0:
            data.xfrc_applied[hb, :3] = np.array([0.0, hand_force_n, 0.0])
        ctrl.step(
            model, data, t,
            q_ref=TORQUE_STAND_CTRL.copy(), qd_ref=qd_ref,
            com_ref_acc=np.zeros(3),
            body_error=np.r_[com[:2] - com0[:2], vel[:2]],
            stance=stance,
            stance_contacts=R.contact_points(model, data, stance),
            stance_targets=targets, base_height_ref=hr, rpy=rpy,
            hand_jac=hj, task_acc_ref=np.zeros(3),
            task_error=np.r_[hp - hand0, hv])
        mujoco.mj_step(model, data); mujoco.mj_forward(model, data)
        hp2, _, _ = hand_state(model, data, hand_sid)
        ts.append(t)
        hand_err.append(1000 * np.linalg.norm(hp2 - hand0))
        com_disp.append(1000 * np.linalg.norm(robot_com(model, data)[:2] - com0[:2]))

    ts = np.array(ts); hand_err = np.array(hand_err); com_disp = np.array(com_disp)
    win = (ts >= 2.0) & (ts < 3.0)
    win_mean = float(np.mean(hand_err[win]))

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.axvspan(2.0, 3.0, color="#c05f28", alpha=0.10, zorder=0)
    ax.axvline(1.0, color=C_ORACLE, lw=1.0, ls=":", zorder=1)
    ax.text(1.02, ax.get_ylim()[1], r"$5$ N force on", fontsize=7.5,
            color=C_ORACLE, va="top", ha="left")
    ax.plot(ts, hand_err, color=C_CONT, lw=1.6, label="hand error", zorder=3)
    ax.plot(ts, com_disp, color=C_REF, lw=1.2, alpha=0.9,
            label="CoM displacement", zorder=2)
    ax.hlines(win_mean, 2.0, 3.0, color="#c05f28", lw=2.0, zorder=4)
    ax.annotate(f"[2,3] s mean {win_mean:.1f} mm", (3.0, win_mean),
                textcoords="offset points", xytext=(6, 2), fontsize=8,
                color="#c05f28")
    ax.set_xlabel("t [s]"); ax.set_ylabel("error [mm]")
    ax.set_xlim(0, duration); ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title("Task-port continuation, observer on: hand tracking under a 5 N force",
                 fontsize=9.5, color=INK)
    fig.tight_layout()
    out = FIGURES / "e6_task_port_timeseries.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print("wrote", out, f"([2,3]s mean {win_mean:.1f} mm)")


if __name__ == "__main__":
    render_admissible_set()
    render_task_port_timeseries()
