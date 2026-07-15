#!/usr/bin/env python3
"""Figure 1: the multirate interaction-dynamics architecture.

The organizing axis is RATE, and the three rates are:

  * 200 Hz OPTIMIZATION NODE -- one real-time thread that, sequentially, reads
    the state and model once, solves the body predictor, then the task predictor
    on the capacity the body left, then EXACTLY ONE whole-body QP, and finally a
    KKT sensitivity on that same factorization; it emits the optimized torque and
    the admissible set.  The dependency is explicit: u_b -> u_t -> tau.
  * 1 kHz SERVO -- holds the last optimized torque (zero-order hold) plus a joint
    PD term.  No optimization, no model update; the only loop that must truly
    close at 1 kHz, and it does.
  * ~50 Hz AUTHORITY REFRESH -- the PWA continuation walk (14c), off the node's
    critical path, publishing the admissible command set the predictors use.

The canonical pair (A,B) is invariant; contact mode and configuration move only
(H_k, h_k).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).with_name("multirate_architecture.png")

C_NODE, E_NODE = "#fdece2", "#c05f28"     # 200 Hz optimization node
C_SERVO, E_SERVO = "#eef0f2", "#5a5a5a"   # 1 kHz servo
C_AUTH, E_AUTH = "#e8f2e9", "#37743f"     # ~50 Hz authority refresh
C_ROBOT, E_ROBOT = "#ebebeb", "#565656"
INK = "#171717"


def box(ax, x, y, w, h, title, body, fc, ec, ts=9.5, bs=7.6, lead=0.036):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
        linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=3))
    ty = y + h - 0.028
    if title:
        ax.text(x + w / 2, ty, title, ha="center", va="top",
                fontsize=ts, fontweight="bold", color=INK, zorder=4)
        ty -= 0.044
    for line in body:
        ax.text(x + w / 2, ty, line, ha="center", va="top",
                fontsize=bs, color=INK, zorder=4)
        ty -= lead


def arrow(ax, p, q, color, style="-", lw=1.7, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=13, linewidth=lw,
        linestyle=style, color=color, connectionstyle=f"arc3,rad={rad}",
        zorder=2, shrinkA=2, shrinkB=2))


def note(ax, x, y, text, color, fs=7.4, style="italic"):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color,
            style=style, zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.95))


def main():
    fig, ax = plt.subplots(figsize=(14.2, 7.6))
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.0); ax.axis("off")

    # ---------------- lanes: authority (top) / node (mid) / servo (bot) ----
    ax.add_patch(Rectangle((0.015, 0.795), 0.97, 0.190, facecolor="#f3f9f4",
                           edgecolor="#bcd9c0", lw=1.0, zorder=0))
    ax.add_patch(Rectangle((0.015, 0.300), 0.97, 0.455, facecolor="#fdf7f3",
                           edgecolor="#ebcdb7", lw=1.0, zorder=0))
    ax.add_patch(Rectangle((0.015, 0.035), 0.97, 0.225, facecolor="#f6f7f8",
                           edgecolor="#cfd3d7", lw=1.0, zorder=0))

    ax.text(0.030, 0.968, "$\\approx$50 Hz  AUTHORITY REFRESH", fontsize=9.5,
            fontweight="bold", color=E_AUTH, va="top", zorder=6)
    ax.text(0.030, 0.940, "PWA continuation walk (14c), off the node's critical path",
            fontsize=8.0, color=E_AUTH, va="top", style="italic", zorder=6)

    ax.text(0.030, 0.742, "200 Hz  OPTIMIZATION NODE", fontsize=9.5,
            fontweight="bold", color=E_NODE, va="top", zorder=6)
    ax.text(0.030, 0.7, "one real-time thread, sequential\n— exactly ONE whole-body QP per update",
            fontsize=8.0, color=E_NODE, va="top", style="italic", zorder=6)

    ax.text(0.030, 0.248, "1 kHz  SERVO", fontsize=9.5,
            fontweight="bold", color=E_SERVO, va="top", zorder=6)
    ax.text(0.030, 0.221, "holds the last optimized torque (ZOH) + joint PD — no optimization",
            fontsize=8.0, color=E_SERVO, va="top", style="italic", zorder=6)

    # ---------------- authority -------------------------------------------
    box(ax, 0.400, 0.812, 0.34, 0.108, "",
        [r"Authority estimator  $\to$  admissible set  $H_k u \leq h_k$   (14e)",
         r"recovers the exact feasible set at no extra whole-body QP solve"],
        C_AUTH, E_AUTH, bs=8.0, lead=0.040)

    # ---------------- 200 Hz node: sequential pipeline --------------------
    yb, hb = 0.4, 0.200
    xs = [0.035, 0.223, 0.410, 0.598, 0.786]
    wb = 0.14
    box(ax, xs[0], yb, wb, hb, "State + model",
        [r"$q,\dot q$,  contact mode $\rho$",
         r"$M,h,J_c,J_t$",
         "nominal feedforward"], C_NODE, E_NODE, lead=0.040)
    box(ax, xs[1], yb, wb, hb, "Body predictor",
        [r"$x_{b}^+=A_b x_b+B_b(u_b+\hat d_b)$",
         r"$(A_b,B_b)$ constant",
         r"s.t.  $H_{b}u_b\leq h_{b}$"], C_NODE, E_NODE, lead=0.040)
    box(ax, xs[2], yb, wb, hb, "Task predictor",
        [r"$x_{t}^+=A_t x_t+B_t(u_t+\hat d_t)$",
         r"$(A_t,B_t)$ constant",
         r"s.t.  $H_{t}u_t\leq h_{t}-H_{tb}u_b^{\star}$"], C_NODE, E_NODE, lead=0.040)
    box(ax, xs[3], yb, wb, hb, "Whole-body QP 14b",
        [r"one solve $\to z_0$",
         r"$\tau_{\mathrm{ff}},\ \lambda_{\mathrm{ff}}$",
         "hard: torque, friction,",
         "unilateral, joint limits"], C_NODE, E_NODE, lead=0.036)
    box(ax, xs[4], yb, wb, hb, "KKT sensitivity 14c",
        [r"$K=\partial z^{\star}/\partial u$",
         r"$\tau=\tau_{\mathrm{ff}}+K_\tau u$",
         r"$\lambda=\lambda_{\mathrm{ff}}+K_\lambda u$",
         "same factorization"], C_NODE, E_NODE, lead=0.036)

    for i in range(4):
        arrow(ax, (xs[i] + wb, yb + hb / 2), (xs[i + 1], yb + hb / 2), E_NODE, lw=1.9)
    note(ax, 0.4500, 0.690, r"sequential:  $u_b(k)\to u_t(k)\to\tau(k)$", E_NODE, fs=10.0)

    # authority -> predictors ; node -> authority
    arrow(ax, (0.4, 0.86), (0.30, 0.86), E_AUTH, lw=1.6, style="--")
    arrow(ax, (0.3, 0.86), (0.30, 0.6), E_AUTH, lw=1.6, style="--")
    arrow(ax, (0.66, 0.812), (0.66, 0.6), E_AUTH, lw=1.6, style="--")
    note(ax, 0.630, 0.72, r"$H_k u\leq h_k$", E_AUTH, fs=8.2, style="normal")
    arrow(ax, (0.87, 0.60), (0.87, 0.86), E_AUTH, lw=1.6)
    arrow(ax, (0.87, 0.86), (0.74, 0.86), E_AUTH, lw=1.6)
    note(ax, 0.85, 0.72, r"$z_0,K$", E_AUTH, fs=8.0, style="normal")

    # ---------------- 1 kHz servo -----------------------------------------
    box(ax, 0.55, 0.055, 0.220, 0.130, "",
        [r"$\tau=\tau^{\star}+K_q(q_d-q)+D_q(\dot q_d-\dot q)$",
         "zero-order hold between node updates"],
        C_SERVO, E_SERVO, bs=8.0, lead=0.040)
    box(ax, 0.330, 0.055, 0.120, 0.130, "Robot / plant",
        [r"$q,\dot q$,  contacts", "measured state"], C_ROBOT, E_ROBOT, lead=0.040)

    # node -> servo: optimized torque
    arrow(ax, (0.684, 0.40), (0.6840, 0.185), E_NODE, lw=1.9)
    note(ax, 0.60, 0.320, r"optimized torque  $\tau^{\star}$  (200 Hz)", E_NODE, fs=8.0, style="normal")
    # servo -> robot
    arrow(ax, (0.55, 0.120), (0.45, 0.120), E_SERVO, lw=1.9)
    note(ax, 0.5, 0.152, r"$\tau$ (1 kHz)", E_SERVO, fs=7.8, style="normal")
    # state feedback: up the far-left margin, label pinned left so it clears the gap
    ax.plot([0.4, 0.4], [0.185, 0.282], color=E_ROBOT, lw=1.5, zorder=2)
    ax.plot([0.4, 0.078], [0.283, 0.283], color=E_ROBOT, lw=1.5, zorder=2)
    arrow(ax, (0.078, 0.283), (0.078, 0.40), E_ROBOT, lw=1.5)
    note(ax, 0.180, 0.283, r"state feedback", E_ROBOT, fs=7.4)

    # ---------------- invariance claim ------------------------------------
    ax.text(0.5, -0.028,
            r"Contact mode and configuration move only $(H_k,h_k)$ — never $(A,B)$.",
            ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")

    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
