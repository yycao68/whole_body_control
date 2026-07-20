#!/usr/bin/env python3
"""Figure 1: the multirate interaction-dynamics architecture.

Four rates, slowest at the top:

  * 50--100 Hz EXTERNAL MOTION PLANNER -- publishes the nominal body, swing-foot,
    and contact-schedule reference; it is never modified by terrain or push.
  * 100 Hz INTERACTION LAYER -- a low-pass measured-acceleration-residual
    estimator produces the effective residual d_eff and its conditioned horizon
    sequence, and the interaction MPC picks a task-acceleration correction a_e
    on the fixed double integrator (A_d,B_d).
  * 500 Hz WHOLE-BODY REALIZER -- an inverse-dynamics/contact QP realizes the
    requested task acceleration under hard multibody, contact, and torque limits,
    and reports the realized-minus-commanded acceleration back to the estimator.
  * 1 kHz TORQUE HOLD / PLANT -- zero-order-hold torque and MuJoCo integration.

The predictor pair (A_d,B_d) is invariant across gait phase, terrain, and push;
robot and environment dependence live only in d_eff, the task constraints, and
the realizer.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).with_name("multirate_architecture.png")

C_PLAN, E_PLAN = "#e9f0f8", "#3b6ea5"     # 50-100 Hz planner
C_MPC, E_MPC = "#fdece2", "#c05f28"       # 100 Hz interaction layer
C_WBC, E_WBC = "#e8f2e9", "#37743f"       # 500 Hz realizer
C_SERVO, E_SERVO = "#eef0f2", "#5a5a5a"   # 1 kHz torque hold
C_ROBOT, E_ROBOT = "#ebebeb", "#565656"
INK = "#171717"


def box(ax, x, y, w, h, title, body, fc, ec, ts=10.0, bs=8.2, lead=0.040):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
        linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=3))
    ty = y + h - 0.010
    if title:
        ax.text(x + w / 2, ty, title, ha="center", va="top",
                fontsize=ts, fontweight="bold", color=INK, zorder=4)
        ty -= 0.040
    for line in body:
        ax.text(x + w / 2, ty, line, ha="center", va="top",
                fontsize=bs, color=INK, zorder=4)
        ty -= lead


def arrow(ax, p, q, color, style="-", lw=1.9, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=14, linewidth=lw,
        linestyle=style, color=color, connectionstyle=f"arc3,rad={rad}",
        zorder=2, shrinkA=2, shrinkB=2))


def poly(ax, pts, color, lw=1.7):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, lw=lw, zorder=2, solid_capstyle="round")


def note(ax, x, y, text, color, fs=8.0, style="italic"):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color,
            style=style, zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.95))


def lane(ax, y0, h, fc, ec, label, sub, ecol):
    ax.add_patch(Rectangle((0.015, y0), 0.97, h, facecolor=fc, edgecolor=ec,
                           lw=1.0, zorder=0))
    ax.text(0.078, y0 + h - 0.018, label, fontsize=10.0, fontweight="bold",
            color=ecol, va="top", zorder=6)
    ax.text(0.078, y0 + h - 0.052, sub, fontsize=8.2, color=ecol, va="top",
            style="italic", zorder=6)


def main():
    fig, ax = plt.subplots(figsize=(12.5, 8.4))
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.0); ax.axis("off")

    lane(ax, 0.830, 0.155, "#eef4fb", "#c3d6ea",
         "50-100 Hz EXTERNAL MOTION PLANNER",
         "nominal reference; not modified by terrain or push", E_PLAN)
    lane(ax, 0.545, 0.260, "#fdf7f3", "#ebcdb7",
         "100 Hz INTERACTION LAYER",
         "residual estimation +\ninteraction MPC on\nfixed model", E_MPC)
    lane(ax, 0.300, 0.220, "#f3f9f4", "#bcd9c0",
         "500 Hz WHOLE-BODY REALIZER",
         "instantaneous inverse-dynamics /\ncontact QP", E_WBC)
    lane(ax, 0.035, 0.230, "#f6f7f8", "#cfd3d7",
         "1 kHz TORQUE HOLD / PLANT",
         "zero-order-hold torque\n& integration", E_SERVO)

    # ---- planner -------------------------------------------------------
    box(ax, 0.465, 0.848, 0.21, 0.11, "External motion planner",
        ["nominal body, swing-foot,\nand contact-schedule reference"],
        C_PLAN, E_PLAN, lead=0.036)

    # ---- interaction layer: estimator + MPC ----------------------------
    box(ax, 0.20, 0.56, 0.2, 0.17, "Interaction estimator",
        [r"low-pass measured residual",
         r"$\hat d_{\rm eff}=d_{\rm int}+d_{\rm real}+d_{\rm mod}$",
         r"conditioned horizon $\tilde d_{k+i|k}=\tilde d_k$"], C_MPC, E_MPC, lead=0.044)
    box(ax, 0.465, 0.56, 0.23, 0.17, "Interaction-Dynamics MPC",
        [r"fixed $(A_d,B_d)$ double integrator",
         r"$\min\ \sum\|x_j\|_Q^2+\|a_{e,j}+\tilde d_k\|_R^2$",
         r"bounded correction $a_e$"], C_MPC, E_MPC, lead=0.044)

    # ---- realizer ------------------------------------------------------
    box(ax, 0.3, 0.310, 0.36, 0.150,
        "Whole-body inverse-dynamics / contact QP",
        [r"realizes $\ddot y_d+a_e$; hard dynamics, contact,",
         r"friction, unilateral-force, torque limits",
         r"soft body / swing-foot task tracking"], C_WBC, E_WBC, lead=0.036)

    # ---- torque hold + plant ------------------------------------------
    box(ax, 0.24, 0.070, 0.21, 0.140, "Torque hold + joint servo",
        [r"$\tau$ held between 500 Hz updates",
         r"clipped to torque limits"], C_SERVO, E_SERVO, lead=0.040)
    box(ax, 0.52, 0.070, 0.23, 0.14, "Robot / plant (MuJoCo)",
        [r"$q,\dot q$,  contacts,  contact force",
         r"terrain and applied push enter here"], C_ROBOT, E_ROBOT, lead=0.040)

    # ---- forward flow --------------------------------------------------
    # planner -> MPC (reference)
    arrow(ax, (0.560, 0.848), (0.560, 0.73), E_PLAN)
    note(ax, 0.59, 0.78, r"$y_d,\ \ddot y_d$", E_PLAN, fs=8.4, style="normal")
    # planner -> realizer (swing-foot + contact schedule) down the right margin
    poly(ax, [(0.67, 0.888), (0.8, 0.888), (0.8, 0.395), (0.700, 0.395)], E_PLAN)
    arrow(ax, (0.71, 0.395), (0.660, 0.395), E_PLAN)
    note(ax, 0.73, 0.915, "swing-foot +\ncontact schedule", E_PLAN, fs=7.6)
    # estimator -> MPC (horizon residual)
    arrow(ax, (0.40, 0.65), (0.465, 0.65), E_MPC)
    note(ax, 0.43, 0.6900, r"$\hat d_{k+i|k}$", E_MPC, fs=8.2, style="normal")
    # MPC -> realizer (correction a_e)
    arrow(ax, (0.60, 0.565), (0.6, 0.460), E_MPC)
    note(ax, 0.67, 0.5, r"correction $a_e$ (100 Hz)", E_MPC, fs=8.0, style="normal")
    # realizer -> torque hold
    arrow(ax, (0.38, 0.320), (0.38, 0.210), E_WBC)
    note(ax, 0.43, 0.27, r"$\tau$ (500 Hz)", E_WBC, fs=8.0, style="normal")
    # torque hold -> plant
    arrow(ax, (0.42, 0.145), (0.52, 0.145), E_SERVO)
    note(ax, 0.48, 0.176, r"$\tau$ (1kHz)", E_SERVO, fs=7.8, style="normal")

    # ---- feedback ------------------------------------------------------
    # plant -> estimator: measured task motion + contact force (far-left bus,
    # left of the lane labels so it crosses nothing)
    poly(ax, [(0.605, 0.070), (0.605, 0.050), (0.045, 0.050), (0.045, 0.640),
              (0.075, 0.640)], E_ROBOT)
    arrow(ax, (0.068, 0.640), (0.2, 0.640), E_ROBOT)
    note(ax, 0.175, 0.360, "measured task motion,\ncontact force", E_ROBOT, fs=7.4)
    # realizer -> estimator: realized-minus-commanded acceleration (short riser
    # straight up from the QP into the estimator)
    arrow(ax, (0.380, 0.460), (0.380, 0.565), E_WBC, lw=1.7)
    note(ax, 0.470, 0.50, r"$d_{\rm real}$: realized $-$ commanded", E_WBC, fs=7.4)

    # ---- invariance banner --------------------------------------------
    ax.text(0.5, -0.020,
            r"Fixed $(A_d,B_d)$ across gait phase, terrain, and push "
            r"$-$environment enters only $d_{\rm eff}$, task limits, and realizer.",
            ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="bold")

    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
