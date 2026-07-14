#!/usr/bin/env python3
"""Figure 1: the multirate interaction-dynamics architecture.

The point of the figure is the RATE separation, which the old dual-MPC diagram
did not show at all:

  * one 1 kHz realization loop that solves exactly ONE whole-body QP per cycle,
    emits torques, and publishes an authority snapshot from ONE KKT solve;
  * two asynchronous canonical predictors (body 200 Hz, task 500 Hz) that read
    the latest snapshot and never block the fast loop;
  * the canonical pair (A, B) is constant -- contact mode and configuration move
    only the admissible set (H_k, h_k).

Layout is laid out in three reserved horizontal regions so nothing overlaps:
    predictors   y in [0.63, 0.99]
    publish/read y in [0.42, 0.61]   (left / centre / right kept clear of each other)
    realization  y in [0.02, 0.40]
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).with_name("multirate_architecture.png")

C_SLOW, E_SLOW = "#e6edf7", "#38639f"     # predictors
C_FAST, E_FAST = "#fdece2", "#c05f28"     # 1 kHz realization loop
C_PUB, E_PUB = "#e8f2e9", "#37743f"       # published snapshot
C_ROBOT, E_ROBOT = "#ebebeb", "#565656"
GREY = "#8d8d8d"
INK = "#171717"


def box(ax, x, y, w, h, title, body, fc, ec, ts=10.5, bs=8.2, lead=0.042):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.010",
        linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=3))
    ty = y + h - 0.032
    if title:
        ax.text(x + w / 2, ty, title, ha="center", va="top",
                fontsize=ts, fontweight="bold", color=INK, zorder=4)
        ty -= 0.050
    for line in body:
        ax.text(x + w / 2, ty, line, ha="center", va="top",
                fontsize=bs, color=INK, zorder=4)
        ty -= lead


def arrow(ax, p, q, color, style="-", lw=1.7, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=14, linewidth=lw,
        linestyle=style, color=color, connectionstyle=f"arc3,rad={rad}",
        zorder=2, shrinkA=2, shrinkB=2))


def note(ax, x, y, text, color, fs=7.8, weight="normal", style="italic"):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color,
            style=style, fontweight=weight, zorder=6,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.95))


def main():
    fig, ax = plt.subplots(figsize=(14.2, 7.8))
    ax.set_xlim(0, 1); ax.set_ylim(-0.055, 1.0); ax.axis("off")

    # ---------------- lanes -------------------------------------------------
    ax.add_patch(Rectangle((0.015, 0.630), 0.970, 0.360, facecolor="#f6f9fd",
                           edgecolor="#c6d7ea", lw=1.0, zorder=0))
    ax.add_patch(Rectangle((0.015, 0.020), 0.970, 0.380, facecolor="#fdf8f4",
                           edgecolor="#ebcdb7", lw=1.0, zorder=0))

    ax.text(0.030, 0.978, "ASYNCHRONOUS CANONICAL PREDICTORS", fontsize=10,
            fontweight="bold", color=E_SLOW, va="top", zorder=6)
    ax.text(0.030, 0.949, "slow, interruptible — never block the loop below",
            fontsize=8.2, color=E_SLOW, va="top", style="italic", zorder=6)

    ax.text(0.140, 0.390, "1 kHz REALIZATION LOOP", fontsize=10,
            fontweight="bold", color=E_FAST, va="top", zorder=6)
    ax.text(0.140, 0.361, "hard real-time — exactly ONE whole-body QP per cycle; never waits",
            fontsize=8.2, color=E_FAST, va="top", style="italic", zorder=6)

    # ---------------- predictors (y 0.655 .. 0.920) -------------------------
    box(ax, 0.055, 0.655, 0.25, 0.265,
        "Body predictor   ·   200 Hz",
        [r"$x_{b,k+1}=A_b x_{b,k}+B_b(u_b+\hat d_b)$",
         r"$(A_b,B_b)$  constant",
         r"s.t.   $H_{b,k}\,u_b \leq h_{b,k}$",
         r"$\hat d_b \to d^{\mathrm{eff}}=d+r_{\mathrm{matched}}$"],
        C_SLOW, E_SLOW)

    box(ax, 0.65, 0.655, 0.255, 0.265,
        "Task predictor   ·   500 Hz",
        [r"$x_{t,k+1}=A_t x_{t,k}+B_t(u_t+\hat d_t)$",
         r"$(A_t,B_t)$  constant",
         r"s.t.   $H_{t,k}u_t \leq h_{t,k}-H_{tb,k}\,u_b^{\star}$",
         r"observer  $\hat d_t$"],
        C_SLOW, E_SLOW)

    arrow(ax, (0.30, 0.760), (0.65, 0.760), E_SLOW, lw=1.8)
    note(ax, 0.4825, 0.800, "body-priority allocation\ntask receives the remaining capacity",
         E_SLOW, fs=7.9)

    # ---------------- realization loop (boxes y 0.090 .. 0.320) -------------
    box(ax, 0.040, 0.090, 0.150, 0.230, "Robot / plant",
        [r"$q,\ \dot q$,  contact mode $\rho$", "measured state"],
        C_ROBOT, E_ROBOT)
    box(ax, 0.245, 0.090, 0.18, 0.230, "Dynamics + feedforward",
        [r"$M,\ h,\ J_c,\ J_t$",
         r"requests:  $\ddot c_d+u_b^{\star}$",
         r"$\ddot x_{t,d}+u_t^{\star}$"],
        C_FAST, E_FAST, lead=0.040)
    box(ax, 0.485, 0.090, 0.18, 0.230, "Whole-body QP  (14b)",
        [r"one solve $\to z_0$", r"$\tau_{\mathrm{ff}},\ \lambda_{\mathrm{ff}}$",
         "hard: torque, friction,", "unilateral, joint limits"],
        C_FAST, E_FAST, lead=0.038)
    box(ax, 0.735, 0.090, 0.18, 0.230, "KKT sensitivity  (14c)",
        [r"$K=\partial z^{\star}/\partial u$",
         r"$\tau=\tau_{\mathrm{ff}}+K_\tau u$",
         r"$\lambda=\lambda_{\mathrm{ff}}+K_\lambda u$",
         "same factorization · 0.15 ms"],
        C_FAST, E_FAST, lead=0.038)

    for x0, x1 in ((0.170, 0.245), (0.410, 0.485), (0.66, 0.735)):
        arrow(ax, (x0, 0.205), (x1, 0.205), E_FAST, lw=1.9)

    # torque return: explicit polyline below the boxes so it is unambiguous
    ax.plot([0.560, 0.560], [0.090, 0.055], color=E_FAST, lw=1.9, zorder=2)
    ax.plot([0.560, 0.125], [0.055, 0.055], color=E_FAST, lw=1.9, zorder=2)
    arrow(ax, (0.125, 0.055), (0.125, 0.090), E_FAST, lw=1.9)
    note(ax, 0.345, 0.055, r"joint torques  $\tau$", E_FAST, fs=8.8, style="normal")

    # ---------------- published snapshot (centre of the middle strip) -------
    box(ax, 0.40, 0.437, 0.30, 0.140, "",
        [r"published snapshot  ·  $H_k u \leq h_k$   (14e)",
         r"$\tau_{\mathrm{ff}},\ \lambda_{\mathrm{ff}},\ K_\tau,\ K_\lambda$,  margins,",
         r"timestamp,  contact mode"],
        C_PUB, E_PUB, bs=8.4, lead=0.038)

    # realizer -> snapshot
    arrow(ax, (0.8, 0.30), (0.8, 0.50), E_PUB, lw=1.8)
    arrow(ax, (0.8, 0.50), (0.7, 0.5), E_PUB, lw=1.8)
    # snapshot -> predictors (dashed = read asynchronously)
    arrow(ax, (0.395, 0.5), (0.28, 0.5), E_PUB, lw=1.7, style="--")
    arrow(ax, (0.28, 0.5), (0.28, 0.68), E_PUB, lw=1.7, style="--")
    arrow(ax, (0.68, 0.577), (0.68, 0.655), E_PUB, lw=1.7, style="--")

    note(ax, 0.180, 0.585, "read latest\n(non-blocking)", E_PUB)
    note(ax, 0.795, 0.550, "stale or wrong mode\n$\\Rightarrow$ conservative fallback", E_PUB)

    # ---------------- commands down (last valid) ----------------------------
    arrow(ax, (0.085, 0.655), (0.085, 0.404), E_SLOW, lw=1.8, style="--")
    arrow(ax, (0.87, 0.655), (0.87, 0.404), E_SLOW, lw=1.8, style="--")
    note(ax, 0.085, 0.545, r"$u_b^{\star}$", E_SLOW, fs=10.5, style="normal")
    note(ax, 0.87, 0.545, r"$u_t^{\star}$", E_SLOW, fs=10.5, style="normal")
    note(ax, 0.085, 0.470, "last valid command\nloop never waits", E_SLOW, fs=7.4)

    # ---------------- logged realization residual ---------------------------
    arrow(ax, (0.57, 0.320), (0.57, 0.437), GREY, lw=1.4, style=":")
    note(ax, 0.660, 0.392, r"logged residual  $r=\ddot e^{\mathrm{real}}-\ddot e^{\mathrm{req}}$",
         GREY, fs=7.8)

    # ---------------- the invariance claim ----------------------------------
    ax.text(0.5, -0.038,
            r"Contact mode and configuration move only $(H_k,h_k)$ — never $(A,B)$.",
            ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")

    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
