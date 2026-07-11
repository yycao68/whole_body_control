#!/usr/bin/env python3
"""Render the v3 dual-MPC architecture figure as a PNG for Markdown preview."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).with_name("interaction_dynamics_ports_architecture.png")


def box(ax, x, y, w, h, title, lines, fc, ec, title_size=11, line_size=8.5):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.8,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    title_lines = title.split("\n")
    for j, title_line in enumerate(title_lines):
        ax.text(x + w / 2, y + h - 0.014 - j * 0.033, title_line,
                ha="center", va="top", fontsize=title_size,
                fontweight="bold", color="#111827")
    first_line_y = y + h - 0.068 - (len(title_lines) - 1) * 0.033
    for i, line in enumerate(lines):
        ax.text(x + w / 2, first_line_y - i * 0.038, line,
                ha="center", va="top", fontsize=line_size, color="#1f2937")


def arrow(ax, p0, p1, color="#1f2937", dashed=False, rad=0.0):
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.8,
        color=color,
        linestyle=(0, (6, 5)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)


def poly_arrow(ax, points, color="#1f2937", dashed=False):
    linestyle = (0, (6, 5)) if dashed else "solid"
    for p0, p1 in zip(points[:-2], points[1:-1]):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                color=color, linewidth=1.8, linestyle=linestyle)
    arrow(ax, points[-2], points[-1], color=color, dashed=dashed)


def main():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blue = "#2563eb"
    green = "#059669"
    amber = "#d97706"
    gray = "#6b7280"

    ax.text(
        0.5,
        0.965,
        "Interaction-Dynamics Ports for Floating-Base Whole-Body Manipulation",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.5,
        0.925,
        "Prediction is normalized; robot-dependent physics is recovered through wrenches, contacts, and torque constraints.",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#374151",
    )

    box(ax, 0.11, 0.77, 0.12, 0.08, "Body Reference", ["CoM, attitude, gait mode"], "#f9fafb", gray)
    box(ax, 0.755, 0.77, 0.14, 0.08, "Task Reference", ["End-effector motion and force"], "#f9fafb", gray)

    box(
        ax,
        0.07,
        0.49,
        0.20,
        0.13,
        "Body Interaction MPC",
        [
            r"$x_b^+=A_bx_b+B_b(u_b+d_b)$",
            "constant exact-ZOH predictor",
        ],
        "#eff6ff",
        blue,
        title_size=10.5,
        line_size=8.2,
    )
    box(
        ax,
        0.73,
        0.49,
        0.20,
        0.13,
        "Task Interaction MPC",
        [
            r"$x_t^+=A_tx_t+B_t(u_t+d_t)$",
            "constant exact-ZOH predictor",
        ],
        "#ecfdf5",
        green,
        title_size=10.5,
        line_size=8.2,
    )

    box(
        ax,
        0.07,
        0.24,
        0.22,
        0.13,
        "Centroidal Wrench Recovery",
        [
            r"$W_b^{des}$ = feedforward + inertia x $u_b$",
            "contacts, friction, CoP, torque surrogate",
        ],
        "#eff6ff",
        blue,
    )
    box(
        ax,
        0.71,
        0.24,
        0.22,
        0.12,
        "Task Wrench Recovery",
        [
            r"$F_t=F_{ff}+\Lambda_tu_t$",
            "contact-consistent inertia and force limits",
        ],
        "#ecfdf5",
        green,
    )

    box(
        ax,
        0.40,
        0.23,
        0.22,
        0.15,
        "Unitree G1 / MuJoCo Plant",
        [
            "floating-base rigid-body dynamics",
            "contacts, wrenches, and actuator response",
            "measured state and contact feedback",
        ],
        "#f3f4f6",
        "#111827",
        title_size=10.5,
        line_size=8.1,
    )

    box(
        ax,
        0.40,
        0.025,
        0.22,
        0.115,
        "Whole-Body Interaction Realizer",
        [
            "inverse-dynamics QP",
            r"feasible generalized torque $\tau$",
        ],
        "#fffbeb",
        amber,
        title_size=10.2,
        line_size=7.8,
    )

    box(
        ax,
        0.40,
        0.49,
        0.2,
        0.12,
        "Kalman Disturbance Est",
        [
            r"random-walk states $d_b,d_t$",
            "innovation contact/event gating",
        ],
        "#f9fafb",
        gray,
        title_size=10.5,
        line_size=7.8,
    )

    box(
        ax,
        0.40,
        0.755,
        0.20,
        0.11,
        "Shared Prediction Object",
        ["constant A,B", "robot dependence enters recovery"],
        "#f9fafb",
        gray,
        line_size=8.1,
    )

    arrow(ax, (0.18, 0.755), (0.18, 0.63), blue)
    arrow(ax, (0.82, 0.76), (0.82, 0.63), green)
    arrow(ax, (0.18, 0.48), (0.18, 0.38), blue)
    arrow(ax, (0.82, 0.48), (0.82, 0.37), green)
    poly_arrow(ax, [(0.18, 0.22), (0.18, 0.095), (0.385, 0.095)], blue)
    poly_arrow(ax, [(0.82, 0.22), (0.82, 0.095), (0.635, 0.095)], green)
    arrow(ax, (0.50, 0.155), (0.50, 0.22), "#1f2937")

    arrow(ax, (0.71, 0.63), (0.28, 0.63), green, dashed=True, rad=0.12)
    ax.text(0.505, 0.70, r"arm-reaction preview $W_{b\leftarrow t}$",
            ha="center", va="center", fontsize=8, color="#374151",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.9))

    arrow(ax, (0.50, 0.395), (0.50, 0.480), "#1f2937", dashed=True)
    ax.text(0.535, 0.445, "state, contacts, wrenches", ha="left", va="center",
            fontsize=8, color="#4b5563")

    arrow(ax, (0.38, 0.565), (0.285, 0.565), "#1f2937", dashed=True)
    arrow(ax, (0.615, 0.565), (0.715, 0.565), "#1f2937", dashed=True)
    ax.text(0.36, 0.595, r"$\hat d_b$", ha="center", va="center", fontsize=8)
    ax.text(0.64, 0.595, r"$\hat d_t$", ha="center", va="center", fontsize=8)

    poly_arrow(ax, [(0.38, 0.78), (0.26, 0.78), (0.26, 0.63)],
               "#1f2937", dashed=True)
    poly_arrow(ax, [(0.62, 0.78), (0.725, 0.78), (0.725, 0.63)],
               "#1f2937", dashed=True)

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.18)
    print(OUT)


if __name__ == "__main__":
    main()
