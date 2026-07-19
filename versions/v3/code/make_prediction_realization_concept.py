#!/usr/bin/env python3
"""Figure 2: the interaction-prediction layer between planner and realizer.

Regenerates ``figures/prediction_realization_concept.png`` to match the current
Interaction-Dynamics framing: the predictor runs the fixed, robot-independent
model e_ddot = a_e + d_eff and emits the task-acceleration correction a_e; the
realizer projects it onto the instantaneous robot/contact constraints; a low-pass
observer estimates d_eff from the measured innovation.  (The prior version still
carried the retired authority-set notation u in U_hat_k.)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures" / "prediction_realization_concept.png"

GREY = "#3d3d3d"
BLUE = "#1f6fc4"
GREEN = "#2e8b57"
BLUE_FILL = "#e8f1fb"
GREEN_FILL = "#e9f5ee"
GREY_FILL = "#eef0f2"


def box(ax, x, y, w, h, title, lines, edge, fill, title_color=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=2.0, edgecolor=edge, facecolor=fill, zorder=2))
    cx = x + w / 2
    ax.text(cx, y + h - 0.16, title, ha="center", va="top",
            fontsize=13.5, fontweight="bold",
            color=title_color or edge, zorder=3)
    ax.text(cx, y + h - 0.44, "\n".join(lines), ha="center", va="top",
            fontsize=10.0, color=GREY, zorder=3)


def arrow(ax, p0, p1, color, label=None, lx=0.0, ly=0.0, rad=0.0,
          fontsize=10.5, style="italic", va="center", ha="center"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=15, linewidth=2.0,
        color=color, zorder=1))
    if label:
        mx, my = (p0[0] + p1[0]) / 2 + lx, (p0[1] + p1[1]) / 2 + ly
        ax.text(mx, my, label, color=color, fontsize=fontsize,
                fontstyle=style, ha=ha, va=va, zorder=3)


def main() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    # top-row band labels
    ax.text(3.9, 4.45, "prediction\nfuture, robot-invariant", ha="center",
            va="top", fontsize=11.5, fontweight="bold", color=BLUE)
    ax.text(7.3, 4.45, "realization\npresent, robot-specific", ha="center",
            va="top", fontsize=11.5, fontweight="bold", color=GREY)

    box(ax, 0.25, 2.55, 2.0, 1.15, "Reference",
        ["planner /", "learned policy"], GREY, GREY_FILL)
    box(ax, 2.95, 2.4, 2.0, 1.4, "PREDICTOR",
        [r"$\ddot e = a_e + d_{\mathrm{eff}}$",
         r"fixed $(A_d,B_d)$"], BLUE, BLUE_FILL)
    box(ax, 6.3, 2.4, 2.05, 1.4, "REALIZER",
        ["inverse-dyn. QP", "contacts, limits"], GREY, GREY_FILL,
        title_color=GREY)
    box(ax, 9.05, 2.55, 1.9, 1.15, "Robot",
        [r"actual $\ddot e$"], "#111111", "#f6f6f6", title_color="#111111")
    box(ax, 6.35, 0.35, 2.0, 1.2, "OBSERVER",
        ["low-pass:", r"estimates $d_{\mathrm{eff}}$"], GREEN, GREEN_FILL)

    # forward path
    arrow(ax, (2.25, 3.12), (2.95, 3.12), GREY,
          r"intent $\ddot e_d$", ly=0.22)
    arrow(ax, (4.95, 3.1), (6.3, 3.1), BLUE,
          r"correction $a_e$", ly=0.22)
    arrow(ax, (8.35, 3.12), (9.05, 3.12), GREY)

    # observer loop
    arrow(ax, (7.35, 2.4), (7.35, 1.55), GREY,
          "innovation", lx=0.72, ly=0.0, ha="left")
    arrow(ax, (6.35, 0.95), (3.9, 0.95), GREEN)
    arrow(ax, (3.55, 0.95), (3.55, 2.4), GREEN,
          r"estimate $\hat d_{\mathrm{eff}}$", lx=-0.15, ly=-0.75,
          ha="right", va="center")

    # realization-feedback (residual) into the predictor
    arrow(ax, (6.3, 2.62), (4.95, 2.62), GREEN,
          r"realization residual $d_{\mathrm{real}}$", ly=-0.5, rad=0.18,
          fontsize=9.5)

    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
