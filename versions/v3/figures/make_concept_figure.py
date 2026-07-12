#!/usr/bin/env python3
"""Conceptual prediction-realization figure for the interaction-dynamics paper.

Renders the interface view: a robot-independent predictor (the "interface") emits
a residual-acceleration command u; the whole-body realizer (the "implementation")
projects it onto the feasible dynamics, carrying all robot mechanics and returning
the realization residual r; a Kalman observer feeds back the interaction
disturbance d. Saves figures/prediction_realization_concept.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"; OUT.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(11, 3.6))
ax.set_xlim(0, 11); ax.set_ylim(0, 3.6); ax.axis("off")

BLUE = "#2c6fbb"; GRAY = "#555555"; GREEN = "#2e8b57"; LGRAY = "#eef2f7"; LBLUE = "#e6eff7"


def box(x, y, w, h, title, sub, ec, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                ec=ec, fc=fc, lw=1.8))
    ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="center", fontsize=11, fontweight="bold", color=ec)
    ax.text(x + w / 2, y + 0.34, sub, ha="center", va="center", fontsize=8.5, color="#333333")


def arrow(x0, y0, x1, y1, color, label=None, lx=0, ly=0, style="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15,
                                 lw=1.6, color=color, ls=style,
                                 connectionstyle="arc3,rad=%.2f" % rad))
    if label:
        ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label, ha="center", va="center",
                fontsize=8.5, color=color, style="italic")


# top row: generator -> predictor -> realizer -> plant
box(0.2, 1.9, 2.2, 1.2, "Reference", "planner / learned policy", GRAY, LGRAY)
box(3.1, 1.9, 2.7, 1.2, "PREDICTOR", r"interface:  $\ddot e = u + d$" + "\nrobot-independent $(A,B)$", BLUE, LBLUE)
box(6.5, 1.9, 2.7, 1.2, "REALIZER", "whole-body QP: carries\n$M_p$, contacts, limits", GRAY, LGRAY)
box(9.6, 1.9, 1.2, 1.2, "Robot", "actual $\\ddot e$", "#111111", "#f4f4f4")

arrow(2.4, 2.5, 3.1, 2.5, GRAY, r"intent $\ddot e_d$", 0, 0.28)
arrow(5.8, 2.5, 6.5, 2.5, BLUE, r"command $u$", 0, 0.28)
arrow(9.2, 2.5, 9.6, 2.5, GRAY)

# feedback: residual r (realizer -> predictor) and disturbance d (observer)
arrow(6.5, 2.1, 5.8, 2.1, GREEN, r"residual $r$", 0, -0.30, rad=0.0)
box(6.5, 0.25, 2.7, 0.95, "OBSERVER", "Kalman: estimates $d$", GREEN, "#e9f5ee")
arrow(7.85, 1.9, 7.85, 1.2, GRAY, "innovation", 1.15, 0)
arrow(6.5, 0.72, 4.45, 0.72, GREEN)
arrow(4.45, 0.72, 4.45, 1.9, GREEN, r"disturbance $d$", -0.75, 0)

ax.text(4.45, 3.35, "prediction  (future, robot-invariant)", ha="center", fontsize=9,
        color=BLUE, fontweight="bold")
ax.text(7.85, 3.35, "realization  (present, robot-specific)", ha="center", fontsize=9,
        color=GRAY, fontweight="bold")

fig.tight_layout()
fig.savefig(OUT / "prediction_realization_concept.png", dpi=170, bbox_inches="tight")
print("saved:", OUT / "prediction_realization_concept.png")
