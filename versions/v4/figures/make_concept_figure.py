#!/usr/bin/env python3
"""Conceptual prediction-realization figure for the interaction-dynamics paper.

Renders the evaluated interface view: a fixed-model predictor emits a bounded
task-acceleration correction; the whole-body realizer maps it to feasible torque
and contact force; and a low-pass measured-acceleration residual is fed back.
Saves figures/prediction_realization_concept.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
OUT = HERE; OUT.mkdir(exist_ok=True)   # this script lives in figures/

fig, ax = plt.subplots(figsize=(8.2, 3.2))
ax.set_xlim(0, 7.8); ax.set_ylim(-0.05, 3.05); ax.axis("off")

BLUE = "#2c6fbb"; GRAY = "#555555"; GREEN = "#2e8b57"; AMBER = "#c56b00"; LGRAY = "#eef2f7"; LBLUE = "#e6eff7"


def box(x, y, w, h, title, sub, ec, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                ec=ec, fc=fc, lw=1.8))
    ax.text(x + w / 2, y + h - 0.20, title, ha="center", va="center", fontsize=11, fontweight="bold", color=ec)
    ax.text(x + w / 2, y + 0.29, sub, ha="center", va="center", fontsize=8.5, color="#333333")


def arrow(x0, y0, x1, y1, color, label=None, lx=0, ly=0, style="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15,
                                 lw=1.6, color=color, ls=style,
                                 connectionstyle="arc3,rad=%.2f" % rad))
    if label:
        ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label, ha="center", va="center",
                fontsize=8.5, color=color, style="italic")


# top row: generator -> predictor -> realizer -> plant
box(0.1, 1.4, 1.2, 1.0, "Reference", "planner\nlearned policy", GRAY, LGRAY)
box(2.1, 1.4, 1.6, 1.0, "PREDICTOR", r"$\ddot e = a_e + \tilde d$" + "\nfixed $(A_d,B_d)$; fixed bounds", BLUE, LBLUE)
box(4.6, 1.4, 1.6, 1.0, "REALIZER", "contact-constrained QP\nrobot-specific limits", GRAY, LGRAY)
box(6.8, 1.4, 0.8, 1.0, "Robot", "actual $\\ddot e$", "#111111", "#f4f4f4")

arrow(1.3, 1.9, 2.1, 1.9, GRAY, r"intent $\ddot e_d$", 0, 0.2)
arrow(3.7, 2.1, 4.6, 2.1, BLUE, r"correction $a_e$", 0, 0.2)
arrow(6.2, 1.9, 6.8, 1.9, GRAY)

# The evaluated predictor uses fixed acceleration bounds. The feedback object
# is the aggregate measured-minus-commanded residual, not an online authority
# polytope or a QP-sensitivity extrapolation.
box(4.6, 0.1, 1.6, 0.8, "ESTIMATOR",
    "low-pass residual\n" + r"conditions $\hat d_{\rm eff}$",
    GREEN, "#e9f5ee")
arrow(7.2, 1.4, 5.4, 0.9, GRAY, "measured motion", 0.35, -0.02, rad=0.18)
arrow(4.6, 0.5, 3.2, 0.5, GREEN)
arrow(3.2, 0.5, 3.2, 1.4, GREEN, r"conditioned $\tilde d$", -0.6, 0)

ax.text(2.8, 2.6, "prediction\nfuture, configuration-invariant", ha="center", fontsize=9,
        color=BLUE, fontweight="bold")
ax.text(5.4, 2.6, "realization\npresent, robot-specific", ha="center", fontsize=9,
        color=GRAY, fontweight="bold")

fig.subplots_adjust(left=0.015, right=0.985, bottom=0.03, top=0.97)
fig.savefig(OUT / "prediction_realization_concept.png", dpi=170,
            bbox_inches="tight", pad_inches=0.08)
print("saved:", OUT / "prediction_realization_concept.png")
