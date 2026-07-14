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
OUT = HERE; OUT.mkdir(exist_ok=True)   # this script lives in figures/

fig, ax = plt.subplots(figsize=(7.8, 3.0))
ax.set_xlim(0, 7.8); ax.set_ylim(0, 3.0); ax.axis("off")

BLUE = "#2c6fbb"; GRAY = "#555555"; GREEN = "#2e8b57"; AMBER = "#c56b00"; LGRAY = "#eef2f7"; LBLUE = "#e6eff7"


def box(x, y, w, h, title, sub, ec, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                ec=ec, fc=fc, lw=1.8))
    ax.text(x + w / 2, y + h - 0.15, title, ha="center", va="center", fontsize=11, fontweight="bold", color=ec)
    ax.text(x + w / 2, y + 0.3, sub, ha="center", va="center", fontsize=8.5, color="#333333")


def arrow(x0, y0, x1, y1, color, label=None, lx=0, ly=0, style="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15,
                                 lw=1.6, color=color, ls=style,
                                 connectionstyle="arc3,rad=%.2f" % rad))
    if label:
        ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label, ha="center", va="center",
                fontsize=8.5, color=color, style="italic")


# top row: generator -> predictor -> realizer -> plant
box(0.1, 1.6, 1.2, 0.8, "Reference", "planner\nlearned policy", GRAY, LGRAY)
box(2.1, 1.6, 1.6, 0.8, "PREDICTOR", r"$\ddot e = u + d$;  $u\in\widehat{\mathcal{U}}_k$" + "\nrobot-independent $(A,B)$", BLUE, LBLUE)
box(4.6, 1.6, 1.6, 0.8, "REALIZER", "nominal QP + sensitivity\n$M_p$, contacts, limits", GRAY, LGRAY)
box(6.8, 1.6, 0.8, 0.8, "Robot", "actual $\\ddot e$", "#111111", "#f4f4f4")

arrow(1.3, 2.0, 2.1, 2.0, GRAY, r"intent $\ddot e_d$", 0, 0.2)
arrow(3.7, 2.1, 4.6, 2.1, BLUE, r"command $u$", 0, 0.2)
arrow(6.2, 2.0, 6.8, 2.0, GRAY)

# Capability and residual are separate feedback objects: authority constrains
# the command before optimization; the residual accounts for execution after.
arrow(4.9, 1.6, 3.4, 1.6, AMBER, r"authority $\widehat{\mathcal{U}}_k$", 0, -0.23, rad=-0.18)
arrow(4.6, 1.9, 3.7, 1.9, GREEN, r"residual $r$", 0, -0.20, rad=0.0)
box(4.6, 0.1, 1.6, 0.8, "OBSERVER", "Kalman: estimates $d$", GREEN, "#e9f5ee")
arrow(5.4, 1.6, 5.4, 0.9, GRAY, "innovation", 0.5, 0)
arrow(4.6, 0.5, 2.9, 0.5, GREEN)
arrow(2.9, 0.5, 2.9, 1.6, GREEN, r"disturbance $d$", -0.6, 0)

ax.text(2.8, 2.6, "prediction\nfuture, robot-invariant", ha="center", fontsize=9,
        color=BLUE, fontweight="bold")
ax.text(5.4, 2.6, "realization\npresent, robot-specific", ha="center", fontsize=9,
        color=GRAY, fontweight="bold")

fig.tight_layout()
fig.savefig(OUT / "prediction_realization_concept.png", dpi=170, bbox_inches="tight")
print("saved:", OUT / "prediction_realization_concept.png")
