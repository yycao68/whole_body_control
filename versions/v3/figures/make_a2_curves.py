#!/usr/bin/env python3
"""Appendix A.2 figure: lateral base-error curves e_y(t) for the six Unitree
open-source locomotion-stack probes (three comparison panels matching the three
retained videos). Reads the gitignored npz logs from ../unitree_locomotion_demo/
results and writes figures/unitree_a2_curves.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "unitree_locomotion_demo" / "results"
OUT = HERE

GRAY, BLUE = "#b0392b", "#2c6fbb"


def ey(name):
    z = np.load(RES / f"{name}_log.npz")
    return z["time"], z["y_error"]


panels = [
    ("(a) no push", [("unitree_base_only", "Unitree policy only", GRAY, "--"),
                     ("unitree_base_idmpc", "+ interaction layer", BLUE, "-")], None),
    ("(b) 40 N lateral push (0.35 s)", [("unitree_push_layer_off", "policy only", GRAY, "--"),
                     ("unitree_push_layer_on", "+ interaction layer", BLUE, "-")], (3.0, 3.35)),
    ("(c) 60 N planned lateral load", [("unitree_load_no_preview", "feedback, no preview", GRAY, "--"),
                     ("unitree_load_preview", "feedback + preview", BLUE, "-")], (3.0, 3.35)),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for ax, (title, curves, push) in zip(axes, panels):
    if push:
        ax.axvspan(push[0], push[1], color="0.85", zorder=0, label="push/load")
    for name, lab, col, ls in curves:
        t, e = ey(name)
        ax.plot(t, e, ls, color=col, lw=1.6, label=lab)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("t [s]"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
axes[0].set_ylabel(r"lateral base error $e_y$ [m]")
fig.tight_layout()
fig.savefig(OUT / "unitree_a2_curves.png", dpi=160, bbox_inches="tight")
print("saved:", OUT / "unitree_a2_curves.png")
