#!/usr/bin/env python3
"""Figure for the sustained-force rejection study (reduced body model)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "figures"
d = json.loads((HERE / "results" / "sustained_push_benchmark.json").read_text())
forces = d["forces_n"]
authority_n = d["body_mpc_u_max_lateral_mps2"] * d["mass_kg"]


def val(controller, f, key):
    return d["cells"][f"{controller}|{int(f)}"][key]


fig, ax = plt.subplots(figsize=(6.6, 3.6))
x = np.arange(len(forces))
w = 0.38
nom = [val("nominal", f, "steady_offset_mm") for f in forces]
inter = [val("interaction", f, "steady_offset_mm") for f in forces]
ax.bar(x - w / 2, nom, w, label="nominal MPC", color="#8c8c8c")
ax.bar(x + w / 2, inter, w, label="ID-MPC", color="#c05f28")
for xi, (a, b) in enumerate(zip(nom, inter)):
    ax.text(xi - w / 2, a + 4, f"{a:.0f}", ha="center", fontsize=8, color="#555")
    ax.text(xi + w / 2, b + 4, f"{b:.0f}", ha="center", fontsize=8, color="#c05f28")
ax.axvline(0.5 + (authority_n - 50) / 20, color="#3b6ea5", ls="--", lw=1.2)
ax.text(1.02, max(nom) * 0.92, f"command\nauthority\n≈{authority_n:.0f} N",
        fontsize=7.6, color="#3b6ea5", ha="left", va="top")
ax.set_xticks(x, [f"{int(f)} N" for f in forces])
ax.set_ylabel("steady-state lateral CoM offset [mm]")
ax.set_title("Sustained 1 s lateral force: offset-free rejection and its authority limit",
             fontsize=9.5)
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
out = FIGURES / "sustained_push_offset.png"
fig.savefig(out, dpi=200)
print("wrote", out)
