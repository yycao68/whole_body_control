#!/usr/bin/env python3
"""Figures for the external-push study, generated only from the push JSON/NPZ.

  external_push_summary.png     -- post-push CoM peak error and recovery time,
                                   grouped by controller across the four
                                   direction/phase conditions.
  external_push_response.png    -- representative lateral single-support push:
                                   CoM planar error and applied force in time.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE.parent / "figures"
DATA = json.loads((RESULTS / "external_push_benchmark.json").read_text())

CONTROLLERS = ("impedance", "nominal_mpc", "interaction_mpc")
if DATA.get("schema_version") != 2:
    raise RuntimeError("refusing to plot legacy push artifact; regenerate schema 2")
if tuple(DATA.get("controllers", ())) != CONTROLLERS:
    raise RuntimeError("push artifact is not the exact publication controller matrix")
CONDITIONS = DATA["conditions"]  # "direction:phase"
CLABEL = {"impedance": "impedance", "nominal_mpc": "nominal MPC",
          "interaction_mpc": "ID-MPC"}
COLORS = {"impedance": "#8c8c8c", "nominal_mpc": "#4c78a8",
          "interaction_mpc": "#c05f28"}
CONDLABEL = {"lateral:double_support": "lat / DS", "lateral:single_support": "lat / SS",
             "forward:double_support": "fwd / DS", "forward:single_support": "fwd / SS"}


def cell(direction, phase, controller):
    return DATA["cells"].get(f"{direction}|{phase}|{controller}", {})


def _grouped(ax, metric, ylabel, title):
    x = np.arange(len(CONDITIONS))
    w = 0.26
    for i, c in enumerate(CONTROLLERS):
        vals = []
        for cond in CONDITIONS:
            d, p = cond.split(":")
            value = cell(d, p, c).get(metric)
            vals.append(np.nan if value is None else value)
        ax.bar(x + (i - 1) * w, vals, w, label=CLABEL[c], color=COLORS[c])
    ax.set_xticks(x)
    ax.set_xticklabels([CONDLABEL[c] for c in CONDITIONS], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)


def summary_figure():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    _grouped(a1, "com_peak_mm", "CoM peak error [mm]", "Post-push peak CoM error")
    _grouped(a2, "recovery_time_s", "recovery time [s]", "Recovery to 12 mm band")
    a1.legend(fontsize=7.5, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "external_push_summary.png", dpi=200)
    plt.close(fig)
    print("wrote external_push_summary.png")


def response_figure():
    cond = ("lateral", "single_support")
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax2 = ax.twinx()
    seed = DATA["seed_start"]
    plotted = False
    for c, color in (("nominal_mpc", "#4c78a8"), ("interaction_mpc", "#c05f28")):
        f = RESULTS / f"push_{cond[0]}_{cond[1]}_{c}_seed{seed}.npz"
        if not f.exists():
            continue
        log = np.load(f)
        t = log["t"]
        planar = 1000 * np.linalg.norm(log["task_error"][:, :2], axis=1)
        ax.plot(t, planar, color=color, lw=1.6, label=f"{CLABEL[c]} CoM error")
        if not plotted and "applied_force" in log:
            fmag = np.linalg.norm(log["applied_force"][:, :3], axis=1)
            ax2.fill_between(t, 0, fmag, color="#d0b0a0", alpha=0.35, zorder=0)
            plotted = True
    ax.axhline(12.0, color="#888", ls=":", lw=1.0)
    ax.set_xlabel("t [s]"); ax.set_ylabel("CoM planar error [mm]")
    ax2.set_ylabel("applied force [N]", color="#a06a4a")
    ax.set_xlim(1.2, 4.0); ax.set_ylim(bottom=0)
    ax.set_title("Representative lateral single-support push (seed %d)" % seed)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "external_push_response.png", dpi=200)
    plt.close(fig)
    print("wrote external_push_response.png")


if __name__ == "__main__":
    summary_figure()
    response_figure()
