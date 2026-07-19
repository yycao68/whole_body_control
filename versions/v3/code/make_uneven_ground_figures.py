#!/usr/bin/env python3
"""Generate every manuscript figure derived from the uneven-ground benchmark."""

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
TERRAINS = ("flat", "depression", "obstacle", "rough")
CONTROLLERS = ("impedance", "nominal_mpc", "interaction_mpc")
LABELS = ("Impedance", "Nominal MPC", "ID-MPC")
COLORS = ("#777777", "#4477AA", "#CC6677")


def load() -> dict:
    return json.loads((RESULTS / "uneven_ground_benchmark.json").read_text())


def tracking_figure(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), constrained_layout=True)
    x = np.arange(len(TERRAINS))
    width = 0.19
    for ci, (controller, label, color) in enumerate(zip(CONTROLLERS, LABELS, COLORS)):
        rms, peak, falls = [], [], []
        for terrain in TERRAINS:
            cell = data["cells"][f"{terrain}/{controller}"]
            rms.append(cell["com_xyz_rms_mm"]["median"])
            peak.append(cell["com_xyz_peak_mm"]["median"])
            falls.append(cell["falls"])
        offset = (ci - 1.5) * width
        bars = axes[0].bar(x + offset, rms, width, label=label, color=color)
        axes[1].bar(x + offset, peak, width, color=color)
        for bar, nfall in zip(bars, falls):
            if nfall:
                axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.16,
                             f"{nfall}/10 falls", ha="center", va="bottom", fontsize=7,
                             rotation=90, color="#8B0000")
    for ax, ylabel in zip(axes, ("CoM tracking RMS (mm)", "CoM tracking peak (mm)")):
        ax.set_xticks(x, [s.capitalize() for s in TERRAINS])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    fig.savefig(FIGURES / "uneven_ground_tracking.png", dpi=240)
    plt.close(fig)


def prediction_figure(data: dict) -> None:
    # Prediction quality is evaluated on nominal-MPC trials so it is not
    # confounded by the different closed-loop commands of the four controllers.
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), constrained_layout=True)
    horizons = (1, 5, 10)
    styles = {"nominal": ("--", "o"), "augmented": ("-", "s")}
    terrain_colors = ("#4477AA", "#EE6677", "#228833", "#CCBB44")
    for ti, terrain in enumerate(TERRAINS):
        trials = [t for t in data["trials"]
                  if t["terrain"] == terrain and t["controller"] == "nominal_mpc"]
        for model in ("nominal", "augmented"):
            com = [np.median([t["prediction"][str(h)][f"{model}_com_rmse_mm"] for t in trials])
                   for h in horizons]
            rp = [np.median([t["prediction"][str(h)][f"{model}_roll_pitch_rmse_mrad"]
                             for t in trials]) for h in horizons]
            ls, marker = styles[model]
            label = f"{terrain.capitalize()} - {model}"
            axes[0].plot(horizons, com, ls=ls, marker=marker, color=terrain_colors[ti], label=label)
            axes[1].plot(horizons, rp, ls=ls, marker=marker, color=terrain_colors[ti])
    for ax, ylabel in zip(axes, ("CoM prediction RMSE (mm)", "Roll/pitch prediction RMSE (mrad)")):
        ax.set_xlabel("Prediction horizon (ms)")
        ax.set_xticks(horizons)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(FIGURES / "uneven_ground_prediction.png", dpi=240)
    plt.close(fig)


def timeseries_figure() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 6.0), sharex=True, constrained_layout=True)
    for controller, label, color in zip(CONTROLLERS, LABELS, COLORS):
        z = np.load(RESULTS / f"uneven_obstacle_{controller}_seed4200.npz")
        t = z["t"]
        com_err = 1e3 * np.linalg.norm(z["task_error"][:, :3], axis=1)
        rp = 1e3 * np.linalg.norm(z["rpy"][:, :2], axis=1)
        residual = np.linalg.norm(z["realization_residual"], axis=1)
        axes[0].plot(t, com_err, color=color, label=label, lw=1.05)
        axes[1].plot(t, rp, color=color, lw=1.05)
        axes[2].plot(t, residual, color=color, lw=1.05)
    axes[0].set_ylabel("CoM error norm (mm)")
    axes[1].set_ylabel("Roll/pitch norm (mrad)")
    axes[2].set_ylabel(r"Realization residual (m/s$^2$)")
    axes[2].set_xlabel("Time (s)")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGURES / "uneven_ground_timeseries.png", dpi=240)
    plt.close(fig)


def timing_figure(data: dict) -> None:
    wbc_median = np.asarray([t["timing"]["wbc"]["median_ms"] for t in data["trials"]])
    wbc_p99 = np.asarray([t["timing"]["wbc"]["p99_ms"] for t in data["trials"]])
    mpc_median = np.asarray([t["timing"]["mpc"]["median_ms"] for t in data["trials"]])
    mpc_p99 = np.asarray([t["timing"]["mpc"]["p99_ms"] for t in data["trials"]])
    vals = [np.median(wbc_median), np.median(wbc_p99), np.median(mpc_median), np.median(mpc_p99)]
    fig, ax = plt.subplots(figsize=(6.8, 3.5), constrained_layout=True)
    bars = ax.bar(np.arange(4), vals, color=(COLORS[1], COLORS[1], COLORS[2], COLORS[2]))
    ax.set_xticks(np.arange(4), ("WBC median", "WBC p99", "MPC median", "MPC p99"))
    ax.set_ylabel("Wall-clock time (ms)")
    # Schedule periods shown as neutral context, not missed deadlines: this is a
    # prototype measurement on a general-purpose, non-real-time host in Python.
    ax.axhline(2.0, color="#9a9a9a", ls="--", lw=1.1, label="500 Hz period (2 ms)")
    ax.axhline(10.0, color="#c8c8c8", ls=":", lw=1.3, label="100 Hz period (10 ms)")
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.15, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, max(10.8, max(vals) + 1.0))
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.set_title("Prototype wall-clock timing (non-real-time host, unoptimized Python)",
                 fontsize=9.5)
    fig.savefig(FIGURES / "uneven_ground_timing.png", dpi=240)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load()
    tracking_figure(data)
    prediction_figure(data)
    timeseries_figure()
    timing_figure(data)
    for name in ("tracking", "prediction", "timeseries", "timing"):
        print(FIGURES / f"uneven_ground_{name}.png")


if __name__ == "__main__":
    main()
