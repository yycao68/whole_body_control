"""Envelope-curve sweeps for the wrench-unified ID controller.

Transient push envelope: fall fraction vs push magnitude (lateral DS/SS).
Sustained force envelope: steady lateral offset vs force magnitude.
Compares policy / regime-specialist impedance / wrench-unified id_mpc, then
writes results/wrench_envelope.json and figures/wrench_envelope.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stage2_id_on_policy as S

HERE = Path(__file__).resolve().parent
(HERE / "results").mkdir(exist_ok=True)
(HERE / "figures").mkdir(exist_ok=True)

SEEDS = list(range(1000, 1010))
PUSHES = [200, 240, 280, 320, 360]
FORCES = [6, 10, 14, 18, 22]
PUSH_NOISE = 6.0
FORCE_NOISE = 4.0

# (label, kind, id_mode) — impedance is the regime-appropriate specialist
PUSH_CTRL = [("policy", "policy", "transient"),
             ("impedance (capture)", "impedance", "transient"),
             ("id_mpc (wrench)", "id_mpc", "wrench")]
FORCE_CTRL = [("policy", "policy", "sustained"),
              ("impedance (against-P)", "impedance", "sustained"),
              ("id_mpc (wrench)", "id_mpc", "wrench")]


def push_cell(kind, id_mode, push, phase):
    falls = 0
    for s in SEEDS:
        r = S.run(kind, push_n=push, push_dir=(0, 1), push_phase=phase,
                  seed=s, process_noise=PUSH_NOISE, id_mode=id_mode)
        falls += int(r["fell"])
    return falls / len(SEEDS)


def force_cell(kind, id_mode, force):
    offs = []
    for s in SEEDS:
        r = S.run(kind, push_n=force, push_t=3.0, push_dir=(0, 1), push_dur=3.0,
                  push_phase="time", duration=8.0, seed=s,
                  process_noise=FORCE_NOISE, id_mode=id_mode)
        if not r["fell"]:
            offs.append(abs(r["lat_offset_mm"]))
    return float(np.median(offs)) if offs else float("nan")


def main():
    out = {"pushes": PUSHES, "forces": FORCES, "push_fall_frac": {}, "force_offset_mm": {}}

    print("== transient push envelope (fall fraction) ==")
    for phase in ("DS", "SS"):
        for label, kind, mode in PUSH_CTRL:
            key = f"{phase}|{label}"
            out["push_fall_frac"][key] = [push_cell(kind, mode, p, phase) for p in PUSHES]
            print(f"  {key:32s} " + " ".join(f"{v:.1f}" for v in out["push_fall_frac"][key]), flush=True)

    print("== sustained force envelope (median offset mm) ==")
    for label, kind, mode in FORCE_CTRL:
        out["force_offset_mm"][label] = [force_cell(kind, mode, f) for f in FORCES]
        print(f"  {label:24s} " + " ".join(f"{v:5.0f}" for v in out["force_offset_mm"][label]), flush=True)

    (HERE / "results" / "wrench_envelope.json").write_text(json.dumps(out, indent=2))

    # ---- figure: colorblind-safe, light theme, clear labels ----
    C = {"policy": "#4C4C4C", "impedance": "#0072B2", "id_mpc": "#D55E00"}
    def color(label):
        return C["id_mpc"] if "wrench" in label else (C["impedance"] if "impedance" in label else C["policy"])
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))

    for j, phase in enumerate(("DS", "SS")):
        for label, kind, mode in PUSH_CTRL:
            y = np.array(out["push_fall_frac"][f"{phase}|{label}"]) * 100
            ax[j].plot(PUSHES, y, "-o", color=color(label), label=label, lw=2, ms=5)
        ax[j].set_title(f"Transient push — lateral {phase}")
        ax[j].set_xlabel("push magnitude [N]"); ax[j].set_ylabel("fall rate [%]")
        ax[j].set_ylim(-5, 105); ax[j].grid(alpha=0.3)
    ax[0].legend(fontsize=8, loc="upper left")

    for label, kind, mode in FORCE_CTRL:
        ax[2].plot(FORCES, out["force_offset_mm"][label], "-o", color=color(label), label=label, lw=2, ms=5)
    ax[2].set_title("Sustained force — steady offset")
    ax[2].set_xlabel("lateral force [N]"); ax[2].set_ylabel("lateral CoM offset [mm]")
    ax[2].grid(alpha=0.3); ax[2].legend(fontsize=8, loc="upper left")

    fig.suptitle("Wrench-unified interaction layer on the Unitree G1 policy — recovery envelopes", fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "wrench_envelope.png", dpi=160)
    print("saved figures/wrench_envelope.png and results/wrench_envelope.json")


if __name__ == "__main__":
    main()
