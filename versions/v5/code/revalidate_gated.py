"""Paper-ready re-run of the affected studies WITH the capture gate.

Transient: fall-based, phase-locked, process_noise=6 (push >> noise).
Sustained: floor-corrected PAIRED metric (offset_force - same-seed offset_noforce),
           process_noise=1 (small absolute noise for 8-12 N forces; the wrench
           sustained variance grows with noise/force ratio because capture
           amplifies present drift during the detection delay — reported honestly).
Produces: A) 6-controller ablation, B) 40-seed paired stats (McNemar), C) envelope
(+figure), D) sustained sensor-bias with the gate.
"""
import json
from math import comb
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stage2_id_on_policy as S

HERE = Path(__file__).resolve().parent
ZERO = lambda t: (0.0, 0.0)
TN, SN = 6, 1                                    # transient / sustained process noise


def kindmode(mode):
    return ("policy", "policy") if mode == "policy" else ("id_mpc", mode)


def falls(mode, push, seeds, phase="SS", pdir=(0, 1)):
    k, m = kindmode(mode)
    return np.array([int(S.run(k, push_n=push, push_dir=pdir, push_phase=phase, seed=s,
                    process_noise=TN, id_mode=m)["fell"]) for s in seeds])


def sdrift(mode, force, seeds):
    k, m = kindmode(mode); d = []
    for s in seeds:
        rf = S.run(k, push_n=force, push_t=3, push_dir=(0, 1), push_dur=3, push_phase="time",
                   duration=8, seed=s, process_noise=SN, id_mode=m)
        r0 = S.run(k, push_n=1.0, push_t=3, push_dir=(0, 1), push_dur=3, push_phase="time",
                   duration=8, seed=s, process_noise=SN, id_mode=m, force_override=ZERO)
        if not rf["fell"] and not r0["fell"]:
            d.append(abs(rf["lat_offset_mm"] - r0["lat_offset_mm"]))
    d = np.array(d)
    return (float(np.median(d)), float(np.percentile(d, 25)), float(np.percentile(d, 75))) if len(d) else (float("nan"),) * 3


def mcnemar(a, b):
    b01 = int(np.sum((a == 1) & (b == 0))); b10 = int(np.sum((a == 0) & (b == 1)))
    n = b01 + b10; k = min(b01, b10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)) if n else 1.0
    return b01, b10, p


def main():
    out = {}
    s20 = list(range(2000, 2020)); s40 = list(range(2000, 2040)); s15 = list(range(2000, 2015))
    CORE = ["policy", "transient", "sustained", "unified", "wrench", "oracle"]
    NAME = {"transient": "capture-spec", "sustained": "hold-spec", "unified": "CoM-unified"}

    print("=== A. Ablation (gate ON): transient 300N latSS falls | sustained floor-corr drift ===")
    print(f"{'controller':14s}{'trans falls':>12s}{'8N drift':>16s}{'12N drift':>16s}")
    out["ablation"] = {}
    for mode in CORE:
        tf = int(falls(mode, 300, s20).sum())
        m8, l8, h8 = sdrift(mode, 8, s20); m12, l12, h12 = sdrift(mode, 12, s20)
        out["ablation"][mode] = {"trans_falls": tf, "8N": [m8, l8, h8], "12N": [m12, l12, h12]}
        print(f"{NAME.get(mode, mode):14s}{tf:>9d}/20{m8:8.0f} [{l8:.0f}-{h8:.0f}]{m12:8.0f} [{l12:.0f}-{h12:.0f}]", flush=True)

    print("\n=== B. 40 paired seeds (gate ON): policy vs wrench ===")
    out["stats"] = {"transient": {}, "sustained": {}}
    for push in (280, 300, 320):
        pol = falls("policy", push, s40); wr = falls("wrench", push, s40)
        b01, b10, p = mcnemar(pol, wr)
        out["stats"]["transient"][push] = [int(pol.sum()), int(wr.sum()), b01, b10, p]
        print(f"  transient {push}N latSS: policy {int(pol.sum())}/40  wrench {int(wr.sum())}/40  "
              f"(pol-only {b01}, wr-only {b10}, McNemar p={p:.4f})", flush=True)
    for F in (8, 12):
        pm = sdrift("policy", F, s40); wm = sdrift("wrench", F, s40); hm = sdrift("sustained", F, s40)
        out["stats"]["sustained"][F] = {"policy": pm, "wrench": wm, "hold_spec": hm}
        print(f"  sustained {F}N drift: policy {pm[0]:.0f}  hold-spec {hm[0]:.0f} [{hm[1]:.0f}-{hm[2]:.0f}]  "
              f"wrench {wm[0]:.0f} [{wm[1]:.0f}-{wm[2]:.0f}]", flush=True)

    print("\n=== C. Envelope (gate ON) ===")
    pushes = [240, 280, 320, 360]; forces = [6, 10, 14, 18, 22]
    ENV = ["policy", "sustained", "wrench"]; ENVN = {"sustained": "hold-spec"}
    out["envelope"] = {"pushes": pushes, "forces": forces, "trans": {}, "sust": {}}
    for mode in ENV:
        tr = [float(falls(mode, p, s15).mean()) for p in pushes]
        su = [sdrift(mode, f, s15)[0] for f in forces]
        out["envelope"]["trans"][mode] = tr; out["envelope"]["sust"][mode] = su
        print(f"  {ENVN.get(mode, mode):10s} trans_fall%={[round(x*100) for x in tr]}  sust_drift={[round(x) for x in su]}", flush=True)

    print("\n=== D. Sustained sensor-bias (gate ON, wrench, 8N, floor-corr) ===")
    out["bias"] = {}
    for b in (-5, 0, 5):
        k, m = kindmode("wrench"); d = []
        for s in s20:
            rf = S.run(k, push_n=8, push_t=3, push_dir=(0, 1), push_dur=3, push_phase="time",
                       duration=8, seed=s, process_noise=SN, id_mode=m, grf_bias=(0.0, b))
            r0 = S.run(k, push_n=1.0, push_t=3, push_dir=(0, 1), push_dur=3, push_phase="time",
                       duration=8, seed=s, process_noise=SN, id_mode=m, grf_bias=(0.0, b), force_override=ZERO)
            if not rf["fell"] and not r0["fell"]: d.append(abs(rf["lat_offset_mm"] - r0["lat_offset_mm"]))
        out["bias"][b] = float(np.median(d)) if d else float("nan")
        print(f"  b_F={b:+d} N: wrench 8N drift = {out['bias'][b]:.0f} mm", flush=True)

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "revalidate_gated.json").write_text(json.dumps(out, indent=2))

    # figure: envelope (gate ON)
    C = {"policy": "#4C4C4C", "hold-spec": "#0072B2", "wrench": "#D55E00"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for mode in ENV:
        nm = ENVN.get(mode, mode)
        ax[0].plot(pushes, np.array(out["envelope"]["trans"][mode]) * 100, "-o",
                   color=C.get(nm, "#888"), label=nm, lw=2)
        ax[1].plot(forces, out["envelope"]["sust"][mode], "-o", color=C.get(nm, "#888"), label=nm, lw=2)
    ax[0].set_title("Transient push — lateral SS"); ax[0].set_xlabel("push [N]"); ax[0].set_ylabel("fall rate [%]")
    ax[0].set_ylim(-5, 105); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].set_title("Sustained force — floor-corrected drift"); ax[1].set_xlabel("force [N]")
    ax[1].set_ylabel("force-induced lateral drift [mm]"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    fig.suptitle("Wrench-unified with capture gate — recovery envelopes (paper-ready)")
    fig.tight_layout(); fig.savefig(HERE / "figures" / "wrench_envelope_gated.png", dpi=160)
    print("saved results/revalidate_gated.json, figures/wrench_envelope_gated.png")


if __name__ == "__main__":
    main()
