"""Step V3 — held-out tuning check.

The wrench-unified hyperparameters (f_thresh=3, tf0=0.4, decay, blend, gains)
were set on ~0.15 s impulses and 8/12/16 N forces. Here they are FROZEN and
evaluated on held-out cases never used for tuning:
  A. impulse DURATIONS 0.10/0.15/0.25/0.40 s (stresses the persistence timer),
  B. sustained MAGNITUDES interpolated off the tuning grid (6/10/14 N),
  C. RAMPED force (gradual onset, 1 s ramp to 12 N),
  D. INTERMITTENT force (12 N, 0.5 s on / 0.5 s off).
Wrench-unified is compared to policy and the regime specialists as reference.
"""
import json
from pathlib import Path
import numpy as np
import stage2_id_on_policy as S

HERE = Path(__file__).resolve().parent
SEEDS = list(range(1000, 1010))


def falls(mode, push, dur, phase="SS", noise=6):
    return sum(int(S.run("policy" if mode == "policy" else "id_mpc",
                         push_n=push, push_dir=(0, 1), push_dur=dur, push_phase=phase,
                         seed=s, process_noise=noise, id_mode=mode)["fell"]) for s in SEEDS)


def offset(mode, force_fn, noise=4, dur_win=3.0):
    offs = []
    for s in SEEDS:
        r = S.run("policy" if mode == "policy" else "id_mpc", push_n=1.0, push_t=3.0,
                  push_dir=(0, 1), push_dur=dur_win, push_phase="time", duration=8.0,
                  seed=s, process_noise=noise, id_mode=mode, force_override=force_fn)
        if not r["fell"]:
            offs.append(abs(r["lat_offset_mm"]))
    return float(np.median(offs)) if offs else float("nan")


def main():
    out = {}
    modes = [("policy", "policy"), ("capture-spec", "transient"),
             ("hold-spec", "sustained"), ("wrench", "wrench")]

    print("A. HELD-OUT impulse durations (200 N lateral SS, falls/10)")
    print(f"  {'controller':14s}" + "".join(f"{d:>8}s" for d in (0.10, 0.15, 0.25, 0.40)))
    out["impulse_duration"] = {}
    for label, mode in modes:
        row = [falls(mode, 200, d) for d in (0.10, 0.15, 0.25, 0.40)]
        out["impulse_duration"][label] = row
        print(f"  {label:14s}" + "".join(f"{v:>7d} " for v in row), flush=True)

    print("B. HELD-OUT sustained magnitudes (offset mm)")
    print(f"  {'controller':14s}" + "".join(f"{f:>7}N" for f in (6, 10, 14)))
    out["sustained_mag"] = {}
    for label, mode in modes:
        row = [offset(mode, (lambda t, F=F: (0.0, F if t < 3.0 else 0.0))) for F in (6, 10, 14)]
        out["sustained_mag"][label] = row
        print(f"  {label:14s}" + "".join(f"{v:>7.0f} " for v in row), flush=True)

    print("C. HELD-OUT ramped force (0->12 N over 1 s, held to 3 s; offset mm)")
    ramp = lambda t: (0.0, 12.0 * min(max(t / 1.0, 0.0), 1.0) if t < 3.0 else 0.0)
    out["ramped_12N"] = {}
    for label, mode in modes:
        v = offset(mode, ramp)
        out["ramped_12N"][label] = v
        print(f"  {label:14s}{v:7.0f} mm", flush=True)

    print("D. HELD-OUT intermittent force (12 N, 0.5 s on/off, over 3 s; offset mm)")
    interm = lambda t: (0.0, 12.0 if (t < 3.0 and int(t / 0.5) % 2 == 0) else 0.0)
    out["intermittent_12N"] = {}
    for label, mode in modes:
        v = offset(mode, interm)
        out["intermittent_12N"][label] = v
        print(f"  {label:14s}{v:7.0f} mm", flush=True)

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "heldout_validation.json").write_text(json.dumps(out, indent=2))
    print("saved results/heldout_validation.json")


if __name__ == "__main__":
    main()
