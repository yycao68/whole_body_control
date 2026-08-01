"""Step V2 — oracle ablation.

Six controllers on the SAME transient-push and sustained-force tests:
  policy / capture-specialist / hold-specialist / CoM-only-unified /
  wrench-unified / oracle-unified (knows the true class at onset, no delay).
The oracle is the upper bound; oracle->wrench gap = cost of causal wrench detection.
"""
import json
from pathlib import Path
import numpy as np
import stage2_id_on_policy as S

HERE = Path(__file__).resolve().parent
SEEDS = list(range(1000, 1010))
CTRL = [("policy", "policy", "transient"),
        ("capture-specialist", "id_mpc", "transient"),
        ("hold-specialist", "id_mpc", "sustained"),
        ("CoM-only-unified", "id_mpc", "unified"),
        ("wrench-unified", "id_mpc", "wrench"),
        ("oracle-unified", "id_mpc", "oracle")]


def transient_falls(kind, mode, push=300):
    return sum(int(S.run(kind, push_n=push, push_dir=(0, 1), push_phase="SS",
                         seed=s, process_noise=6, id_mode=mode)["fell"]) for s in SEEDS)


def sustained_offset(kind, mode, force=8):
    offs = []
    for s in SEEDS:
        r = S.run(kind, push_n=force, push_t=3.0, push_dir=(0, 1), push_dur=3.0,
                  push_phase="time", duration=8.0, seed=s, process_noise=4, id_mode=mode)
        if not r["fell"]:
            offs.append(abs(r["lat_offset_mm"]))
    return float(np.median(offs)) if offs else float("nan")


def main():
    out = {}
    print(f"{'controller':20s}{'transient 300N latSS falls':>28s}{'sustained 8N offset(mm)':>26s}")
    for label, kind, mode in CTRL:
        tf = transient_falls(kind, mode)
        so = sustained_offset(kind, mode)
        out[label] = {"transient_300N_latSS_falls": tf, "sustained_8N_offset_mm": so}
        print(f"{label:20s}{tf:>20d}/10{so:>24.0f}", flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "oracle_ablation.json").write_text(json.dumps(out, indent=2))
    print("saved results/oracle_ablation.json")


if __name__ == "__main__":
    main()
