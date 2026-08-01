"""Step V4 — statistical strength: 40 paired seeds at the failure boundary.

Policy vs wrench-unified, same seeds (paired process noise, gait init, phase,
magnitude). Transient: fall counts + McNemar paired test on discordant pairs.
Sustained: median offset + IQR.
"""
import json
from pathlib import Path
import numpy as np
import stage2_id_on_policy as S

HERE = Path(__file__).resolve().parent
SEEDS = list(range(2000, 2040))          # 40 paired seeds (held out from dev 1000-1009)


def transient_outcomes(mode, push):
    return np.array([int(S.run("policy" if mode == "policy" else "id_mpc",
                    push_n=push, push_dir=(0, 1), push_phase="SS", seed=s,
                    process_noise=6, id_mode=mode)["fell"]) for s in SEEDS])


def sustained_offsets(mode, force):
    v = []
    for s in SEEDS:
        r = S.run("policy" if mode == "policy" else "id_mpc", push_n=force, push_t=3.0,
                  push_dir=(0, 1), push_dur=3.0, push_phase="time", duration=8.0,
                  seed=s, process_noise=4, id_mode=mode)
        v.append(np.nan if r["fell"] else abs(r["lat_offset_mm"]))
    return np.array(v)


def mcnemar(a, b):
    # a,b: paired fall indicators (1=fell). discordant: policy fell & unified survived (b01)
    b01 = int(np.sum((a == 1) & (b == 0)))   # policy fell, wrench survived (favors wrench)
    b10 = int(np.sum((a == 0) & (b == 1)))   # policy survived, wrench fell
    n = b01 + b10
    # exact binomial two-sided p under H0 p=0.5
    from math import comb
    k = min(b01, b10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)) if n > 0 else 1.0
    return b01, b10, p


def main():
    out = {"n_seeds": len(SEEDS)}
    print(f"=== Transient (lateral SS, {len(SEEDS)} paired seeds) ===")
    print(f"{'push':>6}{'policy falls':>14}{'wrench falls':>14}"
          f"{'pol_only':>10}{'wr_only':>9}{'McNemar p':>11}")
    out["transient"] = {}
    for push in (280, 300, 320):
        pol = transient_outcomes("policy", push)
        wr = transient_outcomes("wrench", push)
        b01, b10, p = mcnemar(pol, wr)
        out["transient"][push] = {"policy_falls": int(pol.sum()), "wrench_falls": int(wr.sum()),
                                  "policy_only_fall": b01, "wrench_only_fall": b10, "mcnemar_p": p}
        print(f"{push:>6}{int(pol.sum()):>10}/{len(SEEDS):<3}{int(wr.sum()):>10}/{len(SEEDS):<3}"
              f"{b01:>10}{b10:>9}{p:>11.4f}", flush=True)

    print(f"\n=== Sustained ({len(SEEDS)} paired seeds; offset mm median [IQR]) ===")
    print(f"{'force':>6}{'policy':>22}{'wrench':>22}")
    out["sustained"] = {}
    for force in (8, 12):
        pol = sustained_offsets("policy", force); wr = sustained_offsets("wrench", force)
        def q(v): v = v[~np.isnan(v)]; return (np.median(v), np.percentile(v, 25), np.percentile(v, 75))
        pm, pl, ph = q(pol); wm, wl, wh = q(wr)
        out["sustained"][force] = {"policy_med": pm, "policy_iqr": [pl, ph],
                                   "wrench_med": wm, "wrench_iqr": [wl, wh]}
        print(f"{force:>6}{pm:>10.0f} [{pl:.0f}-{ph:.0f}]{'':>4}{wm:>10.0f} [{wl:.0f}-{wh:.0f}]", flush=True)

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "stats_validation.json").write_text(json.dumps(out, indent=2))
    print("saved results/stats_validation.json")


if __name__ == "__main__":
    main()
