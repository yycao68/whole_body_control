# V5 result provenance

Which script and which committed artifact produces each number in `wbc_v5.tex`.

Added after an external review (2026-08-30) reported that the reported effect
size "is not traceable to one immutable experimental protocol". Every mapping
below was verified by re-reading the artifacts and, where noted, by re-running
the generator.

## Authoritative artifact

**`revalidate_gated.py` → `results/revalidate_gated.json` is the source of every
number in the paper's result tables and figures.** Its docstring says so
("Paper-ready re-run of the affected studies WITH the capture gate"), but the
manuscript never named it. It is the only generator that produces all four
reported studies under one protocol, and the only one that emits the
`hold_spec` (hold-specialist) column the ablation and sustained tables use.

| Paper location | Claim | Source |
| --- | --- | --- |
| Abstract, Table `tab:stats` (transient) | 280 N: 24/40 → 2/40, $p<10^{-4}$, 0 worse; 300 N: 40/40 → 20/40; 320 N: 40/40 → 36/40, $p=0.125$ | `revalidate_gated.json` `stats.transient` — exact match on all 12 cells |
| Table `tab:stats` (sustained) | 8 N: 462 / 12 [7–19] / **13 [8–17]**; 12 N: 728 / 121 [114–132] / 202 [186–218] | `revalidate_gated.json` `stats.sustained` — exact match on all 6 cells |
| Table `tab:ablation`, transient column | 20 / 11 / 20 / 16 / 12 / 11 falls | `revalidate_gated.json` `ablation.*.trans_falls` — exact match |
| Table `tab:ablation`, sustained column | 462 / 1080 / 12 / 719 / 13 / 12 mm | **Mixed** — see "Known inconsistency" below |
| §V5 sensing robustness | drift 10/14/22 mm at $b_F=-5/0/+5$ N | `revalidate_gated.json` `bias` = {−5: 10.41, 0: 13.83, +5: 21.57} |
| §V5 nominal protection | base roll 6.8–6.9° across all bias/noise | `sensorbias_validation.json` `bias_sweep.*.nominal_roll` = 6.8634 (constant) |
| Fig. `fig:envelope` | 0% single-support falls through 280 N; policy already 60–87% | `revalidate_gated.json` `envelope.trans`: wrench [0.0, 0.0, …] at 240/280 N; policy [0.867, 0.600, …] |
| Table `tab:estimator` | 0.1 N residual, 0% false positive, ≤2 ms detect, 0.19 s decay, 1 N tracking | `estimator_validation.json` |

## Protocol differences that matter

Two different sustained metrics exist in this repository. They are not
interchangeable, and the paper uses the second:

* **Raw offset** (`validate_stats.py`, `validate_sensorbias.py`):
  `process_noise=4`, reports `|lat_offset_mm|` directly.
* **Floor-corrected paired drift** (`revalidate_gated.py`, `resustained.py`):
  `process_noise=1` (`SN`), and subtracts a *matched same-seed zero-force run*
  so gait-sway and reference phase drift cancel. This is what the paper
  reports and what §V4 describes ("the offset with force minus the same-seed
  offset without force").

The floor-corrected metric is the defensible one, but the difference is large,
so reading the wrong artifact is misleading. Re-running `validate_sensorbias.py`
today gives 14.8 / 15.4 / 85.3 mm at $b_F=-5/0/+5$ — nowhere near the paper's
10/14/22, because it is the *raw* metric, not because either is wrong.

## Superseded artifacts

These are committed but are **not** the source of any paper number. They
predate the capture gate and/or use the raw sustained metric:

* `oracle_ablation.json` — pre-gate; its policy row records 10 transient falls
  against the paper's 20, on a different seed count.
* `sensorbias_validation.json` — raw metric; its sustained column
  (14.8/20.2/25.7, and 394 mm at $b_F=+3$) is not the reported 10/14/22.
* `sustained_floorcorrected.json` — superseded by `revalidate_gated.json`'s
  `stats.sustained`.

`stats_validation.json` **was** in this category and has been regenerated. The
previously committed copy was written before the capture gate was added and
recorded 6/40 wrench falls, 2 trials made worse, and $p=1.211\times10^{-4}$ at
280 N — which contradicted the paper and, on the $p$ value, was not actually
below the $10^{-4}$ the abstract claims. Re-running the *same, documented*
`validate_stats.py` against current code reproduces the paper exactly
(24/40 → 2/40, 0 worse, $p=4.77\times10^{-7}$). The artifact was stale, not the
paper.

## Known inconsistency (unresolved)

`tab:ablation`'s **sustained** column is not drawn from a single block.
Four of six cells match `ablation.*` exactly (policy 462, capture 1080,
CoM-only 719); but hold-specialist (12) and wrench-gated (13) match the
40-seed `stats.sustained` block (11.62, 13.43) rather than the 20-seed
`ablation` block (10.41, 13.83). The transient column in the same table *is*
from `ablation`. Every value is real and defensible; the table simply mixes a
20-seed and a 40-seed source without saying so. Either relabel the column or
regenerate it from one block.

Note that `ablation.sustained` and `ablation.oracle` carry byte-identical 8 N
and 12 N arrays. This is correct by construction, not aliasing: the sustained
test uses `push_dur=3`, so the oracle's `p_oracle` is 1.0 and the oracle
reduces to pure hold — i.e. exactly the hold specialist. Their `trans_falls`
(20 vs 11) still differ, as expected.

## Reproducing

```bash
cd code
python3 check_platform.py        # verify meshes, reference, policy, packages
python3 revalidate_gated.py      # regenerates the authoritative artifact
python3 validate_stats.py        # V4 cross-check (transient must match)
python3 -m unittest test_gate_semantics.py
```

Runtime is roughly 0.6 s per simulated trial on an M-series Mac;
`validate_stats.py` is about 4 minutes.
