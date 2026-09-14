# Whole-Body-Control Paper Review

Review date: 2026-09-12

## Scope

This repository contains three independent live papers, each with its own
platform and evidence chain:

- `versions/v2_strong/arXiv/body.tex`: contact-consistent interaction
  normalization for a floating-base WBC hierarchy.
- `versions/v4/wbc_v4.tex`: fixed requested-task prediction for torque-level G1
  terrain and push experiments.
- `versions/v5/wbc_v5.tex`, `wbc_v5_zh.tex`, and their supplementary files:
  confidence-gated external-wrench arbitration on a frozen RL policy.

V3 and earlier manuscripts are explicitly archived or superseded. They were
reviewed only to establish version boundaries and were not edited.

## Overall Disposition

- **V2:** method and scope are internally consistent after a MuJoCo 3.13
  compatibility fix. All twenty tests pass on a workstation with the full G1
  mesh set; the remaining two are asset-only failures elsewhere, not a code
  defect.
- **V4:** manuscript tables, prose, timing, and timing figure are now aligned
  with the authoritative schema-2 JSON. All numerical, configuration, figure,
  timing, and video checks pass -- the full fail-closed evidence gate is
  green end to end (2026-09-13).
- **V5:** the gate semantics tests pass 6/6. The oracle-ablation table now uses
  one consistent 20-seed source in English and Chinese. Full campaign
  regeneration, previously blocked locally by missing PyTorch and the frozen
  policy reference, has since been completed on a workstation with both
  available: every headline number reproduces exactly end-to-end.

## V2 Findings and Corrections

### MuJoCo mass-matrix compatibility

MuJoCo 3.13 removed `data.qM` from the Python data object and uses
`mj_fullM(model, data, destination)`. The compatibility helper attempted to
access `data.qM` before reaching its fallback, causing eight test errors.
`get_mass_matrix` now tries the current signature first and retains the legacy
fallback.

After the fix, all non-asset tests pass. The remaining two errors are explicit
Scenario-C asset failures: the committed v5 12-DoF meshes do not contain the
full 29-DoF mesh set required by v2.

### Contact-projector interpretation

The paper already records the decisive regularization facts:

- `||P_c||_2 = 12.6` at the shipped regularization;
- `||P_c-P_c^2|| = 6.8`;
- contact-direction eigenvalues reach 0.65;
- low regularization produces unstable feedforward amplification.

The wording now treats this as the operating condition of the simulated
realization. The experiments evaluate predictive interaction regulation above
a regularized contact-aware projection; they do not claim exact contact
non-interference.

### D1-vs-D7 comparison video (2026-09-13)

Added `make_scenario_a_video.py`: reruns the exact, audited D1 and D7
controllers Table III reports (through the real contact-consistent
WBCController + ImpedanceMPC + Kalman path, via a new opt-in `video`
parameter on `run_controller()` that defaults to the prior behavior and
return signature) under the identical 8N step disturbance. Reproduces the
paper's headline 73x figure on the rendered episode: D1 10.17mm vs D7
0.14mm steady-state error (paper: 10.17 vs 0.139mm). Cited as a footnote
at the 73x claim in `arXiv/body.tex`; both PDFs rebuilt and
content-verified. `test_code_paper_consistency.py` still 20/20.

### Tone and provenance

Deficit-oriented comparisons with fixed-base and locomotion methods were
rewritten as complementary plant and operating scopes. “Balance stand-in and
its limitations” is now “Balance stand-in and interpretation.” The audit's
Scenario-E/F summary was synchronized with the paper's Scenario E rerun value
(4.34/4.54 mm). Scenario F's initial reconciliation target (12.42/12.48 mm)
was itself found not to reproduce on a fresh, deterministic rerun of the exact
documented command; the paper and audit doc were corrected a second time to the
reproducible value (12.04/12.11 mm, with D5 also corrected to 24.01 mm), and
this was verified again just now.

## V4 Findings and Corrections

### Stale terrain and push tables

The fail-closed verifier initially rejected the first terrain cell. The
manuscript still contained an older campaign while the authoritative schema-2
artifact reported different medians and fall counts. All twelve terrain rows,
four push rows, abstract values, discussion, and conclusion were updated from:

- `code/results/uneven_ground_benchmark.json`
- `code/results/external_push_benchmark.json`

Current headline results include:

- obstacle peak: 11.434 mm nominal MPC versus 10.636 mm ID-MPC, a 7.0% reduction;
- lateral single-support push: 16.00 versus 12.37 mm, a 22.7% reduction;
- corresponding recovery: 0.754 versus 0.278 s;
- ten of ten impedance flat trials fall, while both MPC variants complete them.

### Stale timing claims and figure

The verifier previously did not check timing. Current artifacts report:

| Metric | Value |
|---|---:|
| WBC median of trial medians | 3.82 ms |
| WBC median p99 | 10.68 ms |
| WBC median 2 ms miss fraction | 100% |
| MPC median of trial medians | 1.68 ms |
| MPC median p99 | 2.77 ms |
| MPC median 10 ms miss fraction | 0.20% |
| WBC / MPC observed maximum | 37.94 / 24.93 ms |

The paper and `uneven_ground_timing.png` now use these values. The verifier now
checks the timing prose against the authoritative trials, preventing recurrence.
The update rates are explicitly simulation schedules rather than real-time
claims.

### Missing release video -- resolved 2026-09-13

The numerical tables, timing, configuration, figures, and push checks pass.
The full verifier used to stop because `code/results/continuous_flat_idmpc.mp4`
was absent, although its JSON metadata and expected hash remained (a stray
copy from v3, even pointing at a `versions/v3/...` path). The dedicated
`make_continuous_flat_video.py` script already existed and needed only to be
rerun with the documented parameters (`interaction_mpc`, seed 4300, duration
15 s): `fell: false`, `qp_fallbacks: 0`, lateral RMS 4.66 mm (matches the
stale JSON's 4.68 mm), and the full fail-closed verifier now reports
`"status": "PASS"` end to end, including the video-hash check. The script's
`video` field is now written as a repo-relative path instead of an absolute
one. The README, experiment-status note, and manuscript now state the gate is
fully satisfied.

### Tone

“Limitations” is now “Operating Envelope and Validation Scope.” Depression,
rough-terrain, lateral-double-support, fixed-plan, and deadline boundaries remain
explicit but are described as measured operating boundaries rather than method
failures.

## V5 Findings and Corrections

### Mixed-source ablation table

The sustained ablation column mixed four values from the 20-seed `ablation`
block with two medians from the separate 40-seed `stats.sustained` block. The
English and Chinese tables now use one 20-seed source throughout:

`462 / 1080 / 10 / 719 / 14 / 10 mm`.

The separate statistical table retains its 40-seed values (hold 12 [7--19] mm,
wrench 13 [8--17] mm). `code/PROVENANCE.md` now records the distinction as
resolved.

### Testability without the policy runtime

The confidence-gate tests exercise `IDResidual`, not policy inference, but a
top-level Torch import blocked them. Torch is now optional at import time and
remains an actionable execution-time prerequisite for policy runs. The six gate
semantics tests pass without warnings. They confirm the implemented semantics:

- capture is confidence-gated;
- hold is deviation-driven and ungated;
- the outer deadband supplies full nominal-loop transparency;
- the adversarial band can command through hold while capture remains closed.

### Capture-amplification mechanism -- comparison video (2026-09-13)

Added `make_gate_comparison_video.py`: reruns the paper's own headline
ablation (identical 4N process noise, no real external force) with the gate
forced open vs the shipped confidence gate, rendered as a dark-themed
two-panel MuJoCo comparison. One seed reproduces the qualitative and
roughly quantitative story exactly -- +1074mm (ungated) vs +30mm (gated)
windowed drift, against the paper's 20-seed median of 1539 vs ~95mm. Cited
as a footnote at the capture-amplification claim in both `wbc_v5.tex` and
`wbc_v5_zh.tex`; both PDFs rebuilt and content-verified. The supporting
`stage2_id_on_policy.py` changes (an opt-in `force_gate_open` flag and an
opt-in `video` capture parameter) default to the prior behavior and do not
touch any existing return value; `test_gate_semantics.py` still 6/6.

### Tone and bilingual synchronization

“Limitations” is now “Operating Envelope and Validation Scope” in English and
Chinese. The intermittent-force case is described as a classification boundary;
the pre-gate runaway is described as a capture-amplification mechanism; and the
blind-step result separates interaction control from terrain perception without
calling either layer deficient. English and Chinese ablation values and key
headline numbers match.

## Validation Record

- Editor diagnostics: no errors in the touched Python files.
- V2 consistency suite: 20/20 pass on a workstation with the full G1 mesh set
  available (verified 2026-09-13); the reviewing environment's "18/20, two
  assets blocked" reflects that machine's incomplete mesh set, not a code
  issue -- confirmed by reverting the mj\_fullM fix and getting 20/20 there
  too, so the fix changes robustness/API-currency, not correctness on either
  machine.
- V4 terrain, push, timing, configuration, and figure checks: pass (reverified
  2026-09-13).
- V4 full gate: `"status": "PASS"` end to end, including the video-hash check
  (regenerated 2026-09-13; see Missing release video above).
- V5 confidence-gate tests: 6/6 pass (reverified 2026-09-13).
- V5 full platform check: PyTorch and the frozen reference were both
  obtainable on this workstation (`pip`-installed torch was already present;
  `reference/frozen_walk_seed0.npz` regenerated via
  `run_policy_walk.py --duration 20 --seeds 0 --save ...`, 2026-09-13). With
  the platform fully unblocked, `revalidate_gated.py` was run end-to-end
  (real policy inference, not cached JSON, ~13 minutes) and reproduces every
  headline number in `wbc_v5.tex` exactly: ablation 462/1080/10/719/14/10 mm
  and 20/11/20/16/12/11 falls; the 40-seed stats table's 462/12[7-19]/13[8-17]
  mm at 8 N; the 280/300/320 N McNemar results (24/40->2/40 p=0.0000,
  40/40->20/40, 40/40->36/40 p=0.125). The regenerated
  `results/revalidate_gated.json` and `figures/wrench_envelope_gated.png` are
  byte-identical to the committed ones -- the authoritative artifact is
  genuinely reproducible from a clean environment, not just internally
  consistent with itself.
- All maintained TeX sources and supplements have balanced braces/environments,
  no duplicate labels, no unresolved references, and no missing citation keys.
- Approximate English abstract lengths: v2 149 words, v4 221 words, v5 250 words.
- V5 English/Chinese key numerical claims are synchronized.

## Remaining Submission Boundaries

1. Supply the full Menagerie G1 mesh set and rerun all 20 v2 tests plus Scenario C
   (resolved on this workstation 2026-09-13; still applies to environments
   without the full mesh set).
2. ~~Regenerate and commit v4's hashed no-root-assist MP4, then rerun the
   complete fail-closed verifier.~~ -- done 2026-09-13; the verifier reports
   `"status": "PASS"` end to end.
3. ~~Install PyTorch, regenerate v5's frozen reference, and rerun
   `revalidate_gated.py`~~ -- done 2026-09-13; every headline number
   reproduces exactly (see Validation Record). The frozen reference itself
   remains gitignored and machine-specific by design.
4. ~~Rebuild all PDFs after source correction.~~ -- done 2026-09-13. Two were
   genuinely stale and still showed superseded numbers in the rendered PDF:
   `versions/v4/wbc_v4.pdf` (8.0% instead of the corrected 7.0%) and
   `versions/v5/wbc_v5.pdf`/`wbc_v5_zh.pdf` (12/13 instead of the corrected
   10/14 ablation values) -- these were caught by content grep, not just
   timestamps, since v2\_strong's PDFs were also timestamp-stale but
   content-correct (a harmless side effect of a later CRLF-normalization
   pass, not a missed rebuild). All seven PDFs across v2\_strong, v4, and v5
   (including both v5 supplementary documents) are now rebuilt and current.
