# Whole-Body Control V3 — SUPERSEDED, ARCHIVED

> **This version is superseded by [`../v4/`](../v4/) and is retained for
> history only.** Use v4 for the current manuscript, code, and results.
>
> An external review on 2026-08-30 confirmed several defects in this version.
> Only the one that also affected v4 was fixed (portable G1 mesh resolution,
> so the torque benchmarks run from a clean checkout). The rest are recorded
> here and left unfixed by decision, because v4 already addresses the
> substantive ones independently:
>
> | v3 defect (confirmed) | Status in v4 |
> | --- | --- |
> | Torque benchmarks unrunnable from a fresh clone (`*.STL` is gitignored, so `models/assets/` is empty) | Same defect; **fixed in both** |
> | Default benchmark runs 4 controllers / 160 push trials; paper describes 3 / 120 | Fixed: defaults to the 3 published controllers, diagnostics opt-in |
> | Obstacle lane present from the initial pose, so the obstacle condition can include pre-evaluation contact adaptation | Fixed: patch spans x=0.22–0.34 m with a hard check rejecting patch contact during settling |
> | `sync_markdown_to_tex.py` fails (`KeyError: 'Locomotion Interaction Dynamics'`), so `wbc_v3.tex` cannot be regenerated from its source `wbc_ieee.md` | Works: `wbc_v4.tex` regenerates byte-identically from `wbc_ieee_v4.md` |
> | Conclusion says "provably fixed across gait phase, terrain, and push" without the abstract's "under one modeling assumption" qualifier | Not present; v4 uses more careful wording throughout |
> | 11.13 pt overfull `\hbox` in the results table | Not present (0 overfull boxes) |
> | README below is stale: `verify_v3_artifacts.py`, `BENCHMARK_PLAN.md`, and `unitree_locomotion_demo/` do not exist; `code/` is not self-contained | v4's README references only files that exist |
>
> The verifier (`code/verify_interaction_paper_claims.py`) targets
> `wbc_ieee.md` rather than `wbc_v3.tex`. That is *by design* — the Markdown
> is the editing source and the TeX is generated from it — but because this
> version's sync script is broken, that derivation cannot currently be
> reproduced here.

**Historical description follows.**

V3 develops a dual interaction-dynamics architecture:

- Level 1: centroidal double-integrator MPC;
- Level 2: instantaneous whole-body physical interface;
- Level 3: task double-integrator MPC.

## Contents

- `wbc_ieee.md`: new manuscript draft.
- `BENCHMARK_PLAN.md`: detailed Unitree G1 comparison protocol.
- `code/`: self-contained v3 implementation package and local G1 model.
- `unitree_locomotion_demo/`: demo-only plan and video composer for showing the
  interaction-dynamics layer on top of a validated Unitree locomotion base.

## Status

The formulation, figures, and current H1--H6 simulation artifacts are
implemented under `code/`. The implemented artifact set can be checked with:

```bash
cd code
python3 verify_v3_artifacts.py
```

The walking/running material is deliberately separated. The paper does not
claim to solve dynamic walking; walking/running demos should use an existing
Unitree locomotion stack as the base controller and place the interaction layer
on top.

## Demo Package

The demo package is for videos, not paper evidence:

```bash
cd unitree_locomotion_demo
python3 scripts/compose_demo_video.py \
  --manifest demo_manifest.json \
  --use-placeholders \
  --out results/placeholder_storyboard.mp4
```

Replace the placeholder clips in `videos/` with official Unitree locomotion
clips when available, then run the same composer without `--use-placeholders`.
