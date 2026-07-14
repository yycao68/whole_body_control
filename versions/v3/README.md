# Whole-Body Control V3

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
