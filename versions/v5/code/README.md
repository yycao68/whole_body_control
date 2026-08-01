# v5 code — Interaction Dynamics layer on a frozen Unitree G1 policy

Everything lives flat in this directory.

## Core

- `run_policy_walk.py` — the frozen RL policy loop (platform; do not modify).
- `stage2_id_on_policy.py` — the interaction layer: wrench estimator + confidence
  gate + capture/hold persistence blend (`IDResidual`).
- `stage2_wbc_track.py` — the WBC-QP realizer (reported negative result).
- `normalized_mpc.py`, `interaction_estimator.py` — dependencies imported by `stage2_id_on_policy.py`.

## Validation campaign

- `validate_estimator.py` (V1), `validate_oracle_ablation.py` (V2),
  `validate_heldout.py` (V3), `validate_stats.py` (V4, McNemar),
  `validate_sensorbias.py` (V5), `validate_observer.py` (V6).
- `revalidate_gated.py`, `resustained.py`, `wrench_envelope_sweep.py` — gated re-runs and envelope sweeps.

## Figures & docs

- `make_architecture.py`, `make_schematic.py` — paper figures (Fig. 1, Fig. 2) → `figures/`.
- `STAGE2_FINDINGS.md` — full findings log; `FROZEN_PLATFORM.md` — frozen-platform record.

## Assets

- `motion.pt`, `g1_description/`, `configs/`, `reference/` — policy, robot model, config, frozen reference.
- `results/`, `figures/` — experiment outputs.
