# v5 — Interaction Dynamics: a Confidence-Gated External-Wrench Layer

The new-direction work: a fast interaction-dynamics layer built on top of a **frozen**
Unitree G1 RL locomotion policy. Interaction is treated as a residual on a
configuration-invariant task double integrator, discriminated by wrench persistence,
and arbitrated by a self-calibrating confidence gate.

(The original DCM / ID-MPC terrain+push paper is `../v3/`, a separate conference candidate.)

## Contents

- `wbc_ieee_v5.md` — the manuscript (paper draft). `wbc_ieee_v5.pdf` is a pandoc render.
- `PAPER_REDESIGN.md` — paper contract, figure plan, claim matrix (C1–C11).
- `Interaction_Dynamics_Change_Direction_Plan.md` — the two-stage change-of-direction plan + V1–V6 validation.
- `code/` — implementation and experiments (see `code/README.md`).

The tex (`wbc_v5.tex`) is not yet written; it will be synced from the markdown after review.
