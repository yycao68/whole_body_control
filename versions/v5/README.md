# v5 — Interaction Dynamics: a Confidence-Gated External-Wrench Layer

The new-direction work: a fast interaction-dynamics layer built on top of a **frozen**
Unitree G1 RL locomotion policy. Interaction is treated as a residual on a
configuration-invariant task double integrator, discriminated by wrench persistence,
and arbitrated by a self-calibrating confidence gate.

(The original DCM / ID-MPC terrain+push paper is `../v3/`, a separate conference candidate.)

## Contents

- **`wbc_v5.tex` — the canonical manuscript.** Built with `latexmk -pdf wbc_v5.tex`.
  `wbc_v5_supplementary.tex` is its supplement; `*_zh.tex` are Chinese translations.
- `PAPER_REDESIGN.md` — paper contract, figure plan, claim matrix (C1–C11).
- `Interaction_Dynamics_Change_Direction_Plan.md` — the two-stage change-of-direction plan + V1–V6 validation.
- `code/` — implementation and experiments (see `code/README.md`).
- `code/PROVENANCE.md` — which script and artifact produces each reported number.

## Reproducing

```bash
cd code
python3 check_platform.py     # reports every missing prerequisite at once
```

The G1 meshes are committed (a deliberate `.gitignore` exception -- three of the
27 have no public equivalent, so they cannot be auto-fetched). The frozen nominal
reference is gitignored (`*.npz`) because it is regenerable;
`check_platform.py` prints the exact command to rebuild it. See `code/ASSETS.md`.

Building the TeX needs a LaTeX install providing
`IEEEtran`, `algorithmicx`/`algpseudocode`, `amsmath`, `amssymb`, `booktabs`,
`graphicx`, and `hyperref` (TeX Live `collection-latexrecommended`; a minimal
install may need `tlmgr install algorithmicx IEEEtran booktabs`).
