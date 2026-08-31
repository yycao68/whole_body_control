# Interaction Dynamics for Floating-Base and Humanoid Robots

Research repository behind a series of papers on treating physical interaction
as a **residual on a configuration-invariant predictive model**, rather than as
a disturbance to be suppressed or a full contact model to be re-derived per
configuration.

Each paper lives in its own directory under `versions/`. They are not drafts of
one document: v2_strong, v4 and v5 are **separate papers** with different
platforms, claims, and target venues. Read the table before picking one.

## Which version is which

| Directory | Paper | Platform | Status |
| --- | --- | --- | --- |
| `versions/v5/` | *Interaction Dynamics: A Physical-Interaction Reasoning Layer for Humanoid Robots* | Unitree G1 + **frozen RL walking policy** | **Live** (RA-L target; `cover_letter_RAL.txt`) |
| `versions/v4/` | *Interaction Dynamics: A Configuration-Invariant Predictive Model for Humanoid Locomotion under Terrain and External Disturbances* | Torque-level G1, DCM/ID-MPC | **Live**; supersedes v3 |
| `versions/v3/` | *Interaction Dynamics: A Fixed-Model Predictive Framework for Physical Interaction in Humanoid Locomotion* | Torque-level G1, DCM/ID-MPC | **Archived** — see caveat below |
| `versions/v2_strong/` | *Contact-Consistent Interaction Dynamics Normalization for Predictive Physical Human--Robot Interaction* | 17-DOF biped + Menagerie G1, WBC/OSC | Live |
| `versions/v1_honest/`, `versions/v0_original/` | early drafts | — | Superseded; kept for history |

**v3 is archived, not disposable.** v4 supersedes it *as a manuscript*, but the
multirate/authority line of work (`multirate_control.py`,
`realization_authority.py`, `run_multirate_benchmarks.py`,
`run_authority_*.py`) exists **only** in v3 — 17 modules with no v4 counterpart
— and `CORRECTED_EXPERIMENT_STATUS.md` records data whose generating code was
destroyed by a `git checkout`/`reset` and is unrecoverable. `versions/v3/README.md`
carries a superseded header tabulating its known defects against v4's status.

## The common idea

Interaction (an external push, a sustained lean, terrain mismatch, realization
error) is lumped into one residual acting on a task-space model whose discrete
transition and input matrices are **independent of configuration and contact
mode**. Robot- and contact-dependent quantities move into force recovery,
constraints, and the realizer — not into the predictor. A small fixed-structure
QP then regulates that model.

The versions differ in *where* the layer sits: above a whole-body
inverse-dynamics stack (v2_strong), above a DCM/capture-point locomotion
controller with a torque-level ID/contact QP (v3, v4), or as a velocity-command
bias on a frozen learned policy (v5).

## Reproducing

Each version's `code/` is self-contained apart from assets noted below. Start
from the version's own README (`versions/v3|v4|v5/README.md`) or, for
v2_strong, `versions/v2_strong/CODE_PAPER_AUDIT.md`.

```bash
# v2_strong — WBC/OSC pHRI scenarios
cd versions/v2_strong/code
python3 -m unittest test_code_paper_consistency.py   # 19 tests
python3 scenario_a.py        # and scenario_b / _brace / _c_g1 / _qstatic

# v4 — torque-level terrain and push study
cd versions/v4/code
python3 verify_interaction_paper_claims.py           # fail-closed evidence gate
python3 run_uneven_ground_benchmark.py --help

# v5 — confidence-gated wrench layer on the frozen policy
cd versions/v5/code
python3 check_platform.py                            # reports missing assets
python3 -m unittest test_gate_semantics.py           # 6 tests
python3 revalidate_gated.py                          # authoritative artifact
```

### Assets, and what the repository does not carry

`*.STL` and `*.npz` are gitignored to keep the repository small, with two
deliberate exceptions recorded in `.gitignore`:

* **v5's 27 G1 meshes are committed.** Three of them
  (`torso_link_23dof_rev_1_0.STL`, the two `*_wrist_roll_rubber_hand.STL`) have
  no `mujoco_menagerie` equivalent and are also the largest — 14.8 of 25.2 MB —
  so no auto-fetch can reconstruct the set. See `versions/v5/code/ASSETS.md`.
* **v3's demo videos and archived PDFs are committed**, because they cannot be
  cheaply regenerated.

v3 and v4 meshes are *not* committed: they resolve at load time from
`$MENAGERIE_G1_ASSETS` or the `robot_descriptions` package. v5's frozen
reference (`reference/frozen_walk_seed0.npz`) is regenerable, so it is not
committed either — `check_platform.py` prints the exact command.

Compiled paper PDFs **are** tracked deliverables. Note for v5: `wbc_v5.tex`
writes `\includegraphics{name}` without an extension, so LaTeX prefers the
`.pdf` figure over the `.png` of the same basename; `versions/v5/code/figures/*.pdf`
must stay tracked or a clone silently builds against the raster fallbacks.

### Building the papers

```bash
cd versions/v5 && latexmk -pdf wbc_v5.tex
cd versions/v5 && latexmk -xelatex wbc_v5_zh.tex   # Chinese: xeCJK needs XeLaTeX
```

The `_zh` translations **fail under `latexmk -pdf`** — they need `-xelatex`.
They are maintained in parallel and carry claims verbatim, so a correction to an
English manuscript must be applied to its `_zh` sibling too.

Manuscripts are LaTeX-only. The former Markdown sources and their PDF exports
were retired for v2_strong, v4 and v5; only the archived v3 still has one, and
its `sync_markdown_to_tex.py` is broken, so `wbc_v3.tex` cannot be regenerated
from it.

TeX requirements: `IEEEtran`, `amsmath`, `amssymb`, `amsfonts`, `amsthm`,
`graphicx`, `booktabs`, `bm`, `cite`, `hyperref`, and `algorithmicx` for v5
(TeX Live `collection-latexrecommended`).

## Provenance and audit trail

Reported numbers are bound to the code that produced them, and past audits are
kept rather than overwritten:

* `versions/v2_strong/CODE_PAPER_AUDIT.md` — two audits, including the
  oblique-projector conditioning bug (`‖Pc‖₂ = 27.6`) that made the MPC
  controllers diverge once the arm feedforward was implemented.
* `versions/v5/code/PROVENANCE.md` — which script and artifact produces each
  reported number, and which committed artifacts are superseded.
* `versions/v4/code/verify_interaction_paper_claims.py` — fail-closed gate that
  recomputes the manuscript's tables from the authoritative JSON and rejects
  stale values.

## Other directories

`g1_ab_simulation/` is a MuJoCo scaffold for the multi-rate architecture;
`simulation/` holds shared controllers and models. Root-level `*.md`/`*.pdf`
files are design notes and downloaded reference papers, not deliverables.
