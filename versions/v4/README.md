# Whole-Body Interaction Dynamics V4

V4 is the publication package for the interaction-dynamics locomotion study.
It evaluates a fixed canonical task predictor with a 100 Hz residual-augmented
MPC and a torque-level Unitree G1 inverse-dynamics/contact QP scheduled at
500 Hz. The study covers uneven terrain and measured-phase external pushes;
it does not claim terrain preview, online footstep replanning, or real-time
execution of the current Python QP.

## Publication artifacts

- `wbc_v4.tex`: **the manuscript.** Edit this directly.
- `wbc_v4.pdf`: compiled paper (tracked, rebuilt with the command below).
- `figures/`: the eight cited figures and their two architecture-source scripts.
- `code/`: benchmark, verification, plotting, and video source.
- `code/results/uneven_ground_benchmark.json`: 120-trial terrain record.
- `code/results/external_push_benchmark.json`: 120-trial push record.
- `code/results/continuous_flat_idmpc.mp4`: no-root-assist torque-level video.
- `code/results/uneven_ground_verification.json`: hashes and evidence-gate result.
- `V3_TO_V4_DETAILED_COMPARISON.md`: version comparison.
- `CORRECTED_EXPERIMENT_STATUS.md`: experiment-history and validity notes.

## Verify the accepted evidence

From this directory:

```bash
python3 code/verify_interaction_paper_claims.py
python3 -m py_compile code/*.py figures/*.py
```

The verifier checks the exact controller/terrain/seed matrices, obstacle-contact
validity, measured-phase push gating, zero QP fallback, frozen controller
settings, paper figures, and the hash of the no-root-assist video.

## Regenerate benchmarks and figures

The complete campaigns are computationally expensive. Use writable cache
locations on macOS:

```bash
export MPLCONFIGDIR=/private/tmp/mplconfig
export XDG_CACHE_HOME=/private/tmp/cache

python3 code/run_uneven_ground_benchmark.py \
  --trials 10 --seed 4200 --artifact uneven_ground_benchmark.json
python3 code/run_external_push_benchmark.py \
  --seeds 10 --seed-start 4200 --save-representative \
  --artifact external_push_benchmark.json
python3 code/make_uneven_ground_figures.py
python3 code/make_external_push_figures.py
python3 code/make_continuous_flat_video.py
python3 code/verify_interaction_paper_claims.py
```

## Build the paper

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error wbc_v4.tex
```

`wbc_v4.tex` is the single source of truth. The former Markdown source
(`wbc_ieee_v4.md`) and its PDF export were retired; `sync_markdown_to_tex.py`
was removed with them, so nothing regenerates the TeX and edits to it are
never clobbered. `code/verify_interaction_paper_claims.py` reads `wbc_v4.tex`
directly and still checks every table cell against the authoritative JSON.
