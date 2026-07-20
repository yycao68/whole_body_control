# Corrected experiment status

Date: 2026-07-19

This file separates the completed publication matrices from diagnostic gates.
One-seed smoke values remain diagnostic only; manuscript numbers come from the
accepted ten-seed artifacts.

## Corrections now in code

- Cartesian body, contact, hand, and swing-foot acceleration objectives include
  the kinematic bias term $\dot J\dot q$.
- The interaction estimator and logs use finite-differenced measured task
  acceleration. QP-predicted acceleration is stored separately.
- Publication artifacts contain exactly `impedance`, `nominal_mpc`, and
  `interaction_mpc`; `no_realization_feedback` is diagnostic only.
- The implemented residual conditioning is frozen and recorded: 0.30-unit
  task-acceleration deadband, 0.50-unit cap, and 0.70-unit command slew per
  100 Hz update (m/s^2 for translation, rad/s^2 for attitude).
- The obstacle is a finite future patch at x = 0.22--0.34 m. A valid trial must
  show no patch contact during settling and a later measured patch contact.
- The shared publication gait uses 1.4 s steps, 1.0 s double support, 0.03 m
  step length, a 0.05 s continuous ZMP load transfer, and a 15 s terrain window.
- `verify_interaction_paper_claims.py` rejects legacy schema-1 and mixed-controller
  artifacts.

## Current smooth-reference smoke gates

| gate | outcome |
|---|---|
| smooth-transfer flat, nominal MPC, seed 4300, 15 s | no fall; 0.228 m travel; 4.49 mm lateral RMS error; zero QP fallback |
| smooth-transfer flat, ID-MPC, seed 4300, 15 s | no fall; 0.231 m travel; 4.68 mm lateral RMS error; zero QP fallback |
| smooth-transfer flat, impedance, seed 4300 | falls at 12.53 s; retained as a reported baseline failure |
| smooth-reference ID-MPC video, seed 4300, 15 s | no fall; 0.231 m travel; lateral error 4.68 mm RMS, -1.48 mm mean, and 3.08 mm final-second RMS; zero QP fallback |

Nominal MPC and ID-MPC also completed the 15 s flat gate without falls or QP
fallbacks for the additional paired seeds 4301 and 4302. This three-seed smoke
gate preceded the completed ten-seed publication matrix reported below.

The complete torque-level video is `code/results/continuous_flat_idmpc.mp4`,
with metrics and its SHA-256 recorded in `code/results/continuous_flat_idmpc.json`.
It is labeled as ID-MPC with no root assistance and displays the moving lateral
reference, actual CoM, and zero-centered lateral error.

Earlier instantaneous-ZMP smoke tests remain superseded. The current flat/video
values are software gates only; the accepted comparisons below use the complete
paired matrices.

## Completed publication rerun

The corrected terrain matrix contains four terrains, three controllers, and ten
paired seeds (120 trials). The corrected push matrix contains four phase/direction
conditions, three controllers, and ten paired seeds (120 trials). Both use the
frozen continuous gait and zero QP fallback. The verifier reports `PASS`.

Key outcomes:

- Nominal MPC and ID-MPC complete all flat and obstacle trials; impedance falls
  in all flat trials.
- Every controller falls on the fixed 20 mm depression and rough sequence,
  identifying a planner/shared-stack failure boundary.
- On the valid future obstacle, ID-MPC reduces median peak CoM error from
  11.434 to 10.636 mm (7.0%) versus nominal MPC, while RMS is essentially equal.
- Across the four push conditions, ID-MPC reduces median peak error versus
  nominal MPC by 6.2--22.6%; all controllers fall for lateral double-support
  pushes.
- The full verification record is `code/results/uneven_ground_verification.json`.

Reproduction commands:

```bash
MPLCONFIGDIR=/tmp/mpl-cache XDG_CACHE_HOME=/tmp/xdg-cache \
python3 code/run_uneven_ground_benchmark.py
```

Push matrix:

```bash
MPLCONFIGDIR=/tmp/mpl-cache XDG_CACHE_HOME=/tmp/xdg-cache \
python3 code/run_external_push_benchmark.py --duration 4 --save-representative
```

Regenerate figures and verify:

```bash
MPLCONFIGDIR=/tmp/mpl-cache XDG_CACHE_HOME=/tmp/xdg-cache \
python3 code/make_uneven_ground_figures.py
MPLCONFIGDIR=/tmp/mpl-cache XDG_CACHE_HOME=/tmp/xdg-cache \
python3 code/make_external_push_figures.py
MPLCONFIGDIR=/tmp/mpl-cache XDG_CACHE_HOME=/tmp/xdg-cache \
python3 code/verify_interaction_paper_claims.py
```

The current accepted run reports `PASS`; the numerical tables, bounded claims,
timing results, and no-root-assist video are now recorded in `wbc_ieee_v4.md`.
