# Corrected experiment status

Date: 2026-07-19

This file separates completed diagnostic gates from publication evidence. None
of the one-seed numbers below may be copied into the paper as a benchmark
result.

> NOTE (recovery): the corrected-protocol *code* overhaul that this file
> describes was reverted by a `git checkout`/`reset` and is not recoverable from
> git (it was never committed). The corrected-protocol terrain rerun *data* it
> produced survives in `code/results/RECOVERED_terrain_trials.json` (120 trials,
> extracted from the run log). The overhaul code must be recovered from the
> editor's local history or rebuilt before the numbers below can be regenerated.

## Corrections that were in code (now reverted; to recover or rebuild)

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
  step length, and a 15 s terrain window.
- `verify_interaction_paper_claims.py` rejects legacy schema-1 and mixed-controller
  artifacts.

## Recovered corrected-protocol terrain result (RECOVERED_terrain_trials.json)

From the 120-trial recovered rerun (15 s conservative gait, future-patch obstacle):

| terrain | impedance falls | nominal MPC | ID-MPC |
|---|---|---|---|
| flat | 8 | 2 falls, 15 s | 0 falls, 15 s |
| obstacle (patch contacted 10/10) | 7 | 1 fall, peak 12.9 mm | 0 falls, peak 10.6 mm |
| depression | 10 @ 8.2 s | 10 @ 8.2 s | 10 @ 8.2 s |
| rough | 10 @ 8.2 s | 10 @ 8.2 s | 10 @ 8.2 s |

- The obstacle ID-MPC benefit survives and is now a valid future-contact result:
  peak 12.9 -> 10.6 mm (-18%), 0 falls vs nominal's 1.
- ID-MPC is the only controller with zero falls on flat and obstacle.
- Depression and rough destabilize every controller at ~8.2 s (identical across
  controllers -> a shared gait/terrain-geometry issue, not the interaction
  layer). Those two terrains do not yield clean 15 s data under this gait.

## Required publication rerun (after code is recovered/rebuilt)

```bash
MPLCONFIGDIR=/tmp/mpl-cache XDG_CACHE_HOME=/tmp/xdg-cache \
python3 code/run_uneven_ground_benchmark.py
MPLCONFIGDIR=/tmp/mpl-cache XDG_CACHE_HOME=/tmp/xdg-cache \
python3 code/run_external_push_benchmark.py --duration 4 --save-representative
MPLCONFIGDIR=/tmp/mpl-cache XDG_CACHE_HOME=/tmp/xdg-cache \
python3 code/make_uneven_ground_figures.py
MPLCONFIGDIR=/tmp/mpl-cache XDG_CACHE_HOME=/tmp/xdg-cache \
python3 code/make_external_push_figures.py
MPLCONFIGDIR=/tmp/mpl-cache XDG_CACHE_HOME=/tmp/xdg-cache \
python3 code/verify_interaction_paper_claims.py
```

Only after that command reports `PASS` should numerical tables, abstract
percentages, conclusion claims, timing numbers, or publication videos be
restored to `wbc_ieee.md`.
