# V3 Simulation Package

Self-contained implementation for the paper *Interaction Dynamics for
Floating-Base Whole-Body Manipulation*. Every artifact the paper figures and
`verify_v3_artifacts.py` depend on is regenerated from this folder without
importing another version.

## Quick start

Regenerate all artifacts and verify them in one deterministic pass:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig python3 whole_body_control/versions/v3/code/run_all.py
```

`run_all.py` runs H1–H6, the two torque-realizer smoke gates, and the
root-assisted walking visualization in a fixed order, then runs the verifier.
Use `--skip-verify` to regenerate only, or `--keep-going` to continue past a
failing step.

To verify already-generated artifacts without regenerating:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig python3 whole_body_control/versions/v3/code/verify_v3_artifacts.py
```

This checks the exact-ZOH normalized MPC matrices, input-centered disturbance
cancellation, G1 model loading, H1/H2 result files, summary/log consistency,
dual-MPC flags, visible foot lift, H3/H4 result files, fixed-support torque
smoke gates, the Unitree A.2 probe outputs, and the intentionally retained
contact-switch/walking failure status.

## Files in this package

- `run_all.py` — deterministic benchmark entrypoint (runs everything below).
- `normalized_mpc.py` — dimension-generic exact-ZOH interaction MPC and the
  `RandomWalkDisturbanceObserver`; imported by every runner.
- `run_g1_root_assist_demo.py` — 10 s dual-MPC root-assisted G1 walking
  visualization (Appendix A.1).
- `run_h1_multirobot.py` — H1 predictor invariance across G1/H1/Talos
  (imports `centroidal_rotational_inertia` from `run_h1_h2.py`).
- `run_h1_h2.py` — H1 command equivalence and H2 offset-free regulation.
- `run_h3_coupling.py` — H3 external-wrench vs internal-momentum preview.
- `run_h4_detection.py` — H4 innovation-based contact-event detection.
- `run_h5_constraints.py` — H5 constrained vs unconstrained recovery.
- `run_h6_onbase.py` — H6 interaction layer on a moving base.
- `run_g1_torque_realizer_benchmark.py` — torque-actuated inverse-dynamics /
  contact-QP realizer (H2–H5 smoke gates); imports `DCMWalk` from
  `run_gait_dcm.py` for the dynamically feasible CoM sway.
- `run_gait_dcm.py` — DCM-based CoM-sway helper used by the torque benchmark.
- `verify_v3_artifacts.py` — artifact and internal-consistency checker.
- `benchmark_config.yaml` — shared benchmark configuration.
- `models/` — local Menagerie-derived `g1_wbc.xml` and assets; the torque
  benchmark generates the torque-actuated `g1_wbc_torque.xml` variant from it.
- `results/` — generated logs, summaries, and figures.

## Root-assisted walking visualization (Appendix A.1)

A 10 s dual-MPC root-assisted Unitree G1 walking demo on the same scaffold as
`whole_body_control/g1_ab_simulation`. It moves the G1 10.8 m in MuJoCo with
alternating one-foot swing phases and no external push:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig python3 whole_body_control/versions/v3/code/run_g1_root_assist_demo.py
```

It uses the local `models/g1_wbc.xml`, `normalized_mpc.py` for the body and task
interaction-MPC command layers, a 10 s trapezoidal forward reference (0–1 s ramp
to 1.2 m/s, 1–9 s cruise, 9–10 s stop), the `g1_ab_simulation` alternating
one-foot swing command layer, and root assist with stance-foot pinning.

Current deterministic no-push visual result:

| Metric | Value |
|---|---:|
| Duration | 10.0 s |
| Speed profile | 0–1 s ramp, 1–9 s at 1.2 m/s, 9–10 s stop |
| Commanded distance | 10.8 m |
| External push | none |
| Body/task MPC enabled | true / true |
| Forward distance | 10.800 m |
| Left/right foot lift | 0.083 / 0.083 m |
| Minimum CoM height | 0.752 m |
| Max roll/pitch magnitude | 0.030 rad |
| Hand RMS error | 95.5 mm |
| Support switches | 15 |
| Fall | false |
| Visual pass | true |

Generated files: `results/g1_walk_10s_summary.json`, `results/g1_walk_10s_log.npz`,
`results/g1_walk_10s.png` (and its paper alias `results/g1_walk_10s_1p2ms.png`).

Important limitation: this uses the two MPC command layers but is still a
root-assisted visualization, not a torque-level dynamic walking benchmark. It
shows the G1 model, visible one-foot lifting, MPC command-layer integration, and
10.8 m of forward walking only.

## Torque-level realizer smoke benchmark

The fixed-support torque-actuated runner (used by H2–H5 and the Section X
pointer):

```bash
MPLCONFIGDIR=/private/tmp/mplconfig python3 whole_body_control/versions/v3/code/run_g1_torque_realizer_benchmark.py --scenario stand --duration 3.0 --trials 1 --seed 31
MPLCONFIGDIR=/private/tmp/mplconfig python3 whole_body_control/versions/v3/code/run_g1_torque_realizer_benchmark.py --scenario stand_push --duration 3.0 --trials 3 --seed 21
```

The runner generates `models/g1_wbc_torque.xml` from the local position-actuated
MJCF by replacing the 29 position actuators with torque motors. It uses the same
normalized body/task MPCs and `RandomWalkDisturbanceObserver`, then applies a
present-sample inverse-dynamics/contact QP realizer:

```text
min ||J_t qdd - xdd_task_des||^2 + ||J_c qdd - xdd_contact_des||^2 + posture
s.t. M qdd + h = S^T tau + J_c^T lambda, torque bounds, friction pyramid
```

| Scenario | Trials | Passed | Falls | Median completed time |
|---|---:|---:|---:|---:|
| stand | 1 | 1 | 0 | 2.999 s |
| stand_push | 3 | 3 | 0 | 2.999 s |

The randomized-push trials complete and detect all injected pushes with the
normalized disturbance observer; all reach the torque bounds while the post-QP
clipping residual stays small (0.13–0.30 Nm). The earlier DCM stepping /
contact-switch gates were retired when Appendix A.2 became the Unitree
open-source locomotion-stack probe (see `../unitree_locomotion_demo/`); the
`DCMWalk` helper in `run_gait_dcm.py` is retained only for the CoM sway used by
the torque benchmark.

## H1/H3/H4/H5/H6 hypothesis checks

```bash
MPLCONFIGDIR=/private/tmp/mplconfig python3 whole_body_control/versions/v3/code/run_h1_multirobot.py
MPLCONFIGDIR=/private/tmp/mplconfig python3 whole_body_control/versions/v3/code/run_h3_coupling.py
MPLCONFIGDIR=/private/tmp/mplconfig python3 whole_body_control/versions/v3/code/run_h4_detection.py
MPLCONFIGDIR=/private/tmp/mplconfig python3 whole_body_control/versions/v3/code/run_h5_constraints.py
MPLCONFIGDIR=/private/tmp/mplconfig python3 whole_body_control/versions/v3/code/run_h6_onbase.py
```

H1 (multi-robot) instantiates the *same* task port on three humanoids via
`robot_descriptions`: Unitree G1 (34 kg), Unitree H1 (51 kg), PAL Talos (94 kg).
The exact-ZOH predictor (A_t,B_t) is bit-identical across all three (it depends
only on the sample time), while the contact-consistent task inertia Lambda_t at
the hand spans 1.2–184 kg (155x) and mass 2.8x. All platform dependence is
confined to recovery, per Theorem 1.

H3 contrasts the two preview forms of Section VII at equal reaction magnitude
(~45 N). For the **external** load (Eq. 23), the coupled preview cuts peak CoM
excursion from 37.66 mm to 22.08 mm (1.71x) and RMS from 20.85 mm to 11.89 mm.
For the **internal** arm-momentum reaction (Eq. 23b), the unified QP already
compensates it natively via the shared CoM Jacobian, so the uncompensated split
transient is only 9.18 mm and external-style preview does not help. Preview
belongs to what the realizer does not model (external contact loads).

H4 applies three scripted brace-contact intervals, giving six onset/offset
events. The detector uses only the body observer's normalized innovation and a
quiet-window threshold; the scripted schedule is the scoring oracle only.
Current result: 6/6 detected, 0 missed, 0 false positives, mean latency 56.0 ms,
max latency 58.0 ms.

H5 enforces friction cones and torque limits in the recovery QP, not the
predictor. `run_batch(50)` repeats over 50 randomized pushes (magnitude
U(30,50) N, randomized lateral-dominant direction and onset): constrained stands
50/50 with max friction/torque violation 0.31 N / 0.32 N.m and CoM error
6.1 +/- 1.3 mm; unconstrained falls 50/50.

H6 demonstrates the interaction layer on a *moving* base: the base commands its
own +/-50 mm forward weight-shift (0.25 Hz) while a planned 45 N, 1.6 Hz lateral
trunk load disturbs the CoM. With the layer on, previewing the load cuts the
lateral base-tracking error from 24.38 to 10.32 mm RMS (2.36x) and 45.51 to
21.77 mm peak (2.09x), while the forward weight-shift is tracked essentially
identically (35.61 vs 35.43 mm). Double support throughout; no fall.

Generated files:

- `results/h1_multirobot.json`, `results/h1_multirobot.png`
- `results/h1_equivalence.png`, `results/h2_offset_free.png`, `results/h1_h2_results.json`
- `results/h3_coupling_summary.json`, `results/h3_coupling.png`
- `results/h4_detection_summary.json`, `results/h4_detection.png`
- `results/h5_constraints_summary.json`, `results/h5_constraints_stats.json`, `results/h5_constraints.png`
- `results/h6_onbase_summary.json`, `results/h6_onbase.png`

## Reuse policy

Reuse verified v2 formulas and utilities only after copying them here and adding
v3 tests. The v3 paper must be runnable from this folder without importing
another version. The local `models/g1_wbc.xml` retains the Menagerie-derived
position actuators; the torque-actuated variant is generated and validated by
the torque benchmark before use.

## Minimum tests

1. Exact ZOH pair and lifted matrices.
2. Input-centered offset cancellation.
3. Centroidal wrench recovery residual.
4. Friction/CoP and torque constraint enforcement.
5. Arm-reaction sign and frame consistency.
6. Contact detector latency on a synthetic force step.
7. G1 timestep and actuator-mode audit.
8. Split/coupled equivalence when preview coupling is disabled.
