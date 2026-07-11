# V3 Simulation Package

This folder will contain only the v3 dual-MPC implementation. Do not copy v2
result tables into v3.

## Current G1 Walking Video

The canonical video artifact is a 10 s dual-MPC root-assisted Unitree G1
walking demo based on the same scaffold as `whole_body_control/g1_ab_simulation`.
It moves the G1 10.8 m in MuJoCo with alternating one-foot swing phases and no
external push:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig python3 whole_body_control/versions/v3/code/run_g1_root_assist_demo.py
```

It uses:

- the local Menagerie-derived `models/g1_wbc.xml`;
- `normalized_mpc.py` for the body and task interaction-MPC command layers;
- a 10 s trapezoidal forward reference: 0--1 s ramp to 1.2 m/s, 1--9 s
  cruise at 1.2 m/s, and 9--10 s deceleration to rest;
- the `g1_ab_simulation` alternating one-foot swing command layer;
- root assist, commanded joint-pose mirroring, and stance-foot pinning for a
  clean walking visualization.

Current deterministic no-push visual result:

| Metric | Value |
|---|---:|
| Duration | 10.0 s |
| Speed profile | 0--1 s ramp, 1--9 s at 1.2 m/s, 9--10 s stop |
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

Generated files:

- `results/g1_walk_10s_summary.json`
- `results/g1_walk_10s_log.npz`
- `results/g1_walk_10s.png`

To generate the Neuralink-style MP4 with the actual Unitree G1 MuJoCo viewport
and live plots, run:

```bash
mjpython whole_body_control/versions/v3/code/render_g1_walk_video.py --mode mujoco --fps 30 --out whole_body_control/versions/v3/code/results/g1_walk_10s_video.mp4
```

The output is `results/g1_walk_10s_video.mp4`. The renderer has only the
MuJoCo viewport path; if MuJoCo cannot create an OpenGL context, the command
fails rather than producing a schematic substitute.

`results/g1_walk_10s_video.mp4` is the canonical no-push 10 s walking video.

To verify the implemented v3 artifacts after any cleanup or regeneration, run:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig python3 whole_body_control/versions/v3/code/verify_v3_artifacts.py
```

This checks the exact-ZOH normalized MPC matrices, input-centered disturbance
cancellation, G1 model loading, H1/H2 result files, summary/log consistency,
dual-MPC flags, visible foot lift, MP4 metadata, H3/H4 result files,
fixed-support torque smoke gates, and the intentionally retained
contact-switch/walking failure status.

Important limitation: this uses the two MPC command layers, but it is still a
root-assisted visualization, not the final paper-grade dynamic walking
benchmark. It is useful for showing the G1 model, visible one-foot lifting, MPC
command-layer integration, and 10.8 m of forward walking. The
paper-grade S4 result still requires torque actuation, inverse-dynamics
contact-wrench recovery, and randomized push trials.

## Torque-Level Realizer Smoke Benchmark

The first torque-actuated v3 runner is now implemented:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig python3 whole_body_control/versions/v3/code/run_g1_torque_realizer_benchmark.py --scenario stand --duration 3.0 --trials 1 --seed 31
MPLCONFIGDIR=/private/tmp/mplconfig python3 whole_body_control/versions/v3/code/run_g1_torque_realizer_benchmark.py --scenario stand_push --duration 3.0 --trials 3 --seed 21
MPLCONFIGDIR=/private/tmp/mplconfig python3 whole_body_control/versions/v3/code/run_g1_torque_realizer_benchmark.py --scenario contact_switch --duration 3.0 --trials 1 --seed 41
MPLCONFIGDIR=/private/tmp/mplconfig python3 whole_body_control/versions/v3/code/run_g1_torque_realizer_benchmark.py --scenario walk --duration 3.0 --trials 1 --seed 51
```

The runner generates `models/g1_wbc_torque.xml` from the local position-actuated
MJCF by replacing the 29 position actuators with torque motors. It uses the
same normalized body/task MPCs and `RandomWalkDisturbanceObserver`, then applies
a present-sample inverse-dynamics/contact QP realizer:

```text
min ||J_t qdd - xdd_task_des||^2 + ||J_c qdd - xdd_contact_des||^2 + posture
s.t. M qdd + h = S^T tau + J_c^T lambda, torque bounds, friction pyramid
```

Current status: the fixed-support portion now passes, but this is not yet a
passed replacement for the root-assisted walking video because support switching
and walking still fall.

| Scenario | Trials | Passed | Falls | Median completed time |
|---|---:|---:|---:|---:|
| stand | 1 | 1 | 0 | 2.999 s |
| stand_push | 3 | 3 | 0 | 2.999 s |
| contact_switch | 1 | 0 | 1 | 2.041 s (5 switches) |
| walk | 1 | 0 | 1 | 1.889 s (8 switches) |

The randomized-push trials complete and detect all injected pushes with the
normalized disturbance observer. The contact-switch and walking trials still
fall. The logs now distinguish post-QP clipping residual from torque-limit
utilization: all scenarios reach the torque bounds, but the applied clipping
residual remains small for fixed-support push trials (0.13--0.30 Nm) and is
explicitly reported for stepping failures. The failed contact-switch and
walking runs are now DCM-referenced rather than centered-CoM scaffolds; they
survive several support changes, then fail with negative measured foot-floor
friction margins and many QP fallback samples as single-support balance
authority runs out. The next missing components are a production gait/contact
realizer with hip/angular-momentum balance, capture-point step timing/placement
adaptation, and stable support-mode control.

## H3/H4/H6 Hypothesis Checks

H3, H4, H5, and H6 are implemented as torque-actuated G1 checks:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig python3 whole_body_control/versions/v3/code/run_h3_coupling.py
MPLCONFIGDIR=/private/tmp/mplconfig python3 whole_body_control/versions/v3/code/run_h4_detection.py
MPLCONFIGDIR=/private/tmp/mplconfig python3 whole_body_control/versions/v3/code/run_h5_constraints.py
MPLCONFIGDIR=/private/tmp/mplconfig python3 whole_body_control/versions/v3/code/run_h6_onbase.py
```

H3 contrasts the two preview forms of Section VII at equal reaction magnitude
(~45 N). For the **external** load (Eq. 23), unmodeled by the whole-body QP, the
coupled preview cuts peak CoM excursion from 37.66 mm to 22.08 mm (1.71x) and RMS
from 20.85 mm to 11.89 mm. For the **internal** arm-momentum reaction (Eq. 23b),
the unified QP already compensates it natively via the shared CoM Jacobian, so the
uncompensated (split) transient is only 9.18 mm — 4x smaller than the external
load — and the external-style preview does not help (indeed 15.19 mm, since it
perturbs a CoM command the QP is already satisfying). The takeaway: preview
belongs to what the realizer does not model (external contact loads).

H4 applies three scripted brace-contact intervals, giving six onset/offset
events. The detector uses only the body observer's normalized innovation and a
quiet-window threshold; the scripted schedule is used only as the scoring
oracle and for plotting. Current result: 6/6 detected, 0 missed, 0 false
positives, mean latency 56.0 ms, max latency 58.0 ms.

H5 enforces friction cones and torque limits in the recovery QP, not the
predictor. An illustrative 45 N push shows constrained recovery holding the
recovered wrench/torque at the solver tolerance (0.3 N, 0.06 N.m) and standing,
while unconstrained recovery commands ~900 N / ~945 N.m of infeasible force/torque
and falls. `run_batch(50)` then repeats over 50 randomized pushes (magnitude
U(30,50) N, randomized lateral-dominant direction and onset): constrained stands
50/50 with max friction/torque violation 0.31 N / 0.32 N.m and CoM error
6.1 +/- 1.3 mm; unconstrained falls 50/50. (The unconstrained violation saturates
to a common ceiling once the robot tips, so only its magnitude is meaningful.)

H6 demonstrates the interaction layer on a *moving* base: the base commands its
own +/-50 mm forward weight-shift (0.25 Hz) while a planned 45 N, 1.6 Hz lateral
trunk load disturbs the CoM. With the layer on, previewing the load cuts the
lateral base-tracking error from 24.38 to 10.32 mm RMS (2.36x) and 45.51 to
21.77 mm peak (2.09x), while the base's forward weight-shift is tracked
essentially identically (35.61 vs 35.43 mm). Double support throughout; no fall.

Generated files:

- `results/h3_coupling_summary.json`, `results/h3_coupling.png`
- `results/h4_detection_summary.json`, `results/h4_detection.png`
- `results/h5_constraints_summary.json`, `results/h5_constraints_stats.json`, `results/h5_constraints.png`
- `results/h6_onbase_summary.json`, `results/h6_onbase.png`

## Planned Modules

- `normalized_mpc.py`: dimension-generic exact-ZOH interaction MPC.
- `centroidal_mpc.py`: Level-1 body MPC and contact-wrench recovery.
- `task_mpc.py`: Level-3 arm interaction MPC.
- `run_g1_torque_realizer_benchmark.py`: initial torque-actuated realizer smoke benchmark.
- `whole_body_interface.py`: production Level-2 inverse-dynamics QP.
- `disturbance_observer.py`: body/task augmented estimators.
- `contact_detector.py`: innovation-based event detector with kinematic gating.
- `g1_model.py`: model indexing, state extraction, and torque-actuated variant.
- `benchmark_s1.py` through `benchmark_s5.py`: experiment runners.
- `run_all.py`: deterministic benchmark entrypoint.
- `results/`: generated logs, tables, and figures.

## Reuse Policy

Reuse verified v2 formulas and utilities only after copying them here and
adding v3 tests. The v3 paper must be runnable from this folder without
importing another version.

The local `models/g1_wbc.xml` currently retains the Menagerie-derived position
actuators. Its meshes and license are stored locally. A separate torque-actuated
variant must be created and validated before running the primary benchmark;
position-as-torque results are only an ablation.

## Minimum Tests

1. Exact ZOH pair and lifted matrices.
2. Input-centered offset cancellation.
3. Centroidal wrench recovery residual.
4. Friction/CoP and torque constraint enforcement.
5. Arm-reaction sign and frame consistency.
6. Contact detector latency on a synthetic force step.
7. G1 timestep and actuator-mode audit.
8. Split/coupled equivalence when preview coupling is disabled.
