# Detailed comparison: v3 baseline versus v4 continuous-walking revision

Date: 2026-07-19

## Version boundary

- `v3` is restored to the recorded 2026-07-19 15:09 PDT state, immediately
  before the request to implement the review findings. The preceding paper and
  code reviews were read-only; the first source patch followed this boundary.
  The recovery was checked against the recorded `git status` and manuscript
  SHA-256 at that time. The only outstanding v3 paths were three generated
  obstacle XML files and one untracked rough-terrain XML file.
- `v4` is a complete copy of the accepted state after the continuous-reference
  change, 240-trial evaluation, figure regeneration, and paper rewrite.

No unrelated pre-existing edits in `v3` were reset. The generated terrain XML
files that were already present at 15:09 remain in v3.

## Executive comparison

| topic | v3 baseline | v4 revision |
|---|---|---|
| Gait | 0.8 s step, 0.55 s double support, 0.03 m step, lateral ZMP scale 0.85 | 1.4 s step, 1.0 s double support, 0.03 m step, lateral ZMP scale 1.0 |
| ZMP transition | instantaneous foot-to-foot switch | continuous piecewise-linear transfer over 0.05 s |
| DCM construction | constant-ZMP backward recursion | exact backward recursion for linear-ZMP transfer followed by hold |
| Lateral-error interpretation | rejects DCM tracking error but concludes that the requested lateral correction is unrealizable | explicitly separates the stance-relative DCM diagnostic from controlled error $e_y=y-y_d$ and demonstrates the latter |
| Flat evidence | 4 s evaluation window; no continuous-walking acceptance gate or complete video | ten paired 15 s seeds for three controllers plus a complete torque-level video |
| Publication terrain data | schema-1 artifact: 160 trials including a diagnostic fourth controller; manuscript reports selected three-controller results | schema-2 artifact: 120 trials containing exactly the three publication controllers |
| Publication push data | 160-trial artifact including `no_realization_feedback`; manuscript reports selected three-controller results | 120 trials containing exactly the three publication controllers and the shared gait metadata |
| Verification | `PASS` for the earlier artifact/configuration contract | `PASS` for stricter gait, contact, controller, seed, fallback, video, and hash checks |
| Paper status | complete numerical draft based on the earlier short-window experiments | rewritten abstract, experiments, limitations, and conclusion tied to the corrected artifacts |

## Motion-planning change

The canonical interaction predictor is unchanged. Both versions retain the same
fixed double-integrator task model. The difference is confined to the external
walking reference.

`v3` uses the earlier faster gait and moves the ZMP directly from the previous
support point to the next one at the phase boundary. `v4` first adopts the
slower continuous-walking reference above, then adds:

```text
smooth_double_support = true
zmp_transfer_time = 0.05 s
smooth_lateral_only = false
```

During that interval, `v4` uses a linear ZMP trajectory and analytically
consistent DCM propagation. Numerical checks give a maximum DCM-dynamics
residual of $1.18\times10^{-9}$ and phase-boundary DCM/ZMP discontinuities below
$5.5\times10^{-9}$ m.

The change is deliberately conservative. Trials with 0.10--1.00 s transfers
fell; the 0.05 s coupled transfer was the longest tested configuration that
completed the 15 s torque-level flat gate reliably.

## Controlled lateral error

The 135 mm quantity is not $e_y$. It measures a nominal DCM position relative
to a stance-foot frame in the earlier capture-correction diagnostic. The v4
video measures the actual controlled quantity $e_y=y-y_d$ about a moving lateral
reference:

| v4 ID-MPC video metric | value |
|---|---:|
| duration | 15.0 s |
| forward travel | 0.231 m |
| falls | 0 |
| QP fallbacks | 0 |
| lateral RMS error | 4.68 mm |
| lateral mean error | -1.48 mm |
| lateral peak error | 10.63 mm |
| final-second lateral RMS | 3.08 mm |

This supports near-zero-centered tracking of the moving reference; it does not
mean that the robot's world-frame lateral position stays at zero during walking.

## Terrain results added in v4

All values are medians over ten paired seeds. Error values in falling cells are
truncated at the fall and are not full-window comparisons.

| terrain | controller | CoM RMS (mm) | CoM peak (mm) | falls/10 |
|---|---|---:|---:|---:|
| flat | impedance | 4.782 | 52.204 | 10 |
|  | nominal MPC | 4.110 | 11.434 | 0 |
|  | ID-MPC | 4.508 | 10.636 | 0 |
| depression | impedance | 5.839 | 48.472 | 10 |
|  | nominal MPC | 5.821 | 47.506 | 10 |
|  | ID-MPC | 5.807 | 42.679 | 10 |
| obstacle | impedance | 3.989 | 11.564 | 0 |
|  | nominal MPC | 4.572 | 11.434 | 0 |
|  | ID-MPC | 4.567 | 10.636 | 0 |
| rough | impedance | 6.072 | 48.900 | 10 |
|  | nominal MPC | 6.166 | 48.325 | 10 |
|  | ID-MPC | 6.082 | 43.517 | 10 |

The defensible interpretation is narrow:

- Nominal MPC and ID-MPC continuously walk on flat ground for all ten seeds.
- Every controller completes the valid future-obstacle trials.
- ID-MPC reduces obstacle peak error by 7.0% relative to nominal MPC, but does
  not materially change obstacle RMS and does not improve flat RMS.
- Every controller falls on the fixed 20 mm depression and rough sequence.
  Residual compensation therefore does not repair an infeasible fixed plan.

## Push results added in v4

Relative to nominal MPC, ID-MPC changes median peak CoM error by:

| push condition | peak change |
|---|---:|
| lateral, double support | -14.0% |
| lateral, single support | -22.6% |
| forward, double support | -6.2% |
| forward, single support | -13.7% |

The lateral single-support recovery median improves from 0.754 to 0.279 s.
However, every controller falls for the lateral double-support push, and no
controller satisfies the recovery-band definition for the forward
single-support push. The v4 paper therefore claims phase-dependent transient
attenuation, not universal push rejection.

## Prediction and computation

- At the 10 ms horizon, conditioned-residual CoM prediction changes stay below
  1.2% on all terrains. The paper treats this as near-neutral corroborating
  evidence, not a headline forecasting result.
- Prototype WBC timing is 3.91 ms median and 10.89 ms p99 versus a simulated
  2 ms schedule, so 500 Hz wall-clock feasibility is not claimed.
- MPC timing is 0.31 ms median and 0.45 ms p99 versus its 10 ms schedule.

## Principal file differences

Line counts are intentionally omitted because subsequent manuscript editing and
verification improvements make them brittle. The durable differences are:

| file | principal v4 change |
|---|---|
| `wbc_ieee_v4.md` | Reframed contribution and rewritten evaluation around continuous walking, uneven terrain, and phase-specific pushes |
| `CORRECTED_EXPERIMENT_STATUS.md` | Records the accepted evidence boundary and reproducibility status |
| `code/reference_provider.py` | Adds continuous ZMP transfer and the corresponding walking reference |
| `code/run_uneven_ground_benchmark.py` | Implements the paired 15 s terrain campaign and publication artifact |
| `code/verify_interaction_paper_claims.py` | Checks artifact hashes, configuration, every reported table value, figures, and video metrics |
| `code/capture_point.py` | Aligns DCM propagation with the continuous reference |
| `code/run_g1_torque_realizer_benchmark.py` | Updates the torque-level benchmark to the v4 reference contract |
| `code/run_external_push_benchmark.py` | Adds the paired support-phase push evaluation |

Files retained specifically for the cleaned v4 publication workflow include:

- `code/make_continuous_flat_video.py`
- `code/merge_terrain_artifacts.py`
- `code/results/continuous_flat_idmpc.mp4`
- `code/results/continuous_flat_idmpc.json`
- `code/results/uneven_ground_benchmark.json`
- `code/results/external_push_benchmark.json`
- the five representative NPZ histories required by Figures 5 and 7

Per-terrain intermediate JSON, uncited prediction plots, development campaigns,
and redundant per-seed histories were removed after the accepted aggregate
records and representative histories were verified.

## Verification and traceability

- v3: `python3 code/verify_interaction_paper_claims.py` reports `PASS` for its
  earlier schema-1, short-window evidence contract. That pass is not equivalent
  to the stricter v4 validity contract.
- v4: the same command reports `PASS`.
- v4 terrain artifact SHA-256:
  `da87a08b198abfa82d05eae0e028cea4a484c41a31d5022d5066ca554cb2b4d1`
- v4 push artifact SHA-256:
  `1b74c4453e084d951447e90d09a1a20d752a80322c4ae7c0c5410a6996096bd3`
- v4 video SHA-256:
  `56648b7f1a11d572f6c6733031285d91ed2fe576b4b61982712759099f3735db`

## Restoration caveat

The per-trial `.npz` files are ignored generated products. The v4 campaign
overwrote some identically named old ignored logs, and those previous bytes were
not recoverable from Git. The new logs are preserved in v4 and removed from v3
to prevent the restored v3 source from appearing to own v4 evidence. The
tracked v3 benchmark JSON, figures, and earlier verifier contract were
recoverable and have been restored.
