# V4 Reproducibility Code

This directory contains the minimum source and accepted evidence for the paper
*Interaction Dynamics: A Configuration-Invariant Predictive Model for Humanoid
Locomotion under Terrain and External Disturbances*.

## Controller implementation

- `normalized_mpc.py`: fixed exact-ZOH double-integrator MPC.
- `interaction_estimator.py`: finite-difference acceleration residual and 3 Hz
  low-pass estimate.
- `reference_provider.py`: gait, finite double-support ZMP transfer, and task
  references.
- `run_g1_torque_realizer_benchmark.py`: MuJoCo G1 model and instantaneous
  inverse-dynamics/contact QP.
- `run_g1_root_assist_demo.py`: shared model/contact/rendering utilities only.
  The accepted benchmarks and video explicitly disable root assistance.
- `capture_point.py`: optional capture-point utility imported by the terrain
  runner; the reported campaign leaves its additional stabilizer disabled.

The simulated schedule is 100 Hz for MPC, 500 Hz for the whole-body QP, and
1 kHz for applied torque. Wall-clock timing is recorded separately and the
current Python QP does not satisfy the 2 ms real-time deadline.

## Experiment and artifact scripts

- `run_uneven_ground_benchmark.py`: flat, depression, obstacle, and rough
  terrain campaign for impedance, nominal MPC, and interaction MPC.
- `run_external_push_benchmark.py`: sagittal/lateral pushes gated on measured
  single- or double-support phase.
- `make_continuous_flat_video.py`: 15 s torque-level interaction-MPC video with
  no root assistance.
- `make_uneven_ground_figures.py` and `make_external_push_figures.py`: paper
  plots from the accepted JSON and representative logs.
- `merge_terrain_artifacts.py`: merges independently generated terrain records.
- `verify_interaction_paper_claims.py`: final evidence gate and hash record.

## Accepted results

`results/uneven_ground_benchmark.json` and
`results/external_push_benchmark.json` are schema-version-2 records with ten
paired seeds per controller/condition: 120 terrain trials and 120 push trials.
Only five representative NPZ histories are retained because the JSON files
contain the complete aggregate evidence:

- three obstacle histories (one per controller) for the terrain time series;
- nominal- and interaction-MPC lateral/single-support push histories.

`results/continuous_flat_idmpc.json` accompanies the accepted MP4, and
`results/uneven_ground_verification.json` records the artifact hashes and gate
status.

## Minimum checks

From `whole_body_control/versions/v4`:

```bash
python3 code/verify_interaction_paper_claims.py
python3 -m py_compile code/*.py
```

The result package supports bounded peak-error attenuation under the tested
conditions. It does not establish terrain preview, universal tracking
improvement, fall avoidance, hardware robustness, or real-time feasibility.
