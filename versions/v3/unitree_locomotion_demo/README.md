# Unitree Locomotion Demo

This is a **demo-only** package for showing high-level correction commands on
top of Unitree's open-source G1 MuJoCo locomotion policy.

The paper claim remains narrower: interaction dynamics for floating-base
whole-body manipulation. Walking is supplied by Unitree RL Gym; the v3
interaction-dynamics code adds disturbance observation, load preview, and
high-level command correction.

## Folder Layout

```text
unitree_locomotion_demo/
├── DEMO_PLAN.md
├── demo_manifest.json
├── scripts/
│   ├── compose_demo_video.py
│   ├── generate_all_open_source_videos.py
│   ├── run_unitree_rl_gym_g1.py
│   └── unitree_locomotion_adapter.py
└── results/      # final videos, logs, and summaries
```

## Generate Final Videos

The demo uses the cloned Unitree repository at:

```text
../external_deps/unitree_rl_gym
```

Generate the three final comparison videos:

```bash
cd whole_body_control/versions/v3/unitree_locomotion_demo
mjpython scripts/generate_all_open_source_videos.py
```

This keeps only three MP4 files:

```text
results/unitree_d0_baseline_comparison.mp4
results/unitree_d1_d2_push_comparison.mp4
results/unitree_d3_preview_comparison.mp4
```

The raw Unitree scene videos are generated as intermediate files and deleted
after the final comparison videos are composed. Use
`--keep-intermediate-videos` only when debugging the renderer.

Verify the generated package:

```bash
python3 scripts/verify_demo_package.py --require-generated
```

Generate only one comparison after keeping intermediate videos:

```bash
python3 scripts/compose_d1_d2_comparison.py
```

Generate only the D0 baseline comparison:

```bash
python3 scripts/compose_d0_baseline_comparison.py
```

Generate only the D3 preview comparison:

```bash
python3 scripts/compose_d3_preview_comparison.py
```

## What Is Being Run

`scripts/run_unitree_rl_gym_g1.py` repeats Unitree's public MuJoCo deployment
logic: the pretrained G1 policy outputs joint targets at 50 Hz, and a PD motor
law applies torques in MuJoCo.

The four scenes are:

- `D0`: baseline Unitree G1 walking.
- `D1`: scripted lateral push with Unitree policy only.
- `D2`: the same push with Interaction Dynamics MPC correction enabled.
- `D3`: planned lateral load with preview correction enabled.

The final comparison videos are:

- `results/unitree_d0_baseline_comparison.mp4`: baseline Unitree walking versus the same walking command with Interaction Dynamics MPC correction enabled.
- `results/unitree_d1_d2_push_comparison.mp4`: D1 and D2 stacked vertically with the same reference line, push schedule, and live curves.
- `results/unitree_d3_preview_comparison.mp4`: same planned load with reactive Interaction Dynamics MPC versus preview-enabled Interaction Dynamics MPC.

The claim boundary is strict: Unitree's open-source policy supplies walking.
The v3 interaction-dynamics code supplies disturbance observation, preview, and
high-level correction commands; it does not generate the gait.

## Adapter Design

The future hardware integration point is `scripts/unitree_locomotion_adapter.py`.
It is a thin interface around whatever Unitree exposes:

- commanded base velocity;
- measured base pose/IMU;
- estimated CoM or pelvis state;
- high-level body correction input, if available;
- hand/task reference correction, if available.

The correction module should output high-level commands. It should not bypass
Unitree safety or inject joint torques underneath Unitree's controller unless
the SDK exposes an official, documented low-level torque/current mode.
