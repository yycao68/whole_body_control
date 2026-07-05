# G1 A-to-B MuJoCo Simulation Scaffold

This folder contains a MuJoCo scaffold for testing the multi-rate architecture:

```text
global planner       10 Hz
local planner       100 Hz
G1 command layer    500 Hz
MuJoCo physics     1000 Hz
```

The demo loads the existing G1 model from:

```text
../simulation/models/g1_wbc.xml
```

and moves the robot base from point A to point B while logging the global waypoint layer, the local double-integrator planner, and the command layer.

## Important Scope Note

The default run uses `--root-assist`, which kinematically guides the floating base along the local planner trajectory while still stepping MuJoCo and commanding G1 joint poses. This validates the planner/control interfaces and produces a stable A-to-B simulation artifact. It is not yet a physically validated humanoid walking controller.

The `--wbc-control impedance-mpc` mode uses the repository's `ImpedanceMPC`
and Kalman disturbance estimator for the arm/catch task. Because this G1 MJCF
uses position actuators, the Cartesian MPC force is bridged through a damped
hand-Jacobian update into arm position targets. This exercises the proposed
constant-`A_d` Impedance-MPC task layer, but it is not yet a pure torque-level
whole-body inverse-dynamics implementation.

Replacing root assist with dynamic walking requires:

- footstep/contact schedule generation;
- centroidal MPC for CoM and ground reaction forces;
- torque-level WBC or high-quality position-control gait tracking;
- contact-state estimation and fall recovery.

## Run

```bash
cd /Users/yycao/Documents/git/ai_learn
python3 whole_body_control/g1_ab_simulation/run_g1_ab.py
```

Outputs:

- `results/g1_ab_path.png`
- `results/g1_ab_log.csv`
- optionally `results/g1_ab_demo.mp4` if video recording is enabled.

If MuJoCo cannot create an OpenGL renderer in the current session, the script automatically disables video and still writes the CSV and path plot.

## Useful Options

```bash
python3 whole_body_control/g1_ab_simulation/run_g1_ab.py --duration 8
python3 whole_body_control/g1_ab_simulation/run_g1_ab.py --goal 1.5 0.4
python3 whole_body_control/g1_ab_simulation/run_g1_ab.py --no-video
python3 whole_body_control/g1_ab_simulation/run_g1_ab.py --no-root-assist
python3 whole_body_control/g1_ab_simulation/run_g1_ab.py --goal 5 0 --duration 26 --apple-catch --wbc-control impedance-mpc
```
