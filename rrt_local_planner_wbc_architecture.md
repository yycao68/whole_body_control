# Multi-Rate Global Planning, Local Motion Planning, and WBC Architecture

This note describes how to combine a slow global planner, the configuration-independent predictive motion planner, and the whole-body impedance controller into one robot stack.

## Objective

Move a humanoid or legged manipulator from point A to point B while preserving:

- global route feasibility;
- high-rate local obstacle reaction;
- velocity, acceleration, and contact-aware tracking limits;
- balance and whole-body execution constraints;
- pHRI disturbance rejection when the robot interacts with people or objects.

The intended stack is:

```text
RRT / graph global planner       1-10 Hz
        |
        v
Predictive local motion planner  50-100 Hz
        |
        v
Whole-body controller            500 Hz-1 kHz
        |
        v
MuJoCo / hardware robot
```

## Layer 1: Global Planner

The global planner searches the large-scale environment and outputs a sparse waypoint path:

```math
P = \{p_0, p_1, \ldots, p_M\}, \qquad p_i \in \mathbb{R}^2 \text{ or } \mathbb{R}^3.
```

The global planner may be RRT, RRT*, PRM, A*, hybrid A*, a footstep planner, or a semantic route planner. It should not output final timing or actuator commands. Its job is topological guidance:

- find a route around large static obstacles;
- update when the local planner cannot make progress;
- provide the next local goal or corridor.

Typical rate: 1-10 Hz.

## Layer 2: Configuration-Independent Local Motion Planner

The local planner receives:

- current robot base or task state;
- the next waypoint or short path segment from the global planner;
- dynamic obstacle predictions;
- velocity, acceleration, and clearance limits.

It solves a fixed-structure predictive QP using a virtual linear backbone:

```math
X = \Phi x_k + \Gamma U,
```

where \(\Phi\), \(\Gamma\), and the Hessian or sparsity pattern can be precomputed for a selected horizon. The local planner owns timing. It converts sparse global waypoints into a short-horizon trajectory:

```math
p_d(t), \quad \dot p_d(t), \quad \ddot p_d(t),
```

or joint/task-space references:

```math
q_d(t), \quad \dot q_d(t), \quad \ddot q_d(t).
```

Typical rate: 50-100 Hz.

## Layer 3: Whole-Body Control

The WBC layer receives the local trajectory and converts it into whole-body commands. For a floating-base robot, this layer handles:

- stance and contact constraints;
- centroidal balance;
- joint limits;
- end-effector tracking;
- compliant manipulation;
- pHRI disturbance rejection.

In the WBC paper architecture, the torque structure is:

```math
\tau =
\tau_{\text{contact}}
+ \bar N_1^\top \tau_{\text{balance}}
+ \bar N_{12}^\top
\left(
\tau_{\text{ff,arm}}
+ J_{\text{arm}}^\top F_{\text{mpc}}
+ \tau_{\text{null}}
\right).
```

The local planner should not directly command torques. It should provide references and bounds. The WBC layer is responsible for making those references physically executable.

Typical rate: 500 Hz-1 kHz.

## Data Interfaces

### Global Planner to Local Planner

```python
GlobalPlan:
    waypoints: list[tuple[float, float]]
    corridor_radius: float
    static_obstacles: list[Obstacle]
```

### Local Planner to WBC

```python
LocalTrajectory:
    position: np.ndarray      # desired base/task position
    velocity: np.ndarray      # desired base/task velocity
    acceleration: np.ndarray  # desired virtual acceleration
    heading: float
    phase_hint: str           # "stand", "walk", "turn", etc.
```

### WBC to Robot

```python
WholeBodyCommand:
    q_des: np.ndarray
    dq_des: np.ndarray
    tau_ff: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
```

## Why This Stack Works

RRT is good at global search but produces sparse, nonsmooth paths. The local predictive planner is good at fast, time-indexed feasibility but is not a global maze solver. WBC is good at physical execution but should not decide the global route.

The division of labor is therefore:

- **RRT:** where to go globally.
- **Local predictive planner:** how to move locally in time.
- **WBC:** how to execute on the full robot.

## MuJoCo Scaffold

The folder `whole_body_control/g1_ab_simulation/` contains a MuJoCo scaffold for moving the Unitree G1 model from A to B. The default script uses a kinematic root-assist mode to validate the planner and interface timing before replacing the simplified command layer with a full dynamic walking WBC.

This is intentional: robust humanoid walking requires a footstep planner, contact scheduler, state estimator, centroidal MPC, and torque-level WBC. The scaffold isolates the A-to-B architecture first so the planning interfaces can be tested and extended.

