# Change of Research Direction: Returning to the Interaction Dynamics Vision

## Background

During the development of the whole-body Interaction Dynamics (ID) framework, the research direction gradually shifted away from the original scientific objective.

The original vision was:

> **Interaction Dynamics is the research contribution. Humanoid
> locomotion is only the validation platform.**

However, because the walking reference was not sufficiently stable, much of the effort was spent fixing locomotion itself:

-   gait scheduler
-   phase synchronization
-   stabilizer
-   capture behavior
-   swing clearance
-   touchdown timing
-   support transfer

As a result, the paper gradually became about **making the robot walk without falling**, rather than **designing interaction dynamics**.

------------------------------------------------------------------------

# Fundamental Decision

The project will return to its original objective.

The scientific contribution is:

-   Interaction Dynamics
-   Predictive shaping of robot-environment interaction
-   Constraint-aware realization
-   Configuration/contact-invariant interaction model

The project is **NOT** intended to contribute:

-   a new gait generator
-   a new locomotion controller
-   a new capture-step planner
-   a new terrain planner

Humanoid locomotion exists only because it provides one of the most demanding floating-base, contact-switching validation platforms.

------------------------------------------------------------------------

# Main Lesson Learned

Push recovery and step-up/step-down experiments are still necessary.

However, they should validate

> **interaction response**

instead of

> **whether the robot falls.**

Falling is only a boundary condition.

The primary evaluation metrics should be

-   task tracking transient
-   body motion transient
-   recovery time
-   interaction residual
-   apparent interaction dynamics
-   constraint realization

rather than only survival.

------------------------------------------------------------------------

# New Development Strategy

The entire project will be divided into two completely independent stages.

## Stage 1 --- Build and Freeze a Basic Walking Platform

Objective:

Create a simple, repeatable walking platform.

Requirements:

-   stable flat-ground walking
-   torque-level control
-   repeatable behavior
-   no Interaction Dynamics
-   no experiment-specific tuning

After the platform satisfies acceptance criteria, it will be frozen.

No further gait optimization will be allowed.

Interaction Dynamics must never modify the locomotion platform.

------------------------------------------------------------------------

## Stage 2 --- Develop Interaction Dynamics

Only after the walking platform is frozen will the following modules be implemented:

-   normalized interaction model
-   residual estimator
-   desired interaction dynamics
-   constrained MPC
-   whole-body realization

The locomotion platform remains identical for every controller.

------------------------------------------------------------------------

# Development Rules

## Allowed modifications before freezing

-   fix bugs
-   improve swing clearance
-   correct support-transfer timing
-   improve touchdown event handling
-   reduce phase drift
-   improve repeatability

## Not allowed

-   adding locomotion research modules
-   aggressive phase locking
-   controller-specific gait modifications
-   special push recovery logic
-   terrain-specific gait adaptation

------------------------------------------------------------------------

# Walking Platform Acceptance

The walking platform must satisfy:

-   20 s flat-ground walking
-   at least 9/10 successful trials
-   realistic single-support ratio
-   sufficient swing clearance
-   no gait freezing
-   no continuous foot drag
-   repeatable results

Once accepted:

**The locomotion platform is frozen.**

------------------------------------------------------------------------

# Interaction Dynamics Validation

Interaction Dynamics will then be evaluated using

## Standing

-   impulse push
-   sustained force
-   compliance
-   rejection
-   recovery

## Walking

-   push disturbance
-   unexpected step-up
-   unexpected step-down

The locomotion reference is fixed for every controller.

Interaction Dynamics is responsible only for modifying the interaction response.

------------------------------------------------------------------------

# Experimental Philosophy

Experiments will be divided into two regions.

## Region A --- Shared Feasible Region

All controllers can complete the task.

Comparison focuses on:

-   peak error
-   recovery time
-   interaction residual
-   body transient
-   apparent interaction dynamics

These results provide the main scientific evidence.

## Region B --- Failure Boundary

Controllers approach their disturbance limits. Comparison focuses on:

-   maximum recoverable push
-   maximum recoverable terrain height
-   failure probability

These results only demonstrate disturbance envelope expansion.

------------------------------------------------------------------------

# Immediate Technical Plan

## Step 1

Return to the existing locomotion reference. Configuration:

-   original gait
-   stabilizer ON
-   phase synchronization OFF
-   Interaction Dynamics OFF

Evaluate repeatability.

## Step 2

Recover swing clearance. Investigate why clearance decreased after enabling the stabilizer.

## Step 3

Correct support-transfer timing. Analyze scheduled versus measured contacts.

## Step 4

Replace aggressive phase synchronization with a small event-based phase correction. The correction should never suppress stepping.

## Step 5

Achieve a frozen 20-second walking baseline. If successful: Freeze the locomotion platform.

------------------------------------------------------------------------

# Decision Point

If the current walking reference reaches the acceptance criteria within a limited development effort, it will become the permanent validation platform.

If it cannot achieve a reliable frozen baseline without becoming a locomotion research project, the project will migrate to an existing stable Unitree G1 walking platform (for example, the official Unitree MuJoCo/Isaac locomotion baseline), while keeping Interaction Dynamics as the sole research contribution.

------------------------------------------------------------------------

# Guiding Principle

> **Interaction Dynamics is the contribution.**
>
> **Humanoid locomotion is only the validation platform.**
>
> Every future design decision should be evaluated against this
> principle.

------------------------------------------------------------------------

# Stage 2 Validation Plan (reviewer-driven, 2026-07-21)

Stage 2 built a **wrench-unified interaction controller** on the frozen Unitree G1 policy: the external force is estimated from a CoM linear-momentum residual (horizontal components) `F_ext,xy = m·c̈_xy − ΣF_contact,xy`, and its persistence gates a continuous blend of transient *capture* and sustained-force *offset-free hold*. A review found the architecture sound; the remaining risks are **validation**, not architecture. This plan closes them, executed **step by step** (finish, record, and check each before starting the next). All work lives in `code/unitree_baseline/`; results in `code/unitree_baseline/results/`, figures in `.../figures/`, findings appended to `STAGE2_FINDINGS.md`.

## Step V1 — Estimator vs ground-truth force (most decisive)
Log true applied force and estimated `F_ext` together through: nominal walking, foot touchdown, liftoff, step-up, step-down, and commanded turning/lateral acceleration, plus transient impulses and sustained forces. Report **RMSE, peak error, detection delay, decay time, and false-positive rate**. Goal: confirm the estimator does NOT read normal contact transitions or policy-generated acceleration as external interaction (the biggest surprise risk).

## Step V2 — Oracle ablation (upper bound)
Compare six controllers on the same disturbances: (1) policy, (2) capture specialist, (3) hold specialist, (4) CoM-only unified, (5) wrench-unified, (6) **oracle** unified that knows the true disturbance class/duration at t=0. The oracle establishes the ceiling; the oracle→wrench gap isolates the cost of causal wrench detection.

## Step V3 — Held-out tuning (no tune-on-test)
Fix `f_thresh`, timer `tf0`, decay rate, and blend map on ONE calibration set of disturbances, then report HELD-OUT cases: impulses 180/220/260/300 N × 0.10/0.15/0.25/0.40 s; sustained 6/8/10/12/16 N; ramped (not step) forces; and intermittent contact forces.

## Step V4 — Statistical strength
Increase to 30–50 paired seeds around the failure boundary (same process noise, gait init, disturbance phase and magnitude per seed), and compare policy vs unified by paired outcomes.

## Step V5 — Sensor-bias / realism robustness
Sweep total horizontal foot-force bias `b_F ∈ {−5,−3,0,3,5}` N (and unequal bias across feet), plus realistic noise, delay, and low-pass filtering. Critical because the sustained interactions are only 8–12 N.

## Step V6 — Estimator reformulation (optional polish)
Recast the estimator as a filtered linear-momentum-residual observer `\hat F_ext = L(l − \hat l)`, `l = m·ċ`, instead of raw CoM-velocity finite differencing, and confirm V1 metrics hold or improve.

## Honesty caveats to preserve throughout
Estimator inputs are simulation-clean (kinematic CoM velocity from full state; MuJoCo-exact contact forces). Process noise is a *real* applied force, so the wrench estimator sees it. Report as implementable with IMU + estimated CoM velocity + foot six-axis wrenches — **not** hardware-validated.
