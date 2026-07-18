# Change Plan: Interaction-Dynamics MPC for Uneven-Ground Walking

## 1. Revised paper objective

The paper will no longer present the whole-body QP, local authority polytope, or
contact-transition machinery as its main contribution. Its central question is:

> Can a configuration-invariant double-integrator interaction model predict and
> compensate terrain-induced task-acceleration residuals while preserving precise
> body tracking during uneven-ground walking?

The proposed controller does not generate footsteps and does not replace the
whole-body controller (WBC). It receives a nominal walking reference from an
external motion planner, estimates the effect of emerging robot--terrain
interaction on selected body tasks, and supplies constrained acceleration
corrections to a fast WBC.

The contribution is therefore:

1. a fixed double-integrator prediction backbone;
2. an explicit interaction/realization residual estimate and short-horizon
   prediction;
3. constrained, offset-free interaction compensation; and
4. controlled uneven-ground comparisons using the same planner and WBC for all
   methods.

## 2. Scope boundary

### In scope

- body-task interaction dynamics during externally planned walking;
- terrain-induced contact mismatch after it becomes observable;
- task-space residual estimation and prediction;
- constrained acceleration correction;
- a 100--200 Hz interaction MPC;
- a 500 Hz whole-body inverse-dynamics/contact QP, with 1 kHz as a stretch goal;
- flat- and uneven-ground MuJoCo experiments with controlled baselines;
- timing, prediction, tracking, interaction, and realization metrics.

### Out of scope

- footstep planning or gait generation as a new contribution;
- contact-implicit trajectory optimization;
- autonomous terrain perception;
- pre-contact inference of unseen holes from reaction force alone;
- a new general-purpose WBC framework;
- global capability polytopes or KKT continuation as the paper's central result;
- solving every fall-recovery, impact, and hybrid-feasibility problem;
- claims of superiority over all whole-body NMPC or learning-based locomotion.

The planner and WBC are shared experimental infrastructure. They may be repaired
only enough to provide a repeatable, physically constrained comparison platform.

## 3. Target multirate architecture

```text
External motion planner / prerecorded nominal gait
              50--100 Hz
                    |
                    | nominal body and foot trajectories
                    v
Interaction estimator -------------------------------+
  contact force, IMU, task acceleration               |
  500 Hz update; 100--200 Hz prediction publication   |
                    |                                  |
                    v                                  |
Interaction-Dynamics MPC, 100--200 Hz                 |
  fixed double-integrator model                        |
  predicted interaction-residual sequence              |
  constrained task-acceleration correction             |
                    |                                  |
                    v                                  |
Whole-body inverse-dynamics/contact QP, 500 Hz --------+
  rigid-body dynamics, contact, friction, CoP,
  torque and joint limits; soft task tracking
                    |
                    v
Torque hold / MuJoCo integration, 1 kHz
```

The mandatory implementation target is a 500 Hz WBC QP. A measured 1 kHz QP is
a stretch goal and must not be claimed unless its p99 wall-clock time fits the
1 ms deadline on the reported platform.

Recommended initial rates:

| Layer | Initial rate | Deadline |
|---|---:|---:|
| MuJoCo integration / torque hold | 1 kHz | 1.0 ms |
| inverse-dynamics/contact QP | 500 Hz | 2.0 ms |
| residual estimator update | 500 Hz | 2.0 ms shared cycle |
| interaction MPC | 100 Hz | 10 ms |
| external reference publication | 50--100 Hz | planner-dependent |

After correctness is established, evaluate interaction MPC at 100 and 200 Hz.

## 4. Interaction model

For selected body task coordinates `y` and nominal reference `y_d`, retain

```math
e = y-y_d,\qquad \ddot e = v+d_{\mathrm{eff}}.
```

The exact-ZOH predictor remains

```math
x_{k+1}=A_dx_k+B_dv_k+B_d\hat d_{k|k},
\qquad x=[e^\top,\dot e^\top]^\top,
```

with fixed double-integrator matrices `A_d,B_d`.

Use the conceptual decomposition

```math
d_{\mathrm{eff}}=d_{\mathrm{int}}+d_{\mathrm{real}}+d_{\mathrm{mod}},
```

where:

- `d_int` is the task-acceleration effect of contact-force, contact-timing,
  terrain-height, compliance, and friction mismatch;
- `d_real = a_real-a_cmd` is constrained WBC realization error; and
- `d_mod` contains normalization, state-estimation, and unmodeled-dynamics
  error.

The estimator is not required to identify these three sources uniquely. The
control state is their observable combined task-acceleration effect. When
measured contact forces are available, log an interpretable measured component
separately from the remaining estimated residual.

The paper must claim near-future prediction only after interaction mismatch is
observable. It must not claim that reaction force alone detects unseen terrain
before contact.

## 5. Estimation and prediction

Start with a random-walk residual state:

```math
d_{k+1}=d_k+w_k.
```

Estimate it from measured body-task error and realized task acceleration. The
first implementation can use the existing Kalman-style
`RandomWalkDisturbanceObserver`, but its signal definitions must be corrected:

- estimator input: the command actually sent to the WBC;
- estimator output error: measured/estimated task motion;
- realization feedback: the WBC-reported realized-minus-commanded acceleration;
- contact-force-derived term: optional measured explanatory input;
- publication: current estimate plus a horizon sequence.

Compare two prediction models before implementing anything more complicated:

1. constant residual over the MPC horizon;
2. contact-phase scheduled residual, using the external planner's known phase.

Contact-time adaptation is added only if these two models fail for a documented
reason.

## 6. Interaction-Dynamics MPC

The MPC decision is the task-acceleration correction sequence `V`, not contact
force, torque, or footstep location. Use

```math
\min_V \sum_i \|e_i\|_Q^2
       +\sum_i \|v_i\|_R^2
       +\sum_i \|\Delta v_i\|_S^2,
```

subject to the fixed augmented double-integrator model and task-acceleration
bounds. The first version should use transparent, mode-scheduled bounds derived
from WBC operating limits. A lightweight realization-informed tightening may be
added later, but KKT continuation is not required for the main experiment.

The task set should initially be limited to quantities needed to test the
hypothesis:

- lateral and vertical CoM/body position;
- roll and pitch or an explicitly defined angular-momentum surrogate.

Hand manipulation and a full six-dimensional body--task allocation are removed
from the main experimental path.

## 7. Fast WBC realization layer

Reuse the existing `InverseDynamicsQPRealizer` as infrastructure. Its required
properties are:

- solve at 500 Hz using the current measured state;
- hard rigid-body dynamics, rigid active contact, unilateral force, friction,
  CoP, torque, and one-step joint limits;
- soft body and swing-foot task tracking;
- warm-started, fixed-sparsity solver where possible;
- explicit requested-versus-realized task acceleration;
- no authority search or gait planning inside the 500 Hz cycle.

Only the engineering required to meet this interface is in scope: preallocation,
fixed sparsity, solver reuse, and removal of avoidable Python matrix rebuilding.
Do not redesign priorities, impact control, or recovery behavior unless a failure
blocks the nominal shared benchmark.

Timing acceptance at 500 Hz:

- median WBC QP path below 2.0 ms;
- p99 reported explicitly, with deadline-miss fraction;
- no claim of hard real time unless every measured update meets the deadline;
- CPU, solver settings, warm-up, and sample count reported.

## 8. External motion planner contract

Do not develop a new gait generator for this paper. Define a `ReferenceProvider`
interface returning, at time `t`:

- desired body pose, velocity, and acceleration;
- desired swing-foot pose, velocity, and acceleration;
- nominal contact schedule;
- optional nominal contact forces.

Use either an existing validated planner or prerecorded nominal flat-ground
trajectories. The identical reference must be replayed for every controller.
Terrain is changed underneath that plan so the experiment isolates interaction
mismatch rather than replanning quality.

The currently missing `run_gait_dcm.DCMWalk` dependency must not be rebuilt into
a new research contribution. Either restore it as a simple external reference
provider or replace it with a validated existing reference source.

## 9. Controller comparisons

Use the same planner, WBC, state estimator, contact logic, torque limits, and
terrain realization for every method.

### Required baselines

1. **Conventional task impedance:** fixed body/swing-foot feedback producing WBC
   task accelerations. This represents compliance through tracking error.
2. **Nominal double-integrator MPC:** the same MPC and constraints as the proposed
   method, but `d_hat=0`. This isolates residual augmentation.
3. **Interaction-Dynamics MPC:** estimated/predicted `d_eff` included in the
   double-integrator horizon.

### Useful ablation

4. **Interaction MPC without realization feedback:** estimate terrain/contact
   residual but omit WBC realization residual. This tests the prediction--
   realization feedback path.

Do not compare against an unrelated full NMPC or RL implementation unless an
existing, validated controller can be run under the same robot, terrain,
reference, and metrics. Literature comparisons remain qualitative otherwise.

## 10. Terrain benchmark

### Development gate

Before uneven terrain, all three required controllers must complete the same
nominal flat-ground sequence with the shared planner/WBC. If the baseline WBC
cannot do this, stop and repair or replace only the infrastructure boundary; do
not tune the interaction MPC to hide a gait or WBC failure.

### Minimum publishable terrains

1. **Flat ground:** control and estimator sanity check.
2. **Unilateral depression:** one planned foothold is 20 mm below nominal.
3. **Unilateral obstacle:** one planned foothold is 20 mm above nominal.
4. **Low-amplitude rough sequence:** frozen alternating/random foothold heights,
   initially within +/-20 mm.

After the minimum set passes, sweep terrain amplitude to determine the failure
boundary rather than selecting only successful examples. Low friction or soft
contact is an optional second study, not part of the minimum result.

Each trial must use identical initial state, planner reference, terrain, and
seed across controllers. Start with five development seeds; freeze at least ten
evaluation seeds per terrain after controller parameters are fixed.

## 11. Hypotheses and metrics

### H1: invariant predictor construction

`A_d,B_d` remain identical across gait phase and terrain. Report this as a
construction fact, not as a performance result.

### H2: interaction prediction

The residual-augmented model predicts body motion more accurately than the
nominal double integrator after contact mismatch emerges.

Report horizon-indexed RMSE for predicted lateral/vertical body position and
roll/pitch, plus residual-prediction RMSE.

### H3: precise uneven-ground tracking

Interaction-Dynamics MPC reduces RMS and peak body-position/orientation error
relative to impedance and nominal MPC under the same terrain interaction.

Report RMS, peak, and post-interaction bias for each controlled coordinate.

### H4: interaction and realization response

Interaction-Dynamics MPC reduces recovery time and avoids repeatedly requesting
unrealizable acceleration when WBC constraints become active.

Report:

- requested and realized task acceleration;
- realization-residual RMS and peak;
- command-bound activity;
- torque/friction/CoP margins;
- contact-force peak and impulse;
- recovery time;
- fall and QP-fallback counts as secondary outcomes.

### H5: computational feasibility

Report median, p99, maximum, and deadline-miss fraction separately for:

- 500 Hz WBC QP;
- 100/200 Hz interaction estimator and MPC;
- total combined wall-clock workload.

## 12. File-level implementation plan

### Retain and narrow

- `normalized_mpc.py`: retain the exact-ZOH double integrator and Kalman-style
  observer. Add horizon disturbance-sequence input and input-rate penalty only
  if not already supported.
- `run_g1_torque_realizer_benchmark.py`: retain the reusable
  `InverseDynamicsQPRealizer`; stop using this file as the paper's gait planner.
- `realization_authority.py`: retain as an optional diagnostic or constraint-
  tightening experiment; remove it from the main paper claim chain.

### Refactor

- `multirate_control.py`: split the present 200 Hz combined node into independent
  `interaction_mpc_dt`, `wbc_dt`, and `servo_dt` loops. The WBC must consume the
  latest acceleration correction at 500 Hz and publish realized acceleration
  back to the estimator.
- `run_multirate_benchmarks.py`: reduce to rate/timing and signal-contract tests,
  or replace with the new paper benchmark below.

### Add

- `reference_provider.py`: adapter for an existing or prerecorded nominal gait.
- `interaction_estimator.py`: residual definitions, measured component, Kalman
  estimate, and horizon publication.
- `run_uneven_ground_benchmark.py`: paired controller/terrain/seed runner.
- `make_uneven_ground_figures.py`: prediction, tracking, interaction, and timing
  figures generated only from the benchmark JSON/NPZ.
- `results/uneven_ground_benchmark.json`: single authoritative paper artifact.
- `verify_interaction_paper_claims.py`: exact scenario, field, number, and figure
  checks; it must reject missing scenarios and stale manuscript values.

## 13. Staged execution and stop/go gates

### Stage 0: remove stale claim coupling

- Freeze the current paper and artifacts as an archive.
- Mark KKT continuation, five-transfer, and task-port tables as non-authoritative.
- Create one new benchmark manifest containing code version, parameters, and
  output hashes.

**Gate:** no new manuscript number is entered manually without a JSON source.

### Stage 1: establish the shared walking substrate

- Restore/attach the external nominal reference.
- Run the WBC QP at 500 Hz and simulation at 1 kHz.
- Verify flat-ground walking without interaction compensation.

**Gate:** repeated nominal walking completes with no QP fallback and acceptable
body/swing-foot tracking. Otherwise repair or replace the infrastructure before
continuing.

### Stage 2: verify the residual signal

- Log command, realized task acceleration, measured contact force, and body
  motion at the WBC rate.
- Validate signs, units, time alignment, and observability.
- Compare constant and phase-scheduled residual prediction.

**Gate:** augmented prediction must beat the nominal model on held-out terrain
contacts before closed-loop performance claims are attempted.

### Stage 3: controlled controller comparison

- Freeze gains and run impedance, nominal MPC, and Interaction-Dynamics MPC on
  the minimum terrain set.
- Add the realization-feedback ablation.
- Sweep terrain amplitude after the nominal experiments are frozen.

**Gate:** proposed improvements must appear in tracking/prediction metrics, not
only in fall count.

### Stage 4: computational validation

- Profile WBC and MPC separately with warm-up and repeated trials.
- Optimize only matrix assembly, fixed sparsity, warm starts, and avoidable
  allocations required for the rate contract.

**Gate:** claim 500 Hz only if the measured timing supports it. Treat 1 kHz WBC
as a stretch result.

### Stage 5: rewrite the paper from verified evidence

- Rewrite the Markdown first.
- Sync to LaTeX only after equations, hypotheses, and experiment numbers are
  frozen.
- Rebuild all figures from the authoritative benchmark.
- Run numerical claim verification and visual PDF inspection.

## 14. Proposed paper structure

1. **Introduction:** terrain interaction creates a precision--compliance tradeoff.
2. **Related Work:** terrain locomotion MPC, adaptive/disturbance control,
   impedance, and interaction MPC; acknowledge real-time full-order NMPC and
   learning-based terrain locomotion.
3. **Problem and Scope:** external planner, selected body tasks, WBC interface,
   uneven-contact mismatch.
4. **Configuration-Invariant Interaction Dynamics:** normalization, fixed
   double integrator, residual decomposition, exact ZOH.
5. **Interaction Estimation and Prediction:** measured component, effective
   residual observer, horizon sequence, observability limitations.
6. **Constrained Interaction-Dynamics MPC:** optimization, offset-free property,
   bounds, realization feedback.
7. **Fast Whole-Body Realization:** short infrastructure section and rate
   contract.
8. **Uneven-Ground Experiments:** prediction, tracking, realization, timing, and
   controlled comparisons.
9. **Limitations and Conclusion:** no new planner, no unseen-terrain inference,
   no robust-walking or universal-superiority claim.

## 15. Paper-level success criterion

The revised paper is successful if it demonstrates, under an identical external
walking plan and high-rate constrained WBC, that explicitly estimating and
predicting the effective interaction acceleration improves near-future body-
motion prediction and precise tracking over uneven ground relative to both
conventional impedance and the same MPC without residual augmentation.

It is not necessary to solve autonomous gait planning, general fall recovery,
or every WBC contact-transition problem to establish this result.
