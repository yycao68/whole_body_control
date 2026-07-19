# Change Plan: Interaction-Dynamics MPC for Environmental Disturbances

## 1. Revised paper objective

The paper will no longer present the whole-body QP, local authority polytope, contact-transition machinery, or general push recovery as its main contribution. Its central question is:

> Can a configuration-invariant double-integrator interaction model predict the observable motion effect of environmental disturbances and improve body tracking during locomotion, with the gait plan and whole-body realizer held fixed?

Environmental interaction is evaluated in two complementary classes:

1. **terrain-mediated interaction**, including height mismatch, early or delayed touchdown, impact, and support-force redistribution; and
2. **externally applied body interaction**, including finite-duration pushes or pulls applied to the torso during locomotion.

The proposed controller does not generate footsteps, modify the contact schedule, or replace the whole-body controller (WBC). It receives a nominal walking reference from an external motion planner, estimates the observable task-acceleration effect of an interaction after it occurs, and supplies constrained acceleration corrections to the shared WBC.

The contribution is therefore:

1. a fixed double-integrator prediction backbone;
2. an explicit interaction/realization residual estimate and short-horizon prediction;
3. constrained, offset-free interaction compensation; and
4. controlled terrain and external-push comparisons using the same planner and WBC for all methods.

## 2. Scope boundary

### In scope

- body-task interaction dynamics during externally planned walking;
- terrain-induced contact mismatch after it becomes observable;
- external torso forces after their motion effect becomes observable;
- task-space residual estimation and prediction;
- constrained acceleration correction;
- a 100 Hz interaction MPC;
- a whole-body inverse-dynamics/contact QP scheduled at 500 Hz in simulation, with wall-clock timing reported separately;
- flat-ground, uneven-ground, and controlled-push MuJoCo experiments;
- timing, prediction, tracking, interaction, and realization metrics.

### Out of scope

- footstep planning or gait generation as a new contribution;
- contact-implicit trajectory optimization;
- autonomous terrain perception;
- pre-contact inference of unseen holes from reaction force alone;
- a new general-purpose WBC framework;
- global capability polytopes or KKT continuation as the paper's central result;
- solving every fall-recovery, impact, and hybrid-feasibility problem;
- footstep or contact-schedule adaptation in response to a push;
- claiming capture-point recovery, viability guarantees, or maximum rejectable push;
- claims of superiority over all whole-body NMPC or learning-based locomotion.

The planner and WBC are shared experimental infrastructure. They may be repaired only enough to provide a repeatable, physically constrained comparison platform. A trial that remains upright is not by itself evidence of improved interaction control; prediction, tracking, and recovery metrics remain primary.

## 3. Target multirate architecture

```text
External motion planner / prerecorded nominal gait
              50--100 Hz
                    |
                    | nominal body and foot trajectories
                    v
Interaction estimator -------------------------------+
  contact force, IMU, task acceleration               |
  applied wrench (ground-truth logging only)          |
  500 Hz simulated update; 100 Hz publication         |
                    |                                  |
                    v                                  |
Interaction-Dynamics MPC, 100 Hz                      |
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

The experiments preserve a 500 Hz simulated WBC schedule so that all controllers use the same multirate architecture. This is not a demonstrated real-time rate: the completed terrain benchmark measured 2.77 ms median and 7.71 ms p99 against a 2 ms deadline. The next implementation target is therefore to reduce WBC latency or clearly retain the result as accelerated/offline simulation. A 1 kHz WBC is removed from the current paper plan.

Recommended initial rates:

| Layer | Initial rate | Deadline |
|---|---:|---:|
| MuJoCo integration / torque hold | 1 kHz | 1.0 ms |
| inverse-dynamics/contact QP | 500 Hz | 2.0 ms |
| residual estimator update | 500 Hz | 2.0 ms shared cycle |
| interaction MPC | 100 Hz | 10 ms |
| external reference publication | 50--100 Hz | planner-dependent |

The 100 Hz MPC already meets its measured deadline. A 200 Hz comparison is optional and is not required for the push study unless estimator bandwidth becomes the documented bottleneck.

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
d_{\mathrm{eff}}=d_{\mathrm{terrain}}+d_{\mathrm{ext}}
                 +d_{\mathrm{real}}+d_{\mathrm{mod}},
```

where:

- `d_terrain` is the task-acceleration effect of contact-force, contact-timing,
  terrain-height, compliance, and friction mismatch;
- `d_ext` is the task-acceleration effect of an applied body wrench, payload,
  collision, pull, or push;
- `d_real = a_real-a_cmd` is constrained WBC realization error; and
- `d_mod` contains normalization, state-estimation, and unmodeled-dynamics
  error.

The estimator is not required to identify these sources uniquely. The control state is their observable combined task-acceleration effect. Terrain experiments log foot-contact forces; push experiments additionally log the ground-truth applied wrench for evaluation. Unless that wrench is explicitly supplied to the controller in a separate feedforward ablation, it must not enter the causal estimator input.

The paper must claim near-future prediction only after interaction mismatch is observable. It must not claim that reaction force detects unseen terrain before contact or that the estimator anticipates an unmeasured push before its dynamical effect appears.

## 5. Estimation and prediction

Start with a random-walk residual state:

```math
d_{k+1}=d_k+w_k.
```

Estimate it from measured body-task error and realized task acceleration. The first implementation can use the existing Kalman-style `RandomWalkDisturbanceObserver`, but its signal definitions must be corrected:

- estimator input: the command actually sent to the WBC;
- estimator output error: measured/estimated task motion;
- realization feedback: the WBC-reported realized-minus-commanded acceleration;
- contact-force-derived term: optional measured explanatory input;
- applied-wrench signal: ground-truth logging only in the primary push test;
- publication: current estimate plus a horizon sequence.

Compare two prediction models before implementing anything more complicated:

1. constant residual over the MPC horizon;
2. contact-phase scheduled residual, using the external planner's known phase.

For pushes, also compare the constant-residual predictor with a finite-duration
oracle-wrench rollout only as a noncausal upper bound. The oracle must not be
presented as a competing controller.

Contact-time adaptation is added only if these two models fail for a documented reason.

## 6. Interaction-Dynamics MPC

The MPC decision is the task-acceleration correction sequence `V`, not contact force, torque, or footstep location. Use

```math
\min_V \sum_i \|e_i\|_Q^2
       +\sum_i \|v_i\|_R^2
       +\sum_i \|\Delta v_i\|_S^2,
```

subject to the fixed augmented double-integrator model and task-acceleration bounds. The completed terrain benchmark uses the same transparent fixed bounds and command-slew limit for all MPC controllers. State- or mode-dependent realization-informed tightening remains a separate optional study; it must not be described as part of the evaluated controller unless it is added and the full terrain and push matrices are regenerated. KKT continuation is not required for the main experiment.

The task set should initially be limited to quantities needed to test the hypothesis:

- lateral and vertical CoM/body position;
- roll and pitch or an explicitly defined angular-momentum surrogate.

Hand manipulation and a full six-dimensional body--task allocation are removed from the main experimental path.

## 7. Fast WBC realization layer

Reuse the existing `InverseDynamicsQPRealizer` as infrastructure. Its required properties are:

- solve at 500 Hz using the current measured state;
- hard rigid-body dynamics, rigid active contact, unilateral force, friction, and torque limits;
- soft body and swing-foot task tracking;
- warm-started, fixed-sparsity solver where possible;
- explicit requested-versus-realized task acceleration;
- no authority search or gait planning inside the 500 Hz cycle.

Only the engineering required to meet this interface is in scope: preallocation, fixed sparsity, solver reuse, and removal of avoidable Python matrix rebuilding. Do not redesign priorities, impact control, or recovery behavior unless a failure
blocks the nominal shared benchmark.

Timing acceptance at 500 Hz:

- median WBC QP path below 2.0 ms;
- p99 reported explicitly, with deadline-miss fraction;
- no claim of hard real time unless every measured update meets the deadline;
- CPU, solver settings, warm-up, and sample count reported.

## 8. External motion planner contract

Do not develop a new gait generator for this paper. Define a `ReferenceProvider` interface returning, at time `t`:

- desired body pose, velocity, and acceleration;
- desired swing-foot pose, velocity, and acceleration;
- nominal contact schedule;
- optional nominal contact forces.

Use either an existing validated planner or prerecorded nominal flat-ground trajectories. The identical reference must be replayed for every controller. Terrain is changed underneath that plan so the experiment isolates interaction
mismatch rather than replanning quality.

The currently missing `run_gait_dcm.DCMWalk` dependency must not be rebuilt into a new research contribution. Either restore it as a simple external reference provider or replace it with a validated existing reference source.

## 9. Controller comparisons

Use the same planner, WBC, state estimator, contact logic, torque limits, terrain realization, and external-force schedule for every method.

### Required baselines

1. **Conventional task impedance:** fixed body/swing-foot feedback producing WBC task accelerations. This represents compliance through tracking error.
2. **Nominal double-integrator MPC:** the same MPC and constraints as the proposed method, but `d_hat=0`. This isolates residual augmentation.
3. **Interaction-Dynamics MPC:** estimated/predicted `d_eff` included in the double-integrator horizon.

### Useful ablation

4. **Interaction MPC without realization feedback:** estimate terrain/contact residual but omit WBC realization residual. This tests the prediction--realization feedback path.

Do not compare against an unrelated full NMPC or RL implementation unless an existing, validated controller can be run under the same robot, terrain, reference, and metrics. Literature comparisons remain qualitative otherwise.

## 10. Terrain benchmark

### Development gate

Before uneven terrain, all three required controllers must complete the same nominal flat-ground sequence with the shared planner/WBC. If the baseline WBC cannot do this, stop and repair or replace only the infrastructure boundary; do not tune the interaction MPC to hide a gait or WBC failure.

### Minimum publishable terrains

1. **Flat ground:** control and estimator sanity check.
2. **Unilateral depression:** one planned foothold is 20 mm below nominal.
3. **Unilateral obstacle:** one planned foothold is 20 mm above nominal.
4. **Low-amplitude rough sequence:** frozen alternating/random foothold heights, initially within +/-20 mm.

The completed minimum benchmark uses the fixed amplitudes above. A terrain-amplitude failure-boundary sweep was not performed and must remain a stated limitation. Low friction, soft contact, and an amplitude sweep are optional follow-up studies rather than prerequisites for the push experiment.

Each trial must use identical initial state, planner reference, terrain, and seed across controllers. Start with five development seeds; freeze at least ten evaluation seeds per terrain after controller parameters are fixed.

The completed terrain benchmark is retained as the first evidence block: 160 trials, zero QP fallbacks, modest prediction gains on three terrains, a 13.1% obstacle peak-error reduction relative to nominal MPC, mixed RMS results, and an unfavorable realization-feedback ablation. These outcomes must remain visible when the push study is added; the push results cannot be used to conceal the terrain regressions.

## 11. External-push benchmark

### Experimental principle

Apply a known finite-duration wrench to the torso while replaying the same nominal gait. The applied wrench changes only the simulated plant. It is logged at 1 kHz for ground-truth evaluation but is hidden from the estimator and controllers in the primary comparison. All controllers retain the same contact plan, reference, WBC, limits, estimator measurements, and seed.

### Minimum publishable conditions

Use two push directions and two gait phases:

1. lateral push during double support;
2. lateral push during single support;
3. forward or backward push during double support; and
4. forward or backward push during single support.

Begin with a rectangular or half-sine torso force lasting 100--200 ms. Select one moderate impulse that all required controllers can complete, then freeze it before evaluation. A small three-level impulse sweep may be added to show the response boundary, but do not tune each controller at a different force. Push onset must be phase-locked using the shared reference/contact schedule, with measured contact mode logged for audit.

Run the same four controllers and at least ten paired seeds per condition. With four push conditions, this adds 160 primary trials. Flat walking without a push is already available from the terrain benchmark and need not be rerun unless code changes alter the controller or plant.

### Push metrics

Report:

- one-, five-, and ten-millisecond task-motion prediction RMSE after push onset;
- peak and RMS CoM and roll/pitch error;
- recovery time to a predeclared error band and dwell time;
- maximum body displacement and attitude excursion;
- requested, realized, and residual task acceleration;
- applied force, duration, impulse, point of application, and resulting moment;
- contact-force redistribution, foot slip, torque utilization, and QP fallback;
- contact-plan deviation and fall count as secondary outcomes.

Recovery time must be defined before running the final seeds, for example as the first post-push time at which the controlled error norm remains below a frozen threshold for 200 ms. A controller that merely commands less or falls later is not considered better without improved prediction or tracking.

### Push-study ablations

The required causal comparison uses the same four controllers as the terrain study. Two optional diagnostics are permitted:

1. **measured-wrench feedforward**, in which the known simulated wrench is supplied to the controller, to separate estimation delay from control limitation; and
2. **oracle residual rollout**, which uses the recorded future wrench only offline to bound prediction performance.

Neither diagnostic is a deployable baseline. Footstep replanning, capture-step logic, and contact-schedule changes remain disabled because they would change the research question.

### Continuous 15 s interaction challenge

Add one reader-facing continuous vignette after the controlled terrain and
phase-locked push matrices.  It is a stress test, not a replacement for those
factorial experiments:

1. `0--2 s`: nominal flat walking;
2. `2--3 s`: a smooth flat-top lateral torso force with 0.10 s ramps;
3. `3--5.2 s`: unforced recovery while the nominal gait continues;
4. approximately `5 s`: physical step-up onto a spatial box;
5. a later measured step-down from the box; and
6. continue the same unreplanned gait until `15 s`.

The ground geometry must remain fixed in world coordinates.  Step-up and
step-down times are reported from measured foot height/contact, not assumed
from the reference clock.  Use the paired flat-gate gait (`30 mm` step length,
`1.40 s` step period, `1.00 s` double support) until a separate walking planner
is introduced; both MPC variants complete its 15 s flat test.  Do not claim the
originally suggested `0.8 m/s`, which the present reference does not generate.

Development must remain staged: 15 s flat gate, sustained-force boundary
sweep, platform-height boundary sweep, and only then the combined case.  The
initial 30/50/70 N sustained-force proposal is intentionally treated as a
boundary probe.  Preliminary seed 4300 tests on the earlier 20 mm gait show
that all three forces make both MPC variants fall.  On the frozen 30 mm gait,
both controllers complete at 5 N, while nominal MPC falls late at 10 and 15 N
and Interaction-Dynamics MPC completes.  The common 5 N condition is not
favorable to the interaction controller in full-trial peaks, however: its
single-seed lateral/roll excursions are larger.  Preserve 5 N as the paired
completion condition and 10/15 N as separate failure-boundary probes; do not
pool or selectively promote them.  These single-seed development results are
not paper evidence, and event-window peaks must be separated from later gait
excursions before any causal interpretation.

The first fixed-platform development sweep is a failed infrastructure gate:
with the 30 mm gait, both controllers fall near the same support transfer after
stepping onto each 20/30/40 mm block.  A longer plateau does not move the
failure time, showing that the immediate blocker is elevated-support/contact
realization rather than the exit edge.  In the 15 N + 20 mm combined diagnostic,
nominal MPC falls at 8.606 s and Interaction-Dynamics MPC at 8.828 s.  The
interaction controller's smaller lateral peak is diagnostic only; neither
trial supports a recovery or robustness claim.  Per the scope boundary, stop
here rather than redesigning the WBC inside the MPC paper.  The controlled
flat-push study may proceed independently, while the platform vignette remains
an explicit limitation until shared locomotion infrastructure is repaired.

Report peak lateral error, peak roll, recovery to a frozen error band for at
least 250 ms, measured step event times, fall count, QP fallback, and the same
prediction/realization metrics as the controlled studies.  A trial that later
falls receives no recovery time even if it briefly re-enters the error band.
Generate a synchronized side-by-side video only after the frozen combined
condition completes for both controllers.

## 12. Hypotheses and metrics

### H1: invariant predictor construction

`A_d,B_d` remain identical across gait phase, terrain, and applied-push condition. Report this as a construction fact, not as a performance result.

### H2: interaction prediction

The residual-augmented model predicts body motion more accurately than the nominal double integrator after terrain mismatch or an external push becomes observable.

Report horizon-indexed RMSE for predicted lateral/vertical body position and roll/pitch, plus residual-prediction RMSE.

### H3: terrain-interaction tracking

Interaction-Dynamics MPC reduces RMS and peak body-position/orientation error relative to impedance and nominal MPC under the same terrain interaction.

Report RMS, peak, and post-interaction bias for each controlled coordinate.

### H4: interaction and realization response

Interaction-Dynamics MPC reduces recovery time and avoids repeatedly requesting unrealizable acceleration when WBC constraints become active.

Report:

- requested and realized task acceleration;
- realization-residual RMS and peak;
- command-bound activity;
- torque/friction/CoP margins;
- contact-force peak and impulse;
- recovery time;
- fall and QP-fallback counts as secondary outcomes.

The completed terrain result does not support a general realization-feedback advantage. H4 is therefore retained as an open hypothesis for the deliberately applied-wrench experiment, not carried forward as an established claim.

### H5: external-push response

Relative to nominal MPC and impedance, residual augmentation should reduce post-push prediction error, peak body error, or recovery time under the same applied impulse. Improvements must be reported separately for direction and support phase; they may not be pooled into a single favorable number.

### H6: computational feasibility

Report median, p99, maximum, and deadline-miss fraction separately for:

- 500 Hz WBC QP;
- 100 Hz interaction estimator and MPC;
- total combined wall-clock workload.

## 13. File-level implementation plan

### Retain and narrow

- `normalized_mpc.py`: retain the exact-ZOH double integrator and Kalman-style observer. Add horizon disturbance-sequence input and input-rate penalty only if not already supported.
- `run_g1_torque_realizer_benchmark.py`: retain the reusable `InverseDynamicsQPRealizer`; stop using this file as the paper's gait planner.
- `realization_authority.py`: retain as an optional diagnostic or constraint-tightening experiment; remove it from the main paper claim chain.

### Refactor

- `multirate_control.py`: split the present 200 Hz combined node into independent `interaction_mpc_dt`, `wbc_dt`, and `servo_dt` loops. The WBC must consume the latest acceleration correction at 500 Hz and publish realized acceleration back to the estimator.
- `run_multirate_benchmarks.py`: reduce to rate/timing and signal-contract tests, or replace with the new paper benchmark below.

### Add

- `reference_provider.py`: adapter for an existing or prerecorded nominal gait.
- `interaction_estimator.py`: residual definitions, measured component, Kalman estimate, and horizon publication.
- `run_uneven_ground_benchmark.py`: paired controller/terrain/seed runner.
- `make_uneven_ground_figures.py`: prediction, tracking, interaction, and timing figures generated only from the benchmark JSON/NPZ.
- `results/uneven_ground_benchmark.json`: single authoritative paper artifact.
- `run_external_push_benchmark.py`: phase-locked torso-wrench benchmark using the same controller factory, reference provider, realizer, metrics, and seed pairing as the terrain runner.
- `make_external_push_figures.py`: direction/phase summary, representative response, prediction, and recovery plots.
- `results/external_push_benchmark.json`: authoritative push-study artifact, separate from the frozen terrain JSON.
- `run_continuous_interaction_challenge.py`: staged 15 s flat/push/platform runner and optional synchronized video.
- `results/continuous_interaction_challenge.json`: development-to-evaluation artifact for the continuous vignette; it does not replace the two primary benchmark artifacts.
- `verify_interaction_paper_claims.py`: exact terrain and push scenario, field, number, and figure checks; it must reject missing scenarios and stale manuscript values.

## 14. Staged execution and stop/go gates

### Stage 0: remove stale claim coupling

- Freeze the current paper and artifacts as an archive.
- Mark KKT continuation, five-transfer, and task-port tables as non-authoritative.
- Create one new benchmark manifest containing code version, parameters, and output hashes.

**Gate:** no new manuscript number is entered manually without a JSON source.

### Stage 1: establish the shared walking substrate

- Restore/attach the external nominal reference.
- Run the WBC QP at 500 Hz and simulation at 1 kHz.
- Verify flat-ground walking without interaction compensation.

**Gate:** repeated nominal walking completes with no QP fallback and acceptable body/swing-foot tracking. Otherwise repair or replace the infrastructure before continuing.

### Stage 2: verify the residual signal

- Log command, realized task acceleration, measured contact force, and body motion at the WBC rate.
- Validate signs, units, time alignment, and observability.
- Compare constant and phase-scheduled residual prediction.

**Gate:** augmented prediction must beat the nominal model on held-out terrain contacts before closed-loop performance claims are attempted.

### Stage 3A: controlled terrain comparison -- complete

- Freeze gains and run impedance, nominal MPC, and Interaction-Dynamics MPC on the minimum terrain set.
- Add the realization-feedback ablation.
- Preserve the completed 160-trial JSON and its SHA-256 as the terrain evidence source.

**Outcome:** partial pass. Prediction improves modestly on three terrains and obstacle peak error improves, but tracking does not improve uniformly and realization feedback is not supported.

### Stage 3B: controlled external-push comparison -- next

- Add a torso-wrench schedule independent of the controller.
- Validate applied force, point, duration, impulse, onset phase, and logging.
- Select and freeze one moderate impulse using development seeds only.
- Run four controllers, four direction/phase conditions, and ten paired seeds.
- Generate the push summary, representative response, prediction, and recovery figures from the authoritative push JSON/NPZ.

**Gate:** any push claim must be supported by prediction, peak-error, or recovery-time metrics. Remaining upright or changing fall count alone is insufficient.

### Stage 4: computational validation -- terrain measurement complete, optimization open

- Profile WBC and MPC separately with warm-up and repeated trials.
- Optimize only matrix assembly, fixed sparsity, warm starts, and avoidable  allocations required for the rate contract.

**Gate:** do not claim real-time 500 Hz with the current Python measurement. Re-profile after any solver implementation change and keep simulated schedule separate from wall-clock capability.

### Stage 5: rewrite the paper from verified evidence

- Rewrite the Markdown first.
- Sync to LaTeX only after equations, hypotheses, and experiment numbers are  frozen.
- Rebuild all figures from the authoritative benchmark.
- Run numerical claim verification and visual PDF inspection.

## 15. Proposed paper structure

1. **Introduction:** locomotion is disturbed by both terrain-mediated contact mismatch and external body wrenches.
2. **Related Work:** terrain locomotion MPC, push recovery, disturbance observers, impedance, and interaction MPC; distinguish fixed-plan disturbance compensation from footstep/capture-point recovery.
3. **Problem and Scope:** external planner, selected body tasks, WBC interface, terrain mismatch, and applied-wrench disturbance.
4. **Configuration-Invariant Interaction Dynamics:** normalization, fixed double integrator, residual decomposition, exact ZOH.
5. **Interaction Estimation and Prediction:** measured component, effective residual observer, horizon sequence, observability limitations.
6. **Constrained Interaction-Dynamics MPC:** optimization, offset-free property, bounds, realization feedback.
7. **Fast Whole-Body Realization:** short infrastructure section and rate contract.
8. **Environmental-Interaction Experiments:** terrain study, external-push study, prediction, tracking, recovery, realization ablation, and timing.
9. **Limitations and Conclusion:** no new planner, no disturbance anticipation, no general push-recovery guarantee, and no universal-superiority claim.

## 16. Paper-level success criterion

The revised paper is successful if it demonstrates, under an identical external walking plan and constrained WBC, that one canonical residual-acceleration model explains and predicts two distinct interaction classes: terrain-mediated contact mismatch and externally applied body force. Performance claims must remain condition-specific. The terrain study already establishes only a limited prediction and obstacle-peak benefit; the push study must independently determine whether residual augmentation improves post-push prediction, peak tracking error, or recovery time relative to conventional impedance and nominal MPC.

It is not necessary to solve autonomous gait planning, capture-step selection, general fall recovery, or every WBC contact-transition problem. If the push study is also mixed, the publishable contribution should be framed as a reproducible interaction-model evaluation with clear validity boundaries rather than a broad robustness improvement.
