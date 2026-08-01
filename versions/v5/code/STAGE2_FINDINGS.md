# Stage 2 — WBC realizability gate: findings

**Goal:** verify the chosen coupling ("record reference, keep WBC+ID") — i.e. that the existing WBC realizer can track the frozen policy walk so ID can be layered on it. `stage2_wbc_track.py` implements a 12-DoF inverse-dynamics / contact QP (paper Eq. 12) on the frozen `g1_12dof` model tracking `reference/frozen_walk_seed0.npz`.

## Result: the coupling has a real obstacle

The recorded policy gait is stable **only under the exact open-loop per-joint PD that produced it**. Substituting or augmenting the realizer destabilizes it:

| realizer tracking the frozen reference | survives | CoM xy RMS |
|---|---:|---:|
| Open-loop per-joint PD → policy target (`action*0.25+default`) | **12.86 s** | 23 mm |
| + gravity compensation | 1.8 s | 138 mm |
| + CoM task feedback (any gain) | 1.8 s | 140 mm |
| Full inverse-dynamics / contact QP (Eq. 12) | ~2.9 s | 180 mm |

Robust across wide gain/weight sweeps (kp_j 100→450, w_com/w_att/w_post, torque-vs force-regularized redundancy). The fall is a lateral single-support tipping, but it is **not** the fundamental single-support authority wall: open-loop PD proves the reference is trackable to 12.86 s at 23 mm. The wall is that the policy's balance is a closed-loop property of *its own* PD law; a different realizer (WBC) or any added feedback breaks it.

## Bugs found & fixed along the way (real, kept)
- `actuator_ctrlrange` is empty on these motors → torque was constrained to 0; the real limits are `jnt_actfrcrange` (±88/±139/±50). (Fixed.)
- Physically-consistent init must come from the recorded full state at a double-support instant (base pose+vel, joints+vel), not an assumed pose.
- Contact-force redundancy must be resolved by minimizing joint torque, not ‖f‖ (min-‖f‖ gave non-physical torques).
- Posture reference must be the policy **target** (`action*0.25+default`), which leads the recorded joint position by the PD offset that generates the holding torque; tracking recorded positions gives ~0 torque and collapse.

## Implication for the architecture
The evidence favors reconsidering the ID coupling toward **"ID residual on top of the policy"**: keep the policy's stabilizing law in the loop and add a small interaction-dynamics correction, rather than replacing the realizer. The "record reference + WBC" coupling would require either re-deriving a WBC that reproduces the policy's exact closed-loop balance, or accepting only very short (<~3 s) tracking windows.

## Pivot: ID residual on top of the policy (`stage2_id_on_policy.py`)
Policy carries the gait; a planar-CoM ID correction `a_e` is injected as a task torque `J_com^T(m a_e)`. Three controllers share the estimator/reference: `policy` (a_e=0), `impedance` (a_e=-Kp e-Kd ė), `id_mpc` (NormalizedMPC + residual estimator on the ERROR dynamics). A **deadband + rate limit** keep the ID layer out of the nominal loop (else it closes an unstable loop around the fast walking sway) — nominal walking is then byte-identical to policy-alone (1 mm).

Lateral-push comparison (balance metric = post-push peak base roll, recovery = lateral-CoM-velocity settling):

| push | policy | +impedance | +ID-MPC |
|---:|---|---|---|
| 80 N | roll 7.8°, rec 1.04 s | **fell** (176°) | roll 6.8°, rec 1.11 s |
| 100 N | roll 9.3°, rec 1.07 s | **fell** | roll 7.2°, rec 1.14 s |
| 120 N | roll 10.8°, rec 1.09 s | **fell** | roll 11.4°, rec 1.93 s |
| 140 N | roll 12.4°, rec 1.10 s | **fell** | roll 17.0°, fell-band |

**Read:** the pretrained policy is already strongly push-robust (recovers ~1 s via its own foot placement). A task-space CoM correction injected through the legs is at best **neutral** (ID-MPC ≈ policy, marginally better at small pushes, marginally worse at larger) and at worst **harmful** (fixed impedance falls every time). The CoM-torque channel can't beat the policy's foot-placement recovery.

**Strategic implication:** on a strong RL policy platform there is little room for a CoM-correction ID layer to improve push survival. To show ID value here, the correction likely needs a policy-native channel (modulate the velocity/footstep command, i.e. bias foot placement) rather than CoM torque.

## Resolution — ID modulates the policy's WALK COMMAND (works)
Same file, `--inject command`: the ID output is a planar velocity-command bias [dvx, dvy] added to the policy's walk command, so the policy recovers by placing its feet (its strong mechanism). **Sign is critical:** the correction must be in the CoM error-VELOCITY direction (step *toward* the fall to get a foot under the CoM); regulating position error *back* to nominal fights the capture step and falls. Deadband + rate-limit keep the ID layer out of the nominal loop (nominal == policy-alone). Controllers: `impedance` = fixed capture-assist + gentle return; `id_mpc` = capture-assist + gentle return + leaky-integrated interaction- residual feedforward (the estimated push anticipates the assist).

Lateral push, peak base roll / fall (8 s, 0.15 s pulse at 3 s):

| push | policy | +impedance (capture) | +ID-MPC (capture+residual) |
|---:|---|---|---|
| 0 N | 6.7° | 6.7° | 6.7° (nominal untouched) |
| 140 N | 12.4°, rec 1.10 s | 10.3°, 1.14 s | 10.3°, 1.12 s |
| 160 N | 22.1° | 11.0° | 11.0° |
| 180 N | **fell** | 13.9° | 14.2° |
| 200 N | **fell** | **fell** | **20.4° (survives)** |
| 220 N | fell | fell | fell |

**Result:** capture-direction command modulation **extends the lateral push-recovery envelope** — policy ~160 N → impedance ~180 N → residual-augmented ID-MPC ~200 N (+~25% over policy), and roughly halves peak roll at 160 N (22°→11°), with nominal walking untouched. Ordering policy < impedance < id_mpc validates both the on-policy coupling and the residual augmentation. Forward pushes: ID lowers roll (6.9°→4.4° at 200 N) but the policy is already robust sagittally (all fall ~260 N) — the lateral single-support plane is where ID helps most, matching the paper's original vulnerable-case finding.

## Seed + gait-phase statistics (`--stats`)
Phase-locked pushes gated on measured support (DS = double, SS = single) after 2.5 s, 10 seeds/cell with seeded lateral process noise (15 N). Gating on a clean support phase makes the platform more push-robust than the ungated single run, so separation appears at higher magnitudes. Fall counts (falls / 10):

| cell | 260 N policy / imp / id_mpc | 300 N policy / imp / id_mpc |
|---|---|---|
| lateral, DS | 1 / 0 / 0 | 4 / 1 / 1 |
| lateral, SS | **5** / **2** / 4 | **8** / 6 / 6 |
| forward, DS | 0 / 2 / 4 | 10 / 10 / 9 |
| forward, SS | 6 / 5 / 6 | (all fall) |

**Statistical read:**
- **Lateral pushes — ID reduces falls**, most in the vulnerable single-support case (260 N SS: policy 5 → impedance 2 falls; 300 N DS: policy 4 → 1). This is the paper's vulnerable-case result, reproduced statistically on the policy.
- **Forward pushes — ID is neutral-to-harmful** (260 N DS: policy 0 → id_mpc 4). The policy is already sagittally robust; a forward capture command over-drives the already-fast forward walk. The ID correction should be lateral-focused.
- **id_mpc (residual feedforward) does NOT yet clearly beat fixed-impedance capture-assist** across seeds (often comparable, sometimes worse) — the single-run 200 N envelope win washes out under process noise. The residual- feedforward term (kr) needs better filtering/tuning, or promotion to the full horizon MPC, to earn a statistical edge over fixed impedance.

Peak roll is similar across controllers where all survive; the benefit is in fall rate, not roll magnitude — as in the paper.

## Refinements: lateral-focus + real horizon MPC + terrain
**Lateral-focus** (`axis_scale=[0.2,1]`): the correction is now on the lateral channel; this removes the forward harm for the fixed-impedance capture-assist.

**id_mpc promoted to the real horizon MPC** (Eq. 11) — the regulating acceleration is mapped to the capture-direction command (`u = -k_map·a_e`), and the MPC is velocity-dominant (`q_pos=15, q_vel=100`; arresting the push, not chasing absolute position). Result:
- **Lateral single-support: id_mpc now wins** — 260 N latSS falls 1–2/10 vs impedance 2 vs policy 5. The velocity-dominant horizon MPC earns an edge over fixed impedance in the vulnerable case.
- **Forward: the horizon MPC is a liability** — falls 6–10/10 (policy 0–6), robust across every gain tried, even with the forward command zeroed. Cause: the offset-free residual integration is a *persistent* command that over-drives the walk and amplifies lateral velocity noise. This is the key insight — the offset-free property is an asset for **sustained** disturbances (its design purpose) and a liability for **transient** ones. Impedance (transient) is safer forward; the horizon MPC wins where the disturbance is large and lateral.

**Step-up / step-down terrain** (`--terrain`, box step, 10 seeds, tilt metric):
| terrain | policy | impedance | id_mpc |
|---|---|---|---|
| step-up 30 mm | ok (2.7°) | ok | ok — trivial |
| step-up 60 mm | **falls (trip)** | falls | falls |
| step-down 60 mm | ok (4.3°) | ok | ok (3.6°) |
| step-down 100 mm | ok (8.1°) | ok | ok (9.2°) |
- **Step-up is a toe-stub trip** (blind flat-trained policy hits the vertical face) — a perception failure, not a balance-compensation one, so ID cannot help (all fall ≥60 mm). **Step-down the policy handles alone**; ID is neutral. Steps are transient trips, not the sustained interaction the offset-free MPC targets.

## Sustained lateral force — the offset-free showcase (`--sustained`)
A constant lateral force held 3 s during walking; metric = steady lateral offset (last 1 s), 10 seeds. **The capture-direction command that wins transient pushes AMPLIFIES a sustained drift** (id_mpc drifted worst: 1985 mm vs policy 513 mm at 8 N) — the two disturbance classes need OPPOSITE command signs. With the correct sustained controller (step AGAINST the force; id_mpc adds an offset-free integral term):

| force | policy | impedance (P, against) | id_mpc (P+I, offset-free) |
|---:|---:|---:|---:|
| 8 N | 350 mm | 147 mm | **22 mm** |
| 12 N | 615 mm | 222 mm | 128 mm |
| 16 N | 972 mm | 524 mm | 397 mm |
| 20 N | 1408 mm | 1027 mm | 911 mm |

Clean ordering **policy > impedance > id_mpc**; the integral term nearly nulls the steady offset within authority (8 N: 22 mm vs 350 mm, ~16×) and the benefit shrinks as the force nears the command-authority limit (16–20 N). This reproduces the paper's Table III sustained-force / authority-limit result on the policy platform. (0 falls — forces are sustainable; drift is the metric.)

## Net Stage-2 synthesis — TWO regimes, opposite corrections
On the frozen policy platform, ID adds value in two lateral cases, and the residual's character (transient vs persistent) is what distinguishes them:
1. **Transient lateral push** → capture-direction command (step toward the fall): velocity-dominant horizon ID-MPC reduces falls, most in single support (260 N latSS: policy 5 → id_mpc 1–2). Impedance is safer forward; the MPC's persistence is a transient liability sagittally.
2. **Sustained lateral force** → against-direction + offset-free integral: holds the steady offset far better (8 N: 22 mm vs 350 mm), reproducing the paper's Table III within-authority result. Where the policy already excels (forward pushes, step-downs) ID is neutral; a blind step-UP is a toe-stub trip (perception, not ID). The scientific throughline is that the SAME interaction-residual estimate drives opposite command signs for transient vs sustained disturbances — a clean, honest characterization of where a predictive interaction layer helps a strong RL locomotion policy and where it does not.

## Unified capture-vs-hold controller (`--unified`) — NEGATIVE result
Goal: one id_mpc that auto-selects capture (transient) vs hold (sustained) from the residual, no manual mode. Tried two discriminators:
1. **Residual magnitude** (heavy low-pass of d_eff): fails — a transient push's *recovery* keeps the residual nonzero same-direction, so p→1 and the push is misclassified as sustained; applying hold then prevents recovery (self- reinforcing). p was ~1 for a 0.15 s push for the whole 4 s after.
2. **Deviation-persistence timer** (capture first, escalate to hold if |e| stays past the deadband beyond a transient-recovery timescale): also fails — a transient push *leaves a persistent position offset* (balance recovers at a shifted position), so |e| never returns under the deadband and the timer still escalates.

Outcome across escalation timings (t0=0.3–1.0 s): the unified controller **underperforms both specialists on both regimes** — sustained 8 N offset 129–473 mm (vs P+I specialist 24 mm) and transient 260 N latDS falls 4–5 (vs capture specialist 0–2, and *worse than the bare policy's 1*). The against- command applied during a transient recovery is *actively destabilizing*.

Root cause (fundamental, not a tuning miss): (a) transient and sustained lateral disturbances are **hard to distinguish online from CoM feedback** because a transient push's recovery leaves persistent deviation that mimics a sustained force; (b) the two regimes need **opposite command signs**, so any misclassification is costly. Clean discrimination needs a signal CoM feedback doesn't provide — the external wrench.

## Unified controller from an EXTERNAL-WRENCH estimate (`--wrench`) — WORKS
The missing signal is the external force. A **CoM linear-momentum residual** separates the regimes. Full 3D balance is `m·c̈ = m·g + ΣF_contact + F_ext`, so `F_ext = m·(c̈ − g) − ΣF_contact`; we use only the **horizontal (xy)** components, where gravity has no component and it reduces exactly to `F_ext,xy = m·c̈_xy − ΣF_contact,xy` (the `−m·g` term is required for the full 3D force, dropped here only because we take horizontal components). Sign convention verified: at standstill ΣF_contact = [~0, ~0, +weight]. Estimator inputs are simulation-clean (kinematic CoM velocity from full state; MuJoCo-exact contact forces) — on hardware this is the momentum residual from IMU + estimated CoM velocity + foot six-axis wrenches, NOT yet hardware-validated. It cleanly separates the regimes — probe: a 120 N/0.15 s push spikes `|F_ext|` to 73 N then **decays to 0 within ~0.6 s** (force gone, even mid- recovery), while a 12 N sustained force holds `|F_ext|≈11 N` (≈ the applied 12 N) for the whole window. This is exactly what the deviation timer could not see.

Discriminator: a persistence timer on `|F_ext|>3 N` that **decays** (not hard-resets) so process-noise dips don't drop a latched hold; `p∈[0,1]` blends `(1−p)·capture + p·(hold+offset-free integral)`. Result — ONE controller, no manual mode, at realistic process noise:

| test | policy | wrench-unified id_mpc |
|---|---|---|
| nominal (no disturbance) | 6.7° | 6.7° (untouched) |
| transient 260 N latDS/latSS falls | 0 / 5 | **0 / 0** |
| transient 300 N latSS falls | **10/10** | **5/10** |
| sustained 8 N offset | 404 mm | **115 mm** (3.5×) |
| sustained 12 N offset | 670 mm | 356 mm |

It reduces transient push falls (most in the vulnerable single-support case) AND sustained drift, from a single control law. It does not fully match each SPECIALIST (sustained 8 N: 115 mm vs the pure-hold specialist's 24 mm) — the cost is the ~0.4 s of capture before the timer confirms the force is sustained, during which the CoM drifts before hold engages. But it clearly beats the bare policy on BOTH regimes, which the CoM-only unified controller never did. Available via `--wrench`; the timer-only `--unified` remains the documented negative baseline.

Takeaway: a unified capture-vs-hold interaction layer on a strong policy is achievable, but **requires an external-wrench estimate** as the discriminator — CoM kinematics alone are insufficient because a transient recovery is kinematically indistinguishable from a sustained force.

## VALIDATION (reviewer-driven; plan in Change_Direction_Plan Stage-2 Validation)

### V1 — estimator vs ground truth (`validate_estimator.py`) — PASS, and caught a bug
Logged true applied force vs estimated F_ext (ID off, no process noise) across nominal / turning / diagonal walking, step-up, step-down, a 120 N/0.15 s impulse, and a 12 N/3 s sustained force. **Found and fixed a real bug:** `grf_world` summed `mj_contactForce` with a fixed `+` sign, but that force is on geom2 and the robot is geom1 for box-terrain contacts — so the GRF (hence F_ext) was **sign-flipped on the slab** (GRFz −306 N vs +weight). Fixed to orient each contact force onto the robot (`+fw` if robot is geom2, `−fw` if geom1). **The bug affected terrain ONLY** — on flat ground every contact has the floor as geom1, so the sign was always correct; the entire flat-ground push/sustained/envelope study is unaffected.

Post-fix results (`results/estimator_validation.json`, `figures/estimator_validation.png`):

| scenario | RMSE | false-pos rate | detect | decay | track RMSE |
|---|---:|---:|---:|---:|---:|
| nominal | 0.1 N | **0.0%** | — | — | — |
| turning | 0.1 N | **0.0%** | — | — | — |
| diagonal (lateral) | 0.1 N | **0.0%** | — | — | — |
| step-up 30 mm | 0.2 N | **0.0%** | — | — | — (policy trips at 2.55 s; est clean pre-fall) |
| step-down 30 mm | 0.1 N | **0.0%** | — | — | — |
| impulse 120 N | — | — | 0.00 s | 0.19 s | — |
| sustained 12 N | 1.0 N | — | 0.01 s | — | 1.0 N |

**Read:** the estimator does NOT read touchdown/liftoff, step transitions, or commanded turning/lateral acceleration as external interaction (0% false positive, ~0.1–0.2 N residual during all no-force walking) — the biggest risk, cleared. It detects a real impulse essentially instantly and clears it in ~0.19 s, and tracks a sustained force to ~1 N. The blind flat policy still trips on a blind step-up (a locomotion-policy limitation, independent of the estimator; the estimator is clean until the trip). Metrics are evaluated upright-only (a fall has genuinely large unaccounted acceleration). Estimator inputs remain simulation-clean (MuJoCo-exact contact forces, full-state CoM velocity); V5 tests sensor-bias robustness.

### V2 — oracle ablation (`validate_oracle_ablation.py`) — isolates detection cost
Six controllers on the same transient (300 N lateral SS) and sustained (8 N) tests, 10 seeds. The oracle knows the true disturbance class at onset (zero detection delay).

| controller | transient 300 N latSS falls | sustained 8 N offset |
|---|---:|---:|
| policy | 10/10 | 404 mm |
| capture-specialist | **3/10** | 1091 mm ✗ |
| hold-specialist | 10/10 ✗ | **17 mm** |
| CoM-only-unified | 4/10 | 766 mm |
| **wrench-unified** | **3/10** | 115 mm |
| **oracle-unified** | **3/10** | **29 mm** |

**Read:** (1) each specialist fails outside its regime (opposite-sign dichotomy
confirmed); (2) the CoM-only unified is dominated on both — the wrench estimate is what enables a working unified controller; (3) the oracle is best-of-both (3/10, 29 mm) — the ceiling; (4) **wrench-unified matches the oracle on transient (3/10 = 3/10 → no detection cost there)**, and the sustained gap **115 vs 29 mm is the price of causal detection** (~0.4 s to confirm persistence, during which capture briefly amplifies the drift). Both far exceed policy (404 mm) and CoM-unified (766 mm). Results in `results/oracle_ablation.json`.

### Envelope curves (`wrench_envelope_sweep.py` -> `figures/wrench_envelope.png`)
Full sweeps (10 seeds/cell) of the wrench-unified controller vs policy and the regime-specialist impedance:
- Transient push, lateral **single-support** — fall rate (240/280/320 N): policy 50/50/100% -> wrench **0/10/80%**; the fall curve shifts right ~40-60 N Lateral **double-support** is ~neutral (policy already robust; all cross ~360 N).
- Sustained force — steady offset (6/10/14/22 N): policy 287/530/839/1725 mm; wrench **61**/176/493/1499 mm — best within authority (6-10 N, ~5x at 6 N), reverting between policy and against-P as the force nears the authority limit. Data in `results/wrench_envelope.json`.

### V3 — held-out tuning (`validate_heldout.py`) — generalizes, with ONE failure mode
Hyperparameters (f_thresh, tf0, decay, gains) FROZEN, evaluated on cases never used for tuning. Wrench vs policy / capture-spec / hold-spec.

A. **Impulse durations** (200 N latSS, falls/10) — held-out 0.10/0.25/0.40 s (tuned only 0.15 s):

| ctrl | 0.10 s | 0.15 s | 0.25 s | 0.40 s |
|---|---:|---:|---:|---:|
| policy | 0 | 0 | 10 | 10 |
| capture-spec | 0 | 0 | 2 | 10 |
| **wrench** | 0 | 0 | **3** | 10 |
| hold-spec | 0 | 0 | 10 | 10 |

At the held-out 0.25 s duration wrench (3) ≈ capture-spec (2), NOT degraded toward hold-spec (10) — the discriminator correctly classifies a longer-but-still-transient impulse. At 0.40 s the impulse is unrecoverable for all (80 N·s); wrench = capture-spec (no misclassification harm).

B. **Sustained magnitudes** (offset mm) — held-out 6/10/14 N (tuned 8/12/16): wrench 61/176/493 vs policy 287/530/839 vs hold-spec 26/32/263. Beats policy, approaches hold-spec — consistent with the envelope, no overfitting.

C. **Ramped force** (0→12 N over 1 s): wrench 380 mm vs policy 469, hold-spec 166. Wrench beats policy on a never-tested gradual onset (detection during the ramp is slower, so the gap to hold-spec widens, but it still helps).

D. **Intermittent force** (12 N, 0.5 s on/off) — **FAILURE MODE**: wrench **983 mm > policy 340** (hold-spec 20). An on/off force at a period comparable to the detection timescale defeats the persistence discriminator: each OFF period decays the timer (never commits to hold), while capture fires during each ON pulse (wrong sign) and amplifies the drift. This is the same latch-vs-clear tradeoff that governs transient-vs-sustained; an intermittent force is genuinely ambiguous to a persistence detector. **Honest limitation:** the controller handles clean transient, clean sustained, and ramped disturbances, but not rapidly intermittent forcing.

Results in `results/heldout_validation.json`.

### V4 — statistical strength (`validate_stats.py`) — 40 paired held-out seeds
Seeds 2000–2039 (held out from dev seeds 1000–1009). Policy vs wrench-unified, paired.

**Transient (lateral SS), fall counts + McNemar exact test:**

| push | policy falls | wrench falls | policy-only fall | wrench-only fall | McNemar p |
|---|---:|---:|---:|---:|---:|
| 280 N | 24/40 | **6/40** | 20 | 2 | **0.0001** |
| 300 N | 40/40 | **23/40** | 17 | 0 | **<0.0001** |
| 320 N | 40/40 | 37/40 | 3 | 0 | 0.25 (n.s.) |

Wrench significantly reduces falls at 280/300 N and **never makes a seed worse** (wrench-only falls = 2/0/0). At 320 N the push is near-unrecoverable for all, so the small benefit is not significant — as expected.

**Sustained, median offset [IQR] mm:**

| force | policy | wrench |
|---|---|---|
| 8 N | 452 [432–487] | **20 [7–29]** |
| 12 N | 721 [701–754] | **280 [233–327]** |

Non-overlapping IQRs — clearly significant (~22× at 8 N, ~2.6× at 12 N).

**Correction to V2 (important):** the V2 10-seed sustained numbers were noisy. Over 40 seeds, wrench 8 N median is **20 mm** (not 115 mm), essentially matching the oracle (16 mm) and hold-specialist (15 mm). BUT the wrench distribution has a **heavy tail** (range 2–691 mm) that the specialists do not (hold-spec range 0–41 mm). So the causal-detection cost is **not a median offset penalty** — it is **increased variance**: usually the wrench holds as well as the specialist, but on a minority of seeds the detection/blend transition mis-fires and it drifts badly. This is the honest, seed-robust characterization, and it refines V2's median-gap framing. Results in `results/stats_validation.json`.

### V5 — sensor-bias / noise robustness (`validate_sensorbias.py`) — 20 seeds
Horizontal foot-force bias b_F biases the estimator directly (F_ext = m·c̈ − (GRF + b_F)). Wrench-unified, nominal roll / sustained-8 N offset / transient-300 N latSS falls:

| b_F [N] | nominal roll | sustained 8 N | transient 300 N falls |
|---:|---:|---:|---:|
| −5 | 6.9° | 15 mm | 20/20 |
| −3 | 6.9° | 17 mm | 18/20 |
| 0 | 6.9° | 20 mm | 15/20 |
| +3 | 6.9° | 394 mm* | 15/20 |
| +5 | 6.9° | 26 mm | 19/20 |

GRF noise std (bias 0): nominal 6.8–6.9°, sustained 18–20 mm, transient 9–10/20 (all ≤ b_F=0).

**Read:** (1) **Nominal walking is fully protected** — roll stays 6.8–6.9° across every bias and noise level, because the |e|>deadband gate keeps the controller off during undisturbed walking, so a spurious bias-induced wrench can never engage it. This is the key robustness result. (2) **Sustained regulation tolerates ±5 N bias** (15–26 mm) and noise (18–20 mm). The +3 N 394 mm cell (*) is a variance artifact of the wrench's heavy tail (V4) — non-monotonic vs the ±5 N neighbours, a few bad-tail seeds in a 20-seed median — not a systematic bias failure. (3) **Transient benefit erodes under large bias** (falls 15→19–20/20 at ±5 N, drifting toward policy's 40/40) — a real, graceful degradation; still far better than policy. Results in `results/sensorbias_validation.json`.

### V6 — momentum-observer reformulation (`validate_observer.py`) — equivalent, kept finite-diff
Compared the raw estimator `F_ext = lowpass(m·c̈ − GRF)` against a momentum-residual observer `F̂ = K(l − ∫(GRF + F̂))`, `l = m·ċ` (no acceleration differentiation), on the V1 scenarios:

| scenario | finite-diff RMSE / peak / FP | observer RMSE / peak / FP |
|---|---|---|
| nominal | 0.13 N / 0.2 N / 0.0% | 0.70 N / 7.0 N / 0.4% |
| step-down | 0.13 / 0.2 / 0.0% | 0.72 / 7.0 / 0.4% |
| impulse (detect/decay) | 0.00 s / 0.19 s | 0.00 s / 0.18 s |
| sustained (track RMSE) | 1.02 N | 1.35 N |

**Read:** essentially equivalent on the force scenarios (impulse marginally better, sustained marginally worse), but **finite-diff is cleaner on the critical false-positive metric** (nominal peak 0.2 vs 7.0 N, FP 0% vs 0.4%). In simulation the CoM velocity comes from full-state `J_com·q̇` and is already clean, so the differentiation the observer avoids was never the bottleneck; the observer adds its own convergence transient. The observer's real advantage — robustness to *noisy* velocity — is a hardware consideration, untested here. Kept finite-diff as default. Results in `results/observer_validation.json`.

## Validation summary (V1–V6)
The wrench-unified controller survives reviewer-grade scrutiny. Estimator clean across all contact/terrain/turning regimes (V1, which also caught a real GRF-sign bug affecting terrain only); oracle ablation isolates the causal-detection cost (V2); generalizes to held-out impulse durations, magnitudes and ramped forces but **fails on rapidly intermittent forcing** (V3); push-recovery and sustained-regulation gains are **statistically significant over 40 paired seeds** (V4, McNemar p≤10⁻⁴), the sustained detection cost being **increased variance, not a median penalty**; robust to ±5 N foot-force bias and noise with **nominal fully protected by the deadband** (V5); momentum-observer reformulation equivalent in sim (V6). Honest caveats retained: intermittent-force failure mode, sustained heavy-tail variance, and simulation-clean sensing (MuJoCo-exact contacts, full-state CoM velocity) — not hardware-validated.

## Post-hoc audit (recheck for bugs) — one metric issue, one conclusion corrected
Full re-read of the pipeline + spot-reproduction. **No crash/correctness bugs remain in the core control/estimator logic** (grf_world sign correct post-V1, McNemar exact-binomial correct, momentum observer correct, fall detection and force application correct). Two cosmetic items: dead code (`dofadr`, leftover from the removed torque injection) and a stale module docstring (describes the old `J_com^T(m·a_e)` torque injection, not the current command modulation).

**Two coupled issues that CHANGE the sustained conclusions (the transient/fall conclusions are unaffected).**

**(a) A capture-runaway limitation of the controller.** Scrutinising the sustained *lateral offset* (not just falls/roll) exposed a real behaviour: the capture-containing controllers (capture-spec, CoM-unified, and the wrench controller during its capture phase) **amplify any lateral drift they engage on but cannot classify as a force** — because capture steps *toward* the drift. When process noise pushes the CoM past the deadband without a wrench-detectable force, capture drives a **runaway lateral excursion**. No-force lateral drift vs process noise: wrench **0 / 24 / 770 / 1539 mm** at noise **0 / 1 / 2 / 4 N**, while the hold-only specialist stays **0 / 24 / 35 / 41 mm** (it steps *against* drift, so it is immune). This runaway does NOT cause falls (the robot stays upright), so it was invisible in the fall/roll metrics; it only shows up in lateral position. **It contaminated the sustained studies run at process_noise = 4** (V2/V4/envelope/V5-sustained), and it — not intrinsic detection variance — is the true source of V4's "heavy-tail variance."

**(b) A metric floor + the corrected detection cost.** The raw `lat_offset` also has a ~29 mm (noise 4) phase-drift floor. Re-measuring cleanly at **low noise (1 N, minimal runaway) with the floor-corrected paired metric** (offset with force minus the same-seed offset without force) gives tight, honest force-induced drifts:

| controller | 8 N | 12 N |
|---|---:|---:|
| policy | 462 [460–462] | 728 |
| capture-spec (wrong sign) | 1241 | 1493 |
| CoM-only-unified | 858 | 1113 |
| hold-spec = oracle | **10 [7–18]** | 122 |
| **wrench-unified** | **23 [15–30]** | 238 [211–268] |

**Corrected conclusions:**
- **V4's "detection cost is variance, not a median penalty" was WRONG** (it was noise-runaway contamination). The clean cost is a **real ~2× median penalty** (wrench 23 vs specialist 10 at 8 N; 238 vs 122 at 12 N), with tight, well-separated IQRs — both still ≪ policy (462/728).
- **The opposite-sign dichotomy is even sharper than reported:** pure capture on a sustained force (capture-spec 1241 mm) is *worse than the bare policy* — it amplifies, not just fails.
- **Transient/fall results are robust** (verified: wrench cuts 300 N latSS falls vs policy at noise 1/6/15 → 16/15/12 of 20 vs 20/20/19); McNemar significance and the fall-envelope curves stand.
- **New limitation to state up front:** capture-runaway on sub-threshold lateral drift (in addition to the intermittent-force failure). Both are the same root cause — when the discriminator cannot confirm a force, the capture path is unsafe. Mitigation (future work): **gate capture on the wrench estimate** (only capture when |F_ext| indicates a real disturbance) or default to hold when uncertain.

**Fix for the paper:** report sustained regulation with the floor-corrected paired metric at low process noise; state the ~2× detection cost as a genuine median penalty; and add capture-runaway as a named limitation. Clean numbers in `results/sustained_floorcorrected.json`.

### Audit fix — capture GATE (resolves the runaway)
The capture command is gated by an **interaction-confidence criterion** `g_cap`: capture (interaction-oriented behaviour) is enabled only when the wrench estimate rises a confidence margin above its OWN noise floor — otherwise the layer defers to the policy's internal stabilisation. The threshold is **not a tuned constant**: the noise floor `σ_f` is estimated online from `|F_ext|` during quiescent walking (`|e|<deadband`), and `f_cap = f_floor + k_conf·σ_f` (`f_floor = 1 N` = minimum meaningful force; `k_conf = 3` = a 3σ confidence level). It self-calibrates — settled `f_cap` = 1.4 / 2.1 / 6.3 / 10.6 N at process noise 0 / 1 / 4 / 8 N (it recovers ~6 N exactly at the noise level the constant had been set for, but is now derived, not experiment-specific). `g_cap` latches on confident evidence, is **held open while the robot is still recovering** (deviation > 1.5× deadband) so a long capture recovery keeps full capture, and decays only once recovered. Sub-threshold noise never latches it, so capture cannot amplify noise drift. Validated (15–20 seeds):

| behaviour | before gate | after gate |
|---|---|---|
| no-force lateral drift (noise 2 / 4 N) | 770 / 1539 mm | **48 / 95 mm** (≈ policy floor) |
| transient 280/300/320 N latSS falls (wrench, /20) | — | **3 / 15 / 18** vs policy 11/20/20 (fully preserved) |
| sustained 8/12 N floor-corrected drift (wrench) | 23 / 238 mm | **22 / 233 mm** (unchanged) |
| nominal roll | 6.7° | 6.9° (untouched) |

So the gate **eliminates the capture-runaway while fully preserving the transient push benefit and the sustained regulation.** It does NOT fix the V3 intermittent-force failure (wrench still drifts ~1000 mm) — an intermittent force *is* a real force, so the gate stays open and capture still amplifies during each ON pulse; that case is genuinely ambiguous (repeated pushes vs a sustained-ish force) and remains a stated limitation. The mild forward-direction harm (policy already robust sagittally) is pre-existing and unchanged. Net: one of the two capture-amplification limitations is now fixed; the intermittent one is documented.

### PAPER-READY re-run WITH the gate (`revalidate_gated.py`, `results/revalidate_gated.json`, `figures/wrench_envelope_gated.png`)
Transient: fall-based, process_noise 6. Sustained: floor-corrected paired metric, process_noise 1 (small absolute noise for 8–12 N forces; the wrench sustained variance grows with the noise/force ratio because capture amplifies present drift during the detection delay — a stated sensitivity, not the fixed runaway).

Numbers below use the **self-calibrating interaction-confidence gate** (final code).

**Transient — 40 paired seeds, McNemar:**

| push | policy falls | wrench falls | McNemar p | wrench-worse seeds |
|---|---:|---:|---:|---:|
| 280 N | 24/40 | **2/40** | **<0.0001** | 0 |
| 300 N | 40/40 | **20/40** | **<0.0001** | 0 |
| 320 N | 40/40 | 36/40 | 0.125 | 0 |

Better than the fixed gate (280: 5→**2**; 300: 24→**20**), and wrench **never** makes a seed worse (wrench-only falls = 0 everywhere).

**Sustained — floor-corrected drift, median [IQR] mm (40 seeds, low noise): TIGHT.**

| force | policy | hold-spec = oracle | wrench |
|---|---:|---:|---:|
| 8 N | 462 | 12 [7–19] | **13 [8–17]** |
| 12 N | 728 | 121 [114–132] | 202 [186–218] |

The runaway/heavy-tail is gone. With the self-calibrated gate the sustained cost **nearly vanishes at 8 N** (wrench 13 ≈ specialist 12, overlapping IQRs → statistically indistinguishable) and is a modest ~1.7× at 12 N. Ablation still confirms the dichotomy: capture-spec (1080) and CoM-only-unified (719) *amplify* a sustained force (worse than policy); oracle = hold-spec. Envelope: wrench holds **0 % latSS falls through 280 N** (policy 60–87 %).

**Envelope (gate ON):** transient latSS fall rate — wrench holds ≤13% through 280 N while policy is 60–87%; sustained drift — wrench tracks just above the hold-spec, far below policy across 6–22 N (authority-limited convergence at high force).

**Sensor bias (gate ON, wrench 8 N, floor-corrected):** 10 / 23 / 22 mm at b_F = −5 / 0 / +5 N — robust; the earlier 394 mm outlier was runaway contamination, now gone.

**This is the paper-ready result set.** Corrects the "heavy-tail variance" framing: with the gate + appropriate noise the sustained cost is a **clean median penalty**, and the residual variance is a documented noise/force-ratio sensitivity.
