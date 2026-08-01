# Paper Redesign — Interaction Dynamics as a Validated Interaction Layer

Written BEFORE touching the manuscript. Three parts: (1) Paper Contract, (2) Section + Figure Plan, (3) Claim Matrix. The rewrite follows only once these are fixed. All evidence references are to `code/unitree_baseline/STAGE2_FINDINGS.md` (V1–V6) and the frozen-platform work.

Guiding principle (unchanged): **Interaction Dynamics is the contribution; humanoid locomotion is only the frozen validation platform.**

---

# DOCUMENT 1 — Paper Contract (the story, no equations)

## Q1. What is the problem?
External interaction on a legged robot spans qualitatively different modes — brief impulses and persistent forces — that require **oppositely directed** control responses (step *toward* a fall to capture momentum vs. lean/step *against* a sustained force to hold position). Center-of-mass kinematics alone **cannot distinguish** these modes online: a transient push and a sustained force can leave the same persistent post-disturbance displacement. Existing interaction/disturbance controllers are therefore effectively disturbance-class-specific, and there is no unified interaction representation that (a) generalizes across interaction modes with one control law and (b) attaches to an existing, unmodified locomotion system.

## Q2. What is the hypothesis?
An interaction layer built on (i) an **external-wrench estimate from centroidal momentum balance** and (ii) a **persistence-gated blend** of predictive *capture* and offset-free *hold* can serve both transient and sustained disturbances with a **single autonomous control law**, added on top of a **frozen** locomotion policy without retuning it. The wrench estimate supplies the interaction-persistence signal that CoM kinematics cannot.

## Q3. What is the contribution?
1. **Unified interaction representation** + identification of the transient/sustained **opposite-sign dichotomy** and why CoM kinematics cannot resolve it.
2. **Validated external-wrench estimator** (CoM linear-momentum residual) — the discriminator that makes the unified layer possible.
3. **Single wrench-gated predictive interaction controller** with a **two-level interaction-confidence arbitration**: (i) an *interaction-confidence gate* — engage interaction-oriented behaviour (capture) only when the wrench estimate exceeds its own noise floor by a confidence margin, else defer to the policy's internal stabilisation (this resolves the capture-runaway failure mode found in validation); (ii) a *transient-vs-sustained blend* — given a confident disturbance, blend predictive capture and offset-free hold by wrench persistence. Realized by **policy command modulation**.
4. **A structured six-stage validation methodology for interaction layers** (estimator correctness → architecture necessity → generalization → statistical significance → sensing robustness → estimator-implementation choice) — reusable beyond this controller.
5. **Humanoid demonstration** on a frozen RL locomotion policy (push recovery, step-up/step-down).

## Q4. What is locomotion?
A **frozen, reproducible floating-base validation platform** (official Unitree G1 pretrained policy, `unitree_baseline/`). It is never tuned during interaction experiments. Nothing more is claimed about it.

## Honesty commitments (must survive into the manuscript)
- **Realization is command modulation, not WBC.** The whole-body-QP realization coupling was found to **destabilize** the policy (its balance is a closed-loop property of its own control law). This is a **reported negative result** that motivates the command-modulation coupling — not a claim of WBC compatibility.
- **Validation altered the implementation** (see "Lessons from Validation"): a real GRF contact-sign bug (V1) and an intermittent-force failure mode (V3) were found by the campaign.
- **Known limits, stated up front:** intermittent forcing defeats the persistence discriminator (V3); sustained regulation has heavy-tail variance (V4); sensing is simulation-clean (MuJoCo-exact contacts, full-state CoM velocity) and not hardware-validated (V5/V6).

---

# DOCUMENT 2 — Section + Figure Plan

## Section flow (redesigned; the old walking-centric order is discarded)

**S1 — Motivation & Interaction Dynamics.** The interaction-mode problem; locomotion as a validation platform; the unified-layer thesis.

**S2 — Interaction Model.** Normalized task dynamics `ẍ = a_e + d_eff` (configuration/contact-invariant fixed model, Theorem 1); the interaction residual; the wrench representation; the transient-vs-sustained dichotomy and its opposite-sign consequence.

**S3 — External-Wrench Estimator.** CoM linear-momentum residual `F_ext = m(c̈ − g) − ΣF_contact` (horizontal form drops gravity); persistence detection. **V1 lives here** (estimator correctness).

**S4 — Unified Interaction Controller.** The two-level interaction-confidence arbitration: (i) confidence gate (engage interaction behaviour only above the wrench noise floor by k·σ; self-calibrating threshold), (ii) transient-vs-sustained blend (predictive capture via the normalized MPC + offset-free hold, weighted by wrench persistence); **command-modulation realization on the frozen policy**; the WBC-realization negative result.

**S5 — Validation of the Interaction Layer (CORE).** The six-stage campaign V1–V6. This is the strongest section, not an appendix.

**S6 — Humanoid Demonstration.** Push recovery, step-up, step-down on the frozen policy. Short — the layer is already validated; this shows it works in-situ.

**S7 — Discussion & Limitations**, including a **"Lessons from Validation"** subsection (the two defects the campaign caught and how they changed the system).

## Figure plan
| # | Figure | Source |
|---|---|---|
| 1 | System architecture: frozen policy + interaction layer + command-modulation channel | new schematic |
| 2 | Unified interaction model + transient/sustained opposite-sign dichotomy | new schematic |
| 3 | Wrench estimator (momentum-residual block diagram) | new schematic |
| 4 | **V1** estimator vs ground truth (nominal/step/impulse/sustained traces) | `figures/estimator_validation.png` |
| 5 | **V2** oracle ablation (6 controllers, transient falls + sustained offset) | `results/oracle_ablation.json` → new bar chart |
| 6 | **V3** held-out generalization + intermittent-force failure | `results/heldout_validation.json` → new |
| 7 | **V4/gated** paired significance + envelope curves (fall rate + floor-corrected sustained drift vs magnitude) | `figures/wrench_envelope_gated.png`, `results/revalidate_gated.json` |
| 8 | **V5** sensor-bias / noise robustness | `results/sensorbias_validation.json` → new |
| 9 | **V6** momentum observer vs finite-diff | `results/observer_validation.json` → new |
| 10 | **S6** push recovery (representative + envelope) | envelope + a time trace |
| 11 | **S6** step-up / step-down | terrain runs |
(Figs 5/6/8/9 may merge into a 2×2 "validation campaign" panel if space is tight; V1 (Fig 4) and V4 (Fig 7) should stay full-size.)

---

# DOCUMENT 3 — Claim Matrix

Every Abstract/Conclusion sentence must map to one row. A claim with no row is weakened or cut.

| # | Claim | Evidence | Strength / caveat |
|---|---|---|---|
| C1 | Transient and sustained interaction need **opposite** command corrections; CoM kinematics cannot separate them | Theory (S2) + V2 (specialists fail out-of-regime) + V3 (CoM-only unified dominated) | Strong, mechanistic |
| C2 | A **CoM momentum-residual wrench estimate** cleanly separates the modes and does not misread contact transitions or commanded motion | **V1** (0% false-positive across nominal/turning/terrain; 1 N sustained tracking; 0.19 s decay) | Strong; sim-clean sensing |
| C3 | The wrench estimate is the **necessary** enabling signal (CoM-only cannot do it) | **V2** (CoM-only-unified dominated on both regimes) | Strong |
| C4 | A **single** wrench-gated controller improves transient survival (strongly) and sustained regulation over the bare policy | Self-cal-gate re-run: transient McNemar p<10⁻⁴ (280 N 24→2, 300 N 40→20 of 40, never worse); floor-corrected sustained wrench 13 [8–17] vs policy 462 mm @8 N | Both strong; sustained cost small at low force |
| C5 | The controller **generalizes** to unseen impulse durations, magnitudes, ramped forces | **V3** (held-out cases; wrench≈capture-spec at 0.25 s) | Strong |
| C6 | It is **robust** to foot-force bias and noise; **nominal balance** is untouched | **V5** (nominal 6.8–6.9° across ±5 N bias) + gated sustained-bias (wrench 8 N drift 10/23/22 mm at −5/0/+5 N) | Strong |
| C7 | The causal-detection cost on sustained forces is **small** — indistinguishable from the specialist at low force, growing modestly with magnitude | Self-cal-gate, floor-corrected, low noise (wrench 13 [8–17] vs specialist 12 [7–19] @8 N → overlapping; 202 vs 121 @12 N ~1.7×, 40 seeds) | Supersedes the "variance" framing (that was the now-fixed runaway); residual variance is a noise/force-ratio sensitivity |
| C8 | A finite-difference estimator suffices **in simulation**; a momentum observer is an equivalent alternative | **V6** | Honest; observer favored only under noisy hardware velocity |
| C9 | The layer works **in-situ on a frozen floating-base locomotion policy** (push, step-up, step-down) | S6 demonstration on the frozen platform | Demonstration, not the core proof |
| C10 | Interaction should be engaged by an **interaction-confidence criterion** (act only when the wrench exceeds its noise floor by k·σ; else defer to internal stabilisation) — a general two-level arbitration that **resolves the capture-runaway** found in validation | Audit + gate validation (no-force drift 1539→95 mm; threshold self-calibrates 1.4/6.3/10.6 N at noise 0/4/8; transient/sustained/nominal preserved) | Architectural contribution + fix; the related intermittent case (C11) remains |
| C11 | **Limitation — intermittent forcing** (and, before the gate, sub-threshold noise): when the discriminator cannot confirm a force, the capture path amplifies | V3 (intermittent) | Stated limitation; genuinely ambiguous |
| — | **NOT claimed:** WBC-QP realization on a strong policy | Coupling study (destabilizes) | Reported negative result |
| — | **NOT claimed:** robustness to rapidly intermittent forcing | V3 (failure mode) | Stated limitation (same root cause as C10) |
| — | **NOT claimed:** hardware performance | — | Sim-clean sensing only |

## Abstract skeleton (each sentence ↔ a claim row)
1. Problem: interaction modes need opposite corrections; kinematics cannot tell them apart. → C1
2. Idea: estimate the external wrench from momentum balance; gate a capture/hold blend by its persistence. → C2, C3
3. One autonomous layer on a frozen policy improves both push survival and sustained regulation. → C4
4. Validated by a six-stage campaign (correctness, necessity, generalization, significance, robustness, implementation). → C2–C8
5. Honest limits: intermittent forcing, heavy-tail variance, sim-clean sensing. → C7, limitations

---

# Recommended process (not starting LaTeX yet)
1. Fix this contract (Doc 1) — one page, no equations.
2. Lock the figure list (Doc 2) so the figures tell the story in order.
3. Fill the claim matrix (Doc 3); delete any manuscript sentence without a row.
4. Only then rewrite, section by section, S1→S7.
