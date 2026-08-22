# Unitree G1 Simulation Design — Reframed WBC (Interaction-Dynamics Case Study)

*Validates the three new claims of the rewrite: (i) the centroidal balance layer as a configuration-invariant double integrator with offset-free regulation; (ii) the unified interaction MPC with anticipatory cross-port coupling; (iii) sensor-free foot-contact detection from the centroidal disturbance state. Built on the existing G1 stack — reuses `wbc_core` (physical interface), `impedance_mpc`+`kalman` (arm channel), and `level1_centroidal` (GRF allocator), adding a centroidal disturbance estimator and the detector.*

---

## 1. Platform and control stack

- **Model:** `simulation/models/g1_wbc.xml` — official Unitree G1 (29 DOF, 33.3 kg), two feet with 4-corner contacts, one right-arm task site. Physics 2 kHz, control 1 kHz.
- **Actuation:** G1 exposes position actuators; use the **position-as-torque** map already validated in Scenario C (`ctrl[i] ← q_i + τ_i/K_p`). A pure-torque variant (`K_p=K_d=0`, `τ_ff≠0`) is the hardware-parity mode and is run as an ablation.
- **Layer 1 — Body interaction MPC (NEW):** upgrade `level1_centroidal` from PD to the double-integrator form of the derivation. State `x_body=[e_c;ė_c;e_θ;ė_θ]`; input the residual acceleration `u=[u_c;u_θ]`; recover GRF `Σfᵢ = m(c̈_d−g)+m u_c`, `G_τ𝐟 = İ_Gω_G+I_G(ω̇_d+u_θ)` under friction cones; **add a centroidal Kalman disturbance state `d̂_com`** (reuse `kalman.py`, `nd=6` for force+moment) — this is the piece foot-touch detection reads.
- **Layer 2 — Physical interface:** `wbc_core` unchanged — maps commanded GRF + arm force → joint torques under contact-consistency, friction, actuator limits (instantaneous QP, no horizon).
- **Layer 3 — Task interaction MPC:** existing `impedance_mpc`+`kalman` on the right-arm site, `Λ_arm^{(m)}` contact-mode-indexed.
- **Unified option:** stack `[x_body;x_task]` into one QP (constant block-diagonal `𝒜,ℬ`), balance friction/CoP as **hard** constraints, task in the objective, arm reaction fed forward into the body channel (`Γ_bt`).

---

## 2. Scenarios

### S1 — Offset-free centroidal + task regulation (baseline, extends Scenario C)
**Protocol.** G1 in fixed double support. Right arm holds a Cartesian target. At `t=0.5 s` a sustained 8 N push is applied at the end-effector, held 4.5 s.
**Controllers.** D1 OS-PD, D3 fixed-base MPC, D5 proposed no-Kalman, D7 proposed full (+ `d̂_com`).
**Metrics.** Arm SS error (target: D7 ≈ 0.1 mm vs D1 ≈ 10 mm — reproduces the existing result); **CoM excursion** under the arm push (new: shows the body layer holds balance as the arm is loaded).
**Validates.** Offset-free regulation on both ports (claim i).

### S2 — Foot-touch detection from `d̂_com` (**headline, new**)
**Protocol.** Quasi-static cycle: G1 shifts weight onto the right foot (ankle-CoP + hip-roll), **lifts the left foot** ~3 cm, holds ~2 s, **places it back**, repeated 5×. The Layer-1 model is told to *keep both feet in its assumed contact set* (i.e. the mode is NOT hand-fed); the detector must find the switch from `d̂_com` alone.
**Detector.** Project the residual force `m d̂_c'` onto each foot's support normal `nᵢ`; declare **liftoff** when `projᵢ` drops below `−ε` (assumed support force vanished) and **touchdown** when it rises above `+ε`, with `ε` = 3× the steady-state `d̂` std. Trigger the mode update + covariance inflation on each event.
**Ground truth.** MuJoCo per-foot normal contact force (`data.contact`/`efc_force`), thresholded, gives the true touchdown/liftoff times.
**Metrics.** (a) **Detection latency** [ms] from true event to flag; (b) **false-positive rate** over a 20 s undisturbed stance; (c) **missed-event rate**; (d) comparison vs a naïve joint-torque/force-threshold detector. Expected: latency a few ms (the step is `O(mg/2)≈160 N` on a 33 kg robot, far above noise), FP ≈ 0.
**Validates.** Sensor-free contact detection; contact-switching *observed, not asserted* (claim iii, reject #3).
**Risk/fallback.** If G1 single-support is unstable in MuJoCo at this control fidelity, fall back to a **balance-safe variant**: both feet planted, the left hand braces/releases a rail (contact set {LF,RF}↔{LF,RF,LH}); `d̂_com` detects the *hand* touch/release with no balance risk. Same detector, same metrics — reported as the primary if the foot lift proves fragile, with the foot lift as the harder demonstration.

### S3 — Anticipatory cross-port coupling (unified vs split)
**Protocol.** S1's arm push, but compare **split** (arm reaction reaches balance only as `d_body`) vs **unified** (arm wrench fed forward into the body channel, `Γ_bt≠0`).
**Metrics.** Peak CoM excursion and settling time at push onset.
**Validates.** The anticipation advantage of the unified MPC (claim ii); expected lower CoM excursion with `Γ_bt`.

### S4 — Contact-mode-invariant predictor (structural check)
**Protocol.** Across the S2 support transition, log the closed-loop poles / `A_d` and the tracking error before/after the switch.
**Metrics.** Pole spread across modes (target ≈ constant, as `A_d` is fixed); no transient blow-up at the switch beyond the inflation window.
**Validates.** The constant-`A_d` invariance under a genuine contact-mode change (claim, §VI).

---

## 3. Implementation plan (reuse vs new)

| Piece | Status |
|---|---|
| G1 model, `wbc_core` interface, arm `impedance_mpc`+`kalman` | **reuse** as-is |
| `level1_centroidal` → double-integrator + `u`-recovery | **modify** (PD → residual-accel input) |
| Centroidal Kalman `d̂_com` (nd=6) | **new** (instantiate `KalmanDisturbanceEstimator` on the CoM channel) |
| Foot-touch detector (project `d̂_com`, threshold, mode trigger) | **new**, ~1 small module |
| Unified stacked QP + `Γ_bt` feedforward | **new** (optional; S3) |
| Scenario runners S1–S4 | **new**, patterned on `scenario_c_g1.py` / `scenario_transition.py` |
| Ground-truth contact logger + detection metrics | **new**, small |

**Deliverables:** `g1_centroidal_id.py` (Layer-1 upgrade), `contact_detector.py`, `scenario_g1_touch.py` (+S1/S3/S4), and a metrics/plots script writing `latency`, `FP/miss rate`, CoM-excursion and `d̂`-vs-ground-truth figures.

---

## 4. Order of work (recommended)
1. **S2 first** — the headline. Add `d̂_com` to Layer 1, build the detector, run the quasi-static (or hand-brace fallback) transition, and produce the `d̂`-vs-ground-truth figure + latency/FP table. If this lands, the paper's novel claim is backed.
2. **S1** — reuses the existing offset-free result, adds the CoM-excursion metric.
3. **S3/S4** — unified MPC + invariance checks.

---

## 5. Open questions for you
- G1 single-support feasibility at 1 kHz in MuJoCo — try the foot lift, or go straight to the balance-safe hand-brace variant for S2?
- Detector threshold: fixed `3σ` of the `d̂` noise, or a likelihood-ratio test on the Kalman innovation (cleaner, slightly more to implement)?
- Torque mode: position-as-torque (matches prior G1 results) as primary, pure-torque as ablation — agreed?
