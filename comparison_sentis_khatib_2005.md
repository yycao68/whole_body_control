# Detailed Comparison: Impedance MPC vs. Sentis & Khatib (2005)

**Reference:** L. Sentis and O. Khatib, "Synthesis of whole-body behaviors through hierarchical control of behavioral primitives," *Int. J. Humanoid Robotics*, vol. 2, no. 4, pp. 505–518, 2005.

**Our work:** "Two-Layer Impedance MPC for Physical Human-Robot Interaction: Predictive Disturbance Rejection with Joint-Limit Safety."

---

## 1. Core Philosophy

### Sentis & Khatib (SK05)
SK05 answers the question: **how do you compose many concurrent robot objectives without interference?** Their answer is a strict task hierarchy. Each behavior is a "primitive" (postural equilibrium, contact maintenance, limb motion, balance). The control law synthesizes torques so that higher-priority primitives are achieved exactly, and lower-priority primitives are executed in whatever null-space degrees of freedom remain. The approach is fundamentally **reactive** — every torque command is an instantaneous solution to the current state; there is no lookahead, no disturbance model, no prediction horizon.

### Impedance MPC (Ours)
We answer: **how do you reject sustained external forces while tracking a planned trajectory with hard actuator constraints?** Our answer is a predictive two-layer architecture. The feedforward layer removes known nonlinearities; the MPC layer uses a 100 ms prediction horizon to anticipate the future error trajectory and compute a corrective Cartesian force. An augmented Kalman filter estimates persistent disturbances and guarantees zero steady-state error. The approach is fundamentally **predictive and estimating** — the controller acts on a forecast of the next 100 ms, not just the instantaneous state.

**Key difference in one sentence:**
SK05 solves a *priority-constrained instantaneous torque allocation* problem; Impedance MPC solves a *horizon-constrained disturbance-rejecting trajectory tracking* problem.

---

## 2. Robot Model and Task Space

### SK05

SK05 uses the full rigid-body dynamics with contact constraints:

$$M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) = \tau + J_c^\top\lambda \tag{SK-1}$$

where $\lambda \in \mathbb{R}^{n_c}$ is the vector of contact forces and $J_c$ is the contact Jacobian. For each task $i$, the operational-space inertia is:

$$\Lambda_i(q) = (J_i \bar{M}^{-1} J_i^\top)^{-1} \tag{SK-2}$$

where $\bar{M}^{-1} = M^{-1}(I - J_c^\top(J_c M^{-1}J_c^\top)^{-1}J_c M^{-1})$ is the contact-constrained effective inertia. The operational-space equation of motion for task $i$ is:

$$\Lambda_i \ddot{x}_i + \mu_i + p_i = F_i + F_{h,i} \tag{SK-3}$$

where $\mu_i = \bar{J}_i^\top(C\dot{q}+G) - \Lambda_i \dot{J}_i \dot{q}$ (operational-space bias) and $F_i$ is the task force command.

### Impedance MPC (Ours)

We use the fixed-base dynamics (no contact constraints) for a position-controlled task:

$$M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) = \tau + J_v^\top F_h \tag{OUR-1}$$

The operational-space inertia for the *translational* task only:

$$\Lambda(q) = (J_v M^{-1} J_v^\top)^{-1} \in \mathbb{R}^{3\times3} \tag{OUR-2}$$

**Key distinction:** SK05 requires $\bar{M}^{-1}$ (contact-consistent mass inverse), which changes with every contact configuration. Our $\Lambda(q)$ uses the standard $M^{-1}$ — simpler but restricted to fixed-base systems.

---

## 3. Control Architecture

### SK05 — Hierarchical Behavioral Synthesis

The control torque is assembled as a strict hierarchy of behavioral primitives:

$$\tau = \underbrace{J_1^\top\Lambda_1(\ddot{x}_{d1} + F_1) + \mu_1 + p_1}_{\text{Priority 1 (e.g. balance)}} + \underbrace{\bar{N}_1^\top\bigl[J_2^\top\Lambda_2(\ddot{x}_{d2}+F_2)+\mu_2+p_2\bigr]}_{\text{Priority 2 (e.g. CoM)}} + \cdots \tag{SK-4}$$

where $\bar{N}_k = \prod_{i=1}^{k-1}(I - \bar{J}_i J_i)$ is the accumulated dynamically-consistent null-space projector that removes all DOF committed by higher-priority tasks. Each tier $F_i$ is a PD (or PID) force in task $i$'s operational space. The entire expression is evaluated **once per control cycle** — no iteration, no optimization across a horizon.

**Hierarchy example for a humanoid:**
1. Contact maintenance (feet on ground — must not slip)
2. Whole-body balance (CoM above support polygon)
3. End-effector position tracking (arm task)
4. Postural equilibrium (comfortable joint configuration)

### Impedance MPC (Ours) — Two-Layer Decoupled Architecture

$$\tau = \underbrace{\tau_\text{ff}}_{\text{Layer 1}} + \underbrace{J_v^\top F_\text{mpc}}_{\text{Layer 2 QP}} + J_\omega^\top F_\text{orient} + \bar{N}^\top\tau_\text{null} \tag{OUR-3}$$

Layer 1 cancels all known dynamics analytically:

$$\tau_\text{ff} = C\dot{q} + G + J_v^\top\Lambda\ddot{p}_d \tag{OUR-4}$$

Layer 2 solves a receding-horizon QP over $N=10$ steps to compute $F_\text{mpc}$. The two layers are **rate-decoupled**: Layer 1 runs at 1 kHz (every control step); Layer 2 runs at 100 Hz (every 10 steps) with zero-order hold on $F_\text{mpc}$.

**Single task, predictive:** Unlike SK05's multi-task hierarchy, our QP optimizes a single Cartesian task. Secondary objectives (orientation, joint limits) are handled outside the QP via separate torque channels.

---

## 4. Null-Space Handling

### SK05

The null-space projector is the **dynamically-consistent** projector $\bar{N}_i = I - \bar{J}_i J_i$ where $\bar{J}_i = M^{-1}J_i^\top\Lambda_i$ is the dynamically-consistent pseudoinverse. This ensures that null-space torques add zero wrench at task $i$'s operational point — a property called **task consistency**. The hierarchy guarantees that priority-$k$ torques produce zero disturbance at all higher-priority tasks, not just zero motion.

SK05 uses null-space primarily for:
- Postural equilibrium (last priority)
- Redundancy resolution toward a comfortable configuration
- Does **not** use inverse-barrier potentials for joint limits

### Impedance MPC (Ours)

We use the same dynamically-consistent null-space projector $\bar{N} = I - \bar{J}_v J_v$. The null-space torque contains three components:

$$\tau_\text{null} = \underbrace{-k_\text{null}(q-q_0)}_{\text{centering}} + \underbrace{g(q)}_{\text{barrier}} - \underbrace{d_\text{null}\dot{q}}_{\text{damping}} \tag{OUR-5}$$

The **inverse-barrier** $g(q)$ is new relative to SK05: it fires a repulsive torque growing as $k_b(1/d_i - 1/\delta_i)$ when joint $i$ comes within $\eta = 10\%$ of its range limit. SK05 uses a low-priority postural task that *attracts* toward a comfortable configuration but provides no hard repulsion guarantee.

**Key advantage of our approach:** The barrier provides a $C^\infty$ repulsive potential that diverges as $d_i \to 0$, giving a mathematical guarantee that joints cannot reach their limits under bounded QP forces. SK05's postural task can be overridden by higher-priority tasks, leaving joints unprotected near limits.

---

## 5. Disturbance Handling and Steady-State Error

This is the **largest qualitative difference** between the two frameworks.

### SK05

SK05 uses PD feedback for each task:

$$F_i = \Lambda_i(\ddot{x}_{di} + K_{D,i}\dot{e}_i + K_{P,i}e_i) \tag{SK-5}$$

Under a persistent external wrench $F_h$, the steady-state positional error is:

$$e_{\infty,i} = K_{P,i}^{-1} F_{h,i} \quad \text{(nonzero unless } K_{P,i} \to \infty\text{)} \tag{SK-6}$$

Increasing $K_{P,i}$ reduces steady-state error but also increases the reaction force against the human. There is no disturbance observer, no integral action in the original formulation. Subsequent work (e.g., admittance-augmented SK) adds integral terms, but these are not part of SK05.

### Impedance MPC (Ours)

The Kalman-augmented state $[e;\;\dot{e};\;\hat{d}]$ provides offset-free tracking:

$$\hat{d}(k) \to d_\infty = F_h/\text{(plant gain)} \quad \text{as } k \to \infty \tag{OUR-6}$$

The QP free response $x_\text{free} = \Phi\hat{x}_e + D_\text{bar}\hat{d}$ pre-compensates for the disturbance before it accumulates. **Theorem 2** (in the paper) proves:

$$\lim_{k\to\infty}\|e(k)\| = 0 \quad \text{for any bounded constant disturbance} \tag{OUR-7}$$

Simulation on FR3 confirms: 0.04 mm steady-state error under 10 N step force vs. 12.91 mm for classical impedance (×331 improvement).

**SK05 cannot achieve this:** its task-space PD law has a fundamental steady-state bias under any persistent load. To get zero SS error, one would need to add integral action to each primitive — but this risks windup and stability complications across the task hierarchy.

---

## 6. Constraint Handling

| Constraint type | SK05 | Impedance MPC |
|---|---|---|
| Actuator torque limits | Post-hoc saturation (destroys hierarchy) | Implicit through $F_\text{max}$ on QP force; $\tau \leq \tau_\text{max}$ is a hard QP constraint if added |
| Joint position limits | Low-priority postural task (soft, overridable) | Hard inverse-barrier in null-space + workspace projection |
| Contact forces | Managed through contact-consistent $\bar{M}^{-1}$ | Not handled (fixed-base assumption) |
| Friction cone | Not explicitly constrained | Not applicable |
| End-effector wrench limits | Not handled | $\|F_\text{mpc}\|_\infty \leq F_\text{max}$ hard constraint in QP |

**SK05 advantage:** naturally handles contact and floating-base dynamics.  
**Our advantage:** hard QP constraints on corrective force, formal joint-limit safety guarantee.

---

## 7. Prediction Horizon and Look-Ahead

### SK05

**Zero look-ahead.** The control law (SK-4) is an instantaneous function of the current state $(q, \dot{q})$ and current reference $(\ddot{x}_d, \dot{x}_d, x_d)$. There is no prediction of future states, no anticipation of upcoming trajectory curvature, and no optimization over a time window. The controller reacts to errors after they occur.

**Consequence:** Under a step disturbance, SK05 begins correcting only after the error has built up. The transient peak is governed by the stiffness $K_{P,i}$ — higher stiffness reduces peak but increases human interaction forces.

### Impedance MPC (Ours)

**100 ms look-ahead (N=10 steps at 10 ms).** The QP anticipates the future error trajectory $x_\text{free}$ over the next 100 ms and pre-computes a $F_\text{mpc}$ profile that minimizes the predicted future error. For a circular trajectory, this means the QP starts decelerating before a curvature reversal — something reactive controllers cannot do.

**Measured benefit:** In the figure-8 curvature-reversal test (T4a), Impedance MPC achieves 0.31 mm RMSE vs. 24.41 mm for classical impedance — a feature attributable entirely to predictive look-ahead.

---

## 8. Computational Complexity

### SK05

The computational cost per control step is:

| Operation | Cost |
|---|---|
| Forward kinematics | $O(n)$ |
| Jacobians $J_i$ for each task | $O(n \cdot n_\text{tasks})$ |
| $\Lambda_i = (J_i\bar{M}^{-1}J_i^\top)^{-1}$ per task | $O(m_i^3)$ where $m_i$ is task dimension |
| Null-space projectors $\bar{N}_k$ | $O(n^2)$ per hierarchy level |
| Torque assembly (SK-4) | $O(n \cdot n_\text{tasks})$ |
| **Total** | $O(n^2 \cdot n_\text{tasks})$ — scales with $n^2$ |

For a 30-DOF humanoid with 5 task levels, this is approximately $30^2 \times 5 = 4{,}500$ floating-point operations — extremely fast, well under 0.1 ms. SK05 runs at the full FCI/robot rate (1 kHz) with no batching needed.

### Impedance MPC (Ours)

| Operation | Cost |
|---|---|
| Layer 1 feedforward | $O(n)$ — runs 1 kHz |
| $\Gamma(\rho)$ update (online) | $O(N \cdot n \cdot 9)$ — runs 100 Hz |
| QP solve: $H \in \mathbb{R}^{30\times30}$ | OSQP warm-start $\approx 0.5$ ms — runs 100 Hz |
| Kalman predict + update | $O(81)$ (9-state) — runs 100 Hz |
| **Total per 10 ms** | $<1$ ms for QP; $<0.1$ ms for all other steps |

**Key advantage:** Because $A_d$ is configuration-independent (constant), the free-response matrix $\Phi \in \mathbb{R}^{60\times6}$ is precomputed once at startup and never recomputed. The online cost is dominated by the $30\times30$ OSQP solve, which warm-starts from the previous solution in $< 0.5$ ms.

**Scaling:** QP size is $3N$ regardless of $n$ (robot DOF) — the QP is purely in Cartesian space. For a 30-DOF robot, the QP remains 30 variables. SK05 cost grows as $O(n^2)$; ours grows as $O(n)$ for Layer 1 and $O(1)$ for Layer 2.

---

## 9. Stability and Convergence Guarantees

### SK05

SK05 provides:
- **Task-consistency**: null-space torques produce zero wrench at higher-priority tasks (proven via dynamically-consistent pseudoinverse)
- **Stability within each task**: PD gains can be chosen for stability of each individual task in isolation
- **No global stability proof**: the interaction between task levels in a hierarchy can create instabilities (e.g., competing task forces). SK05 relies on careful gain tuning across levels.
- **No steady-state error guarantee**: under persistent load, each task settles at $K_P^{-1}F_h$
- **Passivity**: not explicitly guaranteed; depends on gain selection

### Impedance MPC (Ours)

We provide two formal theorems:
- **Theorem 1** (Impedance Equivalence): unconstrained infinite-horizon Impedance MPC exactly recovers the classical impedance law
- **Theorem 2** (Zero SS Error): closed-loop system is Input-to-State Stable with $\lim_{k\to\infty}\|e(k)\| = 0$ for any bounded constant disturbance

**Limitation:** Theorems hold for the single-task (3-DOF position) case. Extension to multi-task or floating-base requires additional analysis.

---

## 10. Joint-Limit Safety — Detailed Comparison

SK05 and our work take fundamentally different approaches to joint-limit safety.

### SK05 Postural Task

SK05 places a **postural equilibrium task** at the lowest priority:

$$F_\text{posture} = \Lambda_\text{post}(\ddot{q}_d + K_D(\dot{q}_d - \dot{q}) + K_P(q_d - q)) \tag{SK-7}$$

Projected through all higher null-spaces: $\tau_\text{posture} = \bar{N}_{1,\ldots,k}^\top F_\text{posture}$. This **attracts** the robot toward a comfortable configuration but:
1. Can be fully overridden by higher-priority tasks — if balance and arm motion consume all DOF, the postural task contributes nothing
2. Does not provide hard repulsion — a joint can reach its limit if the task requires it
3. Requires manual tuning of $K_P, K_D$ per joint

### Impedance MPC Dual Barrier

We use an inverse-barrier that grows **unboundedly** as $d_i \to 0$:

$$g_i(q) = k_b\!\left(\frac{1}{d_i} - \frac{1}{\delta_i}\right) \cdot \text{sign}(q_i^\text{free}) \quad d_i < \delta_i \tag{OUR-8}$$

This guarantees that within the barrier zone, the null-space torque exceeds any bounded centering spring, preventing the joint from reaching the limit. Additionally, the workspace projection (eq. 13 in paper) projects the barrier gradient into task space via $(J_vJ_v^\top)^{-1}J_v g$, covering joints that are task-constrained.

**Quantitative result (T5, R=20 cm):**

| Method | Min joint margin | Result |
|---|:---:|:---:|
| Classical impedance | 0.048 | ✗ violates |
| SK05-style postural (low-priority attract) | ~0.048 | ✗ similar (centering, no repulsion) |
| Impedance MPC + barrier | **0.084** | ✓ safe |

---

## 11. Application Domain

### SK05

**Designed for:** whole-body humanoid control with multiple simultaneous tasks and contact constraints. The paper demonstrates:
- Manipulation while maintaining balance
- Multi-contact tasks (foot + hand contacts)
- Prioritized reaction to unexpected disturbances
- Works for floating-base robots (no fixed-base assumption)

**Requires:**
- Full contact Jacobian and contact-consistent dynamics at every step
- Careful task priority ordering by the designer
- The robot must be in a stable contact configuration for $\bar{M}^{-1}$ to be well-defined

### Impedance MPC (Ours)

**Designed for:** precision trajectory tracking under human contact forces on fixed-base manipulators. Demonstrated on:
- 7-DOF fixed-base arm (FR3)
- Disturbance rejection under step, sinusoidal, and payload loads
- Four 3-D circle planes, figure-8, and speed scan trajectories
- Boundary workspace with joint-limit protection

**Requires:**
- Fixed base (or stable contact that can be treated as fixed)
- Single Cartesian task (3-DOF position)
- Known reference trajectory with derivatives $(p_d, \dot{p}_d, \ddot{p}_d)$

---

## 12. Extension to Body Control — Can Both Be Combined?

### Gap Analysis

The user's question is whether Impedance MPC applies to **robot body control** (humanoid whole-body, legged locomotion). The answer is: not directly, but the two frameworks are **complementary** and can be composed.

**What SK05 has that we lack:**
1. Contact-consistent dynamics for floating base
2. Multi-task strict hierarchy
3. Simultaneous balance + manipulation

**What we have that SK05 lacks:**
1. Prediction horizon (100 ms look-ahead)
2. Kalman disturbance augmentation (zero SS error)
3. Formal joint-limit safety (inverse-barrier)
4. Formal stability theorem

### Proposed Combination Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  OUTER LAYER (SK05-style hierarchy, runs at 1 kHz)                  │
│                                                                      │
│  Priority 1: Contact maintenance (feet)  → F_contact                │
│  Priority 2: Whole-body balance (CoM)    → F_balance                │
│  Priority 3: End-effector task           → F_ee  ← ← ← ─────────┐  │
│  Priority 4: Postural equilibrium        → F_post + barrier g(q) │  │
└──────────────────────────────┬──────────────────────────────────-┘  │
                               │  τ_hierarchy (SK04 law)              │
                               ▼                                       │
┌─────────────────────────────────────────────────────────────────────┐│
│  INNER LAYER (Impedance MPC, runs at 100 Hz QP / 1 kHz ff)         ││
│                                                                      ││
│  Layer 1: τ_ff = Cq̇+g + J_v^T Λ p̈_d     (1 kHz)                  ││
│  Layer 2: F_mpc from Kalman + OSQP       (100 Hz)                  ││
│  Output: corrective F_mpc → J_ee^T F_mpc  ──────────────────────────┘│
└──────────────────────────────────────────────────────────────────────┘
```

**Mechanism:**
- SK05's hierarchy allocates DOF across contact, balance, and postural tasks
- The end-effector task slot (Priority 3) receives its force command from the Impedance MPC layer instead of a simple PD law
- The Impedance MPC operates on the **residual plant** seen by the end-effector after contact/balance tasks have consumed their DOF
- The Kalman disturbance estimator now estimates the unmodeled wrench in the end-effector task space — which includes contact reaction forces propagated through the hierarchy

**Key technical challenge:** The "residual plant" seen by the Impedance MPC changes as contact configurations change (balance task consumes different DOF depending on foot contacts). The $A_d$ constant property holds **within** a fixed contact mode but needs to be re-derived at contact transitions.

---

## 13. Summary Comparison Table

| Property | Sentis & Khatib (2005) | Impedance MPC (Ours) |
|---|---|---|
| **Architecture** | Instantaneous hierarchical torque synthesis | Two-layer: feedforward + receding-horizon QP |
| **Robot type** | Floating-base humanoid, multi-contact | Fixed-base manipulator (any n-DOF) |
| **Task model** | Multi-task strict priority hierarchy | Single Cartesian task + secondary null-space |
| **Prediction horizon** | Zero (purely reactive) | 100 ms (N=10 steps) |
| **Disturbance handling** | PD with nonzero SS error | Kalman augmentation, zero SS error (Theorem 2) |
| **Steady-state error** | $K_P^{-1} F_h$ (nonzero) | **Effectively zero** (proved, verified 0.04 mm) |
| **Joint-limit safety** | Low-priority postural attract (soft, overridable) | Inverse-barrier + workspace projection (hard guarantee) |
| **Contact constraints** | Contact-consistent $\bar{M}^{-1}$, friction cones | Not handled (fixed base) |
| **Actuator constraints** | Post-hoc saturation | Hard QP box constraint on $F_\text{mpc}$ |
| **Stability guarantee** | Task-consistency; no global proof | Theorems 1 (impedance equiv.) + 2 (ISS, zero SS) |
| **Computational cost** | $O(n^2 \cdot n_\text{tasks})$, < 0.1 ms | QP $O(1)$ in robot DOF, OSQP < 0.5 ms |
| **Look-ahead** | None | 100 ms horizon |
| **Key strength** | Multi-task whole-body with contacts | Disturbance rejection + formal convergence guarantees |
| **Key limitation** | No disturbance estimation, no prediction | Single task, no contact/floating-base |
| **Application** | Whole-body humanoid, legged robots | pHRI on serial manipulators |

---

## 14. What Each Paper Is Missing That the Other Provides

### What SK05 needs from Impedance MPC:
1. **Offset-free end-effector tracking** under persistent human contact: add Kalman augmentation to the end-effector task slot in the hierarchy
2. **Predictive look-ahead** for planned trajectories with curvature: replace the PD force with a QP force computed over a 100 ms horizon
3. **Hard joint-limit safety**: replace the low-priority postural attract with an inverse-barrier torque in the null-space projector

### What Impedance MPC needs from SK05:
1. **Floating-base and contact dynamics**: use contact-consistent $\bar{M}^{-1}$ and add contact Jacobian $J_c$ to the operational space formulation
2. **Multi-task priority handling**: extend the single end-effector task to a hierarchy (CoM, end-effector, posture)
3. **Balance guarantee**: add balance as a higher-priority primitive above the Impedance MPC task

### Synthesis opportunity
The most impactful extension would be: **replace the end-effector control primitive in SK05's hierarchy with an Impedance MPC layer**. The hierarchy handles balance and contact; the Impedance MPC handles precise end-effector tracking with disturbance rejection. This would combine SK05's whole-body capability with our formal disturbance-rejection guarantees — directly addressing the "robot body control" question.

---

## 15. Relevant Follow-On Work

**Building on SK05:**
- Mistry & Righetti (2011): operational space control with contacts via optimization
- Righetti et al. (2013): hierarchical operational space control for legged robots
- Koolen et al. (2016): momentum-based whole-body controller for Atlas

**Building on Impedance MPC:**
- Haninger et al. (2023): GP-based Impedance MPC for uncertain environments
- Wu et al. (2025): MPC + ADRC for pHRI
- **This work**: Two-layer architecture with formal guarantees and joint-limit safety

**Bridging both:**
- Bellicoso et al. (2016): whole-body MPC for ANYmal (combines centroidal MPC + WBC)
- Winkler et al. (2018): gait and foothold optimization via MPC on legged robots
- Kim et al. (2019): highly dynamic quadruped control via MPC (MIT Cheetah 3)

---

## 16. MIT Cheetah 3 — Centroidal MPC + WBC Split

**Key reference:** D. Kim, J. Di Carlo, B. Katz, G. Bledt, and S. Kim, "Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control and Model Predictive Control," *Proc. IEEE/RSJ IROS*, 2019; and J. Di Carlo, P. M. Wensing, B. Katz, G. Bledt, and S. Kim, "Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control," *Proc. IEEE/RSJ IROS*, 2018.

### 16.1 Architecture Overview

The MIT Cheetah 3 controller is the most direct real-world implementation of the "two-level split" mentioned above. It uses:

```
┌─────────────────────────────────────────────────────────────────────┐
│  OUTER: Centroidal MPC  (runs at ~30–100 Hz)                        │
│                                                                      │
│  State:  [CoM_pos (3), CoM_vel (3), orientation (3), ω (3)] = 12D  │
│  Input:  ground reaction forces F_i ∈ ℝ³ per foot (n_c feet)       │
│  Model:  single rigid-body dynamics (SRBD) — ignores leg inertia    │
│                                                                      │
│  min Σ ||x[k] - x_ref[k]||²_Q + ||F[k]||²_R                       │
│  s.t.  contact model, friction cone, gait schedule                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  desired F_i* (contact forces)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  INNER: Whole-Body Impulse Control (WBC)  (runs at 1 kHz)           │
│                                                                      │
│  Full rigid-body dynamics:  M(q)q̈ + h = Sτ + J_c^T λ              │
│  Resolve F_i* → joint torques τ via QP                              │
│  Includes: joint-limit constraints, foot slip prevention            │
└─────────────────────────────────────────────────────────────────────┘
```

### 16.2 Outer MPC — Centroidal Dynamics Model

The Single Rigid-Body Dynamics (SRBD) approximation:

$$\dot{p} = v, \quad \dot{v} = \frac{1}{m}\sum_i F_i + g, \quad \dot{R} = [\omega]_\times R, \quad I\dot{\omega} = \sum_i r_i \times F_i \tag{C3-1}$$

where $p, v \in \mathbb{R}^3$ are CoM position and velocity, $R$ is body orientation, $\omega$ is angular velocity, $F_i$ are foot contact forces, and $r_i$ are foot positions relative to CoM. SRBD ignores leg mass and inertia — valid when leg mass $\ll$ body mass (Cheetah 3: legs ~20% of total).

**Why this model?** The full rigid-body dynamics of a quadruped have ~18 DOF and nonlinear terms that make MPC intractable at 100 Hz. The SRBD reduces the state to 12D and the dynamics to linear (in a body-frame approximation), enabling a **convex QP** solvable in < 0.5 ms.

The MPC QP (Di Carlo et al. 2018) is:

$$\min_{U} \;\frac{1}{2}U^\top H U + c^\top U \quad \text{s.t.} \quad DU \leq d \tag{C3-2}$$

where $U$ stacks the contact forces $F_i[k]$ over $N$ horizon steps, $H$ and $c$ come from the linearized SRBD, and the constraints encode friction cones ($\|F_{i,xy}\| \leq \mu F_{i,z}$) and contact scheduling (swing feet have $F_i = 0$).

**Comparison with our MPC:** Both use a convex QP with warm-starting. Key differences:

| Aspect | Cheetah 3 outer MPC | Impedance MPC (Ours) |
|---|---|---|
| State dimension | 12D (CoM + rotation) | 6D (EE position + velocity) |
| Decision variable | Contact forces $F_i \in \mathbb{R}^3$ at each foot | Corrective Cartesian force $F_\text{mpc} \in \mathbb{R}^3$ |
| QP size | $3 n_c N$ (e.g. $12 \times 10 = 120$ for 4 feet) | $3N = 30$ |
| Model | Linear SRBD (approximation) | Exact double-integrator (after nonlinear cancellation) |
| Constraints | Friction cone + gait schedule | Box constraint on $F_\text{mpc}$ |
| Disturbance | No Kalman — MPC re-plans each step | Kalman augmented state, zero SS error |
| Update rate | 30–100 Hz | 100 Hz |

### 16.3 Inner WBC — Whole-Body Impulse Control

Given the desired contact forces $F_i^*$ from the outer MPC, the WBC solves:

$$\min_{\tau, \ddot{q}, \delta F} \;\|\ddot{q} - \ddot{q}_\text{ref}\|^2 + w_F\|\delta F\|^2 \tag{C3-3}$$

$$\text{s.t.} \quad M(q)\ddot{q} + h = S\tau + J_c^\top\lambda, \quad \lambda_i = F_i^* + \delta F_i, \quad \tau_\text{min} \leq \tau \leq \tau_\text{max}$$

where $S \in \mathbb{R}^{n\times(n+6)}$ is the selection matrix (zeros for the 6 unactuated floating-base DOF), and $\delta F_i$ is the slack on contact forces. This QP is solved at 1 kHz using an active-set or OSQP solver.

**Comparison with our Layer 1:** Both cancel nonlinear dynamics analytically, but the WBC handles the contact constraint explicitly via $J_c^\top\lambda$ while our Layer 1 uses the simpler fixed-base cancellation. Our $A_d$ constant property does not hold in the WBC because the effective inertia changes with contact mode.

### 16.4 What the Cheetah 3 Split Does and Does Not Have

**Has (vs. our work):**
- Floating-base dynamics (legs leaving and touching ground)
- Contact scheduling in the outer QP (which feet are down)
- Friction cone constraints
- Full rigid-body torque resolution at 1 kHz inner loop

**Does not have (vs. our work):**
- Kalman disturbance augmentation: the outer MPC replans every step but does not estimate a persistent disturbance state. Under sustained external forces (e.g., a human pushing the robot body), the MPC will compensate by adjusting contact forces, but there is no formal zero-SS-error guarantee
- Formal stability theorem: the SRBD approximation introduces model error; stability relies on the replanning frequency being high enough
- Joint-limit inverse-barrier: the WBC includes joint-limit inequality constraints but these are soft QP constraints, not inverse-barriers with guaranteed repulsion
- Impedance equivalence: the outer MPC does not recover a classical impedance law in any limit

**Operational performance (reported in Kim et al. 2019):**
- Running at 3.0 m/s on flat ground
- Robust to 150 N lateral push while trotting
- Stair climbing with unknown step heights

---

## 17. Atlas Whole-Body MPC — Humanoid Extension

**Key references:**
- T. Koolen *et al.*, "Design of a Momentum-Based Control Framework and Application to the Humanoid Robot Atlas," *Int. J. Humanoid Robotics*, vol. 13, no. 1, 2016.
- G. Feng *et al.*, "Optimization Based Full Body Control for the Atlas Robot," *Proc. IEEE-RAS Humanoids*, 2014.

### 17.1 Architecture Overview

Atlas (Boston Dynamics) uses a framework closer to a **hierarchical QP (HQP)** than a strict SK05 hierarchy. Rather than the closed-form null-space projection of SK05, Atlas uses a sequence of QPs where each tier is solved subject to the solution of the previous tier.

```
┌───────────────────────────────────────────────────────────────────┐
│  MOMENTUM CONTROLLER (outer, ~100–500 Hz)                         │
│                                                                    │
│  Tracks desired centroidal momentum: h_d = [L_d; k_d]            │
│  Koolen (2016): min ||Ȧ(q)q̈ - ḣ_d + Ȧ̇q̇||² s.t. contacts       │
│  Output: desired joint accelerations q̈_d and contact forces λ_d  │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │  q̈_d, λ_d
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│  WHOLE-BODY CONTROLLER (inner, 1 kHz)                             │
│                                                                    │
│  Torque-level QP:  M(q)q̈ + h = Sτ + J_c^T λ                     │
│  Prioritized tasks: balance > CoM > end-effector > posture        │
│  Output: joint torques τ                                          │
└───────────────────────────────────────────────────────────────────┘
```

### 17.2 Centroidal Momentum Controller

The centroidal momentum $h = A(q)\dot{q}$ (where $A \in \mathbb{R}^{6\times n}$ is the centroidal momentum matrix) captures the robot's aggregate linear and angular momentum. Koolen et al. (2016) show that tracking $\dot{h}_d$ is sufficient for balance:

$$\dot{h} = \sum_i r_i \times F_i^c + F_i^c = A(q)\ddot{q} + \dot{A}(q,\dot{q})\dot{q} \tag{AT-1}$$

The outer controller computes $\ddot{q}_d$ to track $\dot{h}_d$ while respecting contact constraints. This is equivalent to the Cheetah 3 outer MPC but formulated as a single-step QP (no horizon) rather than a receding-horizon optimization.

**Key distinction from Cheetah 3:** Atlas's outer loop is typically an **instantaneous** QP (no prediction horizon), while Cheetah 3's outer MPC explicitly optimizes over $N$ future steps. This means Atlas is reactive at the balance level, while Cheetah 3 can anticipate foot placements.

### 17.3 Comparison with Our Impedance MPC

| Aspect | Atlas WBC (Koolen 2016) | Cheetah 3 (Kim 2019) | Impedance MPC (Ours) |
|---|---|---|---|
| **Outer loop** | Instantaneous centroidal QP | Centroidal MPC (N steps) | Impedance MPC (N=10, EE task) |
| **Inner loop** | Torque-level HQP | WBC impulse QP | Layer 1 feedforward |
| **Horizon** | Zero (outer), Zero (inner) | 100–200 ms (outer), Zero (inner) | 100 ms (Layer 2) |
| **Disturbance** | Not estimated | Not estimated | Kalman, zero SS error |
| **Task space** | Centroidal momentum (6D) | CoM position/velocity (12D) | EE position (3D) |
| **Joint limits** | Soft QP inequality | Soft QP inequality | Hard inverse-barrier |
| **Contacts** | Full contact model | Full contact model | Fixed base only |
| **SS error** | Nonzero under persistent load | Nonzero under persistent load | **Formally zero** |

---

## 18. Unified Comparison: All Four Frameworks

### 18.1 Architecture Spectrum

The four frameworks occupy distinct positions on two axes:

```
                    ← Reactive ──────────── Predictive →
                    
High    SK05 ──────────────────────────────────────────
n-DOF   (instantaneous                                  
        hierarchy)                                      
        |                                               
        Atlas                                           
        (centroidal                                     
        QP, no horizon)                                 
        |                                               
        |                          Cheetah 3            
Low     |                          (centroidal MPC,     
contact |                          N-step horizon)      
DOF     |                                               
        |                                        Impedance MPC
        |                                        (EE task,
Low     ──────────────────────────────────────── Kalman + barrier)
```

More precisely:

| | Reactive | Predictive |
|---|---|---|
| **Multi-task / whole-body** | SK05, Atlas | Cheetah 3 (outer centroidal MPC), future work |
| **Single task / manipulator** | Classical impedance, PI impedance | **Impedance MPC (this work)** |

### 18.2 Extended Four-Way Summary Table

| Property | SK05 (2005) | Atlas WBC (2016) | Cheetah 3 (2019) | Impedance MPC (Ours) |
|---|---|---|---|---|
| **Architecture** | Hierarchical torque synthesis | Centroidal QP + torque HQP | Centroidal MPC + WBC | Feedforward + receding-horizon QP |
| **Robot type** | Fixed/floating humanoid | Floating humanoid | Floating quadruped | Fixed-base arm (any n-DOF) |
| **Outer loop model** | None (instantaneous) | Centroidal momentum | Single rigid-body (SRBD) | Double-integrator (EE) |
| **Inner loop model** | Full operational-space | Full contact dynamics | Full contact dynamics | Full fixed-base dynamics |
| **Prediction horizon** | Zero | Zero | 100–200 ms | 100 ms |
| **Disturbance estimation** | None | None | None (replanning) | Kalman, zero SS error |
| **SS error under load** | $K_P^{-1}F_h$ | $K_P^{-1}F_h$ (outer) | Nonzero | **Zero (formal proof)** |
| **Contact handling** | Contact-consistent $\bar{M}^{-1}$ | Full contact QP | Friction cone in MPC | Not handled |
| **Joint-limit safety** | Soft postural task | Soft QP inequality | Soft QP inequality | **Hard inverse-barrier** |
| **Actuator constraints** | Post-hoc clipping | QP torque bounds | QP torque bounds | Hard QP force bound |
| **QP size** | None (closed-form) | $O(n+6) \times n_\text{tasks}$ | $3 n_c N$ (typ. 120) | $3N = 30$ (fixed) |
| **Stability guarantee** | Task-consistency | Not proved globally | Not proved globally | Theorems 1+2 (ISS) |
| **Real-time rate** | 1 kHz | 1 kHz (inner) | 1 kHz (inner) / 100 Hz (outer) | 1 kHz (L1) / 100 Hz (L2) |
| **Key strength** | Simple, elegant hierarchy | Balance under manipulation | Fast dynamic locomotion | Disturbance rejection + formal guarantees |
| **Key limitation** | No prediction, no estimation | No horizon, no estimation | No estimation, simplified model | Single task, fixed base |

### 18.3 Where Each Framework Falls Short for Body Control

For the use case of a **mobile manipulator doing pHRI** (robot body moving + arm interacting with humans), all four frameworks have gaps:

| Gap | SK05 | Atlas | Cheetah 3 | Impedance MPC |
|---|---|---|---|---|
| Predict future contact forces | ✗ | ✗ | ✓ (outer MPC) | ✗ |
| Zero SS error under persistent force | ✗ | ✗ | ✗ | ✓ |
| Hard joint-limit guarantee | ✗ | ✗ | ✗ | ✓ |
| Floating base | ✓ | ✓ | ✓ | ✗ |
| Multi-task balance + manipulation | ✓ | ✓ | Partial | ✗ |
| Formal global stability | ✗ | ✗ | ✗ | ✓ (single-task) |

---

## 19. Toward Unified Body + pHRI Control

### 19.1 The Proposed Split and Why It Works

The "MIT Cheetah 3 / Atlas split" that extends our Impedance MPC to body control is:

```
┌──────────────────────────────────────────────────────────────────┐
│  LEVEL 1 — Centroidal / Balance MPC  (10–30 Hz)                  │
│                                                                   │
│  State:   [p_CoM, v_CoM, R, ω]  (12D centroidal)                │
│  Vars:    contact forces F_i per foot                            │
│  Horizon: 500 ms – 1 s (locomotion planning)                     │
│  Output:  desired base motion + contact force schedule           │
└───────────────────────────┬──────────────────────────────────────┘
                            │  desired centroidal trajectory
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  LEVEL 2 — Whole-Body Task Hierarchy  (200–500 Hz)               │
│  (SK05-style or Atlas-style HQP)                                  │
│                                                                   │
│  Priority 1: Contact + friction constraints                       │
│  Priority 2: Follow centroidal reference (from Level 1)          │
│  Priority 3: Arm end-effector ← ← ← ← ← ← ← ← ← ← ← ← ──────┐ │
│  Priority 4: Postural equilibrium + barrier                      │ │
└───────────────────────────┬──────────────────────────────────────┘ │
                            │  joint torques τ                       │
                            ▼                                        │
                       Robot joints                                  │
                                                                     │
┌──────────────────────────────────────────────────────────────────┐ │
│  LEVEL 3 — Impedance MPC (this work)  (1 kHz / 100 Hz QP)       │ │
│                                                                   │ │
│  Layer 1: τ_ff = Cq̇ + g + J_v^T Λ p̈_d      (1 kHz)            │ │
│  Layer 2: Kalman + OSQP → F_mpc             (100 Hz)            │ │
│  Barrier: g(q) + workspace projection       (1 kHz)             │ │
│  Output: corrective force → J_ee^T F_mpc ────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

This three-level architecture directly answers the body-control question:
- **Level 1** plans global locomotion and foot placements over a long horizon
- **Level 2** allocates the n-DOF of the whole body across contact, balance, and arm tasks
- **Level 3** (our Impedance MPC) provides the end-effector of Level 2's arm task with disturbance rejection and joint-limit safety

### 19.2 Key Technical Challenges

**Challenge 1 — Constant $A_d$ across contact transitions.**
Our $A_d$ is constant because the fixed-base effective inertia $\Lambda(q) = (J_vM^{-1}J_v^\top)^{-1}$ depends only on the kinematic configuration. In a floating-base robot, $M^{-1}$ changes with contact mode (which feet are on the ground). Within a fixed contact mode, $A_d$ remains constant and our framework applies unchanged. At contact transitions, $A_d$ must be recomputed — a short (< 1 ms) operation given current $(q, \dot{q})$ and the new contact Jacobian.

**Challenge 2 — Disturbance estimation across task hierarchy.**
The Kalman estimator currently treats $d(t)$ as all unmodeled wrenches on the end-effector. In the whole-body setting, some of this "disturbance" is actually contact reaction forces transmitted through the body. The estimator will correctly learn to reject these, but the convergence time depends on how fast the centroidal dynamics change relative to the Kalman time constant.

**Challenge 3 — Null-space barrier in a reduced null space.**
With Level 2 consuming DOF for balance and contact (priority 1–2), fewer null-space DOF remain for the arm's internal barrier (priority 4). For a 30-DOF humanoid with 6 DOF per constraint (contact + balance), the arm has ~18 DOF for its task + null-space objectives. The barrier still operates correctly in this reduced null space, but the workspace projection formula $(J_vJ_v^\top + \epsilon I)^{-1}J_vg$ must use the **contact-consistent** Jacobian, not the plain $J_v$.

### 19.3 Summary: Inheritance and Extension

| Concept | From SK05 | From Cheetah 3 | From Atlas | From Impedance MPC (Ours) |
|---|---|---|---|---|
| Operational-space inertia $\Lambda$ | ✓ | — | — | ✓ |
| Null-space projector $\bar{N}$ | ✓ | — | ✓ (HQP) | ✓ |
| Contact-consistent $\bar{M}^{-1}$ | ✓ | ✓ (SRBD) | ✓ | — (fixed base) |
| Centroidal MPC outer loop | — | ✓ | — | — |
| Receding-horizon QP | — | ✓ (outer) | — | ✓ (EE task) |
| Kalman disturbance augmentation | — | — | — | ✓ |
| Zero SS error proof | — | — | — | ✓ |
| Inverse-barrier joint safety | — | — | — | ✓ |
| Formal stability theorem | — | — | — | ✓ (fixed base) |
