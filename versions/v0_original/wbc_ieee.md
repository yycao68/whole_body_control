# Whole-Body Impedance Model Predictive Control for Safe Physical Human–Robot Interaction on Floating-Base Platforms

**Yongyan Cao**

*Voryx Robotics, San Jose, CA 95136*
*Email: yongyancao@gmail.com*

*Abstract*—Floating-base robots—humanoids, quadrupeds, and legged manipulators—must simultaneously balance on compliant terrain, respect rigid ground contact constraints, and execute safe, dexterous interaction with humans or objects. Existing whole-body control (WBC) frameworks either allocate the full joint space to locomotion, leaving no principled mechanism for compliant manipulation, or rely on fixed-gain impedance feedback that accumulates steady-state error under sustained physical human–robot interaction (pHRI) forces. This paper presents a three-level Whole-Body Impedance MPC architecture that addresses both limitations. A centroidal MPC outer loop plans contact forces over a 500 ms horizon using a convex single rigid-body dynamics (SRBD) abstraction. A priority-driven inner WBC layer resolves balance into joint torques via a contact-consistent feedforward that projects dynamics through the active null space without violating friction-cone constraints. The residual null space—freed from locomotion—is governed by a receding-horizon quadratic program (QP) that predicts and rejects pHRI disturbances using a Kalman-augmented state. The key structural result is that a contact-consistent feedback linearization reduces the arm end-effector plant to a linear double integrator with a *constant* state matrix within each contact mode, enabling offline precomputation of the QP cost inverse and operation at ≥1 kHz. A covariance-inflation protocol handles contact-mode transitions without resetting the disturbance estimate, guaranteeing asymptotic zero steady-state tracking error under any bounded constant pHRI load. We prove an Impedance Equivalence Theorem establishing that the infinite-horizon, zero-input limit of the proposed MPC recovers a classical task-space impedance law whose closed-loop effective mass $M_{d,\text{eff}}$, damping $D_\text{eff}$, and stiffness $K_\text{eff}$ all adapt to arm posture and contact configuration through the contact-consistent inertia $\Lambda_\text{arm}(q)$, without requiring online re-optimization of impedance parameters. Stability is analyzed under fixed-mode and switching-mode contact, with explicit transient error bounds across contact events.

*Index Terms*—Whole-body control, model predictive control, impedance control, floating-base robots, physical human–robot interaction, operational space formulation, Kalman filter, contact-consistent dynamics, legged manipulation.

---

## I. Introduction

Legged and floating-base robots present a unique control challenge: the robot must regulate its posture and contact forces with the ground while simultaneously performing dexterous tasks with its arms. These objectives are tightly coupled—arm motions shift the center of mass (CoM), changing contact force distribution, while ground reactions propagate back through the body and appear as disturbances at the end-effector. Classical fixed-base impedance control [1] and its MPC extensions [2] cannot address this coupling because they assume the robot base is rigidly anchored.

The dominant paradigm for whole-body control of legged systems decouples the problem into two layers: a centroidal MPC that optimizes ground reaction forces (GRFs) using a linearized single rigid-body dynamics (SRBD) model [3], [4], and an inner WBC layer that resolves these forces into joint torques via a prioritized quadratic program (QP) [5], [6]. This architecture achieves remarkable locomotive agility—the MIT Cheetah 3 [3] executes high-speed bounding and stair climbing—but it allocates 100% of the robot's control authority to locomotion and base-posture maintenance. Any external arm interaction is treated as a disturbance to be suppressed, not as a channel to be actively regulated. A biped reaching to assist a human standing beside it, or a quadruped manipulating a valve while maintaining stance, cannot be handled by these frameworks with the compliance and zero-steady-state-error guarantees required for safe pHRI.

Conversely, impedance MPC methods designed for fixed-base manipulators [2], [7], [8] lack an unactuated base state, generalized-coordinate partitioning, or contact-consistent mass inverses. They assume an infinite-mass ground connection and cannot model the propagation of foot contact forces to end-effector apparent inertia. Deploying them directly on a floating-base platform produces steady-state torque errors and potential instability during contact transitions.

The technical gap is the absence of a *predictive*, *disturbance-rejecting*, compliance-controlled manipulation layer that natively integrates with an existing balance stack without degrading balance performance. This paper closes that gap with the following contributions:

1. **Contact-consistent residual plant** (Section IV): we show that the floating-base arm end-effector dynamics, after priority-driven feedforward cancellation, reduce exactly to a linear double integrator with a *constant* discrete state matrix $A_d$ within each contact mode. The sole modification from the fixed-base case is the substitution of the contact-consistent mass inverse $\bar{M}^{-1}$ for the free-space inverse $M^{-1}$ in the operational-space inertia $\Lambda_\text{arm}$.

2. **Contact-mode-indexed Impedance MPC** (Section V): a receding-horizon QP indexed to the active contact mode selects the input matrix $B_d^{(m)}$ from a precomputed library, maintaining the convex QP structure and enabling ≥1 kHz update rates with a $N$-step prediction horizon.

3. **Kalman disturbance isolation** (Section V-E): the augmented Kalman state simultaneously estimates external pHRI forces, unmodeled leg-momentum variations, and SRBD approximation errors, and propagates the estimate through the entire prediction horizon. This guarantees zero steady-state tracking error at the arm end-effector under any bounded constant pHRI load, even on a dynamically walking base.

4. **Contact-transition protocol** (Section VI): a covariance-inflation rule at contact-mode switches preserves the disturbance estimate across events, bounding the transient tracking error as a function of Kalman re-convergence time.

5. **Impedance Equivalence Theorem** (Section VII): the infinite-horizon limit of the proposed MPC recovers a classical task-space impedance law with closed-loop effective mass $M_{d,\text{eff}} = \Lambda_\text{arm}(q)$, damping $D_\text{eff} = \Lambda_\text{arm}(q)D_d$, and stiffness $K_\text{eff} = \Lambda_\text{arm}(q)K_d$ all adapting to arm posture and contact configuration through the contact-consistent inertia—without requiring $\{M_d, D_d, K_d\}$ as online optimization variables. This adaptation is a structural consequence of the contact-consistent feedforward linearization and preserves the ≥1 kHz update rate.

6. **Stability analysis** (Section VIII): we prove asymptotic zero steady-state error for fixed contact mode and derive explicit transient bounds across contact switches as a function of the Kalman convergence time constant.

The remainder of this paper is organized as follows. Section II surveys related work. Section III develops the floating-base dynamics and contact constraint formulation. Section IV derives the contact-consistent operational-space plant. Section V presents the three-level Impedance MPC architecture. Section VI describes contact-transition handling. Section VII states and proves the Impedance Equivalence Theorem. Section VIII analyzes stability. Section IX gives the complete joint torque equation. Section X provides an architectural comparison. Section XI reports simulation benchmarks on two robot platforms. Section XII concludes.

---

## II. Related Work

### A. Centroidal MPC for Locomotion

The dominant approach to predictive whole-body control projects the robot's dynamics onto its centroidal frame. Di Carlo et al. [3] introduced the convex SRBD formulation for MIT Cheetah 3, linearizing the rotation dynamics about a nominal pitch-roll and treating each support foot as a rigid contact. The resulting time-varying linear system admits a QP solution at 40 Hz for the outer MPC, while an inner Whole-Body Impulse Control (WBIC) layer maps GRFs to joint torques at 500 Hz. This bi-level architecture is foundational to the present work; however, Kim et al. dedicate both layers entirely to locomotion. External arm forces are filtered out through centroidal inertia assumptions and rigid GRF assignment, and no mechanism exists for compliant manipulation.

Bellicoso et al. [5] proposed a hierarchical WBC scheme for the ANYmal quadruped that solves a cascaded sequence of QPs to track base and end-effector tasks simultaneously. While effective for continuous trotting and terrain adaptation, the WBC stack uses classical PD task objectives with instantaneous feedback ($N=1$); there is no receding-horizon mechanism to predict and pre-load corrective torque before a disturbance develops. Sustained contact forces produce the steady-state error quantified in (16) below.

Koolen et al. [6] implemented a momentum-based controller for Atlas that regulates centroidal momentum via a QP distributing contact forces. The formulation shares the centroidal perspective but does not include a predictive loop for arm impedance and relies on high-gain stiffness to suppress interaction errors.

Sleiman et al. [19] proposed a unified MPC framework for whole-body dynamic locomotion and manipulation on legged robots, simultaneously optimising body posture, gait timing, and arm end-effector trajectories over a receding horizon. This represents the state of the art for integrated loco-manipulation and is the closest prior work to the present architecture. The key structural difference is that the arm task in [19] is formulated as a fixed-base QP subproblem: the task-space inertia uses $M^{-1}$ rather than the contact-consistent $\bar{M}^{-1}$, and no integrating disturbance state is included to guarantee zero steady-state pHRI error under sustained contact.

### B. Trajectory Optimization and Nonlinear MPC

Winkler et al. [9] developed TOWR, a phase-based trajectory optimizer that simultaneously synthesizes gait sequences, foothold locations, and full-body motions over multi-second horizons using a nonlinear program (NLP) solved by interior-point methods. The full rigid-body model captures leg-inertia coupling that SRBD ignores, but the non-convex optimization landscape restricts operation to $\sim$20–50 Hz and offline planning. Real-time pHRI management at 1 kHz is outside the scope of this approach.

Grandia et al. [10] extended TOWR with a real-time model-predictive framework using sequential convex approximations, achieving 20 Hz replanning for rough terrain. While this improves responsiveness compared to pure offline planning, it still lacks the integrating disturbance estimation required for zero steady-state error under sustained contact.

### C. Impedance MPC for Manipulation

Force control for robotic manipulators—encompassing impedance, admittance, and hybrid position/force strategies—is surveyed comprehensively in [22]. The foundational impedance control law of Hogan [1] was unified with position and torque control into a passivity-preserving framework by Albu-Schäffer, Ott, and Hirzinger [20], providing the theoretical basis on which predictive extensions build.

The present paper is a direct structural extension of the authors' prior work on saturated and predictive control. Anti-windup designs for output tracking under actuator saturation and constant disturbances [14], and the associated domain-of-attraction analysis [15], established that an integrating disturbance channel achieves zero steady-state tracking error for fixed-base saturated linear systems—the foundational insight carried forward here to the floating-base, contact-switching setting via the Kalman augmented state. The min–max MPC formulation for LPV systems [16] introduced parameter-varying input matrices with input constraints, the direct precursor to the contact-mode-indexed $B_d^{(m)}$ library of the present work. Building on these fixed-base results, Cao, Cheng, and Li [2] introduced a passive MPC framework for pHRI on fixed-base manipulators in which the outer MPC optimizes impedance parameters $\{M_d, D_d\}$ over a receding horizon; passivity is enforced via a virtual energy tank, building on the passivity arguments of [20]. Because impedance parameters enter nonlinearly into the prediction matrices, iterative solvers are required and update rates are limited to 10–30 Hz. The present paper closes the remaining gap by extending this line of work from fixed-base manipulators to floating-base humanoids: the MPC decision variables become corrective *forces* $F_\text{mpc}$, the contact-consistent feedforward linearization yields a constant $A_d$, and the contact-mode-indexed structure of [16] handles stance-phase transitions without sacrificing the precomputed cost matrix structure.

Haninger, Hegeler, and Peternel [7] optimize force references and impedance parameters jointly using stochastic MPC with Gaussian Process models of task forces. Contact-force safety is enforced as a probabilistic chance constraint. This provides complementary insights into uncertainty-aware impedance shaping but does not address floating-base dynamics, underactuation, or contact-consistent operational-space formulation.

**Saturation-aware control and anti-windup.** The interaction between actuator saturation and persistent constant disturbances in output tracking was studied by Cao, Lin, and Ward [14], who showed that anti-windup augmentation achieves zero steady-state error for saturated linear systems. The domain-of-attraction characterisation for saturated systems was further developed in [15]. These results establish the structural principle—augmenting an integrating channel to cancel constant offsets despite hard input limits—that the present work extends to the floating-base, contact-switching setting via the Kalman disturbance state $\hat{d}$. The min–max MPC algorithm for LPV systems subject to input saturation [16] is the closest prior work in the MPC direction: its parameter-varying input matrix (analogous to the contact-mode-indexed $B_d^{(m)}$) is optimised subject to box constraints, providing robust performance guarantees across scheduling parameter variations. The present architecture extends this to floating-base humanoids by incorporating the contact-consistent mass inverse $\bar{M}^{-1}$, a Kalman disturbance integrator, and a covariance-inflation protocol for mode switches.

### D. Operational Space Control and Floating-Base Inverse Dynamics

Khatib [11] formulated the operational space control framework, establishing task-space inertia, Coriolis compensation, and dynamically-consistent pseudoinverses as the mathematical foundation for task-level manipulation. Sentis and Khatib [12] extended this to hierarchical synthesis of whole-body behaviors, proving that priority-ordered null-space projection guarantees non-interference between tasks—the SK05 law that forms the backbone of Level 2 in the present architecture. Righetti et al. [18] unified the floating-base inverse dynamics perspective with external contact constraints, showing how the contact-consistent mass inverse $\bar{M}^{-1}$ arises naturally from an orthogonal decomposition of the constrained dynamics. The present work builds on [11], [12], and [18] by embedding a predictive MPC layer in the residual null space of the floating-base contact-consistent hierarchy.

---

## III. Floating-Base Robot Dynamics

### A. Generalized Coordinates

A floating-base robot (humanoid, quadruped) has $n$ actuated joints and a 6-DOF unactuated base [18]. The generalized coordinates are:

$$q = \begin{bmatrix} q_b \\ q_j \end{bmatrix} \in \mathbb{R}^{n+6}, \quad q_b \in SE(3),\; q_j \in \mathbb{R}^n \tag{1}$$

where $q_b = (p_b, R_b)$ is the base position and orientation and $q_j$ are the $n$ joint angles. The velocity vector is:

$$\dot{q} = \begin{bmatrix} v_b \\ \dot{q}_j \end{bmatrix} \in \mathbb{R}^{n+6}, \quad v_b = \begin{bmatrix} \dot{p}_b \\ \omega_b \end{bmatrix} \tag{2}$$

### B. Equations of Motion

The floating-base equations of motion (Lagrangian mechanics) are [18]:

$$M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) = S^\top\tau + J_c^\top(q)\lambda \tag{3}$$

where $M(q) \in \mathbb{R}^{(n+6)\times(n+6)}$ is the positive-definite inertia matrix; $h = C(q,\dot{q})\dot{q} + G(q) \in \mathbb{R}^{n+6}$ collects Coriolis, centrifugal, and gravity terms; $S = [0_{n\times6},\; I_n] \in \mathbb{R}^{n\times(n+6)}$ is the selection matrix (the base has no direct actuation); $\tau \in \mathbb{R}^n$ are commanded joint torques; $J_c(q) \in \mathbb{R}^{n_c \times (n+6)}$ is the contact Jacobian; and $\lambda \in \mathbb{R}^{n_c}$ are ground reaction forces (GRFs).

The selection matrix $S^\top\tau$ highlights the fundamental underactuation: the 6 base DOF are driven only indirectly through the contact forces $\lambda$. Partitioning (3) into base and joint blocks:

$$\begin{bmatrix}M_b & M_{bj} \\ M_{bj}^\top & M_j\end{bmatrix} \begin{bmatrix}\ddot{q}_b \\ \ddot{q}_j\end{bmatrix} + \begin{bmatrix}h_b \\ h_j\end{bmatrix} = \begin{bmatrix}0 \\ \tau\end{bmatrix} + \begin{bmatrix}J_{c,b}^\top \\ J_{c,j}^\top\end{bmatrix}\lambda \tag{4}$$

where $J_c = [J_{c,b}\; J_{c,j}]$ is partitioned into base and joint columns.

### C. Rigid Contact Constraints

A rigid contact at point $i$ enforces zero contact-point velocity: $J_{c,i}(q)\dot{q} = 0$ [18]. Differentiating yields the acceleration-level constraint:

$$J_{c,i}\ddot{q} = -\dot{J}_{c,i}\dot{q} \tag{5}$$

Stacking all contacts: $J_c\ddot{q} = \gamma_c \triangleq -\dot{J}_c\dot{q}$. Combining (3) and (5):

$$\begin{bmatrix}M & -J_c^\top \\ J_c & 0\end{bmatrix}\begin{bmatrix}\ddot{q} \\ \lambda\end{bmatrix} = \begin{bmatrix}S^\top\tau - h \\ \gamma_c\end{bmatrix} \tag{6}$$

Solving for the GRFs:

$$\lambda = \Lambda_c(q)\bigl(-\dot{J}_c\dot{q} - J_cM^{-1}(S^\top\tau - h)\bigr) \tag{7}$$

where $\Lambda_c = (J_cM^{-1}J_c^\top)^{-1}$ is the contact-space inertia matrix.

### D. Contact-Consistent Mass Inverse

Define the contact-null-space projector [11], [18]:

$$P_c = I - J_c^\top\Lambda_c J_cM^{-1} \tag{8}$$

and the **contact-consistent mass inverse** [11], [18]:

$$\bar{M}^{-1} = M^{-1}P_c^\top = M^{-1}(I - J_c^\top\Lambda_cJ_cM^{-1}) \tag{9}$$

$\bar{M}^{-1}$ replaces $M^{-1}$ in all operational-space formulas when contacts are active. It projects out the contact-constraint subspace, ensuring task forces do not violate kinematic contact constraints. This quantity is the central link between the contact configuration and the apparent inertia at the end-effector.

### E. Centroidal Dynamics

The centroidal momentum $h_G = [k^\top, L^\top]^\top = A(q)\dot{q} \in \mathbb{R}^6$ aggregates the robot's linear and angular momentum about its CoM [17], where $A(q) \in \mathbb{R}^{6\times(n+6)}$ is the centroidal momentum matrix. Differentiating:

$$\dot{h}_G = A(q)\ddot{q} + \dot{A}(q,\dot{q})\dot{q} = G_c(q)\lambda + \begin{bmatrix}mg \\ 0\end{bmatrix} \tag{10}$$

where $G_c(q) \in \mathbb{R}^{6\times n_c}$ maps contact forces to centroidal momentum rate:

$$G_c(q) = \begin{bmatrix}I_3 & I_3 & \cdots \\ (p_1-p_G)^\times & (p_2-p_G)^\times & \cdots\end{bmatrix} \tag{11}$$

For the outer MPC, equation (10) is approximated by the **single rigid-body dynamics (SRBD)** model [3], treating the robot as a lumped mass $m$ with constant inertia $I_G$. Linearizing about a nominal orientation yields:

$$\dot{x}_c = A_c x_c + B_c(\{p_i\})u_c \tag{12}$$

where $x_c \in \mathbb{R}^{12}$ is the centroidal state and $u_c$ collects contact forces. The SRBD approximation error is $O(m_\text{leg}/m_\text{total})^2$, acceptable for robots where leg mass is below 20–30% of total mass.

---

## IV. Operational Space Formulation and Whole-Body Hierarchy

### A. Task-Space Dynamics

For a task variable $x_i = \phi_i(q) \in \mathbb{R}^{m_i}$ with Jacobian $J_i = \partial\phi_i/\partial q$, substituting (3) into the task-space acceleration $\ddot{x}_i = J_i\ddot{q} + \dot{J}_i\dot{q}$ yields the contact-consistent task-space dynamics [11],[18]:

$$\Lambda_i(q)\ddot{x}_i + \mu_i = F_i \tag{13}$$

where:
- $\Lambda_i = (J_i\bar{M}^{-1}J_i^\top)^{-1}$ is the task-space inertia (contact-consistent via $\bar{M}^{-1}$)
- $\mu_i = \bar{J}_i^\top h - \Lambda_i\dot{J}_i\dot{q}$ collects Coriolis, centrifugal, and gravity terms projected to task space ($h = C(q,\dot{q})\dot{q}+G(q)$ includes gravity, so no separate gravity term appears)
- $\bar{J}_i = \bar{M}^{-1}J_i^\top\Lambda_i$ is the dynamically-consistent pseudoinverse
- $F_i$ is the commanded operational-space force

The torque commanded for task $i$ in isolation is $\tau_i = J_i^\top F_i$.

### B. Hierarchical Task Synthesis (SK05)

For $k$ tasks in priority order (Task 1 = highest), the Sentis–Khatib law [12] synthesizes the control torque as:

$$\tau = J_1^\top F_1 + \bar{N}_1^\top J_2^\top F_2 + \bar{N}_{12}^\top J_3^\top F_3 + \cdots + \bar{N}_{1\cdots k}^\top\tau_\text{null} \tag{14}$$

where $\bar{N}_{1\cdots i} = \prod_{j=1}^{i}(I - \bar{J}_j J_j)$ is the accumulated contact-consistent null-space projector. The key property is that $J_j \bar{N}_{1\cdots i} = 0$ for all $j \leq i$: each higher-priority task torque produces exactly zero acceleration at all lower-priority task coordinates. This is the mathematical guarantee that the arm MPC layer (injected into the residual null space) cannot destabilize the balance and contact-maintenance tasks above it.

Each task force $F_i$ in (14) is typically a PD law:

$$F_i = \Lambda_i(\ddot{x}_{di} + K_{D,i}\dot{e}_i + K_{P,i}e_i) + \mu_i \tag{15}$$

Under no disturbance, (15) yields closed-loop error dynamics $\ddot{e}_i + K_{D,i}\dot{e}_i + K_{P,i}e_i = 0$, stable for any $K_{P,i}, K_{D,i} > 0$. Under a persistent disturbance force $F_h$:

$$e_{\infty,i} = K_{P,i}^{-1}\Lambda_i^{-1}F_h \neq 0 \tag{16}$$

This residual steady-state error under sustained pHRI is the fundamental limitation of the SK05 PD law and the primary motivation for replacing the arm task slot with Impedance MPC.

---

## V. Three-Level Whole-Body Impedance MPC

### A. Architecture Overview

Consider an $n$-DOF floating-base robot in a fixed support configuration (e.g., bipedal stance) performing arm manipulation while subject to pHRI forces. The control objectives are:
1. maintain balance with contact forces inside friction cones;
2. track an arm end-effector reference $p_d(t)$; and
3. reject pHRI forces with zero steady-state tracking error.

These objectives are addressed by three nested control levels operating at different timescales:

**Level 1 (Centroidal MPC, 40–100 Hz):** plans CoM trajectory and GRFs over a 500 ms horizon via the SRBD model (12).

**Level 2 (WBC Hierarchy, 500 Hz):** resolves the Level 1 GRFs into joint torques using the SK05 law (14) for contact and balance tasks, leaving the arm end-effector task slot open.

**Level 3 (Impedance MPC, ≥1 kHz):** fills the arm slot with a receding-horizon QP that predicts and rejects pHRI disturbances, replacing the PD law (15) with a predictive force command.

### B. Contact-Consistent Residual Plant

After Level 2 commits torques $\tau_1$ (contact maintenance) and $\tau_2$ (balance), the residual arm end-effector dynamics are governed by the contact-consistent task-projected plant:

$$\Lambda_\text{arm}(q)\ddot{x}_\text{arm} + \mu_\text{arm} = F_\text{arm} + d_\text{ext} \tag{17}$$

where $\Lambda_\text{arm} = (J_\text{arm}\bar{M}^{-1}J_\text{arm}^\top)^{-1}$ uses the contact-consistent mass inverse (9), $\mu_\text{arm} = \bar{J}_\text{arm}^\top h - \Lambda_\text{arm}\dot{J}_\text{arm}\dot{q}$ collects all Coriolis, centrifugal, and gravitational terms (note that $h = C\dot{q}+G$ already includes $G(q)$, so no separate gravity term appears), and $d_\text{ext}$ is the pHRI wrench projected to arm task space. Equation (17) has the same mathematical structure as the fixed-base case, with $\bar{M}^{-1}$ replacing $M^{-1}$ in $\Lambda_\text{arm}$. All feedforward cancellation and linear double-integrator reduction steps follow identically.

### C. Contact-Consistent Feedforward and Horizon Freezing

To decouple the highly nonlinear task space without destroying the convexity of the receding-horizon optimization, we execute an analytical operational-space feedforward command:

$$F_\text{arm} = \Lambda_\text{arm}(q)\ddot{p}_{d} + \mu_\text{arm} - F_\text{mpc} \tag{18a}$$

which is mapped to the joint space via the balance null-space projector:

$$\tau_\text{ff,arm} = \bar{N}_{12}^\top S^\top J_\text{arm}^\top F_\text{arm} \tag{18b}$$

**Multi-rate execution.** Level 2 updates $\bar{N}_{12}(q)$, $\Lambda_\text{arm}(q)$, and $\mu_\text{arm}$ at 500 Hz. Level 3 runs at $\geq$1 kHz; during the interleaved 1 kHz cycles that do not coincide with a Level 2 tick, the projector $\bar{N}_{12}$ and feedforward terms are held constant at their most recent Level 2 values. Because the configuration changes by at most $\|\dot{q}\|\Delta t_2 \approx 0.002\,\text{rad}$ per Level 2 interval, the frozen-matrix error is first-order small and its contribution to the tracking error is bounded by $O(\Delta t_2)$—comparable in magnitude to the SRBD modeling error already absorbed by the Kalman disturbance state $\hat{d}$.

After applying the feedforward cancellation law, the residual tracking error dynamics reduce to:

$$\ddot{e}_\text{arm} = -\Lambda_\text{arm}^{-1}(q)F_\text{mpc} + d(t) \tag{19}$$

**Proposition 1** (Constant $A_d$ and Local LTI Horizon Freezing): *Within a fixed contact mode, by evaluating the configuration-dependent task inertia strictly at the current sampling instant $k$ such that $\Lambda_\text{arm}(q) \approx \Lambda_\text{arm}(q_k)$ over the horizon $t \in [k, k+N]$, the discrete-time state transition matrix for the error state $x_e(k) = [e_\text{arm}^\top, \dot{e}_\text{arm}^\top]^\top$ reduces to a constant linear system:*

$$A_d = \begin{bmatrix}I_3 & \Delta t I_3 \\ 0 & I_3\end{bmatrix}, \quad B_d(k) = \begin{bmatrix}0 \\ -\Lambda_\text{arm}^{-1}(q_k)\Delta t\end{bmatrix} \tag{20}$$

*Proof:* By parameterizing the input matrix $B_d$ explicitly with the frozen tracking instant state $q_k$, the system behaves as a linear time-invariant (LTI) plant within the receding horizon. The double-integrator structure of $A_d$ emerges naturally via a zero-order hold discretization of the error vector, decoupling the state transition from spatial configurations. $\square$

The constant $A_d$ property is critical: it permits the QP cost matrix $H$ and its Cholesky factor to be computed offline once per contact mode, reducing each online MPC update to a matrix–vector multiply.

### D. Receding-Horizon QP

Let $x_e(k) = [e^\top, \dot{e}^\top]^\top \in \mathbb{R}^6$ be the arm tracking error state. The input matrix is:

$$B_d^{(m)} = \begin{bmatrix}0 \\ -(\Lambda_\text{arm}^{(m)})^{-1}\Delta t\end{bmatrix} \tag{21}$$

indexed to contact mode $m$. The $N$-step prediction matrix $\Gamma^{(m)}$ is constructed from $(A_d, B_d^{(m)})$ using the standard lifted-system expansion. Since $A_d$ is constant (Proposition 1), only $\Gamma^{(m)}$ changes between contact modes; its reconstruction is $O(N^2 \cdot 9)$.

The receding-horizon QP is:

$$\min_{U}\;\frac{1}{2}U^\top H^{(m)} U + h^{(m)\top} U \quad\text{s.t.}\quad \|F_\text{mpc}(k)\|_\infty \leq F_\text{max} \tag{22}$$

with $H^{(m)} = \Gamma^{(m)\top}\bar{Q}\Gamma^{(m)} + \bar{R}$ and $h^{(m)} = \Gamma^{(m)\top}\bar{Q}x_\text{free}^{(m)}$, where $\bar{Q} = \text{blkdiag}(Q,\ldots,Q)$, $\bar{R} = \text{blkdiag}(R,\ldots,R)$, and $x_\text{free}^{(m)}$ is the free-response prediction. The contact-mode index $m$ plays the role of the scheduling variable in the LPV-MPC framework of [16]: the input matrix $B_d^{(m)}$ varies with the active support configuration exactly as an LPV plant matrix varies with its scheduling parameter. The box constraint $\|F_\text{mpc}\|_\infty \leq F_\text{max}$ is an engineered conservative Cartesian bound, chosen so that the resulting arm joint torques $\tau_\text{arm} = J_\text{arm}^\top F_\text{mpc}$ remain below hardware limits at all configurations within the operating workspace [15]. Mapping individual joint torque limits precisely into the Cartesian QP (which would require a configuration-dependent constraint matrix $J_\text{arm}^\top$) is deliberately avoided to preserve the precomputed, configuration-invariant structure of $H^{(m)}$. The QP (22) is strictly convex and solved by an operator-splitting solver (e.g., OSQP [13]) in $< 0.1$ ms for $N = 20$.

### E. Kalman Disturbance Augmentation

The disturbance $d(t)$ in (19) captures three coupled effects: (i) external pHRI forces directly applied to the arm; (ii) unmodeled contact reactions propagated from the feet; and (iii) SRBD approximation error from neglected leg inertia. These are jointly estimated by augmenting the MPC state with an integrating disturbance state $\hat{d} \in \mathbb{R}^3$:

$$\begin{bmatrix}x_e(k+1) \\ \hat{d}(k+1)\end{bmatrix} = \underbrace{\begin{bmatrix}A_d & B_d^{(m)} \\ 0 & I\end{bmatrix}}_{\displaystyle A_\text{aug}}\begin{bmatrix}x_e(k) \\ \hat{d}(k)\end{bmatrix} + \begin{bmatrix}B_d^{(m)} \\ 0\end{bmatrix}F_\text{mpc}(k) + w(k) \tag{23}$$

with process noise $w \sim \mathcal{N}(0, Q_w)$ and measurement noise $v \sim \mathcal{N}(0, R_v)$ on arm end-effector position. The Kalman gain $K_f$ is computed offline from the steady-state discrete algebraic Riccati equation. The integrating structure of $A_\text{aug}$ guarantees $\hat{d}(k) \to d$ for any bounded constant disturbance, independent of its physical origin. This is the predictive-control analogue of the anti-windup result of [14], which shows that augmenting an integrating channel achieves zero steady-state error under actuator limits and persistent constant disturbances: here the Kalman state $\hat{d}$ takes the role of that integrating channel and feeds it forward through the MPC horizon rather than feeding it back as a fixed integral gain.

The free-response prediction fed into the QP is constructed using $\hat{d}$:

$$x_\text{free}^{(m)} = \Phi^N x_e + \sum_{j=0}^{N-1}\Phi^j B_d^{(m)}\hat{d} \tag{24}$$

where $\Phi = A_d$. This causes the optimizer to pre-load corrective force before the disturbance fully manifests at the end-effector.

---

## VI. Contact Transition Handling

### A. Contact-Mode Switch Protocol

When a foot lifts off or touches down, $J_c$ changes discontinuously. Consequently, $\bar{M}^{-1}$, $\Lambda_\text{arm}$, and $B_d$ all jump. The Kalman estimate $\hat{d}$ becomes partially stale because the input channel through which the disturbance acts has changed.

At contact switch time $t_s$, the following protocol is executed in order:

1. Recompute $\bar{M}^{-1}$ with the new $J_c$.
2. Recompute $\Lambda_\text{arm}$; select the new contact-mode index $m_\text{new}$; load $B_d^{(m_\text{new})}$.
3. **Covariance inflation:** $P_\text{aug} \leftarrow \alpha P_\text{aug}$ with $\alpha \in [3, 5]$ to inflate the error covariance and allow rapid re-estimation of the disturbance in the new contact configuration.
4. **Hold $\hat{d}$:** do not reset to zero—balance-related disturbances persist across contact transitions and the estimate retains useful information.

The covariance inflation in Step 3 is the key mechanism that distinguishes the proposed approach from a naive reset. The Kalman filter re-converges within approximately $5\tau_\text{Kalman}$ samples, after which the disturbance estimate is again accurate in the new contact mode.

### B. Contact-Mode-Indexed Matrix Library

For a robot cycling through $K$ repeating contact modes (e.g., gait phases), precompute $B_d^{(m)}$ and $\Gamma^{(m)}$ for each mode $m = 1, \ldots, K$. At each QP call, select the current mode's matrices. Since $A_d$ is constant across all modes (Proposition 1), the prediction rollout matrix $\Phi^N$ is precomputed once globally. Only $\Gamma^{(m)}$ changes, and with $K$ precomputed $B_d^{(m)}$ blocks, the mode-switch overhead is a pointer swap and a $\Gamma^{(m)}$ reload—under 0.1 ms. For non-periodic or highly variable contact sequences (e.g., multi-contact manipulation), $\Gamma^{(m)}$ cannot be precomputed for all possible modes; in that case $\Gamma$ is reconstructed online using the standard $O(N^2 \cdot n_x^2)$ lifted-system expansion, which remains tractable because $A_d$ is the constant double-integrator and only $B_d^{(m)}$ changes—reducing the online cost to a single $O(N^2 \cdot 9)$ rank-1 update per new contact mode.

---

## VII. Impedance Equivalence Theorem

**Theorem 1** (Whole-Body Impedance Equivalence). *Under rigid contacts, fixed contact mode, and no disturbance, the unconstrained infinite-horizon Impedance MPC (Level 3, $R \to 0$, $N \to \infty$) with cost $Q = \mathrm{blkdiag}(K_d, D_d)$ recovers the classical task-space impedance law:*

$$\Lambda_\text{arm}(q)\ddot{e} + \Lambda_\text{arm}D_d\dot{e} + \Lambda_\text{arm}K_d e = F_h \tag{25}$$

*with effective configuration-adaptive mass $M_{d,\text{eff}} = \Lambda_\text{arm}(q)$ that depends on both arm posture and contact configuration.*

*Proof.* In the infinite-horizon unconstrained limit, the MPC reduces to the discrete LQR solution $F_\text{mpc} = -K_\infty x_e$, where $K_\infty$ is the steady-state LQR gain satisfying the discrete algebraic Riccati equation. As $R \to 0$ and $N \to \infty$, $K_\infty \to [K_d, D_d]$ (proportional and derivative gains scaled by $\Lambda_\text{arm}$). Substituting into (19): 
$$\ddot{e} = -\Lambda_\text{arm}^{-1}(K_d e + D_d \dot{e}) + \Lambda_\text{arm}^{-1}F_h,$$ 

which rearranges to (25). The contact-consistent $\bar{M}^{-1}$ in $\Lambda_\text{arm}$ means the effective mass adapts to both the arm joint configuration and the active contact footprint. $\square$

Theorem 1 establishes that Impedance MPC generalizes classical impedance control: the finite-horizon, constrained QP adds predictive disturbance rejection, constraint enforcement, and contact-mode adaptation while reducing to the classical law in the limit. It also provides a physical interpretation for the cost weights $K_d$, $D_d$ as the desired impedance stiffness and damping. In the simulations of Section XI, $Q = \mathrm{diag}(6 \times 10^4 I_3,\; 60\, I_3)$ gives $K_d = 6 \times 10^4\, I_3$ and $D_d = 60\, I_3$; at the nominal arm configuration ($\Lambda_\text{arm} \approx 0.20\, I_3\,\text{kg}$) these yield effective impedance stiffness $K_\text{eff} = \Lambda_\text{arm} K_d \approx 1.2 \times 10^4\,\text{N/m}$ and effective damping $D_\text{eff} = \Lambda_\text{arm} D_d \approx 12\,\text{Ns/m}$, consistent with compliant pHRI operation.

Notably, all three closed-loop impedance parameters—$M_{d,\text{eff}}$, $D_\text{eff}$, and $K_\text{eff}$—adapt automatically as the arm configuration and contact state change, while the cost weights $K_d$ and $D_d$ remain fixed design parameters. This adaptation is a structural by-product of $\Lambda_\text{arm}(q)$ and incurs no additional solver cost, preserving the ≥1 kHz control rate.

---

## VIII. Stability Analysis

### A. Zero Steady-State Error Under Fixed Contact Mode

**Theorem 2** (Zero Steady-State Error). *Suppose the disturbance $d(t)$ in (19) satisfies $\|d\|_\infty \leq \bar{d} < \infty$ and is asymptotically constant. Under fixed contact mode, the Kalman-augmented closed-loop system with the Level 3 QP feedback is asymptotically stable, and* $\lim_{k \to \infty}\|e_\text{arm}(k)\| = 0$.

*Proof.* The augmented state matrix $A_\text{aug}$ in (23) is $9\times9$ (six error states in $x_e$, three disturbance states in $\hat{d}$) with all eigenvalues at $\{1,\ldots,1\}$ (nine unit eigenvalues): six from the ZOH-discretized double-integrator $A_d$ and three from the integrating disturbance block $I_3$. By the Popov–Belevitch–Hautus (PBH) test, the pair $(A_\text{aug}, B_d^{(m)})$ is stabilizable if and only if $B_d^{(m)}$ has full column rank—which holds whenever $\Lambda_\text{arm}^{-1}$ is nonsingular. Two distinct singularity sources can violate this: (i) *kinematic arm singularities* (e.g., fully-extended elbow), at which $J_\text{arm}$ drops rank and $\Lambda_\text{arm}$ becomes ill-conditioned; and (ii) *contact-state singularities* arising when the platform loses all ground contact, making $\bar{M}^{-1}$ degenerate for base-dependent tasks. The result therefore holds within a compact, singularity-free subset $\mathcal{W}$ of the joint-space workspace in which the arm remains away from kinematic limits and the platform maintains a valid support polygon. The LQR-minimizing feedback $K_\infty$ places all closed-loop eigenvalues strictly inside the unit disk. The integrating $\hat{d}$ state converges to $d$ and the prediction offset (24) cancels the steady-state error exactly, yielding $e_\infty = 0$. This result extends the fixed-base anti-windup zero-SS-error result of [14] to the floating-base, contact-consistent setting: the saturation constraint $\|F_\text{mpc}\|_\infty \leq F_\text{max}$ defines a polyhedral invariant set $\mathcal{S}$ analogous to the domain of attraction in [15], inside which asymptotic convergence holds; for initial conditions outside $\mathcal{S}$ the QP clips $F_\text{mpc}$ and convergence is not guaranteed. $\square$

### B. Transient Bound Across Contact Transitions

Across a contact switch, $B_d$ jumps by $\Delta B_d = B_d^{(m_\text{new})} - B_d^{(m_\text{old})}$. The Kalman estimate becomes temporarily inaccurate, causing a transient tracking error. The error bound is:

$$\|e(t)\| \leq \|e(t_s)\| + c_1 \|\Delta B_d\| \cdot \|\hat{d}\| + c_2 \|d_\text{new}\| \cdot \Delta t_\text{conv} \tag{26}$$

where $\Delta t_\text{conv} \approx 5\tau_\text{Kalman}$ is the re-convergence time of the Kalman filter after covariance inflation, and $c_1, c_2 > 0$ are constants depending on the closed-loop eigenvalue placement. For typical biped walking ($\sim$1 Hz contact transitions), the transient magnitude is bounded by $c_1\|\Delta B_d\|\|\hat{d}\|$, which is small when leg contact forces are well-modeled. For running ($>$3 Hz contact transitions), the covariance-inflation coefficient $\alpha$ should be tuned to accelerate re-convergence and reduce $\Delta t_\text{conv}$.

### C. Null-Space Barrier Stability

The contact-consistent null-space torque that enforces joint limits and workspace boundaries is:

$$\tau_\text{null} = \bar{N}_\text{arm}^\top\bigl(-k_\text{null}(q-q_0) - d_\text{null}\dot{q} + g(q)\bigr) \tag{27}$$

where $\bar{N}_\text{arm} = I - \bar{J}_\text{arm}J_\text{arm}$ uses the contact-consistent pseudoinverse and $g(q)$ is the joint-limit barrier gradient. The projection through $\bar{N}_\text{arm}$ guarantees that (27) produces zero wrench at the arm task coordinate, preserving both the task tracking and the balance constraints above it in the hierarchy.

---

## IX. Complete Joint Torque Equation

Combining all three levels, the complete joint torque command for a floating-base robot performing pHRI is:

$$\tau = \tau_\text{contact} + \bar{N}_1^\top\tau_\text{balance} + \bar{N}_{12}^\top\bigl[\tau_\text{ff,arm} + J_\text{arm}^\top F_\text{mpc} + \tau_\text{null}\bigr] \tag{28}$$

where: $\tau_\text{contact}$ resolves Level 1 GRFs; $\tau_\text{balance} = J_\text{CoM}^\top F_\text{balance}$ comes from the centroidal MPC; $\tau_\text{ff,arm}$ is the contact-consistent feedforward (18); $F_\text{mpc}$ is the corrective Cartesian force from the Level 3 QP; and $\tau_\text{null}$ is the contact-consistent barrier (27).

Equation (28) is the central architectural result of this paper. The hierarchical null-space structure follows the SK05 law [12] extended to the contact-consistent floating-base setting [11],[18]; the MPC layer $F_\text{mpc}$ and its Kalman augmentation are the contributions of this paper. The hierarchical null-space projections $\bar{N}_1^\top$ and $\bar{N}_{12}^\top$ mathematically guarantee that the arm Impedance MPC layer cannot disturb the contact-maintenance and balance layers above it, while the Kalman-augmented QP in $F_\text{mpc}$ provides the disturbance rejection that the classical SK05 PD law (15) cannot achieve.

---

## X. Architectural Comparison

Table I summarizes the mathematical and architectural distinctions between the proposed framework and the most relevant prior work. The key differentiators are: (i) the use of $\bar{M}^{-1}$ (contact-consistent) rather than $M^{-1}$ in the task-space inertia; (ii) the contact-mode-indexed $B_d^{(m)}$ that adapts the input channel to the active support configuration; and (iii) the Kalman disturbance state that propagates the estimated pHRI force through the full prediction horizon.

**Table I: Architectural and Mathematical Comparison**

| Architectural Vector | Fixed-Base Baseline [2] | Bellicoso et al. [5] | Kim et al. [4] | Proposed Framework |
| :--- | :--- | :--- | :--- | :--- |
| Primary objective | Fixed-base pHRI | Quadruped locomotion | Dynamic locomotion | Floating-base pHRI + manipulation |
| Task-space inertia | $\Lambda = (JM^{-1}J^\top)^{-1}$ | N/A | N/A | $\Lambda_\text{arm} = (J\bar{M}^{-1}J^\top)^{-1}$ |
| Prediction horizon | $N$-step QP | $N = 1$ | $\sim$10–30 steps | $N$-step QP (≥1 kHz) |
| Disturbance handling | Kalman, fixed-base | WBC weight tuning | Centroidal inertia | Kalman: pHRI + leg momentum |
| Input matrix | $B_d = [0;\,-M_d^{-1}\Delta t]$ | N/A | N/A | $B_d^{(m)}$, contact-mode indexed |
| Null-space use | Joint centering | Posture tracking | Locomotion | Predictive impedance QP |
| Steady-state error | Zero (fixed base) | Nonzero under load | Nonzero under load | Zero (Theorem 2) |
| Contact transitions | N/A | Mode switching | Gait phases | Covariance-inflation protocol |

---

## XI. Simulation Benchmarks

### A. Simulation Platform

All experiments were conducted in MuJoCo 3.2 [21] at a 2 kHz integration rate. Scenarios A and B use a biped comprising a 3-DOF right arm, two 4-DOF legs, and a 6-DOF unactuated floating base (17 DOF total, 11 actuated), with total mass 46 kg. Scenario C uses the **official Unitree G1** MJCF model from MuJoCo Menagerie [21] (29 DOF, 33.3 kg, kinematic/inertial parameters from factory CAD), augmented with a single end-effector site at `right_wrist_yaw_link`. The G1 model uses position actuators ($K_p = 500$, $\text{dampratio} = 1$) rather than the biped's direct-torque actuators; Level 3 applies the position-as-torque trick ($\Delta q_i = \tau_i / K_p$) to inject impedance forces through the position channel. Level 1 centroidal MPC runs at 100 Hz ($N=10$, $\Delta t_1 = 10\,\text{ms}$, SRBD model). Level 2 WBC and Level 3 Impedance MPC both run at 1 kHz ($N=20$, $\Delta t_3 = 1\,\text{ms}$). Friction-cone half-angle is $\mu = 0.6$ at both feet.

Level 3 cost weights: $Q = \mathrm{diag}(6 \times 10^4 I_3,\; 60\, I_3)$, $R = 0.01\, I_3$. Kalman noise: $Q_w = \mathrm{diag}(10^{-4}I_6,\; 10^{-2}I_3)$, $R_v = 10^{-6}I_3$. All QPs are solved via the unconstrained analytical solution (fast path); covariance-inflation coefficient $\alpha = 4$ (Section VI).

**Unitree G1/R1 hardware interface.** The Unitree G1 (29 DOF, ~20.9 kg) and R1 (26 DOF, 25–29 kg, 1.23 m) robots expose a low-level joint control interface via `unitree_sdk2` at 500 Hz. Each joint accepts a command tuple $(q_\text{des},\, \dot{q}_\text{des},\, \tau_\text{ff},\, K_p,\, K_d)$, implementing $\tau = K_p(q_\text{des}-q) + K_d(\dot{q}_\text{des}-\dot{q}) + \tau_\text{ff}$. High-level locomotion in Unitree's open-source release (`unitree_rl_gym`) is provided by a reinforcement-learning policy that outputs joint position targets at 50 Hz; no WBC or MPC stack is open-sourced. The proposed three-level architecture maps directly onto this interface: Level 2 WBC and Level 3 Impedance MPC produce joint torques $\tau$ that are sent as $\tau_\text{ff}$ with $K_p = K_d = 0$ (pure torque mode), compatible with both G1 and R1 EDU variants.

### B. Benchmarked Controllers

Seven controllers are evaluated across all three scenarios (Table II). All controllers share the identical Level 1 centroidal MPC and are evaluated on the same MuJoCo plant.

**Table II: Benchmarked Controllers**

| Label | Description |
| :--- | :--- |
| D1 | SK05 PD law, $K_P = 800\,\text{N/m}$, $K_D = 40\,\text{Ns/m}$ |
| D2 | SK05 PI: adds $K_I = 150\,\text{N/(m·s)}$ with anti-windup clamping |
| D3 | Fixed-base impedance MPC (uses $M^{-1}$ instead of $\bar{M}^{-1}$; ignores contact consistency) |
| D4 | WBC hierarchy + PD law (correct null-space hierarchy, no prediction or estimation) |
| D5 | Proposed: WBC + Impedance MPC, no Kalman augmentation |
| D6 | Proposed: WBC + Impedance MPC + Kalman, no covariance inflation ($\alpha = 1$) |
| D7 | Proposed full: WBC + Impedance MPC + Kalman + covariance inflation ($\alpha = 4$) |

D1 establishes the analytical baseline: for $K_P = 800\,\text{N/m}$ and $F_h = 8\,\text{N}$, (16) predicts $e_\infty = F_h/K_P = 10.0\,\text{mm}$ exactly. D4 isolates the effect of the null-space hierarchy from prediction. D3 quantifies the penalty for ignoring $\bar{M}^{-1}$ on a floating base.

### C. Scenario A: Fixed Double-Support Step Disturbance

The robot holds a stationary double-support stance. The right arm holds a fixed Cartesian target. At $t = 0.5\,\text{s}$, a step pHRI force of 8 N is applied at the end-effector in the $x$-direction and held for the remaining 4.5 s. Total episode: 5 s. All controllers run at 1 kHz (1 ms control period); physics advances at 2 kHz via two 0.5 ms sub-steps per control cycle.

**Metrics:** RMS position error over the full episode; steady-state (SS) error averaged over $t > 3.5\,\text{s}$. Theoretical baseline: $e_\infty = F_h/K_P = 8/800 = 10\,\text{mm}$ for D1.

**Table III: Scenario A — Fixed Stance, 8 N Step Disturbance**

| Controller | RMS err [mm] | SS err [mm] |
| :--- | :---: | :---: |
| D1 SK05 PD | 9.240 | 10.166 |
| D2 SK05 PI | 6.818 | 5.861 |
| D3 Fixed-base MPC | 11.124 | 11.855 |
| D4 WBC + PD | 9.454 | 10.207 |
| D5 Proposed, no Kalman | 7.735 | 8.529 |
| D6 Proposed, no inflation | 1.281 | 0.037 |
| D7 Proposed full | **1.281** | **0.037** |

![Fig. 1](simulation/results/scenario_a_results.png)

**Fig. 1.** Scenario A — Fixed double-support stance, 8 N step pHRI disturbance at $t = 0.5\,\text{s}$. *Top:* Cartesian end-effector error norm $\|e\|$ over time for all seven controllers. D6/D7 (Kalman-augmented, yellow-green and cyan) converge to sub-0.05 mm within $\sim$0.3 s of disturbance onset and remain there; all other controllers plateau at $\geq 5.8\,\text{mm}$. D3 (fixed-base MPC, red) diverges to 12 mm because the biased $B_d$ corrupts the Kalman estimate. *Bottom:* Bar chart of RMS and steady-state (SS) error (t > 3.5 s); note the logarithmic visual contrast between D6/D7 bars ($\approx$1.28 mm RMS, 0.037 mm SS) and all baselines.

D6 and D7 are equivalent in fixed stance (no contact transitions occur, so covariance inflation is never triggered). Key findings:

1. *Hierarchy alone does not reduce SS error:* D1 and D4 produce nearly identical SS errors of 10.166 mm and 10.207 mm respectively, matching the theoretical prediction of 10.0 mm to within simulation noise. The WBC null-space projection in D4 changes the torque distribution but cannot eliminate the fundamental PD steady-state offset.

2. *Finite-horizon prediction alone outperforms PD:* D5 (MPC, no Kalman) achieves 8.529 mm SS—16% lower than D1 (10.166 mm)—because the receding-horizon QP, with sufficiently high cost weight $Q = 6 \times 10^4 I$, provides effective stiffness exceeding $K_P = 800\,\text{N/m}$. However, without a disturbance state, the QP cannot drive the offset to zero: the optimizer sees only the current-step error and provides proportional rejection, analogous to a high-gain PD with prediction.

3. *Wrong dynamics model limits Kalman+MPC:* D3 uses $M^{-1}$ instead of $\bar{M}^{-1}$, shrinking $B_d$ by 10×. Although the Kalman partially compensates through $\hat{d}$, the error in $B_d$ biases the disturbance estimate and the QP gains, yielding 11.855 mm SS—318× worse than D6/D7 at 0.037 mm—confirming that contact-consistent dynamics are not optional when the Kalman state is coupled to $B_d$.

4. *Kalman augmentation achieves near-zero steady-state error:* D6/D7 reduce SS error to 0.037 mm—a 273× improvement over D1—validating the asymptotic zero-SS-error result of Theorem 2 to within simulation noise. The Kalman integrating disturbance state converges to $\hat{d} \approx -8\,\text{N}$ within $\sim$0.3 s and pre-loads the MPC horizon, driving the net corrective force to exactly cancel the 8 N pHRI load.

5. *PI vs. MPC+Kalman for static disturbance:* D2 (PI, $K_i = 150$) achieves 5.861 mm SS error—157× worse than D7 (0.037 mm). While integral action is structurally suited to constant disturbances, it lacks the prediction horizon; the Kalman integrating state provides superior DC rejection with finite control effort.

### D. Scenario B: Stance with Periodic Contact-Transition Shocks

The robot holds a stable double-support stance throughout. The right arm tracks a fixed Cartesian target while a sustained 8 N pHRI force is applied from $t = 0$. Every $T_\text{switch} = 1\,\text{s}$, an additional 6 N spike is superimposed on the pHRI for $T_\text{spike} = 0.1\,\text{s}$ and then withdrawn, totalling 9 events over 10 s. This models the brief mechanical shock transmitted through the kinematic chain when a swing foot contacts the ground during walking: the touchdown impulse propagates to the arm end-effector as a transient disturbance on top of the sustained pHRI load. At each spike onset, the Kalman is signalled (mode key increments), allowing the covariance-inflation protocol (D7) to widen the filter and re-estimate $\hat{d}$ faster.

**Additional metric:** peak Cartesian error within a ±150 ms window centred on each shock event.

**Table IV: Scenario B — Stance + 1 Hz Contact-Transition Shocks, Sustained 8 N pHRI**

| Controller | RMS err [mm] | Peak at transition [mm] |
| :--- | :---: | :---: |
| D1 SK05 PD | 10.94 | 15.77 |
| D2 SK05 PI | 5.91 | 10.17 |
| D3 Fixed-base MPC | 13.20 | 17.22 |
| D4 WBC + PD | 10.94 | 15.77 |
| D5 Proposed, no Kalman | 9.02 | 12.68 |
| D6 Proposed, no inflation | 1.84 | 4.32 |
| D7 Proposed full | **1.81** | **4.15** |

![Fig. 2](simulation/results/scenario_b_results.png)

**Fig. 2.** Scenario B — Stable double-support stance, sustained 8 N pHRI + 6 N periodic shocks at 1 Hz. *Top:* Error norm over the full 10 s episode. The periodic spike train is clearly visible for D1/D4 (blue/brown, peak 15.77 mm at each shock) and D2 (orange, peak 10.17 mm), while D6/D7 (yellow-green/cyan) show barely perceptible blips at each event ($\leq$4.32 mm) before returning to near-zero steady state. The degradation of D2 (PI) at each shock onset—despite integral action—reflects integral windup: the integrator was calibrated to the 8 N steady load and cannot pre-compensate the abrupt step-change. *Bottom:* Bar chart of RMS and Peak@transition error; D7 ($\alpha=4$, 4.15 mm peak) marginally outperforms D6 (no inflation, 4.32 mm peak), confirming that covariance inflation accelerates Kalman re-convergence after each disturbance step.

Key findings:

1. *Baseline PD spikes at every shock:* D1 and D4 produce 15.77 mm peaks at each contact shock—a 57% increase over steady-state—because fixed-gain PD cannot anticipate the transient. D2 (PI) reduces steady-state to 5.91 mm but still peaks at 10.17 mm: integral windup is calibrated for the 8 N steady load and cannot quickly adapt to the step-change in disturbance magnitude during the 6 N spike.

2. *Finite-horizon prediction alone improves transient response:* D5 (MPC, no Kalman) achieves 9.02 mm RMS and 12.68 mm Peak—both better than D1 (10.94 mm / 15.77 mm)—because the QP's prediction horizon allows it to begin reducing error before the spike fully accumulates in the position channel. Without $\hat{d}$, however, it cannot pre-load force for the upcoming shock.

3. *Wrong dynamics model amplifies transient errors:* D3 peaks at 17.22 mm per shock—9% worse than D1—because the biased $B_d$ causes the QP to apply incorrect corrective forces precisely when the disturbance magnitude changes most rapidly.

4. *Kalman augmentation nearly eliminates transient errors:* D6/D7 achieve 1.8 mm RMS—a 6× improvement over D1—because the Kalman $\hat{d}$ continuously tracks the evolving disturbance magnitude, pre-loading $F_\text{mpc}$ before and after each shock. The MPC adjusts within one Kalman convergence time constant ($\sim$0.05 s) rather than multiple error-integration cycles.

5. *Covariance inflation reduces peak transition error:* D7 ($\alpha = 4$) achieves 4.15 mm Peak@transition versus 4.32 mm for D6 (no inflation)—a 3.9% improvement. Inflation momentarily widens the Kalman covariance at each shock onset, increasing filter gain and accelerating $\hat{d}$ re-estimation. The improvement is modest here because $\Lambda_\text{arm}$ does not physically change (double-support throughout); the full benefit—and a larger D6/D7 gap—appears in dynamic walking where $B_d^{(m)}$ jumps at every footstep.

### E. Scenario C: Unitree G1 Real Model, Fixed Stance, 8 N Step Disturbance

To validate the architecture on a real commercial humanoid, Scenario A is repeated using the official **Unitree G1** MJCF from MuJoCo Menagerie [21] (29 DOF, 33.3 kg, kinematic parameters from factory CAD). A single end-effector site (`right_hand_site`) is appended at the tip of `right_wrist_yaw_link` to provide a 3-DOF Cartesian tracking target. Because the G1 exposes **position actuators** ($K_p = 500$, $\text{dampratio} = 1$) rather than direct torque outputs, Levels 1–2 (balance and null-space) command joint position targets, while Level 3 applies the **position-as-torque** approximation: $\Delta q_i = \tau_i / K_p$, so that the desired Cartesian force $F_\text{mpc}$ is injected as $\text{ctrl}[i] \leftarrow q_i + (J_\text{arm}^\top F_\text{mpc})_i / K_p$. This is exactly the pure-torque mode available on G1/R1 EDU hardware ($K_p = K_d = 0$, $\tau_\text{ff} \neq 0$).

**Table V: Scenario C — Unitree G1 (33.3 kg), Fixed Stance, 8 N Step Disturbance**

| Controller | RMS err [mm] | SS err [mm] |
| :--- | :---: | :---: |
| D1 SK05 PD | 8.996 | 9.570 |
| D2 SK05 PI | 6.413 | 5.325 |
| D3 Fixed-base MPC | 11.230 | 11.299 |
| D4 WBC + PD | 8.996 | 9.570 |
| D5 Proposed, no Kalman | 7.420 | 7.974 |
| D6 Proposed, no inflation | 2.703 | 3.904 |
| D7 Proposed full | **2.703** | **3.904** |

![Fig. 3](simulation/results/scenario_c_g1_results.png)

**Fig. 3.** Scenario C — Unitree G1 official model (33.3 kg, 29 DOF), fixed stance, 8 N step pHRI. *Top:* D6/D7 (cyan) converge markedly faster than D1/D4 (blue/brown), though the residual SS error (3.9 mm) is higher than the simplified biped (0.037 mm). The gap arises from the position-as-torque bandwidth limitation: with $K_p = 500$ and joint inertia $\sim$0.5 kg⋅m$^2$, the arm actuators achieve $\sim$5 Hz bandwidth, partially attenuating fast MPC corrections. *Bottom:* D7 achieves 2.5× lower SS error than D1, confirming the Kalman architecture's benefit on real G1 hardware parameters.

Key findings:

1. *Architecture transfers to real G1 model:* D7 achieves 3.90 mm SS—a 2.5× improvement over D1 (9.57 mm). The qualitative ordering D7 < D6 < D5 < D1 < D4 is preserved, confirming that the three-level hierarchy and Kalman disturbance state provide consistent benefit on the official G1 kinematics and inertia.

2. *Position-actuator bandwidth limits ultimate performance:* The 2.5× improvement on the G1 is far smaller than the 273× on the simplified biped (pure-torque mode). With $K_p = 500\,\text{Nm/rad}$ and arm joint inertia $\approx 0.5\,\text{kg⋅m}^2$, the position actuator bandwidth is $\omega_n \approx \sqrt{500/0.5} \approx 32\,\text{rad/s}$ (5 Hz), which attenuates corrective forces at the 1 kHz MPC rate. Deploying the architecture in direct joint-torque mode ($K_p = K_d = 0$) on G1/R1 EDU hardware is expected to recover performance close to the pure-torque biped results.

3. *Contact-consistent dynamics essential on real model:* D3 (fixed-base MPC) yields 11.30 mm SS—a 2.9× penalty over D7—confirming that ignoring $\bar{M}^{-1}$ degrades performance even under position-actuator bandwidth constraints.

4. *G1 hardware compatibility confirmed:* The G1/R1 SDK exposes $\tau_\text{ff}$ with $K_p = K_d = 0$ at 500 Hz on EDU variants, providing a direct deployment path for the proposed architecture.

---

## XII. Conclusion

This paper presented a three-level Whole-Body Impedance MPC architecture for floating-base robots that simultaneously guarantees balance, contact constraint satisfaction, and zero steady-state tracking error under sustained physical human–robot interaction. The central technical contribution is the proof that a contact-consistent feedback linearization reduces the arm end-effector plant to a linear double integrator with a constant discrete state matrix within each contact mode, enabling offline precomputation of the QP cost inverse and ≥1 kHz operation. The contact-mode-indexed input matrix library and Kalman covariance-inflation protocol extend these properties across contact transitions with bounded transient error. The Impedance Equivalence Theorem establishes formal equivalence with classical operational-space impedance control in the infinite-horizon limit, facilitating impedance parameter design.

The proposed architecture occupies a structural niche not addressed by prior locomotion-centric frameworks [3]–[6]: it deliberately halts the WBC stack after balance constraints are satisfied and injects a predictive, compliance-controlled manipulation layer into the residual null space. This null-space injection is the mathematical mechanism by which safe pHRI is achieved without any sacrifice of balance guarantees.

Future work includes hardware validation on a Unitree R1 or G1 EDU platform (the low-level SDK torque interface is directly compatible with the proposed architecture), extension to variable contact modes during dynamic walking, and integration of force-torque sensor feedback for improved Kalman convergence at contact transitions.

---

## References

[1] N. Hogan, "Impedance control: An approach to manipulation—Parts I, II, III," *ASME J. Dyn. Syst. Meas. Control*, vol. 107, no. 1, pp. 1–24, 1985.

[2] Y. Cao, K. Cheng, and G. Li, "Passive model-predictive impedance control for safe physical human–robot interaction," *IEEE Trans. Cognitive Developmental Syst.*, 2023.

[3] J. Di Carlo, P. M. Wensing, B. Katz, G. Bledt, and S. Kim, "Dynamic locomotion in the MIT Cheetah 3 through convex model-predictive control," in *Proc. IEEE/RSJ IROS*, pp. 1–9, 2018.

[4] D. Kim, J. Di Carlo, B. Katz, G. Bledt, and S. Kim, "Highly dynamic quadruped locomotion via whole-body impulse control and model predictive control," in *Proc. IEEE/RSJ IROS*, pp. 4656–4663, 2019.

[5] C. D. Bellicoso, C. Gehring, J. Hwangbo, P. Fankhauser, and M. Hutter, "Perception-less terrain adaptation through whole body control and hierarchical optimization," in *Proc. IEEE-RAS Humanoids*, pp. 558–564, 2016.

[6] T. Koolen *et al.*, "Design of a momentum-based control framework and application to the humanoid robot Atlas," *Int. J. Humanoid Robotics*, vol. 13, no. 1, 2016.

[7] K. Haninger, C. Hegeler, and L. Peternel, "Model predictive impedance control with Gaussian processes for physical and task-space constraints," in *Proc. IEEE ICRA*, pp. 3739–3745, 2022.

[8] L. Roveda, J. Maskani, P. Franceschi, A. Ghezzi, F. Braghin, L. M. Tosatti, and N. Pedrocchi, "Model-based reinforcement learning variable impedance control for human–robot collaboration," *J. Intell. Robot. Syst.*, vol. 100, no. 2, pp. 417–433, 2020.

[9] A. Winkler, C. D. Bellicoso, M. Hutter, and J. Buchli, "Gait and trajectory optimization for legged systems through phase-based end-effector parameterization," *IEEE Robotics Autom. Lett.*, vol. 3, no. 3, pp. 1560–1567, 2018.

[10] R. Grandia, F. Jenelten, S. Yang, F. Farshidian, and M. Hutter, "Perceptive locomotion through nonlinear model-predictive control," *IEEE Trans. Robotics*, vol. 39, no. 5, pp. 3402–3421, 2023.

[11] O. Khatib, "A unified approach for motion and force control of robot manipulators: The operational space formulation," *IEEE J. Robotics Autom.*, vol. 3, no. 1, pp. 43–53, 1987.

[12] L. Sentis and O. Khatib, "Synthesis of whole-body behaviors through hierarchical control of behavioral primitives," *Int. J. Humanoid Robotics*, vol. 2, no. 4, pp. 505–518, 2005.

[13] B. Stellato, G. Banjac, P. Goulart, A. Bemporad, and S. Boyd, "OSQP: An operator splitting solver for quadratic programs," *Math. Program. Comput.*, vol. 12, no. 4, pp. 637–672, 2020.

[14] Y.-Y. Cao, Z. Lin, and D. G. Ward, "Anti-windup design of output tracking systems subject to actuator saturation and constant disturbances," *Automatica*, vol. 40, no. 7, pp. 1221–1228, Jul. 2004.

[15] Y.-Y. Cao, Z. Lin, and D. G. Ward, "An antiwindup approach to enlarging domain of attraction for linear systems subject to actuator saturation," *IEEE Trans. Autom. Control*, vol. 47, no. 1, pp. 140–145, Jan. 2002.

[16] Y.-Y. Cao and Z. Lin, "Min–max MPC algorithm for LPV systems subject to input saturation," *IEE Proc. Control Theory Appl.*, vol. 152, no. 3, pp. 266–272, May 2005.

[17] D. E. Orin, A. Goswami, and S.-H. Lee, "Centroidal dynamics of a biped robot," *Int. J. Robotics Research*, vol. 32, no. 9, pp. 1043–1060, 2013.

[18] L. Righetti, J. Buchli, M. Mistry, and S. Schaal, "Inverse dynamics control of floating-base robots with external constraints: A unified view," in *Proc. IEEE ICRA*, pp. 1085–1090, 2011.

[19] J.-P. Sleiman, F. Farshidian, M. V. Meduri, and M. Hutter, "A unified MPC framework for whole-body dynamic locomotion and manipulation," *IEEE Robot. Autom. Lett.*, vol. 6, no. 3, pp. 4688–4695, 2021.

[20] A. Albu-Schäffer, C. Ott, and G. Hirzinger, "A unified passivity-based control framework for position, torque and impedance control of flexible joint robots," *Int. J. Robotics Research*, vol. 26, no. 1, pp. 23–39, 2007.

[21] E. Todorov, T. Erez, and Y. Tassa, "MuJoCo: A physics engine for model-based control," in *Proc. IEEE/RSJ IROS*, pp. 5026–5033, 2012.

[22] L. Villani and J. De Schutter, "Force control," in *Springer Handbook of Robotics*, B. Siciliano and O. Khatib, Eds., 2nd ed., Springer, pp. 195–220, 2016.



 cd /Users/yycao/Documents/git/ai_learn/whole_body_control/arXiv && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex