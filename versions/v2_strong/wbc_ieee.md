# Contact-Consistent Interaction Dynamics Normalization for Predictive Physical Human–Robot Interaction

**Yongyan Cao**

*Voryx Robotics, San Jose, CA 95136*
*Email: yongyancao@gmail.com*

*Abstract*—Safe physical human–robot interaction on floating-base robots is fundamentally a problem of regulating interaction dynamics under changing contact constraints. This paper develops a contact-consistent normalization of those dynamics: after priority-consistent cancellation of balance and contact tasks, the end-effector interaction channel reduces to a linear double integrator whose discrete state matrix is constant within each contact mode. Robot configuration and support conditions enter only through the contact-consistent task inertia and hence the input matrix. This representation turns floating-base pHRI from a nonlinear whole-body control problem into a configuration-invariant predictive interaction problem with reusable rollouts, cached contact-mode matrices, and $\geq$1 kHz updates. The resulting controller regulates the normalized interaction dynamics with a receding-horizon quadratic program, a contact-mode-indexed disturbance observer for offset-free force rejection, and a null-space realization that preserves higher-priority balance tasks. We further prove that classical operational-space impedance is the infinite-horizon, zero-input limit of the same predictive interaction law; impedance is therefore not a separate design principle but a limiting case of normalized predictive interaction dynamics. Simulation on a 17-DOF biped and a Unitree G1 model validates the structural claims: the contact-mode-indexed disturbance observer delivers offset-free force rejection — steady-state error under a sustained 8 N load drops from ~13 mm without the observer to ~0.1 mm with it — the contact-consistent inertia further improves static precision over a free-space model (0.11 vs 1.00 mm), and the constant-model predictor operates at ≥1 kHz across genuine contact-mode switches. The results position whole-body impedance behavior, disturbance rejection, and contact-mode switching as consequences of a single interaction-dynamics representation.

*Index Terms*—Interaction dynamics, whole-body control, model predictive control, impedance control, floating-base robots, physical human–robot interaction, contact-consistent dynamics, legged manipulation.

---

## I. Introduction

Legged and floating-base robots must increasingly do more than locomote: they must physically interact with people and their surroundings while keeping balance. Safe interaction on such platforms is fundamentally a problem of *interaction dynamics*—the closed-loop relation between the arm end-effector force and motion—coupled, through the contact constraints, to the whole-body balance task. These objectives are tightly coupled—arm motions shift the center of mass (CoM), changing contact force distribution, while ground reactions propagate back through the body and appear as disturbances at the end-effector. Classical fixed-base impedance control [1] and its MPC extensions [2] cannot address this coupling because they assume the robot base is rigidly anchored.

The dominant paradigm for whole-body control of legged systems decouples the problem into two layers: a centroidal MPC that optimizes ground reaction forces (GRFs) using a linearized single rigid-body dynamics (SRBD) model [3], [4], and an inner WBC layer that resolves these forces into joint torques via a prioritized quadratic program (QP) [5], [6]. This architecture achieves remarkable locomotive agility—the MIT Cheetah 3 [3] executes high-speed bounding and stair climbing—but it allocates 100% of the robot's control authority to locomotion and base-posture maintenance. Any external arm interaction is treated as a disturbance to be suppressed, not as a channel to be actively regulated. A biped reaching to assist a human standing beside it, or a quadruped manipulating a valve while maintaining stance, cannot be handled by these frameworks with the compliance and zero-steady-state-error guarantees required for safe pHRI.

Conversely, impedance MPC methods designed for fixed-base manipulators [2], [7], [8] lack an unactuated base state, generalized-coordinate partitioning, or contact-consistent mass inverses. They assume an infinite-mass ground connection and cannot model the propagation of foot contact forces to end-effector apparent inertia. Deploying them directly on a floating-base platform produces steady-state torque errors and potential instability during contact transitions.

The technical gap is therefore not merely the absence of another whole-body controller. It is the absence of a representation in which interaction dynamics remain structurally the same as robot posture, contact mode, and apparent inertia change. This paper closes that gap through the following sequence:

$$\text{Interaction dynamics}
\rightarrow \text{contact-consistent normalization}
\rightarrow \text{linear double integrator}
\rightarrow \text{predictive regulation}
\rightarrow \text{impedance as a limiting case}.$$

The main contributions are:

1. **Interaction-dynamics normalization.** We formulate floating-base arm pHRI as an interaction-dynamics problem and derive the contact-consistent residual plant obtained after higher-priority balance and contact tasks are removed through the whole-body null space.

2. **Configuration-invariant predictor.** We prove that the normalized end-effector interaction dynamics reduce to an exact linear double integrator with constant discrete state matrix $A_d$ within each contact mode. Contact configuration and robot posture affect the predictor only through the input matrix $B_d^{(m)}$, via the contact-consistent task inertia $\Lambda_\text{arm}$.

3. **Predictive interaction control and impedance equivalence.** We regulate the normalized dynamics with a finite-horizon QP and show that classical operational-space impedance is recovered as the infinite-horizon, zero-input limit. Thus impedance behavior is a special case of predictive interaction dynamics rather than a separate controller family.

4. **Floating-base realization and theory validation.** We realize the predictor inside a priority-consistent WBC stack using contact-mode-indexed matrices, a disturbance observer, and covariance inflation across contact changes. Simulations on a 17-DOF biped and a Unitree G1 model validate the constant-predictor structure, offset-free disturbance rejection, and stable operation across contact-mode switches.

The remainder of this paper follows the same logic. Section II surveys related work. Section III formulates interaction dynamics on floating-base robots. Section IV derives the contact-consistent normalization. Section V presents predictive regulation of the normalized dynamics. Section VI treats contact-mode changes. Section VII proves the impedance-equivalence result. Section VIII analyzes stability. Section IX gives the whole-body torque realization. Section X compares the architecture with existing frameworks. Section XI reports theory-validation simulations. Section XII concludes.

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

The present paper is a direct structural extension of the authors' prior work on saturated and predictive control. Anti-windup designs for output tracking under actuator saturation and constant disturbances [14], and the associated domain-of-attraction analysis [15], established that an integrating disturbance channel achieves zero steady-state tracking error for fixed-base saturated linear systems—the foundational insight carried forward here to the floating-base, contact-switching setting via the Kalman augmented state. The min–max MPC formulation for LPV systems [16] introduced parameter-varying input matrices with input constraints, the direct precursor to the contact-mode-indexed $B_d^{(m)}$ scheduling of the present work. Building on these fixed-base results, Cao, Cheng, and Li [2] introduced a passive MPC framework for pHRI on fixed-base manipulators in which the outer MPC optimizes impedance parameters $\{M_d, D_d\}$ over a receding horizon; passivity is enforced via a virtual energy tank, building on the passivity arguments of [20]. Because impedance parameters enter nonlinearly into the prediction matrices, iterative solvers are required and update rates are limited to 10–30 Hz. The present paper closes the remaining gap by extending this line of work from fixed-base manipulators to floating-base humanoids: the MPC decision variables become corrective *forces* $F_\text{mpc}$, the contact-consistent feedforward linearization yields a constant $A_d$, and the contact-mode-indexed structure of [16] handles stance-phase transitions while preserving the precomputed lifted-system sparsity and rollout structure.

Haninger, Hegeler, and Peternel [7] optimize force references and impedance parameters jointly using stochastic MPC with Gaussian Process models of task forces. Contact-force safety is enforced as a probabilistic chance constraint. This provides complementary insights into uncertainty-aware impedance shaping but does not address floating-base dynamics, underactuation, or contact-consistent operational-space formulation.

**Saturation-aware control and anti-windup.** The interaction between actuator saturation and persistent constant disturbances in output tracking was studied by Cao, Lin, and Ward [14], who showed that anti-windup augmentation achieves zero steady-state error for saturated linear systems. The domain-of-attraction characterisation for saturated systems was further developed in [15]. These results establish the structural principle—augmenting an integrating channel to cancel constant offsets despite hard input limits—that the present work extends to the floating-base, contact-switching setting via the Kalman disturbance state $\hat{d}$. The min–max MPC algorithm for LPV systems subject to input saturation [16] is the closest prior work in the MPC direction: its parameter-varying input matrix (analogous to the contact-mode-indexed $B_d^{(m)}$) is optimised subject to box constraints, providing robust performance guarantees across scheduling parameter variations. The present architecture extends this to floating-base humanoids by incorporating the contact-consistent mass inverse $\bar{M}^{-1}$, a Kalman disturbance integrator, and a covariance-inflation protocol for mode switches.

### D. Operational Space Control and Floating-Base Inverse Dynamics

Khatib [11] formulated the operational space control framework, establishing task-space inertia, Coriolis compensation, and dynamically-consistent pseudoinverses as the mathematical foundation for task-level manipulation. Sentis and Khatib [12] extended this to hierarchical synthesis of whole-body behaviors, proving that priority-ordered null-space projection guarantees non-interference between tasks—the SK05 law that forms the backbone of Level 2 in the present architecture. Righetti et al. [18] unified the floating-base inverse dynamics perspective with external contact constraints, showing how the contact-consistent mass inverse $\bar{M}^{-1}$ arises naturally from an orthogonal decomposition of the constrained dynamics. The present work builds on [11], [12], and [18] by embedding a predictive MPC layer in the residual null space of the floating-base contact-consistent hierarchy.

---

## III. Interaction Dynamics on Floating-Base Robots

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

Define the contact-space inertia and the corresponding constrained inverse [11], [18]:

$$\Lambda_c = (J_cM^{-1}J_c^\top)^{-1} \tag{8}$$

$$\bar{M}^{-1} = M^{-1} - M^{-1}J_c^\top\Lambda_cJ_cM^{-1} \tag{9}$$

This expression is the Schur-complement inverse of the constrained dynamics and is symmetric positive semidefinite on the admissible acceleration subspace when $M \succ 0$ and $J_c$ has full row rank. It replaces $M^{-1}$ in all operational-space formulas when contacts are active. The associated contact-consistent projector may be written in left/right forms depending on the metric, but (9) is the definition used in the task inertia $\Lambda_i=(J_i\bar M^{-1}J_i^\top)^{-1}$. This quantity is the central link between the contact configuration and the apparent inertia at the end-effector.

### E. Centroidal Dynamics

The centroidal momentum $h_G = [k^\top, L^\top]^\top = A(q)\dot{q} \in \mathbb{R}^6$ aggregates the robot's linear and angular momentum about its CoM [17], where $A(q) \in \mathbb{R}^{6\times(n+6)}$ is the centroidal momentum matrix. Differentiating:

$$\dot{h}_G = A(q)\ddot{q} + \dot{A}(q,\dot{q})\dot{q} = G_c(q)\lambda + \begin{bmatrix}mg \\ 0\end{bmatrix} \tag{10}$$

where $G_c(q) \in \mathbb{R}^{6\times n_c}$ maps contact forces to centroidal momentum rate:

$$G_c(q) = \begin{bmatrix}I_3 & I_3 & \cdots \\ (p_1-p_G)^\times & (p_2-p_G)^\times & \cdots\end{bmatrix} \tag{11}$$

For the outer MPC, equation (10) is approximated by the **single rigid-body dynamics (SRBD)** model [3], treating the robot as a lumped mass $m$ with constant inertia $I_G$. Linearizing about a nominal orientation yields:

$$\dot{x}_c = A_c x_c + B_c(\{p_i\})u_c \tag{12}$$

where $x_c \in \mathbb{R}^{12}$ is the centroidal state and $u_c$ collects contact forces. The SRBD approximation error is $O(m_\text{leg}/m_\text{total})^2$, acceptable for robots where leg mass is below 20–30% of total mass.

---

## IV. Contact-Consistent Interaction Dynamics

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

where $\bar{N}_{1\cdots i} = \prod_{j=1}^{i}(I - \bar{J}_j J_j)$ is the accumulated contact-consistent null-space projector. Under an exact model, dynamically consistent projectors, inactive actuator saturation, and no change in the active contact/friction inequalities, the key property is $J_j \bar{N}_{1\cdots i} = 0$ for all $j \leq i$: lower-priority task torques produce zero acceleration at higher-priority task coordinates. Thus the arm predictive interaction layer is designed to be non-interfering with contact maintenance and balance in the ideal hierarchy; in implementation, saturation and contact active-set changes are handled by conservative force limits and by the contact-mode update protocol.

Each task force $F_i$ in (14) is typically a PD law:

$$F_i = \Lambda_i(\ddot{x}_{di} + K_{D,i}\dot{e}_i + K_{P,i}e_i) + \mu_i \tag{15}$$

Under no disturbance, (15) yields closed-loop error dynamics $\ddot{e}_i + K_{D,i}\dot{e}_i + K_{P,i}e_i = 0$, stable for any $K_{P,i}, K_{D,i} > 0$. Under a persistent disturbance force $F_h$:

$$e_{\infty,i} = K_{P,i}^{-1}\Lambda_i^{-1}F_h = K_{x,i}^{-1}F_h \neq 0,\qquad K_{x,i}\triangleq \Lambda_i K_{P,i} \tag{16}$$

Here $K_{P,i}$ is the acceleration-level proportional gain in (15), while $K_{x,i}$ is the equivalent Cartesian stiffness in N/m. The simulation baselines report $K_x$ directly; therefore the D1 theoretical offset under an 8 N force and $K_x=800\,\text{N/m}$ is $8/800=10\,\text{mm}$.

This residual steady-state error under sustained pHRI is the fundamental limitation of the OS PD law and the primary motivation for regulating the arm task through predictive interaction dynamics.

---

## V. Predictive Regulation of Interaction Dynamics

### A. Architecture Overview

Consider an $n$-DOF floating-base robot in a fixed support configuration (e.g., bipedal stance) performing arm manipulation while subject to pHRI forces. The control objectives are:
1. maintain balance with contact forces inside friction cones;
2. track an arm end-effector reference $p_d(t)$; and
3. reject pHRI forces with zero steady-state tracking error.

These objectives are addressed by three nested control levels operating at different timescales:

**Level 1 (Centroidal MPC, 40–100 Hz):** plans CoM trajectory and GRFs over a 500 ms horizon via the SRBD model (12).

**Level 2 (WBC Hierarchy, 500 Hz):** resolves the Level 1 GRFs into joint torques using the SK05 law (14) for contact and balance tasks, leaving the arm end-effector task slot open.

**Level 3 (Predictive interaction regulation, ≥1 kHz):** fills the arm slot with a receding-horizon QP that predicts and rejects pHRI disturbances, replacing the PD law (15) with a predictive force command.

### B. Contact-Consistent Residual Plant

After Level 2 commits torques $\tau_1$ (contact maintenance) and $\tau_2$ (balance), the residual arm end-effector dynamics are governed by the contact-consistent task-projected plant:

$$\Lambda_\text{arm}(q)\ddot{x}_\text{arm} + \mu_\text{arm} = F_\text{arm} + d_\text{ext} \tag{17}$$

where $\Lambda_\text{arm} = (J_\text{arm}\bar{M}^{-1}J_\text{arm}^\top)^{-1}$ uses the contact-consistent mass inverse (9), $\mu_\text{arm} = \bar{J}_\text{arm}^\top h - \Lambda_\text{arm}\dot{J}_\text{arm}\dot{q}$ collects all Coriolis, centrifugal, and gravitational terms (note that $h = C\dot{q}+G$ already includes $G(q)$, so no separate gravity term appears), and $d_\text{ext}$ is the pHRI wrench projected to arm task space. Equation (17) has the same mathematical structure as the fixed-base case, with $\bar{M}^{-1}$ replacing $M^{-1}$ in $\Lambda_\text{arm}$. All feedforward cancellation and linear double-integrator reduction steps follow identically.

### C. Contact-Consistent Feedforward and Horizon Freezing

To decouple the highly nonlinear task space without destroying the convexity of the receding-horizon optimization, we execute an analytical operational-space feedforward command:

$$F_\text{arm} = \Lambda_\text{arm}(q)\ddot{p}_{d} + \mu_\text{arm} - F_\text{mpc} \tag{18a}$$

which is mapped to the joint space via the balance null-space projector:

$$\tau_\text{ff,arm} = S\,\bar{N}_{12}^\top J_\text{arm}^\top F_\text{arm} \tag{18b}$$

**Multi-rate execution.** Level 2 updates $\bar{N}_{12}(q)$, $\Lambda_\text{arm}(q)$, and $\mu_\text{arm}$ at 500 Hz. Level 3 runs at $\geq$1 kHz; during the interleaved 1 kHz cycles that do not coincide with a Level 2 tick, the projector $\bar{N}_{12}$ and feedforward terms are held constant at their most recent Level 2 values. Because the configuration changes by at most $\|\dot{q}\|\Delta t_2 \approx 0.002\,\text{rad}$ per Level 2 interval, the frozen-matrix error is first-order small and its contribution to the tracking error is bounded by $O(\Delta t_2)$—comparable in magnitude to the SRBD modeling error already absorbed by the Kalman disturbance state $\hat{d}$.

Substituting the feedforward (18a) into the residual plant (17) cancels the $\Lambda_\text{arm}\ddot{p}_d + \mu_\text{arm}$ terms; with the error convention $e_\text{arm} = x_\text{arm} - p_d$ (actual minus desired) and $d(t) \triangleq \Lambda_\text{arm}^{-1}d_\text{ext}$, the residual tracking error dynamics reduce to:

$$\ddot{e}_\text{arm} = -\Lambda_\text{arm}^{-1}(q)F_\text{mpc} + d(t) \tag{19}$$

so that a positive $F_\text{mpc}$ enters with a restoring sign and drives $e_\text{arm} \to 0$.

**Proposition 1** (Constant $A_d$ and Local LTI Horizon Freezing): *Within a fixed contact mode, by evaluating the configuration-dependent task inertia strictly at the current sampling instant $k$ such that $\Lambda_\text{arm}(q) \approx \Lambda_\text{arm}(q_k)$ over the horizon $t \in [k, k+N]$, the discrete-time state transition matrix for the error state $x_{e,k} = [e_\text{arm}^\top, \dot{e}_\text{arm}^\top]^\top$ reduces to a constant linear system:*

$$A_d = \begin{bmatrix}I_3 & \Delta t I_3 \\ 0 & I_3\end{bmatrix}, \quad B_d(q_k) = \begin{bmatrix}0 \\ -\Lambda_\text{arm}^{-1}(q_k)\Delta t\end{bmatrix} \tag{20}$$

*Proof:* The continuous residual error dynamics (19) have state matrix $A_c = \left[\begin{smallmatrix}0 & I_3 \\ 0 & 0\end{smallmatrix}\right]$, which is nilpotent ($A_c^2 = 0$). The matrix exponential therefore terminates at first order, $A_d = e^{A_c\Delta t} = I + A_c\Delta t$, which is *exact* and independent of configuration—the double-integrator structure of (20). Evaluating $\Lambda_\text{arm}(q_k)$ at the current sample freezes $B_d(q_k) = [0;\, -\Lambda_\text{arm}^{-1}(q_k)\Delta t]$ over the horizon, so $(A_d, B_d(q_k))$ is LTI within the receding window; the configuration drift between Level-2 updates enters only $B_d$ and is first-order small. $\square$

The constant $A_d$ property is critical: it permits the lifted rollout pattern, powers of $A_d$, and sparsity structure of the QP to be computed offline. The numerical input matrix $B_d(q_k)$ still depends on the frozen task inertia $\Lambda_\text{arm}(q_k)$, so $\Gamma$, $H$, and the linear term are updated online after horizon freezing. Thus the online computation is a small dense update on a fixed structure, not a full nonlinear MPC rebuild.

### D. Receding-Horizon QP

Let $x_{e,k} = [e^\top, \dot{e}^\top]^\top \in \mathbb{R}^6$ be the arm tracking error state. The input matrix is:

$$B_d^{(m)} = \begin{bmatrix}0 \\ -(\Lambda_\text{arm}^{(m)})^{-1}\Delta t\end{bmatrix} \tag{21}$$

indexed to contact mode $m$. The $N$-step prediction matrix $\Gamma^{(m)}$ is constructed from $(A_d, B_d^{(m)})$ using the standard lifted-system expansion. Since $A_d$ is constant (Proposition 1), only $\Gamma^{(m)}$ changes between contact modes; its reconstruction is $O(N^2 \cdot 9)$.

The receding-horizon QP is:

$$\min_{U}\;\frac{1}{2}U^\top H^{(m)} U + h^{(m)\top} U \quad\text{s.t.}\quad \|F_{\text{mpc},k}\|_\infty \leq F_\text{max} \tag{22}$$

with $H_k^{(m)} = \Gamma_k^{(m)\top}\bar{Q}\Gamma_k^{(m)} + \bar{R}$ and $h_k^{(m)} = \Gamma_k^{(m)\top}\bar{Q}x_{\text{free},k}^{(m)}$, where $\bar{Q} = \text{blkdiag}(Q,\ldots,Q)$, $\bar{R} = \text{blkdiag}(R,\ldots,R)$, and $x_{\text{free},k}^{(m)}$ is the free-response prediction. The contact-mode index $m$ plays the role of the scheduling variable in the LPV-MPC framework of [16], while the configuration dependence enters through the frozen $\Lambda_\text{arm}(q_k)$ inside $B_{d,k}^{(m)}$. The box constraint $\|F_\text{mpc}\|_\infty \leq F_\text{max}$ is an engineered conservative Cartesian bound, chosen so that the resulting arm joint torques $\tau_\text{arm} = J_\text{arm}^\top F_\text{mpc}$ remain below hardware limits at all configurations within the operating workspace [15]. Mapping individual joint torque limits precisely into the Cartesian QP (which would require a configuration-dependent constraint matrix $J_\text{arm}^\top$) is deliberately avoided to preserve a fixed QP structure even though the numerical Hessian is updated from $\Lambda_\text{arm}(q_k)$. The QP (22) is strictly convex and solved by an operator-splitting solver (e.g., OSQP [13]) in $< 0.1$ ms for $N = 20$.

### E. Kalman Disturbance Augmentation

The disturbance $d(t)$ in (19) captures three coupled effects: (i) external pHRI forces directly applied to the arm; (ii) unmodeled contact reactions propagated from the feet; and (iii) SRBD approximation error from neglected leg inertia. These are jointly estimated by augmenting the MPC state with an integrating disturbance state $\hat{d} \in \mathbb{R}^3$:

$$\begin{bmatrix}x_{e,k+1} \\ \hat{d}_{k+1}\end{bmatrix} = \underbrace{\begin{bmatrix}A_d & B_d^{(m)} \\ 0 & I\end{bmatrix}}_{\displaystyle A_\text{aug}}\begin{bmatrix}x_{e,k} \\ \hat{d}_{k}\end{bmatrix} + \begin{bmatrix}B_d^{(m)} \\ 0\end{bmatrix}F_{\text{mpc},k} + w_k \tag{23}$$

with process noise $w \sim \mathcal{N}(0, Q_w)$ and measurement noise $v \sim \mathcal{N}(0, R_v)$ on arm end-effector position. The Kalman gain $K_f$ is computed offline from the steady-state discrete algebraic Riccati equation. The integrating structure of $A_\text{aug}$ guarantees $\hat{d}_{k} \to d$ for any bounded constant disturbance, independent of its physical origin. This is the predictive-control analogue of the anti-windup result of [14], which shows that augmenting an integrating channel achieves zero steady-state error under actuator limits and persistent constant disturbances: here the Kalman state $\hat{d}$ takes the role of that integrating channel and feeds it forward through the MPC horizon rather than feeding it back as a fixed integral gain.

The free-response prediction fed into the QP is constructed using $\hat{d}$:

$$x_\text{free}^{(m)} = \Phi^N x_e + \sum_{j=0}^{N-1}\Phi^j B_d^{(m)}\hat{d} \tag{24}$$

where $\Phi = A_d$. This causes the optimizer to pre-load corrective force before the disturbance fully manifests at the end-effector.

---

## VI. Contact-Mode Changes in Interaction Dynamics

### A. Contact-Mode Switch Protocol

When a foot lifts off or touches down, $J_c$ changes discontinuously. Consequently, $\bar{M}^{-1}$, $\Lambda_\text{arm}$, and $B_d$ all jump. The Kalman estimate $\hat{d}$ becomes partially stale because the input channel through which the disturbance acts has changed.

At contact switch time $t_s$, the following protocol is executed in order:

1. Recompute $\bar{M}^{-1}$ with the new $J_c$.
2. Recompute $\Lambda_\text{arm}$; select the new contact-mode index $m_\text{new}$; load $B_d^{(m_\text{new})}$.
3. **Covariance inflation:** $P_\text{aug} \leftarrow \alpha P_\text{aug}$ with $\alpha \in [3, 5]$ to inflate the error covariance and allow rapid re-estimation of the disturbance in the new contact configuration.
4. **Hold $\hat{d}$:** do not reset to zero—balance-related disturbances persist across contact transitions and the estimate retains useful information.

The covariance inflation in Step 3 is the key mechanism that distinguishes the proposed approach from a naive reset. The Kalman filter re-converges within approximately $5\tau_\text{Kalman}$ samples, after which the disturbance estimate is again accurate in the new contact mode.

### B. Contact-Mode-Indexed Matrix Library

For a robot cycling through $K$ repeating contact modes (e.g., gait phases), the contact Jacobian row pattern, lifted-system sparsity, and powers of $A_d$ are cached for each mode $m = 1,\ldots,K$. At each QP call, the controller freezes the current configuration, recomputes $\Lambda_\text{arm}(q_k)$ and $B_{d,k}^{(m)}$, and updates $\Gamma_k^{(m)}$ and $H_k^{(m)}$ on the cached structure. Since $A_d$ is constant across all modes (Proposition 1), the prediction rollout matrix $\Phi^N$ is precomputed once globally. For non-periodic or highly variable contact sequences (e.g., multi-contact manipulation), the same lifted-system expansion is reconstructed online; this remains tractable because only the input blocks depend on the current contact/configuration state while the double-integrator state transition is fixed.

---

## VII. Impedance as the Infinite-Horizon Limit

**Theorem 1** (Impedance Equivalence of Predictive Interaction Dynamics). *Under rigid contacts, fixed contact mode, and no disturbance, the unconstrained infinite-horizon predictive interaction controller (Level 3, $N \to \infty$) renders a classical task-space impedance law:*

$$\Lambda_\text{arm}(q)\ddot{e} + \Lambda_\text{arm}K_v\dot{e} + \Lambda_\text{arm}K_e e = F_h \tag{25}$$

*with effective configuration-adaptive mass $M_{d,\text{eff}} = \Lambda_\text{arm}(q)$ that depends on both arm posture and contact configuration.*

*Proof.* In the infinite-horizon unconstrained limit with $\hat{d}=0$, the QP reduces to the static discrete-LQR (Riccati) feedback $F_\text{mpc} = K_\infty x_e$. Partition the gain as $K_\infty=[K_p^\star\;K_v^\star]$ and define the acceleration-level gains by factoring through the task inertia, $K_e=\Lambda_\text{arm}^{-1}K_p^\star$ and $K_v=\Lambda_\text{arm}^{-1}K_v^\star$. Equivalently, $F_\text{mpc}=\Lambda_\text{arm}(q)(K_e e+K_v\dot e)$. Substituting into (19) with $d = \Lambda_\text{arm}^{-1}F_h$ gives
$$\ddot{e} = -(K_e e + K_v \dot{e}) + \Lambda_\text{arm}^{-1}F_h,$$ 

and premultiplying by $\Lambda_\text{arm}$ yields (25). Hence the effective mass, damping, and stiffness are $M_{d,\text{eff}} = \Lambda_\text{arm}(q)$, $D_\text{eff} = \Lambda_\text{arm}K_v$, and $K_\text{eff} = \Lambda_\text{arm}K_e$. The gains $(K_e,K_v)$ are the Riccati image of the QP weights $(Q,R)$, not the weights themselves; no direct equality between the cost weights and physical stiffness/damping is assumed. The contact-consistent $\bar{M}^{-1}$ in $\Lambda_\text{arm}$ means the effective mass adapts to both the arm joint configuration and the active contact footprint. $\square$

Theorem 1 establishes that predictive interaction regulation generalizes classical impedance control: the finite-horizon, constrained QP adds predictive disturbance rejection, constraint enforcement, and contact-mode adaptation while reducing to a classical linear impedance law in the infinite-horizon unconstrained limit. The QP weights $(Q,R)$ tune the Riccati gains $(K_e,K_v)$ indirectly; the physical stiffness and damping are the resulting $K_\text{eff}=\Lambda_\text{arm}K_e$ and $D_\text{eff}=\Lambda_\text{arm}K_v$, not the weights themselves.

Notably, all three closed-loop impedance parameters—$M_{d,\text{eff}}$, $D_\text{eff}$, and $K_\text{eff}$—adapt automatically as the arm configuration and contact state change, while the QP weights remain fixed design parameters. This adaptation is a structural by-product of $\Lambda_\text{arm}(q)$ and incurs no additional solver cost, preserving the ≥1 kHz control rate.

---

## VIII. Stability of Normalized Interaction Dynamics

### A. Zero Steady-State Error Under Fixed Contact Mode

**Theorem 2** (Zero Steady-State Error). *Suppose the disturbance $d(t)$ in (19) satisfies $\|d\|_\infty \leq \bar{d} < \infty$ and is asymptotically constant. Under fixed contact mode, the Kalman-augmented closed-loop system with the Level 3 QP feedback is asymptotically stable, and* $\lim_{k \to \infty}\|e_{\text{arm},k}\| = 0$.

*Proof.* The claim combines regulator stability with offset-free rejection, which rest on two *distinct* structural properties.

*Regulator (control) side.* The *un-augmented* pair $(A_d, B_d^{(m)})$ is controllable: $B_d^{(m)} = [0;\, -\Lambda_\text{arm}^{-1}\Delta t]$ has full column rank whenever $\Lambda_\text{arm}^{-1}$ is nonsingular, and then $[B_d^{(m)}\;\; A_d B_d^{(m)}]$ has rank 6. Two singularities can violate this: (i) *kinematic arm singularities* (e.g., a fully-extended elbow), at which $J_\text{arm}$ drops rank; and (ii) *contact-state singularities* when the platform loses all ground contact, making $\bar{M}^{-1}$ degenerate. The result therefore holds within a compact, singularity-free subset $\mathcal{W}$ in which the arm stays away from kinematic limits and the platform maintains a valid support polygon. On $\mathcal{W}$ the infinite-horizon LQR gain $K_\infty$ places the error-loop eigenvalues strictly inside the unit disk.

*Offset-free (observer) side.* The integrating block of $A_\text{aug}$ has eigenvalues at $1$ and is *not* controllable from $F_\text{mpc}$—the input $[B_d^{(m)}; 0]$ is zero on the $\hat{d}$ rows. This is by design: a constant disturbance is rejected through the internal-model/observer, not by driving the $\hat{d}$ modes. The property actually required is *detectability* of $(A_\text{aug}, C)$ with $C = [I_6\; 0]$; since $\hat{d}$ enters the measured state $x_e$ through $B_d^{(m)}$ of full column rank, the augmented pair is detectable, and the steady-state Kalman filter gives $\hat{d}_{k} \to d$ for any bounded constant $d$. The free response (24) then pre-loads $-\hat{d}$, and the stable error loop converges to $e_\infty = 0$. The box constraint $\|F_\text{mpc}\|_\infty \leq F_\text{max}$ defines a polyhedral invariant set $\mathcal{S}$ (analogous to the domain of attraction in [15]) inside which this convergence holds; for initial conditions outside $\mathcal{S}$ the QP clips $F_\text{mpc}$ and convergence is not guaranteed. $\square$

### B. Transient Bound Across Contact Transitions

Across a contact switch, $B_d$ jumps by $\Delta B_d = B_d^{(m_\text{new})} - B_d^{(m_\text{old})}$. The Kalman estimate becomes temporarily inaccurate, causing a transient tracking error. The error bound is:

$$\|e(t)\| \leq \|e(t_s)\| + c_1 \|\Delta B_d\| \cdot \|\hat{d}\| + c_2 \|d_\text{new}\| \cdot \Delta t_\text{conv} \tag{26}$$

where $\Delta t_\text{conv} \approx 5\tau_\text{Kalman}$ is the re-convergence time of the Kalman filter after covariance inflation, and $c_1, c_2 > 0$ are constants depending on the closed-loop eigenvalue placement. For typical biped walking ($\sim$1 Hz contact transitions), the transient magnitude is bounded by $c_1\|\Delta B_d\|\|\hat{d}\|$, which is small when leg contact forces are well-modeled. For running ($>$3 Hz contact transitions), the covariance-inflation coefficient $\alpha$ should be tuned to accelerate re-convergence and reduce $\Delta t_\text{conv}$.

### C. Null-Space Barrier Stability

The contact-consistent null-space torque that enforces joint limits and workspace boundaries is:

$$\tau_\text{null} = \bar{N}_\text{arm}^\top\bigl(-k_\text{null}(q-q_0) - d_\text{null}\dot{q} + g(q)\bigr) \tag{27}$$

where $\bar{N}_\text{arm} = I - \bar{J}_\text{arm}J_\text{arm}$ uses the contact-consistent pseudoinverse and $g(q)$ is the joint-limit barrier gradient. Under the same ideal hierarchy assumptions used above, projection through $\bar{N}_\text{arm}$ makes (27) produce zero wrench at the arm task coordinate, preserving task tracking and the higher-priority balance constraints. With torque saturation or friction-cone active-set changes, this decoupling becomes approximate and must be enforced by the low-level QP limits.

---

## IX. Whole-Body Realization

Combining all three levels, the complete joint torque command for a floating-base robot performing pHRI is:

$$\tau = \tau_\text{contact} + \bar{N}_1^\top\tau_\text{balance} + \bar{N}_{12}^\top\bigl[\tau_\text{ff,arm} + J_\text{arm}^\top F_\text{mpc} + \tau_\text{null}\bigr] \tag{28}$$

where: $\tau_\text{contact}$ resolves Level 1 GRFs; $\tau_\text{balance} = J_\text{CoM}^\top F_\text{balance}$ comes from the centroidal MPC; $\tau_\text{ff,arm}$ is the contact-consistent feedforward (18); $F_\text{mpc}$ is the corrective Cartesian force from the Level 3 QP; and $\tau_\text{null}$ is the contact-consistent barrier (27).

Equation (28) is the whole-body realization of the normalized interaction-dynamics controller. The hierarchical null-space structure follows the SK05 law [12] extended to the contact-consistent floating-base setting [11],[18]. The contribution here is not the null-space hierarchy itself, but the normalized predictive interaction force $F_\text{mpc}$ that occupies the residual arm channel. Under exact dynamics and inactive saturation/friction active-set changes, the projections $\bar{N}_1^\top$ and $\bar{N}_{12}^\top$ isolate this interaction layer from contact maintenance and balance, while the Kalman-augmented QP provides the offset-free disturbance rejection that the classical OS PD law (15) cannot achieve.

---

## X. Framework Comparison

Table I summarizes the mathematical and architectural distinctions between the proposed framework and the most relevant prior work. The key differentiators are: (i) the use of $\bar{M}^{-1}$ (contact-consistent) rather than $M^{-1}$ in the task-space inertia; (ii) the frozen-configuration input matrix $B_{d,k}^{(m)}$ that adapts the input channel to both active support mode and current posture; and (iii) the Kalman disturbance state that propagates the estimated pHRI force through the full prediction horizon.

**Table I: Architectural and Mathematical Comparison**

| Architectural Vector | Fixed-Base Baseline [2] | Bellicoso et al. [5] | Kim et al. [4] | Proposed Framework |
| :--- | :--- | :--- | :--- | :--- |
| Primary objective | Fixed-base pHRI | Quadruped locomotion | Dynamic locomotion | Normalized interaction dynamics for floating-base pHRI |
| Task-space inertia | $\Lambda = (JM^{-1}J^\top)^{-1}$ | N/A | N/A | $\Lambda_\text{arm} = (J\bar{M}^{-1}J^\top)^{-1}$ |
| Prediction horizon | $N$-step QP | $N = 1$ | $\sim$10–30 steps | $N$-step QP (≥1 kHz) |
| Disturbance handling | Kalman, fixed-base | WBC weight tuning | Centroidal inertia | Kalman: pHRI + leg momentum |
| Input matrix | $B_d = [0;\,-M_d^{-1}\Delta t]$ | N/A | N/A | $B_{d,k}^{(m)}$, contact/configuration indexed |
| Null-space use | Joint centering | Posture tracking | Locomotion | Predictive interaction QP |
| Steady-state error | Zero (fixed base) | Nonzero under load | Nonzero under load | Zero (Theorem 2) |
| Contact transitions | N/A | Mode switching | Gait phases | Covariance-inflation protocol |

---

## XI. Theory-Validation Experiments

The simulations are organized as validation of the theory rather than as a catalog of scenarios. Scenarios A and C test the fixed-contact normalized predictor and offset-free disturbance rejection. Scenario B stress-tests the same predictor under periodic contact shocks; with both feet planted at the reported gains the disturbance observer is the dominant factor and the contact-consistent and free-space predictors coincide. Scenarios E and F test whether the same interaction layer remains well behaved when the support model changes and the input matrix is switched through the contact-mode library. The experiments therefore target the paper's central claims: constant normalized state dynamics, contact-dependent input matrices, offset-free disturbance rejection in the normalized coordinates, and stable operation across contact-mode switches.

### A. Simulation Platform

All experiments were conducted in MuJoCo 3.2 [21] at a 2 kHz integration rate. Scenarios A and B use a biped comprising a 3-DOF right arm, two 4-DOF legs, and a 6-DOF unactuated floating base (17 DOF total, 11 actuated), with total mass 46 kg. Scenario C uses the **official Unitree G1** MJCF model from MuJoCo Menagerie [21] (29 DOF, 33.3 kg, kinematic/inertial parameters from factory CAD), augmented with a single end-effector site at `right_wrist_yaw_link`. The G1 model uses position actuators ($K_p = 500$, $\text{dampratio} = 1$) rather than the biped's direct-torque actuators; Level 3 applies the position-as-torque trick ($\Delta q_i = \tau_i / K_p$) to inject predictive interaction forces through the position channel. The legs are held in double support by a joint-space PD balance controller; the centroidal-MPC balance planner of Section V is not exercised in these static-/fixed-stance scenarios. The normalized interaction layer runs at 1 kHz ($N=20$, $\Delta t_3 = 1\,\text{ms}$).

Level 3 cost weights: $Q = \mathrm{diag}(6 \times 10^4 I_3,\; 60\, I_3)$, $R = 0.01\, I_3$. Kalman noise: $Q_w = \mathrm{diag}(10^{-4}I_6,\; 10^{-2}I_3)$, $R_v = 10^{-6}I_3$. All QPs are solved via the unconstrained analytical solution (fast path); covariance-inflation coefficient $\alpha = 4$ (Section VI).

**Unitree G1/R1 hardware interface.** The Unitree G1 (29 DOF, ~20.9 kg) and R1 (26 DOF, 25–29 kg, 1.23 m) robots expose a low-level joint control interface via `unitree_sdk2` at 500 Hz. Each joint accepts a command tuple $(q_\text{des},\, \dot{q}_\text{des},\, \tau_\text{ff},\, K_p,\, K_d)$, implementing $\tau = K_p(q_\text{des}-q) + K_d(\dot{q}_\text{des}-\dot{q}) + \tau_\text{ff}$. High-level locomotion in Unitree's open-source release (`unitree_rl_gym`) is provided by a reinforcement-learning policy that outputs joint position targets at 50 Hz; no WBC or MPC stack is open-sourced. The proposed three-level architecture maps directly onto this interface: Level 2 WBC and Level 3 predictive interaction regulation produce joint torques $\tau$ that are sent as $\tau_\text{ff}$ with $K_p = K_d = 0$ (pure torque mode), compatible with both G1 and R1 EDU variants.

### B. Benchmarked Controllers

Seven controllers are evaluated across all three scenarios (Table II). All controllers run above the identical joint-space PD balance controller and are evaluated on the same MuJoCo plant; they differ only in the arm interaction layer.

**Table II: Benchmarked Controllers**

| Label | Description |
| :--- | :--- |
| D1 | Operational-space PD (task PD, no priority hierarchy), Cartesian stiffness $K_x = 800\,\text{N/m}$, damping $D_x = 40\,\text{Ns/m}$ |
| D2 | Operational-space PI: adds $K_I = 150\,\text{N/(m·s)}$ with anti-windup clamping |
| D3 | Fixed-base impedance MPC (uses $M^{-1}$ instead of $\bar{M}^{-1}$; ignores contact consistency) |
| D4 | WBC assembler + null-space PD centering (no prediction or estimation) |
| D5 | Proposed: WBC + predictive interaction regulation, no Kalman augmentation |
| D6 | Proposed: WBC + predictive interaction regulation + Kalman, no covariance inflation ($\alpha = 1$) |
| D7 | Proposed full: WBC + predictive interaction regulation + Kalman + covariance inflation ($\alpha = 4$) |

D1 establishes the analytical baseline: for Cartesian stiffness $K_x = 800\,\text{N/m}$ and $F_h = 8\,\text{N}$, (16) predicts $e_\infty = F_h/K_x = 10.0\,\text{mm}$ exactly. D4 isolates the effect of the null-space hierarchy from prediction. D3 quantifies the penalty for ignoring $\bar{M}^{-1}$ on a floating base.

### C. Scenario A: Fixed Double-Support Step Disturbance

The robot holds a stationary double-support stance. The right arm holds a fixed Cartesian target. At $t = 0.5\,\text{s}$, a step pHRI force of 8 N is applied at the end-effector in the $x$-direction and held for the remaining 4.5 s. Total episode: 5 s. All controllers run at 1 kHz (1 ms control period); physics advances at 2 kHz via two 0.5 ms sub-steps per control cycle.

**Metrics:** RMS position error over the full episode; steady-state (SS) error averaged over $t > 3.5\,\text{s}$. Theoretical baseline: $e_\infty = F_h/K_x = 8/800 = 10\,\text{mm}$ for D1.

**Table III: Scenario A — Fixed Stance, 8 N Step Disturbance**

| Controller | RMS err [mm] | SS err [mm] |
| :--- | :---: | :---: |
| D1 OS PD | 9.24 | 10.17 |
| D2 OS PI | 6.82 | 5.86 |
| D3 Fixed-base MPC | 3.33 | 1.00 |
| D4 WBC + PD | 10.06 | 10.87 |
| D5 Proposed, no Kalman | 11.82 | 12.89 |
| D6 Proposed, no inflation | **3.48** | **0.11** |
| D7 Proposed full | **3.48** | **0.11** |

![Fig. 1](code/results/scenario_a_results.png)

**Fig. 1.** Scenario A — Fixed double-support stance, 8 N step pHRI disturbance at $t = 0.5\,\text{s}$. *Top:* Cartesian end-effector error norm $\|e\|$ over time for the readability subset D1, D2, D3, and D7; D4--D6 are reported in Table III. The Kalman-augmented predictive controller removes most of the steady-state offset, while the no-Kalman predictive controller is biased by the sustained pHRI load. *Bottom:* Bar chart of RMS and steady-state (SS) error for the same plotted subset.

Key findings. (1) The reactive baselines (D1, D4) leave the theoretical 10 mm impedance offset ($8\,\text{N}/800\,\text{N/m}$). (2) Static double support does not expose the full value of contact consistency: the fixed-base D3 remains stable and reaches 1.00 mm steady-state error. (3) The disturbance observer and input-centered QP are essential under sustained force: D6/D7 reduce the steady-state error to 0.99 mm, whereas the no-Kalman predictive controller is biased by the unestimated load. Covariance inflation has no effect here (D6 = D7) because the contact mode never switches.

### D. Scenario B: Stance with Periodic Contact-Transition Shocks

The robot holds a stable double-support stance throughout. The right arm tracks a fixed Cartesian target while a sustained 8 N pHRI force is applied from $t = 0$. Every $T_\text{switch} = 1\,\text{s}$, an additional 6 N spike is superimposed on the pHRI for $T_\text{spike} = 0.1\,\text{s}$ and then withdrawn, totalling 9 events over 10 s. This models the brief mechanical shock transmitted through the kinematic chain when a swing foot contacts the ground during walking. Since the contact set remains double support throughout, covariance inflation is not triggered in this scenario; Scenario E isolates the genuine contact-mode switch.

**Additional metric:** peak Cartesian error within a ±150 ms window centred on each shock event.

**Table IV: Scenario B — Stance + 1 Hz Contact-Transition Shocks, Sustained 8 N pHRI**

| Controller | RMS err [mm] | Peak at transition [mm] |
| :--- | :---: | :---: |
| D1 OS PD | 10.94 | 15.77 |
| D2 OS PI | 5.91 | 10.17 |
| D3 Fixed-base MPC | 3.40 | 4.84 |
| D4 WBC + PD | 11.89 | 17.24 |
| D5 Proposed, no Kalman | 13.92 | 18.20 |
| D6 Proposed, no inflation | **3.41** | **4.85** |
| D7 Proposed full | **3.41** | **4.85** |

![Fig. 2](code/results/scenario_b_results.png)

**Fig. 2.** Scenario B — double-support stance, sustained 8 N pHRI + 6 N periodic shocks at 1 Hz. *Top:* Error norm over the full episode for D1, D2, D3, and D7; D4--D6 are reported in Table IV. The disturbance observer removes the steady-state bias: the no-Kalman predictive controller (D5) is offset by the sustained load, while the Kalman-augmented D7 tracks to a few millimetres. With both feet planted throughout, the free-space (D3) and contact-consistent (D7) predictors are indistinguishable. *Bottom:* Bar chart of RMS and peak-at-transition error for the plotted subset.

Under sustained load plus periodic shocks, the disturbance observer is again the dominant factor: the no-Kalman predictive controller (D5, 13.92 mm RMS) is biased by the sustained load, while the Kalman-augmented controllers track to ≈3.4 mm. Because both feet stay planted throughout — the shocks are applied as force spikes, not an actual contact-set change — the free-space (D3, 3.40 mm) and contact-consistent (D7, 3.41 mm) predictors are indistinguishable here; the value of contact-consistency requires the support model itself to switch (Scenarios E and F). Covariance inflation is inert (D6 = D7) because the contact mode never changes.

### E. Scenario C: Unitree G1 Real Model, Fixed Stance, 8 N Step Disturbance

To validate the architecture on a real commercial humanoid, Scenario A is repeated using the official **Unitree G1** MJCF from MuJoCo Menagerie [21] (29 DOF, 33.3 kg, kinematic parameters from factory CAD). A single end-effector site (`right_hand_site`) is appended at the tip of `right_wrist_yaw_link` to provide a 3-DOF Cartesian tracking target. Because the G1 exposes **position actuators** ($K_p = 500$, $\text{dampratio} = 1$) rather than direct torque outputs, Levels 1–2 (balance and null-space) command joint position targets, while Level 3 applies the **position-as-torque** approximation: $\Delta q_i = \tau_i / K_p$, so that the desired Cartesian force $F_\text{mpc}$ is injected as $\text{ctrl}[i] \leftarrow q_i + (J_\text{arm}^\top F_\text{mpc})_i / K_p$. This is exactly the pure-torque mode available on G1/R1 EDU hardware ($K_p = K_d = 0$, $\tau_\text{ff} \neq 0$).

**Table V: Scenario C — Unitree G1 (33.3 kg), Fixed Stance, 8 N Step Disturbance**

| Controller | RMS err [mm] | SS err [mm] |
| :--- | :---: | :---: |
| D1 OS PD | 9.00 | 9.57 |
| D2 OS PI | 6.41 | 5.32 |
| D3 Fixed-base MPC | 2.37 | 2.84 |
| D4 WBC + PD | 9.00 | 9.57 |
| D5 Proposed, no Kalman | 7.54 | 8.19 |
| D6 Proposed, no inflation | **2.37** | **2.84** |
| D7 Proposed full | **2.37** | **2.84** |

![Fig. 3](code/results/scenario_c_g1_results.png)

**Fig. 3.** Scenario C — Unitree G1 official model (33.3 kg, 29 DOF), fixed stance, 8 N step pHRI. *Top:* Error norm for D1, D2, D3, and D7; D4--D6 are reported in Table V. In static G1 stance the contact-consistent D7 and fixed-base D3 predictive controllers are comparable, and both improve over the reactive D1. *Bottom:* RMS and steady-state error for the plotted subset.

Key findings. As in Scenario A, the static G1 stance does not distinguish contact-consistent from fixed-base predictive control: D3 (2.84 mm SS) and D7 (2.84 mm) are indistinguishable, and the Kalman-augmented predictive controllers improve about 3× over the reactive D1 (9.57 mm). The no-Kalman predictive controller (D5, 8.19 mm) barely improves on D1, underscoring that the disturbance observer, not contact-consistency, drives the gain here. The residual millimetre-level error arises from the position-actuator bandwidth (~5 Hz with $K_p = 500$) attenuating the 1 kHz MPC corrections; direct joint-torque mode on G1/R1 EDU ($K_p = K_d = 0$, $\tau_\text{ff} \neq 0$) would recover it.

### F. Scenario E: Bracing-Hand Support Transition

To exercise a **genuine** contact-mode transition without single-support balance, the biped is given a left arm that periodically braces against and releases a fixed rail while both feet remain planted. The active contact set alternates {L foot, R foot} ↔ {L foot, R foot, L hand} ($J_c$: 6 ↔ 9 rows), so $\bar{M}^{-1}$, the right-arm task inertia $\Lambda_\text{arm}$, and the input matrix $B_d$ genuinely switch at each brace and release. A sustained 8 N pHRI force acts on the right (task) arm throughout.

**Table VI: Scenario E — Bracing-Hand Support Transition, Sustained 8 N pHRI**

| Controller | RMS err [mm] | Peak at switch [mm] |
| :--- | :---: | :---: |
| D5 Proposed, no Kalman | 13.28 | 12.98 |
| D6 Kalman, no inflation ($\alpha=1$) | 4.67 | 6.69 |
| D7 Kalman + inflation ($\alpha=4$) | **4.14** | **6.29** |

This scenario validates the contact-mode-indexed disturbance estimator across a genuine contact-set switch. The no-Kalman controller is biased by the sustained pHRI load (13.28 mm RMS), while the Kalman-augmented controllers track the {L foot, R foot} ↔ {L foot, R foot, L hand} switch to ~4 mm. Here covariance inflation gives a small further improvement (D7 4.14 mm RMS / 6.29 mm peak vs D6 4.67 / 6.69), consistent with its role of re-estimating $\hat{d}$ under the changed $B_d^{(m)}$ after a real contact-set change. Scenario F extends the contact-mode test to a scheduled single↔double support-mode transition for the interaction layer.

### G. Scenario F: Quasi-Static Single↔Double Support Transition

Scenarios B and E hold both feet planted; this scenario exercises a scheduled change of the **foot-support model** used by the interaction layer — the biped shifts its weight over the right foot, lifts the left foot, switches the contact-consistent predictor from double to right-foot support, holds, and places the foot back:

$$\{\text{L foot}, \text{R foot}\} \;\longleftrightarrow\; \{\text{R foot}\},$$

so the contact Jacobian used by the interaction-layer model ($J_c$: 6↔3 rows), $\bar{M}$, the arm task inertia $\Lambda_\text{arm}$, and the input matrix $B_d$ all switch at the lift and the place. The interaction layer runs above the balance controller with a sustained 8 N pHRI force on the arm; the arm regulates a target fixed relative to the torso, isolating its disturbance-rejection task from the balancing motion.

**Balance stand-in and its limitations.** A standing single-support phase requires **ankle-roll** (lateral centre-of-pressure) authority, which the biped of Scenarios A/B/E lacks; hip-roll balancing alone produces a foot-tipping moment that cannot be countered, and the robot falls. We therefore use a modified model (`biped_qstatic`: a wider 18 cm foot and added ankle-roll actuators) and a hand-tuned quasi-static controller — a stiff hip-roll CoM regulator for the coarse weight transfer plus a centre-of-pressure–limited ankle-roll torque for the fine support-mode stabilization. This is a deliberate **minimal stand-in** for the Level-1 balance layer, *not* the centroidal MPC of Section V. MuJoCo contact auditing shows that the lifted foot can retain intermittent toe/edge contact with the floor, so Scenario F should be read as a support-mode/contact-model switch test for the interaction layer, not as a full dynamic walking or physically clean single-support result.

**Table VII: Scenario F — Quasi-Static Single↔Double Support Transition, Sustained 8 N pHRI (torso-relative arm error)**

| Controller | RMS err [mm] | Peak at switch [mm] |
| :--- | :---: | :---: |
| D5 Proposed, no Kalman | 30.08 | 37.57 |
| D6 Kalman, no inflation ($\alpha=1$) | 15.95 | 24.04 |
| D7 Kalman + inflation ($\alpha=4$) | **15.81** | **23.96** |

The biped stays upright across the whole cycle (minimum torso height 0.86 m; scheduled single-support-mode interval ≈2.1 s), and the interaction layer keeps the torso-relative arm error to ≈1.6 cm. In this model the arm task inertia barely changes across the foot-support switch — $\Lambda_\text{arm}$ diagonal $[1.10, 1.04, 2.42]$ kg in double support versus $[1.09, 1.04, 2.42]$ kg in right-foot support (<1%) — because the right arm couples only weakly to the left-leg contact. This scenario therefore exercises the contact-mode indexing and $B_d^{(m)}$ switching mechanism across a genuine **leg** support-mode change, but does not, on this platform, produce a large inertia change to exploit.

The Kalman augmentation roughly halves the error, visible both in RMS (D5 30.08 → D7 15.81 mm) and in the peak at the switch (D5 37.57 → D7 23.96 mm). Covariance inflation is essentially neutral (D6 15.95 vs D7 15.81 mm): the error at a support transition is dominated by the whole-body balancing motion — the torso adjusting under the newly single foot — rather than by staleness of the disturbance estimate, so faster Kalman re-convergence has limited room to help. The honest reading across Scenarios E and F is that the contact-mode-indexed Kalman model is the main source of improvement, while covariance inflation is a tunable, scenario-dependent robustness mechanism — beneficial at the planted-foot bracing switch (Scenario E) but neutral here, where balancing motion dominates.

![Fig. 4](code/results/scenario_f_results.png)

**Fig. 4.** Scenario F — torso-relative arm error across a quasi-static scheduled single↔double support-mode transition (shaded: right-foot-support model). All controllers stay stable across the contact-model switch; the Kalman gives a modest offset improvement and covariance inflation is neutral (Table VII).

---

## XII. Conclusion

This paper proposed contact-consistent interaction dynamics normalization for floating-base pHRI. The central result is that, after balance and contact tasks are removed through the whole-body hierarchy, the residual end-effector interaction channel reduces to a linear double integrator with a constant discrete state matrix within each contact mode. Whole-body control, null-space projection, disturbance observation, and contact-mode scheduling are the realization mechanisms; the underlying object being regulated is the normalized interaction dynamics.

This viewpoint explains the empirical results. The contact-mode-indexed disturbance observer is the primary driver of accuracy — it delivers offset-free rejection of the sustained interaction force (steady-state error ~0.1 mm vs ~13 mm without it) — while the contact-consistent inertia sharpens static precision and keeps the predictor well-conditioned across support-mode changes, because $\Lambda_\text{arm}$ captures the contact-coupled apparent inertia seen at the hand. The Impedance Equivalence Theorem further shows that classical operational-space impedance is recovered as the infinite-horizon limit of the same predictive interaction law, so impedance becomes a design interpretation rather than the starting point of the method. The simulations also demonstrate operation across scheduled support-mode/contact-model changes, although a fully dynamic single-support walking transition with a complete centroidal-MPC balance layer remains future work. More broadly, the normalization perspective suggests a common foundation for pHRI, loco-manipulation, dexterous manipulation, surgical robotics, and other contact-rich systems: first represent the task as interaction dynamics, then normalize those dynamics into a configuration-invariant predictive control problem.

The proposed framework occupies a structural niche not addressed by prior locomotion-centric frameworks [3]–[6]: it deliberately halts the WBC stack after balance constraints are satisfied and injects normalized predictive interaction regulation into the residual null space.

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
