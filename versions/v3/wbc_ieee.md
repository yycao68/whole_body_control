# Interaction Dynamics for Floating-Base Whole-Body Manipulation

**Yongyan Cao**

---

## Abstract

Floating-base loco-manipulation is usually organized as a stack of different control objects: centroidal force planning for balance, whole-body inverse dynamics for torque realization, and impedance or operational-space control for manipulation. This paper takes a different view. It shows that the body-environment interface and the hand-task interface can both be represented as interaction-dynamics ports of the same form previously established for fixed-base physical human-robot interaction [1]. Only these interaction dynamics are predicted over a horizon; the full robot dynamics are used instantaneously to realize the requested interactions. In both ports, model feedforward removes known dynamics and the remaining regulated error satisfies

$$
\ddot e = u+d,
$$

where $u$ is residual acceleration and $d$ is an estimated interaction disturbance. The normalization, exact zero-order-hold predictor, offset-free result, and impedance interpretation are inherited from [1]; the contribution here is to establish the floating-base realization. The body port recovers residual acceleration as a centroidal wrench through mass, centroidal inertia, and contact geometry, while the task port recovers residual acceleration as an end-effector wrench through contact-consistent task inertia. Thus the prediction model is shared across both ports — the same exact-ZOH double-integrator structure, differing only in dimension and sampling period — whereas contact mode, friction, center-of-pressure, actuator limits, and inverse dynamics are localized to recovery and instantaneous whole-body feasibility. The middle layer is therefore not an MPC: it is a whole-body interaction realizer, or projection onto the robot-feasible set, at the current sample. A coupled realization previews the centroidal reaction of planned arm actions so that balance compensation can be generated before the disturbance is observed. The paper specifies a Unitree G1 MuJoCo evaluation for dual-port offset rejection, cross-port coupling, support transitions, active constraints, and Kalman-based contact/event detection. Four hypotheses are evaluated on the standing torque-actuated G1: (H1) the normalized predictor is port-independent and configuration-invariant — the body and task ports share the identical constant exact-ZOH pair while the task apparent inertia varies by more than an order of magnitude across a kinematic sweep; (H2) disturbance estimation gives offset-free dual-port regulation both at the representation level ($\sim$90–210$\times$ steady-state error reduction) and on the full realizer, once the body port is recovered as a centroidal wrench rather than a posture tilt (CoM error 44.9$\to$2.7 mm, hand error 151$\to$31 mm under sustained loads, no falls); (H3) previewing the planned arm-reaction wrench (coupled realization) cuts the peak cross-port CoM transient by $2.9\times$ versus reactive rejection; and (H4) contact events are detected from the observer innovation with no misses or false positives at $\sim$56 ms latency. The remaining torque-level work is dynamic-gait recovery: a 10 s root-assisted visualization runs with both MPC command layers active and fixed-support torque trials pass no-push and randomized-push gates, while torque-level support switching and sustained walking still require the production gait/contact-wrench realizer.

**Index Terms** - interaction dynamics, centroidal MPC, whole-body control, floating-base robots, loco-manipulation, physical human-robot interaction, model predictive control.

---

## I. Introduction

Humanoid robots must regulate two physical interfaces at once. Their feet exchange forces with the environment to maintain balance and locomotion, while their hands exchange forces with people, tools, and objects to accomplish a task. Existing stacks usually assign these interfaces to different mathematical objects: a centroidal or single-rigid-body MPC plans contact forces, a whole-body QP maps those forces to joint commands, and an impedance-like controller regulates the hand. This decomposition is practical, but it hides the fact that both interfaces are manifestations of the same interaction-dynamics problem.

The question addressed here is not whether another whole-body control architecture can be assembled. It is whether floating-base manipulation admits the same normalized interaction-dynamics representation previously derived for fixed-base systems in [1]. If the answer is yes, then the balance controller and the manipulation controller need not be viewed as unrelated modules. They can be treated as two ports of one predictive representation, with robot-specific mechanics appearing only when the normalized commands are recovered as physical wrenches.

At the body port, the controlled interaction is the relation between centroidal motion and the net contact wrench. At the task port, it is the relation between end-effector motion and task wrench. In both cases, known dynamics and desired acceleration can be placed in feedforward, leaving a residual acceleration input:

$$
\text{physical wrench}
=\text{model feedforward}+\text{interaction inertia}\times u.
\tag{1}
$$

The resulting error model is the interaction-dynamics backbone established in [1]. The floating-base case is nontrivial because contact geometry, support changes, friction, center-of-pressure limits, actuator saturation, and arm-body reactions all affect whether a normalized acceleration can be realized. The central claim of this paper is therefore a prediction-realization separation principle: only interaction dynamics are predicted over a horizon, while robot dynamics are used at the current sample to project interaction requests onto the feasible whole-body dynamics. In this view, the full robot dynamics do not become a second prediction model; they define the instantaneous feasible set onto which normalized interaction commands are realized.

This leads naturally, but secondarily, to a dual predictive controller. The balance port and task port use the same exact-ZOH double-integrator predictor; the body port recovers residual acceleration as a centroidal wrench, and the task port recovers residual acceleration as a contact-consistent task wrench. A whole-body interaction realizer then enforces the rigid-body dynamics and instantaneous feasibility constraints. Thus the architecture follows from the representation rather than serving as the paper's main premise.

The contributions are organized around this separation. First, the paper formulates floating-base balance and manipulation as two interaction ports that instantiate the normalized model of [1]. Second, it derives the centroidal and contact-consistent task recoveries showing that mass, centroidal inertia, contact mode, and task inertia do not alter the prediction matrices. Third, it localizes contact geometry, friction, center-of-pressure, actuator limits, and whole-body dynamics to recovery and instantaneous feasibility, clarifying exactly where floating-base constraints enter. Fourth, it realizes the representation as a split or coupled dual-MPC controller and specifies a Unitree G1 evaluation for offset rejection, arm-body coupling, support switching, active constraints, and Kalman-based event detection.

The central separation is therefore between prediction and physical realization. Level 1 and Level 3 are predictors: they optimize future residual accelerations for the body and task interaction dynamics. Level 2 is a realizer: it has no future state, no prediction horizon, and no future cost. It solves only the current-sample feasibility problem that maps the two interaction requests into generalized torques. Table I summarizes the roles.

| Level | Role | Mathematical object | Time scale |
|---|---|---|---|
| Level 1: body interaction port | Predict body-environment interaction | Interaction-dynamics MPC | Future |
| Level 2: whole-body interaction realizer | Realize interaction physically | Instantaneous constrained inverse-dynamics QP | Present |
| Level 3: task interaction port | Predict hand-task interaction | Interaction-dynamics MPC | Future |

![Fig. 1. Interaction-dynamics ports for floating-base whole-body manipulation.](figures/interaction_dynamics_ports_architecture.png)

**Fig. 1.** Floating-base whole-body manipulation represented as two normalized interaction-dynamics ports. The shared prediction object supplies the same constant exact-ZOH predictor to the body and task MPCs through folded dashed connections. The two MPCs produce residual-acceleration commands, which are converted into centroidal and task wrenches before entering the whole-body interaction realizer. The realizer is not an MPC; it is an instantaneous inverse-dynamics QP that projects the interaction requests onto feasible generalized torque commands for the Unitree G1/MuJoCo plant. The measured state, contacts, and wrenches feed directly into the Kalman disturbance estimator. The green dashed path denotes the optional arm-reaction preview used by the coupled realization.

---

## II. Related Work and Positioning

Centroidal and single-rigid-body MPC methods [2], [3], [8], [13] predict center-of-mass and body orientation while optimizing contact forces over a gait schedule. Their strength is horizon-wide treatment of friction, unilateral contact, and support geometry. The body-port controller retains these physical constraints but changes the decision coordinates from raw contact forces to a normalized residual acceleration plus a contact-wrench recovery.

Whole-body inverse dynamics and hierarchical QPs [4], [5], [7], [9] enforce rigid contacts, task priorities, and actuator limits. They remain essential here. The whole-body layer is not replaced by another predictive model; it is the instantaneous interaction realizer that maps a desired body-interaction wrench and a desired task-interaction acceleration to feasible generalized forces.

Operational-space impedance, admittance, and task-space MPC [6], [11], [12] regulate the end-effector port. Their apparent inertia generally depends on configuration and support. Residual-acceleration coordinates remove this inertia from the prediction dynamics while retaining it in force recovery.

Closest in spirit is unified whole-body MPC for combined locomotion and manipulation [10], which optimizes a single predictive whole-body model. The present work differs in the prediction-realization split: only the two normalized interaction dynamics are predicted, while the full contact-constrained rigid-body dynamics act at the current sample as a feasibility projection rather than as a second predictive model. The normalized model, offset-free regulation, stability conditions, and impedance-limit interpretation belong to [1]; the standard centroidal model [8], [17], whole-body inverse dynamics [9], and covariance inflation are prior tools. This paper contributes their floating-base integration, anticipatory coupling, constraint realization, and empirical evaluation on a Unitree G1 in MuJoCo [15].

---

## III. Floating-Base Interaction Dynamics

Let

$$
q=[q_b^\top,q_j^\top]^\top,\qquad
M(q)\ddot q+h(q,\dot q)=S^\top\tau+J_c^\top\lambda+J_t^\top F_h,
\tag{2}
$$

where $q_b$ is the floating-base coordinate, $q_j$ contains actuated joints, $\lambda$ stacks contact wrenches, and $F_h$ is an external task wrench. Rigid active contacts satisfy

$$
J_c\ddot q+\dot J_c\dot q=0.
\tag{3}
$$

The controller uses two controlled ports. The body port is defined by CoM position and body-orientation errors, and the task port is defined by Cartesian end-effector tracking error.

The active contact mode $\rho$ changes the contact Jacobian and feasible wrench set. Following [1], it does not change the normalized prediction pair used below.

---

## IV. Body Interaction Port

### A. Centroidal Normalization

Let $c$ be the CoM, $m$ the robot mass, $f_i$ the force at active contact $i$, and $g$ the signed gravitational-acceleration vector (so $mg$ is the weight). With lumped disturbance $w_c$,

$$
m\ddot c=\sum_{i\in\mathcal C_\rho}f_i+mg+w_c.
\tag{4}
$$

For $e_c=c-c_d$, define the desired resultant

$$
F_c^{\rm des}=m(\ddot c_d-g)+m u_c.
\tag{5}
$$

When the recovered contact forces satisfy $\sum_i f_i=F_c^{\rm des}$,

$$
\ddot e_c=u_c+d_c,\qquad d_c=m^{-1}w_c.
\tag{6}
$$

The rotational channel is treated in the same residual-acceleration coordinates. Let $k_G=I_G(q)\omega_G$ be centroidal angular momentum expressed through the centroidal composite rigid-body inertia. Then

$$
I_G\dot\omega_G
=M_c-\dot I_G\omega_G+w_\theta,
\tag{7}
$$

where $M_c$ is the net contact moment about the CoM. Define

$$
M_c^{\rm des}
=\dot I_G\omega_G+I_G(\dot\omega_{G,d}+u_\theta).
\tag{8}
$$

For the local orientation error $e_\theta=\log(RR_d^\top)^\vee$, the approximation $\ddot e_\theta\simeq\dot\omega_G-\dot\omega_{G,d}$ gives

$$
\ddot e_\theta=u_\theta+d_\theta,\qquad
d_\theta=I_G^{-1}w_\theta.
\tag{9}
$$

Two approximations are used here and should be stated explicitly. First, $\omega_G$ defined by $k_G=I_G(q)\omega_G$ is the locked-inertia (CCRBI-averaged) angular velocity of the whole body; it is generally *not* the time derivative of any single orientation, so identifying $R$ with the centroidal frame and setting $\dot e_\theta\simeq\omega_G-\omega_{G,d}$ is an approximation, with the resulting kinematic mismatch absorbed into $d_\theta$. Second, the error map is linearized through the logarithm, so the channel is valid locally, away from the singularity of the logarithm map. Large-angle locomotion therefore requires a nonlinear attitude predictor or repeated local relinearization.

Stacking translation and rotation gives

$$
e_b=[e_c^\top,e_\theta^\top]^\top,\quad
x_b=[e_b^\top,\dot e_b^\top]^\top,\quad
u_b=[u_c^\top,u_\theta^\top]^\top.
$$

Applying the exact-ZOH construction of [1] at sampling period $T_b$ gives

$$
x_{b,k+1}=A_bx_{b,k}+B_b(u_{b,k}+d_{b,k}),
\tag{10}
$$

$$
A_b=
\begin{bmatrix}I_6&T_bI_6\\0&I_6\end{bmatrix},
\qquad
B_b=
\begin{bmatrix}\frac12T_b^2I_6\\T_bI_6\end{bmatrix}.
\tag{11}
$$

Both matrices are constant. Mass, centroidal inertia, contact locations, and contact mode appear only in wrench recovery and constraints.

**Lemma 1 (floating-base body port).** Assume the centroidal orientation error remains inside the local logarithmic chart and that the desired centroidal wrench can be recovered, possibly with logged slack, from the active contacts. Then the floating-base body port satisfies the normalized interaction-dynamics model (10) with the constant exact-ZOH pair (11). The active contact mode, contact locations, mass, centroidal inertia, friction limits, and center-of-pressure constraints affect only the recovery equations and the feasible input set; they do not change $(A_b,B_b)$.

**Proof.** The translational channel follows by substituting the recovered resultant force (5) into the centroidal balance equation (4), which gives (6). The rotational channel follows similarly from (7) and (8), giving the local residual-acceleration relation (9). Stacking the translational and rotational residual accelerations gives the continuous-time double integrator. Applying the exact-ZOH construction from [1] yields (10)-(11). The quantities $m,I_G,p_i,\rho$ are used to convert $u_b$ into a centroidal wrench and contact forces through (12)-(15), but the normalized state transition is expressed directly in residual-acceleration coordinates. $\square$

### B. Contact-Wrench Recovery and MPC

For stacked contact forces $f=[f_1^\top,\ldots,f_{n_c}^\top]^\top$, define

$$
\mathcal G_\rho(c,p_i)f=
\begin{bmatrix}
\sum_i f_i\\
\sum_i(p_i-c)\times f_i
\end{bmatrix}.
\tag{12}
$$

The desired centroidal wrench is

$$
W_b^{\rm des}(u_b)=
\begin{bmatrix}
m(\ddot c_d-g)+mu_c\\
\dot I_G\omega_G+I_G(\dot\omega_{G,d}+u_\theta)
\end{bmatrix}.
\tag{13}
$$

Recovery enforces

$$
\mathcal G_\rho f=W_b^{\rm des}(u_b)+s_W,
\tag{14}
$$

where $s_W$ is a penalized wrench slack used only when exact realization is infeasible. When $s_W\neq0$ the recovered resultant no longer equals $F_c^{\rm des}$, so the normalized relation (6) holds only up to a realization residual $m^{-1}s_W^{\rm (force)}$ (and the rotational analogue); this residual is collected into $d_b$ exactly as $d_{\rm rec}$ is collected into the task disturbance $d_t$ in (19), and is estimated by the observer. Equations (6) and (10) are therefore exact when recovery is exact and hold with an estimated residual otherwise. Each active contact additionally satisfies unilateral-force, friction-pyramid/cone, and center-of-pressure constraints. In a scheduled gait, future $\mathcal G_{\rho_j}$ and contact bounds are indexed by the planned mode sequence while $(A_b,B_b)$ remain unchanged.

Joint-torque feasibility cannot be inferred exactly from the reduced centroidal model alone. The body-port recovery therefore uses either a conservative frozen affine surrogate

$$
\tau_j^{\rm pred}=\tau_{\rm ff,k}+\mathcal T_{\rho_j,k}f_j
\tag{14b}
$$

or delegates the torque-limit check to the whole-body interaction realizer. The benchmark reports which option is active and logs disagreement between predicted and realized torque.

The body-port optimization uses this recovery map inside the horizon:

$$
\begin{aligned}
\min_{U_b,F,S_W}\quad&
\sum_{j=0}^{N_b-1}
\left(
\|x_{b,j}\|_{Q_b}^2+
\|u_{b,j}+\hat d_{b,k}\|_{R_b}^2+
\|s_{W,j}\|_{R_W}^2
\right)
+\|x_{b,N_b}\|_{P_b}^2\\
\text{s.t.}\quad&
x_{b,j+1}=A_bx_{b,j}+B_b(u_{b,j}+\hat d_{b,k}),\\
&\mathcal G_{\rho_j}f_j=W_b^{\rm des}(u_{b,j})+s_{W,j},\\
&f_j\in\mathcal F_{\rho_j},\\
&\tau_{\min}\le\tau_j^{\rm pred}\le\tau_{\max}
\quad\text{when (14b) is enabled}.
\end{aligned}
\tag{15}
$$

The input-centered penalty is essential: for a constant estimated disturbance, the cancelling equilibrium is $u_b=-\hat d_b$. Penalizing raw $u_b$ would generally reintroduce a steady-state offset.

---

## V. Task Interaction Port

The task port is not derived from a fixed-base arm model. In (2), $M(q)$ is the full floating-base mass matrix, including the unactuated base and all actuated joints, and $J_{c,\rho}$ is the active whole-body contact Jacobian. Using the contact-consistent inverse associated with mode $\rho$,

$$
\bar M_\rho^{-1}
=M^{-1}-M^{-1}J_{c,\rho}^\top
(J_{c,\rho}M^{-1}J_{c,\rho}^\top)^{-1}
J_{c,\rho}M^{-1},
\tag{16}
$$

the task inertia is

$$
\Lambda_t=(J_t\bar M_\rho^{-1}J_t^\top)^{-1}.
\tag{17}
$$

Thus $\Lambda_t$ already contains the floating-base, stance-contact, and arm-body inertial coupling induced by the full constrained system. The construction does not claim that the arm is dynamically isolated from the base; it only expresses the realized task acceleration in residual-acceleration coordinates after the current contact constraints have been imposed. Base reactions that are predictable from the planned task wrench are handled by the coupled body-port preview in Section VII, while unmodeled coupling and recovery error are collected in $d_t$ and in the realization residuals reported by (22).

For $e_t=x_t-x_{t,d}$, choose

$$
F_t=F_{t,\rm ff}+\Lambda_tu_t,\qquad
F_{t,\rm ff}=\Lambda_t\ddot x_{t,d}+\mu_t.
\tag{18}
$$

The normalized task error is

$$
\ddot e_t=u_t+d_t,\qquad
d_t=\Lambda_t^{-1}F_h+d_{\rm model}+d_{\rm rec}.
\tag{19}
$$

Applying the same construction [1] at period $T_t$ gives

$$
x_{t,k+1}=A_tx_{t,k}+B_t(u_{t,k}+d_{t,k}),
\tag{20}
$$

with the same matrix form as (11), now using dimension three.

**Lemma 2 (contact-consistent task port).** For a fixed active contact mode with $J_{c,\rho}M^{-1}J_{c,\rho}^\top$ nonsingular and $J_t\bar M_\rho^{-1}J_t^\top$ nonsingular on the operating set, the end-effector port satisfies the normalized model (20) with a constant exact-ZOH pair. Configuration and contact mode enter through the full-system apparent inertia $\Lambda_t$, feedforward compensation, wrench bounds, and torque recovery, but not through the prediction matrices.

**Proof.** The constrained inverse (16) is formed from the full floating-base mass matrix and restricts admissible accelerations to directions compatible with the active rigid-contact constraint. In the task coordinates, the resulting contact-consistent apparent inertia is $\Lambda_t$ in (17). Substituting the feedforward-plus-residual wrench (18) into the task dynamics cancels the nominal acceleration and leaves (19), with model mismatch, base-reaction mismatch, realization residual $d_{\rm rec}$, and external wrench collected in $d_t$. The exact-ZOH predictor is therefore the same normalized double integrator used in [1], with dimension three. $\square$

The task MPC uses an input-centered cost and constrains the recovered task wrench. When enabled, the predicted torque row is a conservative surrogate only; final torque feasibility is still imposed by the whole-body interaction realizer in (22). Here $\tau_{{\rm base},j}$ is a frozen bias torque (gravity/Coriolis compensation plus the higher-priority balance contribution) evaluated at the current state and held over the solve, so that $\tau_{{\rm base},j}+J_t^\top F_{t,j}$ is an affine-in-$u_t$ estimate of the arm joint torque:

$$
\|F_{t,\rm ff}+\Lambda_tu_{t,j}\|_\infty\le F_{\max},
\qquad
\tau_{\min}\le\tau_{\rm base,j}+J_t^\top F_{t,j}\le\tau_{\max}.
\tag{21}
$$

At runtime, $\Lambda_t$, $\mu_t$, and the torque-recovery rows are recomputed from the current measured state and active contact mode, then held fixed only inside the short prediction solve. The constant object reused across samples is the normalized exact-ZOH rollout structure $(A_t,B_t)$, not a frozen full robot model or a globally precomputed robot-dynamics Hessian. Fast configuration changes, near-singular task Jacobians, and force-box approximations can still create recovery mismatch; these effects are treated as disturbance and realization residuals rather than as a certified robust-stability guarantee.

---

## VI. Whole-Body Interaction Realizer

The two MPCs output a desired body interaction request and a desired task interaction request. These enter the realizer in their natural coordinates: the body request as a centroidal wrench $W_b^{\rm des}$ realized by the contact forces, and the task request as the desired end-effector acceleration $\ddot x_{t,d}+u_t^\star$ realized by the joint torques (equivalently the task wrench $F_t$, since $F_t$ and $\ddot x_{t,d}+u_t$ are related one-to-one through $\Lambda_t$). This asymmetry is deliberate: the body wrench is what the unilateral, friction, and CoP constraints act on, whereas the task objective is most directly imposed as an acceleration. This middle layer does not predict future robot states and is therefore not an MPC. Its role is physical realization: at the current sample, it projects the two interaction requests onto the set of generalized accelerations, contact wrenches, and joint torques that satisfy the floating-base dynamics and constraints. It is therefore better understood as a projection operator from interaction space to the robot-feasible torque set, not as another controller that plans robot motion over time.

Let $S_j$ select the actuated joint coordinates from $q$, and let $\tau_{\rm ref}$ be a nominal torque used only for regularization, such as the previous command or a gravity-compensating inverse-dynamics torque. With polyhedral friction pyramids, the whole-body interaction realizer is the convex instantaneous inverse-dynamics QP

$$
\begin{aligned}
\min_{\ddot q,\tau,\lambda,s_b,s_t}\quad&
\|s_b\|_{W_b}^2
+\|s_t\|_{W_t}^2
+\|\tau-\tau_{\rm ref}\|_{W_\tau}^2\\
\text{s.t.}\quad&
M\ddot q+h=S^\top\tau+J_c^\top\lambda+J_t^\top F_h,\\
&J_c\ddot q+\dot J_c\dot q=0,\\
&\mathcal G_\rho\lambda+s_b=W_b^{\rm des},\\
&J_t\ddot q+\dot J_t\dot q+s_t=\ddot x_{t,d}+u_t^\star,\\
&\lambda\in\mathcal F_\rho,\quad
\tau_{\min}\le\tau\le\tau_{\max},\\
&q_{j,\min}+\epsilon\le
S_j(q+\Delta t\dot q+\tfrac12\Delta t^2\ddot q)
\le q_{j,\max}-\epsilon.
\end{aligned}
\tag{22}
$$

The external task wrench $F_h$ appearing in the dynamics row is the value measured at the current sample (from a force/torque sensor, or the observer estimate $\Lambda_t\hat d_t$ when no sensor is available); it is a known constant within the instantaneous QP, not a decision variable. The variables are the generalized acceleration $\ddot q$, joint torque $\tau$, active contact wrench vector $\lambda$, body-wrench realization residual $s_b$, and task-acceleration realization residual $s_t$. The first two equality constraints enforce rigid-body dynamics and active-contact consistency at the present sample. The next two constraints define the projection residuals: $s_b$ measures the part of the body-port wrench request that cannot be realized by the active contact wrench, and $s_t$ measures the part of the task-port acceleration request that cannot be realized simultaneously with the other constraints. The final joint-limit row uses a one-step constant-acceleration feasibility check; without the $\frac12\Delta t^2\ddot q$ term, the row would not depend on the QP decision variable and would only check the current state.

With a friction-pyramid approximation, $\mathcal F_\rho$ is polyhedral and (22) is a standard convex QP solvable by an operator-splitting method such as [14]. If exact Coulomb friction cones are used instead, the same realization layer becomes a second-order cone program. Balance feasibility can be made hard by fixing $s_b=0$ or soft by assigning a large $W_b$; task tracking is usually softened through $s_t$ when the two interaction requests conflict. The realizer reports $s_b$, $s_t$, and active constraint margins back to the disturbance estimators and upper layers. These residuals are evidence that the requested interaction acceleration was not physically realizable at the current sample, rather than prediction errors from a second whole-body MPC.

---

## VII. Split and Coupled Prediction Realizations

The split controller solves (15) and the task MPC independently. The reaction of the task wrench on the base appears in $d_b$ and is rejected after being observed. This design is modular and provides the baseline dual-MPC realization.

The coupled controller uses the fact that the planned task wrench produces a known centroidal reaction,

$$
W_{b\leftarrow t}
=-\begin{bmatrix}
F_t\\(x_t-c)\times F_t+M_t
\end{bmatrix}.
\tag{23}
$$

This reaction is included in the body wrench recovery over the horizon. Because $F_t$ is affine in $u_t$, the coupling changes the lifted input map or linear constraint rows but not the double-integrator state matrix. This design is intended to reduce CoM and attitude transients during fast arm motion or contact, because the body controller compensates a predictable disturbance before observer feedback is required.

A single stacked QP is possible when both horizons share a grid. A computationally simpler alternative retains two QPs and passes the planned arm wrench sequence to the body port. The evaluation therefore compares the split controller, the coupled controller with arm-reaction preview, and, if timing permits, a monolithic stacked QP. No equivalence between weighted and strict lexicographic priority is assumed. Hard balance constraints, explicit task slacks, and logged recovery residuals define the actual priority.

---

## VIII. Kalman Estimation and Contact Events

Following the integrating-disturbance principle for zero steady-state error under persistent disturbances [16], and its instantiation in [1], each normalized model is augmented by a constant disturbance state:

$$
\begin{bmatrix}x_{k+1}\\d_{k+1}\end{bmatrix}
=
\begin{bmatrix}A&B\\0&I\end{bmatrix}
\begin{bmatrix}x_k\\d_k\end{bmatrix}
+
\begin{bmatrix}B\\0\end{bmatrix}u_k
+
\begin{bmatrix}0\\I\end{bmatrix}w_k.
\tag{24}
$$

Here $w_k$ drives the disturbance random walk; process noise acting directly on $x_k$ would be modeled by a separate state-noise term. The body observer estimates a six-dimensional residual wrench acceleration; the task observer estimates a three-dimensional residual acceleration. Offset-free regulation requires detectability, estimator convergence for a constant disturbance, and feasibility of the cancelling input.

The feedforward and recovery terms require generalized velocity. On hardware, $\dot q$ should not be obtained by raw encoder differencing inside the high-rate loop. The intended implementation uses filtered velocity estimates, for example a low-pass or observer-based differentiator feeding the rigid-body model, with the remaining phase lag and noise absorbed by the disturbance state and by the realization residuals. This filtering is an implementation requirement, not a theoretical replacement for torque-level validation.

A contact event creates an innovation because the assumed recovery set no longer matches the plant. This motivates a detector based on normalized innovation statistics:

$$
\eta_k=\nu_k^\top S_k^{-1}\nu_k.
\tag{25}
$$

A mode change is declared only after $\eta_k$ exceeds a calibrated threshold for $n_d$ consecutive samples and a candidate contact is geometrically plausible. This is safer than claiming that the aggregate disturbance alone uniquely identifies a particular foot: without additional kinematic information, different external wrenches can be indistinguishable at the centroidal port.

---

## IX. Relation to the Fixed-Base Theory

The normalized predictor, nominal offset-free regulation, stability conditions, and impedance interpretation are taken directly from [1]. They are not restated as propositions or re-proved here. The centroidal and task equations in Sections IV-V establish only that the two ports satisfy the assumptions and coordinate form required to invoke those results.

Several conditions are specific to this floating-base realization and must be checked independently. The centroidal orientation error is assumed to remain inside the local logarithmic coordinate chart, and $I_G$ and $\Lambda_t$ are assumed finite and positive definite on the operating set. The cancelling residual accelerations must be realizable by contact wrenches and joint torques within the physical constraints. Any recovery residual from the whole-body interaction realizer is treated as part of the disturbance model, and estimator convergence is considered only for contact modes that are correctly modeled or correctly detected. Thus citation of [1] does not by itself prove recursive feasibility of the contact-constrained G1 controller.

Constant $(A,B)$ removes model switching from the normalized state dynamics, but the feasible input set still switches with contact mode. Stability under arbitrary switching does not follow automatically. A certified switching claim would require either recursive feasibility for the scheduled mode sequence with a terminal set and terminal cost, or a common Lyapunov certificate for the actual constrained feedback regions. Until such a certificate is completed, the paper reports bounded empirical switching performance rather than certified arbitrary-switching stability.

---

## X. Unitree G1 Evaluation

The evaluation uses a Menagerie-derived Unitree G1 model as the common plant. The comparisons are organized around hypotheses rather than around the software layers themselves. This is important because the paper's claim is representation-level: both interaction ports should share the same predictor, while the floating-base mechanics appear in recovery. H1–H4 are reported below and all hold on the standing torque-actuated G1 (H2 offset-free regulation is demonstrated on the full realizer, not only at the representation level); only H5 and the sustained-walking form of the benchmark depend on the dynamic-gait realizer and remain planned.

| Hypothesis | Comparison | Primary Evidence | Status |
|---|---|---|---|
| H1: normalized prediction is port-independent | conventional centroidal MPC vs. centroidal interaction MPC | same $(A,B)$, command equivalence, $\Lambda_t$ variation | evaluated (Table II) |
| H2: disturbance estimation gives offset-free dual-port regulation | no observer vs. body/task Kalman observers | steady-state hand and CoM error under persistent force | evaluated (Table III): offset-free at the representation level and on the standing G1 realizer |
| H3: arm-reaction preview reduces cross-port transients | split vs. coupled prediction | CoM and attitude peaks during fast reaching | evaluated (Table V) |
| H4: contact events can be detected without an oracle | detected event vs. scripted-oracle event | latency, missed events, false positives | evaluated (Table VI) |
| H5: constraints belong to recovery | constrained vs. unconstrained recovery | friction, CoP, torque violations and slack use | planned |

The controller set is C0 joint-PD/operational-space PD, C1 conventional force-input centroidal MPC, C2 dual interaction MPC without observers, C3 split dual interaction MPC with body/task observers, C4 coupled dual interaction MPC with arm-reaction preview, and C5 the oracle-contact upper bound. Randomized studies use fixed seeds and paired disturbances; failed and fallen trials remain in the success denominator. A claim of dynamic walking interaction is reserved for the torque-actuated inverse-dynamics benchmark with randomized pushes and Kalman/event detection active.

### Results for H1 (Port-Independent Prediction)

H1 is a representation-level claim and is validated directly. First, the body port (dimension two: CoM and planar attitude) and the task port (dimension three) instantiate the *same* constant exact-ZOH double integrator: measured against the closed-form pair (11), $\max\|A-A_\text{ZOH}\|=\max\|B-B_\text{ZOH}\|=0$ for both ports; only the dimension differs. Second, the normalized centroidal MPC (decision = residual acceleration) and a conventional force-input centroidal MPC (decision = CoM force, matched weights $R_f=R/m^2$) are the same feedback map: over 2000 random states and disturbances the maximum command difference is $7.5\times10^{-6}$, i.e., the reparametrization is lossless. Third, over a 36-point right-arm kinematic sweep the contact-consistent task inertia $\Lambda_t$ (17) varies from $0.33$ to $13.1$ kg on its diagonal — up to $3840\%$ — while the predictor $(A_t,B_t)$ stays exactly constant. All configuration and contact dependence therefore lives in recovery, not in prediction, which is the H1 claim (Fig. 2, left).

| H1 evidence | Quantity | Result |
|---|---|---|
| Shared predictor across ports | $\max\|A-A_\text{ZOH}\|$, $\max\|B-B_\text{ZOH}\|$ (body & task) | $0.0$ |
| Normalized $\equiv$ force-input centroidal MPC | $\max\|u_\text{norm}-u_\text{force}\|$, 2000 states | $7.5\times10^{-6}$ |
| Configuration confined to recovery | $\mathrm{diag}(\Lambda_t)$ range over arm sweep | $0.33$–$13.1$ kg (up to $3840\%$), $(A_t,B_t)$ constant |

**Table II.** H1 evidence: the normalized predictor is port-independent and configuration-invariant; all robot-specific mechanics appear only in the recovery inertia $\Lambda_t$.

### Results for H2 (Offset-Free Dual-Port Regulation)

H2 is tested in two settings. First, at the *representation level*, where the recovery is faithful so that $\ddot e=u+d$ holds exactly by construction (using the G1 mass for the body port and the contact-consistent $\Lambda_t$ for the task port), a sustained disturbance is applied and the steady-state error is compared with the disturbance observer disabled and enabled. The observer estimate converges to the true disturbance and the offset is removed. Second, the same test is run on the *full torque-actuated G1 realizer* of (22): the body-port residual acceleration is realized by driving the whole-body CoM linear acceleration to $\ddot c_d+u_c$ (equivalently, allocating the centroidal wrench $m(\ddot c_d-g)+m u_c$ across the contacts through the inverse-dynamics QP), and the task-port residual acceleration is realized as the end-effector acceleration $\ddot x_{t,d}+u_t$. With this faithful recovery in place, $\ddot e=u+d$ holds on the robot and the observer becomes consistent with the plant, so offset-free regulation carries over to the full realizer for both ports:

| Port | Setting | Sustained disturbance | No observer | With observer | Reduction |
|---|---|---|---:|---:|---:|
| Body (CoM) | representation | 12 N pelvis force | 125.0 mm | 1.4 mm | 90$\times$ |
| Task (hand) | representation | 8 N hand force | 18748 mm | 89 mm | 211$\times$ |
| Body (CoM) | full G1 realizer | 12 N pelvis force | 44.9 mm | 2.7 mm | 17$\times$ |
| Task (hand) | full G1 realizer | 8 N hand force | 151.3 mm | 30.9 mm | 4.9$\times$ |

**Table III.** H2, representation level and full torque-actuated G1 realizer. The large no-observer task figures reflect the small task apparent inertia ($\Lambda_t\!\approx\!0.4$ kg laterally), which turns an 8 N force into a large normalized acceleration disturbance; the offset-free observer removes it in both settings, and neither port falls on the robot. The smaller reduction ratios on the robot reflect that the whole-body QP already holds a partial baseline offset without the observer, not a weaker cancellation.

An earlier realizer that mapped the body residual acceleration to an approximate posture *tilt* rather than to a centroidal wrench did **not** achieve this: the observer then degraded the CoM error (48 mm without observer, 137 mm with it) and the task port fell, because $\ddot e=u+d$ was not faithfully realized and the offset-free observer was inconsistent with the plant. Replacing the posture-tilt heuristic with the CoM-acceleration/centroidal-wrench recovery above is precisely what makes H2 hold on the robot; it confirms the paper's thesis that offset-free regulation is a property of the normalized representation that transfers to the robot exactly when recovery is faithful. The remaining production step is dynamic-gait recovery (Section X), where the contact set switches during walking.

![Fig. 2a. H1 configuration invariance and command equivalence.](code/results/h1_equivalence.png)

![Fig. 2b. H2 offset-free regulation on the full G1 realizer.](code/results/h2_offset_free.png)

**Fig. 2.** (a) H1: the task apparent inertia $\mathrm{diag}(\Lambda_t)$ over the arm sweep (left) with the constant $(A,B)$ and the command-equivalence residual annotated (right). (b) H2 on the full torque-actuated G1 realizer: CoM error (left) and hand error (right) under a sustained force with the observer disabled vs enabled — the observer removes the steady-state offset and neither port falls.

### Initial 10 s G1 Walking Visualization

The current implementation includes a dual-MPC root-assisted walking visualization on the local position-actuated Unitree G1 model. This artifact follows the `g1_ab_simulation` scaffold for visual stability, but the body reference is generated by the normalized centroidal interaction MPC and the right-hand motion is regulated by a normalized task interaction MPC. The floating base is still kinematically assisted, the G1 executes alternating one-foot swing commands, and the stance foot is lightly pinned. This is not the final torque-actuated whole-body interaction realizer and should not be interpreted as torque-level dynamic walking validation. Its purpose is to verify the G1 model, rendering pipeline, MPC command-layer integration, gait-command interface, and visible foot-lift behavior before the full S4 benchmark is run.

For a trapezoidal walking command that ramps from rest to $1.2\,{\rm m/s}$ during $0$--$1$ s, cruises at $1.2\,{\rm m/s}$ during $1$--$9$ s, and decelerates to rest during $9$--$10$ s, the robot traveled 10.800 m in 10 s with visible one-foot swing phases. Because the base is kinematically assisted, the 10.800 m forward distance matches the commanded 10.8 m by construction and is a consistency check on the gait-command interface rather than an achieved tracking result; the meaningful quantities in Table IV are the foot-lift, CoM-height, and torso-attitude ranges under active MPC command layers. The left and right feet lift by 8.3 cm and 8.3 cm, respectively. The CoM height remained above 0.752 m, and the largest absolute roll/pitch angle was 0.030 rad. Table IV summarizes the deterministic run, and Fig. 3 shows the CoM tracking, foot lift, torso attitude, and scheduled support sequence.

| Metric | Value |
|---|---:|
| Duration | 10.0 s |
| Speed profile | 0--1 s ramp, 1--9 s at 1.2 m/s, 9--10 s stop |
| Commanded distance | 10.8 m |
| External push | none |
| Body/task MPC enabled | true / true |
| Root assist | true |
| Forward distance | 10.800 m |
| Left/right foot lift | 0.083 / 0.083 m |
| Minimum CoM height | 0.752 m |
| Maximum roll/pitch magnitude | 0.030 rad |
| Hand RMS error | 95.5 mm |
| Support switches | 15 |
| Fall | false |

**Table IV.** Deterministic dual-MPC root-assisted walking visualization on the position-actuated Unitree G1 MuJoCo model. The artifact verifies the G1 model, video pipeline, MPC command-layer integration, gait-command interface, and visible foot-lift behavior. The full S4 claim still requires the torque-actuated whole-body interaction realizer and randomized push trials.

![Fig. 3. Ten-second dual-MPC root-assisted G1 walking visualization.](code/results/g1_walk_10s_1p2ms.png)

**Fig. 3.** Initial dual-MPC root-assisted G1 walking visualization. The top plot compares CoM forward motion with the 10.8 m trapezoidal-speed reference, the second plot reports left/right foot height, the third plot reports torso roll and pitch, and the bottom plot shows the scheduled left/right support sequence.

### Initial Torque-Level Realizer Smoke Test

A torque-actuated smoke benchmark has been added to test whether the root-assisted visualization can be replaced by a physical whole-body realizer. The script generates a local torque-motor variant of the G1 MJCF, runs the normalized body and task MPCs with the random-walk disturbance observers, and applies a present-sample inverse-dynamics/contact QP of the form

$$
\begin{aligned}
\min_{\ddot q,\tau,\lambda}\quad&
\|J_t\ddot q-\ddot x_{t,{\rm des}}\|^2+
\|J_c\ddot q-\ddot x_{c,{\rm des}}\|^2+
\|\ddot q-\ddot q_{\rm post}\|^2\\
{\rm s.t.}\quad&
M(q)\ddot q+h(q,\dot q)=S^\top\tau+J_c^\top\lambda,\\
&\tau_{\min}\le\tau\le\tau_{\max},\qquad
\lambda\in\mathcal F_{\rm pyr},
\end{aligned}
\tag{26}
$$

and logs torque-limit utilization, post-QP clipping residual, friction margin, contact events, push detection, QP residuals, fall status, and realization failures. The current smoke-test realizer drives the whole-body CoM acceleration for body-port recovery, uses four virtual contact points per stance foot, and uses a rubber-sole MuJoCo contact setting in the torque-actuated model. The stepping gates use the DCM (capture-point) lateral-sway reference and a swing-foot task (the same layer analyzed below), so they are no longer a centered-CoM scaffold; what they still lack for the final production implementation in (22) is a center-of-pressure/DCM stabilizer for sustained single support.

The fixed-support portion of the smoke test now passes. The no-push torque-standing trial completes 3.0 s without falling, and three randomized-push standing trials also complete 3.0 s while detecting all injected pushes with the random-walk disturbance observer. The median maximum roll/pitch angle in the push trials is 0.112 rad, and the median hand RMS error is 23.2 mm. All fixed-support trials reach the torque limits, but the post-QP clipping residual remains small (0.13--0.30 Nm), so the constraints are active rather than bypassed by a separate saturation block. The stepping gates, now driven by the DCM lateral-sway reference, no longer tip immediately: the contact-switch gate carries the faithful recovery through five contact-mode switches before falling at 2.041 s (tipping via roll/pitch, pelvis staying at 0.66 m), and the walking gate through eight switches before falling at 1.889 s (pelvis crossing the 0.45 m threshold). Both still fail — the limiter is the single-support balance bandwidth characterized below, not a centered-CoM reference — and are retained in the denominator: the code supports fixed-support torque-level push rejection and DCM-referenced stepping across several contact-mode switches, but has not yet replaced the root-assisted walking artifact as evidence of sustained torque-level dynamic walking.

| Torque smoke gate | Trials | Passed | Falls | Median completed time |
|---|---:|---:|---:|---:|
| No-push standing | 1 | 1 | 0 | 2.999 s |
| Randomized push standing | 3 | 3 | 0 | 2.999 s |
| Contact-switch command (DCM ref) | 1 | 0 | 1 | 2.041 s (5 switches) |
| Walking command (DCM ref) | 1 | 0 | 1 | 1.889 s (8 switches) |

### Extending Faithful Recovery Through Contact-Mode Switches

The faithful centroidal-wrench recovery that makes H2 hold in fixed support (§X, Table III) was then carried into a stepping gait, to test whether the recovery survives contact-mode switches. Two controllers were built on top of the recovery. The first is a quasi-static stepping controller that regulates the CoM to a support-consistent reference (CoM over the current stance foot); it completes **four contact-mode switches (two steps)** with $\sim$4 cm swing-foot lifts, a genuine improvement over the earlier position-scaffold walking gate that fell at 1.1 s, but single support is only *marginally* stable (a slow tip over $\sim$3–4 s) because the reference asks the robot to statically balance on one small foot. The second is a **divergent-component-of-motion (DCM) walking layer**: a footstep plan and backward DCM recursion generate a *dynamically feasible* CoM trajectory (LIPM $\ddot c=\omega^2(c-p_{\rm zmp})$, ZMP inside the stance foot), which the same normalized centroidal MPC tracks — the interaction-dynamics body port is unchanged; only the reference becomes walk-feasible. This carries the recovery through **seven contact-mode switches** while the body observer keeps the CoM tracking the DCM reference.

To close the loop we then implemented the standard **center-of-pressure/DCM stabilizer** (Englsberger law): the divergent component $\xi=c+\dot c/\omega$ is measured from the momentum-based CoM velocity, the commanded ZMP $p_{\rm cmd}=p_{\rm ref}+(1+k_{\rm dcm}/\omega)(\xi-\xi_{\rm ref})$ is clamped to the support polygon (the CoP limit), and the resulting $\ddot c=\omega^2(c-p_{\rm cmd})$ is realized by the same faithful recovery. This did **not** yield sustained walking either, and it isolates the true limit: it is not estimation or reference bandwidth but single-support **actuation authority**. With an upright torso the CoM can only be accelerated through the ankle CoP, whose $\pm$6 cm foot range caps the horizontal CoM acceleration at $\omega^2\times0.06\approx0.9$ m/s$^2$; correcting the DCM excursions of the $\pm$14 cm wide stance requires more, so the CoP saturates and the DCM diverges after about five switches. On this platform the explicit stabilizer does not even outperform the MPC-tracked reference (five vs. seven switches). We then added the two components standard in the locomotion literature but orthogonal to the interaction-dynamics contribution: (i) a **hip / angular-momentum strategy** — the torso-attitude objective in the realizer is given a separate, relaxable weight so the QP can use centroidal angular momentum for balance beyond the ankle CoP; and (ii) **capture-point step adaptation** — the next footstep is placed at the predicted end-of-step DCM minus the nominal offset, $u_{\rm next}=\xi_{\rm eos}-b_{\rm nom}$ (clamped to kinematic limits), so the robot steps *under* its falling CoM rather than arresting it with the CoP, together with a DCM-tracked walking-initiation shift onto the first stance foot. This full stack does carry the recovery through the contact switches, but it still does not achieve *sustained* walking on this platform: the robot completes only about two adapted steps before the wide-stance lateral balance and the co-tuning of initiation, step timing, foot-placement limits, and hip-relaxation exceed what was reachable here — and it does not surpass the simpler MPC-tracked DCM reference (seven switches). We conclude, honestly, that the faithful centroidal-wrench recovery makes the normalized interaction dynamics hold across contact-mode switches (the paper's claim, demonstrated to seven switches, Fig. 4), and that all the standard walking-stabilizer components (DCM tracking, CoP clamping, hip strategy, step adaptation) have been implemented on top of it, but that turning them into robust continuous locomotion — especially given the G1's wide default stance — is a dedicated locomotion-engineering effort left as future work, separable from the interaction-dynamics representation this paper establishes. Fig. 4 shows the seven-switch DCM run.

![Fig. 4. DCM stepping on the faithful centroidal recovery.](code/results/gait_dcm.png)

**Fig. 4.** DCM walking layer on the faithful recovery: measured CoM vs. the DCM-planned reference (top: lateral, showing the $\pm$14 cm sway and the residual single-support tracking lag; middle: forward), and left/right foot lift (bottom) over the contact-mode switches.

### Results for H3 (Arm-Reaction Preview Reduces Cross-Port Transients)

Returning to the fixed-support hypotheses, H3 tests the coupled realization of Section VII. On the standing torque-actuated G1, a fast oscillating arm-reaction wrench (45 N at 1.6 Hz) is applied at the body port. Because this reaction is *planned* — the task wrench $F_t$ is known — the coupled controller previews its center-of-mass effect $-F_t/m$ and feeds it forward into the body-port acceleration command, whereas the split controller rejects the identical reaction reactively through the body disturbance observer, which lags a fast-changing input. The preview cuts the peak lateral CoM excursion by $2.9\times$:

| Cross-port controller | Peak CoM excursion | RMS CoM excursion |
|---|---:|---:|
| Split (reactive rejection) | 65.5 mm | 28.7 mm |
| Coupled (arm-reaction preview) | 22.4 mm | 12.3 mm |
| Reduction | $2.9\times$ | $2.3\times$ |

**Table V.** H3: peak and RMS lateral CoM excursion during the fast arm reaction, split vs. coupled prediction (Fig. 5a). Because $F_t$ is affine in $u_t$, the preview changes only the body-port input, not the predictor.

### Results for H4 (Contact Events Detected Without an Oracle)

H4 tests the innovation-based detector of Section VIII. A sequence of lateral brace-contact onsets and offsets (six events) is applied to the standing G1; each creates an unmodeled wrench, so the body CoM disturbance observer's normalized innovation $\eta_k=\nu_k^\top S_k^{-1}\nu_k$ spikes. A change detector declares an event when $\eta_k$ exceeds a calibrated threshold (mean $+6\sigma$ of a quiet window) for three consecutive samples, with a refractory interval; the detector never reads the event schedule, which serves only as the oracle for scoring:

| Metric | Value |
|---|---:|
| True contact events | 6 |
| Detected | 6 |
| Missed | 0 |
| False positives | 0 |
| Mean / max detection latency | 56 / 58 ms |

**Table VI.** H4: contact-event detection from the body observer innovation, scored against the scripted-event oracle. All six onsets/offsets are detected with no misses or false positives at $\sim$56 ms latency (Fig. 5b). The latency is the time for the CoM disturbance to register in the innovation; a task-port observer or direct force sensing would reduce it further.

![Fig. 5a. H3 arm-reaction preview.](code/results/h3_coupling.png)

![Fig. 5b. H4 contact-event detection.](code/results/h4_detection.png)

**Fig. 5.** (a) H3: lateral CoM excursion (left) and torso roll/pitch (right) during the fast arm reaction, split vs. coupled preview. (b) H4: the normalized innovation (NIS) over the trial with the calibrated threshold; green dotted lines are oracle contact events, orange lines are detections.

---

## XI. Limitations

The centroidal rotational channel is locally linear and depends on $I_G,\dot I_G$. Contact-force recovery remains mode and geometry dependent. The contact-consistent task inertia can become ill-conditioned near singular task configurations or weak support modes, and simple Cartesian force boxes may underrepresent the corresponding joint-torque amplification. The task and body recovery maps are refreshed at each sample and frozen only over the short solve; the paper does not claim an ISS bound for arbitrary horizon-wide variation of $\Lambda_t$, $I_G$, or the contact geometry. Such variation is logged through observer innovation and realization residuals and remains a target for a future robust certificate.

The contact detector observes model inconsistency and requires kinematic gating; it is not guaranteed to uniquely identify arbitrary external contact. The whole-body interaction realizer can make a requested interaction acceleration infeasible. Hardware deployment also requires filtered velocity estimates and actuator-aware torque smoothing, since raw encoder differentiation would inject high-frequency noise into feedforward and inverse dynamics. The reported 10 s walking video uses the two MPC command layers but remains root assisted, so it should be read as a visualization artifact rather than as dynamic walking validation. The torque realizer delivers offset-free regulation in fixed double support and carries the faithful centroidal-wrench recovery through seven contact-mode switches under a DCM walking reference, but it does not yet sustain continuous walking. A standard CoP/DCM stabilizer was implemented and isolates the binding limit as single-support actuation authority: with an upright torso the ankle center-of-pressure caps the horizontal CoM acceleration near $0.9$ m/s$^2$, below what the wide-stance lateral sway demands, so the CoP saturates and the DCM diverges after a few switches. Sustained walking therefore requires a hip/angular-momentum strategy and capture-point step-timing/placement adaptation layered on the recovery — standard locomotion components orthogonal to the normalized predictor — which remain the pieces for the S4 benchmark. Finally, MuJoCo validation does not replace torque-controlled G1 hardware experiments.

---

## XII. Conclusion

This paper formulates floating-base balance and task interaction as two instances of one predictive object. A centroidal interaction MPC and a task-space interaction MPC share the exact double-integrator ZOH structure, while mass, centroidal inertia, contact-consistent task inertia, friction, and actuator limits are localized to physical realization. The key principle is that interaction dynamics should be predicted, while full robot dynamics should be realized instantaneously as a constrained projection onto the feasible whole-body set. Two representation-level hypotheses are confirmed on the G1: the predictor is port-independent and configuration-invariant (H1 — identical constant $(A,B)$ across ports, lossless equivalence to a conventional force-input MPC, and $\Lambda_t$ varying up to $3840\%$ across an arm sweep while $(A,B)$ stay constant), and offset-free dual-port regulation holds both at the representation level and on the full torque-actuated standing G1 realizer (H2 — CoM error 44.9$\to$2.7 mm and hand error 151$\to$31 mm with the observers, no falls), once the body port is recovered as a centroidal wrench rather than a posture tilt. This confirms that offset-free regulation is a property of the normalized representation that transfers to the robot exactly when recovery is faithful. The initial G1 artifact also shows a 10 s, 10.8 m root-assisted walking visualization, and a torque-actuated smoke benchmark passes fixed-support no-push and randomized-push trials while support switching and walking still fail. The remaining work is therefore concrete: extend the faithful-recovery realizer of (22) to dynamic gait — where the contact set switches during walking — then rerun active-constraint and detected contact-mode-transition trials until failures are eliminated or explicitly characterized.

---

## References

[1] Y. Cao and J. Tang, "Toward Interaction Dynamics: A Predictive Framework for Safe Physical Human-Robot Interaction," 2026, arXiv:2606.08281.

[2] J. Di Carlo, P. M. Wensing, B. Katz, G. Bledt, and S. Kim, "Dynamic locomotion in the MIT Cheetah 3 through convex model-predictive control," in *Proc. IEEE/RSJ IROS*, pp. 1–9, 2018.

[3] D. Kim, J. Di Carlo, B. Katz, G. Bledt, and S. Kim, "Highly dynamic quadruped locomotion via whole-body impulse control and model predictive control," in *Proc. IEEE/RSJ IROS*, pp. 4656–4663, 2019.

[4] C. D. Bellicoso, C. Gehring, J. Hwangbo, P. Fankhauser, and M. Hutter, "Perception-less terrain adaptation through whole body control and hierarchical optimization," in *Proc. IEEE-RAS Humanoids*, pp. 558–564, 2016.

[5] T. Koolen *et al.*, "Design of a momentum-based control framework and application to the humanoid robot Atlas," *Int. J. Humanoid Robotics*, vol. 13, no. 1, 2016.

[6] O. Khatib, "A unified approach for motion and force control of robot manipulators: The operational space formulation," *IEEE J. Robotics Autom.*, vol. 3, no. 1, pp. 43–53, 1987.

[7] L. Sentis and O. Khatib, "Synthesis of whole-body behaviors through hierarchical control of behavioral primitives," *Int. J. Humanoid Robotics*, vol. 2, no. 4, pp. 505–518, 2005.

[8] D. E. Orin, A. Goswami, and S.-H. Lee, "Centroidal dynamics of a biped robot," *Autonomous Robots*, vol. 35, no. 2–3, pp. 161–176, 2013.

[9] L. Righetti, J. Buchli, M. Mistry, and S. Schaal, "Inverse dynamics control of floating-base robots with external constraints: A unified view," in *Proc. IEEE ICRA*, pp. 1085–1090, 2011.

[10] J.-P. Sleiman, F. Farshidian, M. V. Meduri, and M. Hutter, "A unified MPC framework for whole-body dynamic locomotion and manipulation," *IEEE Robot. Autom. Lett.*, vol. 6, no. 3, pp. 4688–4695, 2021.

[11] N. Hogan, "Impedance control: An approach to manipulation—Parts I, II, III," *ASME J. Dyn. Syst. Meas. Control*, vol. 107, no. 1, pp. 1–24, 1985.

[12] A. Albu-Schäffer, C. Ott, and G. Hirzinger, "A unified passivity-based control framework for position, torque and impedance control of flexible joint robots," *Int. J. Robotics Research*, vol. 26, no. 1, pp. 23–39, 2007.

[13] R. Grandia, F. Jenelten, S. Yang, F. Farshidian, and M. Hutter, "Perceptive locomotion through nonlinear model-predictive control," *IEEE Trans. Robotics*, vol. 39, no. 5, pp. 3402–3421, 2023.

[14] B. Stellato, G. Banjac, P. Goulart, A. Bemporad, and S. Boyd, "OSQP: An operator splitting solver for quadratic programs," *Math. Program. Comput.*, vol. 12, no. 4, pp. 637–672, 2020.

[15] E. Todorov, T. Erez, and Y. Tassa, "MuJoCo: A physics engine for model-based control," in *Proc. IEEE/RSJ IROS*, pp. 5026–5033, 2012.

[16] Y.-Y. Cao, Z. Lin, and D. G. Ward, "Anti-windup design of output tracking systems subject to actuator saturation and constant disturbances," *Automatica*, vol. 40, no. 7, pp. 1221–1228, Jul. 2004.

[17] D. E. Orin and A. Goswami, "Centroidal momentum matrix of a humanoid robot: Structure and properties," in *Proc. IEEE/RSJ IROS*, pp. 653–659, 2008.
