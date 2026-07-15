# Interaction Dynamics for Floating-Base Whole-Body Manipulation

**Yongyan Cao**

---

## Abstract

Floating-base loco-manipulation is usually split across a centroidal predictor, a task-space impedance controller, and a whole-body inverse-dynamics layer. This paper separates *canonical prediction* from *physical authority* and anchors the split on rate: a $1$ kHz servo holds the last optimized command, while a $200$ Hz node solves both predictors and exactly **one** whole-body quadratic program per update (median $3.0$ ms of a $5$ ms budget), so the $\approx2$ ms QP never has to close a $1$ kHz loop. Requests evolve in residual-acceleration coordinates $x_{k+1}=Ax_k+B(u_k+d_k)$ whose exact-ZOH pair $(A,B)$ is invariant to configuration and contact mode; the robot re-enters only as the *admissible geometry* of the command. Because the command enters the QP through linear cost terms alone, one KKT solve on the factorization the realizer already computes yields the feedforward, the input maps, and the constraints $H_ku\le h_k$ — with no second optimization. That single-cell set is the QP's *critical region*, not its feasible set: it refuses $68\%$ of feasible commands, falls in $4/4$ single-support transitions, and breaks the task port. Crossing an active-set boundary is not a failure but a redistribution of contact forces, so we **walk the piecewise-affine solution map**, locating entering and leaving constraints in closed form from the same KKT system. The walk recovers the exact feasible set (zero false positives *and* zero false negatives) with **no extra whole-body QP solves**, sustains the transition $4/4$, and preserves $14\times$ offset-free hand rejection. Contact transitions change the admissible geometry, never the canonical dynamics.

**Index Terms** - interaction dynamics, centroidal MPC, whole-body control, floating-base robots, loco-manipulation, physical human-robot interaction, model predictive control.

---

## I. Introduction

Humanoid robots regulate two physical interfaces at once: the feet exchange forces with the environment for balance, the hands with people, tools, and objects for a task. Existing stacks assign these to different objects — a centroidal MPC plans contact forces, a whole-body QP maps them to joint commands, an impedance controller regulates the hand — which is practical but hides that both are the same interaction-dynamics problem.

The question is whether floating-base manipulation admits the same normalized interaction-dynamics representation derived for fixed-base systems in [1] without making prediction blind to current physical capability: whether balance and manipulation can share one canonical predictive model while robot-specific mechanics enter only through the admissible command geometry. Such a robot-independent model is also a natural target for learned whole-body *intent*, which a realization-informed feasible set turns into constraint-aware execution.

At the body port the interaction is between centroidal motion and the net contact wrench; at the task port, between end-effector motion and task wrench. In both, known dynamics and desired acceleration go in feedforward, leaving a residual acceleration input:

$$
\begin{aligned}
\text{physical wrench}
&=\text{model feedforward}\\
&\quad+\text{interaction inertia}\times u.
\end{aligned}
\tag{1}
$$

The resulting error model is the interaction-dynamics backbone of [1]. The floating-base case is nontrivial because contact geometry, support changes, friction, center-of-pressure limits, actuator saturation, nominal feedforward, and arm–body reactions together determine which normalized accelerations are *currently* realizable. The central claim is therefore a bidirectional prediction–realization separation: the full robot dynamics do not replace the canonical horizon model, but they must tell it what it may ask for. A capability query supplies the admissible set $\widehat{\mathcal U}_k$; the predictors keep the same exact-ZOH pair across contact modes, constrain their commands to that set, and pass the first request to the instantaneous realizer, which executes it and reports whatever residual remains. The query must be cheap, or the separation is worthless — and it is: because the command enters the whole-body QP only through linear cost terms, the solution is affine on the current active set, and one KKT solve on the factorization the realizer has already computed returns the feedforward and the input maps from which $\widehat{\mathcal U}_k$ follows.

A subtler failure then appears, and it is the technical heart of this paper. A set built from one active set is the QP's *critical region* — where no new constraint activates — not the set of commands the robot can realize; the two differ threefold, enough to topple the robot in single support and break the task port. But an active-set change is not a failure of realization, only a redistribution of contact forces. Following the solution across those regions — entering and leaving constraints located in closed form from the *primal and dual* blocks of the same KKT system — recovers the exact feasible set, still without another whole-body QP. In short:

> **Contact transitions change the admissible input geometry, not the canonical interaction dynamics.**

The contributions follow this separation:

1. **Canonical ports.** Body and task interaction are two ports of one exact-ZOH model whose pair $(A,B)$ is independent of configuration, inertia, and contact mode.
2. **Rate-matched architecture.** The $\approx2$ ms whole-body QP runs with both predictors, sequentially, in one **200 Hz** node inside a $5$ ms budget; a **1 kHz** servo holds the last optimized command between updates.
3. **Authority for free.** The command enters the QP only through linear cost terms, so **one KKT solve** on the realizer's own factorization yields the feedforward, the input maps, and the constraints $H_ku\le h_k$ — no second optimization.
4. **Exact authority by continuation.** The single-cell set is the QP's *critical region* and refuses $68\%$ of feasible commands (falling $4/4$ in single support, breaking the task port); **walking the piecewise-affine solution map** recovers the exact feasible set (zero false positives *and* negatives) with **no extra QP solves**, sustaining the transition $4/4$ and preserving $14\times$ hand rejection.

The scope is bounded. Neither estimator is certified — the single-cell map holds on one critical region, the continuation only under regularity conditions not checked online — so the zero false-positive rates are evidence, not proof, and the realizer stays the hard layer that makes a limit violation impossible regardless. The rates are met in a Python prototype, not on hardware; the multirate structure is decimated in simulation, not threaded; and the scripted support transfer is not a walking result.

Figure 1 summarizes the resulting architecture. A $200$ Hz node reads the state and model once, then solves both predictors and one whole-body QP in sequence, publishing the admissible set $H_ku\le h_k$ from that solve's KKT system; a $1$ kHz servo holds the optimized torque between updates, and the tight authority estimate is refreshed asynchronously. A stale or mode-mismatched set falls back to a conservative box, and the predictor without a fresh command reuses the last valid one.

![Fig. 1. The multirate interaction-dynamics architecture.](figures/multirate_architecture.png)

**Fig. 1.** The multirate interaction-dynamics architecture.

---

## II. Related Work

Centroidal and single-rigid-body MPC [2], [3], [8], [13] predict CoM and orientation while optimizing contact forces over a gait schedule, carrying friction, unilateral contact, and support geometry in the horizon. Our body port instead predicts a normalized residual acceleration and moves the contact forces out of the predictor entirely: at the current sample the realizer maps the physical limits into a residual-command set that constrains the canonical predictor.

Whole-body inverse dynamics and hierarchical QPs [4], [5], [7], [9] enforce rigid contacts, task priorities, and actuator limits. They remain essential here — but as the instantaneous realizer that maps a desired body wrench and task acceleration to feasible generalized forces, not as a second predictive model. Operational-space impedance, admittance, and task-space MPC [6], [11], [12] regulate the end-effector; residual-acceleration coordinates remove their configuration-dependent apparent inertia from the prediction while retaining it in force recovery.

Learning-based pipelines (reinforcement-learning policies, demonstration retargeting) increasingly *generate* whole-body references, but a kinematic reference is not guaranteed executable under real contact forces and actuator limits; it still needs a local, model-based layer for constraints and reactive compliance. This framework is exactly that layer, whether the reference is hand-authored or learned.

Closest is unified whole-body MPC for locomotion and manipulation [10], which optimizes a single predictive model; we differ in the prediction–realization split, predicting only the two normalized interaction dynamics while the full contact-constrained dynamics act at the current sample as a feasibility projection. The normalized model, offset-free regulation, and impedance interpretation belong to [1]; the centroidal model [8], [17], whole-body inverse dynamics [9], and the integrating-disturbance observer [16] are prior tools. We contribute their floating-base integration, the KKT-based authority query and its piecewise-affine continuation, and evaluation on a Unitree G1 in MuJoCo [15].

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

The controller uses two controlled ports. The body port is defined by CoM position and body-orientation errors — the orientation channel regulated in centroidal angular-momentum coordinates (Section IV), with a desired attitude entering only through an outer loop — and the task port is defined by Cartesian end-effector tracking error. The active contact mode $\rho$ changes the contact Jacobian and feasible wrench set. Following [1], it does not change the normalized prediction pair used below. This invariance is the organizing idea of the paper, which we state once and then use throughout.

**Definition 1 (interaction-dynamics representation, under exact feedforward normalization).** An *interaction-dynamics representation* of a controlled port consists of: (i) a canonical requested model $\ddot e=u+d$ whose exact-ZOH state-transition pair $(A,B)$ is independent of the robot's mechanics; (ii) a realization-informed admissible set $\widehat{\mathcal U}_k$ supplied at every MPC update from the current robot state, reference feedforward, contact mode, and physical constraints; and (iii) an instantaneous recovery map that realizes the selected command and reports the residual $r=\ddot e^{\rm real}-(u+d)$. Robot mechanics therefore do not enter $(A,B)$, but they do enter the input geometry $\widehat{\mathcal U}_k$ and recovery map.

The qualifier is essential, and we make its hypotheses explicit.

**Assumption 1 (exact feedforward normalization).** A controlled port has physical coordinate $x$ with tracking error $e = x - x_d$, and constrained dynamics $M_p(q,\rho)\ddot x + \mu_p(q,\dot q,\rho) = F_p^{\rm act} + F_p^{\rm ext}$, where the *interaction inertia* $M_p$ is invertible and well-conditioned on the operating set. The commanded actuation cancels the bias and injects the desired-trajectory plus residual acceleration, $F_p^{\rm act} = \mu_p + M_p(\ddot x_d + u) + \delta$, and the recovery map $u \mapsto F_p^{\rm act}$ is affine on each active-constraint cell.

**Theorem 1 (canonical dynamics invariance).** Under Assumption 1, every such port admits the normalized realized model $\ddot e=u+d+r$, with residual-acceleration input $u$, interaction disturbance $d=M_p^{-1}F_p^{\rm ext}$, and realization residual $r=M_p^{-1}\delta$. The exact-ZOH pair $(A,B)$ of the requested integrator chain is independent of $M_p$, $\mu_p$, and contact mode $\rho$. These quantities may change the admissible input set $\widehat{\mathcal U}_k$ and recovery map at every update, but they do not change the canonical state-transition matrices. Hence a contact transition is constraint switching, not dynamics switching, in the requested coordinates.

*Proof sketch.* Substituting the feedforward of Assumption 1 into the constrained dynamics cancels $\mu_p$, leaving $M_p\ddot x = M_p(\ddot x_d + u) + F_p^{\rm ext} + \delta$; subtracting $M_p\ddot x_d$ and left-multiplying by $M_p^{-1}$ gives $\ddot e = \ddot x - \ddot x_d = u + d + r$. The map from $u$ to $\ddot e$ is the identity, so the sampled predictor is the integrator ZOH pair, which contains no entry of $M_p$, $\mu_p$, or $\rho$. $\square$

**Remark (dynamics-invariant does not mean physics-blind).** Under actuation redundancy the recovery map is generally non-unique: different QP weightings, task priorities, and contact-force allocations produce different local authority sets and residuals, while sharing the same $(A,B)$. The invariant object is the requested dynamics, not the available control authority. Upstream planners may reuse the canonical transition pair, but they must accept the current admissible geometry supplied by the robot-specific realizer.

Intuitively, the representation behaves like a stable software interface with a capability query. The transition contract $\ddot e=u+d$ does not change across robots or contacts, but the realizer reports which $u$ values it can currently honor. As shown in Fig. 2, state/reference/contact data produce $\widehat{\mathcal U}_k$, the constrained predictor selects $u\in\widehat{\mathcal U}_k$, the realizer executes it, the residual $r$ closes the realization account, and the Kalman observer estimates $d$. Different robots therefore share one transition interface while exposing different current authority.

![Fig. 2. Prediction–realization interface.](figures/prediction_realization_concept.png)

**Fig. 2.** Prediction–realization interface.

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

The rotational channel is expressed directly in centroidal **angular-momentum** coordinates, which avoids identifying the locked-inertia angular velocity with a rigid-body attitude rate. Let $k_G=A_G(q)\dot q$ be the centroidal angular momentum, with $A_G$ the angular block of the centroidal momentum matrix, and let $M_c$ be the net contact moment about the CoM. Then

$$
\dot k_G=M_c+w_\theta,
\tag{7}
$$

where $w_\theta$ lumps unmodeled moments. For the angular-momentum error $e_h=k_G-k_{G,d}$, define the desired net moment

$$
M_c^{\rm des}=\dot k_{G,d}+u_\theta.
\tag{8}
$$

When the recovered contacts realize $M_c=M_c^{\rm des}$,

$$
\dot e_h=u_\theta+d_\theta,\qquad d_\theta=w_\theta,
\tag{9}
$$

a **first-order** integrator that is exact under exact moment recovery, with no locked-inertia-to-attitude approximation. A desired attitude, when required, enters through the reference $k_{G,d}$ via an outer regulator (e.g. $k_{G,d}=-K_\theta\,\log(RR_d^\top)^\vee$), whose local validity is a property of that loop, not of the port. The two body channels thus differ in order — second-order on CoM error, first-order on angular-momentum error.

Neither (6) nor (9) is the true plant: the recovered force and moment are realized only up to a **realization residual** (Section VI), kept explicit rather than folded into $d$. Writing $r_b=[r_c^\top,r_h^\top]^\top$, the realized body port obeys

$$
\ddot e_c=u_c+d_c+r_c,\qquad
\dot e_h=u_\theta+d_\theta+r_h,
\tag{6$'$}
$$

with $r_b=0$ exactly when the requested centroidal wrench and moment are feasibly recovered. Stacking the second-order CoM channel and the first-order angular-momentum channel,

$$
x_b=[e_c^\top,\dot e_c^\top,e_h^\top]^\top,\quad
u_b=[u_c^\top,u_\theta^\top]^\top,\quad
d_b=[d_c^\top,d_\theta^\top]^\top,
$$

the exact-ZOH construction of [1] at period $T_b$ (inputs $u_b,d_b,r_b$ held over each interval) gives

$$
x_{b,k+1}=A_bx_{b,k}+B_b(u_{b,k}+d_{b,k}+r_{b,k}),
\tag{10}
$$

$$
A_b=
\begin{bmatrix}I_3&T_bI_3&0\\0&I_3&0\\0&0&I_3\end{bmatrix},
\qquad
B_b=
\begin{bmatrix}
\tfrac12T_b^2I_3&0\\
T_bI_3&0\\
0&T_bI_3
\end{bmatrix}.
\tag{11}
$$

The pair $(A_b,B_b)$ is constant; mass, centroidal inertia, contact locations, and contact mode appear only in wrench recovery and constraints.

**Corollary 1 (canonical port representation).** Each controlled port satisfies Assumption 1, so Theorem 1 applies to it verbatim. For the body port, the *requested* dynamics are the canonical model (10) with the constant exact-ZOH pair (11): a double integrator on the CoM error and a first-order integrator on the centroidal angular-momentum error. The translational channel follows exactly from centroidal force balance (4)–(6); the rotational channel follows exactly from centroidal angular-momentum balance (7)–(9), with no attitude approximation. Mass, centroidal inertia, contact locations, friction, and center-of-pressure limits enter only the recovery map and the feasible input set, not $(A_b,B_b)$. The realized body port equals the requested model up to the realization residual $r_b$ of Section VI, and coincides with it when $r_b=0$; the task port of Section VII is the same instance with $M_p$ the contact-consistent inertia $\Lambda_t$.

**Proof.** Substituting (5) into (4) gives (6) and (8) into (7) gives (9); stacking yields a block-diagonal generator (double integrator $\oplus$ first-order integrator) whose exact ZOH is (11). The quantities $m,A_G,p_i,\rho$ enter only the recovery (12)–(15), and the requested-versus-recovered gap is the residual $r_b$ of Section VI. $\square$

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
\dot k_{G,d}+u_\theta
\end{bmatrix}.
\tag{13}
$$

Recovery enforces

$$
\mathcal G_\rho f=W_b^{\rm des}(u_b)+s_W,
\tag{14}
$$

where $s_W$ is a penalized wrench residual, zero under exact recovery. It maps to the body-port acceleration residual of (6$'$): force part $r_c=m^{-1}s_W^{\rm (force)}$, moment part $r_h=s_W^{\rm (moment)}$. Under a scheduled contact sequence $\mathcal G_{\rho_j}$ is re-formed per mode while $(A_b,B_b)$ stay unchanged.

Under the prediction–realization separation the body MPC predicts only the canonical state transition; it does not duplicate $M$, $J_c$, contact forces, or joint torques across the horizon. It is nevertheless constrained by the current physical capability, and that constraint is obtained from the whole-body QP the realizer is *already solving* — not from a second optimization.

**The solved form of the realizer.** Eliminating the slacks $s_W,s_t$ of (22) against their quadratic penalties turns the realizer into the weighted problem actually solved,

$$
\min_{z}\ \tfrac12 z^\top P z+\big(q_0+Q_u\,u\big)^\top z
\quad\text{s.t.}\quad
Ez=e,\qquad Cz\le c,
\tag{14b}
$$

with $z=[\ddot q^\top,\tau^\top,\lambda^\top]^\top$, $E z=e$ the rigid-body dynamics and rigid-contact rows, and $Cz\le c$ the actuator bounds, friction pyramid, unilateral-force and one-step joint limits — all of which remain **hard**. The stacked residual command $u=[u_b^\top,u_t^\top]^\top$ enters (14b) only through the *linear* term, via the CoM-acceleration, centroidal-wrench and end-effector-acceleration targets; the Hessian $P$ does not depend on $u$. This is the structural fact the rest of the section rests on, and it is a property of the eliminated form (14b), not of the slack form (22).

**Authority from one KKT solve.** Let $\mathcal A_k$ index the rows of $Cz\le c$ active at the current solution $z_0=z_k^\star(0)$ — identified from the solver's dual variables, a row being active when it sits at its bound and its multiplier exceeds a threshold $\varepsilon_\lambda$ — and let $C_{\mathcal A_k}$ stack those rows together with the equality rows $E$, which are always active. On the *critical region* where $\mathcal A_k$ is unchanged, the solution of (14b) is affine in $u$, and its Jacobian solves the KKT system

$$
\begin{bmatrix}P & C_{\mathcal A_k}^{\!\top}\\ C_{\mathcal A_k} & 0\end{bmatrix}
\begin{bmatrix}K\\ \ast\end{bmatrix}
=
\begin{bmatrix}-\,Q_u\\ 0\end{bmatrix},
\qquad
z_k^\star(u)\approx z_0+Ku .
\tag{14c}
$$

The nominal whole-body QP supplies $z_0$ — hence the feedforward $\tau_{\rm ff},\lambda_{\rm ff}$ — and (14c) supplies only the sensitivity $K$; partitioning it gives the input maps $K_\tau,K_\lambda$, so that

$$
\tau=\tau_{\rm ff}+K_\tau u,\qquad
\lambda=\lambda_{\rm ff}+K_\lambda u .
\tag{14d}
$$

**Mapped constraints.** Substituting (14d) into the physical limits the realizer already enforces, and into the realization-tolerance test $\lVert r_k(u)\rVert_\infty\le\epsilon_r$ linearized about the same cell, yields a polyhedral admissible command set

$$
\widehat{\mathcal U}_k=\{u:\ H_k u\le h_k\}.
\tag{14e}
$$

Three properties must be stated plainly. *(i)* The tolerance rows are not optional: a command can satisfy every actuator and contact limit and still fail to be realized, because the realizer trades the request against its other objectives. *(ii)* A row with $h_{k,i}<0$ means the nominal solution already sits past that **margin** — never past a true limit, since (14b) enforces the true limits as hard constraints — and is clamped at zero, which keeps $u=0$ admissible. Soundness therefore rests on the realizer's hard constraints, not on the construction of (14e); $\widehat{\mathcal U}_k$ is an inner *model*, not a certificate. *(iii)* The affine map (14c) is exact only on the current critical region: once a new constraint activates the true QP redistributes and the map bends. The resulting conservatism is measured, not assumed (Section X-E), and the realizer remains the final hard layer.

**Beyond one cell: piecewise-affine continuation.** The set (14e) is, in essence, the *critical region* of the active set $\mathcal A_k$ — the region on which no new constraint activates. But an active-set change is not a failure of realization: when a contact force saturates, another takes over and the request is still met. The true feasible set therefore spans several critical regions, and (14e) refuses most of it.

The remedy uses no new machinery. Along a ray $u=t\,e$, the solution is affine until one of exactly two events occurs: an inactive row of $Cz\le c$ reaches its bound (a constraint **enters** $\mathcal A_k$), or an active row's multiplier reaches zero (a constraint **leaves**). Both events are available in closed form from the KKT system (14c), whose primal block gives $\mathrm dz/\mathrm du$ and whose *dual* block gives $\mathrm d\nu/\mathrm du$; the first event along the ray is the smaller of the two breakpoints. At that point $\mathcal A_k$ is updated, (14c) is re-solved, and the walk continues. Because the realization residual is affine on each region, the tolerance crossing is exact, and *that* crossing is the authority boundary. The walk costs one KKT solve per region traversed and — the essential point — **no whole-body QP solves at all**. Section X-E shows it recovers the exact feasible set.

Because $H_k,h_k$ come from a single KKT solve on a factorization the realizer already possesses, the single-cell set is refreshed inside the $200$ Hz optimization node at a cost of $\approx0.6$ ms, against $\approx150$ ms for the exact realizer query used only as an offline oracle (Section X-E). For split body/task MPCs a **body-priority allocation** is used: the body slice is computed with the task request held at nominal, and the task port then receives the remaining capacity $H_tu_t\le h_t-H_{tb}u_b^\star$ from the same joint sensitivity, at no extra QP or KKT solve.

Authority may be refreshed at every update; the set is frozen over the horizon, so the transition matrices $(A_b,B_b)$ and the predictor's condensed **cost** Hessian $\Psi$ (the quadratic form of (15), which depends only on $A_b,B_b,Q_b,R_b$ and the horizon, never on the robot) stay constant while only the constraint rows change. The invariant *cost* matrix $\Psi$ must not be confused with the *constraint* matrix $H_k$, which is exactly what moves. For a scheduled contact switch a transition window may use $\widehat{\mathcal U}_{\rm tr}=\widehat{\mathcal U}_{\rm DS}\cap\widehat{\mathcal U}_{\rm SS}$, which reduces incompatible requests but does not by itself prove recursive feasibility.

The body MPC is therefore

$$
\begin{aligned}
\min_{U_b}\quad&
\sum_{j=0}^{N_b-1}
\left(
\|x_{b,j}\|_{Q_b}^2+
\|u_{b,j}+\hat d_{b,k}\|_{R_b}^2
\right)
+\|x_{b,N_b}\|_{S_b}^2\\
\text{s.t.}\quad&
x_{b,j+1}=A_bx_{b,j}+B_b(u_{b,j}+\hat d_{b,k}),\\
&u_{b,j}\in\widehat{\mathcal U}_{b,k}.
\end{aligned}
\tag{15}
$$

No robot-specific quantity enters the state transition or cost; current mechanics enter only through $\widehat{\mathcal U}_{b,k}$ and through recovery and residual after the first command. Contact transitions thus schedule admissible geometry while leaving $(A_b,B_b)$ unchanged.

**Why the realization residual does not appear in (15).** The realized plants (10), (20) carry $r$; the predictors do not, and this is deliberate. The residual is not an exogenous signal but a function of the decision variable, $r=r(u)$, zero inside the realizable set and growing outside it; the constraint $u_{b,j}\in\widehat{\mathcal U}_{b,k}$ **is** the predictor's representation of it, so carrying an additive $r$ term as well would double-count the same physics — and the realizer reports $r$ only after execution in any case. Whatever the frozen local set fails to prevent splits in two: its matched, slowly-varying part is indistinguishable from a disturbance, so the observer of Section VIII converges to an **effective disturbance** $d^{\rm eff}=d+r_{\rm matched}$ and cancels it; the unmatched part is bounded, not modelled, by Proposition 4. So (15) predicts the *requested* dynamics and confines physical infeasibility to the constraint set and the logged residual.

The input-centered penalty is essential: for a constant estimated disturbance the cancelling equilibrium is $u_b=-\hat d_b$, so penalizing $\|u_b+\hat d_b\|$ rather than $\|u_b\|$ avoids reintroducing a steady-state offset.

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

Thus $\Lambda_t$ already contains the floating-base, stance-contact, and arm–body inertial coupling of the full constrained system, without assuming the arm is isolated from the base. Predictable base reactions are handled by the body-port preview (Section VII); unmodeled coupling and recovery error go into $d_t$ and the residuals of (22).

Projecting the constrained rigid-body dynamics into the task coordinates gives the contact-consistent task-space dynamics

$$
\Lambda_t\,\ddot x_t+\mu_{t,\rho}=F_t^{\rm act}+F_h+r_{t,\rm dyn},
\tag{17b}
$$

where $\mu_{t,\rho}=\bar J_t^\top h-\Lambda_t\dot J_t\dot q$ collects Coriolis, centrifugal, and gravity terms, $F_t^{\rm act}$ is the task-space wrench *actually* produced by the joint torques, $F_h$ is the external task wrench, and $r_{t,\rm dyn}$ lumps contact-consistency and higher-priority null-space couplings. For $e_t=x_t-x_{t,d}$, the controller **requests** the wrench

$$
F_t^{\rm cmd}=F_{t,\rm ff}+\Lambda_tu_t,\qquad
F_{t,\rm ff}=\Lambda_t\ddot x_{t,d}+\mu_{t,\rho},
\tag{18}
$$

equivalently the acceleration request $\ddot x_t^{\rm req}=\ddot x_{t,d}+u_t$. Let $\ddot x_t^{\rm real}$ be the acceleration the whole-body realizer actually imposes, and define the **realization residual**

$$
r_t=\ddot x_t^{\rm real}-\ddot x_t^{\rm req}.
\tag{18b}
$$

Substituting (18) into (17b): when the requested wrench is realized exactly ($F_t^{\rm act}=F_t^{\rm cmd}$, $r_t=0$) the nominal acceleration cancels and $\ddot e_t=u_t+\Lambda_t^{-1}F_h+d_{\rm model}$. In general the realized task port is

$$
\ddot e_t=u_t+d_{h,t}+r_t,\qquad
d_{h,t}=\Lambda_t^{-1}F_h+d_{\rm model},
\tag{19}
$$

with $d_{h,t}$ the observer-cancelled external/model disturbance and $r_t$ the physical-infeasibility residual (kept distinct, as for the body port). Exactly as for the body port (10), the *realized* task port discretizes by exact ZOH as

$$
x_{t,k+1}=A_tx_{t,k}+B_t(u_{t,k}+d_{h,t,k}+r_{t,k}),
\tag{20}
$$

with

$$
A_t=\begin{bmatrix}I_3&T_tI_3\\0&I_3\end{bmatrix},
\qquad
B_t=\begin{bmatrix}\tfrac12T_t^2I_3\\T_tI_3\end{bmatrix},
\tag{20b}
$$

the canonical three-dimensional exact-ZOH double-integrator pair.

By Corollary 1 the task port is the same instance of Theorem 1 with $M_p=\Lambda_t$. For a fixed active contact mode with $J_{c,\rho}M^{-1}J_{c,\rho}^\top$ and $J_t\bar M_\rho^{-1}J_t^\top$ nonsingular on the operating set, the *requested* end-effector port is the canonical model (20) with a constant exact-ZOH pair; configuration and contact mode enter through $\Lambda_t$, the feedforward $\mu_{t,\rho}$, and the feasible set, not through $(A_t,B_t)$, and the realized port equals the requested model up to $r_t$. Concretely, the constrained inverse (16) gives the contact-consistent inertia $\Lambda_t$ (17); substituting the commanded wrench (18) into (17b) cancels the nominal terms and leaves (19), whose requested part has exact ZOH (20), with gap $r_t$ the realizer residual $s_t$ of (22).

Like the body port, the task MPC retains the canonical transition pair and minimizes $\sum_j\!\big(\|x_{t,j}\|_{Q_t}^2+\|u_{t,j}+\hat d_{t,k}\|_{R_t}^2\big)+\|x_{t,N_t}\|_{S_t}^2$. Its constraint is not a fixed end-effector acceleration box. The same joint KKT sensitivity (14c) that produced the body maps also produces the task maps $K_{\tau,t},K_{\lambda,t}$, at no extra solve. Under the body-priority allocation, the body command $u_{b,k}^\star$ is committed first and consumes part of the shared actuator and contact budget; the task port then receives what remains,

$$
\widehat{\mathcal U}_{t,k}
=\{u_t:\ H_{t,k}\,u_t\le h_{t,k}-H_{tb,k}\,u_{b,k}^\star\}.
\tag{21}
$$

The rows of (21) are the same physical limits mapped through $\tau=\tau_{\rm ff}+K_{\tau,b}u_b^\star+K_{\tau,t}u_t$ and the contact-force map, expressing the authority left after the committed body request. This is an allocation policy, not a joint body–task optimum: a different priority yields a different task set, the set contracts near a task singularity or torque boundary, and it is empty when the nominal task residual already exceeds tolerance, forcing a fallback.

---

## VI. Whole-Body Interaction Realizer

The two MPCs output a body request (a centroidal wrench $W_b^{\rm des}$, realized by contact forces) and a task request (the end-effector acceleration $\ddot x_{t,d}+u_t^\star$, realized by joint torques). The task is imposed as an acceleration with an exposed shortfall $s_t$, while the body request stays a wrench because the unilateral/friction/CoP constraints act on wrenches. This layer predicts no future states — it is an instantaneous projection of both requests onto the generalized accelerations, contact wrenches, and joint torques satisfying the floating-base dynamics and constraints, not a second MPC.

Let $S_j$ select the actuated joint coordinates from $q$, and let $\tau_{\rm ref}$ be a nominal torque used only for regularization, such as the previous command or a gravity-compensating inverse-dynamics torque. With polyhedral friction pyramids, the whole-body interaction realizer is the convex instantaneous inverse-dynamics QP

$$
\begin{aligned}
\min_{\ddot q,\tau,\lambda,s_W,s_t}\quad&
\|s_W\|_{W_b}^2
+\|s_t\|_{W_t}^2
+\|\tau-\tau_{\rm ref}\|_{W_\tau}^2\\
\text{s.t.}\quad&
M\ddot q+h=S^\top\tau+J_c^\top\lambda+J_t^\top F_h,\\
&J_c\ddot q+\dot J_c\dot q=0,\\
&\mathcal G_\rho\lambda=W_b^{\rm des}+s_W,\\
&J_t\ddot q+\dot J_t\dot q=\ddot x_{t,d}+u_t^\star+s_t,\\
&\lambda\in\mathcal F_\rho,\quad
\tau_{\min}\le\tau\le\tau_{\max},\\
&q_j^+=S_j(q+\Delta t\dot q+\tfrac12\Delta t^2\ddot q),\\
&q_{j,\min}+\epsilon\le q_j^+\le q_{j,\max}-\epsilon.
\end{aligned}
\tag{22}
$$

The external wrench $F_h$ is a known constant used in one of two *mutually exclusive* modes, so it is never compensated twice: *measured-wrench feedforward* inserts sensed $F_h$ into the dynamics and removes it from the task disturbance state; *observer-only rejection* sets $F_h=0$ there and lets $u_t$ cancel $\hat d_t$. The two realization slacks are $s_W=\mathcal G_\rho\lambda-W_b^{\rm des}$ (a six-dimensional wrench residual) and $s_t=(J_t\ddot q+\dot J_t\dot q)-(\ddot x_{t,d}+u_t^\star)$ (a task-acceleration residual); they relate to the port residuals by $r_b=\mathcal D_b s_W$, $\mathcal D_b=\operatorname{diag}(m^{-1}I_3,I_3)$, and $s_t=r_t$. The joint-limit row's $\tfrac12\Delta t^2\ddot q$ term makes the one-step check depend on the decision variable.

With a friction-pyramid approximation $\mathcal F_\rho$ is polyhedral and (22) is a convex QP solvable by operator splitting [14]; exact Coulomb cones make it a second-order cone program. Balance can be made hard ($s_W=0$) or soft (large $W_b$), and task tracking is softened through $s_t$ when the two requests conflict.

The realizer has two current-sample roles, and it performs **one** QP solve to serve both. Solving (14b) yields the torques applied this sample and the nominal feedforward $z_0=(\ddot q_0,\tau_{\rm ff},\lambda_{\rm ff})$; one KKT sensitivity (14c) on the same factorization then yields the input maps, and hence the admissible set (14e) published to the predictors. The realizer never runs a second optimization to answer the capability query. Having executed the first stacked request it reports

$$
u_k^{\rm req},\qquad
u_k^{\rm real},\qquad
r_k=u_k^{\rm real}-u_k^{\rm req},
\tag{22b}
$$

together with the nominal torques and wrenches, the friction/CoP/torque/joint margins, and the active-constraint class. These distinguish three events that must not be conflated: the predictor reaching its admissible boundary, the realizer trading the request through $s_W$ or $s_t$, and a final numerical safety clip.

The authority query uses only current measured/estimated state and known references; it does not read future disturbance realizations. For a scheduled contact transition, future mode geometry may be supplied by the contact plan, but state-dependent queries must still be refreshed. The instantaneous realizer remains the final hard safety layer because a sampled box can be invalidated by inter-sample state motion, estimation error, and unmodeled contact.

---

## VII. External-Wrench and Internal-Momentum Preview

The split controller solves the body MPC (15) and the task MPC independently; whatever the arm does reaches the body port only after it is observed, and is rejected reactively through $d_b$. The coupled controller instead previews the *planned* effect of the arm on the centroidal dynamics. Two physically distinct effects must be kept separate, because conflating them double-counts forces.

**External task wrench.** When the hand exchanges a real contact force with the environment — pushing, carrying, leaning on a rail — the environment reaction $F_h^{\rm plan}$ (and moment $M_h^{\rm plan}$) enters the *whole-robot* centroidal-momentum balance as a genuine external wrench, with known centroidal contribution

$$
W_{G,h}^{\rm ext}=
\begin{bmatrix}
F_h^{\rm plan}\\
(x_h-c)\times F_h^{\rm plan}+M_h^{\rm plan}
\end{bmatrix}.
\tag{23}
$$

The body port previews it by adding $-W_{G,h}^{\rm ext}$ to the desired centroidal wrench (13) over the horizon, so the contacts are pre-loaded before $d_b$ has to build up.

**Internal arm motion.** A fast *free-space* reach exerts no external force on the robot. The operational-space control wrench $F_t^{\rm ctrl}=F_{t,\rm ff}+\Lambda_t u_t$ acts through joint torques and is *internal*; it must **not** be inserted into the centroidal balance as an external $-F_t^{\rm ctrl}$, because internal actuation cannot change the total centroidal momentum and doing so double-counts. The genuine base reaction is the rate of change of centroidal momentum carried by the arm. Partitioning the centroidal momentum matrix into base, leg, and arm columns in $h_G=A_G(q)\dot q$, the planned arm motion contributes

$$
\dot h_{G,\rm arm}^{\rm plan}
=A_{G,\rm arm}(q)\,\ddot q_{\rm arm}^{\rm plan}
+\dot A_{G,\rm arm}(q,\dot q)\,\dot q_{\rm arm}^{\rm plan},
\tag{23b}
$$

Equation (23b) is the reaction a *split* body predictor would exchange with an independent arm plan. But in the unified whole-body QP the arm columns already appear in the total CoM Jacobian, so adding $-\dot h_{G,\rm arm}^{\rm plan}$ again double-counts the same internal motion; external wrench preview stays explicit because the environment wrench is not generated by those internal accelerations.

**Status.** This preview is *derived, not evaluated*. The controller of Section X does not implement it, and no experiment in this paper tests it; it is retained because it follows from the centroidal-momentum structure and states precisely which cross-port term is genuine and which double-counts.

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

Here $w_k$ drives the disturbance random walk. The observer sees only the port error, in which a matched realization residual is indistinguishable from an external disturbance, so its estimate converges to the **effective** disturbance $d^{\rm eff}=d+r_{\rm matched}$ of Section IV without needing to separate the two. Offset-free regulation requires detectability, convergence for a constant disturbance, and feasibility of the cancelling input.

The feedforward and recovery terms need generalized velocity, which on hardware must come from a filtered estimate (a low-order low-pass or observer-based differentiator), not raw encoder differencing; the high rate keeps the phase lag non-critical, and residual lag and noise are absorbed by the disturbance state and residuals.

A contact event creates an innovation because the assumed recovery set no longer matches the plant. This motivates a detector based on normalized innovation statistics:

$$
\eta_k=\nu_k^\top S_k^{-1}\nu_k.
\tag{25}
$$

A mode change is declared only after $\eta_k$ exceeds a calibrated threshold for $n_d$ consecutive samples and a candidate contact is geometrically plausible. This is safer than claiming that the aggregate disturbance alone uniquely identifies a particular foot: without additional kinematic information, different external wrenches can be indistinguishable at the centroidal port.

**Status.** This detector is *derived, not evaluated* in the present paper. The support transition of Section X is gated on the mapped authority set and a kinematic readiness test, not on the innovation statistic (25).

---

## IX. Relation to the Fixed-Base Theory

The normalized predictor, nominal offset-free regulation, and impedance interpretation are taken directly from [1] and are not re-proved here. The new issue is whether the whole-body realizer can communicate useful authority to the predictors, from the solve it is already performing, without changing the canonical dynamics.

**Proposition 1 (local affine recovery and mapped authority).** Fix the current state, references and contact mode, and let $z_0$ solve the realizer (14b) with active set $\mathcal A_k$. Suppose the active constraint gradients $C_{\mathcal A_k}$ have full row rank (LICQ), strict complementarity holds, and $P$ is positive definite on the null space of $C_{\mathcal A_k}$. Then there is a neighbourhood of $u=0$ — the *critical region* of $\mathcal A_k$ — on which: (i) $z_k^\star(u)=z_0+Ku$ exactly, with $K$ the unique solution of the KKT system (14c); and (ii) the actuator, friction-pyramid and unilateral-force limits of (14b), together with the linearized realization-tolerance test, are equivalent to the polyhedron $H_ku\le h_k$ of (14e) under the maps (14d), so every $u\in\widehat{\mathcal U}_k$ that remains inside the critical region is realized with componentwise residual at most $\epsilon_r$.

**Proof.** Under LICQ and strict complementarity the active set is locally constant, so on that neighbourhood (14b) reduces to an equality-constrained quadratic program whose optimality conditions are the linear system in (14c). Since $u$ enters only the linear cost term $Q_u u$ and $P$ does not depend on $u$, the right-hand side is affine in $u$ and the solution map is affine with Jacobian $K$, giving (i). Substituting $\tau=\tau_{\rm ff}+K_\tau u$ and $\lambda=\lambda_{\rm ff}+K_\lambda u$ into the affine constraint rows, and $z_0+Ku$ into the residual expression, gives (ii). $\square$

**Remark (what Proposition 1 does *not* give).** The guarantee is conditional on remaining in the critical region, which the predictor cannot verify. Outside it the true QP changes active set and the map bends, so $\widehat{\mathcal U}_k$ is an inner *model*, not a certified inner approximation, and no false-positive-free claim follows by construction. What can be said unconditionally is weaker and comes from the realizer, not the map: because (14b) enforces the actuator and contact limits as hard constraints, **no** command — inside or outside $\widehat{\mathcal U}_k$ — can drive the robot past a physical limit; an inadmissible command is paid for in realization residual, not in constraint violation. The conservatism of the map is therefore an empirical quantity, measured against an exact oracle in Section X-E.

**Remark (constraint-feasible does not mean faithfully realized).** Satisfying every torque and contact limit does not imply $r=0$. The realizer trades the request against balance, posture and regularization objectives, so a request can be physically feasible and still be served poorly — which is precisely why the tolerance rows appear in (14e), and why a task port whose objective is weighted too softly has a large nominal residual and an empty authority set (Section X-G).

**Proposition 2 (exactness of the continuation).** Suppose that along the ray $u=t\,e$ the QP (14b) satisfies LICQ and strict complementarity at every point except finitely many breakpoints, and that no two breakpoints coincide. Then the piecewise-affine walk of Section IV-B reproduces $z_k^\star(t\,e)$ exactly for all $t$ up to the tolerance crossing, and the returned boundary is the exact boundary of $\{t:\lVert r_k(te)\rVert_\infty\le\epsilon_r\}$.

**Proof sketch.** By Proposition 1 the solution is affine on each critical region, so it suffices to locate the region boundaries and continue across them. A boundary occurs exactly when an inactive row becomes active or an active multiplier vanishes; both are affine in $t$ on the region and are therefore solved in closed form from (14c), whose primal block gives $\mathrm dz/\mathrm du$ and whose dual block gives $\mathrm d\nu/\mathrm du$. Updating $\mathcal A_k$ at the first such breakpoint and re-solving (14c) yields the affine piece on the adjacent region. The residual is affine on each piece, so its tolerance crossing is exact. $\square$

**Remark (what remains empirical).** Degeneracy, coincident breakpoints, and the inexactness of the solver's duals can all corrupt the active-set identification and hence the walk; the guarantee above assumes them away. The measured false-positive and false-negative rates of Section X-E, both zero, are therefore evidence and not proof, and the realizer stays the hard layer regardless. The walk is also a *ray* construction: independent axis boundaries need not have feasible Cartesian corners, and a rigorous box would additionally walk the diagonals.

**Proposition 3 (conditional offset-free regulation).** Fix a contact mode and suppose the realization residual is *constant and matched* on the operating cell, so that the effective disturbance $d^{\rm eff}=d+r_{\rm matched}$ is constant. If the augmented observer is detectable and $\hat d\to d^{\rm eff}$, and if the cancelling request $u=-d^{\rm eff}$ lies in $\widehat{\mathcal U}_k$, then the regulated port reaches zero steady-state error under the fixed-base result of [1] — note that the residual is *cancelled along with* the disturbance, so $r=0$ is not required, only that $r$ be matched and constant. Two failure modes remain, and both are observed in Section X: an unmatched or state-dependent residual cannot be cancelled by any input and yields only bounded regulation (Proposition 4); and when $-d^{\rm eff}$ lies outside the admissible set — as it does when the disturbance exceeds the port's authority — no offset claim is made and the port saturates.

**Assumption 2 (nominal ISS).** The nominal requested-model closed loop of [1] — the predictor $\ddot e=u+d$ under the offset-free feedback of Section VIII — is input-to-state stable with respect to its disturbance input.

**Proposition 4 (bounded authority mismatch).** Suppose Assumption 2 holds and that critical-region exit, snapshot staleness or inter-sample state motion leave a uniformly bounded realization residual, $\sup_k\|r_k\|\le\varepsilon$. Then the realized augmented error state is ultimately bounded by the nominal class-$\mathcal{KL}$ transient plus a class-$\mathcal K$ gain on $\varepsilon$. This result does not certify $\varepsilon$; it states how an authority mismatch propagates once the realizer has logged it.

**Imperfect feedforward on a floating base.** The exact-cancellation premise of Assumption 1 is never met exactly on the robot: arm–base reaction, configuration-dependent bias, and velocity-estimate error all leave a feedforward mismatch. This mismatch splits along the same $d$/$r$ line — its matched, slowly varying part enters the model disturbance $d_{\rm model}$ of (19) and is cancelled by the augmented observer of Section VIII, while the unmatched, feasibility-limited part is absorbed into the bounded residual $r$ and governed by Proposition 4. We therefore claim no perfect nonlinear cancellation; robustness to imperfect feedforward is exactly the ISS/ultimate-boundedness statement above, closed by observer feedback.

**Standing assumptions.** Beyond Assumptions 1–2, several conditions must be checked independently: $A_G$ and $\Lambda_t$ remain finite and well-conditioned; the nominal whole-body problem is feasible; the active set satisfies LICQ and strict complementarity so that Proposition 1 applies; the admissible set is nonempty; and state, contact, and reference estimates remain representative while it is used. The body rotational channel is regulated in angular-momentum coordinates, so a desired attitude still requires an outer regulator and local orientation chart. Citing [1] or keeping $(A,B)$ constant does not prove recursive feasibility of the contact-constrained controller.

Two gaps remain open. First, constant $(A,B)$ removes dynamics switching but not constraint switching; recursive feasibility across scheduled sets still requires compatible terminal ingredients, a robust invariant construction, or another switching certificate. Taking the intersection of adjacent-mode boxes is not such a proof. Second, body and task authority are coupled. Separate predictor boxes require an explicit allocation rule; otherwise individually feasible requests can be jointly infeasible. The implementation uses a body-priority allocation, while a coupled authority set remains the principled extension.

---

## X. Multirate Architecture and Evaluation

### A. The Three Rates

The whole-body QP costs $\approx2$ ms, so it cannot close a 1 kHz loop; the rates are separated by what each object actually costs:

- **1 kHz — joint servo.** Holds the most recently optimized torque (zero-order hold) and adds an optional joint impedance, $\tau=\tau^\star+K_q(q_d-q)+D_q(\dot q_d-\dot q)$. It performs no optimization and no model update; this is the only loop that must truly close at 1 kHz, and it does.
- **200 Hz — optimization node.** One real-time thread, executed *sequentially* so both predictors and the realizer see the same synchronized state: read $(q,\dot q,\text{contacts})$; update $M,h,J_c,J_t$ and the nominal feedforward; run the observers; solve the body predictor; solve the task predictor on the capacity the body left, $H_tu_t\le h_t-H_{tb}u_b^\star$; solve **exactly one** whole-body QP (14b); publish $\tau^\star$ to the servo, and the admissible set to the next update. The two predictors are functionally distinct — the body port predicts centroidal interaction, the task port hand interaction — but they share one state read and one model update, and the dependency is explicit, $u_b(k)\to u_t(k)\to\tau(k)$, not two asynchronous consumers of stale data.
- **$\approx50$ Hz — authority refresh.** The tight authority estimate of Section X-B costs $\approx14$ ms and does not fit the node; it is recomputed off the critical path. The node uses the most recent set and falls back to a conservative fixed box when it is stale or was taken in a different contact mode.

The whole-body QP remains the final hard constraint layer at 200 Hz; the servo never violates a limit because it only holds and tracks a command the node already made feasible.

### B. Authority from One KKT Solve

The residual command $u$ enters the whole-body QP only through objective *linear* terms, and its Hessian $P$ of (14b) does not depend on $u$. On the current active-set cell the solution is therefore affine,

$$
z(u)=z_0+Ku,\qquad
\begin{bmatrix}P & A_{\rm act}^\top\\ A_{\rm act} & 0\end{bmatrix}
\begin{bmatrix}K\\ \ast\end{bmatrix}
=\begin{bmatrix}-\,\partial q/\partial u\\ 0\end{bmatrix},
\tag{26}
$$

so one KKT solve yields $\tau=\tau_{\rm ff}+K_\tau u$ and $\lambda=\lambda_{\rm ff}+K_\lambda u$ from the cycle the realizer is running anyway. Substituting these into the limits the realizer already enforces — actuator bounds, friction pyramid, unilateral normal force — gives

$$
H_k u\le h_k .
\tag{27}
$$

A row with $h_{k,i}<0$ means the nominal solution already sits past that margin; the row is clamped at zero ("no room to push further this way"), which keeps $u=0$ admissible so the set is never infeasible. **The map is exact only on the current active-set cell.** Once a constraint activates the true QP redistributes and the map bends, so (27) is a local model, not a certificate — which is precisely why the realizer stays the hard layer and why the mapping is graded offline below.

### C. Real-Time Budget (E1)

The optimization node solves **one** whole-body QP per update (measured maximum: 1) and fits the $5$ ms budget of its $200$ Hz rate, while the servo runs $3000$ ticks to the node's $600$ updates. On a representative controller (nominal gains, horizon 10, both ports active):

| quantity | value |
|---|---:|
| node median | $3.0$ ms |
| node p99 | $3.9$ ms |
| node budget ($200$ Hz) | $5.0$ ms |
| deadline misses | $0.2\%$ |
| — whole-body QP | $2.0$ ms |
| — authority (single-cell KKT) | $0.60$ ms |
| — body predictor | $0.26$ ms |
| — task predictor | $0.11$ ms |

The whole-body QP dominates, but only $\approx0.3$ ms of its $2.0$ ms is the OSQP solve; the remainder is Python matrix assembly, which a compiled realizer removes. We therefore report a node that fits $200$ Hz in the prototype (and with wide margin compiled) and a $1$ kHz servo met exactly because it does no optimization; we do **not** claim the whole-body QP runs at $1$ kHz.

Two prototype details matter. The realizer QP is solved to $\varepsilon=10^{-8}$: the active set is read from its duals with a $10^{-6}$ bound test, which at the looser $10^{-4}$ default sits below the solver's noise floor and flips with the warm start; tightening it makes the identification warm-start independent at no measurable cost. And the admissible polytope is pruned of rows that cannot bind ($h_i>\lVert H_i\rVert_1 u_{\max}$), which is exact and roughly halves the predictor's solve. Even so, near a constraint boundary the identified set can differ from the exact one, so the predictor falls back to a conservative box on a failed or singular solve, and $\widehat{\mathcal U}_k$ is treated as a model with the realizer as the hard layer.

### D. Feedforward and Contact Mode Consume Authority (E3)

Across payload, arm reference, commanded acceleration and support mode, the canonical pair $(A,B)$ and the predictor's condensed cost Hessian $\Psi$ are **bitwise unchanged**, while the **constraint** data $(H_k,h_k)$ move (Table I). The last column of the table reports $(A,B)$ and $\Psi$ — the *cost* matrix — not the constraint matrix $H_k$, which is exactly what the first two columns show changing. A $5$ kg payload widens the reach; a commanded $0.8$ m/s$^2$ forward acceleration roughly halves it, cutting forward reach from $0.664$ to $0.291$ m/s$^2$; and unprepared single support yields an **empty** admissible set, its nominal realization residual ($0.51$ m/s$^2$) already exceeding the $0.35$ m/s$^2$ tolerance so no residual command is realizable there (the readiness gate of Section X-F). This is the thesis made measurable: *the feedforward spends the actuator and contact budget, and what remains is the predictor's admissible set — not its dynamics.*

| Condition | $u_x$ reach | $u_y$ reach | $(A,B),\Psi$ |
|---|---|---|---|
| double support, nominal | $[-1.43,\,0.66]$ | $[-0.91,\,0.89]$ | unchanged |
| $+5$ kg hand payload | $[-1.67,\,0.69]$ | $[-0.94,\,0.94]$ | unchanged |
| extended-arm reference | $[-1.43,\,0.67]$ | $[-0.91,\,0.89]$ | unchanged |
| commanded $0.8$ m/s$^2$ | $[0.00,\,0.29]$ | $[0.00,\,1.60]$ | unchanged |
| left single support | **empty** | **empty** | unchanged |

**Table I.** Realization authority moves with state, feedforward and contact mode; the canonical predictor does not.

### E. Set Fidelity: One Cell versus Continuation (E5)

Both estimators are graded against an **offline oracle** — the exact realizer query, which bisects each signed coordinate ray on the measured acceleration residual and validates the corners, at $\approx62$ whole-body QP solves per query. It is a measurement procedure, not part of the feedback loop. On a $21\times21$ grid over $[-3,3]^2$:

| estimator | false positives | false negatives | time | whole-body QP solves |
|---|---:|---:|---:|---:|
| single-cell map (14e) | $0.0\%$ | $68.2\%$ | $0.6$ ms | **0** |
| **PWA continuation** | $\mathbf{0.0\%}$ | $\mathbf{0.0\%}$ | $15.8$ ms | **0** (97 KKT) |
| exact oracle | — | — | $151$ ms | $62$ |

**Table II.** Authority-set fidelity. The single-cell map is sound but refuses two thirds of the feasible set; continuation recovers all of it without solving a single extra QP. With a $5$ kg payload the pattern is identical ($68.2\to70.5\%$ false negatives for one cell, $0.0\%$ for continuation).

The single-cell map has **no observed false positives** but is severely conservative, confined to one critical region; continuation reproduces the oracle's ray boundaries to $0.11$ m/s$^2$ (the residual gap is the oracle's own $2.3\%$ corner shrink, not a walk error). Soundness is empirical, not proved — Proposition 1 is conditional — and what holds unconditionally is that the realizer's hard constraints make a limit violation impossible regardless: an inadmissible request is paid in realization residual, not constraint violation.

### F. Conservatism at the Limit of Capability (E2, E4)

In **double support** under a deliberately aggressive predictor, the mapped constraint keeps the request inside what the contacts can produce: median realization residual falls from $1.213$ to $0.560$ m/s$^2$ and the maximum from $5.735$ to $2.834$, while a fixed $\pm4$ box lets the predictor demand accelerations no contact force can deliver. The price is tracking: planar RMS error rises from $42.1$ to $46.8$ mm, because commands the realizer cannot honor are refused rather than issued.

**At the limit of capability the difference between the two estimators is decisive.** Balancing on one foot requires nearly all available authority; a set that withholds two thirds of it cannot stabilize. On the scripted, authority-gated double–single–double support transition (4 seeds each):

| authority source | falls | query time | extra whole-body QPs | canonical $(A,B)$ |
|---|---:|---:|---:|---|
| single-cell map (14e) | $4/4$ | $0.15$ ms | $0$ | invariant |
| **PWA continuation** | $\mathbf{0/4}$ | $13.6$ ms | $\mathbf{0}$ | invariant |
| exact oracle | $0/4$ | $127$ ms | $65$ | invariant |

**Table III.** Support transition. Continuation attains the oracle's outcome at roughly one tenth its cost and without a single extra QP solve.

The single-cell map fails not because the representation is wrong — the canonical matrices are invariant in all three runs — but because it answers a narrower question than the predictor asks; continuation answers the right one with the factorization the realizer already computes. At $13.6$ ms it is refreshed asynchronously off the $200$ Hz critical path (every $20$ ms here), the node meanwhile solving one QP per update and falling back to a conservative box on a stale set.

### G. The Task Port (E6)

The task port is exercised in closed loop at 200 Hz on the capacity the body did not spend, under a sustained $5$ N lateral hand force. The force is chosen so the cancelling command lies inside the port's authority: with $\Lambda_{t,y}\approx1$ kg it is a $\approx5$ m/s$^2$ disturbance. (A $12$ N force needs $\approx12$ m/s$^2$, beyond the command box; the port then saturates and cannot reject it — a limit of the port, and exactly what Proposition 3 predicts.)

**A task objective must be weighted hard enough to be a port at all.** The realizer's hand objective is a soft stand-in for the hard task row of (22). The contact-consistent task inertia spans $30\times$ across the hand axes ($\Lambda_t\approx0.4$ kg in $x$, $12.5$ kg in $z$), so a small scalar weight starves the heavy vertical axis. At the body-only weight the nominal hand residual is $1.834$ m/s$^2$ and the task authority set is **empty** — $\ddot e_t=u_t+d$ is not a usable model of the hand. Raising the weight repairs it, and the body port is untouched:

| hand-objective weight | nominal task residual | task set | body (planar) residual |
|---:|---:|---|---:|
| $6$ (body-only) | $1.834$ | **empty** | $0.0016$ |
| $80$ | $0.228$ | valid | $0.0018$ |
| $2\times10^3$ | $0.064$ | valid | $0.0017$ |
| $8\times10^3$ | $0.021$ | valid | $0.0017$ |

**Table IV.** The task port exists only above a weight threshold; the body port is unaffected.

**Offset-free rejection, and the same estimator dependence.** With the observer the hand rejects the sustained force offset-free. Whether the *mapped* authority permits it depends entirely on which estimator supplies it:

| body authority | observer | hand steady-state | CoM steady-state |
|---|---|---:|---:|
| fixed box | off | $364.5$ mm | $4.7$ mm |
| fixed box | **on** | $\mathbf{26.0}$ **mm** | $3.2$ mm |
| single-cell map | on | $502.6$ mm | $148.2$ mm |
| **PWA continuation** | **on** | $\mathbf{26.0}$ **mm** | $\mathbf{3.2}$ **mm** |

**Table V.** Task-port rejection of a sustained $5$ N hand force. The $14\times$ offset reduction survives intact under continuation authority; the single-cell map destroys it, and disturbs the CoM as a side effect.

Both ports run on one whole-body QP per node update, the task authority reusing the same KKT sensitivity at no extra solve. So the single-active-set restriction is not a peculiarity of single-support balance: it fails at *both* ports, and the same continuation repairs *both*.

---

## XI. Limitations

Neither authority estimator is certified. The single-cell map is exact only on the current critical region (Proposition 1); the continuation is exact under LICQ, strict complementarity and non-coincident breakpoints (Proposition 2), none verified online. Both read an active set from the duals of an approximate solve, which can be wrong near a boundary or under degeneracy, so the measured zero false-positive rates are evidence, not proof. The continuation is moreover a ray construction: independent axis boundaries need not have feasible Cartesian corners. The instantaneous QP therefore remains the final hard layer — what makes a limit violation impossible regardless of the map — and no recursive-feasibility claim is made under contact switching.

The rates are a Python prototype, not hardware. The $200$ Hz node fits its $5$ ms budget, but the whole-body QP that dominates it is $2.0$ ms of which only $\approx0.3$ ms is the OSQP solve; a compiled realizer is needed before any hardware timing claim. The continuation walk ($\approx14$ ms) is refreshed asynchronously and would want a factorization update per active-set change rather than a fresh KKT solve. The structure is decimated in simulation, not threaded, so jitter, lock-free publication and real-time misses are untested.

The task port uses a fixed body-priority allocation — a policy, not a joint body–task optimum — and its hand-objective weight is a per-use design parameter (a value serving the task port destabilises the support transition, where the hand task is inert). The angular-momentum body channel is derived and structurally verified but not exercised in closed loop; single support is only marginally stable irrespective of the authority source; and the evaluation is one simulated G1 in MuJoCo, with no actuator dynamics, delay, contact-model mismatch, or locomotion.

---

## XII. Conclusion

This paper anchors the interaction-dynamics representation on the rate at which each object must run. A $200$ Hz optimization node solves both canonical predictors and a single whole-body QP per update, and a $1$ kHz servo holds the optimized torque between updates; the predictors carry an exact-ZOH pair $(A,B)$ invariant to configuration and contact mode. Robot mechanics re-enter not as a second predictive model but as the admissible geometry of the residual command: because that command enters the whole-body QP only through linear cost terms, the KKT system of the solve the realizer is *already performing* returns the feedforward, the input maps, and hence the constraints $H_ku\le h_k$. Contact transitions move that geometry and leave the canonical dynamics untouched.

The accuracy of that capability query was the entire difficulty, and also the solution. Restricted to one active set the query is sound but describes the QP's *critical region*, not what the robot can do — enough to fall in $4/4$ single-support transitions and to break the task port. Yet an active-set change is not a failure of realization; the contact forces simply redistribute. Walking the solution across those regions, with entering and leaving constraints in closed form from the primal and dual blocks of the same KKT system, recovers the exact feasible set with **zero** additional whole-body QP solves. The capability query is therefore not an obstacle to the prediction–realization contract but a by-product of it: the realizer already knows what it can do, and one dual block is enough to make it say so.

What remains is engineering and rigour — a compiled node, factorization updates per active-set change, guarantees under degeneracy, a jointly-allocated body–task set, and hardware. But the interface is agnostic to how the reference is produced: a planner, a learned policy, or a world model may issue residual interaction requests in coordinates that do not depend on the robot's mass, inertia, friction cones or contact state, while the realizer alone reports which requests it can currently honor and turns the chosen one into contact forces and torques. That is the contract this paper set out to make precise.

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
