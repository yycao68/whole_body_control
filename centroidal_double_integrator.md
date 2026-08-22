# Centroidal Interaction Dynamics: The Floating-Base Case Study

*Reframed WBC core. The scientific object is not a new controller but a demonstration that the **canonical centroidal model of legged robotics is an instance of the configuration-invariant interaction-dynamics representation** established for fixed-base pHRI. The generic results (offset-free regulation, workspace/contact-mode stability via a common-$P$ LMI, impedance-as-a-limit) are proved once in the base paper [pHRI]; here they are instantiated, not re-proved. The one genuinely new element is that the interaction-disturbance state, besides giving offset-free centroidal regulation, **detects foot touchdown/liftoff without a contact sensor** — so contact-mode switching is observed rather than asserted.*

---

## 1. The unifying principle: physical input = model feedforward + predictive residual

Every layer of the framework has the same two-part control law and the same normalized error dynamics:

$$
\underbrace{(\text{physical input})}_{\text{force / GRF / torque}}
=\underbrace{(\text{model-based feedforward})}_{\text{cancels known dynamics, injects reference}}
+\underbrace{(\text{inertia})\times u}_{\text{predictive residual}},
\qquad
\boxed{\;\ddot e = u + d\;}
\tag{1}
$$

where $u$ is the **residual (interaction) acceleration** — the MPC decision variable — and $d$ the estimated interaction disturbance. The inertia map ($\Lambda$, $m$, $I_G$) carries $u$ back to a physical input; **all configuration and contact dependence lives in that map and in the input constraints, never in the normalized error dynamics.** For the fixed-base arm this is the operational-space law $F=\Lambda(q)(\ddot x_d-u)+\mu(q,\dot q)$ of [pHRI]. Below we show the centroidal layer has exactly this form.

---

## 2. Canonical centroidal model

For a floating-base robot of mass $m$, CoM $c$, angular momentum $k$ about the CoM, and contacts $i$ at $p_i$ with forces $f_i$ (Orin–Goswami–Lee):

$$
m\ddot c=\textstyle\sum_i f_i+mg+d_c,\qquad
\dot k=\textstyle\sum_i (p_i-c)\times f_i+\sum_i\tau_i+d_k,\qquad
k=I_G(q)\,\omega_G,
\tag{2}
$$

with $g=[0,0,-g_0]^\top$, CCRBI $I_G(q)$, and $d_c,d_k$ the lumped external/model disturbances. Equations (2) are the standard centroidal / single-rigid-body model of legged MPC — taken as given.

---

## 3. Linear channel

Let $e_c=c-c_d$. From (2), $\ddot e_c=\tfrac1m\sum_i f_i+g-\ddot c_d+\tfrac1m d_c$. Choose the GRF resultant as a **model feedforward plus a residual acceleration** $u_c$:

$$
\textstyle\sum_i f_i \;=\; m(\ddot c_d-g)\;+\;m\,u_c .
\tag{3}
$$

Substituting cancels gravity and the reference term exactly:

$$
\boxed{\;\ddot e_c = u_c + d_c',\qquad d_c'=\tfrac1m d_c\;}
\tag{4}
$$

a double integrator whose input is the CoM residual acceleration $u_c$. No mass appears in (4): the linear channel is configuration-invariant with a **constant** input map. After the MPC returns $u_c^\star$, the physical resultant is recovered by (3), $\sum_i f_i^\star=m(\ddot c_d-g)+m\,u_c^\star$.

---

## 4. Angular channel

Differentiating $k=I_G\omega_G$ and using (2): $I_G\dot\omega_G = M-\dot I_G\omega_G+d_k$, with net contact moment $M=\sum_i(p_i-c)\times f_i+\sum_i\tau_i$. Choose

$$
M=\underbrace{\dot I_G(q)\,\omega_G+I_G(q)\,\dot\omega_{G,d}}_{\text{model feedforward}}+\underbrace{I_G(q)\,u_\theta}_{\text{predictive residual}} .
\tag{5}
$$

With $e_\theta=\log(RR_d^\top)^\vee$ and $\ddot e_\theta\approx\dot\omega_G-\dot\omega_{G,d}$,

$$
\boxed{\;\ddot e_\theta = u_\theta + d_\theta',\qquad d_\theta'=I_G^{-1}d_k\;}
\tag{6}
$$

Again a double integrator with a **constant** input map; the CCRBI $I_G(q)$ — the centroidal analog of the operational-space inertia $\Lambda(q)$ — appears only in the feedforward (5) and the moment recovery.

---

## 5. The centroidal interaction-dynamics model

Stack $x=[e_c;\dot e_c;e_\theta;\dot e_\theta]\in\mathbb R^{12}$ and residual acceleration $u=[u_c;u_\theta]\in\mathbb R^{6}$. From (4),(6), $\ddot e=u+d$ per channel; exact ZOH ($A_c^2=0$) gives, **constant across all configurations and contact modes**,

$$
x_{k+1}=A_d\,x_k+B_d\,u_k+B_d\,d_k',\quad
A_d=\begin{bmatrix}I&\Delta t I\\0&I\end{bmatrix}\!\otimes\! I_2,\quad
B_d=\begin{bmatrix}\tfrac{\Delta t^2}{2}I\\ \Delta t\,I\end{bmatrix}\!\otimes\! I_2 .
\tag{7}
$$

Because $u$ is an *acceleration*, the control and the disturbance enter through the **same** $B_d$ — the model is fully configuration-invariant, and identical in form to the fixed-base backbone of [pHRI]. This is the proposition the case study rests on:

> **Instantiation (no new proof).** The canonical centroidal model (2), under feedforwards (3),(5), is a realization of the configuration-invariant interaction-dynamics model $x_{k+1}=A_d x_k+B_d(u_k+d_k)$ with the *same* constant $(A_d,B_d)$ as [pHRI]. Configuration and contact dependence are confined to the feedforward inertias $(m,I_G(q))$ and to the input constraints of §6 — not to $(A_d,B_d)$.

---

## 6. Contact-consistent recovery and constraints

The residual $u^\star$ is realized by per-contact forces. Writing $\mathbf f=[f_1;\dots;f_{n_c}]$, the recovery + wrench-consistency conditions are

$$
S_f\,\mathbf f = m(\ddot c_d-g)+m\,u_c,\qquad
G_\tau(\rho)\,\mathbf f = \dot I_G\omega_G+I_G(\dot\omega_{G,d}+u_\theta),
$$

$S_f=[I\ \cdots\ I]$, $G_\tau(\rho)=[\cdots\,(p_i-c)^\wedge\,\cdots]$, subject to friction cones $f_i\in\mathcal{FC}_i$, unilaterality $f_{i,z}\ge0$, and CoP-in-support. These linear/SOC inequalities are the floating-base analog of the arm's actuator/joint limits and enter the same QP. The **contact mode $\rho_k$** selects which contacts are active — i.e. it changes $\{G_\tau,\mathcal{FC}_i\}$, the *constraint side*, while $(A_d,B_d)$ stay fixed. Because $A_d,B_d$ are constant, the prediction matrices are precomputed once; only the contact-constraint rows refresh per mode.

---

## 7. Offset-free regulation and stability — inherited from [pHRI]

Augment (7) with an integrating disturbance $\hat d$ (random walk), estimated by a steady-state Kalman filter from CoM/orientation (and momentum, when available):

$$
\begin{bmatrix}x_{k+1}\\\hat d_{k+1}\end{bmatrix}
=\begin{bmatrix}A_d&B_d\\0&I\end{bmatrix}
\begin{bmatrix}x_k\\\hat d_k\end{bmatrix}+\begin{bmatrix}B_d\\0\end{bmatrix}u_k .
\tag{8}
$$

Since (8) is the *identical* augmented model of [pHRI] with the same constant $(A_d,B_d)$, its results apply verbatim — **we state them and cite the proofs there rather than reproving:**
- **Offset-free centroidal regulation** ([pHRI, Thm 2]): under a constant unmodeled centroidal wrench (sustained push, unmeasured payload) the CoM/orientation error goes to zero.
- **Workspace/contact-mode stability** ([pHRI, Thm 3]): as $I_G^{-1}(q)$ over the workspace and the finite contact-mode set lie in a compact polytope, a *single* parameter-independent $P$ certifies exponential stability across all configurations and modes via one vertex LMI.
- **Impedance/PD as a corollary** ([pHRI, Thm 1]): the unconstrained infinite-horizon law is a static centroidal PD/impedance feedback — recovered as a limit, not claimed as a theorem.

This is the point of the reframing: the theory is the invariance class; the centroidal layer is one member, so it inherits rather than re-derives.

---

## 8. New: sensor-free foot-touch detection from the disturbance state

Contact events break the momentary model consistency: at **touchdown** an unmodeled support force appears; at **liftoff** an assumed support force vanishes. Under a *fixed* contact-mode feedforward (3),(5) this mismatch is exactly what the estimator books into $\hat d$. Hence the interaction-disturbance state is also a **contact-event signal**:

$$
\text{touchdown/liftoff at foot } i \;\Longleftrightarrow\;
\big|\,\hat d_k - \hat d_{k}^{(\rho)}\,\big|\ \text{or the Kalman innovation}\ \nu_k\ \text{crosses a threshold},
\tag{9}
$$

where $\hat d^{(\rho)}$ is the disturbance predicted under the current assumed mode. Practically: project $\hat d$ (equivalently the residual GRF $m\hat d_c'$) onto each candidate foot's support normal; a step up flags touchdown, a step down flags liftoff, triggering the mode update $\rho_k\!\to\!\rho_{k+1}$ and the covariance inflation of §... . This makes contact-mode switching **observed, not asserted** — the same disturbance observer that confers offset-free regulation doubles as a proprioceptive contact detector, closing reject point #3 with a positive contribution rather than a caveat.

---

## 9. Positioning / what this buys against the reject

- *#1 "impedance = LQR=PD; not a new theorem":* demoted to a cited corollary (§7). The paper's claim is the **invariance of the canonical centroidal model into the interaction-dynamics class** (§5), with the general theorems inherited from [pHRI].
- *#2 "covariance inflation is classical":* stated as an engineering realization of the contact-mode-indexed estimator, not a theorem.
- *#3 "Scenario B has no contact switch":* replaced by a genuine support-transition experiment in which §8 **detects** the switch from $\hat d$ and the controller re-plans across it.
- *#4 "incremental extension":* the modeled object is now the *standard* centroidal model of legged robotics recast as one case of a theory that also covers fixed-base, tendon-driven, and continuum systems — a unification, not an $M^{-1}$ swap.

**Scope / honesty.** Assumes the CCRBI relation $k=I_G\omega_G$, the small-orientation-error linearization, and availability of $\dot I_G$ for (5) — standard SRBD-level assumptions, all confined to the *known* model that the feedforward cancels; residuals go to $\hat d$. This is the reduced centroidal layer; a whole-body QP resolves joint torques beneath it. The hardware gap (MuJoCo-only) remains and is not closed by reframing.
