Yongyan Cao
Voryx Robotic LLC
San Jose, CA 95136, USA
yongyancao@gmail.com

June 12, 2026

To the Editor-in-Chief and Associate Editors,
*IEEE Transactions on Robotics*

**Re: Submission of original research manuscript**

Dear Editors,

We are pleased to submit our manuscript, **"Whole-Body Impedance Model
Predictive Control for Safe Physical Human–Robot Interaction on Floating-Base
Platforms,"** by Yongyan Cao, for consideration as a regular paper in *IEEE
Transactions on Robotics*.

**Problem.** Floating-base robots — humanoids, quadrupeds, legged
manipulators — must balance under rigid contact constraints while interacting
safely with humans. The dominant whole-body control (WBC) frameworks allocate
the entire joint space to locomotion and treat arm interaction as a
disturbance to suppress, while fixed-base impedance MPC methods assume an
infinite-mass ground connection and fail under contact transitions. Neither
provides the compliance and zero-steady-state-error guarantees required for
safe physical human–robot interaction (pHRI) on a balancing platform.

**Approach and contributions.** We propose a three-level Whole-Body Impedance
MPC — centroidal MPC for contact forces, a priority-driven WBC hierarchy for
balance, and a receding-horizon impedance QP in the residual null space:

1. *Contact-consistent residual plant.* After priority-driven feedforward
   cancellation, the floating-base arm end-effector dynamics reduce exactly to
   a double integrator with a *constant* discrete state matrix within each
   contact mode; the sole change from the fixed-base case is the
   contact-consistent mass inverse in the operational-space inertia. The QP
   cost inverse is therefore precomputed offline, enabling ≥1 kHz operation.
2. *Contact-mode-indexed MPC.* The input matrix is selected from a precomputed
   per-contact-mode library, preserving the convex QP structure across
   stance-phase changes.
3. *Kalman disturbance isolation.* An augmented Kalman state jointly estimates
   pHRI forces, leg-momentum variations, and model-approximation errors, and
   propagates the estimate through the full prediction horizon, guaranteeing
   zero steady-state tracking error under any bounded constant pHRI load.
4. *Contact-transition protocol.* A covariance-inflation rule preserves the
   disturbance estimate across contact switches, with explicit transient
   error bounds.
5. *Impedance Equivalence Theorem.* The infinite-horizon limit provably
   recovers a classical task-space impedance law whose effective mass,
   damping, and stiffness all adapt to posture and contact configuration
   through the contact-consistent inertia — with no online re-optimization of
   impedance parameters.

**Key results.** In MuJoCo benchmarks against six alternative controllers
(classical baselines and ablations): on a 17-DOF biped
under a sustained 8 N pHRI force, the full controller attains 0.037 mm
steady-state error versus 10.17 mm for the hierarchical PD baseline (a 273×
reduction); replacing the contact-consistent inertia with the free-space
inverse degrades the result 318×, confirming contact consistency is not
optional. Under periodic contact-transition shocks, peak deflection drops from
15.8 mm to 4.2 mm. The architecture transfers to the official Unitree G1
humanoid model (29 DOF), preserving the controller ordering with a 2.5×
improvement through the position-actuator interface.

**Relation to prior work (disclosure).** This manuscript builds on the base
framework "Impedance MPC for Physical Human–Robot Interaction: Predictive
Disturbance Rejection with Joint-Limit Safety" (arXiv:2606.08281, currently
under review at *IEEE Transactions on Robotics*). The base paper treats a
fixed-base redundant manipulator; the present manuscript contains the
floating-base extension only — contact-consistent dynamics, the
contact-mode-indexed input library, contact-transition handling, the
whole-body Impedance Equivalence Theorem, and all floating-base benchmarks.
The base paper is cited in third person, and there is no overlapping text or
duplicated result between the two manuscripts. A separate letter applying the
same base framework to dexterous hands is under review at *IEEE Robotics and
Automation Letters*; it shares no content with the present manuscript.

**Compliance.** The manuscript is 9 pages, prepared in the IEEEtran journal
format, and anonymized for double-anonymous review (author block and
identifying information removed; references are not anonymized, per the IEEE
RAS guidelines).

**Originality and ethics.** This manuscript is original, has not been
published previously, and is not under consideration for publication
elsewhere. The author has approved the submission and agrees to its content.
The work does not involve human or animal subjects. The author declares no
conflicts of interest.

**Suggested topic areas.** Whole-body motion planning and control; humanoid
and legged robots; compliance and impedance control; model predictive and
optimal control; physical human–robot interaction.

We thank you for your time and look forward to the reviewers' feedback.

Sincerely,

Yongyan Cao (corresponding author)
Voryx Robotic LLC — yongyancao@gmail.com
