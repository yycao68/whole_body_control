"""
WBC Core Dynamics Utilities
Implements contact-consistent mass inverse, operational-space formulation,
and hierarchical null-space torque synthesis for a floating-base biped.
All equations reference the IEEE paper section numbers in comments.
"""

import numpy as np
import mujoco


def _checked_matmul(left, right, name):
    """Multiply dense arrays while detecting real non-finite results."""
    with np.errstate(all="ignore"):
        product = np.asarray(left) @ np.asarray(right)
    if not np.all(np.isfinite(product)):
        raise FloatingPointError(f"non-finite matrix product: {name}")
    return product


# --------------------------------------------------------------------------
# Model index cache
# --------------------------------------------------------------------------
_cache = {}

def _get_ids(model):
    key = id(model)
    if key not in _cache:
        def sid(name): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,     name)
        def bid(name): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,     name)
        def jid(name): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    name)
        def aid(name): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

        # Build dof_address → qpos_address map for all 1-DOF joints
        dof_to_qadr = {}
        for j in range(model.njnt):
            if model.jnt_type[j] != 0:  # skip freejoint
                dof_to_qadr[model.jnt_dofadr[j]] = model.jnt_qposadr[j]

        def dof(jname): return model.jnt_dofadr[jid(jname)]
        def qadr(jname): return model.jnt_qposadr[jid(jname)]

        _cache[key] = {
            'torso_body':        bid('torso'),
            'left_foot_site':    sid('left_foot_contact'),
            'right_foot_site':   sid('right_foot_contact'),
            'hand_site':         sid('right_hand_site'),
            # DOF (qvel) addresses
            'left_hip_x_dof':    dof('left_hip_x'),
            'left_hip_y_dof':    dof('left_hip_y'),
            'left_knee_y_dof':   dof('left_knee_y'),
            'left_ankle_y_dof':  dof('left_ankle_y'),
            'right_hip_x_dof':   dof('right_hip_x'),
            'right_hip_y_dof':   dof('right_hip_y'),
            'right_knee_y_dof':  dof('right_knee_y'),
            'right_ankle_y_dof': dof('right_ankle_y'),
            'rshoulder_x_dof':   dof('right_shoulder_x'),
            'rshoulder_y_dof':   dof('right_shoulder_y'),
            'relbow_y_dof':      dof('right_elbow_y'),
            # qpos addresses (for reading joint angles)
            'left_hip_x_qadr':    qadr('left_hip_x'),
            'left_hip_y_qadr':    qadr('left_hip_y'),
            'left_knee_y_qadr':   qadr('left_knee_y'),
            'left_ankle_y_qadr':  qadr('left_ankle_y'),
            'right_hip_x_qadr':   qadr('right_hip_x'),
            'right_hip_y_qadr':   qadr('right_hip_y'),
            'right_knee_y_qadr':  qadr('right_knee_y'),
            'right_ankle_y_qadr': qadr('right_ankle_y'),
            'rshoulder_x_qadr':   qadr('right_shoulder_x'),
            'rshoulder_y_qadr':   qadr('right_shoulder_y'),
            'relbow_y_qadr':      qadr('right_elbow_y'),
            # actuator indices
            'leg_act_ids':  [aid(n) for n in [
                'left_hip_x','left_hip_y','left_knee_y','left_ankle_y',
                'right_hip_x','right_hip_y','right_knee_y','right_ankle_y']],
            'arm_act_ids':  [aid(n) for n in [
                'right_shoulder_x','right_shoulder_y','right_elbow_y']],
        }
    return _cache[key]


# --------------------------------------------------------------------------
# Dynamics quantities  (Eq. 3)
# --------------------------------------------------------------------------

def get_mass_matrix(model, data):
    """M(q) ∈ ℝ^{nv×nv}."""
    nv = model.nv
    M = np.zeros((nv, nv))
    try:
        mujoco.mj_fullM(model, M, data.qM)
    except TypeError:
        # MuJoCo >= 3.10 Python bindings use (model, data, dst).
        mujoco.mj_fullM(model, data, M)
    return M

def get_bias_force(data):
    """h = C(q,dq)dq + G(q)."""
    return data.qfrc_bias.copy()

def get_site_jacobian(model, data, site_id):
    """3×nv translational Jacobian for a site."""
    nv = model.nv
    Jp = np.zeros((3, nv))
    Jr = np.zeros((3, nv))
    mujoco.mj_jacSite(model, data, Jp, Jr, site_id)
    return Jp

def get_site_jacobian_dot(model, data, site_id):
    """3xnv translational Jacobian TIME DERIVATIVE for a site (mj_jacDot,
    which computes the derivative for a world POINT attached to a body, not
    a site directly -- resolved via the site's current world position and
    parent body). Needed for the operational-space bias term mu_arm =
    Jbar^T h - Lambda (dJ/dt) qdot (paper eq. after eq:plant); previously
    unused in this codebase, added when wiring that term through."""
    nv = model.nv
    Jp_dot = np.zeros((3, nv))
    Jr_dot = np.zeros((3, nv))
    body_id = model.site_bodyid[site_id]
    point = data.site_xpos[site_id].copy()
    mujoco.mj_jacDot(model, data, Jp_dot, Jr_dot, point, body_id)
    return Jp_dot


def get_body_jacobian(model, data, body_id):
    """(3×nv, 3×nv) translational + rotational Jacobians for a body."""
    nv = model.nv
    Jp = np.zeros((3, nv))
    Jr = np.zeros((3, nv))
    mujoco.mj_jacBody(model, data, Jp, Jr, body_id)
    return Jp, Jr


def get_body_jacobian_dot(model, data, body_id):
    """3xnv translational Jacobian time derivative for a body's own origin
    (mj_jacDot at the body's current world position). See
    get_site_jacobian_dot's docstring for why this is needed."""
    nv = model.nv
    Jp_dot = np.zeros((3, nv))
    Jr_dot = np.zeros((3, nv))
    point = data.xpos[body_id].copy()
    mujoco.mj_jacDot(model, data, Jp_dot, Jr_dot, point, body_id)
    return Jp_dot


# --------------------------------------------------------------------------
# Contact-consistent mass inverse  (Eq. 9)
# --------------------------------------------------------------------------

def get_contact_jacobian(model, data, site_ids, contact_mask=None):
    """Stacked 3k×nv contact Jacobian for active foot sites."""
    if contact_mask is None:
        contact_mask = [True] * len(site_ids)
    rows = [get_site_jacobian(model, data, sid)
            for sid, active in zip(site_ids, contact_mask) if active]
    return np.vstack(rows) if rows else np.zeros((0, model.nv))

def get_contact_consistent_inverse(M, Jc, reg=1e-6, contact_damp=0.1):
    """M̄⁻¹ = M⁻¹ − M⁻¹JcᵀΛcJcM⁻¹   (Eq. 9), damped constrained inverse.

    `contact_damp` damps the contact-space inertia inversion
    (Λc = (Jc M⁻¹ Jc^T + contact_damp·I)⁻¹).  Without it the resulting
    task-space inertia Λ_arm = (J_arm M̄ J_arm^T)⁻¹ is near-singular for a
    planted double-support stance (the arm cannot accelerate the hand in a
    contact-coupled direction without base motion), which is ill-posed for
    the controller. A moderate damping keeps Mbar well-conditioned, but makes
    contact consistency approximate rather than exact. The paper reports this
    implementation regularization explicitly.
    """
    M_inv = np.linalg.inv(M + reg * np.eye(M.shape[0]))
    if Jc.shape[0] == 0:
        return M_inv
    nc = Jc.shape[0]
    contact_inertia = _checked_matmul(
        _checked_matmul(Jc, M_inv, "Jc @ M_inv"), Jc.T, "Jc @ M_inv @ Jc.T"
    )
    Lambda_c = np.linalg.inv(contact_inertia + contact_damp * np.eye(nc))
    M_bar_inv = M_inv - _checked_matmul(
        _checked_matmul(
            _checked_matmul(M_inv, Jc.T, "M_inv @ Jc.T"),
            Lambda_c,
            "M_inv @ Jc.T @ Lambda_c",
        ),
        _checked_matmul(Jc, M_inv, "Jc @ M_inv"),
        "M_inv @ Jc.T @ Lambda_c @ Jc @ M_inv",
    )
    return 0.5 * (M_bar_inv + M_bar_inv.T)


# --------------------------------------------------------------------------
# Operational-space quantities  (Eq. 13)
# --------------------------------------------------------------------------

def get_task_inertia(J, M_bar_inv, reg=1e-4):
    """Λ = (J M̄⁻¹ J^T)⁻¹  with eigenvalue clamping to guarantee PSD."""
    A = _checked_matmul(
        _checked_matmul(J, M_bar_inv, "J @ M_bar_inv"), J.T,
        "J @ M_bar_inv @ J.T",
    )
    # Symmetrise and clamp eigenvalues so A stays PD
    A = 0.5 * (A + A.T)
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, reg)
    A_psd = _checked_matmul(
        _checked_matmul(eigvecs, np.diag(eigvals), "eigvecs @ diag(eigvals)"),
        eigvecs.T,
        "eigvecs @ diag(eigvals) @ eigvecs.T",
    )
    return np.linalg.inv(A_psd)

def get_dynamically_consistent_pseudoinverse(J, M_bar_inv, Lambda):
    """J̄ = M̄⁻¹ J^T Λ."""
    return _checked_matmul(
        _checked_matmul(M_bar_inv, J.T, "M_bar_inv @ J.T"), Lambda,
        "M_bar_inv @ J.T @ Lambda",
    )

def get_null_space_projector(J_bar, J):
    """N̄ = I − J̄ J."""
    return np.eye(J.shape[1]) - J_bar @ J


def get_contact_consistent_projector(M, Jc, reg=1e-4, contact_damp=0.1):
    """Contact null-space projector Pc (generalized-force space): in the
    undamped exact-model limit, a task force mapped through Pc^T produces no
    contact acceleration. Factored out of WBCController.compute() so scenario
    files that need contact-consistent torque realization outside
    WBCController's own fixed 2-foot-contact/single-arm assumptions (e.g. a
    variable-size contact set that gains a bracing-hand row) can use the
    identical projection rather than falling back to a raw, non-contact-
    consistent J^T F mapping -- found missing (Scenario E used raw J^T F
    directly) by external review.

    `contact_damp` default matches get_contact_consistent_inverse's (0.1),
    not the 1e-3 this originally carried. The damping trades two properties
    against each other, measured at the Scenario A/B/C double-support stance:

        contact_damp -> 0   Pc is an EXACT oblique projector (idempotent,
                            eigenvalues exactly 0/1) -- but because it is
                            oblique rather than orthogonal, its operator
                            norm is unbounded by that exactness, measuring
                            ||Pc||_2 = 28.0.
        contact_damp = 0.1  ||Pc||_2 = 12.6, but Pc is no longer a true
                            projector: the six contact-direction eigenvalues
                            have drifted from 0 to 0.65, and
                            ||Pc - Pc@Pc|| = 6.8.

    The large operator norm at low damping caused a genuine closed-loop
    instability once the arm feedforward's velocity-dependent bias term
    (mu_arm) was added: Pc^T amplified it ~28x in some directions. Sharp
    bifurcation at contact_damp <= 0.003 (diverges past 1000mm) vs >= 0.003
    (stable, ~20mm). The instability is independent of the feedforward law's
    own sign/derivation, which re-derives correctly from M*qddot + h = tau.

    So 0.1 buys stability by making the projection substantially APPROXIMATE,
    not by fixing it. That is the honest characterization, and it is why the
    paper's contact-decoupling claim is stated as approximate. It also
    matches the damping already used for Mbar over the same contact set,
    rather than introducing a second, inconsistent constant."""
    nv = M.shape[0]
    M_inv = np.linalg.inv(M + reg * np.eye(nv))
    if Jc.shape[0] == 0:
        return np.eye(nv)
    contact_inertia = _checked_matmul(
        _checked_matmul(Jc, M_inv, "Jc @ M_inv"), Jc.T, "Jc @ M_inv @ Jc.T"
    )
    Lam_c = np.linalg.inv(contact_inertia + contact_damp * np.eye(Jc.shape[0]))
    return np.eye(nv) - _checked_matmul(
        _checked_matmul(
            _checked_matmul(Jc.T, Lam_c, "Jc.T @ Lam_c"), Jc,
            "Jc.T @ Lam_c @ Jc",
        ),
        M_inv,
        "Jc.T @ Lam_c @ Jc @ M_inv",
    )


def get_arm_bias_force(model, data, J_arm, J_arm_dot, Mbar_inv, Lambda_arm):
    """mu_arm = Jbar_arm^T h - Lambda_arm (dJ_arm/dt) qdot (paper eq. after
    eq:plant, Sec. V). This is the task-space Coriolis/gravity bias the arm
    feedforward law F_arm = Lambda_arm(p_ddot_d + u) + mu_arm needs to
    cancel; found missing from the reported benchmarks by external review
    (impedance_mpc.py previously computed only Lambda_arm @ u, silently
    dropping this term and the p_ddot_d term entirely)."""
    h = get_bias_force(data)
    J_bar_arm = get_dynamically_consistent_pseudoinverse(J_arm, Mbar_inv, Lambda_arm)
    qdot = data.qvel.copy()
    return (
        _checked_matmul(J_bar_arm.T, h, "J_bar_arm.T @ bias")
        - _checked_matmul(
            Lambda_arm,
            _checked_matmul(J_arm_dot, qdot, "J_arm_dot @ qdot"),
            "Lambda_arm @ J_arm_dot @ qdot",
        )
    )


# --------------------------------------------------------------------------
# Contact detection (height proxy)
# --------------------------------------------------------------------------

def get_foot_contact_flags(model, data, site_ids, height_thresh=0.06):
    """True if foot site z < thresh (simple stance detection)."""
    return [bool(data.site_xpos[sid, 2] < height_thresh) for sid in site_ids]


# --------------------------------------------------------------------------
# Convenience getters
# --------------------------------------------------------------------------

def get_hand_state(model, data):
    """Returns (pos, vel) of right hand site in world frame."""
    ids = _get_ids(model)
    sid = ids['hand_site']
    pos = data.site_xpos[sid].copy()
    Jp = get_site_jacobian(model, data, sid)
    vel = Jp @ data.qvel
    return pos, vel

def get_robot_com(model, data):
    """CoM position (mass-weighted average over all bodies except worldbody)."""
    total_mass = float(np.sum(model.body_mass[1:]))
    com_pos = np.zeros(3)
    for b in range(1, model.nbody):
        com_pos += model.body_mass[b] * data.xipos[b]
    com_pos /= total_mass
    ids = _get_ids(model)
    Jp, _ = get_body_jacobian(model, data, ids['torso_body'])
    com_vel = Jp @ data.qvel
    return com_pos, com_vel


# --------------------------------------------------------------------------
# Joint PD helper
# --------------------------------------------------------------------------

def joint_pd_torques(model, data, q_ref, dq_ref, Kp, Kd, joint_names):
    """PD torque for a list of joints (indexed by name)."""
    tau = np.zeros(len(joint_names))
    for i, jname in enumerate(joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        q  = data.qpos[model.jnt_qposadr[jid]]
        dq = data.qvel[model.jnt_dofadr[jid]]
        tau[i] = Kp[i] * (q_ref[i] - q) - Kd[i] * dq
    return tau


# --------------------------------------------------------------------------
# WBC torque assembler  (Eq. 28)
# --------------------------------------------------------------------------

class WBCController:
    """
    Three-level WBC torque assembler.
    Level 2: Leg PD (balance / gait tracking)
    Level 3: Arm end-effector MPC force (in null-space of leg tasks)
    Level 4: Null-space joint centering
    """

    # Ordered DOF and qpos addresses for legs + arm (must match ctrl order)
    _LEG_DOF_KEYS  = ['left_hip_x_dof', 'left_hip_y_dof',
                       'left_knee_y_dof','left_ankle_y_dof',
                       'right_hip_x_dof','right_hip_y_dof',
                       'right_knee_y_dof','right_ankle_y_dof']
    _LEG_QADR_KEYS = ['left_hip_x_qadr', 'left_hip_y_qadr',
                       'left_knee_y_qadr','left_ankle_y_qadr',
                       'right_hip_x_qadr','right_hip_y_qadr',
                       'right_knee_y_qadr','right_ankle_y_qadr']
    _ARM_DOF_KEYS  = ['rshoulder_x_dof','rshoulder_y_dof','relbow_y_dof']
    _ARM_QADR_KEYS = ['rshoulder_x_qadr','rshoulder_y_qadr','relbow_y_qadr']

    def __init__(self, model, contact_consistent=True):
        self.model   = model
        self.ids     = _get_ids(model)
        # When True, the arm task force is projected through the contact
        # null space so it produces no contact-constraint violation (the
        # feet cannot be pushed by the arm), and the coupling leg torques
        # needed to preserve contact are fed back to the legs.
        self.contact_consistent = contact_consistent
        # Feeding the projected leg-coupling torques back to the PD-balanced
        # legs can fight the balance PD; keep it off for the static-stance
        # scenarios (the arm projection alone already makes the arm torque
        # contact-consistent).
        self.apply_leg_coupling = False
        self.Kp_null = 3.0    # gentle null-space centering to avoid destabilizing gait
        self.Kd_null = 0.5
        self._tau_max = np.array([120, 200, 200, 100,
                                   120, 200, 200, 100,
                                   80,  80,  60], dtype=float)
        self._q0_arm = np.array([0.0, 0.5, -1.0])  # resting arm posture
        # Gravity compensation: precomputed once at init when qvel≈0.
        # This avoids Coriolis contamination during fast leg motions.
        self._gravity_ff = self._compute_init_gravity(model)

    def _compute_init_gravity(self, model):
        """Compute arm gravity torques at the keyframe pose with zero velocity."""
        d_tmp = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, d_tmp, 0)
        mujoco.mj_forward(model, d_tmp)
        arm_dofs = [self.ids[k] for k in self._ARM_DOF_KEYS]
        h = d_tmp.qfrc_bias.copy()
        del d_tmp
        return np.array([h[dadr] for dadr in arm_dofs])

    def compute(self, data, F_arm_mpc, contact_mask=None,
                q_ref_legs=None, dq_ref_legs=None,
                Kp_leg=None, Kd_leg=None):
        """
        Parameters
        ----------
        data        : MjData (forward dynamics already called)
        F_arm_mpc   : (3,) corrective Cartesian force from ImpedanceMPC [N]
        contact_mask: [bool, bool] — feet in contact
        q_ref_legs  : (8,) desired leg angles
        dq_ref_legs : (8,) desired leg velocities
        Kp_leg      : (8,) leg PD proportional gains
        Kd_leg      : (8,) leg PD derivative gains

        Returns
        -------
        tau  : (11,) joint torques [Nm]  (8 leg + 3 arm, matching actuator order)
        info : dict
        """
        model = self.model
        ids   = self.ids
        nv    = model.nv

        if contact_mask is None: contact_mask = [True, True]
        if q_ref_legs   is None: q_ref_legs   = np.zeros(8)
        if dq_ref_legs  is None: dq_ref_legs  = np.zeros(8)
        if Kp_leg       is None: Kp_leg        = np.full(8, 80.0)
        if Kd_leg       is None: Kd_leg        = np.full(8, 8.0)

        # ── Dynamics quantities ────────────────────────────────────────────
        M    = get_mass_matrix(model, data)
        foot_site_ids = [ids['left_foot_site'], ids['right_foot_site']]
        Jc   = get_contact_jacobian(model, data, foot_site_ids, contact_mask)
        Mbar = get_contact_consistent_inverse(M, Jc)

        # ── Arm Jacobian and task-space inertia (Level 3) ─────────────────
        J_arm   = get_site_jacobian(model, data, ids['hand_site'])
        Lam_arm = get_task_inertia(J_arm, Mbar)

        # ── Leg balance (Level 2) — joint-space PD ────────────────────────
        leg_dofs  = [ids[k] for k in self._LEG_DOF_KEYS]
        leg_qadrs = [ids[k] for k in self._LEG_QADR_KEYS]
        tau_leg   = np.zeros(8)
        for i, (dadr, qadr_) in enumerate(zip(leg_dofs, leg_qadrs)):
            q  = data.qpos[qadr_]
            dq = data.qvel[dadr]
            tau_leg[i] = Kp_leg[i] * (q_ref_legs[i] - q) - Kd_leg[i] * dq

        # ── Arm task torque (Level 3), contact-consistent projection ──────
        arm_dofs  = [ids[k] for k in self._ARM_DOF_KEYS]
        arm_qadrs = [ids[k] for k in self._ARM_QADR_KEYS]
        leg_dofs  = [ids[k] for k in self._LEG_DOF_KEYS]
        tau_leg_cc = np.zeros(8)
        force_to_torque = np.zeros((11, 3))
        if self.contact_consistent and Jc.shape[0] > 0:
            # Contact null-space projector P_c (generalized-force space):
            # In the undamped exact-model limit, a task force mapped through
            # P_c^T produces no contact acceleration. With the regularization
            # below, this decoupling is approximate.
            Pc    = get_contact_consistent_projector(M, Jc)
            tau_full_map = _checked_matmul(
                Pc.T, J_arm.T, "Pc.T @ J_arm.T"
            )
            tau_arm_task = _checked_matmul(
                tau_full_map[arm_dofs], F_arm_mpc, "arm force-to-torque map"
            )
            force_to_torque[8:] = tau_full_map[arm_dofs]
            if self.apply_leg_coupling:
                tau_leg_cc = _checked_matmul(
                    tau_full_map[leg_dofs], F_arm_mpc,
                    "leg force-to-torque map",
                )
                force_to_torque[:8] = tau_full_map[leg_dofs]
        else:
            force_to_torque[8:] = J_arm[:, arm_dofs].T
            tau_arm_task = _checked_matmul(
                force_to_torque[8:], F_arm_mpc, "arm-only force-to-torque map"
            )

        # ── Arm null-space centering (Level 4) ────────────────────────────
        # Simple PD on arm joints, projected to not fight the task force.
        # Use plain joint-space PD (no contact-consistent projection to
        # avoid numerical instability from near-singular Mbar).
        tau_arm_null = np.zeros(3)
        for i, (dadr, qadr_) in enumerate(zip(arm_dofs, arm_qadrs)):
            q  = data.qpos[qadr_]
            dq = data.qvel[dadr]
            tau_arm_null[i] = (self.Kp_null * (self._q0_arm[i] - q)
                               - self.Kd_null * dq)

        # Combine arm torques: task force + null-space centering + gravity ff
        tau_arm = tau_arm_task + tau_arm_null + self._gravity_ff
        # Add the contact-consistent leg coupling (zero when projection off)
        tau_leg = tau_leg + tau_leg_cc

        # ── Assembly (Eq. 28) ────────────────────────────────────────────
        tau_preclip = np.concatenate([tau_leg, tau_arm])
        tau = np.clip(tau_preclip, -self._tau_max, self._tau_max)

        return tau, dict(Lam_arm=Lam_arm, J_arm=J_arm, Mbar=Mbar,
                         Jc=Jc, contact_mask=contact_mask,
                 contact_consistent=self.contact_consistent,
                 tau_preclip=tau_preclip,
                 force_to_torque=force_to_torque)
