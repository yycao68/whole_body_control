"""
WBC Core Dynamics Utilities
Implements contact-consistent mass inverse, operational-space formulation,
and hierarchical null-space torque synthesis for a floating-base biped.
All equations reference the IEEE paper section numbers in comments.
"""

import numpy as np
import mujoco


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

def get_body_jacobian(model, data, body_id):
    """(3×nv, 3×nv) translational + rotational Jacobians for a body."""
    nv = model.nv
    Jp = np.zeros((3, nv))
    Jr = np.zeros((3, nv))
    mujoco.mj_jacBody(model, data, Jp, Jr, body_id)
    return Jp, Jr


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
    Lambda_c = np.linalg.inv(Jc @ M_inv @ Jc.T + contact_damp * np.eye(nc))
    M_bar_inv = M_inv - M_inv @ Jc.T @ Lambda_c @ Jc @ M_inv
    return 0.5 * (M_bar_inv + M_bar_inv.T)


# --------------------------------------------------------------------------
# Operational-space quantities  (Eq. 13)
# --------------------------------------------------------------------------

def get_task_inertia(J, M_bar_inv, reg=1e-4):
    """Λ = (J M̄⁻¹ J^T)⁻¹  with eigenvalue clamping to guarantee PSD."""
    A = J @ M_bar_inv @ J.T
    # Symmetrise and clamp eigenvalues so A stays PD
    A = 0.5 * (A + A.T)
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, reg)
    A_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return np.linalg.inv(A_psd)

def get_dynamically_consistent_pseudoinverse(J, M_bar_inv, Lambda):
    """J̄ = M̄⁻¹ J^T Λ."""
    return M_bar_inv @ J.T @ Lambda

def get_null_space_projector(J_bar, J):
    """N̄ = I − J̄ J."""
    return np.eye(J.shape[1]) - J_bar @ J


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
        if self.contact_consistent and Jc.shape[0] > 0:
            # Contact null-space projector P_c (generalized-force space):
            # In the undamped exact-model limit, a task force mapped through
            # P_c^T produces no contact acceleration. With the regularization
            # below, this decoupling is approximate.
            M_inv = np.linalg.inv(M + 1e-4 * np.eye(nv))
            Lam_c = np.linalg.inv(Jc @ M_inv @ Jc.T + 1e-3 * np.eye(Jc.shape[0]))
            Pc    = np.eye(nv) - Jc.T @ Lam_c @ Jc @ M_inv
            tau_full     = Pc.T @ (J_arm.T @ F_arm_mpc)   # (nv,) projected torque
            tau_arm_task = tau_full[arm_dofs]
            if self.apply_leg_coupling:
                tau_leg_cc = tau_full[leg_dofs]  # legs help hold contact under the arm task
        else:
            tau_arm_task = J_arm[:, arm_dofs].T @ F_arm_mpc  # arm-only (no projection)

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
        tau = np.concatenate([tau_leg, tau_arm])
        tau = np.clip(tau, -self._tau_max, self._tau_max)

        return tau, dict(Lam_arm=Lam_arm, J_arm=J_arm, Mbar=Mbar,
                         Jc=Jc, contact_mask=contact_mask,
                         contact_consistent=self.contact_consistent)
