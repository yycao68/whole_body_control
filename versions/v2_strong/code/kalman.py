"""
Kalman Disturbance Estimator  (Section V-E and VI of paper).

Augmented state: [x_e; d_hat] where d_hat ∈ ℝ³ is an integrating
disturbance state that converges to a bounded constant acceleration disturbance.

A_aug = [[A_d, B_d], [0, I]]   (Eq. 23)

Contact-mode transition: inflate covariance by α (Section VI-A).
"""

import numpy as np



def _checked_matmul(left, right, name):
    """Multiply dense arrays while detecting real non-finite results."""
    with np.errstate(all="ignore"):
        product = np.asarray(left) @ np.asarray(right)
    if not np.all(np.isfinite(product)):
        raise FloatingPointError(f"non-finite matrix product: {name}")
    return product


class KalmanDisturbanceEstimator:
    """
    Kalman filter for joint estimation of tracking error and pHRI disturbance.

    Observation: hand Cartesian position (3-DOF) and optionally velocity (3-DOF).
    Velocity observation is passed as an optional argument to update(); when
    provided it gives the Kalman direct access to the d_hat signal, cutting
    convergence time from ~seconds to ~tens of milliseconds.
    """

    def __init__(self, dt,
                 Q_w_xe=1e-4,  Q_w_d=1e-2,
                 R_v=1e-6,     R_v_vel=1e-4):
        self.dt    = dt
        self.nxe   = 6   # [e, e_dot]
        self.nd    = 3   # disturbance
        self.n_aug = 9   # total augmented state

        # Process noise (tuned separately for tracking error and disturbance)
        self.Q_w = np.block([
            [Q_w_xe * np.eye(6), np.zeros((6, 3))],
            [np.zeros((3, 6)),   Q_w_d  * np.eye(3)]
        ])

        # Position-only observation (3-DOF)
        self.R_v     = R_v * np.eye(3)
        self.C_obs   = np.hstack([np.eye(3), np.zeros((3, 6))])   # (3×9)

        # Position + velocity observation (6-DOF) — used when e_vel is passed to update()
        self.R_v_pv  = np.block([
            [R_v     * np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)),   R_v_vel * np.eye(3)]
        ])
        self.C_obs_pv = np.hstack([np.eye(6), np.zeros((6, 3))])  # (6×9)

        # State and covariance
        self.x_aug = np.zeros(9)
        self.P     = 0.01 * np.eye(9)

        # A_d, B_d to be set by first call to set_mode
        self.A_d = None
        self.B_d = None
        self.A_aug = None

    # ------------------------------------------------------------------
    def set_mode(self, A_d, B_d):
        """Update augmented system matrices for a new contact mode."""
        nxe, nd = self.nxe, self.nd
        self.A_d = A_d.copy()
        self.B_d = B_d.copy()

        # A_aug = [[A_d, B_d], [0, I]]  (Eq. 23)
        self.A_aug = np.block([
            [A_d,                B_d],
            [np.zeros((nd, nxe)), np.eye(nd)]
        ])

        # B_ctrl: how residual acceleration enters the state prediction.
        self.B_ctrl = np.vstack([B_d, np.zeros((nd, 3))])

    def inflate_covariance(self, alpha=4.0):
        """Inflate covariance at contact-mode switch (Eq. 26 context)."""
        self.P *= alpha

    # ------------------------------------------------------------------
    def predict(self, u_prev):
        """Prior prediction step."""
        if self.A_aug is None:
            return
        self.x_aug = (
            _checked_matmul(self.A_aug, self.x_aug, "A_aug @ x_aug")
            + _checked_matmul(self.B_ctrl, u_prev, "B_ctrl @ u_prev")
        )
        self.P = (
            _checked_matmul(
                _checked_matmul(self.A_aug, self.P, "A_aug @ P"),
                self.A_aug.T,
                "A_aug @ P @ A_aug.T",
            )
            + self.Q_w
        )
        if not (np.all(np.isfinite(self.x_aug)) and np.all(np.isfinite(self.P))):
            raise FloatingPointError("non-finite Kalman predicted state or covariance")

    def update(self, e_pos_meas, e_vel_meas=None):
        """
        Measurement update.

        Parameters
        ----------
        e_pos_meas : (3,) observed position error
        e_vel_meas : (3,) observed velocity error, optional.
            When provided, a 6-DOF [pos; vel] observation is used, giving the
            Kalman direct access to the velocity signal and dramatically
            accelerating d_hat convergence.

        Returns (x_e, d_hat) where x_e ∈ ℝ^6, d_hat ∈ ℝ^3.
        """
        if self.A_aug is None:
            return self.x_aug[:6], self.x_aug[6:]

        if e_vel_meas is not None:
            C   = self.C_obs_pv                              # 6×9
            R   = self.R_v_pv                                # 6×6
            meas = np.concatenate([e_pos_meas, e_vel_meas])  # (6,)
        else:
            C   = self.C_obs   # 3×9
            R   = self.R_v     # 3×3
            meas = e_pos_meas  # (3,)

        y_pred = _checked_matmul(C, self.x_aug, "C @ x_aug")
        innov  = meas - y_pred

        S = _checked_matmul(
            _checked_matmul(C, self.P, "C @ P"), C.T, "C @ P @ C.T"
        ) + R
        S_inv = np.linalg.inv(S)
        if not np.all(np.isfinite(S_inv)):
            raise FloatingPointError("non-finite Kalman innovation inverse")
        K = _checked_matmul(
            _checked_matmul(self.P, C.T, "P @ C.T"), S_inv, "P @ C.T @ inv(S)"
        )

        self.x_aug = self.x_aug + _checked_matmul(K, innov, "K @ innovation")
        self.P = _checked_matmul(
            np.eye(9) - _checked_matmul(K, C, "K @ C"), self.P,
            "(I - K @ C) @ P",
        )
        self.P     = 0.5 * (self.P + self.P.T)
        if not (np.all(np.isfinite(self.x_aug)) and np.all(np.isfinite(self.P))):
            raise FloatingPointError("non-finite Kalman updated state or covariance")

        return self.x_aug[:6].copy(), self.x_aug[6:].copy()

    # ------------------------------------------------------------------
    def reset(self, x_e_init=None, d_hat_init=None):
        self.x_aug = np.zeros(9)
        if x_e_init is not None:
            self.x_aug[:6] = x_e_init
        if d_hat_init is not None:
            self.x_aug[6:] = d_hat_init
        self.P = 0.01 * np.eye(9)

    @property
    def x_e(self):
        return self.x_aug[:6].copy()

    @property
    def d_hat(self):
        return self.x_aug[6:].copy()
