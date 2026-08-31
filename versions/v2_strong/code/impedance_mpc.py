"""Normalized predictive interaction controller (paper Section V).

The QP input is residual Cartesian acceleration. The current task inertia
appears only in the force recovery and force constraints.
"""

import numpy as np
import osqp
import scipy.sparse as sp


def _checked_matmul(left, right, name):
    """Multiply dense arrays while rejecting real non-finite results.

    Apple Accelerate's arm64 GEMM path can set divide/overflow/invalid status
    flags for finite, well-conditioned products.  NumPy exposes those flags
    as warnings (or ``FloatingPointError`` under ``seterr(...='raise')``), so
    inspect the product itself rather than accepting backend status alone.
    """
    with np.errstate(all="ignore"):
        product = np.asarray(left) @ np.asarray(right)
    if not np.all(np.isfinite(product)):
        raise FloatingPointError(f"non-finite matrix product: {name}")
    return product

class ImpedanceMPC:
    """
    Receding-horizon QP for arm end-effector tracking.

    State:  x_e = [e^T, e_dot^T]^T ∈ ℝ^6   (position + velocity error)
    Input:  u ∈ ℝ^3                        (residual Cartesian acceleration)

    A_d (constant) = [[I, dt*I], [0, I]]
    B_d (constant) = [[½dt² I], [dt I]]

    solve() returns the recovered force Lambda_arm @ u. The corresponding
    acceleration is available as last_u for the disturbance observer.
    """

    def __init__(self, N=20, dt=0.001, Q=None, R=None, F_max=60.0):
        self.N     = N
        self.dt    = dt
        self.F_max = F_max
        self.nxe   = 6   # state dimension
        self.nu    = 3   # input dimension

        # Default cost weights
        if Q is None:
            Q = np.diag([1000., 1000., 1000., 10., 10., 10.])
        if R is None:
            R = 0.1 * np.eye(3)
        self.Q = Q
        self.R = R
        self.last_u = np.zeros(3)
        self.last_u_sequence = np.zeros((N, self.nu))


        # Constant state-transition matrix A_d (Proposition 1)
        self.A_d = np.block([
            [np.eye(3),   dt * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)]
        ])
        self.B_d = np.vstack([
            0.5 * dt * dt * np.eye(3),
            dt * np.eye(3),
        ])

        # Precomputed free-response rollout matrix Phi (N*nxe × nxe)
        nxe, N_ = self.nxe, self.N
        self._Phi = np.zeros((N_ * nxe, nxe))
        A_pow = self.A_d.copy()
        for i in range(N_):
            self._Phi[i*nxe:(i+1)*nxe, :] = A_pow
            A_pow = self.A_d @ A_pow

        # Stacked cost matrices
        self._Q_bar = np.kron(np.eye(N_), Q)
        self._R_bar = np.kron(np.eye(N_), R)

        self._mode_library = {}
        self._constant_model = self._build_lifted_model()

        # OSQP instance (re-used across solves for warm-starting)
        self._osqp = None
        self._osqp_nnz = None
        self._osqp_A_nnz = None
        self._osqp_A_shape = None

    def _build_lifted_model(self):
        N, nxe, nu = self.N, self.nxe, self.nu
        Gamma = np.zeros((N * nxe, N * nu))
        for i in range(N):
            for j in range(i + 1):
                Gamma[i*nxe:(i+1)*nxe, j*nu:(j+1)*nu] = (
                    np.linalg.matrix_power(self.A_d, i - j) @ self.B_d
                )
        H = _checked_matmul(
            _checked_matmul(Gamma.T, self._Q_bar, "Gamma.T @ Q_bar"),
            Gamma,
            "Gamma.T @ Q_bar @ Gamma",
        ) + self._R_bar
        if not np.all(np.isfinite(H)):
            raise FloatingPointError("non-finite lifted MPC Hessian")
        return {
            "B_d": self.B_d,
            "Gamma": Gamma,
            "H": H,
            "H_inv": np.linalg.inv(H),
        }

    # ------------------------------------------------------------------
    def precompute_mode(self, mode_key, Lambda_arm):
        """Register a mode while reusing the constant lifted model."""
        self._mode_library[mode_key] = dict(
            **self._constant_model,
            Lambda_arm=np.asarray(Lambda_arm, float).copy(),
        )
        return self._mode_library[mode_key]

    def get_or_update_mode(self, mode_key, Lambda_arm):
        """Return the constant model and retain the latest recovery inertia."""
        if mode_key not in self._mode_library:
            return self.precompute_mode(mode_key, Lambda_arm)
        cached = self._mode_library[mode_key]
        cached["Lambda_arm"] = np.asarray(Lambda_arm, float).copy()
        return cached

    # ------------------------------------------------------------------
    def solve(self, x_e, Lambda_arm, mode_key=None,
              d_hat=None, use_osqp=True, p_ddot_d=None, mu_arm=None,
              torque_map=None, torque_offset=None,
              torque_min=None, torque_max=None):
        """
        Solve the receding-horizon QP.

        Parameters
        ----------
        x_e        : (6,) tracking error state [e; e_dot]
        Lambda_arm : (3,3) current task-space inertia
        mode_key   : hashable contact-mode id (default: single mode)
        d_hat      : (3,) Kalman disturbance estimate; None → no Kalman
        use_osqp   : use OSQP with box constraints; False → unconstrained
        p_ddot_d   : (3,) desired task-space acceleration (paper eq. ff_a);
                     None → zeros, correct for the regulation scenarios in
                     this benchmark (a held, non-moving setpoint), not a
                     silent omission — callers tracking a moving reference
                     must pass the real value.
        mu_arm     : (3,) task-space Coriolis/gravity bias (paper eq. after
                     eq:plant, wbc_core.get_arm_bias_force); None → zeros.
                     Previously always implicitly zero here (the missing
                     feedforward term an external review found); passing it
                     explicitly is now the caller's responsibility so a
                     missing value is visible at the call site, not buried.
                torque_map : (n_tau,3) local map from residual acceleration to the
                         full pre-clip applied actuator torque. When supplied
                         with the remaining torque arguments, its affine bound is
                         imposed at every MPC horizon stage.
                torque_offset, torque_min, torque_max : (n_tau,) affine torque
                         offset and actuator bounds for ``torque_map``.

        Returns
        -------
        F_mpc : (3,) recovered corrective force [N]
        """
        if mode_key is None:
            mode_key = 'default'
        p_ddot_d = np.zeros(3) if p_ddot_d is None else np.asarray(p_ddot_d, float)
        mu_arm = np.zeros(3) if mu_arm is None else np.asarray(mu_arm, float)
        Lambda_arm_f = np.asarray(Lambda_arm, float)
        # F_ff = Lambda_arm @ p_ddot_d + mu_arm (paper eq. ff_a with u=0):
        # the feedforward force the residual QP's u=0 point should already
        # sit at, so the total applied force is F_ff + Lambda_arm @ u.
        F_ff = Lambda_arm_f @ p_ddot_d + mu_arm

        mode = self.get_or_update_mode(mode_key, Lambda_arm)
        B_d, Gamma, H, H_inv = (mode['B_d'], mode['Gamma'],
                                 mode['H'], mode['H_inv'])

        N, nxe, nu = self.N, self.nxe, self.nu

        # Free response with the estimated disturbance injected through all
        # prediction steps. Both d_hat and the decision input are accelerations.
        d_stack = np.zeros(N * nu)
        if d_hat is not None:
            d_hat = np.asarray(d_hat, float)
            d_stack = np.tile(d_hat, N)
            x_free = np.zeros(N * nxe)
            x_k = x_e.copy()
            for i in range(N):
                x_k = self.A_d @ x_k + B_d @ d_hat
                x_free[i*nxe:(i+1)*nxe] = x_k
        else:
            x_free = _checked_matmul(self._Phi, x_e, "Phi @ x_e")

        # Gradient of QP cost.  The input effort is centered at the estimated
        # cancelling input, ||U + d_hat||_R, not raw ||U||_R.  With
        # D_bar*d_hat = Gamma*(1_N kron d_hat), this converts the offset-free
        # steady-state problem into a nominal regulation problem in
        # V = U + d_hat and yields F_mpc -> -d_hat for constant disturbances.
        h_qp = (
            _checked_matmul(
                _checked_matmul(Gamma.T, self._Q_bar, "Gamma.T @ Q_bar"),
                x_free,
                "Gamma.T @ Q_bar @ x_free",
            )
            + _checked_matmul(self._R_bar, d_stack, "R_bar @ d_stack")
        )
        if not np.all(np.isfinite(h_qp)):
            raise FloatingPointError("non-finite MPC gradient")
        
        if not use_osqp:
            # Unconstrained solution (fast path when far from limits)
            U_star = -_checked_matmul(H_inv, h_qp, "H_inv @ h_qp")
            self.last_u_sequence = U_star.reshape(N, nu).copy()
            u_cmd = U_star[:nu]
        else:
            u_cmd = self._solve_osqp(
                H, h_qp, N, nu, Lambda_arm_f, F_ff,
                torque_map, torque_offset, torque_min, torque_max,
            )

        force = Lambda_arm_f @ (u_cmd + p_ddot_d) + mu_arm
        force = np.clip(force, -self.F_max, self.F_max)
        # Recover the realized residual acceleration from the CLIPPED force
        # (inverting F = Lambda_arm(p_ddot_d + u) + mu_arm for u), same
        # convention as before feedforward was added, just now correctly
        # subtracting the feedforward terms first.
        self.last_u = np.linalg.solve(Lambda_arm_f, force - mu_arm) - p_ddot_d
        return force

    def _solve_osqp(
        self, H, h_qp, N, nu, Lambda_arm, F_ff,
        torque_map=None, torque_offset=None, torque_min=None, torque_max=None,
    ):
        """Solve with horizon-wide total-force and optional total-torque rows."""
        n_dec = N * nu
        # OSQP stores only the upper triangular part of the symmetric Hessian.
        # Passing the full matrix on update changes the data length and triggers
        # "new number of elements out of bounds" warnings.
        P = sp.triu(sp.csc_matrix(H), format='csc')
        q = h_qp
        # Keep all 3x3 entries in the sparse pattern so OSQP can update a
        # rotating, generally dense task inertia without rebuilding the solver.
        Lambda_pattern = np.asarray(Lambda_arm, float) + 1e-16 * np.ones((3, 3))
        force_A = sp.kron(sp.eye(N, format="csc"),
                   sp.csc_matrix(Lambda_pattern), format="csc")
        # Constrain the TOTAL applied force |F_ff + Lambda_arm u| <= F_max
        # (paper eq:QP), not |Lambda_arm u| alone: shift the box by -F_ff,
        # frozen at its current value across the horizon (same frozen-
        # Level-2-value approximation the paper states for Lambda_arm
        # itself). Found omitted by external review -- F_ff was not wired
        # into this solver at all before the feedforward fix.
        F_ff_stack = np.tile(np.asarray(F_ff, float), N)
        lb = np.full(n_dec, -self.F_max) - F_ff_stack
        ub = np.full(n_dec, self.F_max) - F_ff_stack

        use_torque_constraint = any(
            value is not None
            for value in (torque_map, torque_offset, torque_min, torque_max)
        )
        if use_torque_constraint:
            if any(value is None for value in (
                torque_map, torque_offset, torque_min, torque_max
            )):
                raise ValueError("all torque constraint arguments are required")
            torque_map = np.asarray(torque_map, dtype=float)
            torque_offset = np.asarray(torque_offset, dtype=float).reshape(-1)
            torque_min = np.asarray(torque_min, dtype=float).reshape(-1)
            torque_max = np.asarray(torque_max, dtype=float).reshape(-1)
            n_tau = torque_map.shape[0] if torque_map.ndim == 2 else 0
            if (torque_map.shape != (n_tau, nu)
                    or torque_offset.shape != (n_tau,)
                    or torque_min.shape != (n_tau,)
                    or torque_max.shape != (n_tau,)
                    or not np.all(np.isfinite(torque_map))
                    or not np.all(np.isfinite(torque_offset))
                    or not np.all(np.isfinite(torque_min))
                    or not np.all(np.isfinite(torque_max))
                    or np.any(torque_min > torque_max)):
                raise ValueError("invalid affine torque constraint")
            torque_pattern = torque_map + 1e-16 * np.ones_like(torque_map)
            torque_A = sp.kron(
                sp.eye(N, format="csc"), sp.csc_matrix(torque_pattern), format="csc"
            )
            force_A = sp.vstack((force_A, torque_A), format="csc")
            lb = np.concatenate((lb, np.tile(torque_min - torque_offset, N)))
            ub = np.concatenate((ub, np.tile(torque_max - torque_offset, N)))
        A = force_A


        if (self._osqp is None or self._osqp_nnz != P.nnz
                or self._osqp_A_nnz != A.nnz
                or self._osqp_A_shape != A.shape):
            prob = osqp.OSQP()
            prob.setup(P, q, A, lb, ub,
                       warm_starting=True, verbose=False,
                       max_iter=1000, eps_abs=1e-4, eps_rel=1e-4,
                       polishing=False)
            self._osqp = prob
            self._osqp_nnz = P.nnz
            self._osqp_A_nnz = A.nnz
            self._osqp_A_shape = A.shape
        else:
            self._osqp.update(q=q, l=lb, u=ub, Px=P.data, Ax=A.data)

        result = self._osqp.solve(raise_error=False)
        # Validate the actual primal residual, not just the status string
        # (100x this problem's own eps_abs=eps_rel=1e-4 -- same convention
        # used across every other controller reviewed this session).
        # "solved inaccurate" alone does not guarantee A @ U actually stays
        # within [lb, ub]; found missing by external review.
        feas_tol = 100 * 1e-4
        x = result.x
        finite = x is not None and np.all(np.isfinite(x))
        if finite:
            au = A @ x
            residual = float(np.maximum(
                np.maximum(lb - au, 0), np.maximum(au - ub, 0)
            ).max())
        else:
            residual = float("inf")
        if (result.info.status_val in (1, 2)) and finite and residual <= feas_tol:
            self.last_u_sequence = x.reshape(N, nu).copy()
            return x[:nu]
        # Fail SAFE, not fast: an earlier version of this fallback returned
        # the UNCONSTRAINED solution on any OSQP failure -- silently
        # ignoring the force limit entirely, found by external review. u=0
        # instead: the applied force falls back to F_ff alone (the
        # feedforward/gravity-cancelling term, typically much smaller than
        # an unconstrained QP solution near a limit), still passed through
        # solve()'s own np.clip(-F_max, F_max) as the final backstop either
        # way -- this does not itself guarantee F_ff is within F_max, only
        # that the fallback no longer actively discards the constraint.
        self.last_u_sequence = np.zeros((N, nu))
        return np.zeros(nu)
