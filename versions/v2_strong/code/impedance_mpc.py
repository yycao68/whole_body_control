"""Normalized predictive interaction controller (paper Section V).

The QP input is residual Cartesian acceleration. The current task inertia
appears only in the force recovery and force constraints.
"""

import numpy as np
import osqp
import scipy.sparse as sp


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

    def _build_lifted_model(self):
        N, nxe, nu = self.N, self.nxe, self.nu
        Gamma = np.zeros((N * nxe, N * nu))
        for i in range(N):
            for j in range(i + 1):
                Gamma[i*nxe:(i+1)*nxe, j*nu:(j+1)*nu] = (
                    np.linalg.matrix_power(self.A_d, i - j) @ self.B_d
                )
        H = Gamma.T @ self._Q_bar @ Gamma + self._R_bar
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
              d_hat=None, use_osqp=True):
        """
        Solve the receding-horizon QP.

        Parameters
        ----------
        x_e        : (6,) tracking error state [e; e_dot]
        Lambda_arm : (3,3) current task-space inertia
        mode_key   : hashable contact-mode id (default: single mode)
        d_hat      : (3,) Kalman disturbance estimate; None → no Kalman
        use_osqp   : use OSQP with box constraints; False → unconstrained

        Returns
        -------
        F_mpc : (3,) recovered corrective force [N]
        """
        if mode_key is None:
            mode_key = 'default'

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
            x_free = self._Phi @ x_e

        # Gradient of QP cost.  The input effort is centered at the estimated
        # cancelling input, ||U + d_hat||_R, not raw ||U||_R.  With
        # D_bar*d_hat = Gamma*(1_N kron d_hat), this converts the offset-free
        # steady-state problem into a nominal regulation problem in
        # V = U + d_hat and yields F_mpc -> -d_hat for constant disturbances.
        h_qp = Gamma.T @ self._Q_bar @ x_free + self._R_bar @ d_stack

        if not use_osqp:
            # Unconstrained solution (fast path when far from limits)
            U_star = -H_inv @ h_qp
            u_cmd = U_star[:nu]
        else:
            u_cmd = self._solve_osqp(H, h_qp, N, nu, Lambda_arm)

        force = np.asarray(Lambda_arm, float) @ u_cmd
        force = np.clip(force, -self.F_max, self.F_max)
        self.last_u = np.linalg.solve(np.asarray(Lambda_arm, float), force)
        return force

    def _solve_osqp(self, H, h_qp, N, nu, Lambda_arm):
        """OSQP solve with horizon-wide |Lambda_arm u| <= F_max."""
        n_dec = N * nu
        # OSQP stores only the upper triangular part of the symmetric Hessian.
        # Passing the full matrix on update changes the data length and triggers
        # "new number of elements out of bounds" warnings.
        P = sp.triu(sp.csc_matrix(H), format='csc')
        q = h_qp
        # Keep all 3x3 entries in the sparse pattern so OSQP can update a
        # rotating, generally dense task inertia without rebuilding the solver.
        Lambda_pattern = np.asarray(Lambda_arm, float) + 1e-16 * np.ones((3, 3))
        A = sp.kron(sp.eye(N, format="csc"),
                    sp.csc_matrix(Lambda_pattern), format="csc")
        lb = np.full(n_dec, -self.F_max)
        ub = np.full(n_dec,  self.F_max)

        if (self._osqp is None or self._osqp_nnz != P.nnz
                or self._osqp_A_nnz != A.nnz):
            prob = osqp.OSQP()
            prob.setup(P, q, A, lb, ub,
                       warm_starting=True, verbose=False,
                       max_iter=1000, eps_abs=1e-4, eps_rel=1e-4,
                       polishing=True)
            self._osqp = prob
            self._osqp_nnz = P.nnz
            self._osqp_A_nnz = A.nnz
        else:
            self._osqp.update(q=q, l=lb, u=ub, Px=P.data, Ax=A.data)

        result = self._osqp.solve(raise_error=False)
        if result.info.status == 'solved' or result.info.status_val == 1:
            return result.x[:nu]
        # Fallback to unconstrained if OSQP fails
        return -np.linalg.solve(H, h_qp)[:nu]
