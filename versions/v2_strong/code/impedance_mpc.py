"""
Impedance MPC — Level 3 receding-horizon QP  (Section V of paper).

Key properties:
- Constant A_d (Proposition 1): fixed free-response rollout Phi
- Contact-mode-indexed B_d^{(m)}: cached and refreshed when Lambda changes
- Input-centered disturbance preload: cost uses ||U + d_hat||_R for offset-free tracking
- OSQP solver for box-constrained F_mpc
"""

import numpy as np
import osqp
import scipy.sparse as sp


class ImpedanceMPC:
    """
    Receding-horizon QP for arm end-effector tracking.

    State:  x_e = [e^T, e_dot^T]^T ∈ ℝ^6   (position + velocity error)
    Input:  F_mpc ∈ ℝ^3                    (corrective Cartesian force)

    A_d (constant) = [[I, dt*I], [0, I]]
    B_d^(m)        = [[-½dt² Λ_arm^{-1}], [-Λ_arm^{-1} dt]]  — exact ZOH, contact-mode indexed
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

        # Constant state-transition matrix A_d (Proposition 1)
        self.A_d = np.block([
            [np.eye(3),   dt * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)]
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

        # Library of contact-mode-indexed (B_d, Gamma, H, H_inv) tuples.
        # Key: hashable contact-mode / inertia-snapshot id.  Since Lambda_arm
        # still varies with posture inside a contact mode, entries are refreshed
        # when Lambda changes appreciably.
        self._mode_library = {}

        # OSQP instance (re-used across solves for warm-starting)
        self._osqp = None
        self._osqp_Lambda = None   # Lambda_arm for which _osqp was built
        self._osqp_nnz = None

    # ------------------------------------------------------------------
    def precompute_mode(self, mode_key, Lambda_arm):
        """
        Offline precomputation for a given contact mode.
        mode_key   : any hashable id (e.g. frozenset of active foot names)
        Lambda_arm : (3,3) task-space inertia for this mode
        """
        N, nxe, nu = self.N, self.nxe, self.nu
        dt = self.dt

        # Exact zero-order-hold input matrix (ported from the pHRI FR3 controller).
        # A_c is nilpotent (A_c^2 = 0), so A_d = I + A_c*dt is already exact; the ZOH
        # integral B_d = ∫_0^dt e^{A_c s} B_c ds then carries a -½dt² Λ^{-1} top block
        # that the earlier Forward-Euler form dropped. Γ, H, the Kalman augmented
        # model, and the disturbance injection all read this B_d, so the whole
        # pipeline stays on one consistent discretization (no ZOH/Euler mixing).
        Lam_inv = np.linalg.inv(Lambda_arm)
        B_d = np.vstack([
             -0.5 * dt * dt * Lam_inv,
             -Lam_inv * dt
        ])

        # Gamma matrix: (N*nxe) × (N*nu)
        Gamma = np.zeros((N * nxe, N * nu))
        for i in range(N):
            for j in range(i + 1):
                Ak = np.linalg.matrix_power(self.A_d, i - j)
                Gamma[i*nxe:(i+1)*nxe, j*nu:(j+1)*nu] = Ak @ B_d

        H = Gamma.T @ self._Q_bar @ Gamma + self._R_bar

        self._mode_library[mode_key] = dict(
            B_d=B_d, Gamma=Gamma, H=H,
            H_inv=np.linalg.inv(H),
            Lambda_arm=Lambda_arm.copy(),
        )
        return self._mode_library[mode_key]

    def get_or_update_mode(self, mode_key, Lambda_arm):
        """Return cached mode; recompute if Lambda changed significantly."""
        if mode_key not in self._mode_library:
            return self.precompute_mode(mode_key, Lambda_arm)
        cached = self._mode_library[mode_key]
        diff = np.linalg.norm(Lambda_arm - cached['Lambda_arm']) / (
               np.linalg.norm(cached['Lambda_arm']) + 1e-8)
        if diff > 0.05:   # 5% change triggers recompute
            return self.precompute_mode(mode_key, Lambda_arm)
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
        F_mpc : (3,) corrective force [N]
        """
        if mode_key is None:
            mode_key = 'default'

        mode = self.get_or_update_mode(mode_key, Lambda_arm)
        B_d, Gamma, H, H_inv = (mode['B_d'], mode['Gamma'],
                                 mode['H'], mode['H_inv'])

        N, nxe, nu = self.N, self.nxe, self.nu

        # Free response with the estimated disturbance injected through all
        # prediction steps. The disturbance is expressed in the same equivalent
        # force coordinates as F_mpc, so it enters through B_d.
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
            F_mpc = U_star[:nu]
        else:
            F_mpc = self._solve_osqp(H, h_qp, N, nu)

        return np.clip(F_mpc, -self.F_max, self.F_max)

    def _solve_osqp(self, H, h_qp, N, nu):
        """OSQP solve with ||F_mpc||_inf ≤ F_max box constraint."""
        n_dec = N * nu
        # OSQP stores only the upper triangular part of the symmetric Hessian.
        # Passing the full matrix on update changes the data length and triggers
        # "new number of elements out of bounds" warnings.
        P = sp.triu(sp.csc_matrix(H), format='csc')
        q = h_qp
        # Box constraint: I * U ∈ [-F_max, F_max]
        A = sp.eye(n_dec, format='csc')
        lb = np.full(n_dec, -self.F_max)
        ub = np.full(n_dec,  self.F_max)

        if self._osqp is None or self._osqp_nnz != P.nnz:
            prob = osqp.OSQP()
            prob.setup(P, q, A, lb, ub,
                       warm_starting=True, verbose=False,
                       max_iter=1000, eps_abs=1e-4, eps_rel=1e-4,
                       polish=True)
            self._osqp = prob
            self._osqp_nnz = P.nnz
        else:
            self._osqp.update(q=q, l=lb, u=ub, Px=P.data)

        result = self._osqp.solve()
        if result.info.status == 'solved' or result.info.status_val == 1:
            return result.x[:nu]
        # Fallback to unconstrained if OSQP fails
        return np.clip(-np.linalg.solve(H, h_qp)[:nu], -self.F_max, self.F_max)
