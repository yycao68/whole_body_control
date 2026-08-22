"""
Interaction MPC — receding-horizon QP in acceleration-input form (Section V).

Reformulation (force-input -> acceleration-input)
-------------------------------------------------
The decision variable is the residual acceleration u (not the corrective force
F_mpc). With the change of variables  u = -Lambda_arm^{-1} F_mpc  the normalized
error dynamics become

        e_ddot = u + d(t)          (boxed Eq. 19)

whose exact ZOH discretization has BOTH matrices constant and config-invariant:

        A_d = [[I, dt I],[0, I]]        B_d = [½dt² I ; dt I]        (Eq. 20)

Consequences that this file realizes:
  * The free-response rollout Phi, the prediction map Gamma, and the Hessian
    H = Gammaᵀ Q̄ Gamma + R̄ are all CONSTANT — precomputed once, reused across
    every contact mode and configuration (no per-mode Hessian rebuild).
  * All configuration dependence relocates to (a) a static per-sample recovery
    F_arm = Lambda_arm(q)(e_ddot_d + u) + mu = F_ff + Lambda_arm u, and
    (b) the constraint rows, which bound the TOTAL delivered control, not the
    corrective increment:

        ‖F_ff + Lambda_arm(q_k) u_k‖_inf ≤ F_max            (force cap, Eq. 22)
        ‖tau_ff + Jᵀ(q_k) Lambda_arm(q_k) u_k‖_inf ≤ tau_max (optional torque cap)

    Both are affine in u; the Hessian is untouched, only the constraint
    matrix/bounds refresh per step.
  * The augmented disturbance model A_aug = [[A_d, B_d],[0, I]] is likewise
    constant (Eq. 23); the estimated disturbance d_hat is an acceleration.

The input weight R is supplied in force coordinates (the physically meaningful
tuning) and reflected once through the NOMINAL task inertia,
R_u = Lambda_nomᵀ R Lambda_nom, so the constant-H acceleration penalty matches
the former force penalty at the nominal inertia while keeping H config-invariant.
"""

import numpy as np
import osqp
import scipy.sparse as sp


class ImpedanceMPC:
    """
    Receding-horizon QP for arm end-effector tracking, acceleration-input form.

    State:  x_e = [e^T, e_dot^T]^T ∈ ℝ^6   (position + velocity error)
    Input:  u   ∈ ℝ^3                       (residual acceleration)

    A_d = [[I, dt I],[0, I]]   (constant)
    B_d = [½dt² I ; dt I]      (constant, config-invariant)
    Recovery: F_mpc = -Lambda_arm(q) u    (so callers' F_arm = -F_mpc = Lambda_arm u)
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
        self._R_force = R           # input weight in FORCE coordinates (see header)

        # Constant state-transition matrix A_d (Eq. 20)
        self.A_d = np.block([
            [np.eye(3),   dt * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)]
        ])

        # Constant acceleration-input matrix B_d = [½dt² I ; dt I] (Eq. 20).
        # Unlike the former force-input B_d^(m) = [-½dt² Λ^{-1}; -Λ^{-1} dt], this
        # carries NO task inertia — it is identical in every contact mode.
        self.B_d = np.vstack([
            0.5 * dt * dt * np.eye(3),
            dt * np.eye(3)
        ])

        # Precomputed free-response rollout matrix Phi (N*nxe × nxe)
        nxe, N_ = self.nxe, self.N
        self._Phi = np.zeros((N_ * nxe, nxe))
        A_pow = self.A_d.copy()
        for i in range(N_):
            self._Phi[i*nxe:(i+1)*nxe, :] = A_pow
            A_pow = self.A_d @ A_pow

        # Constant prediction map Gamma (N*nxe × N*nu) — built ONCE with the
        # constant B_d, reused for every configuration and contact mode.
        self._Gamma = np.zeros((N_ * nxe, N_ * self.nu))
        for i in range(N_):
            for j in range(i + 1):
                Ak = np.linalg.matrix_power(self.A_d, i - j)
                self._Gamma[i*nxe:(i+1)*nxe, j*self.nu:(j+1)*self.nu] = Ak @ self.B_d

        # Constant lifted state-cost products. Gamma and the prediction coupling
        # Gammaᵀ Q̄ Gamma are configuration-invariant (constant B_d) and computed
        # ONCE. Only the input-weight block R̄ below is configuration-referenced.
        self._Q_bar = np.kron(np.eye(N_), Q)
        self._GtQ   = self._Gamma.T @ self._Q_bar            # gradient map (const)
        self._GtQG  = self._GtQ @ self._Gamma                # prediction coupling (const)

        # Mode bookkeeping. The predictor (A_d, B_d, Gamma, Gammaᵀ Q̄ Gamma) is
        # constant; the input weight R_u = Λᵀ R Λ is referenced to the contact-
        # mode task inertia so the acceleration penalty carries the same Λ-adaptive
        # damping as the force penalty ‖Λu‖²_R. H = Gammaᵀ Q̄ Gamma + R̄(Λ) is
        # therefore assembled from a constant term plus a cheap block-diagonal
        # refresh — not rebuilt from scratch — and cached per mode (5% Λ change
        # triggers a refactorization).
        self._mode_library = {}
        self._Lambda_live  = None

        # Last residual acceleration applied (estimator input for kalman.predict)
        self.last_u = np.zeros(self.nu)

        # OSQP instance (re-used across solves for warm-starting)
        self._osqp = None
        self._osqp_nnz = None

    # ------------------------------------------------------------------
    def _assemble_H(self, Lambda_arm):
        """Assemble H = Gammaᵀ Q̄ Gamma + R̄(Λ) from the constant prediction
        coupling plus the Λ-referenced input weight R_u = Λᵀ R Λ."""
        R_u   = Lambda_arm.T @ self._R_force @ Lambda_arm
        R_bar = np.kron(np.eye(self.N), R_u)
        H     = self._GtQG + R_bar
        return H, np.linalg.inv(H), R_bar

    def precompute_mode(self, mode_key, Lambda_arm):
        """
        Offline registration for a contact mode. The predictor (A_d, B_d, Gamma,
        Gammaᵀ Q̄ Gamma) is constant; here only the input-weight block R̄(Λ) and
        the resulting Hessian factorization are assembled for `mode_key`.
        mode_key   : any hashable id (e.g. frozenset of active foot names)
        Lambda_arm : (3,3) task-space inertia for this mode
        """
        Lambda_arm = np.asarray(Lambda_arm, float)
        H, H_inv, R_bar = self._assemble_H(Lambda_arm)
        self._Lambda_live = Lambda_arm.copy()
        self._mode_library[mode_key] = dict(
            B_d=self.B_d, Gamma=self._Gamma, H=H, H_inv=H_inv, R_bar=R_bar,
            Lambda_arm=Lambda_arm.copy(),
        )
        return self._mode_library[mode_key]

    def get_or_update_mode(self, mode_key, Lambda_arm):
        """Return the cached mode; refresh the input-weight block and Hessian
        factorization if Λ changed by more than 5% (the constant prediction
        coupling Gammaᵀ Q̄ Gamma is never recomputed)."""
        Lambda_arm = np.asarray(Lambda_arm, float)
        self._Lambda_live = Lambda_arm.copy()
        if mode_key not in self._mode_library:
            return self.precompute_mode(mode_key, Lambda_arm)
        cached = self._mode_library[mode_key]
        diff = np.linalg.norm(Lambda_arm - cached['Lambda_arm']) / (
               np.linalg.norm(cached['Lambda_arm']) + 1e-8)
        if diff > 0.05:
            return self.precompute_mode(mode_key, Lambda_arm)
        return cached

    # ------------------------------------------------------------------
    def solve(self, x_e, Lambda_arm, mode_key=None,
              d_hat=None, use_osqp=True, F_ff=None,
              J=None, tau_ff=None, tau_max=None):
        """
        Solve the receding-horizon QP for the residual acceleration u and return
        the corrective force F_mpc = -Lambda_arm u.

        Parameters
        ----------
        x_e        : (6,) tracking error state [e; e_dot]
        Lambda_arm : (3,3) current task-space inertia (recovery + constraint)
        mode_key   : hashable contact-mode id (default: single mode)
        d_hat      : (3,) estimated ACCELERATION disturbance; None → no Kalman
        use_osqp   : enforce the caps as a QP; False → unconstrained + clip
        F_ff       : (3,) task-space feedforward force applied ALONGSIDE F_mpc.
                     The force cap bounds the TOTAL F_ff + Lambda_arm u. Default 0.
        J, tau_ff, tau_max : optional torque cap on the TOTAL joint torque
                     tau_ff + Jᵀ Lambda_arm u (J is the (3×n) task Jacobian block,
                     tau_ff the feedforward torque, tau_max the (n,) limit).

        Returns
        -------
        F_mpc : (3,) corrective force [N]  (caller applies F_arm = -F_mpc)
        """
        if mode_key is None:
            mode_key = 'default'
        mode = self.get_or_update_mode(mode_key, Lambda_arm)
        H, H_inv, R_bar = mode['H'], mode['H_inv'], mode['R_bar']

        N, nxe, nu = self.N, self.nxe, self.nu
        Lam = np.asarray(Lambda_arm, float)
        F_ff = np.zeros(nu) if F_ff is None else np.asarray(F_ff, float)

        # Free response with the estimated acceleration disturbance injected
        # through the constant B_d at every prediction step.
        d_stack = np.zeros(N * nu)
        if d_hat is not None:
            d_hat = np.asarray(d_hat, float)
            d_stack = np.tile(d_hat, N)
            x_free = np.zeros(N * nxe)
            x_k = x_e.copy()
            for i in range(N):
                x_k = self.A_d @ x_k + self.B_d @ d_hat
                x_free[i*nxe:(i+1)*nxe] = x_k
        else:
            x_free = self._Phi @ x_e

        # Gradient of the QP cost. The input effort is centered at the estimated
        # cancelling acceleration, ||U + d_hat||_{R_u}: since e_ddot = u + d, a
        # constant disturbance is cancelled by u -> -d_hat, giving offset-free
        # tracking. (D_bar d_hat = Gamma (1_N ⊗ d_hat).)
        h_qp = self._GtQ @ x_free + R_bar @ d_stack

        if not use_osqp:
            # Unconstrained solution (fast path): H_inv cached per mode, a pure
            # matrix-vector product; the cap is honored by clipping the total.
            u0 = (-H_inv @ h_qp)[:nu]
            F_corr = Lam @ u0
            F_total = np.clip(F_ff + F_corr, -self.F_max, self.F_max)
            F_corr = F_total - F_ff
        else:
            u0 = self._solve_osqp(H, h_qp, Lam, F_ff, J, tau_ff, tau_max)
            F_corr = Lam @ u0

        self.last_u = u0.copy()
        return -F_corr

    def _solve_osqp(self, H, h_qp, Lam, F_ff, J, tau_ff, tau_max):
        """QP over U with the caps on the TOTAL control. The prediction coupling
        is constant; the input-weight block of H and the constraint matrix/bounds
        refresh per contact mode."""
        N, nu = self.N, self.nu
        n_dec = N * nu

        P = sp.triu(sp.csc_matrix(H), format='csc')
        q = h_qp

        # Force cap on the total: -F_max ≤ F_ff + Lambda_arm u_k ≤ F_max.
        A_blocks = [sp.kron(sp.eye(N), sp.csc_matrix(Lam))]
        lb = [np.tile(-self.F_max - F_ff, N)]
        ub = [np.tile( self.F_max - F_ff, N)]

        # Optional torque cap on the total: -tau_max ≤ tau_ff + Jᵀ Lambda u_k ≤ tau_max
        if J is not None and tau_max is not None:
            JL = np.asarray(J, float).T @ Lam                # (n × 3)
            tau_ff = np.zeros(JL.shape[0]) if tau_ff is None else np.asarray(tau_ff, float)
            tau_max = np.asarray(tau_max, float)
            A_blocks.append(sp.kron(sp.eye(N), sp.csc_matrix(JL)))
            lb.append(np.tile(-tau_max - tau_ff, N))
            ub.append(np.tile( tau_max - tau_ff, N))

        A  = sp.vstack(A_blocks, format='csc')
        lb = np.concatenate(lb)
        ub = np.concatenate(ub)

        # The constraint matrix depends on Lambda_arm(q) (and J), so its sparsity
        # pattern/shape can change between modes; rebuild OSQP when it does.
        if self._osqp is None or self._osqp_nnz != (P.nnz, A.nnz, A.shape[0]):
            prob = osqp.OSQP()
            prob.setup(P, q, A, lb, ub,
                       warm_starting=True, verbose=False,
                       max_iter=1000, eps_abs=1e-4, eps_rel=1e-4,
                       polish=True)
            self._osqp = prob
            self._osqp_nnz = (P.nnz, A.nnz, A.shape[0])
        else:
            self._osqp.update(q=q, l=lb, u=ub, Px=P.data, Ax=A.data)

        result = self._osqp.solve()
        if result.info.status == 'solved' or result.info.status_val == 1:
            return result.x[:nu]
        # Fallback: unconstrained then rely on the caller-side clip
        return (-np.linalg.solve(H, h_qp))[:nu]
