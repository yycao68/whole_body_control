"""Dimension-generic normalized interaction MPC.

The model is the exact-ZOH double integrator used throughout the v3 paper:

    x[k+1] = A x[k] + B (u[k] + d_hat)

where x = [e; e_dot].  The optimizer uses the input-centered variable
v = u + d_hat, so constant disturbance cancellation is not penalized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
import scipy.sparse as sp


@dataclass
class NormalizedMPC:
    dim: int
    dt: float
    horizon: int
    q_pos: float
    q_vel: float
    r: float
    qf_pos: float | None = None
    qf_vel: float | None = None
    u_max: float | np.ndarray | None = None

    def __post_init__(self):
        n = self.dim
        self.A = np.block(
            [
                [np.eye(n), self.dt * np.eye(n)],
                [np.zeros((n, n)), np.eye(n)],
            ]
        )
        self.B = np.vstack((0.5 * self.dt**2 * np.eye(n), self.dt * np.eye(n)))
        self.Q = np.diag([self.q_pos] * n + [self.q_vel] * n)
        self.R = self.r * np.eye(n)
        self.Qf = np.diag(
            [self.qf_pos if self.qf_pos is not None else self.q_pos] * n
            + [self.qf_vel if self.qf_vel is not None else self.q_vel] * n
        )
        self._build_lifted_matrices()
        self._solver = None
        self._poly_solver = None
        self._H_poly = None
        self._h_poly = None
        self.last_polytope_failed = False
        self._u_lower = None
        self._u_upper = None
        self.last_u_sequence = None
        self.last_input_lower = None
        self.last_input_upper = None
        self.last_bound_active = False
        if self.u_max is not None:
            lim = np.asarray(self.u_max, dtype=float)
            if lim.ndim == 0:
                lim = np.full(self.dim, float(lim))
            lim = lim.reshape(self.dim)
            if np.any(~np.isfinite(lim)) or np.any(lim < 0.0):
                raise ValueError("u_max must be finite and nonnegative")
            self.update_input_box(-lim, lim)

    def _setup_constrained_solver(self):
        n_dec = self.horizon * self.dim
        self._solver = osqp.OSQP()
        self._solver.setup(
            P=sp.csc_matrix(2.0 * (self.H + 1e-10 * np.eye(self.H.shape[0]))),
            q=np.zeros(n_dec),
            A=sp.eye(n_dec, format="csc"),
            l=-np.inf * np.ones(n_dec),
            u=np.inf * np.ones(n_dec),
            verbose=False,
            polish=False,
            eps_abs=1e-7,
            eps_rel=1e-7,
            max_iter=1000,
        )

    def update_input_box(
        self,
        u_lower: np.ndarray | float,
        u_upper: np.ndarray | float,
    ) -> None:
        """Set asymmetric residual-command bounds for the next MPC solve.

        Each argument may be scalar, shape ``(dim,)`` for a box frozen over
        the horizon, or shape ``(horizon, dim)`` for stage-scheduled boxes.
        Updating these bounds does not rebuild the canonical model or Hessian.
        """

        def _stages(value: np.ndarray | float, name: str) -> np.ndarray:
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 0:
                arr = np.full((self.horizon, self.dim), float(arr))
            elif arr.shape == (self.dim,):
                arr = np.tile(arr, (self.horizon, 1))
            elif arr.shape != (self.horizon, self.dim):
                raise ValueError(
                    f"{name} must be scalar, ({self.dim},), or "
                    f"({self.horizon}, {self.dim})"
                )
            if np.any(~np.isfinite(arr)):
                raise ValueError(f"{name} must contain only finite values")
            return arr.copy()

        lower = _stages(u_lower, "u_lower")
        upper = _stages(u_upper, "u_upper")
        if np.any(lower > upper):
            raise ValueError("u_lower must not exceed u_upper")
        self._u_lower = lower
        self._u_upper = upper
        if self._solver is None:
            self._setup_constrained_solver()

    def clear_input_box(self) -> None:
        """Return to the unconstrained canonical predictor."""
        self._u_lower = None
        self._u_upper = None
        self._H_poly = None
        self._h_poly = None

    def update_input_polytope(self, H: np.ndarray, h: np.ndarray) -> None:
        """Constrain every horizon stage by the realization polytope H u <= h.

        This is the general form of the residual-command constraint supplied by
        the 1 kHz realizer (torque bounds, friction pyramid, unilateral force
        mapped through the input sensitivity).  It is frozen over the short
        horizon.  The canonical pair (A, B) and the condensed Hessian are NOT
        touched -- only the constraint rows change, which is the whole point:
        a contact transition switches the admissible geometry, not the dynamics.
        """
        H = np.atleast_2d(np.asarray(H, dtype=float))
        h = np.asarray(h, dtype=float).reshape(-1)
        if H.shape[1] != self.dim or H.shape[0] != h.size:
            raise ValueError("H must be (n_con, dim) and h must be (n_con,)")
        if not (np.all(np.isfinite(H)) and np.all(np.isfinite(h))):
            raise ValueError("H and h must be finite")
        rebuild = (
            self._poly_solver is None
            or self._H_poly is None
            or self._H_poly.shape != H.shape
        )
        self._H_poly = H.copy()
        self._h_poly = h.copy()
        self._u_lower = None          # polytope supersedes the box
        self._u_upper = None
        # A = blockdiag(H, ..., H) has a FIXED pattern once H's shape is fixed, so
        # it is built once and only its data is refreshed.  Rebuilding the
        # block-diagonal every solve costs ~30 ms and would defeat the whole
        # point of a fast predictor.  The pattern is pinned dense so that a zero
        # entry in H cannot silently drop a column.
        if rebuild:
            n_dec = self.horizon * self.dim
            pattern = sp.block_diag(
                [sp.csc_matrix(np.ones(H.shape))] * self.horizon, format="csc")
            self._poly_A = pattern
            self._poly_solver = osqp.OSQP()
            self._poly_solver.setup(
                P=sp.csc_matrix(2.0 * (self.H + 1e-10 * np.eye(self.H.shape[0]))),
                q=np.zeros(n_dec),
                A=self._poly_A,
                l=-np.inf * np.ones(self.horizon * H.shape[0]),
                u=np.inf * np.ones(self.horizon * H.shape[0]),
                verbose=False, polish=False,
                eps_abs=1e-6, eps_rel=1e-6, max_iter=6000,
            )
        # CSC data of a dense (m, dim) block is column-major; block_diag just
        # concatenates the blocks' data, so the whole array is one tile.
        self._poly_solver.update(Ax=np.tile(H.flatten(order="F"), self.horizon))

    def _build_lifted_matrices(self):
        n_x = 2 * self.dim
        n_u = self.dim
        N = self.horizon

        Phi = np.zeros((N * n_x, n_x))
        Gamma = np.zeros((N * n_x, N * n_u))
        A_power = np.eye(n_x)
        for i in range(N):
            A_power = self.A @ A_power
            Phi[i * n_x:(i + 1) * n_x, :] = A_power
            for j in range(i + 1):
                Aij = np.linalg.matrix_power(self.A, i - j)
                Gamma[i * n_x:(i + 1) * n_x, j * n_u:(j + 1) * n_u] = Aij @ self.B

        Q_blocks = [self.Q] * (N - 1) + [self.Qf]
        Qbar = np.zeros((N * n_x, N * n_x))
        Rbar = np.zeros((N * n_u, N * n_u))
        for i, Qi in enumerate(Q_blocks):
            Qbar[i * n_x:(i + 1) * n_x, i * n_x:(i + 1) * n_x] = Qi
        for i in range(N):
            Rbar[i * n_u:(i + 1) * n_u, i * n_u:(i + 1) * n_u] = self.R

        H = Gamma.T @ Qbar @ Gamma + Rbar
        K_lift = np.linalg.solve(H + 1e-10 * np.eye(H.shape[0]), Gamma.T @ Qbar @ Phi)

        self.Phi = Phi
        self.Gamma = Gamma
        self.H = H
        self.Qbar = Qbar
        self.K0 = K_lift[:n_u, :]

    def solve(self, x: np.ndarray, d_hat: np.ndarray | None = None) -> np.ndarray:
        """Return the first residual-acceleration command u[0]."""
        x = np.asarray(x, dtype=float).reshape(2 * self.dim)
        if d_hat is None:
            d_hat = np.zeros(self.dim)
        d_hat = np.asarray(d_hat, dtype=float).reshape(self.dim)

        if self._H_poly is not None:
            # Stagewise polytope on the physical command U: H u_j <= h.  With the
            # input-centered variable V = U + d_hat this is H v_j <= h + H d_hat.
            gradient = self.Gamma.T @ self.Qbar @ (self.Phi @ x)
            ub = np.tile(self._h_poly + self._H_poly @ d_hat, self.horizon)
            self._poly_solver.update(q=2.0 * gradient,
                                     l=-np.inf * np.ones(ub.size), u=ub)
            result = self._poly_solver.solve()
            if result.x is None or result.info.status_val not in (1, 2):
                # The fallback must itself be ADMISSIBLE.  -d_hat (the offset-free
                # cancelling input) is not: it can lie outside H u <= h, and
                # returning it would hand the realizer a command the contacts
                # cannot produce -- exactly what this constraint exists to stop.
                # u = 0 is always admissible whenever the nominal is feasible
                # (then h >= 0), so fall back to it.
                self.last_u_sequence = None
                self.last_bound_active = False
                self.last_polytope_failed = True
                return np.zeros(self.dim)
            self.last_polytope_failed = False
            self.last_u_sequence = result.x.reshape(self.horizon, self.dim) - d_hat
            u0 = result.x[:self.dim] - d_hat
            slack = self._h_poly - self._H_poly @ u0
            self.last_bound_active = bool(np.any(slack <= 1e-6))
            return u0

        if self._u_lower is None:
            v0 = -self.K0 @ x
            self.last_u_sequence = None
            self.last_input_lower = None
            self.last_input_upper = None
            self.last_bound_active = False
            return v0 - d_hat

        # Optimize the complete input-centered horizon V.  The physical
        # residual command is U = V - d_hat, so stagewise bounds on U become
        # d_hat-lim <= V_j <= d_hat+lim at every horizon sample.
        dbar = np.tile(d_hat, self.horizon)
        lower = self._u_lower.reshape(-1)
        upper = self._u_upper.reshape(-1)
        gradient = self.Gamma.T @ self.Qbar @ (self.Phi @ x)
        self._solver.update(q=2.0 * gradient, l=dbar + lower, u=dbar + upper)
        result = self._solver.solve()
        if result.x is None or result.info.status_val not in (1, 2):
            raise RuntimeError(f"normalized MPC solve failed: {result.info.status}")
        self.last_u_sequence = result.x.reshape(self.horizon, self.dim) - d_hat
        self.last_input_lower = self._u_lower.copy()
        self.last_input_upper = self._u_upper.copy()
        tol = 5e-6
        self.last_bound_active = bool(
            np.any(self.last_u_sequence <= self._u_lower + tol)
            or np.any(self.last_u_sequence >= self._u_upper - tol)
        )
        return result.x[:self.dim] - d_hat


class RandomWalkDisturbanceObserver:
    """Small steady-state-friendly observer for x=[e,e_dot], d constant."""

    def __init__(self, dim: int, dt: float, q_d: float = 0.25, r_y: float = 2e-4):
        self.dim = dim
        self.dt = dt
        n = dim
        self.Aa = np.block(
            [
                [np.eye(n), dt * np.eye(n), 0.5 * dt**2 * np.eye(n)],
                [np.zeros((n, n)), np.eye(n), dt * np.eye(n)],
                [np.zeros((n, n)), np.zeros((n, n)), np.eye(n)],
            ]
        )
        self.Ba = np.vstack((0.5 * dt**2 * np.eye(n), dt * np.eye(n), np.zeros((n, n))))
        self.C = np.hstack((np.eye(n), np.zeros((n, n)), np.zeros((n, n))))
        self.Q = np.diag([1e-7] * n + [1e-5] * n + [q_d] * n)
        self.R = r_y * np.eye(n)
        self.z = np.zeros(3 * n)
        self.P = np.eye(3 * n)

    def reset(self):
        self.z[:] = 0.0
        self.P[:] = np.eye(3 * self.dim)

    def step(self, y: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y, dtype=float).reshape(self.dim)
        u = np.asarray(u, dtype=float).reshape(self.dim)

        # Correct the current prior with y_k, then propagate with u_k to form
        # the prior for k+1.  This avoids comparing y_k against a state already
        # advanced by the newly computed u_k.
        innovation = y - self.C @ self.z
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.z = self.z + K @ innovation
        self.P = (np.eye(self.P.shape[0]) - K @ self.C) @ self.P
        d_corrected = self.z[2 * self.dim:].copy()
        self.z = self.Aa @ self.z + self.Ba @ u
        self.P = self.Aa @ self.P @ self.Aa.T + self.Q
        return d_corrected, innovation
