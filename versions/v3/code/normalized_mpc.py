"""Dimension-generic normalized interaction MPC.

The model is the exact-ZOH double integrator used throughout the v3 paper:

    x[k+1] = A x[k] + B (u[k] + d_hat)

where x = [e; e_dot].  The optimizer uses the input-centered variable
v = u + d_hat, so constant disturbance cancellation is not penalized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
        self.K0 = K_lift[:n_u, :]

    def solve(self, x: np.ndarray, d_hat: np.ndarray | None = None) -> np.ndarray:
        """Return the first residual-acceleration command u[0]."""
        x = np.asarray(x, dtype=float).reshape(2 * self.dim)
        if d_hat is None:
            d_hat = np.zeros(self.dim)
        d_hat = np.asarray(d_hat, dtype=float).reshape(self.dim)

        v0 = -self.K0 @ x
        u0 = v0 - d_hat
        if self.u_max is not None:
            lim = np.asarray(self.u_max, dtype=float)
            if lim.ndim == 0:
                lim = np.full(self.dim, float(lim))
            u0 = np.clip(u0, -lim, lim)
        return u0


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

        self.z = self.Aa @ self.z + self.Ba @ u
        self.P = self.Aa @ self.P @ self.Aa.T + self.Q

        innovation = y - self.C @ self.z
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.z = self.z + K @ innovation
        self.P = (np.eye(self.P.shape[0]) - K @ self.C) @ self.P
        return self.z[2 * self.dim:].copy(), innovation
