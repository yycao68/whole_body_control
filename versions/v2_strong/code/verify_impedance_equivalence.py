"""
Verify Theorem 1: finite-horizon predictive interaction feedback converges
to the infinite-horizon LQR/impedance feedback as N increases.

This script does not use the full MuJoCo plant. It isolates the normalized
double-integrator interaction dynamics used in the paper and compares:
  - finite-horizon QP first-step feedback gains, and
  - the infinite-horizon discrete LQR gain.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are

from impedance_mpc import ImpedanceMPC


OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DT = 0.001
Q = np.diag([6e4, 6e4, 6e4, 60.0, 60.0, 60.0])
R = 0.01 * np.eye(3)
LAMBDA = np.diag([0.93, 1.07, 1.98])
HORIZONS = [2, 5, 10, 20, 40, 80, 160]


def first_step_gain(mpc: ImpedanceMPC, Lambda: np.ndarray) -> np.ndarray:
    """Return K_N where the unconstrained optimizer applies U0 = -K_N x."""
    mode = mpc.precompute_mode("nominal", Lambda)
    gain = mode["H_inv"] @ mode["Gamma"].T @ mpc._Q_bar @ mpc._Phi
    return gain[:3, :]


def infinite_lqr_gain(dt: float, Lambda: np.ndarray) -> np.ndarray:
    """Return K_inf for U = -K_inf x."""
    A = np.block([
        [np.eye(3), dt * np.eye(3)],
        [np.zeros((3, 3)), np.eye(3)],
    ])
    B = np.vstack([np.zeros((3, 3)), -np.linalg.inv(Lambda) * dt])
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def rollout(A: np.ndarray, B: np.ndarray, K: np.ndarray, x0: np.ndarray, steps=700):
    xs = np.zeros((steps, len(x0)))
    x = x0.copy()
    for i in range(steps):
        xs[i] = x
        u = -K @ x
        x = A @ x + B @ u
    return xs


def main():
    A = np.block([
        [np.eye(3), DT * np.eye(3)],
        [np.zeros((3, 3)), np.eye(3)],
    ])
    B = np.vstack([np.zeros((3, 3)), -np.linalg.inv(LAMBDA) * DT])
    K_inf = infinite_lqr_gain(DT, LAMBDA)

    rel_errors = []
    gains = {}
    for N in HORIZONS:
        mpc = ImpedanceMPC(N=N, dt=DT, Q=Q, R=R, F_max=1e6)
        K_N = first_step_gain(mpc, LAMBDA)
        gains[N] = K_N
        rel = np.linalg.norm(K_N - K_inf) / (np.linalg.norm(K_inf) + 1e-12)
        rel_errors.append(rel)

    x0 = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
    t = np.arange(700) * DT
    selected = [2, 10, 20, 80, 160]

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax[0].semilogy(HORIZONS, rel_errors, "o-", lw=2)
    ax[0].set_xlabel("Horizon N")
    ax[0].set_ylabel(r"$\|K_N-K_\infty\|/\|K_\infty\|$")
    ax[0].set_title("Finite-horizon gain convergence")
    ax[0].grid(True, which="both", alpha=0.3)

    xs_inf = rollout(A, B, K_inf, x0)
    ax[1].plot(t, 1000 * xs_inf[:, 0], "k--", lw=2.5, label=r"$N=\infty$ LQR")
    for N in selected:
        xs = rollout(A, B, gains[N], x0)
        ax[1].plot(t, 1000 * xs[:, 0], lw=1.5, label=f"N={N}")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("x-error [mm]")
    ax[1].set_title("Closed-loop response convergence")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    out = OUT_DIR / "impedance_equivalence.png"
    fig.savefig(out, dpi=180)

    print("Impedance-equivalence horizon sweep")
    for N, rel in zip(HORIZONS, rel_errors):
        print(f"  N={N:3d}: relative gain error={rel:.4e}")
    print(f"Figure saved to {out}")


if __name__ == "__main__":
    main()
