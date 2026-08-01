"""Task-acceleration residual estimator used by the uneven-ground benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InteractionEstimate:
    interaction: np.ndarray
    realization: np.ndarray
    effective: np.ndarray


class FilteredAccelerationResidualEstimator:
    """Estimate matched residuals from measured and WBC-realized acceleration.

    ``interaction = a_measured - a_realized`` isolates plant/contact mismatch;
    ``realization = a_realized - a_commanded`` is the WBC execution mismatch;
    their sum is the effective residual seen by the canonical requested model.
    A first-order low-pass filter is used because finite-difference acceleration
    is noisy even in simulation.  No terrain height or future force is used.
    """

    def __init__(self, dim: int, dt: float, bandwidth_hz: float = 8.0) -> None:
        self.dim = int(dim)
        self.dt = float(dt)
        self.alpha = float(1.0 - np.exp(-2.0 * np.pi * bandwidth_hz * dt))
        self._velocity_previous: np.ndarray | None = None
        self._interaction = np.zeros(self.dim)
        self._realization = np.zeros(self.dim)

    def reset(self, velocity: np.ndarray | None = None) -> None:
        self._velocity_previous = (
            None if velocity is None else np.asarray(velocity, float).reshape(self.dim).copy()
        )
        self._interaction[:] = 0.0
        self._realization[:] = 0.0

    def step(
        self,
        velocity: np.ndarray,
        commanded_acceleration: np.ndarray,
        realized_acceleration: np.ndarray,
    ) -> InteractionEstimate:
        velocity = np.asarray(velocity, float).reshape(self.dim)
        commanded = np.asarray(commanded_acceleration, float).reshape(self.dim)
        realized = np.asarray(realized_acceleration, float).reshape(self.dim)
        if self._velocity_previous is None:
            measured = realized.copy()
        else:
            measured = (velocity - self._velocity_previous) / self.dt
        self._velocity_previous = velocity.copy()
        measured = np.clip(measured, -30.0, 30.0)
        interaction_raw = measured - realized
        realization_raw = realized - commanded
        self._interaction += self.alpha * (interaction_raw - self._interaction)
        self._realization += self.alpha * (realization_raw - self._realization)
        return InteractionEstimate(
            self._interaction.copy(),
            self._realization.copy(),
            self._interaction + self._realization,
        )

