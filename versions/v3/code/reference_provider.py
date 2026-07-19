"""External nominal walking-reference provider for the uneven-ground benchmark.

The provider deliberately has no terrain input.  It produces the same DCM/ZMP
body reference, contact schedule, and swing-foot targets for every controller
and terrain.  Terrain therefore perturbs execution without changing the plan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


G = 9.81


@dataclass(frozen=True)
class ReferenceSample:
    stance: tuple[str, ...]
    swing: str | None
    swing_progress: float
    swing_start_xy: np.ndarray | None
    swing_target_xy: np.ndarray | None
    dcm_xy: np.ndarray
    zmp_xy: np.ndarray
    phase_index: int
    target_index: int = -1     # index in self.zmp of the swing foot's target
    seg_tau: float = 0.0       # time into the current segment
    seg_duration: float = 1.0  # duration of the current segment


class DCMReferenceProvider:
    """Small external DCM/footstep reference used as shared infrastructure."""

    def __init__(
        self,
        left0: np.ndarray,
        right0: np.ndarray,
        *,
        step_length: float,
        n_steps: int,
        com_height: float,
        step_time: float = 0.62,
        double_support_time: float = 0.12,
        settle_time: float = 0.8,
        lateral_zmp_scale: float = 1.0,
    ) -> None:
        if com_height <= 0.0 or step_time <= 0.0 or n_steps < 1:
            raise ValueError("positive com_height/step_time and n_steps >= 1 required")
        self.omega = float(np.sqrt(G / com_height))
        self.step_time = float(step_time)
        self.double_support_time = float(double_support_time)
        self.settle_time = float(settle_time)
        self.left_plant = np.asarray(left0, float).reshape(2).copy()
        self.right_plant = np.asarray(right0, float).reshape(2).copy()

        x0 = 0.5 * (self.left_plant[0] + self.right_plant[0])
        yl = self.left_plant[1] * lateral_zmp_scale
        yr = self.right_plant[1] * lateral_zmp_scale
        self.zmp = [np.array([x0, 0.0])]
        self.side = ["both"]
        for k in range(n_steps):
            side = "left" if k % 2 == 0 else "right"
            self.side.append(side)
            self.zmp.append(np.array([x0 + k * step_length, yl if side == "left" else yr]))
        self.zmp.append(self.zmp[-1].copy())
        self.side.append("both")

        self.times = [0.0, self.settle_time]
        for _ in range(1, len(self.zmp) - 1):
            self.times.append(self.times[-1] + self.step_time)
        self.total_time = self.times[-1] + self.settle_time

        self.dcm_initial = [None] * len(self.zmp)
        self.dcm_initial[-1] = self.zmp[-1].copy()
        for k in range(len(self.zmp) - 2, -1, -1):
            dt = self.times[k + 1] - self.times[k]
            self.dcm_initial[k] = self.zmp[k] + (
                self.dcm_initial[k + 1] - self.zmp[k]
            ) * np.exp(-self.omega * dt)

        # Phase-synchronized playback state (used only by sample_synced).
        self._pt = 0.0        # reference phase time (decoupled from the clock)
        self._wait = False    # frozen at a single-support boundary awaiting touchdown

    def reset_sync(self) -> None:
        self._pt = 0.0
        self._wait = False

    def sample_synced(self, dt: float, swing_down: bool) -> "ReferenceSample":
        """Advance the reference by gait *phase*, not the wall clock, and
        re-synchronize the single-support boundary to measured touchdown.

        ``swing_down`` is the measured contact of the current swing foot. The
        reference DCM is held at the step's capture point (the segment-boundary
        value) if the foot lands late, and snapped forward if it lands early, so
        the reference stays phase-consistent with the actual contact sequence
        instead of drifting on the clock. ``sample`` is left untouched.
        """
        if self._wait:
            if swing_down:            # late touchdown finally occurred
                self._wait = False
                self._pt += dt
        else:
            self._pt += dt
            k, _, _ = self._segment(self._pt)
            single_support = self.side[k] != "both"
            boundary = self.times[min(k + 1, len(self.times) - 1)]
            ss_start = self.times[k] + self.double_support_time
            if single_support and swing_down and ss_start <= self._pt < boundary:
                self._pt = boundary   # early touchdown: snap phase forward
            elif single_support and self._pt >= boundary and not swing_down:
                self._pt = boundary - 1e-6   # freeze at the capture point
                self._wait = True
        return self.sample(self._pt)

    def _segment(self, t: float) -> tuple[int, float, float]:
        for k in range(len(self.zmp) - 1):
            if self.times[k] <= t < self.times[k + 1]:
                return k, t - self.times[k], self.times[k + 1] - self.times[k]
        return len(self.zmp) - 1, 0.0, 1.0

    def sample(self, t: float) -> ReferenceSample:
        k, tau, duration = self._segment(float(t))
        dcm = self.zmp[k] + (self.dcm_initial[k] - self.zmp[k]) * np.exp(
            self.omega * tau
        )
        side = self.side[k]
        if side == "both" or tau < self.double_support_time:
            return ReferenceSample(
                ("left", "right"), None, 0.0, None, None,
                dcm, self.zmp[k].copy(), k, -1, tau, duration,
            )

        swing = "right" if side == "left" else "left"
        target = None
        target_index = -1
        for j in range(k + 1, len(self.side)):
            if self.side[j] == swing:
                target = self.zmp[j].copy()
                target_index = j
                break
        start = self.right_plant if swing == "right" else self.left_plant
        if target is None:
            target = start.copy()
        progress = np.clip(
            (tau - self.double_support_time)
            / max(duration - self.double_support_time, 1e-3),
            0.0,
            1.0,
        )
        return ReferenceSample(
            (side,), swing, float(progress), start.copy(), target,
            dcm, self.zmp[k].copy(), k, target_index, tau, duration,
        )

    def set_zmp(self, idx: int, xy: np.ndarray) -> None:
        """Move a footstep's ZMP and re-solve the backward DCM recursion so the
        body reference follows the stepped feet (closed-loop DCM plan)."""
        if idx < 1 or idx >= len(self.zmp) - 1:
            return
        self.zmp[idx] = np.asarray(xy, float).reshape(2).copy()
        for k in range(len(self.zmp) - 2, -1, -1):
            dt = self.times[k + 1] - self.times[k]
            self.dcm_initial[k] = self.zmp[k] + (
                self.dcm_initial[k + 1] - self.zmp[k]
            ) * np.exp(-self.omega * dt)

    def commit_plant(self, foot: str, xy: np.ndarray) -> None:
        if foot == "left":
            self.left_plant = np.asarray(xy, float).reshape(2).copy()
        elif foot == "right":
            self.right_plant = np.asarray(xy, float).reshape(2).copy()
        else:
            raise ValueError(f"unknown foot {foot!r}")

