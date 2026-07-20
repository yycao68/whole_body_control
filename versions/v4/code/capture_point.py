#!/usr/bin/env python3
"""Shared, controller-independent capture-point gait stabilizer.

Two strictly separated modules, per the design of the diagnostic study:

  A. Discrete touchdown gait-frame anchoring.  At each reliable touchdown the
     gait frame is translated by ``actual_stance - nominal_stance`` and held
     constant for the step, so the CoM/DCM/footstep reference follows the
     stepped feet.  This is a geometric-consistency fix, not controller tuning.

  B. One-step predicted-touchdown DCM foot placement.  The DCM at the next
     touchdown is predicted from the *measured* CoM state; its deviation from
     the anchored nominal is mapped to a bounded lateral foothold correction by
     a one-variable closed-form QP (dcm / nominal / smoothness), projected onto
     a legal step width, and frozen in late swing.

The stabilizer reads only the physical CoM state and the measured/nominal
footholds — never a controller's disturbance estimate or command — so it is
identical for every controller with one parameter set.

EXPERIMENTAL / NOT ENABLED IN THE REPORTED BENCHMARKS.  The historical 135 mm
quantity was a DCM-to-stance diagnostic of the superseded 0.8 s-step planner,
not the controlled lateral tracking error.  The shared publication planner now
uses a finite double-support ZMP transfer and a conservative continuous gait;
``run_trial`` still leaves this online foot-placement module disabled so its
effect cannot be confused with interaction-residual compensation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StabilizerParams:
    q_dcm: float = 1.0            # weight on the predicted-touchdown DCM error
    r_nom: float = 0.2           # weight on deviation from the nominal foothold
    r_smooth: float = 0.5        # weight on step-to-step correction change
    max_lat_corr: float = 0.04   # |lateral foothold correction| bound [m]
    min_step_width: float = 0.06  # lateral stance->swing separation bounds [m]
    max_step_width: float = 0.24
    freeze_time_s: float = 0.13  # freeze the target when remaining swing < this
    growth_cap: float = 1.5      # cap on omega*T_r inside exp() for safety
    lateral_only: bool = True


@dataclass
class CapturePointStabilizer:
    omega: float
    params: StabilizerParams = field(default_factory=StabilizerParams)
    gait_offset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    _prev_corr_y: float = 0.0
    _swing: str | None = None
    _nominal_target: np.ndarray | None = None
    _frozen_target: np.ndarray | None = None

    # ---- Module A: discrete gait-frame anchoring -----------------------
    def anchor(self, xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy, float) + self.gait_offset

    def on_touchdown(self, actual_stance_xy: np.ndarray,
                     nominal_stance_xy: np.ndarray) -> None:
        """Discrete update at a reliable touchdown; held for the step."""
        self.gait_offset = (np.asarray(actual_stance_xy, float).reshape(2)
                            - np.asarray(nominal_stance_xy, float).reshape(2))
        self._frozen_target = None

    def on_new_swing(self, swing: str, nominal_target_xy: np.ndarray) -> None:
        self._swing = swing
        self._nominal_target = np.asarray(nominal_target_xy, float).reshape(2).copy()
        self._frozen_target = None

    # ---- Module B: one-step predicted-touchdown DCM foot placement -----
    def _predict_td_dcm(self, dcm_xy, stance_xy, remaining_time):
        g = math.exp(min(max(self.omega * remaining_time, 0.0), self.params.growth_cap))
        return np.asarray(stance_xy, float) + g * (np.asarray(dcm_xy, float)
                                                   - np.asarray(stance_xy, float))

    def foot_placement(self, *, measured_dcm, desired_dcm,
                       measured_stance_xy, nominal_stance_xy,
                       nominal_next_foot_xy, remaining_time, next_is_left):
        """Return the corrected swing-foot target (anchored + bounded DCM step)."""
        p = self.params
        anchored_next = self.anchor(nominal_next_foot_xy)
        if remaining_time <= p.freeze_time_s and self._frozen_target is not None:
            return self._frozen_target.copy()

        xi_td = self._predict_td_dcm(measured_dcm, measured_stance_xy, remaining_time)
        xi_td_des = self._predict_td_dcm(self.anchor(desired_dcm),
                                         self.anchor(nominal_stance_xy), remaining_time)
        e_y = float(xi_td[1] - xi_td_des[1])
        # One-variable closed-form QP: min q(e - dp)^2 + r_nom dp^2 + r_smooth(dp - dp_prev)^2
        raw = (p.q_dcm * e_y + p.r_smooth * self._prev_corr_y) / (
            p.q_dcm + p.r_nom + p.r_smooth)
        corr_y = float(np.clip(raw, -p.max_lat_corr, p.max_lat_corr))

        target = anchored_next.copy()
        target[1] += corr_y
        if not p.lateral_only:
            pass  # sagittal correction reserved for a later version
        # Project onto a legal step width relative to the actual stance foot.
        rel = float(target[1] - measured_stance_xy[1])
        if next_is_left:
            rel = float(np.clip(rel, p.min_step_width, p.max_step_width))
        else:
            rel = float(np.clip(rel, -p.max_step_width, -p.min_step_width))
        target[1] = measured_stance_xy[1] + rel

        if remaining_time <= p.freeze_time_s:
            self._frozen_target = target.copy()
            self._prev_corr_y = corr_y
        return target
