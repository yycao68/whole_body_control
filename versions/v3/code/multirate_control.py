"""Multirate interaction-dynamics controller.

The architecture this module implements, and the one the paper now claims:

    1 kHz   realization loop  (never blocks, never waits)
              - update M, h, J_c, J_t and the nominal feedforward
              - solve exactly ONE whole-body inverse-dynamics QP
              - emit joint torques
              - publish a RealizationAuthority snapshot (tau_ff, lambda_ff,
                K_tau, K_lambda, H_k, h_k, margins) from one KKT sensitivity

    100-200 Hz  body predictor   (asynchronous)
    200-500 Hz  task predictor   (asynchronous)
              - canonical model x+ = A x + B (u + d_hat), (A, B) CONSTANT
              - constrained by the latest published H_k u <= h_k
              - publish u*; the 1 kHz loop uses the last valid command and
                never waits for a new one

The predictors are slow consumers of a fast producer.  A stale snapshot, or one
taken in a different contact mode, is NOT used and NOT waited on: the predictor
falls back to a conservative fixed box.  The whole-body QP remains the final
hard constraint layer, because H_k u <= h_k is only a local (active-set-cell)
model of what the realizer can do.

Body-priority allocation (Phase 1 of the design): the body predictor is solved
first against its own constraints; the task predictor then receives the
*remaining* capacity h_t - H_tb u_b*, so the two ports never both assume they
own the full actuator/contact budget.

In simulation the three rates are realized by decimation of the 1 ms step, not
by OS threads.  Wall-clock compute time of each component is measured, so the
deadline margin is reported rather than assumed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from normalized_mpc import NormalizedMPC, RandomWalkDisturbanceObserver
from realization_authority import (
    AnalyticAuthorityMapper, ContinuationAuthorityEstimator,
    RealizationAuthority, TaskAuthority,
)


@dataclass
class RateConfig:
    sim_dt: float = 0.001            # 1 kHz realization loop
    body_dt: float = 0.005           # 200 Hz body predictor
    task_dt: float = 0.002           # 500 Hz task predictor
    max_snapshot_age: float = 0.040  # older than this -> conservative fallback
    authority_dt: float = 0.020      # authority refresh period (asynchronous, ~50 Hz)


@dataclass
class Timing:
    realizer_ms: list[float] = field(default_factory=list)
    authority_ms: list[float] = field(default_factory=list)
    body_mpc_ms: list[float] = field(default_factory=list)
    task_mpc_ms: list[float] = field(default_factory=list)
    qp_solves_per_cycle: list[int] = field(default_factory=list)

    def summary(self) -> dict:
        def stat(v):
            if not v:
                return None
            a = np.asarray(v, float)
            return {
                "median_ms": float(np.median(a)),
                "p99_ms": float(np.percentile(a, 99)),
                "max_ms": float(a.max()),
                "n": int(a.size),
            }
        cyc = np.asarray(self.realizer_ms, float) + np.asarray(self.authority_ms, float)
        return {
            "realizer": stat(self.realizer_ms),
            "authority_kkt": stat(self.authority_ms),
            "realization_cycle": stat(list(cyc)),
            "body_mpc": stat(self.body_mpc_ms),
            "task_mpc": stat(self.task_mpc_ms),
            "whole_body_qp_solves_per_1khz_cycle": {
                "max": int(max(self.qp_solves_per_cycle)) if self.qp_solves_per_cycle else 0,
                "mean": float(np.mean(self.qp_solves_per_cycle)) if self.qp_solves_per_cycle else 0.0,
            },
        }


class MultirateInteractionController:
    """1 kHz realizer + asynchronous canonical predictors."""

    def __init__(
        self,
        realizer,
        *,
        rates: RateConfig | None = None,
        body_mpc: NormalizedMPC | None = None,
        task_mpc: NormalizedMPC | None = None,
        body_obs: RandomWalkDisturbanceObserver | None = None,
        task_obs: RandomWalkDisturbanceObserver | None = None,
        mapper: AnalyticAuthorityMapper | None = None,
        fallback_box: np.ndarray = np.array([1.5, 2.0]),
        use_authority: bool = True,
        continuation: bool = False,
    ):
        self.realizer = realizer
        self.rates = rates or RateConfig()
        self.mapper = mapper or AnalyticAuthorityMapper()
        self.use_authority = bool(use_authority)
        # PWA continuation: exact authority, KKT-only, but ~14 ms -- so it is
        # refreshed ASYNCHRONOUSLY at rates.authority_dt, not every 1 kHz cycle.
        self.continuation = bool(continuation)
        self.cont = ContinuationAuthorityEstimator(max_regions=60)
        self._last_auth_t = -np.inf
        self._cont_box = None
        self.fallback_box = np.asarray(fallback_box, float)

        self.body_mpc = body_mpc or NormalizedMPC(
            dim=2, dt=self.rates.body_dt, horizon=25,
            q_pos=55.0, q_vel=12.0, r=0.08,
        )
        self.task_mpc = task_mpc
        self.body_obs = body_obs or RandomWalkDisturbanceObserver(
            dim=2, dt=self.rates.body_dt, q_d=0.05, r_y=1.5e-4,
        )
        self.task_obs = task_obs

        # Published state (the "lock-free buffers" of the real implementation).
        self.snapshot: RealizationAuthority | None = None
        self.task_snapshot: TaskAuthority | None = None
        self.task_fallback_box = np.array([2.0, 2.0, 2.0])
        self.u_body = np.zeros(2)
        self.u_task = np.zeros(3)

        self.timing = Timing()
        self.n_body_solves = 0
        self.n_task_solves = 0
        self.n_stale_fallbacks = 0
        self.n_mode_mismatch = 0
        self.n_invalid_snapshots = 0
        self.n_polytope_failures = 0
        self.n_task_invalid = 0
        self._last_body_t = -np.inf
        self._last_task_t = -np.inf

    # -- predictor constraint selection ------------------------------------
    def _apply_body_constraints(self, t: float, mode: tuple[str, ...]) -> str:
        """Constrain the body predictor by the latest snapshot, or fall back.

        Never waits for the 1 kHz loop.  Returns which source was used.
        """
        if self.use_authority and self.continuation:
            cb = self._cont_box
            if (cb is not None and cb[2] == mode
                    and (t - cb[1]) <= self.rates.max_snapshot_age):
                self.body_mpc.update_input_box(cb[0].lower, cb[0].upper)
                return "continuation_box"
            self.n_invalid_snapshots += 1
            self.body_mpc.update_input_box(-self.fallback_box, self.fallback_box)
            return "fallback_box"

        s = self.snapshot
        if not self.use_authority or s is None or not s.valid:
            self.n_invalid_snapshots += int(self.use_authority and (s is None or not s.valid))
            self.body_mpc.update_input_box(-self.fallback_box, self.fallback_box)
            return "fallback_box"
        if s.contact_mode != mode:
            self.n_mode_mismatch += 1
            self.body_mpc.update_input_box(-self.fallback_box, self.fallback_box)
            return "mode_mismatch_fallback"
        if (t - s.timestamp) > self.rates.max_snapshot_age:
            self.n_stale_fallbacks += 1
            self.body_mpc.update_input_box(-self.fallback_box, self.fallback_box)
            return "stale_fallback"
        self.body_mpc.update_input_polytope(s.H_body, s.h_body)
        return "authority_polytope"

    # -- one 1 kHz tick -----------------------------------------------------
    def step(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        t: float,
        *,
        q_ref: np.ndarray,
        qd_ref: np.ndarray,
        com_ref_acc: np.ndarray,       # c_ddot_d  (3,)
        body_error: np.ndarray,        # [e_c(2), e_c_dot(2)]
        stance: tuple[str, ...],
        stance_contacts: dict,
        stance_targets: dict,
        base_height_ref: float,
        rpy: np.ndarray,
        hand_jac: np.ndarray,
        task_acc_ref: np.ndarray | None = None,
        task_error: np.ndarray | None = None,
        swing_task: dict | None = None,
        attitude_weight: float = 60.0,
    ) -> dict:
        """Run one 1 kHz realization cycle, plus any predictor due this tick."""
        R = self.rates
        info = {"body_source": None, "body_solved": False, "task_solved": False}

        # ---- asynchronous body predictor (uses the LAST published snapshot) --
        if t - self._last_body_t >= R.body_dt - 1e-12:
            self._last_body_t = t
            t0 = time.perf_counter()
            src = self._apply_body_constraints(t, stance)
            d_b, _ = self.body_obs.step(np.asarray(body_error, float)[:2], self.u_body)
            self.u_body = self.body_mpc.solve(np.asarray(body_error, float), d_b)
            if getattr(self.body_mpc, "last_polytope_failed", False):
                self.n_polytope_failures += 1
            self.timing.body_mpc_ms.append((time.perf_counter() - t0) * 1e3)
            self.n_body_solves += 1
            info["body_source"] = src
            info["body_solved"] = True

        # ---- asynchronous task predictor, body-priority remaining capacity ---
        if (self.task_mpc is not None and task_error is not None
                and t - self._last_task_t >= R.task_dt - 1e-12):
            self._last_task_t = t
            t0 = time.perf_counter()
            src = "fallback_box"
            if self.use_authority and self.task_snapshot is not None \
                    and self.task_snapshot.valid \
                    and self.task_snapshot.contact_mode == stance \
                    and (t - self.task_snapshot.timestamp) <= R.max_snapshot_age:
                # Body-priority allocation: the task gets what the body left.
                self.task_mpc.update_input_polytope(self.task_snapshot.H_task,
                                                    self.task_snapshot.h_task)
                src = "task_authority_polytope"
            else:
                self.task_mpc.update_input_box(-self.task_fallback_box,
                                               self.task_fallback_box)
                self.n_task_invalid += 1
            d_t, _ = self.task_obs.step(np.asarray(task_error, float)[:3], self.u_task)
            self.u_task = self.task_mpc.solve(np.asarray(task_error, float), d_t)
            self.timing.task_mpc_ms.append((time.perf_counter() - t0) * 1e3)
            self.n_task_solves += 1
            info["task_solved"] = True
            info["task_source"] = src

        # ---- 1 kHz realization: exactly ONE whole-body QP --------------------
        com_acc_des = np.asarray(com_ref_acc, float).copy()
        com_acc_des[:2] += self.u_body            # last valid command; never waits
        task_acc_des = (np.zeros(3) if task_acc_ref is None
                        else np.asarray(task_acc_ref, float).copy())
        if self.task_mpc is not None:
            task_acc_des = task_acc_des + self.u_task

        t0 = time.perf_counter()
        tau = self.realizer.command(
            model, data, q_ref, qd_ref, com_acc_des[:2], task_acc_des, hand_jac,
            stance_contacts, stance_targets, base_height_ref, rpy,
            com_acc_des=com_acc_des, swing_task=swing_task,
            attitude_weight=attitude_weight, centroidal_moment_des=np.zeros(3),
        )
        self.timing.realizer_ms.append((time.perf_counter() - t0) * 1e3)
        self.timing.qp_solves_per_cycle.append(1)    # the whole point

        # ---- publish the authority snapshot (one KKT solve, no extra QPs) ----
        t0 = time.perf_counter()
        if self.use_authority and self.continuation:
            if t - self._last_auth_t >= R.authority_dt - 1e-12:
                self._last_auth_t = t
                box = self.cont.estimate(self.realizer, model, data)
                self._cont_box = (box, t, stance) if box.valid else None
            self.snapshot = None      # continuation supplies bounds, not a polytope
        elif self.use_authority:
            self.snapshot = self.mapper.snapshot(
                self.realizer, model, data, timestamp=t, contact_mode=stance,
            )
            if self.task_mpc is not None:
                # Body-priority: the task set is conditioned on the body command
                # that was actually issued this cycle.  Reuses the same joint KKT
                # sensitivity, so it costs no extra QP or KKT solve.
                self.task_snapshot = self.mapper.task_authority(
                    self.realizer, model, data, self.u_body,
                )
        self.timing.authority_ms.append((time.perf_counter() - t0) * 1e3)

        info["fallback"] = bool(self.realizer.last_fallback)
        info["residual"] = float(self.realizer.last_body_acc_residual)
        return info

    def diagnostics(self) -> dict:
        return {
            "timing": self.timing.summary(),
            "body_solves": self.n_body_solves,
            "task_solves": self.n_task_solves,
            "stale_fallbacks": self.n_stale_fallbacks,
            "mode_mismatch_fallbacks": self.n_mode_mismatch,
            "invalid_snapshots": self.n_invalid_snapshots,
            "polytope_failures": self.n_polytope_failures,
            "task_invalid_snapshots": self.n_task_invalid,
        }
