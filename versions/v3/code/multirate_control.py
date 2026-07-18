"""Multirate interaction-dynamics controller.

The architecture, corrected. An earlier version put the whole-body QP inside a
1 kHz loop.  It does not fit: the QP alone is ~2 ms.  The rates are therefore

    1 kHz   JOINT SERVO (JointServo)
              - holds the latest optimized command (zero-order hold)
              - tau = tau_star + Kq (q_d - q) + Dq (qd_d - qd)
              - no optimization, no model update: this is the only loop that
                must actually close at 1 kHz, and it does

    200 Hz  OPTIMIZATION NODE (OptimizationNode), one real-time thread,
            executed SEQUENTIALLY so both predictors and the realizer see the
            same synchronized state:
              1. read (q, qdot, contacts)
              2. update M, h, J_c, J_t and the nominal feedforward
              3. run the disturbance observers
              4. solve the body predictor          (canonical, constant (A,B))
              5. solve the task predictor          (on the capacity the body left)
              6. solve ONE whole-body realization QP
              7. publish tau* and the joint reference to the servo
            Measured: roughly 2--3 ms median of a 5 ms nominal period in the
            Python prototype; this is not a hard-real-time guarantee.

    ~50 Hz  AUTHORITY REFRESH (rate-gated, NOT threaded in this prototype)
                            - the PWA continuation costs roughly 11--14 ms and does NOT fit
                                a synchronous 200 Hz critical path.
              - this implementation only rate-LIMITS that work: when due, it
                still runs synchronously inside the same node-update call and
                blocks it for ~16 ms.  A deployment needs a genuinely separate
                worker thread/process publishing the set lock-free; that is
                NOT what this prototype does.  The node otherwise uses the
                most recent set, and falls back to a conservative fixed box
                if that set is stale or was taken in a different contact mode.

The two predictors share one state read, one model update, and one set of
prediction matrices.  They remain functionally distinct -- the body port
predicts centroidal interaction, the task port hand interaction -- but they are
solved in sequence in the same node, not as two asynchronous consumers of stale
data.  The dependency is explicit: u_b(k) -> u_t(k) -> tau(k).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np

from normalized_mpc import NormalizedMPC, RandomWalkDisturbanceObserver
from realization_authority import (
    AnalyticAuthorityMapper, ContinuationAuthorityEstimator,
    RealizationAuthority, TaskAuthority,
)


@dataclass
class RateConfig:
    servo_dt: float = 0.001          # 1 kHz joint servo (ZOH + PD), no optimization
    node_dt: float = 0.005           # 200 Hz optimization node (predictors + WBC QP)
    authority_dt: float = 0.020      # ~50 Hz authority refresh (rate-gated, synchronous in this prototype)
    max_snapshot_age: float = 0.040  # older than this -> conservative fallback

    # backward-compatible aliases
    @property
    def sim_dt(self) -> float:
        return self.servo_dt

    @property
    def body_dt(self) -> float:
        return self.node_dt

    @property
    def task_dt(self) -> float:
        return self.node_dt


@dataclass
class Timing:
    realizer_ms: list[float] = field(default_factory=list)
    authority_ms: list[float] = field(default_factory=list)
    body_mpc_ms: list[float] = field(default_factory=list)
    task_mpc_ms: list[float] = field(default_factory=list)
    qp_solves_per_cycle: list[int] = field(default_factory=list)
    node_ms: list[float] = field(default_factory=list)

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
            "optimization_node": stat(self.node_ms),
            "whole_body_qp_solves_per_node_update": {
                "max": int(max(self.qp_solves_per_cycle)) if self.qp_solves_per_cycle else 0,
                "mean": float(np.mean(self.qp_solves_per_cycle)) if self.qp_solves_per_cycle else 0.0,
            },
        }


class JointServo:
    """1 kHz joint servo.  Zero-order-holds the last optimized command.

    This is the only loop that must actually close at 1 kHz, and it can: it
    performs no optimization and no model update.  Between optimization updates
    it holds tau* and tracks the joint reference the node published,

        tau = tau_star + Kq (q_d - q) + Dq (qd_d - qd),

    which is what keeps the joints stiff at 1 kHz while the optimizer runs at
    200 Hz.  Reporting this loop and the optimization node as one 1 kHz loop --
    as an earlier version of this controller did -- is simply wrong: the
    whole-body QP alone costs ~2 ms.
    """

    def __init__(self, realizer, *, kp: float = 0.0, kd: float = 0.0):
        self.realizer = realizer
        self.kp = float(kp)
        self.kd = float(kd)
        self.tau_star = None          # last optimized torque (ZOH)
        self.q_des = None
        self.qd_des = None
        self.n_holds = 0              # servo ticks served from a held command

    def publish(self, tau_star, q_des, qd_des) -> None:
        """Called by the optimization node at 200 Hz."""
        self.tau_star = np.asarray(tau_star, float).copy()
        self.q_des = None if q_des is None else np.asarray(q_des, float).copy()
        self.qd_des = None if qd_des is None else np.asarray(qd_des, float).copy()

    def step(self, model, data) -> np.ndarray | None:
        """Called every 1 ms.  No optimization here."""
        if self.tau_star is None:
            return None
        tau = self.tau_star
        if self.kp > 0.0 and self.q_des is not None:
            q, qd = self.realizer.joint_state(data)
            tau = tau + self.kp * (self.q_des - q) + self.kd * (self.qd_des - qd)
        # The impedance term can move the command outside the QP solution's
        # torque bounds.  Enforce the actuator envelope again at servo rate.
        # Hardware still needs independent drive limits and watchdogs; this is
        # only the controller-side bound guard.
        if not np.all(np.isfinite(tau)):
            tau = np.zeros_like(self.tau_star)
        tau = np.clip(tau, self.realizer.torque_min, self.realizer.torque_max)
        data.ctrl[self.realizer.ctrl_id] = tau
        self.n_holds += 1
        return tau


class MultirateInteractionController:
    """1 kHz realizer + rate-gated canonical predictors.

    The continuation authority refresh below is rate-limited to
    ``rates.authority_dt``, but when it runs it executes synchronously inside
    this call -- it is NOT offloaded to another thread in this prototype.
    """

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
        task_continuation: bool = False,
        task_realization_tolerance: float = 0.5,
    ):
        self.realizer = realizer
        self.rates = rates or RateConfig()
        self.mapper = mapper or AnalyticAuthorityMapper()
        self.use_authority = bool(use_authority)
        # PWA continuation: exact authority, KKT-only, but ~14 ms -- so it is
        # rate-limited to rates.authority_dt, not every 1 kHz cycle -- but the
        # call below still runs synchronously in this node update, not on a
        # separate thread; see the class docstring.
        self.continuation = bool(continuation)
        # Independent flag: the task port may use continuation while the body
        # port uses the analytic map, or vice versa.  Both share one estimator
        # instance (stateless between calls other than a solve counter).
        self.task_continuation = bool(task_continuation)
        self.task_realization_tolerance = float(task_realization_tolerance)
        self.cont = ContinuationAuthorityEstimator(max_regions=60)
        self._last_auth_t = -np.inf
        self._last_task_auth_t = -np.inf
        self._cont_box = None
        self._task_cont_box = None
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

        self.servo = JointServo(realizer)
        self._last_node_t = -np.inf
        self.n_node_updates = 0
        self.n_servo_ticks = 0
        self.n_deadline_miss = 0
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

        Never waits for a fresh QP-node snapshot.  Returns which source was used.
        """
        if self.use_authority and self.continuation:
            cb = self._cont_box
            if (cb is not None and cb[2] == mode
                    and (t - cb[1]) <= self.rates.max_snapshot_age):
                self.body_mpc.update_input_box(cb[0].lower + cb[3], cb[0].upper + cb[3])
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
        external_hand_force_ff: np.ndarray | None = None,
        task_command_selector: Callable[[np.ndarray, TaskAuthority | None], np.ndarray] | None = None,
    ) -> dict:
        """One servo tick (1 kHz).  The optimization node runs on its own 200 Hz
        sub-schedule; between node updates the servo simply holds the last
        optimized command, which is the whole point of the rate separation."""
        R = self.rates
        info = {"body_source": None, "body_solved": False, "task_solved": False,
            "node_ran": False, "task_selector_applied": False}

        # ---- not a node tick: 1 kHz servo holds the last optimized command ----
        if t - self._last_node_t < R.node_dt - 1e-12:
            self.servo.step(model, data)
            self.n_servo_ticks += 1
            info["residual"] = float(self.realizer.last_body_acc_residual)
            info["fallback"] = bool(self.realizer.last_fallback)
            return info

        self._last_node_t = t
        self.n_node_updates += 1
        info["node_ran"] = True
        t_node0 = time.perf_counter()

        # ---- body predictor: step 4 of the node, sequential, same state --------
        if True:
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

        # ---- task predictor: step 5, on the capacity the body just committed --
        if self.task_mpc is not None and task_error is not None:
            t0 = time.perf_counter()
            src = "fallback_box"
            snapshot_used: TaskAuthority | None = None
            if self.use_authority and self.task_continuation:
                tcb = self._task_cont_box
                if (tcb is not None and tcb[2] == stance
                        and (t - tcb[1]) <= R.max_snapshot_age):
                    self.task_mpc.update_input_box(tcb[0].lower + tcb[3],
                                                   tcb[0].upper + tcb[3])
                    src = "task_continuation_box"
                else:
                    self.task_mpc.update_input_box(-self.task_fallback_box,
                                                   self.task_fallback_box)
                    self.n_task_invalid += 1
            elif self.use_authority and self.task_snapshot is not None \
                    and self.task_snapshot.valid \
                    and self.task_snapshot.contact_mode == stance \
                    and (t - self.task_snapshot.timestamp) <= R.max_snapshot_age:
                # Body-priority allocation: the task gets what the body left.
                self.task_mpc.update_input_polytope(self.task_snapshot.H_task,
                                                    self.task_snapshot.h_task)
                src = "task_authority_polytope"
                snapshot_used = self.task_snapshot
            else:
                self.task_mpc.update_input_box(-self.task_fallback_box,
                                               self.task_fallback_box)
                self.n_task_invalid += 1
            d_t, _ = self.task_obs.step(np.asarray(task_error, float)[:3], self.u_task)
            self.u_task = self.task_mpc.solve(np.asarray(task_error, float), d_t)
            info["task_nominal_command"] = self.u_task.copy()
            info["task_snapshot_used"] = snapshot_used
            if task_command_selector is not None:
                selected = np.asarray(
                    task_command_selector(self.u_task.copy(), snapshot_used), dtype=float
                ).reshape(3)
                if not np.all(np.isfinite(selected)):
                    raise ValueError("task_command_selector must return a finite shape-(3,) command")
                self.u_task = selected
                info["task_selector_applied"] = True
            info["task_command"] = self.u_task.copy()
            self.timing.task_mpc_ms.append((time.perf_counter() - t0) * 1e3)
            self.n_task_solves += 1
            info["task_solved"] = True
            info["task_source"] = src

        # ---- one active-mode whole-body realization QP at this node ----------
        com_acc_des = np.asarray(com_ref_acc, float).copy()
        com_acc_des[:2] += self.u_body            # last valid command; never waits
        task_acc_des = (np.zeros(3) if task_acc_ref is None
                        else np.asarray(task_acc_ref, float).copy())
        if self.task_mpc is not None:
            task_acc_des = task_acc_des + self.u_task

        t0 = time.perf_counter()
        tau, _tau_unsat, _sat = self.realizer.command(
            model, data, q_ref, qd_ref, com_acc_des[:2], task_acc_des, hand_jac,
            stance_contacts, stance_targets, base_height_ref, rpy,
            com_acc_des=com_acc_des, swing_task=swing_task,
            attitude_weight=attitude_weight, centroidal_moment_des=np.zeros(3),
            external_hand_force_ff=external_hand_force_ff,
        )
        self.timing.realizer_ms.append((time.perf_counter() - t0) * 1e3)
        self.timing.qp_solves_per_cycle.append(1)    # the whole point

        # ---- publish the authority snapshot (one KKT solve, no extra QPs) ----
        t0 = time.perf_counter()
        if self.use_authority and self.continuation:
            if t - self._last_auth_t >= R.authority_dt - 1e-12:
                self._last_auth_t = t
                box = self.cont.estimate(self.realizer, model, data)
                self._cont_box = (
                    box, t, stance, self.u_body.copy()
                ) if box.valid else None
            self.snapshot = None      # continuation supplies bounds, not a polytope
        elif self.use_authority:
            self.snapshot = self.mapper.snapshot(
                self.realizer, model, data, timestamp=t, contact_mode=stance,
                command_reference=self.u_body,
            )
            if self.task_mpc is not None and not self.task_continuation:
                # Body-priority: the task set is conditioned on the body command
                # that was actually issued this cycle.  Reuses the same joint KKT
                # sensitivity, so it costs no extra QP or KKT solve.
                self.task_snapshot = self.mapper.task_authority(
                    self.realizer, model, data, self.u_body,
                    body_reference=self.u_body, task_reference=self.u_task,
                )
        if self.use_authority and self.task_continuation and self.task_mpc is not None:
            if t - self._last_task_auth_t >= R.authority_dt - 1e-12:
                self._last_task_auth_t = t
                tbox = self.cont.estimate_task(
                    self.realizer, model, data,
                    task_tolerance=self.task_realization_tolerance,
                )
                self._task_cont_box = (
                    tbox, t, stance, self.u_task.copy()
                ) if tbox.valid else None
        self.timing.authority_ms.append((time.perf_counter() - t0) * 1e3)

        # ---- step 7: publish the optimized command to the 1 kHz servo ---------
        self.servo.publish(tau, None, None)
        self.n_servo_ticks += 1

        node_ms = (time.perf_counter() - t_node0) * 1e3
        self.timing.node_ms.append(node_ms)
        if node_ms > 1000.0 * R.node_dt:
            self.n_deadline_miss += 1

        info["fallback"] = bool(self.realizer.last_fallback)
        info["residual"] = float(self.realizer.last_body_acc_residual)
        info["node_ms"] = node_ms
        return info

    def diagnostics(self) -> dict:
        return {
            "timing": self.timing.summary(),
            "node_updates": self.n_node_updates,
            "servo_ticks": self.n_servo_ticks,
            "servo_holds": self.servo.n_holds,
            "node_deadline_misses": self.n_deadline_miss,
            "body_solves": self.n_body_solves,
            "task_solves": self.n_task_solves,
            "stale_fallbacks": self.n_stale_fallbacks,
            "mode_mismatch_fallbacks": self.n_mode_mismatch,
            "invalid_snapshots": self.n_invalid_snapshots,
            "polytope_failures": self.n_polytope_failures,
            "task_invalid_snapshots": self.n_task_invalid,
        }
