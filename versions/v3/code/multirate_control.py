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

import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable

import mujoco
import numpy as np

from normalized_mpc import NormalizedMPC, RandomWalkDisturbanceObserver
from realization_authority import (
    AnalyticAuthorityMapper, BoxAuthority, ContinuationAuthorityEstimator,
    PhysicalFailureManager, PhysicalFailureResponse, PhysicalRealizabilityPredictor,
    RealizationAuthority, ResponseLevel, RouteCandidate, RouteEvaluator, TaskAuthority,
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
    # Populated only when a RealizerWorker is in use.  worker_compute_ms is
    # the worker THREAD's own wall time per job (compute_finished -
    # compute_started) -- this is the ~3.5-11 ms cost that used to block
    # step() and, under the async design, no longer does.  node_ms above
    # keeps its existing meaning (this cycle's step() wall time) and should
    # now be small, since the expensive part moved off that path.
    # snapshot_staleness_ms is how old the published result was, in ms, at
    # the moment step() actually read it -- the number that answers "is the
    # staleness this design tolerates actually small in practice."
    worker_compute_ms: list[float] = field(default_factory=list)
    snapshot_staleness_ms: list[float] = field(default_factory=list)

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
            "worker_compute": stat(self.worker_compute_ms),
            "snapshot_staleness": stat(self.snapshot_staleness_ms),
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


@dataclass
class RealizerJob:
    """Everything RealizerWorker needs for one node cycle, captured at
    submit time so the worker can run entirely against its own MjData copy
    without racing the caller's mj_step. Only used when the controller has
    committed to the analytic (non-continuation) authority path for BOTH
    ports -- see MultirateInteractionController._use_worker.
    """

    t: float
    stance: tuple[str, ...]
    q_ref: np.ndarray
    qd_ref: np.ndarray
    com_acc_des: np.ndarray          # full (3,); body-port correction already added in
    task_acc_des: np.ndarray
    hand_jac: np.ndarray
    stance_contacts: dict
    stance_targets: dict
    base_height_ref: float
    rpy: np.ndarray
    swing_task: dict | None
    attitude_weight: float
    external_hand_force_ff: np.ndarray | None
    u_body: np.ndarray
    u_task: np.ndarray


@dataclass
class RealizerResult:
    t: float                    # timestamp of the STATE this result was computed from
    contact_mode: tuple[str, ...]
    tau: np.ndarray
    fallback: bool
    residual: float
    snapshot: RealizationAuthority | None
    task_snapshot: TaskAuthority | None
    compute_started: float      # perf_counter() the worker picked the job up
    compute_finished: float     # perf_counter() the worker published this
    realizer_ms: float
    authority_ms: float
    # Copies of realizer.last_hand_jac/last_qdd/last_task_acc_des at solve
    # time -- callers that want the task-tracking residual
    # (hand_jac @ qdd - task_acc_des) must read it from here, not from
    # self.realizer directly, for the same reason as everything else on
    # this dataclass: those attributes are worker-owned while this thread
    # runs.
    hand_jac: np.ndarray | None
    qdd: np.ndarray
    task_acc_des: np.ndarray


class RealizerWorker:
    """Runs realizer.command() + mapper.snapshot()/task_authority() on a
    background thread, decoupled from the caller's live ``data``.

    This is the "genuinely separate worker thread/process publishing the
    set lock-free" the class docstring above already names as the correct
    fix and says this prototype doesn't build. It targets the dominant real
    cost identified by profiling (the whole-body QP + KKT sensitivity, not
    body_mpc/task_mpc or anything in the Module A/B/C wiring).

    IMPORTANT -- opt-in only (MultirateInteractionController's
    enable_async_worker, default False): this design trades a fixed ~1-cycle
    staleness for a VARIABLE one bounded by rates.max_snapshot_age, with a
    safe (gravity-compensation-only) torque published whenever the bound is
    exceeded. That bound only makes the failure mode safe -- it does not
    guarantee the worker keeps up. Measured directly: fine for a benign
    standing scenario (worker comfortably keeps pace, 0 deadline misses on
    the now-decoupled critical path); NOT fine for a demanding, fast-changing
    one (an aggressive high-gain predictor drove the worker into the
    safe-fallback branch 60-75% of node updates, i.e. no active balance
    control most of the time, and the robot fell despite the fallback torque
    being individually safe). The deficit there is throughput -- the
    worker's own compute time is comparable to or exceeds the node period --
    not staleness, so it is not fixable by a smarter fallback alone. Enable
    this only for a scenario where you have actually measured that the
    worker's compute time fits comfortably inside rates.node_dt.

    Scope note: this worker is used only when NEITHER port uses the PWA
    continuation authority (see _use_worker). Continuation's own estimators
    (ContinuationAuthorityEstimator.estimate/estimate_task) read the SAME
    self.realizer post-solve state that command() mutates; running one of
    them synchronously on the main thread while this worker's command() call
    mutates that state on a different thread would be exactly the kind of
    race this design exists to avoid. Rather than also relocate the
    continuation estimators into this worker (a materially larger change,
    with its own separate rate-limiting/staleness bookkeeping), the two
    paths are kept mutually exclusive per controller instance; continuation
    keeps its existing fully-synchronous behavior unchanged.

    Owns a PRIVATE MjData copy (mjData is not safe to read/write
    concurrently from two threads) that is resynced from the caller's live
    data at the start of each accepted job -- just the raw state
    (qpos/qvel/ctrl), not a full deep clone -- then brought current with
    mj_forward on that private copy before the realizer touches it.

    Publishing is lock-free by construction: CPython guarantees a single
    attribute assignment is atomic, so the worker thread simply does
    ``self._latest = new_result`` when done, and the reader always reads
    ``self._latest`` -- no lock needed for that swap, only for the (much
    cheaper) data-sync step below.
    """

    def __init__(self, model: mujoco.MjModel, realizer, mapper, *,
                 task_mpc_present: bool):
        self._model = model
        self._data = mujoco.MjData(model)
        self._realizer = realizer
        self._mapper = mapper
        self._task_mpc_present = bool(task_mpc_present)
        self._sync_lock = threading.Lock()
        self._wakeup = threading.Event()
        self._pending: RealizerJob | None = None
        self._busy = False
        self._latest: RealizerResult | None = None
        self._stop = False
        # If a job raises (a bad input, a genuine realizer bug), _busy must
        # still be cleared or every future submit() would silently starve
        # forever -- the caller would just see "no new result" and keep
        # holding the last-good command indefinitely with no visibility into
        # why. last_error/n_errors make that visible instead of silent.
        self.last_error: BaseException | None = None
        self.n_errors = 0
        self._thread = threading.Thread(
            target=self._run, name="RealizerWorker", daemon=True)
        self._thread.start()

    def submit(self, job: RealizerJob, live_data: mujoco.MjData) -> bool:
        """Non-blocking. Drops the job (mailbox semantics) if the worker is
        still busy with the previous one -- only the freshest state is ever
        useful, so a backlog of stale jobs would be actively harmful, not
        just wasted work. Returns whether the job was accepted.
        """
        if self._busy:
            return False
        with self._sync_lock:
            self._data.qpos[:] = live_data.qpos
            self._data.qvel[:] = live_data.qvel
            self._data.ctrl[:] = live_data.ctrl
        self._pending = job
        self._busy = True
        self._wakeup.set()
        return True

    def latest(self) -> RealizerResult | None:
        return self._latest

    def stop(self) -> None:
        self._stop = True
        self._wakeup.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            self._wakeup.wait()
            self._wakeup.clear()
            if self._stop:
                return
            job = self._pending
            try:
                self._process(job)
            except BaseException as exc:  # noqa: BLE001 -- see class docstring
                # Deliberately broad: this thread has no other way to
                # surface a failure, and swallowing it silently while
                # leaving _busy=True would starve every future submit()
                # forever. self._latest is simply left at its last GOOD
                # value, so the controller keeps holding the last-known-safe
                # command (the same behavior as any other "no new result
                # yet" cycle) rather than crashing or hanging.
                self.last_error = exc
                self.n_errors += 1
            finally:
                self._busy = False

    def _process(self, job: "RealizerJob") -> None:
        compute_started = time.perf_counter()
        mujoco.mj_forward(self._model, self._data)

        t0 = time.perf_counter()
        tau, _tau_unsat, _sat = self._realizer.command(
            self._model, self._data, job.q_ref, job.qd_ref,
            job.com_acc_des[:2], job.task_acc_des, job.hand_jac,
            job.stance_contacts, job.stance_targets, job.base_height_ref,
            job.rpy, com_acc_des=job.com_acc_des, swing_task=job.swing_task,
            attitude_weight=job.attitude_weight,
            centroidal_moment_des=np.zeros(3),
            external_hand_force_ff=job.external_hand_force_ff,
        )
        realizer_ms = (time.perf_counter() - t0) * 1e3
        fallback = bool(self._realizer.last_fallback)
        residual = float(self._realizer.last_body_acc_residual)
        hand_jac = (None if self._realizer.last_hand_jac is None
                    else self._realizer.last_hand_jac.copy())
        qdd = self._realizer.last_qdd.copy()
        task_acc_des = self._realizer.last_task_acc_des.copy()

        t0 = time.perf_counter()
        snapshot = self._mapper.snapshot(
            self._realizer, self._model, self._data, timestamp=job.t,
            contact_mode=job.stance, command_reference=job.u_body,
        )
        task_snapshot = None
        if self._task_mpc_present:
            task_snapshot = self._mapper.task_authority(
                self._realizer, self._model, self._data, job.u_body,
                body_reference=job.u_body, task_reference=job.u_task,
            )
        authority_ms = (time.perf_counter() - t0) * 1e3

        self._latest = RealizerResult(
            t=job.t, contact_mode=job.stance, tau=tau, fallback=fallback,
            residual=residual, snapshot=snapshot, task_snapshot=task_snapshot,
            compute_started=compute_started, compute_finished=time.perf_counter(),
            realizer_ms=realizer_ms, authority_ms=authority_ms,
            hand_jac=hand_jac, qdd=qdd, task_acc_des=task_acc_des,
        )


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
        body_margin_safe: float = 0.15,
        task_margin_safe: float = 0.15,
        enable_async_worker: bool = False,
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

        # A SEPARATE instance for the fallback-box candidate, not a second
        # constraint mode toggled on self.body_mpc/self.task_mpc: switching a
        # single NormalizedMPC between update_input_polytope and
        # update_input_box resets _H_poly, which forces the (~30 ms) OSQP
        # solver rebuild on the NEXT polytope call instead of the fast
        # incremental .update() path -- see update_input_polytope's own
        # rebuild comment. Two independent instances keep each candidate's
        # solver state (and rebuild-avoidance) undisturbed by the other.
        self.body_fallback_mpc = replace(self.body_mpc)
        self.task_fallback_mpc = None if self.task_mpc is None else replace(self.task_mpc)

        # Published state (the "lock-free buffers" of the real implementation).
        self.snapshot: RealizationAuthority | None = None
        self.task_snapshot: TaskAuthority | None = None
        self.task_fallback_box = np.array([2.0, 2.0, 2.0])
        self.u_body = np.zeros(2)
        self.u_task = np.zeros(3)

        # Modules A/B/C from the code-to-paper review: A (predictor) folds an
        # already-predicted command horizon through an already-published
        # authority polytope; B (classifier) and C (route evaluator) are
        # wired below into the analytic-snapshot body/task branches only --
        # the continuation branches keep their own, unchanged, box-only
        # fallback (see _select_body_route/_select_task_route).
        self.predictor = PhysicalRealizabilityPredictor()
        self.body_margin_safe = float(body_margin_safe)
        self.task_margin_safe = float(task_margin_safe)
        self.body_response_manager = PhysicalFailureManager(margin_safe=self.body_margin_safe)
        self.task_response_manager = PhysicalFailureManager(margin_safe=self.task_margin_safe)
        self.body_route_evaluator = RouteEvaluator(
            margin_safe=self.body_margin_safe, predictor=self.predictor)
        self.task_route_evaluator = RouteEvaluator(
            margin_safe=self.task_margin_safe, predictor=self.predictor)
        self.body_response: PhysicalFailureResponse | None = None
        self.task_response: PhysicalFailureResponse | None = None

        # RealizerWorker offloads realizer.command() + mapper.snapshot()/
        # task_authority() (the dominant real-time cost per E1 profiling) to
        # a background thread. OPT-IN, default off: validated safe only for
        # scenarios where the worker's own compute time comfortably fits the
        # node period (E1's benign standing case: 0 deadline misses, worker
        # keeps up). Measured UNSAFE for demanding/fast-changing scenarios
        # (E2's aggressive high-gain predictor): even with the staleness-
        # bound safe-torque fallback below, the worker cannot keep pace with
        # E2's submission rate (busy on 60-75% of node updates in testing),
        # meaning there is effectively no active balance control for most of
        # a short aggressive maneuver -- the robot falls regardless of how
        # safe the fallback torque is, because the deficit is throughput, not
        # staleness. Do not enable this for a scenario until you've measured
        # (not assumed) that the worker keeps up, the same way E1 was here.
        #
        # Also requires NEITHER port to be on the PWA continuation path:
        # ContinuationAuthorityEstimator.estimate/estimate_task read the same
        # self.realizer post-solve state that command() mutates, and running
        # one synchronously on the main thread while the worker's command()
        # call runs on another thread would race -- see RealizerWorker's
        # docstring. Constructed lazily on the first step() call, once
        # `model` is available.
        self._use_worker = (
            bool(enable_async_worker)
            and not self.continuation and not self.task_continuation
        )
        self.worker: RealizerWorker | None = None
        self._last_published_result_t: float | None = None
        self.n_stale_command_holds = 0   # cycles where the worker's latest result
                                          # was unusable (wrong contact mode or too
                                          # old) and the safe gravity-compensation
                                          # fallback was published instead

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
        # Cycles where the authority snapshot passed its validity/mode/
        # freshness checks, but RouteEvaluator still chose the fallback box
        # because the authority candidate's own predicted margin (Module A)
        # did not clear margin_safe -- a strictly stronger check than the
        # metadata gate above ever performed on its own.
        self.n_body_route_downgrade = 0
        self.n_task_route_downgrade = 0
        self._last_body_t = -np.inf
        self._last_task_t = -np.inf

    # -- predictor constraint selection (Modules B/C wired in here) ---------
    def _select_body_route(
        self, t: float, mode: tuple[str, ...],
        body_error: np.ndarray, d_hat: np.ndarray,
    ) -> tuple[np.ndarray, str, PhysicalFailureResponse | None]:
        """Solve the body predictor via the authority-polytope route or the
        conservative fallback-box route, choosing between them with Module
        C's ``RouteEvaluator`` instead of trusting the snapshot's validity/
        mode/freshness metadata alone.

        The metadata checks are still necessary preconditions -- an invalid,
        wrong-mode, or stale snapshot cannot be evaluated at all -- but when
        they pass, the box is now solved and certified too (as a
        ``BoxAuthority``), so RouteEvaluator's ``M(P) >= margin_safe`` check
        can catch a snapshot that LOOKS fine by metadata but whose own
        predicted margin does not clear ``margin_safe`` (Module A), not just
        one that is outright stale/invalid/wrong-mode.  Cost: one extra
        NormalizedMPC solve on cycles where the authority route is attempted
        (the box solve is cheap relative to the polytope solve; see the
        commit message for measured overhead).

        Never waits for a fresh QP-node snapshot.  The continuation-authority
        path is unchanged: it publishes bounds, not a polytope, so there is
        nothing for Module A's H/h machinery to evaluate, and it keeps its
        own existing conservative-box fallback.
        """
        if self.use_authority and self.continuation:
            cb = self._cont_box
            if (cb is not None and cb[2] == mode
                    and (t - cb[1]) <= self.rates.max_snapshot_age):
                self.body_mpc.update_input_box(cb[0].lower + cb[3], cb[0].upper + cb[3])
                return self.body_mpc.solve(body_error, d_hat), "continuation_box", None
            self.n_invalid_snapshots += 1
            self.body_mpc.update_input_box(-self.fallback_box, self.fallback_box)
            return self.body_mpc.solve(body_error, d_hat), "fallback_box", None

        s = self.snapshot
        authority_ok = (
            self.use_authority and s is not None and s.valid
            and s.contact_mode == mode
            and (t - s.timestamp) <= self.rates.max_snapshot_age
        )
        u_by_name: dict[str, np.ndarray] = {}
        candidates: list[RouteCandidate] = []
        if authority_ok:
            self.body_mpc.update_input_polytope(s.H_body, s.h_body, u_ref=s.command_reference)
            u_a = self.body_mpc.solve(body_error, d_hat)
            polytope_failed = self.body_mpc.last_polytope_failed
            if polytope_failed:
                self.n_polytope_failures += 1
            seq_a = (self.body_mpc.last_u_sequence if self.body_mpc.last_u_sequence is not None
                     else np.tile(u_a, (self.body_mpc.horizon, 1)))
            u_by_name["authority_polytope"] = u_a
            candidates.append(RouteCandidate(
                "authority_polytope", s, seq_a, cost=0.0,
                last_polytope_failed=polytope_failed,
            ))
        else:
            if not self.use_authority or s is None or not s.valid:
                self.n_invalid_snapshots += int(self.use_authority and (s is None or not s.valid))
            elif s.contact_mode != mode:
                self.n_mode_mismatch += 1
            else:
                self.n_stale_fallbacks += 1

        box = BoxAuthority.from_box(-self.fallback_box, self.fallback_box,
                                    timestamp=t, contact_mode=mode)
        self.body_fallback_mpc.update_input_box(-self.fallback_box, self.fallback_box)
        u_b = self.body_fallback_mpc.solve(body_error, d_hat)
        seq_b = (self.body_fallback_mpc.last_u_sequence
                 if self.body_fallback_mpc.last_u_sequence is not None
                 else np.tile(u_b, (self.body_fallback_mpc.horizon, 1)))
        u_by_name["fallback_box"] = u_b
        candidates.append(RouteCandidate("fallback_box", box, seq_b, cost=1.0))

        sel = self.body_route_evaluator.evaluate(candidates)
        response = None
        if authority_ok:
            auth_cert = next(
                e.certificate for e in sel.evaluations if e.name == "authority_polytope")
            # NOT self.body_mpc.last_polytope_failed: the box solve above ran
            # after the authority solve and overwrote that instance state.
            response = self.body_response_manager.classify(
                auth_cert, last_polytope_failed=polytope_failed)

        chosen = sel.selected or "fallback_box"     # box is always a safe default
        if authority_ok and chosen != "authority_polytope":
            self.n_body_route_downgrade += 1
        return u_by_name[chosen], chosen, response

    def _select_task_route(
        self, t: float, mode: tuple[str, ...],
        task_error: np.ndarray, d_hat: np.ndarray,
    ) -> tuple[np.ndarray, str, TaskAuthority | None, PhysicalFailureResponse | None]:
        """Task-port mirror of :meth:`_select_body_route`.

        Returns ``(u_task, source, snapshot_used, response)``; ``snapshot_used``
        is the ``TaskAuthority`` RouteEvaluator actually selected (or ``None``
        for the fallback box), preserved for ``task_command_selector`` callers
        that key off it.
        """
        if self.use_authority and self.task_continuation:
            tcb = self._task_cont_box
            if (tcb is not None and tcb[2] == mode
                    and (t - tcb[1]) <= self.rates.max_snapshot_age):
                self.task_mpc.update_input_box(tcb[0].lower + tcb[3], tcb[0].upper + tcb[3])
                return (self.task_mpc.solve(task_error, d_hat), "task_continuation_box",
                        None, None)
            self.n_task_invalid += 1
            self.task_mpc.update_input_box(-self.task_fallback_box, self.task_fallback_box)
            return self.task_mpc.solve(task_error, d_hat), "fallback_box", None, None

        ts = self.task_snapshot
        authority_ok = (
            self.use_authority and ts is not None and ts.valid
            and ts.contact_mode == mode
            and (t - ts.timestamp) <= self.rates.max_snapshot_age
        )
        u_by_name: dict[str, np.ndarray] = {}
        candidates: list[RouteCandidate] = []
        if authority_ok:
            self.task_mpc.update_input_polytope(ts.H_task, ts.h_task, u_ref=ts.command_reference)
            u_a = self.task_mpc.solve(task_error, d_hat)
            polytope_failed = self.task_mpc.last_polytope_failed
            seq_a = (self.task_mpc.last_u_sequence if self.task_mpc.last_u_sequence is not None
                     else np.tile(u_a, (self.task_mpc.horizon, 1)))
            u_by_name["task_authority_polytope"] = u_a
            candidates.append(RouteCandidate(
                "task_authority_polytope", ts, seq_a, cost=0.0,
                last_polytope_failed=polytope_failed,
            ))
        else:
            self.n_task_invalid += 1

        box = BoxAuthority.from_box(-self.task_fallback_box, self.task_fallback_box,
                                    timestamp=t, contact_mode=mode)
        self.task_fallback_mpc.update_input_box(-self.task_fallback_box, self.task_fallback_box)
        u_b = self.task_fallback_mpc.solve(task_error, d_hat)
        seq_b = (self.task_fallback_mpc.last_u_sequence
                 if self.task_fallback_mpc.last_u_sequence is not None
                 else np.tile(u_b, (self.task_fallback_mpc.horizon, 1)))
        u_by_name["fallback_box"] = u_b
        candidates.append(RouteCandidate("fallback_box", box, seq_b, cost=1.0))

        sel = self.task_route_evaluator.evaluate(candidates)
        response = None
        if authority_ok:
            auth_cert = next(
                e.certificate for e in sel.evaluations if e.name == "task_authority_polytope")
            response = self.task_response_manager.classify(
                auth_cert, last_polytope_failed=polytope_failed)

        chosen = sel.selected or "fallback_box"
        if authority_ok and chosen != "task_authority_polytope":
            self.n_task_route_downgrade += 1
        snapshot_used = ts if chosen == "task_authority_polytope" else None
        return u_by_name[chosen], chosen, snapshot_used, response

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
            # When the worker is in use, self.realizer's mutable attributes
            # (last_fallback/last_body_acc_residual) can be mid-mutation on
            # the worker thread at ANY time, not just during a node tick --
            # so every read, including here, must come from the last
            # published RealizerResult, never self.realizer directly.
            if self._use_worker:
                result = self.worker.latest() if self.worker is not None else None
                info["residual"] = float(result.residual) if result is not None else float("inf")
                info["fallback"] = result.fallback if result is not None else True
            else:
                info["residual"] = float(self.realizer.last_body_acc_residual)
                info["fallback"] = bool(self.realizer.last_fallback)
            return info

        self._last_node_t = t
        self.n_node_updates += 1
        info["node_ran"] = True
        t_node0 = time.perf_counter()

        if self._use_worker and self.worker is None:
            self.worker = RealizerWorker(
                model, self.realizer, self.mapper,
                task_mpc_present=self.task_mpc is not None)

        if self._use_worker:
            # Adopt whatever the worker has published so far -- from
            # whenever it finished, not necessarily this cycle -- BEFORE
            # route selection runs, exactly mirroring the ordering the
            # synchronous path already had (this cycle's route selection
            # uses the PREVIOUS cycle's published snapshot; here it uses
            # the most recently published one, whatever its actual age).
            # _select_body_route/_select_task_route's existing
            # contact_mode/max_snapshot_age gate is untouched and does the
            # rest -- it already tolerates "not from this exact cycle."
            result = self.worker.latest()
            if result is not None:
                self.snapshot = result.snapshot
                self.task_snapshot = result.task_snapshot

        # ---- body predictor: step 4 of the node, sequential, same state --------
        if True:
            t0 = time.perf_counter()
            # The observer update needs the PREVIOUS cycle's u_body, so it
            # must run before _select_body_route overwrites self.u_body.
            d_b, _ = self.body_obs.step(np.asarray(body_error, float)[:2], self.u_body)
            self.u_body, src, self.body_response = self._select_body_route(
                t, stance, np.asarray(body_error, float), d_b)
            self.timing.body_mpc_ms.append((time.perf_counter() - t0) * 1e3)
            self.n_body_solves += 1
            info["body_source"] = src
            info["body_response_level"] = (
                None if self.body_response is None else self.body_response.level.name)
            info["body_solved"] = True

        # ---- task predictor: step 5, on the capacity the body just committed --
        if self.task_mpc is not None and task_error is not None:
            t0 = time.perf_counter()
            d_t, _ = self.task_obs.step(np.asarray(task_error, float)[:3], self.u_task)
            self.u_task, src, snapshot_used, self.task_response = self._select_task_route(
                t, stance, np.asarray(task_error, float), d_t)
            info["task_nominal_command"] = self.u_task.copy()
            info["task_snapshot_used"] = snapshot_used
            info["task_response_level"] = (
                None if self.task_response is None else self.task_response.level.name)
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

        if self._use_worker:
            # ---- hand this cycle's state off to the background worker -------
            # (see RealizerWorker: command() + mapper.snapshot()/task_authority()
            # run there, against the worker's OWN MjData copy, not this `data`.)
            job = RealizerJob(
                t=t, stance=stance, q_ref=np.asarray(q_ref, float),
                qd_ref=np.asarray(qd_ref, float), com_acc_des=com_acc_des,
                task_acc_des=task_acc_des, hand_jac=hand_jac,
                stance_contacts=stance_contacts, stance_targets=stance_targets,
                base_height_ref=base_height_ref, rpy=rpy, swing_task=swing_task,
                attitude_weight=attitude_weight,
                external_hand_force_ff=external_hand_force_ff,
                u_body=self.u_body.copy(), u_task=self.u_task.copy(),
            )
            self.worker.submit(job, data)

            # ---- publish the worker's latest result, but only if it is
            # SAFE to keep using: still for the CURRENT contact mode (a
            # result computed for a mode the robot has since left, e.g.
            # mid-DS->SS-transition, is wrong by construction) AND no older
            # than max_snapshot_age. The age check matters even when nothing
            # NEW has arrived: if the worker falls behind for several node
            # cycles in a row (measured under an aggressive, high-gain
            # predictor -- E2 -- where the worker cannot keep up with the
            # 200 Hz submission rate), blindly continuing to ZOH an
            # increasingly stale QP-derived torque is not just conservative,
            # it actively destabilized balance. Once a result is too old,
            # fall back to the SAME cheap, QP-free command the realizer
            # itself already uses when its own solve fails --
            # gravity/Coriolis compensation clipped to the torque envelope
            # (data.qfrc_bias, already populated by this cycle's mj_forward,
            # no extra computation) -- and keep recomputing that fallback
            # fresh every cycle for as long as the worker stays behind,
            # rather than freezing it at whatever it was when staleness was
            # first detected.
            result = self.worker.latest()
            is_new = result is not None and result.t != self._last_published_result_t
            age = (t - result.t) if result is not None else float("inf")
            usable = (
                result is not None and result.contact_mode == stance
                and age <= R.max_snapshot_age
            )
            if usable:
                if is_new:
                    self._last_published_result_t = result.t
                    self.servo.publish(result.tau, None, None)
                    self.timing.realizer_ms.append(result.realizer_ms)
                    self.timing.authority_ms.append(result.authority_ms)
                    self.timing.worker_compute_ms.append(
                        (result.compute_finished - result.compute_started) * 1e3)
                    self.timing.snapshot_staleness_ms.append(age * 1e3)
                    self.timing.qp_solves_per_cycle.append(1)    # the whole point
                # else: already published and still within its freshness
                # bound -- the servo's own ZOH already holds the right command.
                info["fallback"] = result.fallback
                info["residual"] = result.residual
            else:
                tau_fallback = np.clip(
                    data.qfrc_bias[self.realizer.dof],
                    self.realizer.torque_min, self.realizer.torque_max,
                )
                self.servo.publish(tau_fallback, None, None)
                self.n_stale_command_holds += 1
                info["fallback"] = True
                info["residual"] = result.residual if result is not None else float("inf")
        else:
            # ---- unchanged, fully-synchronous path (continuation ports) -----
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

            self.servo.publish(tau, None, None)
            info["fallback"] = bool(self.realizer.last_fallback)
            info["residual"] = float(self.realizer.last_body_acc_residual)

        self.n_servo_ticks += 1
        node_ms = (time.perf_counter() - t_node0) * 1e3
        self.timing.node_ms.append(node_ms)
        if node_ms > 1000.0 * R.node_dt:
            self.n_deadline_miss += 1
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
            "body_route_downgrades": self.n_body_route_downgrade,
            "task_route_downgrades": self.n_task_route_downgrade,
            "use_worker": self._use_worker,
            "stale_command_holds": self.n_stale_command_holds,
            "worker_errors": self.worker.n_errors if self.worker is not None else 0,
        }

    def stop(self) -> None:
        """Join the background worker thread, if one was started. Every
        caller that constructs a controller with continuation disabled on
        both ports (the default) should call this when done with it -- the
        thread is a daemon so it won't block process exit on its own, but a
        script that constructs many controllers in a loop (as the benchmark
        scripts here do) would otherwise leak one live thread per run."""
        if self.worker is not None:
            self.worker.stop()
