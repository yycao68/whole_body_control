"""Realization-informed command authority for the canonical predictors.

Two estimators live here, with very different roles.

``AnalyticAuthorityMapper`` (the controller path).  The 1 kHz realization loop
already solves one whole-body QP.  The residual command ``u`` enters that QP
only through objective *linear* terms, and the QP Hessian does not depend on
``u``; therefore, on the current active-set cell, the solution is affine,

    z(u) = z0 + K u,      K = dz/du   (one KKT solve, no extra QP solves)

so the nominal feedforward and the input maps

    tau    = tau_ff    + K_tau    u
    lambda = lambda_ff + K_lambda u

come out of the cycle the realizer is running anyway.  Substituting them into
the *physical* limits the realizer enforces -- actuator bounds, friction
pyramid, unilateral normal force -- yields linear constraints on the residual
command,

    H_k u <= h_k,

which is exactly the set the canonical predictor should be constrained by.  The
canonical pair (A, B) is untouched: contact mode and configuration change only
(H_k, h_k).  Cost is one KKT solve (~0.15 ms), not 62 QP solves (~154 ms).

The affine map is exact only on the *current active-set cell*.  Once a
constraint activates, the true QP redistributes and the map bends, so H_k u <=
h_k is a local model and NOT a certificate.  The instantaneous realizer remains
the final hard safety layer, and the mapping's optimism/conservatism is
quantified offline against the exact estimator below.

``ExactResidualBisectionEstimator`` (offline ground truth).  Repeatedly re-solves
the hard-constrained realizer and bisects the signed coordinate rays on the
realized-vs-requested acceleration residual.  Accurate but ~62 QP solves and
O(100 ms); it is a *measurement procedure*, not part of the feedback loop.  Its
job is to grade the analytic mapping (false-positive / false-negative rates).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import mujoco
import numpy as np


# --------------------------------------------------------------------------
# Controller path: analytic mapping from the 1 kHz realizer
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RealizationAuthority:
    """Snapshot published by the 1 kHz loop; read by the slower predictors.

    Consumers must check ``timestamp`` and ``contact_mode``: a stale snapshot,
    or one taken in a different contact mode, must be replaced by a conservative
    fallback rather than used or waited on.
    """

    timestamp: float
    contact_mode: tuple[str, ...]

    tau_ff: np.ndarray
    lambda_ff: np.ndarray

    K_tau_body: np.ndarray
    K_lambda_body: np.ndarray

    H_body: np.ndarray
    h_body: np.ndarray

    torque_margin: np.ndarray
    friction_margin: np.ndarray
    normal_margin: np.ndarray

    valid: bool = True
    status: str = "ok"
    # Diagnostics
    nominal_residual: np.ndarray = field(default_factory=lambda: np.zeros(2))
    residual_gain: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))

    def contains(self, u: np.ndarray, tol: float = 1e-9) -> bool:
        if not self.valid or self.H_body.size == 0:
            return False
        return bool(np.all(self.H_body @ np.asarray(u, float) <= self.h_body + tol))

    def axis_extent(self, absolute_limit: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
        """Per-axis reach of the polytope: max t with t*(+/-e_i) admissible.

        This is the interpretable summary of remaining authority.  It is NOT the
        largest inscribed box: the polytope has coupled rows, so the max-area
        inscribed box can be far smaller than the axis reach (and can even pin an
        upper bound at zero while the axis itself is admissible).  Report this.
        """
        if not self.valid or self.H_body.size == 0:
            z = np.zeros(2)
            return z, z.copy()
        H, h = self.H_body, self.h_body
        lo, hi = np.zeros(2), np.zeros(2)
        for i in range(2):
            for sgn, out in ((+1.0, hi), (-1.0, lo)):
                d = np.zeros(2); d[i] = sgn
                Hd = H @ d
                # t*Hd <= h for all rows -> t <= min over rows with Hd > 0
                pos = Hd > 1e-12
                t = absolute_limit if not np.any(pos) else float(np.min(h[pos] / Hd[pos]))
                t = float(np.clip(t, 0.0, absolute_limit))
                out[i] = sgn * t
        return lo, hi

    def box(self, absolute_limit: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
        """Largest axis-aligned box inside {u : H u <= h}, as (lower, upper).

        Solved as a tiny LP over (center, radius); the row-wise support of the
        box is H_i c + |H_i| r.  Used where a predictor takes only bounds.
        """
        from scipy.optimize import linprog

        if not self.valid or self.H_body.size == 0:
            z = np.zeros(2)
            return z, z.copy()
        H, h = self.H_body, self.h_body
        A_ub = np.vstack((
            np.hstack((H, np.abs(H))),
            np.hstack((np.eye(2), -np.eye(2))),     # lower <= 0 (contain u=0)
            np.hstack((-np.eye(2), -np.eye(2))),    # upper >= 0
        ))
        b_ub = np.concatenate((h, np.zeros(2), np.zeros(2)))
        res = linprog(
            np.r_[np.zeros(2), -np.ones(2)],
            A_ub=A_ub, b_ub=b_ub,
            bounds=[(-absolute_limit, absolute_limit)] * 2 + [(0.0, absolute_limit)] * 2,
            method="highs",
        )
        if not res.success:
            z = np.zeros(2)
            return z, z.copy()
        c, r = res.x[:2], np.maximum(res.x[2:], 0.0)
        return c - r, c + r


@dataclass(frozen=True)
class TaskAuthority:
    """Residual capacity left for the task port AFTER the body allocation.

    Body-priority allocation: the body predictor is solved first against its own
    set; the task predictor then receives what is left,

        H_t u_t <= h_t - H_tb u_b*,

    so the two ports never both assume they own the whole actuator/contact
    budget.  This is an allocation POLICY, not an equivalence to a joint
    body-task optimization -- a different priority yields a different task set.
    """

    timestamp: float
    contact_mode: tuple[str, ...]
    H_task: np.ndarray
    h_task: np.ndarray
    K_tau_task: np.ndarray
    K_lambda_task: np.ndarray
    valid: bool = True
    status: str = "ok"

    def contains(self, u_t: np.ndarray, tol: float = 1e-9) -> bool:
        if not self.valid or self.H_task.size == 0:
            return False
        return bool(np.all(self.H_task @ np.asarray(u_t, float) <= self.h_task + tol))

    def axis_extent(self, absolute_limit: float = 6.0):
        if not self.valid or self.H_task.size == 0:
            z = np.zeros(3)
            return z, z.copy()
        lo, hi = np.zeros(3), np.zeros(3)
        for i in range(3):
            for sgn, out in ((+1.0, hi), (-1.0, lo)):
                d = np.zeros(3); d[i] = sgn
                Hd = self.H_task @ d
                pos = Hd > 1e-12
                t = (absolute_limit if not np.any(pos)
                     else float(np.min(self.h_task[pos] / Hd[pos])))
                out[i] = sgn * float(np.clip(t, 0.0, absolute_limit))
        return lo, hi


class AnalyticAuthorityMapper:
    """Build H_k u <= h_k from the realizer's own KKT sensitivity.

    Margins are fractional tightenings that absorb the local-cell approximation,
    inter-sample state motion, and estimation error.  They make the mapping
    conservative but not certified; the realizer stays the hard layer.
    """

    def __init__(
        self,
        *,
        torque_margin_fraction: float = 0.02,
        friction_margin_fraction: float = 0.04,
        normal_force_margin_n: float = 1.0,
        absolute_limit: float = 4.0,
        realization_tolerance: float = 0.35,
    ):
        self.torque_margin_fraction = float(torque_margin_fraction)
        self.friction_margin_fraction = float(friction_margin_fraction)
        self.normal_force_margin_n = float(normal_force_margin_n)
        self.absolute_limit = float(absolute_limit)
        self.realization_tolerance = float(realization_tolerance)

    def snapshot(
        self,
        realizer: Any,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        timestamp: float,
        contact_mode: tuple[str, ...],
    ) -> RealizationAuthority:
        """Called right after realizer.command() in the 1 kHz loop."""
        nv, nu = realizer.nv, realizer.nu
        empty = np.zeros((0, 2))

        def _invalid(status: str) -> RealizationAuthority:
            return RealizationAuthority(
                timestamp=timestamp, contact_mode=contact_mode,
                tau_ff=np.zeros(nu), lambda_ff=np.zeros(0),
                K_tau_body=np.zeros((nu, 2)), K_lambda_body=np.zeros((0, 2)),
                H_body=empty, h_body=np.zeros(0),
                torque_margin=np.zeros(nu), friction_margin=np.zeros(0),
                normal_margin=np.zeros(0),
                valid=False, status=status,
            )

        if realizer.last_fallback or realizer._com_dq_du is None:
            return _invalid("realizer fallback or no CoM objective")
        if realizer._com_clipped:
            return _invalid("CoM request on the clip; affine map invalid")

        K = realizer.input_sensitivity(realizer._com_dq_du)
        if K is None or not np.all(np.isfinite(K)):
            return _invalid("KKT sensitivity unavailable")

        z0 = realizer._qp_z
        nlam = realizer.last_contact_force.size
        tau_ff = z0[nv:nv + nu]
        lam_ff = z0[nv + nu:nv + nu + nlam]
        K_tau = K[nv:nv + nu]
        K_lam = K[nv + nu:nv + nu + nlam]

        rows: list[np.ndarray] = []
        bounds: list[float] = []

        # --- actuator torque bounds:  tau_min <= tau_ff + K_tau u <= tau_max
        span = realizer.torque_max - realizer.torque_min
        tau_lo = realizer.torque_min + self.torque_margin_fraction * span
        tau_hi = realizer.torque_max - self.torque_margin_fraction * span
        for j in range(nu):
            rows.append(K_tau[j]);      bounds.append(tau_hi[j] - tau_ff[j])
            rows.append(-K_tau[j]);     bounds.append(tau_ff[j] - tau_lo[j])

        # --- friction pyramid and unilateral normal force on each contact
        mu = realizer.mu * (1.0 - self.friction_margin_fraction)
        for c in range(nlam // 3):
            b = 3 * c
            fz_ff, Kz = lam_ff[b + 2], K_lam[b + 2]
            for t in (0, 1):
                ft_ff, Kt = lam_ff[b + t], K_lam[b + t]
                #  ft - mu fz <= 0   and   -ft - mu fz <= 0
                rows.append(Kt - mu * Kz);   bounds.append(mu * fz_ff - ft_ff)
                rows.append(-Kt - mu * Kz);  bounds.append(mu * fz_ff + ft_ff)
            #  fz >= fz_min   ->   -Kz u <= fz_ff - fz_min
            rows.append(-Kz)
            bounds.append(fz_ff - self.normal_force_margin_n)

        # --- realization tolerance: the criterion the set is DEFINED by.
        # Mapping only the physical limits is not enough: the QP may satisfy every
        # limit and still fail to deliver the requested CoM acceleration (it trades
        # the request against its other objectives).  Feasibility means the
        # realized-minus-requested residual stays inside the tolerance, so that
        # test must appear as constraint rows, linearized about the current cell:
        #     |residual0 + (J_com K - I) u| <= tol
        # Without these rows the set is unsound in exactly the regime that matters
        # -- e.g. single support, where the nominal residual already exceeds the
        # tolerance and every "admissible" command is in fact unrealizable.
        Jcom = np.zeros((3, nv))
        mujoco.mj_jacSubtreeCom(model, data, Jcom, realizer.root_body)
        residual0 = (Jcom[:2] @ realizer.last_qdd
                     - np.asarray(realizer.last_com_acc_des, float)[:2])
        residual_gain = Jcom[:2] @ K[:nv] - np.eye(2)
        tol = self.realization_tolerance
        if np.max(np.abs(residual0)) > tol:
            # u = 0 itself is not realizable: the set is empty, not merely tight.
            return _invalid("nominal realization residual exceeds tolerance")
        for i in range(2):
            rows.append(residual_gain[i]);   bounds.append(tol - residual0[i])
            rows.append(-residual_gain[i]);  bounds.append(tol + residual0[i])

        # --- numerical backstop
        for i in range(2):
            e = np.zeros(2); e[i] = 1.0
            rows.append(e);   bounds.append(self.absolute_limit)
            rows.append(-e);  bounds.append(self.absolute_limit)

        H = np.vstack(rows)
        h = np.asarray(bounds, dtype=float)

        # h_k < 0 handling.  Clamping such a row to H_i u <= 0 RELAXES the
        # linearized margined row (u = 0 violated it), so the set is no longer
        # conservative *by construction* against the margined limits.  It is
        # nevertheless safe against the PHYSICAL limits, and it is worth being
        # precise about why: the whole-body QP enforces tau in [tau_min,tau_max]
        # and the friction pyramid as HARD constraints, so tau_ff and lambda_ff
        # always satisfy the TRUE limits.  h_i < 0 can therefore only mean the
        # *safety margin* (torque/friction/normal fractions above) has been eaten
        # -- never that a true limit is violated.  Clamping says "do not push
        # further this way", which keeps tau <= tau_ff <= tau_max.
        #
        # Two regimes are still distinguished, because they mean different things:
        #   |h_i| <= EPS   numerical noise on an active row -> clamp silently
        #   h_i  <  -EPS   the margin is genuinely consumed -> clamp, but report
        #                  it so the caller can treat the snapshot as degraded
        #                  rather than trusting it as a fresh margin estimate.
        # Soundness here rests on the realizer's hard constraints, NOT on the
        # mapper's construction; the mapper is not a certificate.
        EPS_H = 1e-6
        n_numerical = int(np.sum((h < 0.0) & (h >= -EPS_H)))
        n_margin_exhausted = int(np.sum(h < -EPS_H))
        h = np.maximum(h, 0.0)

        # Joint body+task sensitivity, so the task port can be given the capacity
        # the body did not spend.  u_t enters the QP through the hand-task
        # objective's target only, exactly like u_c, so the same KKT system
        # returns dz/du_t at no extra QP solve.
        self._last_joint = None
        if realizer._task_dq_du is not None and not realizer._task_clipped:
            dq_du_joint = np.hstack((realizer._com_dq_du, realizer._task_dq_du))
            Kj = realizer.input_sensitivity(dq_du_joint)
            if Kj is not None and np.all(np.isfinite(Kj)):
                self._last_joint = dict(
                    K_tau=Kj[nv:nv + nu], K_lam=Kj[nv + nu:nv + nu + nlam],
                    K_qdd=Kj[:nv], tau_ff=tau_ff, lam_ff=lam_ff,
                    timestamp=timestamp, contact_mode=contact_mode,
                )

        return RealizationAuthority(
            timestamp=timestamp, contact_mode=contact_mode,
            tau_ff=tau_ff.copy(), lambda_ff=lam_ff.copy(),
            K_tau_body=K_tau.copy(), K_lambda_body=K_lam.copy(),
            H_body=H, h_body=h,
            torque_margin=np.minimum(tau_hi - tau_ff, tau_ff - tau_lo),
            friction_margin=np.array([
                mu * lam_ff[3 * c + 2] - max(abs(lam_ff[3 * c]), abs(lam_ff[3 * c + 1]))
                for c in range(nlam // 3)
            ]),
            normal_margin=np.array([
                lam_ff[3 * c + 2] - self.normal_force_margin_n for c in range(nlam // 3)
            ]),
            valid=True,
            status=("ok" if n_margin_exhausted == 0
                    else f"margin_exhausted ({n_margin_exhausted} rows; "
                         f"{n_numerical} numerical)"),
            nominal_residual=residual0.copy(),
            residual_gain=residual_gain,
        )

    def task_authority(
        self,
        realizer: Any,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        u_body: np.ndarray,
        *,
        task_tolerance: float = 0.5,
    ) -> TaskAuthority:
        """Task capacity remaining after the body allocation u_body.

        Must be called after ``snapshot()`` on the same realizer solve.  Uses the
        joint sensitivity taken there, so it costs no additional QP or KKT solve.
        """
        j = getattr(self, "_last_joint", None)
        if j is None:
            z3 = np.zeros((0, 3))
            return TaskAuthority(0.0, (), z3, np.zeros(0),
                                 np.zeros((realizer.nu, 3)), np.zeros((0, 3)),
                                 valid=False, status="joint sensitivity unavailable")
        nv, nu = realizer.nv, realizer.nu
        nlam = realizer.last_contact_force.size
        ub = np.asarray(u_body, float).reshape(2)

        K_tau_b, K_tau_t = j["K_tau"][:, :2], j["K_tau"][:, 2:]
        K_lam_b, K_lam_t = j["K_lam"][:, :2], j["K_lam"][:, 2:]
        # Feedforward already committed by the body command.
        tau0 = j["tau_ff"] + K_tau_b @ ub
        lam0 = j["lam_ff"] + K_lam_b @ ub

        rows: list[np.ndarray] = []
        bnds: list[float] = []
        span = realizer.torque_max - realizer.torque_min
        tau_lo = realizer.torque_min + self.torque_margin_fraction * span
        tau_hi = realizer.torque_max - self.torque_margin_fraction * span
        for a in range(nu):
            rows.append(K_tau_t[a]);   bnds.append(tau_hi[a] - tau0[a])
            rows.append(-K_tau_t[a]);  bnds.append(tau0[a] - tau_lo[a])
        mu = realizer.mu * (1.0 - self.friction_margin_fraction)
        for c in range(nlam // 3):
            b = 3 * c
            fz0, Kz = lam0[b + 2], K_lam_t[b + 2]
            for t in (0, 1):
                ft0, Kt = lam0[b + t], K_lam_t[b + t]
                rows.append(Kt - mu * Kz);   bnds.append(mu * fz0 - ft0)
                rows.append(-Kt - mu * Kz);  bnds.append(mu * fz0 + ft0)
            rows.append(-Kz);  bnds.append(fz0 - self.normal_force_margin_n)

        # Task realization tolerance, linearized about this cell.
        Jt = np.zeros((3, nv))
        sid = None
        Jh = realizer.last_hand_jac if hasattr(realizer, "last_hand_jac") else None
        if Jh is None:
            z3 = np.zeros((0, 3))
            return TaskAuthority(j["timestamp"], j["contact_mode"], z3, np.zeros(0),
                                 K_tau_t, K_lam_t, valid=False,
                                 status="hand Jacobian unavailable")
        Jt = np.asarray(Jh, float)
        res_t0 = (Jt @ realizer.last_qdd
                  - np.asarray(realizer.last_task_acc_des, float))
        gain_t = Jt @ j["K_qdd"][:, 2:] - np.eye(3)
        if np.max(np.abs(res_t0)) > task_tolerance:
            z3 = np.zeros((0, 3))
            return TaskAuthority(j["timestamp"], j["contact_mode"], z3, np.zeros(0),
                                 K_tau_t, K_lam_t, valid=False,
                                 status="nominal task residual exceeds tolerance")
        for i in range(3):
            rows.append(gain_t[i]);   bnds.append(task_tolerance - res_t0[i])
            rows.append(-gain_t[i]);  bnds.append(task_tolerance + res_t0[i])
        for i in range(3):
            e = np.zeros(3); e[i] = 1.0
            rows.append(e);   bnds.append(6.0)
            rows.append(-e);  bnds.append(6.0)

        H = np.vstack(rows)
        h = np.asarray(bnds, float)
        n_exh = int(np.sum(h < -1e-6))
        h = np.maximum(h, 0.0)
        return TaskAuthority(
            j["timestamp"], j["contact_mode"], H, h, K_tau_t, K_lam_t,
            valid=True,
            status="ok" if n_exh == 0 else f"margin_exhausted ({n_exh} rows)",
        )


class ContinuationAuthorityEstimator:
    """Tight authority by walking the QP's piecewise-affine solution map.

    The single-active-set map of ``AnalyticAuthorityMapper`` is conservative for a
    precise reason: {H_k u <= h_k} is essentially the CRITICAL REGION of the
    current active set -- the set on which no new constraint activates.  But
    crossing an active-set boundary is not a failure: the QP simply redistributes
    (a contact force saturates, another takes over) and keeps realizing the
    request.  The true feasible set therefore spans several critical regions, and
    the single-cell map refuses ~68% of it.

    This estimator follows the solution across those regions.  Along a ray, the
    solution is affine until one of two events occurs:

        * an inactive constraint reaches its bound   (a row ENTERS the active set)
        * an active constraint's multiplier hits 0   (a row LEAVES it)

    Both events are found in closed form from the same KKT system, whose primal
    block gives dz/du and whose dual block gives dnu/du.  At the event the active
    set is updated and the KKT is re-solved; the walk continues.  The ray stops
    where the realization residual -- which is affine on each region -- crosses the
    tolerance.  That crossing IS the authority boundary.

    Cost: one KKT solve per region traversed, and ZERO extra whole-body QP solves.
    Contrast with the offline oracle, which needs ~62 QP solves.
    """

    def __init__(
        self,
        *,
        realization_tolerance: float = 0.35,
        absolute_limit: float = 4.0,
        max_regions: int = 12,
        active_tol: float = 1e-6,
        dual_tol: float = 1e-7,
    ):
        self.realization_tolerance = float(realization_tolerance)
        self.absolute_limit = float(absolute_limit)
        self.max_regions = int(max_regions)
        self.active_tol = float(active_tol)
        self.dual_tol = float(dual_tol)
        self.kkt_solves = 0

    # -- one ray -----------------------------------------------------------
    def _walk_ray(self, realizer, Jout, nom, direction, Qu):
        """Return the distance along `direction` at which the residual leaves tol."""
        P = realizer._qp_P
        A = realizer._qp_A
        lo, hi = realizer._qp_l, realizer._qp_u
        z = realizer._qp_z.copy()
        y = realizer._qp_y.copy()
        nv = realizer.nv
        n = P.shape[0]
        d = np.asarray(direction, float)

        equality = np.isclose(lo, hi)
        Az = A @ z
        at_hi = np.isfinite(hi) & (np.abs(Az - hi) <= self.active_tol)
        at_lo = np.isfinite(lo) & (np.abs(Az - lo) <= self.active_tol)
        active = equality | ((at_hi | at_lo) & (np.abs(y) > self.active_tol))

        t = 0.0
        r = Jout @ z[:nv] - nom             # residual at u = 0 (t = 0)
        tol = self.realization_tolerance

        for _ in range(self.max_regions):
            Aa = A[active]
            na = int(Aa.shape[0])
            K = np.block([[P, Aa.T], [Aa, np.zeros((na, na))]]) if na else P
            rhs = np.concatenate((-(Qu @ d), np.zeros(na)))
            try:
                sol = np.linalg.solve(K + 1e-9 * np.eye(K.shape[0]), rhs)
            except np.linalg.LinAlgError:
                return t                      # degenerate: stop where we are
            self.kkt_solves += 1
            dz = sol[:n]
            dnu = sol[n:]

            # --- residual crossing on this region (affine in s)
            dr = Jout @ dz[:nv] - d
            s_tol = np.inf
            for i in range(d.size):
                if abs(dr[i]) > 1e-12:
                    for bound in (tol, -tol):
                        s = (bound - r[i]) / dr[i]
                        if s > 1e-9:
                            s_tol = min(s_tol, s)

            # --- a currently-inactive row ENTERS the active set (vectorized)
            rate = A @ dz
            val = A @ z
            inact = ~active
            s_hi = np.where(inact & (rate > 1e-10) & np.isfinite(hi),
                            (hi - val) / np.where(np.abs(rate) > 1e-12, rate, 1.0), np.inf)
            s_lo = np.where(inact & (rate < -1e-10) & np.isfinite(lo),
                            (lo - val) / np.where(np.abs(rate) > 1e-12, rate, 1.0), np.inf)
            s_row = np.minimum(s_hi, s_lo)
            s_row = np.where(s_row > 1e-9, s_row, np.inf)
            enter_idx = int(np.argmin(s_row))
            s_enter = float(s_row[enter_idx])
            if not np.isfinite(s_enter):
                enter_idx = -1

            # --- an active row LEAVES (its multiplier reaches zero) (vectorized)
            act_idx = np.flatnonzero(active)
            movable = ~equality[act_idx]
            s_act = np.full(act_idx.size, np.inf)
            safe = movable & (np.abs(dnu) > self.dual_tol)
            s_act[safe] = -y[act_idx][safe] / dnu[safe]
            s_act = np.where(s_act > 1e-9, s_act, np.inf)
            if s_act.size:
                k_leave = int(np.argmin(s_act))
                s_leave = float(s_act[k_leave])
                leave_idx = int(act_idx[k_leave]) if np.isfinite(s_leave) else -1
            else:
                s_leave, leave_idx = np.inf, -1

            s_cap = self.absolute_limit - t
            s_star = min(s_tol, s_enter, s_leave, s_cap)
            if not np.isfinite(s_star):
                return min(t + s_cap, self.absolute_limit)

            # residual tolerance reached first -> this is the authority boundary
            if s_star == s_tol:
                return t + s_tol
            if s_star == s_cap:
                return self.absolute_limit

            # advance to the breakpoint and switch the active set
            z = z + s_star * dz
            y = y.copy()
            y[act_idx] = y[act_idx] + s_star * dnu
            r = r + s_star * dr
            t += s_star
            if s_star == s_enter and enter_idx >= 0:
                active[enter_idx] = True
            elif s_star == s_leave and leave_idx >= 0:
                active[leave_idx] = False
        return t

    # -- box ----------------------------------------------------------------
    def estimate(self, realizer, model, data, *, timestamp=0.0, contact_mode=()):
        self.kkt_solves = 0
        nv = realizer.nv
        if realizer.last_fallback or realizer._com_dq_du is None or realizer._qp_z is None:
            z = np.zeros(2)
            return AuthorityBox(z, z.copy(), z.copy(), z.copy(), "invalid",
                                np.full(2, np.inf), False, "realizer unavailable", 0, 0.0)
        Jcom = np.zeros((3, nv))
        mujoco.mj_jacSubtreeCom(model, data, Jcom, realizer.root_body)
        Jcom2 = Jcom[:2]
        nom2 = np.asarray(realizer.last_com_acc_des, float)[:2]
        r0 = Jcom2 @ realizer.last_qdd - nom2
        if np.max(np.abs(r0)) > self.realization_tolerance:
            z = np.zeros(2)
            return AuthorityBox(z, z.copy(), z.copy(), z.copy(), "invalid", r0, False,
                                "nominal realization residual exceeds tolerance", 0, 0.0)

        lower, upper = np.zeros(2), np.zeros(2)
        for axis in range(2):
            e = np.zeros(2); e[axis] = 1.0
            upper[axis] = self._walk_ray(realizer, Jcom2, nom2, e, realizer._com_dq_du)
            lower[axis] = -self._walk_ray(realizer, Jcom2, nom2, -e, realizer._com_dq_du)

        center = 0.5 * (lower + upper)
        radius = 0.5 * (upper - lower)
        return AuthorityBox(lower, upper, center, radius, "realization-residual",
                            r0, bool(np.all(radius > 0.0)),
                            "pwa continuation", self.kkt_solves, 1.0)


    def estimate_task(self, realizer, model, data, *, task_tolerance: float = 0.5):
        """Continuation authority for the TASK port (3-D walk).

        Same machinery, different output map: the residual is measured on the hand
        (J_t) and the input enters through the task objective's target, so the cost
        gradient is ``realizer._task_dq_du``.  Zero extra whole-body QP solves.
        """
        self.kkt_solves = 0
        if (realizer.last_fallback or realizer._task_dq_du is None
                or realizer._qp_z is None or realizer.last_hand_jac is None):
            z = np.zeros(3)
            return AuthorityBox(z, z.copy(), z.copy(), z.copy(), "invalid",
                                np.full(3, np.inf), False, "realizer unavailable", 0, 0.0)
        Jt = np.asarray(realizer.last_hand_jac, float)
        nom = np.asarray(realizer.last_task_acc_des, float)
        r0 = Jt @ realizer.last_qdd - nom
        if np.max(np.abs(r0)) > task_tolerance:
            z = np.zeros(3)
            return AuthorityBox(z, z.copy(), z.copy(), z.copy(), "invalid", r0, False,
                                "nominal task residual exceeds tolerance", 0, 0.0)
        saved_tol = self.realization_tolerance
        self.realization_tolerance = float(task_tolerance)
        lower, upper = np.zeros(3), np.zeros(3)
        try:
            for axis in range(3):
                e = np.zeros(3); e[axis] = 1.0
                upper[axis] = self._walk_ray(realizer, Jt, nom, e, realizer._task_dq_du)
                lower[axis] = -self._walk_ray(realizer, Jt, nom, -e, realizer._task_dq_du)
        finally:
            self.realization_tolerance = saved_tol
        center = 0.5 * (lower + upper)
        radius = 0.5 * (upper - lower)
        return AuthorityBox(lower, upper, center, radius, "task-residual", r0,
                            bool(np.all(radius > 0.0)), "pwa continuation (task)",
                            self.kkt_solves, 1.0)



# --------------------------------------------------------------------------
# Offline ground truth: exact residual bisection (NOT in the control path)
# --------------------------------------------------------------------------


@dataclass
class AuthorityBox:
    lower: np.ndarray
    upper: np.ndarray
    center: np.ndarray
    radius: np.ndarray
    active_constraint: str
    nominal_residual: np.ndarray
    valid: bool
    status: str
    solve_count: int = 0
    corner_scale: float = 1.0

    @property
    def area(self) -> float:
        return float(np.prod(np.maximum(self.upper - self.lower, 0.0)))

    def corners(self) -> np.ndarray:
        return np.asarray(list(product(*zip(self.lower, self.upper))), dtype=float)


class ExactResidualBisectionEstimator:
    """Offline reference: bisect the exact realizer's residual along each ray.

    ~62 QP solves per query.  Used to grade AnalyticAuthorityMapper, never to
    run the robot.
    """

    def __init__(
        self,
        *,
        absolute_limit: float | np.ndarray = 4.0,
        realization_tolerance: float = 0.35,
        bisection_iterations: int = 7,
    ):
        lim = np.asarray(absolute_limit, dtype=float)
        if lim.ndim == 0:
            lim = np.full(2, float(lim))
        self.absolute_limit = lim.reshape(2)
        self.realization_tolerance = float(realization_tolerance)
        self.bisection_iterations = int(bisection_iterations)

    def estimate(
        self,
        realizer: Any,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        q_ref: np.ndarray,
        qd_ref: np.ndarray,
        task_acc_des: np.ndarray,
        hand_jac: np.ndarray,
        stance_contacts: dict,
        stance_targets: dict,
        base_height_ref: float,
        rpy: np.ndarray,
        nominal_com_acc_des: np.ndarray,
        swing_task: dict | None = None,
        attitude_weight: float = 8.0,
        centroidal_moment_des: np.ndarray | None = None,
    ) -> AuthorityBox:
        nominal_com = np.asarray(nominal_com_acc_des, dtype=float).reshape(3)
        saved_ctrl = data.ctrl.copy()
        cache: dict[tuple[float, float], tuple[bool, np.ndarray]] = {}

        def evaluate(offset: np.ndarray) -> tuple[bool, np.ndarray]:
            key = tuple(np.round(np.asarray(offset, dtype=float), 12))
            if key in cache:
                return cache[key]
            request = nominal_com.copy()
            request[:2] += offset
            realizer.command(
                model, data, q_ref, qd_ref, request[:2], task_acc_des,
                hand_jac, stance_contacts, stance_targets, base_height_ref, rpy,
                com_acc_des=request, swing_task=swing_task,
                attitude_weight=attitude_weight,
                centroidal_moment_des=centroidal_moment_des,
            )
            if realizer.last_fallback or not np.all(np.isfinite(realizer.last_qdd)):
                result = (False, np.full(2, np.inf))
            else:
                Jcom = np.zeros((3, realizer.nv))
                mujoco.mj_jacSubtreeCom(model, data, Jcom, realizer.root_body)
                residual = Jcom[:2] @ realizer.last_qdd - request[:2]
                result = (
                    bool(np.all(np.isfinite(residual))
                         and np.max(np.abs(residual)) <= self.realization_tolerance),
                    residual,
                )
            cache[key] = result
            return result

        def ray_boundary(direction: np.ndarray, maximum: float) -> float:
            high = float(maximum)
            if evaluate(high * direction)[0]:
                return high
            low = 0.0
            for _ in range(self.bisection_iterations):
                mid = 0.5 * (low + high)
                if evaluate(mid * direction)[0]:
                    low = mid
                else:
                    high = mid
            return low

        try:
            ok, residual0 = evaluate(np.zeros(2))
            if not ok:
                return AuthorityBox(
                    np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2),
                    "invalid", residual0, False,
                    "nominal realization residual exceeds tolerance",
                    len(cache), 0.0,
                )
            lower, upper = np.zeros(2), np.zeros(2)
            for axis in range(2):
                d = np.zeros(2); d[axis] = 1.0
                upper[axis] = ray_boundary(d, self.absolute_limit[axis])
                d[axis] = -1.0
                lower[axis] = -ray_boundary(d, self.absolute_limit[axis])

            def corners_ok(scale: float) -> bool:
                return all(evaluate(np.asarray(c))[0]
                           for c in product(*zip(scale * lower, scale * upper)))

            corner_scale = 1.0
            if not corners_ok(1.0):
                lo, hi = 0.0, 1.0
                for _ in range(self.bisection_iterations):
                    mid = 0.5 * (lo + hi)
                    if corners_ok(mid):
                        lo = mid
                    else:
                        hi = mid
                corner_scale = lo
                lower *= corner_scale
                upper *= corner_scale

            center = 0.5 * (lower + upper)
            radius = 0.5 * (upper - lower)
            active = ("absolute-limit"
                      if np.allclose(np.maximum(np.abs(lower), np.abs(upper)),
                                     self.absolute_limit)
                      else "realization-residual")
            return AuthorityBox(
                lower, upper, center, radius, active, residual0.copy(),
                bool(np.all(radius > 0.0)), "exact residual bisection",
                len(cache), corner_scale,
            )
        finally:
            data.ctrl[:] = saved_ctrl


# Backwards-compatible alias: the old name referred to the bisection estimator.
PlanarBodyAuthorityEstimator = ExactResidualBisectionEstimator
