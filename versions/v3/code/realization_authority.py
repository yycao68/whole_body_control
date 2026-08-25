"""Realization-informed command authority for the canonical predictors.

Two estimators live here, with very different roles.

``AnalyticAuthorityMapper`` (the controller path).  The 200 Hz active-mode
optimization node already solves one whole-body QP.  The 1 kHz servo holds its
published torque. The residual command ``u`` enters that QP
only through objective *linear* terms, and the QP Hessian does not depend on
``u``; therefore, under regularity on the current active-set cell, the solution
is affine,

    z(u) = z0 + K u,      K = dz/du   (one KKT solve, no extra QP solves)

so the nominal feedforward and the input maps

    tau    = tau_ff    + K_tau    u
    lambda = lambda_ff + K_lambda u

come out of the cycle the realizer is running anyway.  Substituting them into
the *physical* limits the realizer enforces -- actuator bounds, friction
pyramid, unilateral normal force -- yields linear constraints on the residual
command,

    H_k u <= h_k,

Together with primal feasibility of every inactive QP row and sign feasibility
of each active inequality multiplier, these rows form the fixed-active-set
critical region.  The canonical pair (A, B) is untouched: contact mode and
configuration change only (H_k, h_k).  Cost is one KKT solve (~0.15 ms), not
62 QP solves (~154 ms).

The affine map is exact only on the *current active-set cell*.  Once a
constraint activates, the true QP redistributes and the map bends, so H_k u <=
h_k is a local model and NOT a certificate.  The instantaneous realizer remains
the final modeled-constraint layer, and the mapping's optimism/conservatism is
quantified offline against the repeated-QP numerical reference below.

``ExactResidualBisectionEstimator`` (offline repeated-QP numerical reference).
Repeatedly re-solves the modeled-constrained realizer and bisects sampled signed
coordinate rays on the realized-vs-requested acceleration residual.  It is not
an exact full-dimensional oracle or certificate.  At ~62 QP solves and O(100
ms), it is a *measurement procedure*, not part of the feedback loop; its job is
to grade the analytic mapping on the sampled rays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from itertools import product
from typing import Any

import mujoco
import numpy as np


# --------------------------------------------------------------------------
# Controller path: analytic mapping from the 1 kHz realizer
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RealizationAuthority:
    """Snapshot published by the active-mode QP node; read by the predictors.

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
    # The absolute command this H_body/h_body was translated relative to
    # (H(u-u_ref)<=h_local published as Hu<=h_local+H@u_ref). This is the one
    # point the polytope's own construction guarantees is admissible -- the
    # consumer's solve-failure fallback, not u=0.
    command_reference: np.ndarray = field(default_factory=lambda: np.zeros(2))

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
    command_reference: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def contains(self, u_t: np.ndarray, tol: float = 1e-9) -> bool:
        if not self.valid or self.H_task.size == 0:
            return False
        return bool(np.all(self.H_task @ np.asarray(u_t, float) <= self.h_task + tol))

    def ray_extent(
        self,
        direction: np.ndarray,
        *,
        origin: np.ndarray | None = None,
        maximum: float = 1.0,
    ) -> float:
        """Return the coupled-polytope reach on ``origin + s * direction``.

        This is a directional query through the complete task polytope.  It is
        intentionally distinct from ``axis_extent()``, whose independent
        coordinate reaches do not define jointly feasible task commands.
        """
        d = np.asarray(direction, dtype=float).reshape(3)
        o = np.zeros(3) if origin is None else np.asarray(origin, dtype=float).reshape(3)
        if maximum <= 0.0 or not np.all(np.isfinite(d)) or not self.contains(o):
            return 0.0
        Hd = self.H_task @ d
        rhs = self.h_task - self.H_task @ o
        positive = Hd > 1e-12
        if not np.any(positive):
            return float(maximum)
        return float(np.clip(np.min(rhs[positive] / Hd[positive]), 0.0, maximum))

    def on_polytope_face(self, u_t: np.ndarray, tol: float = 1e-6) -> bool:
        """Return whether a feasible task command is numerically on a face."""
        if not self.contains(u_t, tol=tol):
            return False
        slack = self.h_task - self.H_task @ np.asarray(u_t, dtype=float).reshape(3)
        return bool(np.min(slack) <= tol)

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


@dataclass(frozen=True)
class AugmentedBodyAuthority:
    """Local authority for $[u_x,u_y,u_\psi]$ in absolute coordinates.

    This development-path object is deliberately separate from
    ``RealizationAuthority`` so existing 2-D paper experiments are unchanged.
    ``u_psi`` is a base yaw-acceleration request to the realizer's attitude
    objective; it is not yet the paper's angular-momentum coordinate.
    """

    timestamp: float
    contact_mode: tuple[str, ...]
    H: np.ndarray
    h: np.ndarray
    valid: bool = True
    status: str = "ok"
    command_reference: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def contains(self, u: np.ndarray, tol: float = 1e-9) -> bool:
        """Return whether an absolute three-dimensional command is in the map."""
        if not self.valid or self.H.size == 0:
            return False
        return bool(np.all(self.H @ np.asarray(u, dtype=float).reshape(3) <= self.h + tol))

    def ray_extent(
        self,
        direction: np.ndarray,
        *,
        origin: np.ndarray | None = None,
        maximum: float = 1.0,
    ) -> float:
        """Return the coupled-polytope reach on ``origin + s * direction``.

        This is a ray query, not an axis-aligned box construction.  In
        particular, the result along a simultaneous three-axis direction can be
        much smaller than the independent coordinate-ray reaches.
        """
        d = np.asarray(direction, dtype=float).reshape(3)
        o = np.zeros(3) if origin is None else np.asarray(origin, dtype=float).reshape(3)
        if maximum <= 0.0 or not np.all(np.isfinite(d)) or not self.contains(o):
            return 0.0
        Hd = self.H @ d
        rhs = self.h - self.H @ o
        positive = Hd > 1e-12
        if not np.any(positive):
            return float(maximum)
        return float(np.clip(np.min(rhs[positive] / Hd[positive]), 0.0, maximum))

    def box(self, absolute_limit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return independent coordinate-ray reaches for diagnostics only.

        The returned lower/upper values do not define a jointly feasible box;
        callers that need a feasible coupled command must enforce ``H u <= h``.
        """
        limit = np.asarray(absolute_limit, dtype=float).reshape(3)
        if not self.valid or self.H.size == 0:
            return np.zeros(3), np.zeros(3)
        lo, hi = -limit.copy(), limit.copy()
        for i in range(3):
            for sign, target in ((1.0, hi), (-1.0, lo)):
                direction = np.zeros(3); direction[i] = sign
                Hd = self.H @ direction
                pos = Hd > 1e-12
                reach = limit[i] if not np.any(pos) else float(np.min(self.h[pos] / Hd[pos]))
                target[i] = sign * float(np.clip(reach, 0.0, limit[i]))
        return lo, hi


class AugmentedBodyAuthorityMapper:
    """Experimental local map from the current QP cell to planar-CoM plus yaw.

    The map carries the same complete fixed-active-set primal and dual
    critical-region rows as :class:`AnalyticAuthorityMapper`, but remains an
    offline development object until it has passed the separate 3-D sampled
    repeated-QP validation.  It must not be used to expand the paper's 2-D
    control claim before that validation is complete.
    """

    def __init__(
        self,
        *,
        torque_margin_fraction: float = 0.02,
        friction_margin_fraction: float = 0.04,
        normal_force_margin_n: float = 1.0,
        realization_tolerance: float = 0.35,
    ):
        self.torque_margin_fraction = float(torque_margin_fraction)
        self.friction_margin_fraction = float(friction_margin_fraction)
        self.normal_force_margin_n = float(normal_force_margin_n)
        self.realization_tolerance = float(realization_tolerance)

    def snapshot_augmented(
        self,
        realizer: Any,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        timestamp: float,
        contact_mode: tuple[str, ...],
        command_reference: np.ndarray,
        yaw_tolerance: float = 1.0,
        absolute_limit: np.ndarray = np.array([4.0, 4.0, 8.0]),
    ) -> AugmentedBodyAuthority:
        u_ref = np.asarray(command_reference, dtype=float).reshape(3)
        limit = np.asarray(absolute_limit, dtype=float).reshape(3)
        if (realizer.last_fallback or realizer._body_dq_du is None
                or realizer._body_dq_du.shape[1] != 3
                or not np.all(np.isfinite(u_ref))):
            return AugmentedBodyAuthority(timestamp, contact_mode,
                                          np.zeros((0, 3)), np.zeros(0), False,
                                          "realizer fallback or missing body sensitivity")
        sensitivity = realizer.input_sensitivity_with_duals(realizer._body_dq_du)
        if sensitivity is None:
            return AugmentedBodyAuthority(timestamp, contact_mode,
                                          np.zeros((0, 3)), np.zeros(0), False,
                                          "KKT sensitivity unavailable")
        K, dnu, active, at_lower, at_upper = sensitivity
        if not (np.all(np.isfinite(K)) and np.all(np.isfinite(dnu))):
            return AugmentedBodyAuthority(timestamp, contact_mode,
                                          np.zeros((0, 3)), np.zeros(0), False,
                                          "KKT sensitivity unavailable")
        nv, nu = realizer.nv, realizer.nu
        z0 = realizer._qp_z
        if z0 is None or not np.all(np.isfinite(z0)):
            return AugmentedBodyAuthority(timestamp, contact_mode,
                                          np.zeros((0, 3)), np.zeros(0), False,
                                          "non-finite nominal QP point")
        nlam = realizer.last_contact_force.size
        tau, lam = z0[nv:nv + nu], z0[nv + nu:nv + nu + nlam]
        Ktau, Klam = K[nv:nv + nu], K[nv + nu:nv + nu + nlam]
        rows: list[np.ndarray] = []
        bounds: list[float] = []

        # Keep the local affine sensitivity inside the QP active-set cell.  The
        # physical-margin rows below are insufficient by themselves: an inactive
        # QP row may become infeasible, or an active inequality dual may change
        # sign, before a torque/contact margin is reached.
        A_qp, l_qp, u_qp, y_qp = (
            realizer._qp_A, realizer._qp_l, realizer._qp_u, realizer._qp_y,
        )
        if not (
            A_qp is not None and l_qp is not None and u_qp is not None
            and y_qp is not None and A_qp.shape[1] == K.shape[0]
            and np.all(np.isfinite(A_qp)) and np.all(np.isfinite(y_qp))
        ):
            return AugmentedBodyAuthority(timestamp, contact_mode,
                                          np.zeros((0, 3)), np.zeros(0), False,
                                          "QP rows unavailable for critical-region mapping")
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            Az0 = A_qp @ z0
        if not np.all(np.isfinite(Az0)):
            return AugmentedBodyAuthority(timestamp, contact_mode,
                                          np.zeros((0, 3)), np.zeros(0), False,
                                          "non-finite nominal QP row value")
        equality = np.isclose(l_qp, u_qp)
        for i in np.flatnonzero(~active):
            row_gain = A_qp[i] @ K
            if np.isfinite(u_qp[i]):
                rows.append(row_gain)
                bounds.append(float(u_qp[i] - Az0[i]))
            if np.isfinite(l_qp[i]):
                rows.append(-row_gain)
                bounds.append(float(Az0[i] - l_qp[i]))
        active_indices = np.flatnonzero(active)
        for local_index, qp_index in enumerate(active_indices):
            if equality[qp_index]:
                continue
            if at_upper[qp_index]:
                rows.append(-dnu[local_index])
                bounds.append(float(y_qp[qp_index]))
            elif at_lower[qp_index]:
                rows.append(dnu[local_index])
                bounds.append(float(-y_qp[qp_index]))
            else:
                return AugmentedBodyAuthority(timestamp, contact_mode,
                                              np.zeros((0, 3)), np.zeros(0), False,
                                              "active inequality has no finite bound side")
        critical_row_count = len(rows)

        span = realizer.torque_max - realizer.torque_min
        tau_lo = realizer.torque_min + self.torque_margin_fraction * span
        tau_hi = realizer.torque_max - self.torque_margin_fraction * span
        for j in range(nu):
            rows.extend((Ktau[j], -Ktau[j]))
            bounds.extend((tau_hi[j] - tau[j], tau[j] - tau_lo[j]))
        mu = realizer.mu * (1.0 - self.friction_margin_fraction)
        for c in range(nlam // 3):
            base = 3 * c
            for axis in (0, 1):
                rows.extend((Klam[base + axis] - mu * Klam[base + 2],
                             -Klam[base + axis] - mu * Klam[base + 2]))
                bounds.extend((mu * lam[base + 2] - lam[base + axis],
                               mu * lam[base + 2] + lam[base + axis]))
            rows.append(-Klam[base + 2])
            bounds.append(lam[base + 2] - self.normal_force_margin_n)
        Jcom = np.zeros((3, nv))
        mujoco.mj_jacSubtreeCom(model, data, Jcom, realizer.root_body)
        planar_residual = Jcom[:2] @ realizer.last_qdd - np.asarray(realizer.last_com_acc_des)[:2]
        planar_gain = Jcom[:2] @ K[:nv] - np.eye(3)[:2]
        if realizer.last_yaw_output_kind == "moment":
            wrench_map = realizer._last_wrench_map
            if wrench_map is None or wrench_map.shape[1] != nlam:
                return AugmentedBodyAuthority(timestamp, contact_mode,
                                              np.zeros((0, 3)), np.zeros(0), False,
                                              "yaw wrench sensitivity unavailable")
            residual = np.r_[planar_residual,
                             realizer.last_yaw_moment - realizer.last_yaw_moment_target]
            gain = np.vstack((planar_gain,
                              wrench_map[5] @ K[nv + nu:] - np.eye(3)[2]))
        else:
            residual = np.r_[planar_residual,
                             realizer.last_qdd[5] - float(realizer.last_yaw_acc_target)]
            gain = np.vstack((planar_gain, K[5] - np.eye(3)[2]))
        tolerance = np.array([self.realization_tolerance, self.realization_tolerance,
                              float(yaw_tolerance)])
        if np.any(np.abs(residual) > tolerance):
            return AugmentedBodyAuthority(timestamp, contact_mode,
                                          np.zeros((0, 3)), np.zeros(0), False,
                                          "nominal augmented residual exceeds tolerance")
        for i in range(3):
            rows.extend((gain[i], -gain[i]))
            bounds.extend((tolerance[i] - residual[i], tolerance[i] + residual[i]))
        for i in range(3):
            e = np.zeros(3); e[i] = 1.0
            rows.extend((e, -e)); bounds.extend((limit[i], limit[i]))
        H = np.vstack(rows)
        local_h = np.asarray(bounds, dtype=float)
        if np.any(local_h[:critical_row_count] < -1e-6):
            return AugmentedBodyAuthority(timestamp, contact_mode,
                                          np.zeros((0, 3)), np.zeros(0), False,
                                          "nominal point violates a critical-region row")
        if np.any(local_h < -1e-6):
            return AugmentedBodyAuthority(timestamp, contact_mode,
                                          np.zeros((0, 3)), np.zeros(0), False,
                                          "tightened nominal torque/contact margin exhausted")
        local_h = np.maximum(local_h, 0.0)
        # Increment-to-absolute translation: H (u-u_ref) <= h becomes H u <= h+H u_ref.
        h = local_h + H @ u_ref
        return AugmentedBodyAuthority(timestamp, contact_mode, H, h, True, "ok",
                                       command_reference=u_ref.copy())


@dataclass(frozen=True)
class CentroidalBodyAuthority:
    """Local authority for ``[u_x, u_y, M_x, M_y, M_z]``.

    This experimental object deliberately remains separate from the paper's
    two-dimensional body authority and the yaw-only development map.
    """

    timestamp: float
    contact_mode: tuple[str, ...]
    H: np.ndarray
    h: np.ndarray
    valid: bool = True
    status: str = "ok"
    command_reference: np.ndarray = field(default_factory=lambda: np.zeros(5))


class CentroidalBodyAuthorityMapper:
    """Map the current QP cell into planar-CoM plus 3-axis moment authority."""

    def __init__(
        self,
        *,
        torque_margin_fraction: float = 0.02,
        friction_margin_fraction: float = 0.04,
        normal_force_margin_n: float = 1.0,
        realization_tolerance: float = 0.35,
        moment_tolerance_nm: float = 1.0,
    ):
        self.torque_margin_fraction = float(torque_margin_fraction)
        self.friction_margin_fraction = float(friction_margin_fraction)
        self.normal_force_margin_n = float(normal_force_margin_n)
        self.realization_tolerance = float(realization_tolerance)
        self.moment_tolerance_nm = float(moment_tolerance_nm)

    def snapshot(
        self,
        realizer: Any,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        timestamp: float,
        contact_mode: tuple[str, ...],
        command_reference: np.ndarray,
        absolute_limit: np.ndarray = np.array([4.0, 4.0, 40.0, 40.0, 20.0]),
    ) -> CentroidalBodyAuthority:
        u_ref = np.asarray(command_reference, dtype=float).reshape(5)
        limit = np.asarray(absolute_limit, dtype=float).reshape(5)
        if (realizer.last_fallback or realizer._body_dq_du is None
                or realizer._body_dq_du.shape[1] != 5):
            return CentroidalBodyAuthority(timestamp, contact_mode,
                                           np.zeros((0, 5)), np.zeros(0), False,
                                           "five-dimensional sensitivity unavailable")
        K = realizer.input_sensitivity(realizer._body_dq_du)
        if K is None or not np.all(np.isfinite(K)):
            return CentroidalBodyAuthority(timestamp, contact_mode,
                                           np.zeros((0, 5)), np.zeros(0), False,
                                           "KKT sensitivity unavailable")
        nv, nu = realizer.nv, realizer.nu
        z0 = realizer._qp_z
        nlam = realizer.last_contact_force.size
        wrench_map = realizer._last_wrench_map
        if wrench_map is None or wrench_map.shape[1] != nlam:
            return CentroidalBodyAuthority(timestamp, contact_mode,
                                           np.zeros((0, 5)), np.zeros(0), False,
                                           "centroidal wrench map unavailable")
        tau = z0[nv:nv + nu]
        lam = z0[nv + nu:nv + nu + nlam]
        Ktau = K[nv:nv + nu]
        Klam = K[nv + nu:nv + nu + nlam]
        rows: list[np.ndarray] = []
        bounds: list[float] = []
        span = realizer.torque_max - realizer.torque_min
        tau_lo = realizer.torque_min + self.torque_margin_fraction * span
        tau_hi = realizer.torque_max - self.torque_margin_fraction * span
        for j in range(nu):
            rows.extend((Ktau[j], -Ktau[j]))
            bounds.extend((tau_hi[j] - tau[j], tau[j] - tau_lo[j]))
        mu = realizer.mu * (1.0 - self.friction_margin_fraction)
        for c in range(nlam // 3):
            base = 3 * c
            for axis in (0, 1):
                rows.extend((Klam[base + axis] - mu * Klam[base + 2],
                             -Klam[base + axis] - mu * Klam[base + 2]))
                bounds.extend((mu * lam[base + 2] - lam[base + axis],
                               mu * lam[base + 2] + lam[base + axis]))
            rows.append(-Klam[base + 2])
            bounds.append(lam[base + 2] - self.normal_force_margin_n)
        Jcom = np.zeros((3, nv))
        mujoco.mj_jacSubtreeCom(model, data, Jcom, realizer.root_body)
        residual = np.r_[
            Jcom[:2] @ realizer.last_qdd - np.asarray(realizer.last_com_acc_des)[:2],
            realizer.last_centroidal_moment - realizer.last_centroidal_moment_target,
        ]
        gain = np.vstack((
            Jcom[:2] @ K[:nv] - np.eye(5)[:2],
            wrench_map[3:] @ K[nv + nu:] - np.eye(5)[2:],
        ))
        tolerance = np.r_[
            np.full(2, self.realization_tolerance),
            np.full(3, self.moment_tolerance_nm),
        ]
        if np.any(np.abs(residual) > tolerance):
            return CentroidalBodyAuthority(timestamp, contact_mode,
                                           np.zeros((0, 5)), np.zeros(0), False,
                                           "nominal centroidal residual exceeds tolerance")
        for i in range(5):
            rows.extend((gain[i], -gain[i]))
            bounds.extend((tolerance[i] - residual[i], tolerance[i] + residual[i]))
        for i in range(5):
            e = np.zeros(5); e[i] = 1.0
            rows.extend((e, -e)); bounds.extend((limit[i], limit[i]))
        H = np.vstack(rows)
        local_bounds = np.asarray(bounds, dtype=float)
        if np.any(local_bounds < -1e-8):
            return CentroidalBodyAuthority(timestamp, contact_mode,
                                           np.zeros((0, 5)), np.zeros(0), False,
                                           "tightened nominal torque/contact margin exhausted")
        h = np.maximum(local_bounds, 0.0) + H @ u_ref
        return CentroidalBodyAuthority(timestamp, contact_mode, H, h, True, "ok",
                                        command_reference=u_ref.copy())


class AnalyticAuthorityMapper:
    """Build a local authority polytope from the realizer's KKT sensitivity.

    Margins are fractional tightenings that absorb the local-cell approximation,
    inter-sample state motion, and estimation error.  They make the mapping
    conservative but not certified.  The map includes complete critical-region
    conditions, plus selected physical-margin and realization-tolerance rows;
    the instantaneous modeled QP remains the final constraint-enforcement
    layer under its feasibility, model, and contact assumptions.
    """

    def __init__(
        self,
        *,
        torque_margin_fraction: float = 0.02,
        friction_margin_fraction: float = 0.04,
        normal_force_margin_n: float = 1.0,
        absolute_limit: float = 4.0,
        realization_tolerance: float = 0.35,
        per_foot_normal_force_margin: bool = False,
    ):
        self.torque_margin_fraction = float(torque_margin_fraction)
        self.friction_margin_fraction = float(friction_margin_fraction)
        self.normal_force_margin_n = float(normal_force_margin_n)
        self.absolute_limit = float(absolute_limit)
        self.realization_tolerance = float(realization_tolerance)
        # Opt-in only (default False): published Table I/II numbers were
        # generated, and remain byte-identical to publication, with the
        # original per-corner normal-force floor.  See Stage L4 in
        # LOCOMOTION_REPORT_PLAN.md for why the per-foot aggregation exists
        # and where it is enabled.
        self.per_foot_normal_force_margin = bool(per_foot_normal_force_margin)

    def snapshot(
        self,
        realizer: Any,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        timestamp: float,
        contact_mode: tuple[str, ...],
        command_reference: np.ndarray | None = None,
    ) -> RealizationAuthority:
        """Publish a local model in absolute residual-command coordinates.

        The KKT solution is linearized about the command used by the immediately
        preceding realizer solve.  ``command_reference`` is that planar residual
        command.  The local inequalities initially describe an increment
        ``delta_u``; translating them to ``u = command_reference + delta_u`` is
        essential before passing them to an MPC that optimizes absolute commands.
        """
        nv, nu = realizer.nv, realizer.nu
        empty = np.zeros((0, 2))
        u_ref = np.zeros(2) if command_reference is None else np.asarray(
            command_reference, dtype=float
        ).reshape(2)
        if not np.all(np.isfinite(u_ref)):
            raise ValueError("command_reference must be finite")

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

        sensitivity = realizer.input_sensitivity_with_duals(realizer._com_dq_du)
        if sensitivity is None:
            return _invalid("KKT sensitivity unavailable")
        K, dnu, active, at_lower, at_upper = sensitivity
        if not (np.all(np.isfinite(K)) and np.all(np.isfinite(dnu))):
            return _invalid("non-finite KKT sensitivity")

        z0 = realizer._qp_z
        nlam = realizer.last_contact_force.size
        tau_ff = z0[nv:nv + nu]
        lam_ff = z0[nv + nu:nv + nu + nlam]
        K_tau = K[nv:nv + nu]
        K_lam = K[nv + nu:nv + nu + nlam]

        rows: list[np.ndarray] = []
        bounds: list[float] = []

        # --- complete primal/dual critical-region conditions.
        #
        # The affine KKT map is exact only while its active set remains valid.
        # Mapping selected torque/contact margins alone is insufficient: an
        # otherwise-unmapped inactive row can become infeasible, or an active
        # inequality multiplier can change sign.  Include both conditions in
        # the increment coordinates used by the KKT solve:
        #
        #   l_i <= A_i (z0 + K delta_u) <= u_i  for inactive rows,
        #   y_i(delta_u) >= 0                  for active upper rows,
        #   y_i(delta_u) <= 0                  for active lower rows.
        #
        # Equalities are active by construction and impose no multiplier-sign
        # condition.  OSQP's signed dual convention is retained here.
        A_qp, l_qp, u_qp, y_qp = (
            realizer._qp_A, realizer._qp_l, realizer._qp_u, realizer._qp_y,
        )
        if not (
            A_qp is not None and l_qp is not None and u_qp is not None
            and y_qp is not None and A_qp.shape[1] == K.shape[0]
        ):
            return _invalid("QP rows unavailable for critical-region mapping")
        if not (
            np.all(np.isfinite(z0)) and np.all(np.isfinite(A_qp))
            and np.all(np.isfinite(y_qp))
        ):
            return _invalid("non-finite nominal QP data")
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            Az0 = A_qp @ z0
        if not np.all(np.isfinite(Az0)):
            return _invalid("non-finite nominal QP row value")
        equality = np.isclose(l_qp, u_qp)
        inactive = ~active
        for i in np.flatnonzero(inactive):
            row_gain = A_qp[i] @ K
            if np.isfinite(u_qp[i]):
                rows.append(row_gain)
                bounds.append(float(u_qp[i] - Az0[i]))
            if np.isfinite(l_qp[i]):
                rows.append(-row_gain)
                bounds.append(float(Az0[i] - l_qp[i]))
        active_indices = np.flatnonzero(active)
        for local_index, qp_index in enumerate(active_indices):
            if equality[qp_index]:
                continue
            if at_upper[qp_index]:
                # y_i + dy_i/d(delta_u) delta_u >= 0.
                rows.append(-dnu[local_index])
                bounds.append(float(y_qp[qp_index]))
            elif at_lower[qp_index]:
                # y_i + dy_i/d(delta_u) delta_u <= 0.
                rows.append(dnu[local_index])
                bounds.append(float(-y_qp[qp_index]))
            else:
                return _invalid("active inequality has no finite bound side")
        critical_row_count = len(rows)

        # --- actuator torque bounds:  tau_min <= tau_ff + K_tau u <= tau_max
        span = realizer.torque_max - realizer.torque_min
        tau_lo = realizer.torque_min + self.torque_margin_fraction * span
        tau_hi = realizer.torque_max - self.torque_margin_fraction * span
        for j in range(nu):
            rows.append(K_tau[j]);      bounds.append(tau_hi[j] - tau_ff[j])
            rows.append(-K_tau[j]);     bounds.append(tau_ff[j] - tau_lo[j])

        # --- friction pyramid on each contact (still evaluated per corner:
        # tangential slip is a genuine per-contact-point physical limit).
        mu = realizer.mu * (1.0 - self.friction_margin_fraction)
        n_corners = nlam // 3
        for c in range(n_corners):
            b = 3 * c
            fz_ff, Kz = lam_ff[b + 2], K_lam[b + 2]
            for t in (0, 1):
                ft_ff, Kt = lam_ff[b + t], K_lam[b + t]
                #  ft - mu fz <= 0   and   -ft - mu fz <= 0
                rows.append(Kt - mu * Kz);   bounds.append(mu * fz_ff - ft_ff)
                rows.append(-Kt - mu * Kz);  bounds.append(mu * fz_ff + ft_ff)

        # --- unilateral normal-force floor.
        #
        # Default (``per_foot_normal_force_margin=False``, unchanged from
        # publication): the floor is enforced per corner, exactly as
        # published.
        #
        # Opt-in (``per_foot_normal_force_margin=True``): the floor is
        # aggregated PER FOOT instead.  A rigid foot modeled as several
        # discrete corner contacts (see FOOT_CONTACT_OFFSETS) is statically
        # indeterminate: at a real, actively weight-shifting stance one
        # corner routinely carries near-zero load even though the foot as a
        # whole is nowhere close to lifting off.  A per-corner floor
        # over-tightens on exactly that expected, physical load imbalance and
        # can reject an otherwise perfectly good nominal point (see
        # LOCOMOTION_REPORT_PLAN.md Stage L4).  The margin that is physically
        # meaningful is on each foot's total vertical load, so corners are
        # summed per foot before the margin is applied.  Either way, the hard
        # fz >= 0 unilateral bound remains enforced per corner by the QP
        # itself (``lb[base + 2] = 0.0`` in ``_solve_qp``) and is untouched by
        # this option.
        if self.per_foot_normal_force_margin:
            n_feet = len(contact_mode) if contact_mode else 0
            if n_feet and n_corners % n_feet == 0:
                corners_per_foot = n_corners // n_feet
            else:
                # Corner count doesn't split evenly across contact_mode
                # (should not happen for the current contact model): fall
                # back to the per-corner grouping rather than guess.
                corners_per_foot = 1
                n_feet = n_corners
        else:
            corners_per_foot = 1
            n_feet = n_corners
        for foot_index in range(n_feet):
            lo = foot_index * corners_per_foot
            hi = lo + corners_per_foot
            fz_total = sum(lam_ff[3 * c + 2] for c in range(lo, hi))
            Kz_total = sum(K_lam[3 * c + 2] for c in range(lo, hi))
            #  sum(fz) >= fz_min_foot  ->  -Kz_total u <= fz_total - fz_min_foot
            rows.append(-Kz_total)
            bounds.append(fz_total - self.normal_force_margin_n)

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

        # A negative critical-region margin would mean that the reported nominal
        # QP point and its active-set classification are inconsistent at the
        # stated numerical tolerance.  Do not repair that inconsistency by
        # clamping: reject the snapshot and use the caller's conservative
        # fallback instead.
        if np.any(h[:critical_row_count] < -1e-6):
            return _invalid("nominal point violates a critical-region row")

        # Prune rows that CANNOT bind.  A row H_i u <= h_i is unreachable within
        # the command range if h_i exceeds the largest value H_i u can take over
        # ||u||_inf <= absolute_limit, i.e. if h_i > |H_i|_1 * u_max.  Most of the
        # ~100 torque/friction rows are of this kind (huge margins), and keeping
        # them costs real time: the predictor lifts every row over its horizon,
        # so 102 rows x 25 stages = 2550 constraints in its QP.  Pruning is exact
        # -- it removes only rows that are vacuous on the admissible range.
        reach = np.abs(H).sum(axis=1) * self.absolute_limit
        keep = h <= reach + 1e-9
        n_pruned = int((~keep).sum())
        if keep.any():
            H, h = H[keep], h[keep]

        # h_k < 0 handling. A negative mapped physical-margin row means that the
        # nominal point has already consumed the configured tightening. Replacing
        # H_i delta_u <= h_i < 0 with H_i delta_u <= 0 would enlarge the mapped
        # set, so do not publish a relaxed authority polytope. Reject the
        # snapshot and let the controller use its explicitly configured fallback
        # box. Tiny negative values are treated as numerical roundoff only.
        EPS_H = 1e-6
        n_margin_exhausted = int(np.sum(h < -EPS_H))
        if n_margin_exhausted:
            return _invalid(f"tightened physical margin exhausted ({n_margin_exhausted} rows)")
        h = np.maximum(h, 0.0)

        # The KKT rows currently constrain delta_u.  Consumers optimize the
        # absolute residual command u, so H (u - u_ref) <= h must be published
        # as H u <= h + H u_ref.
        h = h + H @ u_ref

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
                status="ok",
            nominal_residual=residual0.copy(),
            residual_gain=residual_gain,
            command_reference=u_ref.copy(),
        )

    def task_authority(
        self,
        realizer: Any,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        u_body: np.ndarray,
        *,
        task_tolerance: float = 0.5,
        body_reference: np.ndarray | None = None,
        task_reference: np.ndarray | None = None,
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
        ub_ref = np.zeros(2) if body_reference is None else np.asarray(
            body_reference, dtype=float
        ).reshape(2)
        ut_ref = np.zeros(3) if task_reference is None else np.asarray(
            task_reference, dtype=float
        ).reshape(3)
        if not (np.all(np.isfinite(ub_ref)) and np.all(np.isfinite(ut_ref))):
            raise ValueError("authority command references must be finite")

        K_tau_b, K_tau_t = j["K_tau"][:, :2], j["K_tau"][:, 2:]
        K_lam_b, K_lam_t = j["K_lam"][:, :2], j["K_lam"][:, 2:]
        # The joint KKT model is centered at (ub_ref, ut_ref).  Commit the
        # requested absolute body command through its increment from that point.
        delta_ub = ub - ub_ref
        tau0 = j["tau_ff"] + K_tau_b @ delta_ub
        lam0 = j["lam_ff"] + K_lam_b @ delta_ub

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
                                 status="nominal task tracking slack exceeds tolerance")
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
        if n_exh:
            z3 = np.zeros((0, 3))
            return TaskAuthority(
                j["timestamp"], j["contact_mode"], z3, np.zeros(0), K_tau_t, K_lam_t,
                valid=False,
                status=f"body allocation exhausts {n_exh} task-margin rows",
            )
        h = np.maximum(h, 0.0)
        # Rows above constrain delta_ut.  Translate to the absolute task command
        # expected by the task MPC.
        h = h + H @ ut_ref
        return TaskAuthority(
            j["timestamp"], j["contact_mode"], H, h, K_tau_t, K_lam_t,
            valid=True,
            status="ok",
            command_reference=ut_ref.copy(),
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
    Contrast with the offline repeated-QP numerical reference, which needs
    roughly 62 QP solves.
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
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            Az = A @ z
        if not np.all(np.isfinite(Az)):
            return 0.0
        at_hi = np.isfinite(hi) & (np.abs(Az - hi) <= self.active_tol)
        at_lo = np.isfinite(lo) & (np.abs(Az - lo) <= self.active_tol)
        active = equality | ((at_hi | at_lo) & (np.abs(y) > self.active_tol))

        t = 0.0
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            r = Jout @ z[:nv] - nom
        if not np.all(np.isfinite(r)):
            return 0.0
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
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                dr = Jout @ dz[:nv] - d
            if not np.all(np.isfinite(dr)):
                return t
            s_tol = np.inf
            for i in range(d.size):
                if abs(dr[i]) > 1e-12:
                    for bound in (tol, -tol):
                        s = (bound - r[i]) / dr[i]
                        if s > 1e-9:
                            s_tol = min(s_tol, s)

            # --- a currently-inactive row ENTERS the active set (vectorized)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                rate = A @ dz
                val = A @ z
            if not (np.all(np.isfinite(rate)) and np.all(np.isfinite(val))):
                return t
            inact = ~active
            s_hi = np.where(inact & (rate > 1e-10) & np.isfinite(hi),
                            (hi - val) / np.where(np.abs(rate) > 1e-12, rate, 1.0), np.inf)
            s_lo = np.where(inact & (rate < -1e-10) & np.isfinite(lo),
                            (lo - val) / np.where(np.abs(rate) > 1e-12, rate, 1.0), np.inf)
            s_row = np.minimum(s_hi, s_lo)
            s_row = np.where(s_row > 1e-9, s_row, np.inf)
            s_enter = float(np.min(s_row)) if s_row.size else np.inf

            # --- an active row LEAVES (its multiplier reaches zero) (vectorized)
            act_idx = np.flatnonzero(active)
            movable = ~equality[act_idx]
            s_act = np.full(act_idx.size, np.inf)
            safe = movable & (np.abs(dnu) > self.dual_tol)
            s_act[safe] = -y[act_idx][safe] / dnu[safe]
            s_act = np.where(s_act > 1e-9, s_act, np.inf)
            s_act_full = np.full(active.shape, np.inf)
            s_act_full[act_idx] = s_act
            s_leave = float(np.min(s_act)) if s_act.size else np.inf

            s_cap = self.absolute_limit - t
            s_star = min(s_tol, s_enter, s_leave, s_cap)
            if not np.isfinite(s_star):
                return min(t + s_cap, self.absolute_limit)

            # residual tolerance reached first -> this is the authority boundary
            if s_star == s_tol:
                return t + s_tol
            if s_star == s_cap:
                return self.absolute_limit

            # advance to the breakpoint and switch the active set.  Apply every
            # row whose event falls within EVENT_TOL of s_star together, not
            # just a single argmin winner -- rows tied at the same critical-
            # region boundary (a symmetric configuration, or two contacts
            # saturating together) are a genuine SIMULTANEOUS transition, and
            # since s_enter/s_leave come from independent formulas ((hi-val)/rate
            # vs -y/dnu) a true tie almost never lands on the same float.
            # Picking only the numerically-smaller one per iteration would
            # silently split one event across iterations and could exhaust
            # max_regions before reaching the true tolerance-crossing boundary.
            EVENT_TOL = 1e-9 + 1e-6 * s_star
            z = z + s_star * dz
            y = y.copy()
            y[act_idx] = y[act_idx] + s_star * dnu
            r = r + s_star * dr
            t += s_star
            active[inact & (s_row <= s_star + EVENT_TOL)] = True
            active[active & (s_act_full <= s_star + EVENT_TOL)] = False
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
                                "nominal task tracking slack exceeds tolerance", 0, 0.0)
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
# Offline repeated-QP numerical reference: residual bisection (NOT in control)
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
    """Offline numerical reference: bisect modeled-QP residuals along rays.

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
                bool(np.all(radius > 0.0)), "repeated-QP residual bisection",
                len(cache), corner_scale,
            )
        finally:
            data.ctrl[:] = saved_ctrl


# Backwards-compatible alias: the old name referred to the bisection estimator.
PlanarBodyAuthorityEstimator = ExactResidualBisectionEstimator


@dataclass(frozen=True)
class BoxAuthority:
    """A plain box ``[lower, upper]``, expressed as a polytope ``H u <= h``.

    Exists so a box constraint (e.g. ``NormalizedMPC.update_input_box``'s
    conservative fixed-box fallback) can be evaluated by
    ``PhysicalRealizabilityPredictor``/``RouteEvaluator`` through the exact
    same code path as a real mapped authority polytope, rather than needing
    special-cased handling for "this candidate has no H/h."
    """

    timestamp: float
    contact_mode: tuple[str, ...]
    H: np.ndarray
    h: np.ndarray
    valid: bool = True
    status: str = "ok"
    command_reference: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @classmethod
    def from_box(
        cls, lower: np.ndarray, upper: np.ndarray, *,
        timestamp: float = 0.0, contact_mode: tuple[str, ...] = (),
    ) -> "BoxAuthority":
        lower = np.asarray(lower, dtype=float).reshape(-1)
        upper = np.asarray(upper, dtype=float).reshape(-1)
        if lower.size != upper.size:
            raise ValueError("lower and upper must have the same size")
        n = lower.size
        H = np.vstack((np.eye(n), -np.eye(n)))
        h = np.concatenate((upper, -lower))
        return cls(timestamp, contact_mode, H, h, valid=True, status="ok",
                   command_reference=np.zeros(n))


# --------------------------------------------------------------------------
# Horizon physical-realizability certificate (predictor only, not a control
# layer -- nothing below this line feeds back into execution on its own)
# --------------------------------------------------------------------------


def _authority_H_h(authority: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return (H, h) from whichever authority-snapshot type is passed.

    The snapshot dataclasses in this module name their polytope fields
    differently (``H_body``/``h_body``, ``H_task``/``h_task``, ``H``/``h``),
    so callers of :class:`PhysicalRealizabilityPredictor` should not have to
    know which one they hold.
    """
    for H_attr, h_attr in (("H_body", "h_body"), ("H_task", "h_task"), ("H", "h")):
        if hasattr(authority, H_attr):
            return getattr(authority, H_attr), getattr(authority, h_attr)
    raise TypeError(f"{type(authority).__name__} exposes no H/h authority polytope")


@dataclass(frozen=True)
class PhysicalRealizabilityCertificate:
    """m_phys(k+j) over an already-predicted command horizon U_{k:k+N}.

    ``horizon_margin[j]`` is the worst-case row slack of the authority
    polytope at the predicted stage-``j`` command ``u_{k+j}``,

        horizon_margin[j] = min_i ( h_i - H_i @ u_{k+j} ),

    and ``min_margin = min_j horizon_margin[j]`` is the scalar certificate
    m_phys(k). Since H/h already fold torque, friction-cone, unilateral
    normal-force, and realization-tolerance rows into one polytope (see
    ``AnalyticAuthorityMapper``), this IS the combined multi-physical-quantity
    margin -- evaluated jointly across constraint types rather than isolated
    per-actuator torque headroom. Recovering a strict per-quantity breakdown
    at every horizon stage would require re-deriving the mapper's own
    torque/friction/normal-force bounds here, a second copy of logic that
    could silently drift from the mapper's -- so this predictor deliberately
    reads the mapper's already-published polytope instead of recomputing it.
    (``RealizationAuthority.torque_margin/friction_margin/normal_margin`` give
    that breakdown, but only at the single nominal point k+0 the snapshot was
    taken at, not across the horizon.)

    THIS IS A LOCAL CERTIFICATE, NOT YET THE PAPER'S FULL PREDICTIVE ONE: H
    and h are frozen over the horizon by
    ``NormalizedMPC.update_input_polytope`` (the same ``(H, h)`` is applied at
    every stage), so ``horizon_margin`` varies only because the predicted
    command ``u_{k+j}`` varies -- not because of any predicted change in
    contact mode, configuration, or environment. A genuine future-authority
    certificate needs stage-varying ``(H_j, h_j)`` built from predicted
    ``(q, qdot, contact)_{k+j}``, which this codebase does not yet compute.
    """

    timestamp: float
    contact_mode: tuple[str, ...]
    horizon_margin: np.ndarray
    min_margin: float
    first_violation_index: int | None
    valid: bool = True
    status: str = "ok"


class PhysicalRealizabilityPredictor:
    """Fold a predicted command horizon through an authority polytope.

    This is "Module A" from the code-to-paper review: it turns the existing
    *instantaneous* authority machinery (``AnalyticAuthorityMapper`` /
    ``ContinuationAuthorityEstimator``) plus the existing *predicted command
    sequence* (``NormalizedMPC.last_u_sequence``, already computed by the
    horizon MPC every solve) into a horizon-wide margin profile, without
    changing either of those two pieces. It adds no new state and performs no
    extra QP or KKT solves -- both inputs already exist by the time a control
    cycle finishes.
    """

    def predict(
        self, authority: Any, u_sequence: np.ndarray,
    ) -> PhysicalRealizabilityCertificate:
        """Compute m_phys(k+j) for the predicted stages in ``u_sequence``.

        ``authority`` is any of ``RealizationAuthority``/``TaskAuthority``/
        ``AugmentedBodyAuthority``/``CentroidalBodyAuthority``. ``u_sequence``
        is the predicted ABSOLUTE command trajectory, shape ``(horizon, dim)``
        or ``(dim,)`` for a single stage -- e.g. ``NormalizedMPC.last_u_sequence``.
        """
        u_seq = np.atleast_2d(np.asarray(u_sequence, dtype=float))
        n_stages = u_seq.shape[0]
        if not authority.valid:
            return PhysicalRealizabilityCertificate(
                timestamp=authority.timestamp, contact_mode=authority.contact_mode,
                horizon_margin=np.full(n_stages, -np.inf), min_margin=-np.inf,
                first_violation_index=0, valid=False,
                status=f"authority invalid: {authority.status}",
            )
        H, h = _authority_H_h(authority)
        if H.size == 0:
            # An empty polytope means "unconstrained": every row-min is
            # vacuously +inf, not a violation.
            return PhysicalRealizabilityCertificate(
                timestamp=authority.timestamp, contact_mode=authority.contact_mode,
                horizon_margin=np.full(n_stages, np.inf), min_margin=np.inf,
                first_violation_index=None, valid=True,
                status="unconstrained authority (no rows)",
            )
        if u_seq.shape[1] != H.shape[1]:
            raise ValueError(
                f"u_sequence has dim {u_seq.shape[1]}, authority polytope expects {H.shape[1]}"
            )
        if not np.all(np.isfinite(u_seq)):
            return PhysicalRealizabilityCertificate(
                timestamp=authority.timestamp, contact_mode=authority.contact_mode,
                horizon_margin=np.full(n_stages, -np.inf), min_margin=-np.inf,
                first_violation_index=0, valid=False,
                status="non-finite predicted command sequence",
            )
        slack = h[None, :] - u_seq @ H.T          # (n_stages, n_rows)
        horizon_margin = np.min(slack, axis=1)
        min_margin = float(np.min(horizon_margin))
        violations = np.flatnonzero(horizon_margin < 0.0)
        first_violation_index = int(violations[0]) if violations.size else None
        return PhysicalRealizabilityCertificate(
            timestamp=authority.timestamp, contact_mode=authority.contact_mode,
            horizon_margin=horizon_margin, min_margin=min_margin,
            first_violation_index=first_violation_index,
            valid=True, status="ok",
        )


# --------------------------------------------------------------------------
# Response classifier (Module B).  Classifies a certificate into a response
# level; it does NOT execute retiming, reshaping, or rerouting -- this
# codebase has no such executors (see the module-level review notes above).
# --------------------------------------------------------------------------


class ResponseLevel(IntEnum):
    """The five-tier response hierarchy from the code-to-paper review.

    Only EXECUTE and FALLBACK correspond to an action this codebase actually
    performs today: EXECUTE is "use the authority-constrained MPC solve as
    given," and FALLBACK is "the solve found no feasible point in H u <= h at
    all, so NormalizedMPC already fell back to u_ref" (see
    ``NormalizedMPC.solve``'s ``_admissible_fallback``). RETIME, RESHAPE, and
    REROUTE are RECOMMENDATIONS only -- there is no trajectory-retiming,
    command-reshaping, or route-rerouting executor in this codebase to carry
    them out (confirmed absent by inspection: no retiming/reshaping module,
    and Level 3 rerouting is the review's own "largest missing component").
    """

    EXECUTE = 0
    RETIME = 1
    RESHAPE = 2
    REROUTE = 3
    FALLBACK = 4


@dataclass(frozen=True)
class PhysicalFailureResponse:
    level: ResponseLevel
    reason: str
    margin_now: float             # horizon_margin[0]: the nearest-term margin
    margin_worst: float           # cert.min_margin: the horizon-wide certificate
    first_violation_index: int | None
    violation_fraction: float     # fraction of predicted stages below margin_safe


class PhysicalFailureManager:
    """Module B: classify a :class:`PhysicalRealizabilityCertificate` into a
    :class:`ResponseLevel`, using only signals that already exist by the time
    a control cycle finishes (Module A's certificate, plus
    ``NormalizedMPC.last_polytope_failed``).

    The classification below the EXECUTE/FALLBACK boundary is a fixed
    HEURISTIC over the shape of the predicted margin profile, not a verified
    outcome -- there is no retime/reshape/reroute engine here to test the
    recommendation against (see :class:`ResponseLevel`). The heuristic:

    * FALLBACK  -- the authority snapshot is invalid, or the realization-
      constrained solve already found no feasible command at all
      (``last_polytope_failed``). This is the one tier backed by a real
      "nothing worked" signal, not a shape read off the margin profile.
    * EXECUTE   -- the certificate's worst-case margin over the WHOLE horizon
      already clears ``margin_safe``. Nothing needs to change.
    * RETIME    -- the near-term margin (stage 0) clears ``margin_safe`` but
      some later stage does not: there is time before the shortfall arrives,
      which is exactly the lever retiming (slowing down) trades on.
    * REROUTE   -- the near-term margin is already below ``margin_safe`` AND
      the violation is pervasive (more than ``reroute_violation_fraction`` of
      the predicted horizon is below ``margin_safe``): not a local, transient
      dip, so no adaptation within the CURRENT authority polytope is likely to
      restore feasibility.
    * RESHAPE   -- the near-term margin is already below ``margin_safe`` but
      the violation is not pervasive: retiming can't buy time (the shortfall
      is already here), but it also is not structural enough to call for a
      different route.
    """

    def __init__(
        self, *, margin_safe: float, reroute_violation_fraction: float = 0.5,
    ):
        if margin_safe < 0.0:
            raise ValueError("margin_safe must be nonnegative")
        if not (0.0 < reroute_violation_fraction <= 1.0):
            raise ValueError("reroute_violation_fraction must be in (0, 1]")
        self.margin_safe = float(margin_safe)
        self.reroute_violation_fraction = float(reroute_violation_fraction)

    def classify(
        self, cert: PhysicalRealizabilityCertificate, *,
        last_polytope_failed: bool = False,
    ) -> PhysicalFailureResponse:
        n = cert.horizon_margin.size
        violation_fraction = (
            float(np.mean(cert.horizon_margin < self.margin_safe)) if n else 1.0
        )
        margin_now = float(cert.horizon_margin[0]) if n else cert.min_margin

        if not cert.valid or last_polytope_failed:
            reason = (
                f"authority invalid ({cert.status})" if not cert.valid
                else "realization-constrained solve found no feasible command "
                     "(last_polytope_failed)"
            )
            return PhysicalFailureResponse(
                ResponseLevel.FALLBACK, reason, margin_now, cert.min_margin,
                cert.first_violation_index, violation_fraction,
            )

        if cert.min_margin >= self.margin_safe:
            return PhysicalFailureResponse(
                ResponseLevel.EXECUTE,
                "worst-case predicted margin clears margin_safe across the horizon",
                margin_now, cert.min_margin, cert.first_violation_index,
                violation_fraction,
            )

        if margin_now >= self.margin_safe:
            return PhysicalFailureResponse(
                ResponseLevel.RETIME,
                "near-term margin is safe; the shortfall only appears later in "
                "the predicted horizon, so there is time to slow down",
                margin_now, cert.min_margin, cert.first_violation_index,
                violation_fraction,
            )

        if violation_fraction > self.reroute_violation_fraction:
            return PhysicalFailureResponse(
                ResponseLevel.REROUTE,
                f"near-term margin already violated and {violation_fraction:.0%} "
                "of the predicted horizon is below margin_safe -- pervasive, "
                "not a local transient dip",
                margin_now, cert.min_margin, cert.first_violation_index,
                violation_fraction,
            )

        return PhysicalFailureResponse(
            ResponseLevel.RESHAPE,
            "near-term margin already violated, but the shortfall is not "
            "pervasive across the predicted horizon",
            margin_now, cert.min_margin, cert.first_violation_index,
            violation_fraction,
        )


# --------------------------------------------------------------------------
# Route evaluator (Module C).  Selects among ALREADY-GENERATED candidate
# routes; it does not generate candidates itself -- this codebase has no
# route/plan generator (confirmed absent by inspection, same as Module B's
# missing retime/reshape/reroute executors). A caller supplies each candidate
# already reduced to what Module A needs (an authority snapshot and a
# predicted command sequence) plus its own planning cost J(P_i).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteCandidate:
    """One candidate route/plan, already reduced to what Module A needs.

    ``authority`` and ``u_sequence`` are exactly :meth:`PhysicalRealizability
    Predictor.predict`'s two arguments for THIS candidate -- e.g. a body/task
    authority snapshot taken (or predicted) along the candidate route, and the
    predicted command horizon a canonical MPC would issue while following it.
    ``cost`` is the candidate's planning cost J(P_i); lower is better. It is
    supplied by the caller -- this module has no notion of trajectory time,
    path length, or any other planning objective.
    """

    name: str
    authority: Any
    u_sequence: np.ndarray
    cost: float
    last_polytope_failed: bool = False


@dataclass(frozen=True)
class RouteEvaluation:
    name: str
    certificate: PhysicalRealizabilityCertificate
    cost: float
    feasible: bool          # M(P_i) = certificate.min_margin >= margin_safe


@dataclass(frozen=True)
class RouteSelection:
    """P* = argmin_P J(P) subject to M(P) >= margin_safe, or no feasible P."""

    selected: str | None
    evaluations: tuple[RouteEvaluation, ...]
    reason: str
    best_infeasible: str | None = None   # highest-M(P) candidate, when selected is None


class RouteEvaluator:
    """Module C: evaluate candidate routes and select
    ``P* = argmin_P J(P)`` subject to ``M(P) >= margin_safe``, where
    ``M(P_i) = min_j m_phys^(i)(k+j)`` is Module A's certificate for that
    candidate. This is the review's own formula, unmodified -- the part of it
    that this module does NOT do is generate the candidates ``P_i`` in the
    first place, since this codebase has no route/plan generator.

    If no candidate reaches ``margin_safe``, ``selected`` is ``None`` rather
    than silently returning the least-bad infeasible candidate as if it were
    safe -- the same "the fallback must itself be admissible" principle
    ``NormalizedMPC.solve`` already follows for its own single-route fallback.
    A caller integrating this with Module B would treat "no feasible route"
    the same way Module B treats "no feasible continuation": as FALLBACK.
    """

    def __init__(
        self, *, margin_safe: float,
        predictor: PhysicalRealizabilityPredictor | None = None,
    ):
        if margin_safe < 0.0:
            raise ValueError("margin_safe must be nonnegative")
        self.margin_safe = float(margin_safe)
        self.predictor = predictor or PhysicalRealizabilityPredictor()

    def evaluate(self, candidates: list[RouteCandidate]) -> RouteSelection:
        if not candidates:
            raise ValueError("candidates must be non-empty")
        evaluations = []
        for c in candidates:
            cert = self.predictor.predict(c.authority, c.u_sequence)
            feasible = (
                cert.valid and not c.last_polytope_failed
                and cert.min_margin >= self.margin_safe
            )
            evaluations.append(RouteEvaluation(c.name, cert, c.cost, feasible))

        feasible_evals = [e for e in evaluations if e.feasible]
        if feasible_evals:
            best = min(feasible_evals, key=lambda e: e.cost)
            reason = (
                f"selected '{best.name}': lowest cost ({best.cost:g}) among "
                f"{len(feasible_evals)}/{len(evaluations)} candidates with "
                f"M(P) >= margin_safe ({self.margin_safe:g})"
            )
            return RouteSelection(best.name, tuple(evaluations), reason)

        worst_case_best = max(evaluations, key=lambda e: e.certificate.min_margin)
        reason = (
            f"no candidate reaches margin_safe ({self.margin_safe:g}); best "
            f"worst-case margin was {worst_case_best.certificate.min_margin:g} "
            f"('{worst_case_best.name}')"
        )
        return RouteSelection(None, tuple(evaluations), reason, worst_case_best.name)
