#!/usr/bin/env python3
"""Torque-level Unitree G1 benchmark for the v3 interaction realizer.

This is the first non-root-assisted v3 benchmark.  It converts the local
Menagerie-derived G1 MJCF from position actuators to torque motors at runtime,
uses the normalized body/task interaction MPCs as command generators, and
realizes those requests through a present-time inverse-dynamics/contact QP:

   min ||J_t qdd - xdd_task_des||^2 + ||J_c qdd - xdd_contact_des||^2
   s.t. M qdd + h = S^T tau + J_c^T lambda, torque bounds, friction bounds

The controller is intentionally logged as an initial torque-level realizer, not
as a finished dynamic walking stack.  Failed and fallen trials remain in the
reported denominator.
"""

from  __future__ import annotations

import argparse
import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import osqp
import scipy.sparse as sp

from normalized_mpc import NormalizedMPC, RandomWalkDisturbanceObserver
from run_g1_root_assist_demo import (
    ACTUATED_JOINT_NAMES,
    G1CommandLayer,
    G1_STAND_CTRL,
    LocalTrajectory,
    robot_com,
    roll_pitch_yaw_from_body,
    site_jac,
    trapezoid_profile,
)


HERE = Path(__file__).resolve().parent
MODEL_POSITION = HERE / "models" / "g1_wbc.xml"
MODEL_TORQUE = HERE / "models" / "g1_wbc_torque.xml"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

SIM_DT = 0.001
COMMAND_DT = 0.002
G = np.array([0.0, 0.0, 9.81])
TORQUE_STAND_CTRL = G1_STAND_CTRL.copy()
TORQUE_STAND_CTRL[:6] = np.array([-0.12, 0.03, 0.0, 0.28, -0.14, -0.03])
TORQUE_STAND_CTRL[6:12] = np.array([-0.12, -0.03, 0.0, 0.28, -0.14, 0.03])
FOOT_CONTACT_OFFSETS = (
    np.array([-0.05, 0.025, -0.03]),
    np.array([-0.05, -0.025, -0.03]),
    np.array([0.12, 0.03, -0.03]),
    np.array([0.12, -0.03, -0.03]),
)
USE_VIRTUAL_FOOT_POLYGON = True
POSTURE_RECOVERY_GAIN = 1.0


@dataclass
class PushSpec:
    start: float
    duration: float
    force: np.ndarray


def generate_torque_model(src: Path = MODEL_POSITION, dst: Path = MODEL_TORQUE) -> Path:
    """Create a torque-actuated MJCF variant from the local position model."""
    tree = ET.parse(src)
    root = tree.getroot()
    actuator = root.find("actuator")
    if actuator is None:
        raise RuntimeError(f"{src} has no <actuator> block")
    actuator.clear()

    # Joint force ranges are present in the local Menagerie-derived MJCF.
    joint_range: dict[str, tuple[float, float]] = {}
    for joint in root.iter("joint"):
        name = joint.attrib.get("name")
        if not name:
            continue
        fr = joint.attrib.get("actuatorfrcrange")
        if fr:
            lo, hi = (float(x) for x in fr.split())
        else:
            lo, hi = -30.0, 30.0
        joint_range[name] = (lo, hi)

    for joint_name in ACTUATED_JOINT_NAMES:
        lo, hi = joint_range[joint_name]
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": joint_name,
                "joint": joint_name,
                "gear": "1",
                "ctrllimited": "true",
                "ctrlrange": f"{lo:g} {hi:g}",
            },
        )

    # The position-control visualization uses the Menagerie default foot
    # spheres.  The torque-level benchmark needs an explicit rubber-sole
    # contact model so the physical realizer can produce support-polygon
    # moments instead of sliding on nearly point-like contacts.
    for geom in root.iter("geom"):
        if geom.attrib.get("class") == "foot":
            geom.set("friction", "1.0 0.02 0.001")
            geom.set("condim", "6")

    tmp = dst.with_name(f"{dst.stem}.{os.getpid()}.tmp.xml")
    tree.write(tmp, encoding="unicode")
    tmp.replace(dst)
    return dst

def body_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

def site_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)

def joint_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

def hand_state(model: mujoco.MjModel, data: mujoco.MjData, sid: int):
    jp = site_jac(model, data, sid)
    return data.site_xpos[sid].copy(), jp @ data.qvel, jp



def measured_foot_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    out = np.zeros(2, dtype=bool)
    for i in range(data.ncon):
        con = data.contact[i]
        body_names = []
        geom_names = []
        for gid in (con.geom1, con.geom2):
            geom_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")
            bid = model.geom_bodyid[gid]
            body_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or "")
        left = any("left_ankle" in n or "left_foot" in n for n in body_names)
        right = any("right_ankle" in n or "right_foot" in n for n in body_names)
        floor = any(n == "floor" or n.startswith("terrain_") for n in geom_names)
        if floor and left:
            out[0] = True
        if floor and right:
            out[1] = True
    return out

def friction_margin(model: mujoco.MjModel, data: mujoco.MjData, mu: float = 0.9) -> float:
    """Return min(mu fz - ||ft||) across active foot-floor contacts.

    The QP friction constraints apply to the stance-foot support model. MuJoCo
    may also report incidental self/contact-pair forces, so the logged margin
    should use the same foot-floor scope as the support detector.
    """
    margins = []
    wrench = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        body_names = []
        geom_names = []
        for gid in (con.geom1, con.geom2):
            geom_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")
            bid = model.geom_bodyid[gid]
            body_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or "")
        floor = any(name == "floor" or name.startswith("terrain_") for name in geom_names)
        foot = any("ankle" in name or "foot" in name for name in body_names)
        if not (floor and foot):
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        normal = abs(float(wrench[0]))
        tang = float(np.linalg.norm(wrench[1:3]))
        margins.append(mu * normal - tang)
    return float(min(margins)) if margins else float("inf")

def com_velocity(model: mujoco.MjModel, data: mujoco.MjData, root_body: int) -> np.ndarray:
    """Whole-robot CoM velocity from the subtree CoM Jacobian."""
    jac = np.zeros((3, model.nv))
    mujoco.mj_jacSubtreeCom(model, data, jac, root_body)
    return jac @ data.qvel

def smooth_forward_reference(t: float, duration: float, distance: float):
    x, xd, xdd = trapezoid_profile(t, duration=duration, distance=distance)
    return (
        np.array([x, 0.0]),
        np.array([xd, 0.0]),
        np.array([xdd, 0.0]),
    )

def scheduled_stance_feet(t: float, scenario: str) -> tuple[str, ...]:
    if scenario in ("stand", "stand_push"):
        return ("left", "right")
    cycle = (0.78 * t) % 1.0
    left_swing = cycle < 0.5
    phase = cycle * 2.0 if left_swing else (cycle - 0.5) * 2.0
    # Short double-support window around lift-off/touch-down improves numerical
    # conditioning without turning the layer into a horizon controller.
    if phase < 0.10 or phase > 0.90:
        return ("left", "right")
    return ("right",) if left_swing else ("left",)

class WholeBodyTorqueRealizer:
    """Present-sample torque projection used as the Level-2 realizer."""

    def __init__(self, model: mujoco.MjModel):
        self.qadr = np.array([model.jnt_qposadr[joint_id(model, n)] for n in ACTUATED_JOINT_NAMES])
        self.dof = np.array([model.jnt_dofadr[joint_id(model, n)] for n in ACTUATED_JOINT_NAMES])
        self.ctrl_id = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTUATED_JOINT_NAMES])
        self.torque_min = model.actuator_ctrlrange[self.ctrl_id, 0].copy()
        self.torque_max = model.actuator_ctrlrange[self.ctrl_id, 1].copy()

        self.kp = np.array(
            [220, 160, 100, 220, 45, 35] * 2
            + [65, 45, 45]
            + [35, 30, 25, 28, 8, 6, 5] * 2,
            dtype=float,
        )
        self.kd = np.array(
            [9.0, 7.0, 4.5, 9.0, 2.2, 1.8] * 2
            + [3.0, 2.5, 2.5]
            + [1.5, 1.3, 1.0, 1.2, 0.25, 0.20, 0.18] * 2,
            dtype=float,
        )

    def joint_state(self, data: mujoco.MjData):
        return data.qpos[self.qadr].copy(), data.qvel[self.dof].copy()

    def command(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        q_ref: np.ndarray,
        qd_ref: np.ndarray,
        task_wrench: np.ndarray,
        hand_jac: np.ndarray,
    ):
        q, qd = self.joint_state(data)
        tau = data.qfrc_bias[self.dof].copy()
        tau += self.kp * (q_ref - q) + self.kd * (qd_ref - qd)
        tau_task_full = hand_jac.T @ task_wrench
        tau += tau_task_full[self.dof]

        tau_unsat = tau.copy()
        tau = np.clip(tau, self.torque_min, self.torque_max)
        data.ctrl[self.ctrl_id] = tau
        saturation = np.maximum(tau_unsat - self.torque_max, 0.0) + np.maximum(self.torque_min - tau_unsat, 0.0)
        return tau, tau_unsat, saturation

class InverseDynamicsQPRealizer:
    """Instantaneous inverse-dynamics/contact projection for Level 2.

    The prediction layers request body and task accelerations. This class does
    not predict future robot dynamics; it solves one convex present-sample QP
    over generalized acceleration, actuator torque, and stance contact forces.
    """

    def __init__(self, model: mujoco.MjModel, *, exact_realizer: bool = False):
        self.nv = model.nv
        self.nu = model.nu
        self.qadr = np.array([model.jnt_qposadr[joint_id(model, n)] for n in ACTUATED_JOINT_NAMES])
        self.dof = np.array([model.jnt_dofadr[joint_id(model, n)] for n in ACTUATED_JOINT_NAMES])
        self.ctrl_id = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTUATED_JOINT_NAMES])
        self.torque_min = model.actuator_ctrlrange[self.ctrl_id, 0].copy()
        self.torque_max = model.actuator_ctrlrange[self.ctrl_id, 1].copy()
        self.Btau = np.zeros((self.nv, self.nu))
        for col, dof in enumerate(self.dof):
            self.Btau[dof, col] = 1.0
        self.root_body = body_id(model, "pelvis")   # subtree root for whole-body CoM Jacobian
        self.foot_body = {
            "left": body_id(model, "left_ankle_roll_link"),
            "right": body_id(model, "right_ankle_roll_link"),
        }
        self.foot_site = {
            "left": site_id(model, "left_foot"),
            "right": site_id(model, "right_foot"),
        }

        self.joint_kp = np.array(
            [65, 55, 35, 70, 20, 16] * 2
            + [28, 20, 20]
            + [18, 16, 12, 14, 5, 4, 3] * 2,
            dtype=float,
        )
        self.joint_kd = np.array(
            [9.0, 7.0, 4.5, 8.0, 2.0, 1.8] * 2
            + [3.0, 2.5, 2.5]
            + [1.4, 1.2, 0.9, 1.0, 0.20, 0.18, 0.15] * 2,
            dtype=float,
        )
        self.mu = 0.9
        self.fz_max = 900.0
        # Hand-task objective weight.  This is a SOFT approximation of the hard
        # task row of Eq. (22) (J_t qdd + Jdot qdot = x_ddot_td + u_t + s_t).
        # The default below is the BODY-ONLY value: the hand task is inert in the
        # balance/transition experiments, and a large weight there destabilises
        # the support transition.  A closed-loop TASK PORT needs it ~1000x larger
        # (see run_multirate_benchmarks.e6_*): the contact-consistent task inertia
        # spans 30x across the hand axes (Lambda_t ~0.4 kg in x, ~12.5 kg in z),
        # so a small scalar weight starves the heavy vertical axis -- at w=6 the
        # nominal hand residual is 1.8 m/s^2 and the task authority set is EMPTY.
        # At w=8e3 the residual is <0.3 m/s^2 on every axis and the body port is
        # essentially unaffected.  The weight is therefore a per-experiment design
        # parameter, and that dependence is itself a reported result.
        self.task_weight = 6.0
        self.task_acc_clip = 18.0     # hand acceleration clip [m/s^2]
        self.wrench_weight = 80.0     # Eq. (22) body-slack priority
        self.com_task_weight = 40.0
        # When False, the friction-pyramid and joint-torque limits are dropped
        # from the recovery QP (unconstrained recovery, for the H5 ablation).
        self.constraints_on = True
        # In exact_realizer mode the implementation follows (22): the body
        # request is a six-dimensional centroidal wrench, virtual stance-point
        # accelerations are equality constraints, and one-step joint-position
        # limits are imposed.  The wrench/task slacks are eliminated from the
        # QP (their squared norms are the corresponding least-squares terms)
        # but reconstructed and logged after every solve.
        self.exact_realizer = exact_realizer
        self.enforce_one_step_joint_limits = bool(exact_realizer)
        self.last_status = "not_solved"
        self.last_eq_residual = np.inf          # rigid-body dynamics-equality residual
        self.last_body_acc_residual = np.inf    # ||Jcom qdd - com_acc_des|| (realized - requested)
        # Norm of the QP hand-acceleration tracking slack, not a pure
        # force-based actuation realization residual.  External task dynamics
        # must be separated explicitly before interpreting this quantity as an
        # actuation deficit.
        self.last_task_acc_tracking_slack_norm = np.inf
        # Deprecated compatibility alias for the task acceleration-tracking
        # slack norm above; this is not a pure actuation residual.
        self.last_task_acc_residual = np.inf
        self.last_contact_force = np.zeros(0)
        self.last_wrench_slack = np.full(6, np.inf)
        self.last_task_slack = np.full(3, np.inf)
        self.last_contact_acc_residual = np.inf
        self.last_fallback = False
        self.last_qdd = np.full(self.nv, np.nan)
        self.last_com_bias_acc = np.zeros(3)
        self.last_tau_qp = np.full(self.nu, np.nan)
        self.last_qdd_lower = np.full(self.nv, -120.0)
        self.last_qdd_upper = np.full(self.nv, 120.0)
        # Retained QP data for the exact input-sensitivity (dz/du) KKT solve.
        self._qp_P = None
        self._qp_A = None
        self._qp_l = None
        self._qp_u = None
        self._qp_z = None
        self._qp_y = None
        self._com_dq_du = None   # d(linear cost)/d(u_c); set when a CoM objective is active
        self._com_clipped = False
        self._qp_pattern = None   # cached OSQP solver + sparsity pattern for the warm 1 kHz path
        self.last_com_acc_des = np.zeros(3)   # last requested CoM acceleration (c_ddot_d + u_c)
        self.last_task_acc_des = np.zeros(3)  # last requested hand acceleration (x_ddot_td + u_t)
        self.last_hand_jac = None             # hand Jacobian of the last solve
        # Local affine map used by the task MPC: tau_QP ~= offset + map @ u,
        # where u = [CoM xyz acceleration; roll/pitch acceleration].
        self.last_mpc_torque_map = None
        self.last_mpc_torque_offset = None
        self.last_mpc_qdd_map = None
        self.last_mpc_qdd_offset = None
        self._task_dq_du = None
        self._task_clipped = False
        self._qp_row_labels = np.zeros(0, dtype="<U32")
        self.last_active_constraint_counts: dict[str, int] = {}
        self.last_yaw_acc_target = 0.0
        self.last_yaw_moment_target = 0.0
        self.last_yaw_moment = 0.0
        self.last_yaw_output_kind = "acceleration"
        self._last_wrench_map = None
        self._moment_dq_du = None
        self.last_centroidal_moment_target = np.zeros(3)
        self.last_centroidal_moment = np.zeros(3)

    def joint_state(self, data: mujoco.MjData):
        return data.qpos[self.qadr].copy(), data.qvel[self.dof].copy()

    def _mass_matrix(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        M = np.zeros((model.nv, model.nv))
        # MuJoCo 3.3+ exposes ``mj_fullM(model, dst, data.qM)``; older Python
        # bindings accepted ``(model, data, dst)``. Support both so the
        # torque-level authority examples are runnable across supported wheels.
        try:
            mujoco.mj_fullM(model, M, data.qM)
        except TypeError:
            mujoco.mj_fullM(model, data, M)
        return M

    def contact_points(self, model: mujoco.MjModel, data: mujoco.MjData, stance_feet: tuple[str, ...]):
        points = {}
        for foot in stance_feet:
            if not USE_VIRTUAL_FOOT_POLYGON:
                sid = self.foot_site[foot]
                points[foot] = (data.site_xpos[sid].copy(), site_jac(model, data, sid))
                continue
            bid = self.foot_body[foot]
            R = data.xmat[bid].reshape(3, 3)
            for i, offset in enumerate(FOOT_CONTACT_OFFSETS):
                key = f"{foot}_{i}"
                pos = data.xpos[bid] + R @ offset
                jp = np.zeros((3, model.nv))
                jr = np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jp, jr, pos, bid)
                points[key] = (pos.copy(), jp)
        return points

    def _add_ls(self, P: np.ndarray, q: np.ndarray, C: np.ndarray, target: np.ndarray, weight: float):
        if C.size == 0:
            return
        # Some macOS BLAS builds emit spurious floating-point warnings for
        # finite dense products.  Check the operands and results explicitly so
        # genuine non-finite QP data remains a hard failure rather than relying
        # on warning behaviour that differs across numerical backends.
        if not (
            np.all(np.isfinite(C))
            and np.all(np.isfinite(target))
            and np.isfinite(weight)
        ):
            raise FloatingPointError("non-finite least-squares QP term")
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            p_add = weight * (C.T @ C)
            q_add = -weight * (C.T @ target)
        if not (np.all(np.isfinite(p_add)) and np.all(np.isfinite(q_add))):
            raise FloatingPointError("least-squares QP product overflow")
        P += p_add
        q += q_add

    def _solve_qp(
        self,
        M: np.ndarray,
        bias: np.ndarray,
        Jc: np.ndarray,
        P: np.ndarray,
        q: np.ndarray,
        nlam: int,
        Aeq_extra: np.ndarray | None = None,
        beq_extra: np.ndarray | None = None,
        qdd_lb: np.ndarray | None = None,
        qdd_ub: np.ndarray | None = None,
        lam_z_ub: np.ndarray | None = None,
    ):
        n = self.nv + self.nu + nlam
        A_dyn = np.zeros((self.nv, n))
        A_dyn[:, :self.nv] = M
        A_dyn[:, self.nv:self.nv + self.nu] = -self.Btau
        if nlam:
            A_dyn[:, self.nv + self.nu:] = -Jc.T
        l_dyn = -bias
        u_dyn = -bias

        rows = [sp.csc_matrix(A_dyn)]
        lows = [l_dyn]
        ups = [u_dyn]
        row_labels = [np.full(self.nv, "dynamics", dtype="<U32")]

        if Aeq_extra is not None and len(Aeq_extra):
            rows.append(sp.csc_matrix(Aeq_extra))
            lows.append(beq_extra)
            ups.append(beq_extra)
            row_labels.append(np.full(len(Aeq_extra), "rigid_contact", dtype="<U32"))

        if nlam and self.constraints_on:
            A_fric = []
            for i in range(nlam // 3):
                base = self.nv + self.nu + 3 * i
                for fx_sign, fy_sign in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
                    row = np.zeros(n)
                    row[base] = fx_sign
                    row[base + 1] = fy_sign
                    row[base + 2] = -self.mu
                    A_fric.append(row)
            rows.append(sp.csc_matrix(np.vstack(A_fric)))
            lows.append(-np.inf * np.ones(len(A_fric)))
            ups.append(np.zeros(len(A_fric)))
            row_labels.append(np.full(len(A_fric), "friction", dtype="<U32"))

        lb = np.full(n, -np.inf)
        ub = np.full(n, np.inf)
        lb[:self.nv] = -120.0
        ub[:self.nv] = 120.0
        if qdd_lb is not None:
            lb[:self.nv] = np.maximum(lb[:self.nv], qdd_lb)
        if qdd_ub is not None:
            ub[:self.nv] = np.minimum(ub[:self.nv], qdd_ub)
        if self.constraints_on:
            lb[self.nv:self.nv + self.nu] = self.torque_min
            ub[self.nv:self.nv + self.nu] = self.torque_max
        for i in range(nlam // 3):
            base = self.nv + self.nu + 3 * i
            lb[base:base + 2] = -self.fz_max
            ub[base:base + 2] = self.fz_max
            lb[base + 2] = 0.0
            ub[base + 2] = self.fz_max if lam_z_ub is None else float(lam_z_ub[i])
        rows.append(sp.eye(n, format="csc"))
        lows.append(lb)
        ups.append(ub)
        variable_labels = np.full(n, "acceleration_bound", dtype="<U32")
        variable_labels[self.nv:self.nv + self.nu] = "torque_bound"
        for i in range(nlam // 3):
            base = self.nv + self.nu + 3 * i
            variable_labels[base:base + 2] = "tangential_force_bound"
            variable_labels[base + 2] = "normal_force_bound"
        row_labels.append(variable_labels)

        A = sp.vstack(rows, format="csc")
        l = np.concatenate(lows)
        u = np.concatenate(ups)
        # Retain the exact QP data so the input sensitivity dz/du can be taken
        # from the active-set KKT system instead of by re-solving the QP.
        Psym = 0.5 * (P + P.T)
        Adense = A.toarray()
        self._qp_P = Psym
        self._qp_A = Adense
        self._qp_l = l
        self._qp_u = u
        self._qp_row_labels = np.concatenate(row_labels)

        # 1 kHz path: set the solver up once per sparsity pattern and thereafter
        # only push new values.  OSQP's update() requires an unchanged pattern,
        # so the pattern itself is the cache key: if it ever changes (different
        # contact mode, different active structure) we fall back to a fresh
        # setup rather than corrupting the factorization.
        pat = self._qp_pattern
        reuse = (
            pat is not None
            and pat["n"] == n
            and pat["m"] == Adense.shape[0]
            # A structurally-new nonzero outside the cached pattern would be
            # silently dropped, so verify the pattern still covers the data.
            and int(np.count_nonzero(np.triu(Psym))) <= pat["pnnz"]
            and int(np.count_nonzero(Adense)) <= pat["annz"]
        )
        if reuse:
            Px = Psym[pat["pr"], pat["pc"]]
            Ax = Adense[pat["ar"], pat["ac"]]
            solver = pat["solver"]
            solver.update(Px=Px, Ax=Ax, q=q, l=l, u=u)
        else:
            Pcsc = sp.triu(sp.csc_matrix(Psym), format="csc")
            Acsc = sp.csc_matrix(Adense)
            solver = osqp.OSQP()
            solver.setup(
                P=Pcsc, q=q, A=Acsc, l=l, u=u,
                verbose=False, polish=True,
                # The authority estimators identify the active set from this
                # solve's duals with a 1e-6 bound test.  At eps=1e-4 that test
                # sits BELOW the solver's own noise floor, and the identified
                # active set flips with the warm start (2 vs 14 weakly-active
                # rows on the same problem).  1e-8 makes it warm-start
                # independent at no measurable cost (QP ~2.1 ms either way).
                eps_abs=1e-8, eps_rel=1e-8, max_iter=20000, adaptive_rho=True,
            )
            pr_ = Pcsc.indices
            pc_ = np.repeat(np.arange(n), np.diff(Pcsc.indptr))
            ar_ = Acsc.indices
            ac_ = np.repeat(np.arange(n), np.diff(Acsc.indptr))
            self._qp_pattern = {
                "n": n, "m": Adense.shape[0], "solver": solver,
                "pr": pr_, "pc": pc_, "ar": ar_, "ac": ac_,
                "pnnz": int(Pcsc.nnz), "annz": int(Acsc.nnz),
            }
        result = solver.solve()
        if result.x is None or result.info.status_val not in (1, 2):
            # A contact transition can make a cached OSQP factorization reject
            # one otherwise feasible update.  Rebuild once from the current
            # dense matrices before exposing a torque-level fallback; this is a
            # numerical recovery, not an unconstrained second control solve.
            Pcsc = sp.triu(sp.csc_matrix(Psym), format="csc")
            Acsc = sp.csc_matrix(Adense)
            retry_solver = osqp.OSQP()
            retry_solver.setup(
                P=Pcsc, q=q, A=Acsc, l=l, u=u,
                verbose=False, polish=True,
                eps_abs=1e-8, eps_rel=1e-8, max_iter=20000, adaptive_rho=True,
            )
            result = retry_solver.solve()
            if result.x is not None and result.info.status_val in (1, 2):
                self._qp_pattern = {
                    "n": n, "m": Adense.shape[0], "solver": retry_solver,
                    "pr": Pcsc.indices,
                    "pc": np.repeat(np.arange(n), np.diff(Pcsc.indptr)),
                    "ar": Acsc.indices,
                    "ac": np.repeat(np.arange(n), np.diff(Acsc.indptr)),
                    "pnnz": int(Pcsc.nnz), "annz": int(Acsc.nnz),
                }
        self._qp_z = None if result.x is None else np.asarray(result.x, dtype=float)
        self._qp_y = None if result.y is None else np.asarray(result.y, dtype=float)
        self.last_active_constraint_counts = self.active_constraint_counts()
        return result

    def active_constraint_counts(self, active_tol: float = 1e-6) -> dict[str, int]:
        """Return typed active-row counts from the most recent QP solution."""
        if (self._qp_z is None or self._qp_y is None or self._qp_A is None
                or self._qp_row_labels.size != self._qp_A.shape[0]):
            return {}
        A, l, u, y = self._qp_A, self._qp_l, self._qp_u, self._qp_y
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            Az = A @ self._qp_z
        if not np.all(np.isfinite(Az)):
            return {}
        equality = np.isclose(l, u)
        at_bound = (
            (np.isfinite(l) & (np.abs(Az - l) <= 1e-6))
            | (np.isfinite(u) & (np.abs(Az - u) <= 1e-6))
        )
        active = equality | (at_bound & (np.abs(y) > active_tol))
        labels, counts = np.unique(self._qp_row_labels[active], return_counts=True)
        return {str(label): int(count) for label, count in zip(labels, counts)}

    def input_sensitivity_with_duals(
        self, dq_du: np.ndarray, active_tol: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Return current-cell primal and dual KKT sensitivities.

        ``dq_du`` is d(linear cost)/du with shape (n, m).  The QP Hessian does
        not depend on u, so on a fixed active set the solution is affine in u
        and its Jacobian solves

            [P   Aa^T] [dz/du]   [-dq_du]
            [Aa   0  ] [dnu  ] = [   0  ]

        where ``Aa`` are the rows active at the current solution.  The returned
        tuple is ``(dz_du, dnu_du, active, at_lower, at_upper)``.  ``dnu_du``
        uses the same signed dual convention as OSQP: upper-bound multipliers
        are nonnegative and lower-bound multipliers are nonpositive.  This lets
        a caller impose both primal feasibility of inactive rows and sign
        feasibility of active multipliers when constructing a critical region.

        Returns ``None`` if the QP has not been solved or the KKT system is
        singular.
        """
        if self._qp_z is None or self._qp_y is None:
            return None
        if not (np.all(np.isfinite(self._qp_z)) and np.all(np.isfinite(self._qp_y))):
            return None
        A, l, u, y = self._qp_A, self._qp_l, self._qp_u, self._qp_y
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            Az = A @ self._qp_z
        if not np.all(np.isfinite(Az)):
            return None
        # Equalities (l == u) are always active; inequalities are active when the
        # row sits at a finite bound with a nonzero multiplier.
        equality = np.isclose(l, u)
        at_lower = np.isfinite(l) & (np.abs(Az - l) <= 1e-6)
        at_upper = np.isfinite(u) & (np.abs(Az - u) <= 1e-6)
        at_bound = at_lower | at_upper
        active = equality | (at_bound & (np.abs(y) > active_tol))
        Aa = A[active]
        n = self._qp_P.shape[0]
        na = int(Aa.shape[0])
        K = np.zeros((n, na))
        kkt = np.block([
            [self._qp_P, Aa.T],
            [Aa, np.zeros((na, na))],
        ]) if na else self._qp_P
        dq_du = np.asarray(dq_du, dtype=float)
        if not np.all(np.isfinite(dq_du)):
            return None
        rhs = np.vstack((-dq_du, np.zeros((na, dq_du.shape[1]))))
        try:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                sol = np.linalg.solve(kkt + 1e-9 * np.eye(kkt.shape[0]), rhs)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(sol)):
            return None
        return sol[:n], sol[n:], active, at_lower, at_upper

    def input_sensitivity(self, dq_du: np.ndarray, active_tol: float = 1e-6) -> np.ndarray | None:
        """Return ``dz/du`` on the current active-set cell.

        This compatibility wrapper retains the original mapper API.  New
        critical-region construction should call
        :meth:`input_sensitivity_with_duals` so active dual feasibility is not
        silently omitted.
        """
        sensitivity = self.input_sensitivity_with_duals(dq_du, active_tol)
        return None if sensitivity is None else sensitivity[0]

    def command(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        q_ref: np.ndarray,
        qd_ref: np.ndarray,
        body_acc_des: np.ndarray,
        task_acc_des: np.ndarray,
        hand_jac: np.ndarray,
        stance_contacts: dict[str, tuple[np.ndarray, np.ndarray]],
        stance_targets: dict[str, np.ndarray],
        base_height_ref: float,
        rpy: np.ndarray,
        com_acc_des: np.ndarray | None = None,
        swing_task: dict | None = None,
        attitude_weight: float = 8.0,
        centroidal_moment_des: np.ndarray | None = None,
        yaw_acc_des: float = 0.0,
        yaw_moment_des: float | None = None,
        yaw_moment_tracking_weight: float = 0.0,
        centroidal_moment_residual_des: np.ndarray | None = None,
        external_hand_force_ff: np.ndarray | None = None,
        contact_force_z_max_override: dict[str, float] | None = None,
        attitude_acc_correction: np.ndarray | None = None,
        mpc_correction: np.ndarray | None = None,
    ):
        self.last_com_bias_acc = np.zeros(3)
        self._yaw_dq_du = None
        self._body_dq_du = None
        self._moment_dq_du = None
        self.last_yaw_acc_target = 0.0
        self.last_yaw_moment_target = 0.0
        self.last_yaw_moment = 0.0
        self.last_yaw_output_kind = "acceleration"
        self.last_centroidal_moment_target = np.zeros(3)
        self.last_centroidal_moment = np.zeros(3)
        external_hand_force_ff = (
            np.zeros(3) if external_hand_force_ff is None
            else np.asarray(external_hand_force_ff, dtype=float).reshape(3)
        )
        self.last_external_hand_force_ff = external_hand_force_ff.copy()
        if yaw_moment_tracking_weight < 0.0:
            raise ValueError("yaw_moment_tracking_weight must be nonnegative")
        q_act, qd_act = self.joint_state(data)
        # One forward-kinematics perturbation supplies the convective
        # acceleration Jdot*qdot terms used by every soft Cartesian task.  The
        # earlier practical realizer omitted these terms even though the paper
        # formulated J*qdd + Jdot*qdot tracking.
        fd_eps = 1e-6
        data_fd = mujoco.MjData(model)
        data_fd.qpos[:] = data.qpos
        data_fd.qvel[:] = data.qvel
        mujoco.mj_integratePos(model, data_fd.qpos, data.qvel, fd_eps)
        mujoco.mj_forward(model, data_fd)

        def _site_jdot_qdot(sid: int, J_now: np.ndarray) -> np.ndarray:
            J_fd = site_jac(model, data_fd, sid)
            return ((J_fd - J_now) / fd_eps) @ data.qvel

        def _point_jdot_qdot(point: np.ndarray, body: int) -> np.ndarray:
            Jdot = np.zeros((3, self.nv))
            mujoco.mj_jacDot(model, data, Jdot, None,
                             np.asarray(point, dtype=float), body)
            return Jdot @ data.qvel
        ncontacts = len(stance_contacts)
        nlam = 3 * ncontacts
        n = self.nv + self.nu + nlam
        if mpc_correction is None:
            correction = None
            authority_dq_du = None
        else:
            correction = np.asarray(mpc_correction, dtype=float).reshape(-1)
            if correction.shape != (5,) or not np.all(np.isfinite(correction)):
                raise ValueError("mpc_correction must be a finite 5-D vector")
            if com_acc_des is None or attitude_acc_correction is None:
                raise ValueError("mpc_correction requires CoM and attitude corrections")
            authority_dq_du = np.zeros((n, 5))
        authority_clipped = False
        self.last_mpc_torque_map = None
        self.last_mpc_torque_offset = None
        self.last_mpc_qdd_map = None
        self.last_mpc_qdd_offset = None
        P = 1e-5 * np.eye(n)
        q = np.zeros(n)
        foot_eq_rows = []
        foot_eq_targets = []
        wrench_map = np.zeros((6, nlam))
        wrench_des = None

        C_joint = np.zeros((self.nu, n))
        C_joint[:, self.dof] = np.eye(self.nu)
        qdd_joint_des = self.joint_kp * (q_ref - q_act) + self.joint_kd * (qd_ref - qd_act)
        self._add_ls(P, q, C_joint, np.clip(qdd_joint_des, -80.0, 80.0), 1.0)

        if com_acc_des is not None:
            # Practical body-port recovery: penalize the difference between the
            # realized whole-body CoM acceleration and c_ddot_d + u_c.  This is a
            # weighted objective, so its realization residual is logged below.
            Jcom = np.zeros((3, self.nv))
            mujoco.mj_jacSubtreeCom(model, data, Jcom, self.root_body)
            Jcom_fd = np.zeros((3, self.nv))
            mujoco.mj_jacSubtreeCom(model, data_fd, Jcom_fd, self.root_body)
            com_bias_acc = ((Jcom_fd - Jcom) / fd_eps) @ data.qvel
            self.last_com_bias_acc = com_bias_acc.copy()
            C_com = np.zeros((3, n))
            C_com[:, :self.nv] = Jcom
            self._add_ls(
                P, q, C_com,
                np.clip(com_acc_des - com_bias_acc, -35.0, 35.0),
                self.com_task_weight,
            )
            # u_c enters the QP only through objective targets, so dq/du is a sum
            # of -w * C^T (dtarget/du) over every objective whose target depends
            # on u.  This is the CoM-acceleration term; the exact realizer adds a
            # centroidal-wrench term below.  The map is affine only off the clip.
            self._com_dq_du = -self.com_task_weight * C_com[:2, :].T
            self._com_clipped = bool(np.any(np.abs(np.asarray(com_acc_des)[:2]) >= 35.0))
            if authority_dq_du is not None:
                authority_dq_du[:, :3] -= self.com_task_weight * C_com.T
                authority_clipped |= bool(
                    np.any(np.abs(np.asarray(com_acc_des)) >= 35.0)
                )
            self.last_com_acc_des = np.asarray(com_acc_des, dtype=float).copy()
            # Vertical height stays stiff (weight 8). Torso attitude uses a
            # separate weight: relaxing it (attitude_weight < 8) lets the QP use
            # centroidal angular momentum (torso/arm rotation) for balance beyond
            # the ankle CoP -- the hip strategy.
            C_h = np.zeros((1, n)); C_h[:, 2] = 1.0
            self._add_ls(P, q, C_h,
                        np.clip(np.array([80.0 * (base_height_ref - data.qpos[2]) - 14.0 * data.qvel[2]]), -35.0, 35.0),
                        8.0)
            C_att = np.zeros((3, n)); C_att[:, 3:6] = np.eye(3)
            attitude_acc_des = -42.0 * rpy - 9.0 * data.qvel[3:6]
            if attitude_acc_correction is not None:
                attitude_acc_des[:2] += np.asarray(
                    attitude_acc_correction, dtype=float
                ).reshape(2)
            # The legacy controller leaves yaw to this fixed PD target.  A
            # centroidal yaw-moment controller instead owns the yaw channel
            # through the contact-wrench objective below; retaining both
            # objectives would request mutually inconsistent yaw dynamics.
            # Roll/pitch PD remains active in either mode.
            if yaw_moment_des is None and centroidal_moment_residual_des is None:
                attitude_acc_des[2] += float(yaw_acc_des)
            else:
                attitude_acc_des[2] = float(yaw_acc_des)
            self.last_yaw_acc_target = float(attitude_acc_des[2])
            self._add_ls(P, q, C_att, np.clip(attitude_acc_des, -35.0, 35.0), attitude_weight)
            if authority_dq_du is not None:
                authority_dq_du[:, 3:5] -= attitude_weight * C_att[:2, :].T
                authority_clipped |= bool(np.any(np.abs(attitude_acc_des[:2]) >= 35.0))
            # The yaw command changes only the yaw row of this least-squares
            # target.  It is retained separately until the CoM sensitivity has
            # also received its centroidal-wrench contribution below.
            self._yaw_dq_du = -float(attitude_weight) * C_att[2:3, :].T
        else:
            C_base = np.zeros((6, n))
            C_base[:6, :6] = np.eye(6)
            base_target = np.zeros(6)
            base_target[:2] = body_acc_des[:2]
            base_target[2] = 80.0 * (base_height_ref - data.qpos[2]) - 14.0 * data.qvel[2]
            base_target[3:6] = -42.0 * rpy - 9.0 * data.qvel[3:6]
            self._add_ls(P, q, C_base, np.clip(base_target, -35.0, 35.0), 8.0)

        C_task = np.zeros((3, n))
        C_task[:, :self.nv] = hand_jac
        hand_bias_acc = _site_jdot_qdot(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site"),
            hand_jac,
        )
        self.last_hand_jac = np.asarray(hand_jac, dtype=float).copy()
        self._add_ls(
            P, q, C_task,
            np.clip(task_acc_des - hand_bias_acc,
                    -self.task_acc_clip, self.task_acc_clip),
            self.task_weight,
        )
        # u_t enters only this objective's target, so dq/du_t = -w_task * C_task^T.
        self.last_task_acc_des = np.asarray(task_acc_des, dtype=float).copy()
        self._task_dq_du = -self.task_weight * C_task.T
        self._task_clipped = bool(
            np.any(np.abs(np.asarray(task_acc_des, dtype=float)) >= self.task_acc_clip)
        )

        Jc_blocks = []
        lam_z_ub = None
        if ncontacts and contact_force_z_max_override:
            lam_z_ub = np.full(ncontacts, self.fz_max)
            for i, key in enumerate(stance_contacts):
                foot = key.split("_", 1)[0]
                if foot in contact_force_z_max_override:
                    lam_z_ub[i] = min(self.fz_max, float(contact_force_z_max_override[foot]))
                elif key in contact_force_z_max_override:
                    lam_z_ub[i] = min(self.fz_max, float(contact_force_z_max_override[key]))
            self.last_contact_force_z_max_override = lam_z_ub.copy()
        else:
            self.last_contact_force_z_max_override = None
        if ncontacts:
            fz_nom = float(np.sum(model.body_mass) * 9.81 / ncontacts)
            C_lam = np.zeros((nlam, n))
            target_lam = np.zeros(nlam)
            for i, (key, (pos, J)) in enumerate(stance_contacts.items()):
                Jc_blocks.append(J)
                C_foot = np.zeros((3, n))
                C_foot[:, :self.nv] = J
                pos_err = pos - stance_targets[key]
                vel = J @ data.qvel
                foot_acc_des = -360.0 * pos_err - 36.0 * vel
                foot_acc_des = np.clip(foot_acc_des, -80.0, 80.0)
                foot = key.split("_", 1)[0]
                contact_bias_acc = _point_jdot_qdot(pos, self.foot_body[foot])
                self._add_ls(P, q, C_foot,
                             foot_acc_des - contact_bias_acc, 10.0)
                if not USE_VIRTUAL_FOOT_POLYGON:
                    foot_eq_rows.append(C_foot)
                    foot_eq_targets.append(foot_acc_des)
                base = 3 * i
                C_lam[base:base + 3, self.nv + self.nu + base:self.nv + self.nu + base + 3] = np.eye(3)
                target_lam[base + 2] = fz_nom
                wrench_map[:3, base:base + 3] = np.eye(3)
                arm = pos - data.subtree_com[self.root_body]
                wrench_map[3:, base:base + 3] = np.array(
                    [[0.0, -arm[2], arm[1]],
                    [arm[2], 0.0, -arm[0]],
                    [-arm[1], arm[0], 0.0]]
                )
            self._add_ls(P, q, C_lam, target_lam, 2e-4)
            Jc = np.vstack(Jc_blocks)

            if self.exact_realizer:
                # Contact forces remain distributed over virtual sole corners,
                # but rigid-contact acceleration is imposed once per stance
                # foot as a six-dimensional body constraint.  Constraining all
                # four corner Jacobians separately is redundant and becomes
                # numerically inconsistent when Jdot*qdot is approximated.
                stance_feet = sorted({key.split("_", 1)[0]
                                    for key in stance_contacts})
                for foot in stance_feet:
                    jp = np.zeros((3, self.nv))
                    jr = np.zeros((3, self.nv))
                    mujoco.mj_jacBody(model, data, jp, jr, self.foot_body[foot])
                    Jfoot6 = np.vstack((jp, jr))
                    jp_fd = np.zeros((3, self.nv))
                    jr_fd = np.zeros((3, self.nv))
                    mujoco.mj_jacBody(model, data_fd, jp_fd, jr_fd, self.foot_body[foot])
                    Jdot_qdot = ((np.vstack((jp_fd, jr_fd)) - Jfoot6) / fd_eps) @ data.qvel
                    Cfoot6 = np.zeros((6, n))
                    Cfoot6[:, :self.nv] = Jfoot6
                    # Acceleration-level rigid contact with a Baumgarte velocity
                    # term and translational drift correction.  The latter is
                    # essential after touchdown: velocity damping alone permits
                    # a newly added stance foot to separate slowly from its
                    # measured world target.  Corner-target errors provide the
                    # corresponding sole translation without imposing four
                    # redundant six-dimensional hard constraints.
                    foot_keys = [key for key in stance_contacts
                                if key.split("_", 1)[0] == foot]
                    sole_pos_error = np.mean(
                        [stance_contacts[key][0] - stance_targets[key]
                        for key in foot_keys],
                        axis=0,
                    )
                    afoot6 = -Jdot_qdot - 36.0 * (Jfoot6 @ data.qvel)
                    afoot6[:3] -= 360.0 * sole_pos_error
                    afoot6 = np.clip(afoot6, -80.0, 80.0)
                    foot_eq_rows.append(Cfoot6)
                    foot_eq_targets.append(afoot6)

            if self.exact_realizer and com_acc_des is not None:
                # Eq. (22), with s_W eliminated: ||G lambda-W_des||^2.
                # MuJoCo's bias term contains gravity, hence static support
                # corresponds to F_des=m*g_up.
                total_mass = float(np.sum(model.body_mass))
                moment_des = (np.zeros(3) if centroidal_moment_des is None
                            else np.asarray(centroidal_moment_des, dtype=float))
                if centroidal_moment_residual_des is not None:
                    moment_des = moment_des + np.asarray(
                        centroidal_moment_residual_des, dtype=float
                    ).reshape(3)
                    self.last_yaw_output_kind = "moment"
                if yaw_moment_des is not None:
                    moment_des = moment_des.copy()
                    moment_des[2] += float(yaw_moment_des)
                    self.last_yaw_moment_target = float(moment_des[2])
                    self.last_yaw_output_kind = "moment"
                wrench_des = np.r_[total_mass * (np.asarray(com_acc_des) + G), moment_des]
                self.last_centroidal_moment_target = moment_des.copy()
                C_wrench = np.zeros((6, n))
                C_wrench[:, self.nv + self.nu:] = wrench_map
                self._add_ls(P, q, C_wrench, wrench_des, self.wrench_weight)
                # The exact realizer also drives u through the centroidal-wrench
                # target: d(wrench_des[:2])/du = m*I, so this term must be added
                # to dq/du or the sensitivity is wrong (it dominates at the
                # wrench weight actually used).
                if self._com_dq_du is not None:
                    self._com_dq_du = self._com_dq_du + (
                        -self.wrench_weight * total_mass * C_wrench[:2, :].T
                    )
                if yaw_moment_des is not None:
                    self._yaw_dq_du = -self.wrench_weight * C_wrench[5:6, :].T
                    if yaw_moment_tracking_weight > 0.0:
                        # Preserve the six-dimensional wrench term but add a
                        # yaw-only allocation priority.  This is deliberately
                        # opt-in: its contact-force trade-off is a gait
                        # redesign, not part of the validated baseline.
                        self._add_ls(
                            P, q, C_wrench[5:6], moment_des[2:3],
                            float(yaw_moment_tracking_weight),
                        )
                        self._yaw_dq_du = self._yaw_dq_du - (
                            float(yaw_moment_tracking_weight) * C_wrench[5:6, :].T
                        )
                if centroidal_moment_residual_des is not None:
                    self._moment_dq_du = -self.wrench_weight * C_wrench[3:6, :].T
        else:
            Jc = np.zeros((0, self.nv))

        if self._com_dq_du is not None and self._moment_dq_du is not None:
            self._body_dq_du = np.hstack((self._com_dq_du, self._moment_dq_du))
        elif self._com_dq_du is not None and self._yaw_dq_du is not None:
            self._body_dq_du = np.hstack((self._com_dq_du, self._yaw_dq_du))

        # Swing-foot Cartesian task (single support): track a lift-and-place
        # trajectory for the non-stance foot as an acceleration objective.
        if swing_task is not None:
            sid = swing_task["sid"]
            Jsw = site_jac(model, data, sid)
            swing_bias_acc = _site_jdot_qdot(sid, Jsw)
            pos = data.site_xpos[sid].copy()
            vel = Jsw @ data.qvel
            acc_des = swing_task["kp"] * (swing_task["pos_des"] - pos) + swing_task["kd"] * (swing_task["vel_des"] - vel)
            C_sw = np.zeros((3, n))
            C_sw[:, :self.nv] = Jsw
            self._add_ls(P, q, C_sw,
                         np.clip(acc_des - swing_bias_acc, -60.0, 60.0),
                         swing_task.get("weight", 12.0))

        C_tau = np.zeros((self.nu, n))
        C_tau[:, self.nv:self.nv + self.nu] = np.eye(self.nu)
        self._add_ls(P, q, C_tau, data.qfrc_bias[self.dof], 2e-4)

        M = self._mass_matrix(model, data)
        Aeq_extra = np.vstack(foot_eq_rows) if foot_eq_rows else None
        beq_extra = np.concatenate(foot_eq_targets) if foot_eq_targets else None
        qdd_lb = qdd_ub = None
        if self.exact_realizer and self.enforce_one_step_joint_limits:
            # One-step actuated-joint position row of (22), expressed as qdd
            # bounds and intersected with the generic acceleration bounds.
            eps = 1e-3
            dt = COMMAND_DT
            qdd_lb = np.full(self.nv, -np.inf)
            qdd_ub = np.full(self.nv, np.inf)
            for qadr, dof in zip(self.qadr, self.dof):
                jid = int(model.dof_jntid[dof])
                if not model.jnt_limited[jid]:
                    continue
                lo, hi = model.jnt_range[jid]
                predicted_no_acc = data.qpos[qadr] + dt * data.qvel[dof]
                # The Menagerie stand keyframe places a few joints inside the
                # numerical epsilon band.  A one-step invariant constraint is
                # undefined there (it would demand acceleration beyond the
                # physical qdd bound), so activate it after the state is inside
                # the shrunken interval; torque/acceleration bounds still apply
                # during recovery into that interval.
                if predicted_no_acc <= lo + eps or predicted_no_acc >= hi - eps:
                    continue
                qdd_lb[dof] = 2.0 * (lo + eps - predicted_no_acc) / dt**2
                qdd_ub[dof] = 2.0 * (hi - eps - predicted_no_acc) / dt**2
        # The physical plant may be subject to an external hand force.  The
        # controller receives only its feedforward estimate; observer-only
        # operation leaves this value at zero.  Moving J_t^T F_hat to the left
        # side makes the QP dynamics explicit about which load it models.
        modeled_bias = data.qfrc_bias.copy() - hand_jac.T @ external_hand_force_ff
        result = self._solve_qp(M, modeled_bias, Jc, P, q, nlam,
                                Aeq_extra, beq_extra, qdd_lb, qdd_ub, lam_z_ub)
        self.last_qdd_lower = np.full(self.nv, -120.0)
        self.last_qdd_upper = np.full(self.nv, 120.0)
        if qdd_lb is not None:
            self.last_qdd_lower = np.maximum(self.last_qdd_lower, qdd_lb)
        if qdd_ub is not None:
            self.last_qdd_upper = np.minimum(self.last_qdd_upper, qdd_ub)
        self.last_status = result.info.status
        self.last_fallback = result.x is None or result.info.status_val not in (1, 2)

        if self.last_fallback:
            tau = np.clip(data.qfrc_bias[self.dof], self.torque_min, self.torque_max)
            data.ctrl[self.ctrl_id] = tau
            self.last_eq_residual = np.inf
            self.last_body_acc_residual = np.inf
            self.last_task_acc_tracking_slack_norm = np.inf
            self.last_task_acc_residual = np.inf
            self.last_contact_force = np.zeros(nlam)
            self.last_wrench_slack = np.full(6, np.inf)
            self.last_task_slack = np.full(3, np.inf)
            self.last_contact_acc_residual = np.inf
            self.last_qdd = np.full(self.nv, np.nan)
            self.last_tau_qp = np.full(self.nu, np.nan)
            return tau, tau.copy(), np.zeros_like(tau)

        z = result.x
        qdd = z[:self.nv]
        tau_qp = z[self.nv:self.nv + self.nu]
        self.last_qdd = qdd.copy()
        self.last_tau_qp = tau_qp.copy()
        saturation = np.maximum(tau_qp - self.torque_max, 0.0) + np.maximum(self.torque_min - tau_qp, 0.0)
        tau = np.clip(tau_qp, self.torque_min, self.torque_max)
        if authority_dq_du is not None and not authority_clipped:
            sensitivity = self.input_sensitivity(authority_dq_du)
            if sensitivity is not None:
                torque_map = sensitivity[self.nv:self.nv + self.nu]
                torque_offset = tau_qp - torque_map @ correction
                if np.all(np.isfinite(torque_map)) and np.all(np.isfinite(torque_offset)):
                    self.last_mpc_torque_map = torque_map
                    self.last_mpc_torque_offset = torque_offset
                    qdd_map = sensitivity[:self.nv]
                    qdd_offset = qdd - qdd_map @ correction
                    if np.all(np.isfinite(qdd_map)) and np.all(np.isfinite(qdd_offset)):
                        self.last_mpc_qdd_map = qdd_map
                        self.last_mpc_qdd_offset = qdd_offset
        lam = z[self.nv + self.nu:] if nlam else np.zeros(0)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            dyn = M @ qdd + data.qfrc_bias - self.Btau @ tau
        if nlam:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                dyn -= Jc.T @ lam
            dyn -= hand_jac.T @ external_hand_force_ff
        if not np.all(np.isfinite(dyn)):
            raise FloatingPointError("non-finite inverse-dynamics residual")
        self.last_eq_residual = float(np.linalg.norm(dyn))
        # True body/task realization residuals: realized minus requested acceleration
        # (the soft-objective residuals of the CoM and hand acceleration objectives).
        self.last_task_slack = hand_jac @ qdd + hand_bias_acc - task_acc_des
        self.last_task_acc_tracking_slack_norm = float(np.linalg.norm(self.last_task_slack))
        self.last_task_acc_residual = self.last_task_acc_tracking_slack_norm
        if com_acc_des is not None:
            Jcom_r = np.zeros((3, self.nv))
            mujoco.mj_jacSubtreeCom(model, data, Jcom_r, self.root_body)
            self.last_body_acc_residual = float(np.linalg.norm(
                Jcom_r @ qdd + com_bias_acc - com_acc_des
            ))
        else:
            self.last_body_acc_residual = float("nan")
        self.last_contact_force = lam.copy()
        self.last_wrench_slack = (wrench_map @ lam - wrench_des
                                if wrench_des is not None else np.full(6, np.nan))
        self._last_wrench_map = wrench_map.copy()
        if wrench_des is not None:
            self.last_centroidal_moment = (wrench_map[3:] @ lam).copy()
            self.last_yaw_moment = float(wrench_map[5] @ lam)
        if foot_eq_rows:
            Cc = np.vstack(foot_eq_rows)[:, :self.nv]
            ac = np.concatenate(foot_eq_targets)
            self.last_contact_acc_residual = float(np.linalg.norm(Cc @ qdd - ac))
        else:
            self.last_contact_acc_residual = float("nan")
        data.ctrl[self.ctrl_id] = tau
        return tau, tau_qp, saturation

def make_push(seed: int, enabled: bool) -> PushSpec:
    if not enabled:
        return PushSpec(start=1e9, duration=0.0, force=np.zeros(3))
    rng = np.random.default_rng(seed)
    start = float(rng.uniform(1.0, 2.0))
    duration = float(rng.uniform(0.08, 0.18))
    mag = float(rng.uniform(55.0, 95.0))
    angle = float(rng.uniform(-math.pi, math.pi))
    return PushSpec(start=start, duration=duration, force=mag * np.array([math.cos(angle), math.sin(angle), 0.0]))

def run_trial(
    seed: int,
    duration: float,
    scenario: str,
    distance: float,
    push_enabled: bool,
    exact_realizer: bool = False,
):
    model_path = generate_torque_model()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for value, joint_name in zip(TORQUE_STAND_CTRL, ACTUATED_JOINT_NAMES):
        jid = joint_id(model, joint_name)
        data.qpos[model.jnt_qposadr[jid]] = value
        data.qvel[model.jnt_dofadr[jid]] = 0.0
        mujoco.mj_forward(model, data)

    command_layer = G1CommandLayer()
    realizer = InverseDynamicsQPRealizer(model, exact_realizer=exact_realizer)
    # The revised uneven-terrain study regulates locomotion body coordinates;
    # a simultaneous hand trajectory is outside its scope and competes with
    # stance attitude during single support.  Keep the legacy hand benchmark
    # for fixed-support scenarios only.
    if scenario in ("walk", "contact_switch"):
        realizer.task_weight = 0.0
        realizer.com_task_weight = 400.0
        # Do not intersect a newly activated rigid-touchdown equality with the
        # optional one-step position invariant set.  Torque, acceleration, and
        # the model's joint limits remain in force.
        realizer.enforce_one_step_joint_limits = False
    body_mpc = NormalizedMPC(dim=2, dt=COMMAND_DT, horizon=35, q_pos=55.0, q_vel=12.0, r=0.08, u_max=np.array([3.5, 3.0]))
    task_mpc = NormalizedMPC(dim=3, dt=COMMAND_DT, horizon=18, q_pos=100.0, q_vel=10.0, r=0.12, u_max=np.array([4.5, 4.5, 4.5]))
    body_obs = RandomWalkDisturbanceObserver(dim=2, dt=COMMAND_DT, q_d=0.05, r_y=1.5e-4)
    task_obs = RandomWalkDisturbanceObserver(dim=3, dt=COMMAND_DT, q_d=0.04, r_y=2.0e-4)

    torso = body_id(model, "torso_link")
    pelvis = body_id(model, "pelvis")
    hand_sid = site_id(model, "right_hand_site")
    left_sid = site_id(model, "left_foot")
    right_sid = site_id(model, "right_foot")
    push = make_push(seed, push_enabled)

    q_nom = TORQUE_STAND_CTRL.copy()
    qd_ref = np.zeros_like(q_nom)
    if scenario in ("walk", "contact_switch") or exact_realizer:
        # The keyframe starts 7--11 mm above the floor.  Establish physical sole
        # contacts before freezing stance targets or starting the gait clock.
        # This is required for both soft and exact walking realizers; otherwise
        # the soft path incorrectly anchors airborne virtual sole points.
        warm = InverseDynamicsQPRealizer(model, exact_realizer=False)
        warm_com = robot_com(model, data)
        warm_height = float(data.qpos[2])
        for _ in range(int(round(0.35 / SIM_DT))):
            warm_contacts = warm.contact_points(model, data, ("left", "right"))
            warm_targets = {key: pos.copy() for key, (pos, _) in warm_contacts.items()}
            _, _, warm_hand_jac = hand_state(model, data, hand_sid)
            warm_rpy = roll_pitch_yaw_from_body(data, torso)
            warm_acc = -25.0 * (robot_com(model, data) - warm_com) - 8.0 * com_velocity(
                model, data, warm.root_body)
            warm.command(
                model, data, q_nom, qd_ref, np.zeros(2), np.zeros(3),
                warm_hand_jac, warm_contacts, warm_targets, warm_height,
                warm_rpy, com_acc_des=np.clip(warm_acc, -3.0, 3.0),
                attitude_weight=60.0,
            )
            mujoco.mj_step(model, data); mujoco.mj_forward(model, data)
        data.time = 0.0
    com0 = robot_com(model, data)
    hand0, _, _ = hand_state(model, data, hand_sid)
    base_height_ref = float(data.qpos[2])

    # Stepping scenarios use the DCM (capture-point) reference (unified with
    # run_gait_dcm): a dynamically feasible CoM sway that places the CoM over the
    # current stance foot, instead of a centered CoM that tips in single support.
    dcm_plan = None
    ground_z_walk = float(min(data.site_xpos[left_sid][2], data.site_xpos[right_sid][2]))
    com_ref_xy = com0[:2].copy()
    com_ref_vel_xy = np.zeros(2)
    swing_prev = None
    last_sw_target = None
    swing_cmd = None
    walk_body_mpc = body_mpc
    if scenario in ("walk", "contact_switch"):
        from run_gait_dcm import DCMWalk
        z_c = float(com0[2] - ground_z_walk)
        left0 = data.site_xpos[left_sid][:2].copy()
        right0 = data.site_xpos[right_sid][:2].copy()
        # This file supplies only a conservative shared reference for controller
        # comparisons; fast gait generation is outside the paper's scope.
        step_len = 0.03 if scenario == "walk" else 0.0
        quasi_static_switch = scenario == "contact_switch"
        dcm_plan = DCMWalk(left0, right0, step_len, n_steps=12, z_c=z_c,
                        t_step=1.20 if quasi_static_switch else 0.80,
                        t_ds=1.00 if quasi_static_switch else 0.55,
                        t_settle=1.0,
                        zmp_y_scale=1.0 if quasi_static_switch else 0.85)
        walk_body_mpc = NormalizedMPC(dim=2, dt=COMMAND_DT, horizon=35,
                                    q_pos=90.0, q_vel=16.0, r=0.05, u_max=np.array([6.0, 6.0]))

    d_body = np.zeros(2)
    d_task = np.zeros(3)
    u_body = np.zeros(2)
    u_task = np.zeros(3)
    innovation_body = np.zeros(2)
    innovation_task = np.zeros(3)
    contact_prev = measured_foot_contacts(model, data)
    contact_events = []
    last_contact_event_t = -1e9
    stance_prev: tuple[str, ...] = ()
    stance_targets: dict[str, np.ndarray] = {}

    steps = int(round(duration / SIM_DT))
    command_period = max(1, int(round(COMMAND_DT / SIM_DT)))
    log = {
        "t": np.zeros(steps),
        "com": np.zeros((steps, 3)),
        "com_ref": np.zeros((steps, 3)),
        "rpy": np.zeros((steps, 3)),
        "hand": np.zeros((steps, 3)),
        "hand_ref": np.zeros((steps, 3)),
        "contact": np.zeros((steps, 2), dtype=int),
        "u_body": np.zeros((steps, 2)),
        "d_body": np.zeros((steps, 2)),
        "u_task": np.zeros((steps, 3)),
        "d_task": np.zeros((steps, 3)),
        "tau": np.zeros((steps, model.nu)),
        "tau_sat_norm": np.zeros(steps),
        "tau_limit_utilization": np.zeros(steps),
        "dynamics_equality_residual": np.zeros(steps),
        "body_acc_residual": np.zeros(steps),
        "task_acc_tracking_slack": np.zeros(steps),
        "realizer_fallback": np.zeros(steps, dtype=int),
        "contact_force_norm": np.zeros(steps),
        "wrench_slack": np.zeros((steps, 6)),
        "task_slack": np.zeros((steps, 3)),
        "contact_acc_residual": np.zeros(steps),
        "push_force": np.zeros((steps, 3)),
        "friction_margin": np.zeros(steps),
        "fall": np.zeros(steps, dtype=int),
        "qpos": np.zeros((steps, model.nq)),
    }

    q_ref = q_nom.copy()
    body_acc_des = np.zeros(2)
    task_acc_des = np.zeros(3)
    for k in range(steps):
        t = k * SIM_DT
        com = robot_com(model, data)
        rpy = roll_pitch_yaw_from_body(data, torso)
        hand, hand_vel, hand_jac = hand_state(model, data, hand_sid)

        if dcm_plan is not None:
            # ---- DCM stepping reference (walk / contact_switch) ----
            # A dynamically feasible CoM sway (LIPM/DCM) places the CoM over the
            # current stance foot, so single support is feasible; the swing foot
            # tracks a lift-and-place task. This replaces the centered CoM +
            # position-scaffold gait that tipped laterally in single support.
            stance, swing, s, sw_start, sw_target = dcm_plan.schedule(t)
            if swing_prev is not None and swing is None and last_sw_target is not None:
                dcm_plan.commit_plant(swing_prev, last_sw_target)
            swing_prev = swing
            if swing is not None:
                last_sw_target = sw_target
            xi, zmp = dcm_plan.xi_and_zmp(t)
            q_nom = TORQUE_STAND_CTRL.copy()
            com_ref = np.array([com_ref_xy[0], com_ref_xy[1], com0[2]])
            hand_ref = hand0 + np.array([0.0, 0.03 * math.sin(2.0 * math.pi * 0.35 * t), 0.0])
            hand_v_ref = np.array([0.0, 0.03 * 2.0 * math.pi * 0.35 * math.cos(2.0 * math.pi * 0.35 * t), 0.0])
            hand_a_ref = np.array([0.0, -0.03 * (2.0 * math.pi * 0.35) ** 2 * math.sin(2.0 * math.pi * 0.35 * t), 0.0])
            if exact_realizer:
                hand_ref = hand0.copy(); hand_v_ref[:] = 0.0; hand_a_ref[:] = 0.0
            if stance != stance_prev:
                current_contacts = realizer.contact_points(model, data, stance)
                for key, (pos, _) in current_contacts.items():
                    foot = key.split("_", 1)[0]
                    if foot not in stance_prev or key not in stance_targets:
                        stance_targets[key] = pos.copy()
                for key in list(stance_targets):
                    if key not in current_contacts:
                        del stance_targets[key]
                stance_prev = stance
            stance_contacts = realizer.contact_points(model, data, stance)

            if k % command_period == 0:
                xi_dot = dcm_plan.w * (xi - zmp)
                com_ref_vel_xy = dcm_plan.w * (xi - com_ref_xy)
                com_ref_acc = dcm_plan.w * (xi_dot - com_ref_vel_xy)
                com_ref_xy = com_ref_xy + com_ref_vel_xy * COMMAND_DT
                x_body = np.r_[com[:2] - com_ref_xy,
                            com_velocity(model, data, realizer.root_body)[:2] - com_ref_vel_xy]
                u_body = walk_body_mpc.solve(x_body, d_body)
                d_body, innovation_body = body_obs.step(com[:2] - com_ref_xy, u_body)
                x_task = np.r_[hand - hand_ref, hand_vel - hand_v_ref]
                u_task = task_mpc.solve(x_task, d_task)
                d_task, innovation_task = task_obs.step(hand - hand_ref, u_task)
                q_ref = q_nom.copy()
                body_acc_des = com_ref_acc + u_body
                task_acc_des = hand_a_ref + u_task
                if swing is not None:
                    sid = right_sid if swing == "right" else left_sid
                    xy = sw_start + (sw_target - sw_start) * (0.5 - 0.5 * np.cos(np.pi * s))
                    lift = 0.025 if (exact_realizer and scenario == "contact_switch") else 0.05
                    zt = ground_z_walk + lift * np.sin(np.pi * s)
                    swing_cmd = dict(sid=sid, pos_des=np.array([xy[0], xy[1], zt]),
                                    vel_des=np.zeros(3),
                                    kp=180.0 if exact_realizer else 280.0,
                                    kd=25.0 if exact_realizer else 32.0,
                                    weight=10.0 if exact_realizer else 14.0)
                else:
                    swing_cmd = None
        else:
            # ---- fixed-support scenarios (stand / stand_push) ----
            traj = LocalTrajectory(position=np.zeros(2), velocity=np.zeros(2), acceleration=np.zeros(2), heading=0.0)
            q_nom = TORQUE_STAND_CTRL.copy()
            com_ref = com0 + np.array([traj.position[0], traj.position[1], 0.0])
            hand_ref = hand0 + np.array([traj.position[0], 0.03 * math.sin(2.0 * math.pi * 0.35 * t), 0.0])
            hand_v_ref = np.array([traj.velocity[0], 0.03 * 2.0 * math.pi * 0.35 * math.cos(2.0 * math.pi * 0.35 * t), 0.0])
            hand_a_ref = np.array([traj.acceleration[0], -0.03 * (2.0 * math.pi * 0.35) ** 2 * math.sin(2.0 * math.pi * 0.35 * t), 0.0])
            if exact_realizer:
                hand_ref = hand0.copy(); hand_v_ref[:] = 0.0; hand_a_ref[:] = 0.0
            stance = scheduled_stance_feet(t, scenario)
            if stance != stance_prev:
                current_contacts = realizer.contact_points(model, data, stance)
                valid_target_keys = set(current_contacts)
                for key, (pos, _) in current_contacts.items():
                    foot = key.split("_", 1)[0]
                    if foot not in stance_prev or key not in stance_targets:
                        stance_targets[key] = pos.copy()
                for key in list(stance_targets):
                    if key not in valid_target_keys:
                        del stance_targets[key]
                stance_prev = stance
            stance_contacts = realizer.contact_points(model, data, stance)

            if k % command_period == 0:
                x_body = np.r_[com[:2] - com_ref[:2],
                            com_velocity(model, data, realizer.root_body)[:2] - traj.velocity]
                u_body = body_mpc.solve(x_body, d_body)
                d_body, innovation_body = body_obs.step(com[:2] - com_ref[:2], u_body)
                x_task = np.r_[hand - hand_ref, hand_vel - hand_v_ref]
                u_task = task_mpc.solve(x_task, d_task)
                d_task, innovation_task = task_obs.step(hand - hand_ref, u_task)
                # The body MPC request is realized by the CoM-acceleration
                # objective; this posture bias is only a secondary shaping term.
                q_ref = q_nom.copy()
                q_ref[0] += -0.035 * POSTURE_RECOVERY_GAIN * u_body[0]
                q_ref[6] += -0.035 * POSTURE_RECOVERY_GAIN * u_body[0]
                q_ref[1] += 0.045 * POSTURE_RECOVERY_GAIN * u_body[1]
                q_ref[7] += 0.045 * POSTURE_RECOVERY_GAIN * u_body[1]
                q_ref[4] += 0.018 * POSTURE_RECOVERY_GAIN * u_body[0]
                q_ref[10] += 0.018 * POSTURE_RECOVERY_GAIN * u_body[0]
                body_acc_des = traj.acceleration + u_body
                task_acc_des = hand_a_ref + u_task
        swing_task = swing_cmd if dcm_plan is not None else None

        active_push = push.start <= t < push.start + push.duration
        data.xfrc_applied[:] = 0.0
        if active_push:
            data.xfrc_applied[pelvis, :3] = push.force

        tau, tau_unsat, saturation = realizer.command(
            model,
            data,
            q_ref,
            qd_ref,
            body_acc_des,
            task_acc_des,
            hand_jac,
            stance_contacts,
            stance_targets,
            base_height_ref,
            rpy,
            com_acc_des=np.array([body_acc_des[0], body_acc_des[1], 0.0]),
            swing_task=swing_task,
            attitude_weight=(240.0 if exact_realizer else 120.0)
                            if scenario in ("walk", "contact_switch")
                            else (60.0 if exact_realizer else 8.0),
        )
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        t_log = float(data.time)
        com_log = robot_com(model, data)
        rpy_log = roll_pitch_yaw_from_body(data, torso)
        hand_log, _, _ = hand_state(model, data, hand_sid)
        contacts = measured_foot_contacts(model, data)
        if np.any(contacts != contact_prev) and (t_log - last_contact_event_t) >= 0.08:
            contact_events.append(t_log)
            last_contact_event_t = t_log
            contact_prev = contacts.copy()
        elif np.any(contacts != contact_prev):
            contact_prev = contacts.copy()

        fall = bool(data.qpos[2] < 0.45 or np.max(np.abs(rpy_log[:2])) > 0.85)
        log["t"][k] = t_log
        log["com"][k] = com_log
        log["com_ref"][k] = com_ref
        log["rpy"][k] = rpy_log
        log["hand"][k] = hand_log
        log["hand_ref"][k] = hand_ref
        log["contact"][k] = contacts.astype(int)
        log["u_body"][k] = u_body
        log["d_body"][k] = d_body
        log["u_task"][k] = u_task
        log["d_task"][k] = d_task
        log["tau"][k] = tau
        log["tau_sat_norm"][k] = float(np.linalg.norm(saturation))
        tau_limit = np.maximum(np.abs(realizer.torque_min), np.abs(realizer.torque_max))
        tau_limit = np.maximum(tau_limit, 1e-9)
        log["tau_limit_utilization"][k] = float(np.max(np.abs(tau) / tau_limit))
        log["dynamics_equality_residual"][k] = realizer.last_eq_residual
        log["body_acc_residual"][k] = realizer.last_body_acc_residual
        log["task_acc_tracking_slack"][k] = realizer.last_task_acc_tracking_slack_norm
        log["realizer_fallback"][k] = int(realizer.last_fallback)
        log["contact_force_norm"][k] = float(np.linalg.norm(realizer.last_contact_force))
        log["wrench_slack"][k] = realizer.last_wrench_slack
        log["task_slack"][k] = realizer.last_task_slack
        log["contact_acc_residual"][k] = realizer.last_contact_acc_residual
        log["push_force"][k] = push.force if active_push else 0.0
        log["friction_margin"][k] = friction_margin(model, data)
        log["fall"][k] = int(fall)
        log["qpos"][k] = data.qpos
        if fall:
            # Keep logging arrays rectangular; remaining entries repeat terminal state.
            for key in ("t",):
                log[key][k + 1:] = log[key][k]
            for key, value in log.items():
                if key != "t" and value.shape[0] == steps:
                    value[k + 1:] = value[k]
            break

    summary = summarize_trial(log, seed, scenario, duration, distance, push,
                            contact_events, model_path, exact_realizer)
    return log, summary

def summarize_trial(log, seed, scenario, requested_duration, distance, push,
                    contact_events, model_path, exact_realizer=False):
    t = log["t"]
    valid = np.flatnonzero(t >= 0.0)
    end = int(valid[-1]) if len(valid) else len(t) - 1
    fell = bool(np.any(log["fall"][:end + 1]))
    contact_switches = int(len(contact_events))
    push_mask = np.linalg.norm(log["push_force"][:end + 1], axis=1) > 0
    detected_push = bool(np.max(np.linalg.norm(log["d_body"][:end + 1], axis=1)) > 0.35 and np.any(push_mask))
    residual = log["dynamics_equality_residual"][:end + 1]
    finite_residual = residual[np.isfinite(residual)]
    body_res = log["body_acc_residual"][:end + 1]; body_res = body_res[np.isfinite(body_res)]
    task_res = log["task_acc_tracking_slack"][:end + 1];
    task_res = task_res[np.isfinite(task_res)]
    wrench_res = np.linalg.norm(log["wrench_slack"][:end + 1], axis=1)
    wrench_res = wrench_res[np.isfinite(wrench_res)]
    contact_res = log["contact_acc_residual"][:end + 1]
    contact_res = contact_res[np.isfinite(contact_res)]
    contact_log = log["contact"][:end + 1]
    single_support_s = float(SIM_DT * np.sum(np.sum(contact_log, axis=1) == 1))
    double_support_s = float(SIM_DT * np.sum(np.sum(contact_log, axis=1) == 2))
    return {
        "seed": int(seed),
        "scenario": scenario,
        "exact_realizer": bool(exact_realizer),
        "model": str(model_path.relative_to(HERE)),
        "duration_requested_s": float(requested_duration),
        "duration_completed_s": float(t[end]),
        "distance_commanded_m": float(distance),
        "distance_actual_m": float(log["com"][end, 0] - log["com"][0, 0]),
        "push_enabled": bool(push.duration > 0),
        "push_start_s": float(push.start) if push.duration > 0 else None,
        "push_duration_s": float(push.duration),
        "push_force_n": push.force.tolist(),
        "detected_push": detected_push,
        "contact_switches_detected": contact_switches,
        "contact_event_times_s": contact_events,
        "physical_single_support_s": single_support_s,
        "physical_double_support_s": double_support_s,
        "max_abs_roll_pitch_rad": float(np.max(np.abs(log["rpy"][:end + 1, :2]))),
        "min_pelvis_height_m": float(np.min(log["qpos"][:end + 1, 2])),
        "max_tau_abs_nm": float(np.max(np.abs(log["tau"][:end + 1]))),
        "max_tau_saturation_norm": float(np.max(log["tau_sat_norm"][:end + 1])),
        "max_tau_limit_utilization": float(np.max(log["tau_limit_utilization"][:end + 1])),
        "max_realizer_residual": finite_or_none(np.max(finite_residual)) if finite_residual.size else None,
        "max_dynamics_equality_residual": finite_or_none(np.max(finite_residual)) if finite_residual.size else None,
        "max_body_acc_residual": finite_or_none(np.max(body_res)) if body_res.size else None,
        "max_task_acc_tracking_slack": finite_or_none(np.max(task_res)) if task_res.size else None,
        "max_wrench_slack_norm": finite_or_none(np.max(wrench_res)) if wrench_res.size else None,
        "max_contact_acc_residual": finite_or_none(np.max(contact_res)) if contact_res.size else None,
        "num_realizer_fallbacks": int(np.sum(log["realizer_fallback"][:end + 1])),
        "max_contact_force_norm_n": float(np.max(log["contact_force_norm"][:end + 1])),
        "min_friction_margin": finite_or_none(np.min(log["friction_margin"][:end + 1])),
        "hand_rms_error_mm": float(1000.0 * np.sqrt(np.mean(np.sum((log["hand"][ :end + 1] - log["hand_ref"][ :end + 1]) ** 2, axis=1)))),
        "fell": fell,
        "passes_torque_realizer_smoke": bool(
            (not fell) and t[end] > 0.95 * t[-1]
            and (not exact_realizer or int(np.sum(log["realizer_fallback"][:end + 1])) == 0)
            and (scenario != "contact_switch" or single_support_s >= 0.30)
        ),
        "claim_scope": (
            "Torque-actuated G1 smoke benchmark with normalized body/task MPCs, "
            "RandomWalkDisturbanceObserver detection, and a present-sample "
            "inverse-dynamics/contact QP realizer. This is a replacement for "
            "root assist only after it passes; failures are intentionally retained "
            "in the metrics."
        ),
    }


def save_plot(log, summary, out_png):
    t = log["t"]
    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(t, log["com"][:, 0], label="CoM x")
    axes[0].plot(t, log["com_ref"][:, 0], "--", label="ref")
    axes[0].set_ylabel("x [m]")
    axes[0].legend(loc="best")

    axes[1].plot(t, log["com"][:, 1], label="CoM y")
    axes[1].plot(t, log["com_ref"][:, 1], "--", label="ref")
    axes[1].set_ylabel("y [m]")
    axes[1].legend(loc="best")

    axes[2].plot(t, log["rpy"][:, 0], label="roll")
    axes[2].plot(t, log["rpy"][:, 1], label="pitch")
    axes[2].set_ylabel("rad")
    axes[2].legend(loc="best")

    axes[3].plot(t, np.linalg.norm(log["d_body"], axis=1), label="||d_body||")
    axes[3].plot(t, np.linalg.norm(log["push_force"], axis=1) / 100.0, label="push/100")
    axes[3].set_ylabel("detect")
    axes[3].legend(loc="best")

    axes[4].plot(t, log["tau_sat_norm"], label="post-QP clipping residual")
    axes[4].plot(t, log["tau_limit_utilization"], label="max torque utilization")
    axes[4].plot(t, log["dynamics_equality_residual"] / 100.0, label="ID residual / 100")
    axes[4].step(t, log["contact"][:, 0], where="post", label="left contact")
    axes[4].step(t, log["contact"][:, 1] + 1.2, where="post", label="right contact")
    axes[4].set_ylabel("constraints")
    axes[4].set_xlabel("time [s]")
    axes[4].legend(loc="best")

    fig.suptitle(
        f"Torque realizer {summary['scenario']} seed={summary['seed']} "
        f"pass={summary['passes_torque_realizer_smoke']} fell={summary['fell']}"
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def aggregate(summaries):
    return {
        "num_trials": len(summaries),
        "num_passed": int(sum(s["passes_torque_realizer_smoke"] for s in summaries)),
        "num_fell": int(sum(s["fell"] for s in summaries)),
        "success_rate": float(np.mean([s["passes_torque_realizer_smoke"] for s in summaries])) if summaries else 0.0,
        "median_completed_s": float(np.median([s["duration_completed_s"] for s in summaries])) if summaries else 0.0,
        "median_max_roll_pitch_rad": float(np.median([s["max_abs_roll_pitch_rad"] for s in summaries])) if summaries else 0.0,
        "median_hand_rms_error_mm": float(np.median([s["hand_rms_error_mm"] for s in summaries])) if summaries else 0.0,
        "trials": summaries,
    }

def finite_or_none(value: float):
    value = float(value)
    return value if np.isfinite(value) else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["stand", "stand_push", "contact_switch", "walk"], default="stand_push")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--distance", type=float, default=0.6)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--push", action="store_true", default=False)
    parser.add_argument("--exact-realizer", action="store_true",
                        help="use the residual-aware six-dimensional realizer of Eq. (22)")
    args = parser.parse_args()

    summaries = []
    for i in range(args.trials):
        seed = args.seed + i
        push_enabled = args.push or args.scenario == "stand_push"
        log, summary = run_trial(seed, args.duration, args.scenario, args.distance,
                                push_enabled, args.exact_realizer)
        mode = "_exact" if args.exact_realizer else ""
        prefix = f"g1_torque_{args.scenario}{mode}_seed{seed}"
        np.savez_compressed(RESULTS / f"{prefix}_log.npz", **log)
        with (RESULTS / f"{prefix}_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        save_plot(log, summary, RESULTS / f"{prefix}.png")
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    agg = aggregate(summaries)
    mode = "_exact" if args.exact_realizer else ""
    out = RESULTS / f"g1_torque_{args.scenario}{mode}_aggregate.json"
    with out.open("w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))
    print(f"saved: {out}")

if __name__ == "__main__":
    main()
