#!/usr/bin/env python3
"""Experiments for the multirate interaction-dynamics architecture.

E1  Real-time budget      -- one whole-body QP per 1 kHz cycle; component timing.
E2  Fixed vs mapped bound -- C1 (+/-4 fixed box) vs C2 (analytic H_k u <= h_k).
E3  Feedforward occupancy -- H_k, h_k move with payload/arm/reference/mode while
                             the canonical (A, B) stay bitwise constant.
E4  Contact switching     -- DS -> SS -> DS with mode-updated constraints.
E5  Mapping fidelity      -- analytic set graded against the exact-QP bisection
                             (false positives / false negatives / boundary error).

Usage: MPLCONFIGDIR=/private/tmp/mplconfig python3 run_multirate_benchmarks.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from multirate_control import MultirateInteractionController, RateConfig
from normalized_mpc import NormalizedMPC, RandomWalkDisturbanceObserver
from realization_authority import (
    AnalyticAuthorityMapper, ContinuationAuthorityEstimator,
    ExactResidualBisectionEstimator,
)
from run_authority_benchmarks import settle_model, scenario_context
from run_g1_torque_realizer_benchmark import (
    ACTUATED_JOINT_NAMES, TORQUE_STAND_CTRL, InverseDynamicsQPRealizer,
    com_velocity, hand_state, robot_com,
)
from run_g1_root_assist_demo import roll_pitch_yaw_from_body

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)
FIGURES = HERE.parent / "figures"; FIGURES.mkdir(exist_ok=True)

TOL = 0.35          # componentwise realization tolerance [m/s^2]
TASK_WEIGHT = 8.0e3 # hand-task objective weight needed for a faithful task port
FIXED_BOX = 4.0     # C1


# ---------------------------------------------------------------------------
# shared driver
# ---------------------------------------------------------------------------

def make_controller(model, *, use_authority: bool, rates: RateConfig,
                    fixed_box: float = FIXED_BOX,
                    q_pos: float = 55.0, r_cost: float = 0.08):
    realizer = InverseDynamicsQPRealizer(model, exact_realizer=True)
    body_mpc = NormalizedMPC(dim=2, dt=rates.body_dt, horizon=25,
                             q_pos=q_pos, q_vel=12.0, r=r_cost)
    if not use_authority:
        body_mpc.update_input_box(-fixed_box, fixed_box)
    ctrl = MultirateInteractionController(
        realizer, rates=rates, body_mpc=body_mpc,
        body_obs=RandomWalkDisturbanceObserver(dim=2, dt=rates.body_dt,
                                               q_d=0.05, r_y=1.5e-4),
        mapper=AnalyticAuthorityMapper(),
        use_authority=use_authority,
        fallback_box=np.array([fixed_box, fixed_box]),
    )
    return ctrl


def lateral_step_run(use_authority: bool, *, rates: RateConfig,
                     duration: float = 0.8, step_y: float = 0.10,
                     q_pos: float = 1500.0, r_cost: float = 0.02):
    """Lateral CoM reference step tracked by an AGGRESSIVE predictor.

    The predictor is deliberately high-gain: its unconstrained command exceeds
    what the standing realizer can deliver.  That is the regime the residual
    bound exists for -- a fixed +/-4 box lets the predictor ask for accelerations
    the contacts cannot produce, while the mapped set H_k u <= h_k does not."""
    model, data, torso, hand_sid, hr = settle_model()
    ctrl = make_controller(model, use_authority=use_authority, rates=rates,
                           q_pos=q_pos, r_cost=r_cost)
    R = ctrl.realizer
    stance = ("left", "right")
    com0 = robot_com(model, data)
    targets = {k: p.copy() for k, (p, _) in R.contact_points(model, data, stance).items()}
    qd_ref = np.zeros(model.nu)

    N = int(round(duration / rates.sim_dt))
    log = {k: np.zeros(N) for k in ("t", "res", "ey", "ex")}
    log["u"] = np.zeros((N, 2))
    src_counts: dict[str, int] = {}
    fell = False

    for k in range(N):
        t = k * rates.sim_dt
        com = robot_com(model, data)
        vel = com_velocity(model, data, R.root_body)
        rpy = roll_pitch_yaw_from_body(data, torso)
        _, _, hj = hand_state(model, data, hand_sid)
        contacts = R.contact_points(model, data, stance)

        c_ref = com0.copy()
        c_ref[1] += step_y                      # step reference
        err = np.r_[com[:2] - c_ref[:2], vel[:2]]

        info = ctrl.step(
            model, data, t,
            q_ref=TORQUE_STAND_CTRL.copy(), qd_ref=qd_ref,
            com_ref_acc=np.zeros(3), body_error=err,
            stance=stance, stance_contacts=contacts, stance_targets=targets,
            base_height_ref=hr, rpy=rpy, hand_jac=hj,
        )
        if info["body_source"]:
            src_counts[info["body_source"]] = src_counts.get(info["body_source"], 0) + 1

        mujoco.mj_step(model, data); mujoco.mj_forward(model, data)
        log["t"][k] = t
        log["res"][k] = info["residual"]
        log["u"][k] = ctrl.u_body
        e = robot_com(model, data) - c_ref
        log["ex"][k], log["ey"][k] = e[0], e[1]
        if data.qpos[2] < 0.45 or np.max(np.abs(rpy[:2])) > 0.85:
            fell = True
            break

    res = log["res"][np.isfinite(log["res"])]
    rms = float(np.sqrt(np.mean(log["ex"] ** 2 + log["ey"] ** 2)) * 1000)
    return dict(
        variant="C2_analytic_authority" if use_authority else "C1_fixed_box",
        rms_planar_error_mm=rms,
        median_residual=float(np.median(res)) if res.size else float("nan"),
        max_residual=float(res.max()) if res.size else float("nan"),
        residual_over_tol_fraction=float(np.mean(res > TOL)) if res.size else float("nan"),
        fell=bool(fell),
        qp_fallbacks=int(ctrl.realizer.last_fallback),
        body_constraint_sources=src_counts,
        max_abs_u=[round(float(np.abs(log["u"][:, 0]).max()), 3),
                   round(float(np.abs(log["u"][:, 1]).max()), 3)],
        diagnostics=ctrl.diagnostics(),
        _log=log,
    )


# ---------------------------------------------------------------------------
# E1 real-time budget
# ---------------------------------------------------------------------------

def e1_realtime(rates: RateConfig) -> dict:
    # A REPRESENTATIVE controller, not the deliberately-aggressive E2 stress
    # predictor: nominal gains, short horizon, and both ports active -- the
    # configuration a deployment would actually run.  E2's high-gain predictor
    # exists to make a fixed box overshoot and is not a real-time reference.
    model, data, torso, hand_sid, hr = settle_model()
    realizer = InverseDynamicsQPRealizer(model, exact_realizer=True)
    ctrl = MultirateInteractionController(
        realizer, rates=rates,
        body_mpc=NormalizedMPC(dim=2, dt=rates.node_dt, horizon=10,
                               q_pos=55.0, q_vel=12.0, r=0.08),
        task_mpc=NormalizedMPC(dim=3, dt=rates.node_dt, horizon=10,
                               q_pos=800.0, q_vel=40.0, r=0.05),
        body_obs=RandomWalkDisturbanceObserver(dim=2, dt=rates.node_dt, q_d=0.05, r_y=1.5e-4),
        task_obs=RandomWalkDisturbanceObserver(dim=3, dt=rates.node_dt, q_d=0.04, r_y=2.0e-4),
        mapper=AnalyticAuthorityMapper(), use_authority=True)
    R = ctrl.realizer
    stance = ("left", "right")
    com0 = robot_com(model, data)
    targets = {k: p.copy() for k, (p, _) in R.contact_points(model, data, stance).items()}
    hand0, _, _ = hand_state(model, data, hand_sid)
    for k in range(int(round(3.0 / rates.servo_dt))):
        t = k * rates.servo_dt
        com = robot_com(model, data); vel = com_velocity(model, data, R.root_body)
        rpy = roll_pitch_yaw_from_body(data, torso)
        hp, hv, hj = hand_state(model, data, hand_sid)
        ctrl.step(model, data, t,
                  q_ref=TORQUE_STAND_CTRL.copy(), qd_ref=np.zeros(model.nu),
                  com_ref_acc=np.zeros(3), body_error=np.r_[com[:2] - com0[:2], vel[:2]],
                  stance=stance, stance_contacts=R.contact_points(model, data, stance),
                  stance_targets=targets, base_height_ref=hr, rpy=rpy, hand_jac=hj,
                  task_acc_ref=np.zeros(3), task_error=np.r_[hp - hand0, hv])
        mujoco.mj_step(model, data); mujoco.mj_forward(model, data)
    diag = ctrl.diagnostics()
    d = diag["timing"]
    node = d["optimization_node"]
    budget = 1000.0 * rates.node_dt
    return {
        "claim": ("a 1 kHz servo holds the last optimized command; a 200 Hz "
                  "optimization node solves EXACTLY ONE whole-body QP per update"),
        "servo_hz": int(round(1.0 / rates.servo_dt)),
        "node_hz": int(round(1.0 / rates.node_dt)),
        "authority_hz": int(round(1.0 / rates.authority_dt)),
        "servo_ticks": diag["servo_ticks"],
        "node_updates": diag["node_updates"],
        "whole_body_qp_solves_per_node_update": d["whole_body_qp_solves_per_node_update"],
        "node_ms": node,
        "node_budget_ms": budget,
        "node_deadline_misses": diag["node_deadline_misses"],
        "node_deadline_miss_fraction": round(
            diag["node_deadline_misses"] / max(diag["node_updates"], 1), 4),
        "component_ms": {
            "whole_body_qp": d["realizer"],
            "authority_kkt": d["authority_kkt"],
            "body_predictor": d["body_mpc"],
            "task_predictor": d.get("task_mpc"),
        },
        "note": (
            "The whole-body QP (~2.4 ms, of which only ~0.3 ms is the OSQP solve; "
            "the rest is Python matrix assembly) does not fit a 1 kHz cycle, which "
            "is why the QP runs in the 200 Hz node and a torque servo runs at 1 kHz. "
            "Compare against the previous design, whose 62-QP authority search cost "
            "~154 ms per update."
        ),
    }


# ---------------------------------------------------------------------------
# E2 fixed box vs mapped constraints
# ---------------------------------------------------------------------------

def e2_fixed_vs_mapped(rates: RateConfig) -> dict:
    out = {}
    logs = {}
    for name, auth in (("C1_fixed_box", False), ("C2_analytic_authority", True)):
        r = lateral_step_run(auth, rates=rates)
        logs[name] = r.pop("_log")
        out[name] = r
    out["_logs"] = logs
    return out


# ---------------------------------------------------------------------------
# E3 feedforward occupancy changes the set, not the dynamics
# ---------------------------------------------------------------------------

def e3_occupancy() -> dict:
    model, data, torso, hand_sid, hr = settle_model()
    mapper = AnalyticAuthorityMapper()
    scenarios = [
        {"name": "double_support_nominal", "stance": ["left", "right"]},
        {"name": "double_support_payload", "stance": ["left", "right"], "payload_kg": 5.0},
        {"name": "double_support_arm_reach", "stance": ["left", "right"], "arm_reach": True},
        {"name": "double_support_accel_ref", "stance": ["left", "right"],
         "nominal_com_acc": [0.8, 0.0, 0.0]},
        {"name": "single_support_left", "stance": ["left"]},
    ]
    ref = None
    rows = []
    for sc in scenarios:
        ctx = scenario_context(model, data, torso, hand_sid, hr, sc)
        R = ctx["realizer"]
        req = ctx["nominal_com_acc_des"].copy()
        R.command(model, data, ctx["q_ref"], ctx["qd_ref"], req[:2], ctx["task_acc_des"],
                  ctx["hand_jac"], ctx["stance_contacts"], ctx["stance_targets"],
                  ctx["base_height_ref"], ctx["rpy"], com_acc_des=req,
                  attitude_weight=ctx["attitude_weight"],
                  centroidal_moment_des=ctx["centroidal_moment_des"])
        snap = mapper.snapshot(R, model, data, timestamp=0.0,
                               contact_mode=tuple(sc["stance"]))
        lo, hi = snap.axis_extent()
        mpc = NormalizedMPC(dim=2, dt=0.005, horizon=25, q_pos=55., q_vel=12., r=0.08)
        AB = (mpc.A.copy(), mpc.B.copy(), mpc.H.copy())
        if snap.valid:
            mpc.update_input_polytope(snap.H_body, snap.h_body)
        if ref is None:
            ref = AB
        rows.append({
            "scenario": sc["name"],
            "contact_mode": list(sc["stance"]),
            "valid": bool(snap.valid),
            "status": snap.status,
            "n_constraints": int(snap.H_body.shape[0]),
            "axis_extent_lower": [round(float(v), 3) for v in lo],
            "axis_extent_upper": [round(float(v), 3) for v in hi],
            "axis_extent_area": round(float(np.prod(np.maximum(hi - lo, 0.0))), 3),
            "min_torque_margin_nm": round(float(np.min(snap.torque_margin)), 2) if snap.valid else None,
            "min_friction_margin_n": round(float(np.min(snap.friction_margin)), 2) if snap.valid and snap.friction_margin.size else None,
            "min_normal_margin_n": round(float(np.min(snap.normal_margin)), 2) if snap.valid and snap.normal_margin.size else None,
            "canonical_A_unchanged": bool(np.array_equal(mpc.A, ref[0])),
            "canonical_B_unchanged": bool(np.array_equal(mpc.B, ref[1])),
            "canonical_hessian_unchanged": bool(np.array_equal(mpc.H, ref[2])),
        })
    return {
        "claim": ("current feedforward, payload, arm reference and contact mode all move "
                  "H_k, h_k; the canonical (A, B) and condensed Hessian never change"),
        "rows": rows,
        "all_canonical_invariant": bool(all(
            r["canonical_A_unchanged"] and r["canonical_B_unchanged"]
            and r["canonical_hessian_unchanged"] for r in rows)),
    }


# ---------------------------------------------------------------------------
# E4 contact switching: the set switches, the canonical dynamics do not
# ---------------------------------------------------------------------------

def e4_contact_switch(rates: RateConfig, use_authority: bool) -> dict:
    """DS -> left-SS -> DS on the *validated* authority-gated transition.

    Same scripted cycle, same realizer, same gate; the only thing that changes is
    where the predictor's residual-command constraint comes from:

      exact    -- the 62-QP ray-bisection query (the previous design)
      analytic -- the KKT mapping published by the 1 kHz loop (this design)

    This isolates the architecture change from the locomotion problem.
    """
    from run_authority_transition import run_cycle
    mode = use_authority if isinstance(use_authority, str) else (
        "analytic" if use_authority else "exact")
    summary, _ = run_cycle(4.0, seed=None, authority=mode)
    return {
        "variant": f"{mode}_authority",
        "authority_source": mode,
        "fell": bool(summary["fell"]),
        "qp_fallbacks": int(summary["qp_fallbacks"]),
        "lift_authorized": bool(summary["lift_authorized"]),
        "lift_time_s": summary.get("lift_authorized_time_s"),
        "single_support_s": summary.get("measured_single_support_duration_s"),
        "foot_lift_m": summary.get("max_right_foot_lift_m"),
        "max_roll_pitch_rad": summary.get("max_roll_pitch_rad"),
        "rms_planar_error_mm": summary.get("rms_planar_error_mm"),
        "max_realization_residual": summary.get("max_realization_residual_inf_mps2"),
        "authority_valid_fraction": summary.get("authority_valid_update_fraction"),
        "active_query_ms_median": summary.get("active_query_time_ms_median"),
        "active_query_whole_body_qp_solves": summary.get("active_query_solve_count_max"),
        "canonical_matrices_bitwise_invariant": summary.get("canonical_matrices_bitwise_invariant"),
    }


# ---------------------------------------------------------------------------
# E6 task port: 500 Hz predictor on body-priority remaining capacity
# ---------------------------------------------------------------------------

def e6_task_port(rates: RateConfig, *, hand_force_n: float = 5.0,
                 fallback_box: float = 10.0, duration: float = 3.0) -> dict:
    """Task port closed loop at 500 Hz under a sustained hand force.

    Two comparisons, 2x2:
      observer off/on        -- does the canonical task port reject a constant
                                interaction force offset-free?
      task authority off/on  -- does the analytic (single-active-set) task set
                                let it?

    The hand force is chosen so the required cancelling command is inside the
    port's physical authority: with Lambda_t,y ~ 1 kg a 5 N force is a ~5 m/s^2
    disturbance.  (A 12 N force needs ~12 m/s^2, beyond the command box, so the
    port saturates and cannot reject it -- a limit of the port, not a bug.)
    """
    out = {}
    for use_auth in (False, True):
        for obs in (False, True):
            # Warm-up and experiment share one controller configuration.
            model, data, torso, hand_sid, hr = settle_model(task_weight=TASK_WEIGHT)
            realizer = InverseDynamicsQPRealizer(model, exact_realizer=True)
            realizer.task_weight = TASK_WEIGHT      # the task port needs a hard-ish task row
            ctrl = MultirateInteractionController(
                realizer, rates=rates,
                body_mpc=NormalizedMPC(dim=2, dt=rates.body_dt, horizon=25,
                                       q_pos=55.0, q_vel=12.0, r=0.08),
                task_mpc=NormalizedMPC(dim=3, dt=rates.task_dt, horizon=30,
                                       q_pos=800.0, q_vel=40.0, r=0.05),
                body_obs=RandomWalkDisturbanceObserver(dim=2, dt=rates.body_dt,
                                                       q_d=0.05, r_y=1.5e-4),
                task_obs=RandomWalkDisturbanceObserver(
                    dim=3, dt=rates.task_dt, q_d=(0.04 if obs else 0.0), r_y=2.0e-4),
                mapper=AnalyticAuthorityMapper(), use_authority=use_auth,
            )
            ctrl.task_fallback_box = np.array([fallback_box] * 3)
            R = ctrl.realizer
            stance = ("left", "right")
            com0 = robot_com(model, data)
            targets = {k: p.copy() for k, (p, _) in
                       R.contact_points(model, data, stance).items()}
            hand0, _, _ = hand_state(model, data, hand_sid)
            hb = int(model.site_bodyid[hand_sid])
            qd_ref = np.zeros(model.nu)
            N = int(round(duration / rates.sim_dt))
            he, be, tr = [], [], []
            for k in range(N):
                t = k * rates.sim_dt
                com = robot_com(model, data)
                vel = com_velocity(model, data, R.root_body)
                rpy = roll_pitch_yaw_from_body(data, torso)
                hp, hv, hj = hand_state(model, data, hand_sid)
                data.xfrc_applied[:] = 0.0
                if t >= 1.0:
                    data.xfrc_applied[hb, :3] = np.array([0.0, hand_force_n, 0.0])
                ctrl.step(model, data, t,
                          q_ref=TORQUE_STAND_CTRL.copy(), qd_ref=qd_ref,
                          com_ref_acc=np.zeros(3),
                          body_error=np.r_[com[:2] - com0[:2], vel[:2]],
                          stance=stance,
                          stance_contacts=R.contact_points(model, data, stance),
                          stance_targets=targets, base_height_ref=hr, rpy=rpy,
                          hand_jac=hj, task_acc_ref=np.zeros(3),
                          task_error=np.r_[hp - hand0, hv])
                mujoco.mj_step(model, data); mujoco.mj_forward(model, data)
                hp2, _, _ = hand_state(model, data, hand_sid)
                if t > 2.0:
                    he.append(1000 * np.linalg.norm(hp2 - hand0))
                    be.append(1000 * np.linalg.norm(robot_com(model, data)[:2] - com0[:2]))
                tr.append(float(np.max(np.abs(
                    R.last_hand_jac @ R.last_qdd - R.last_task_acc_des))))
            d = ctrl.diagnostics()
            key = ("mapped" if use_auth else "fixed_box") + ("_observer" if obs else "_no_observer")
            out[key] = dict(
                task_authority=("mapped" if use_auth else "fixed_box"),
                observer=obs,
                ss_hand_error_mm=round(float(np.mean(he)), 1),
                ss_com_error_mm=round(float(np.mean(be)), 1),
                median_task_residual=round(float(np.median(tr)), 3),
                task_solves=d["task_solves"],
                whole_body_qp_per_cycle=d["timing"]["whole_body_qp_solves_per_node_update"]["max"],
                task_mpc_ms=d["timing"]["task_mpc"],
            )
    out["hand_offset_reduction_x"] = round(
        out["fixed_box_no_observer"]["ss_hand_error_mm"]
        / max(out["fixed_box_observer"]["ss_hand_error_mm"], 1e-9), 1)
    out["hand_force_N"] = hand_force_n
    out["claim"] = (
        "the canonical task port rejects a sustained hand force offset-free "
        "(observer on vs off); the analytic task authority is too conservative "
        "to permit it -- the same single-active-set conservatism that blocks "
        "single-support balance. Zero extra whole-body QP solves either way.")
    return out


def e6_task_weight_sweep() -> dict:
    """The task port only EXISTS as a port above a weight threshold."""
    model, data, torso, hand_sid, hr = settle_model()
    mapper = AnalyticAuthorityMapper()
    rows = []
    for w in (6.0, 80.0, 800.0, 2000.0, 8000.0, 30000.0):
        ctx = scenario_context(model, data, torso, hand_sid, hr,
                               {"name": "ds", "stance": ["left", "right"]})
        R = ctx["realizer"]; R.task_weight = w
        req = ctx["nominal_com_acc_des"].copy()
        R.command(model, data, ctx["q_ref"], ctx["qd_ref"], req[:2], np.zeros(3),
                  ctx["hand_jac"], ctx["stance_contacts"], ctx["stance_targets"],
                  ctx["base_height_ref"], ctx["rpy"], com_acc_des=req,
                  attitude_weight=ctx["attitude_weight"],
                  centroidal_moment_des=ctx["centroidal_moment_des"])
        Jc = np.zeros((3, R.nv))
        mujoco.mj_jacSubtreeCom(model, data, Jc, R.root_body)
        body_res = float(np.max(np.abs(Jc[:2] @ R.last_qdd - req[:2])))
        task_res = float(np.max(np.abs(
            R.last_hand_jac @ R.last_qdd - R.last_task_acc_des)))
        mapper.snapshot(R, model, data, timestamp=0.0, contact_mode=("left", "right"))
        ta = mapper.task_authority(R, model, data, np.zeros(2))
        rows.append(dict(task_weight=w,
                         body_residual=round(body_res, 4),
                         task_residual=round(task_res, 3),
                         task_set_valid=bool(ta.valid),
                         task_set_status=ta.status))
    return {"claim": ("the task port is only a faithful port above a weight "
                      "threshold; the body port is unaffected"),
            "rows": rows}


# ---------------------------------------------------------------------------
# E5 analytic mapping vs exact-QP ground truth
# ---------------------------------------------------------------------------

def e5_mapping_fidelity(n_grid: int = 21, span: float = 3.0) -> dict:
    model, data, torso, hand_sid, hr = settle_model()
    exact = ExactResidualBisectionEstimator(realization_tolerance=TOL)
    mapper = AnalyticAuthorityMapper()
    rows = []
    for sc in ({"name": "double_support_nominal", "stance": ["left", "right"]},
               {"name": "double_support_payload", "stance": ["left", "right"], "payload_kg": 5.0}):
        ctx = scenario_context(model, data, torso, hand_sid, hr, sc)
        R = ctx["realizer"]
        req = ctx["nominal_com_acc_des"].copy()

        def realize(u):
            rq = req.copy(); rq[:2] += u
            R.command(model, data, ctx["q_ref"], ctx["qd_ref"], rq[:2], ctx["task_acc_des"],
                      ctx["hand_jac"], ctx["stance_contacts"], ctx["stance_targets"],
                      ctx["base_height_ref"], ctx["rpy"], com_acc_des=rq,
                      attitude_weight=ctx["attitude_weight"],
                      centroidal_moment_des=ctx["centroidal_moment_des"])
            Jcom = np.zeros((3, R.nv))
            mujoco.mj_jacSubtreeCom(model, data, Jcom, R.root_body)
            return float(np.max(np.abs(Jcom[:2] @ R.last_qdd - rq[:2])))

        realize(np.zeros(2))
        t0 = time.perf_counter()
        snap = mapper.snapshot(R, model, data, timestamp=0.0,
                               contact_mode=tuple(sc["stance"]))
        t_analytic = (time.perf_counter() - t0) * 1e3

        realize(np.zeros(2))
        cont = ContinuationAuthorityEstimator(realization_tolerance=TOL, max_regions=60)
        t0 = time.perf_counter()
        cbox = cont.estimate(R, model, data)
        t_cont = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        box = exact.estimate(R, model, data, **{k: v for k, v in ctx.items() if k != "realizer"})
        t_exact = (time.perf_counter() - t0) * 1e3

        grid = np.linspace(-span, span, n_grid)

        def score(inside):
            fp = fn = tp = tn = 0
            for ux in grid:
                for uy in grid:
                    u = np.array([ux, uy])
                    truly = realize(u) <= TOL
                    claimed = inside(u)
                    if claimed and truly: tp += 1
                    elif claimed and not truly: fp += 1
                    elif truly: fn += 1
                    else: tn += 1
            n = tp + fp + fn + tn
            return dict(true_positive=tp, true_negative=tn,
                        false_positive=fp, false_negative=fn,
                        false_positive_rate=round(fp / n, 4),
                        false_negative_rate=round(fn / n, 4))

        in_cont = (lambda u: bool(cbox.valid and np.all(u >= cbox.lower - 1e-9)
                                  and np.all(u <= cbox.upper + 1e-9)))
        rows.append({
            "scenario": sc["name"],
            "grid": f"{n_grid}x{n_grid} over [{-span},{span}]^2",
            "analytic": dict(ms=round(t_analytic, 3), whole_body_qp=0, kkt=1,
                             **score(snap.contains)),
            "continuation": dict(ms=round(t_cont, 2), whole_body_qp=0,
                                 kkt=int(cbox.solve_count), **score(in_cont)),
            "exact_oracle": dict(ms=round(t_exact, 1), whole_body_qp=62,
                                 box_lower=[round(float(v), 3) for v in box.lower],
                                 box_upper=[round(float(v), 3) for v in box.upper]),
            "continuation_box_lower": [round(float(v), 3) for v in cbox.lower],
            "continuation_box_upper": [round(float(v), 3) for v in cbox.upper],
            "continuation_vs_oracle_max_boundary_error": round(float(np.max(np.abs(
                np.r_[cbox.lower, cbox.upper] - np.r_[box.lower, box.upper]))), 3),
            "speedup_continuation_over_oracle": round(t_exact / max(t_cont, 1e-9), 1),
            "note": ("false positives are the safety-relevant error. The 1-cell analytic "
                     "map is sound but refuses ~68% of feasible commands; PWA continuation "
                     "recovers them with zero extra whole-body QP solves."),
        })
    return {"claim": "analytic mapping graded against exact-QP bisection", "rows": rows}


# ---------------------------------------------------------------------------

def main():
    rates = RateConfig(servo_dt=0.001, node_dt=0.005, authority_dt=0.020)
    out = {"rates": {"realizer_hz": 1000, "body_mpc_hz": int(1 / rates.body_dt),
                     "task_mpc_hz": int(1 / rates.task_dt)},
           "realization_tolerance_mps2": TOL}

    print("E1 real-time budget ...")
    out["E1_realtime"] = e1_realtime(rates)
    t = out["E1_realtime"]
    print("   node %d Hz: median %.2f ms (budget %.1f ms), p99 %.2f, misses %.1f%% | "
          "servo %d Hz, %d ticks | QP/node %d"
          % (t["node_hz"], t["node_ms"]["median_ms"], t["node_budget_ms"],
             t["node_ms"]["p99_ms"], 100 * t["node_deadline_miss_fraction"],
             t["servo_hz"], t["servo_ticks"],
             t["whole_body_qp_solves_per_node_update"]["max"]))

    print("E2 fixed box vs mapped authority ...")
    e2 = e2_fixed_vs_mapped(rates)
    logs = e2.pop("_logs")
    out["E2_fixed_vs_mapped"] = e2
    for k, v in e2.items():
        print("   %-24s RMS %6.2f mm | median res %.3f | max res %.3f | >tol %.1f%% | fell=%s"
              % (k, v["rms_planar_error_mm"], v["median_residual"], v["max_residual"],
                 100 * v["residual_over_tol_fraction"], v["fell"]))

    print("E3 feedforward occupancy ...")
    out["E3_occupancy"] = e3_occupancy()
    for r in out["E3_occupancy"]["rows"]:
        print("   %-26s valid=%-5s reach=%s..%s area=%.2f"
              % (r["scenario"], r["valid"], r["axis_extent_lower"], r["axis_extent_upper"],
                 r["axis_extent_area"]))
    print("   canonical (A,B,Hessian) invariant across all:",
          out["E3_occupancy"]["all_canonical_invariant"])

    print("E4 contact switching (validated authority-gated DS->SS->DS) ...")
    e4 = {}
    for auth in ("exact", "analytic", "continuation"):
        r = e4_contact_switch(rates, auth)
        e4[r["variant"]] = r
        print("   %-18s fell=%-5s lift@%-5s SS=%.2fs lift=%.3fm | query %7.2f ms, %d extra WB-QP | canonical inv=%s"
              % (r["variant"], r["fell"], r["lift_time_s"], r["single_support_s"] or 0,
                 r["foot_lift_m"] or 0, r["active_query_ms_median"] or 0,
                 r["active_query_whole_body_qp_solves"] or 0,
                 r["canonical_matrices_bitwise_invariant"]))
    out["E4_contact_switch"] = e4

    print("E6 task port (500 Hz, body-priority remaining capacity) ...")
    out["E6_task_weight_sweep"] = e6_task_weight_sweep()
    for r in out["E6_task_weight_sweep"]["rows"]:
        print("   w=%5.0f  body res %.4f  task res %6.3f  task set: %s"
              % (r["task_weight"], r["body_residual"], r["task_residual"],
                 "valid" if r["task_set_valid"] else r["task_set_status"]))
    out["E6_task_port"] = e6_task_port(rates)
    e6 = out["E6_task_port"]

    print("E5 analytic vs exact mapping fidelity ...")
    out["E5_mapping_fidelity"] = e5_mapping_fidelity()
    for r in out["E5_mapping_fidelity"]["rows"]:
        a, c, e = r["analytic"], r["continuation"], r["exact_oracle"]
        print("   %-24s analytic  FP %.1f%% FN %5.1f%%  %6.2f ms  0 QP" %
              (r["scenario"], 100 * a["false_positive_rate"],
               100 * a["false_negative_rate"], a["ms"]))
        print("   %-24s CONTINU.  FP %.1f%% FN %5.1f%%  %6.2f ms  0 QP, %d KKT  (err vs oracle %.3f)" %
              ("", 100 * c["false_positive_rate"], 100 * c["false_negative_rate"],
               c["ms"], c["kkt"], r["continuation_vs_oracle_max_boundary_error"]))
        print("   %-24s oracle                       %6.1f ms  62 QP" % ("", e["ms"]))

    with (RESULTS / "multirate_benchmark.json").open("w") as f:
        json.dump(out, f, indent=2)

    # figure: E2 residual + command traces
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for name, lg in logs.items():
        ax[0].plot(lg["t"], lg["res"], label=name)
        ax[1].plot(lg["t"], lg["u"][:, 1], label=name)
        ax[2].plot(lg["t"], 1000 * lg["ey"], label=name)
    ax[0].axhline(TOL, ls="--", c="k", lw=1, label="tolerance")
    ax[0].set_title("realization residual [m/s$^2$]")
    ax[1].set_title("body command $u_y$ [m/s$^2$]")
    ax[2].set_title("lateral CoM error [mm]")
    for a in ax:
        a.set_xlabel("t [s]"); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "multirate_e2.png", dpi=160)
    print("\nsaved: results/multirate_benchmark.json, figures/multirate_e2.png")


if __name__ == "__main__":
    main()
