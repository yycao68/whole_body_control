"""Stage 2, increment 1 — WBC realizability gate on the frozen platform.

Chosen coupling ("record reference, keep WBC+ID"): the frozen policy walk
(`reference/frozen_walk_seed0.npz`) is the nominal reference; a whole-body
inverse-dynamics/contact QP (the paper's Eq. 12, sized to the 12-DoF G1) tracks
it. This module runs the WBC ALONE (no Interaction Dynamics) to answer the open
Stage-2 risk: can the WBC track the recorded gait and stay upright, or does it
hit the same single-support realizability wall the in-house reference did?

No policy is in the loop here — the recorded trajectory is the reference.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import mujoco
import osqp
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
SCENE = HERE / "g1_description" / "scene.xml"
REF = HERE / "reference" / "frozen_walk_seed0.npz"

LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
# Foot corner contact offsets in the ankle_roll_link frame (from the model's
# 4 sole spheres): heel-in/out and toe-in/out at sole height z=-0.03.
FOOT_CORNERS = np.array([
    [-0.05, 0.025, -0.03], [-0.05, -0.025, -0.03],
    [0.12, 0.03, -0.03], [0.12, -0.03, -0.03],
])
SIM_DT = 0.002


def bid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def sid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def jid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def quat_to_rpy(q):
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


class RecordedReference:
    """Replays the frozen policy walk as the nominal reference."""

    def __init__(self, path=REF):
        d = np.load(path)
        self.t = d["t"]
        self.com = d["com"]
        self.base_pos = d["base_pos"]
        self.base_linvel = d["base_linvel"]
        self.base_quat = d["base_quat"]
        self.base_angvel = d["base_angvel"]
        self.qj = d["qj"]
        self.dqj = d["dqj"]
        self.lfoot = d["lfoot"]
        self.rfoot = d["rfoot"]
        self.contact = d["contact"]
        # The policy's joint TARGET (action*0.25 + default) leads the actual joint
        # position by the PD offset that produces the holding/driving torque.
        # Tracking recorded positions gives ~0 torque and collapse; the target is
        # the reference that actually reproduces the gait.
        default = np.array([-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0])
        self.qj_target = d["action"] * 0.25 + default
        self.dt = float(self.t[1] - self.t[0])
        # CoM velocity/acceleration by finite difference of the recorded CoM.
        self.vcom = np.gradient(self.com, self.dt, axis=0)
        self.acom = np.gradient(self.vcom, self.dt, axis=0)
        # recorded joint acceleration (feedforward for posture tracking)
        self.ddqj = np.gradient(self.dqj, self.dt, axis=0)

    def idx(self, t):
        return int(np.clip(round(t / self.dt), 0, len(self.t) - 1))

    def sample(self, t):
        k = self.idx(t)
        stance = tuple(f for f, c in (("left", self.contact[k, 0]),
                                      ("right", self.contact[k, 1])) if c)
        if not stance:                       # flight (rare); hold both as target
            stance = ("left", "right")
        swing = None
        if len(stance) == 1:
            swing = "right" if stance == ("left",) else "left"
        # stance-foot centroid (the frame in which lateral CoM balance is stable)
        feet = {"left": self.lfoot[k], "right": self.rfoot[k]}
        pstance = np.mean([feet[f] for f in stance], axis=0)
        return {
            "k": k, "com": self.com[k], "vcom": self.vcom[k], "acom": self.acom[k],
            "rpy": quat_to_rpy(self.base_quat[k]), "angvel": self.base_angvel[k],
            "qj": self.qj[k], "dqj": self.dqj[k], "ddqj": self.ddqj[k],
            "qj_target": self.qj_target[k],
            "stance": stance, "swing": swing, "pstance": pstance,
            "lfoot": self.lfoot[k], "rfoot": self.rfoot[k],
        }


class WBC12:
    """12-DoF whole-body inverse-dynamics/contact QP (paper Eq. 12).

    Soft-tracks CoM, base attitude, swing foot, contact no-acceleration, and
    reference posture as weighted least squares on generalized acceleration;
    enforces floating-base dynamics, friction pyramid, and torque limits as hard
    constraints. Torque is eliminated via the actuated dynamics rows.
    """

    def __init__(self, model):
        self.model = model
        self.nv = model.nv
        self.dof = np.array([model.jnt_dofadr[jid(model, n)] for n in LEG_JOINTS])
        self.qadr = np.array([model.jnt_qposadr[jid(model, n)] for n in LEG_JOINTS])
        self.act = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                             for n in LEG_JOINTS])
        # These motors are unlimited in actuator_ctrlrange; the real torque
        # limits live on the joints' actuatorfrcrange (±88/±139/±50).
        jidx = np.array([jid(model, n) for n in LEG_JOINTS])
        self.tau_min = model.jnt_actfrcrange[jidx, 0].copy()
        self.tau_max = model.jnt_actfrcrange[jidx, 1].copy()
        self.pelvis = bid(model, "pelvis")
        self.foot_body = {"left": bid(model, "left_ankle_roll_link"),
                          "right": bid(model, "right_ankle_roll_link")}
        self.foot_site = {"left": sid(model, "left_foot"), "right": sid(model, "right_foot")}
        self.mu = 0.7
        self.fz_max = 900.0
        # objective weights
        # Posture-dominant: the recorded gait is self-stabilizing on its own
        # trajectory, so track recorded joints + swing feet tightly and use
        # CoM/attitude only as light anchors against base drift.
        self.w_com = 25.0
        self.w_att = 40.0
        self.w_swing = 80.0
        self.w_post = 150.0
        self.w_freg = 1e-4
        self.w_tau = 0.02          # minimize joint torque to resolve contact redundancy
        # task feedback gains
        self.kp_com, self.kd_com = 60.0, 16.0
        self.kp_att, self.kd_att = 120.0, 22.0
        self.kp_sw, self.kd_sw = 300.0, 35.0
        self.kp_j, self.kd_j = 80.0, 6.0
        self._prev = {}                       # for finite-difference Jdot*qd
        self.last_status = "none"
        self.last_tau = np.zeros(12)

    # -- kinematics helpers ------------------------------------------------
    def _com_jac(self, data):
        J = np.zeros((3, self.nv))
        mujoco.mj_jacSubtreeCom(self.model, data, J, self.pelvis)
        return J

    def _site_jac(self, data, s):
        jp = np.zeros((3, self.nv)); jr = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.model, data, jp, jr, s)
        return jp

    def _site_jac6(self, data, s):
        jp = np.zeros((3, self.nv)); jr = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.model, data, jp, jr, s)
        return np.vstack([jp, jr])

    def _point_jac(self, data, pos, body):
        jp = np.zeros((3, self.nv)); jr = np.zeros((3, self.nv))
        mujoco.mj_jac(self.model, data, jp, jr, pos, body)
        return jp

    def _bias(self, key, J, qd):
        """Jdot*qd via one-step finite difference of the Jacobian."""
        prev = self._prev.get(key)
        self._prev[key] = J.copy()
        if prev is None or prev.shape != J.shape:
            return np.zeros(J.shape[0])
        return (J - prev) / SIM_DT @ qd

    def _contact_points(self, data, stance):
        pts = []
        for foot in stance:
            b = self.foot_body[foot]
            R = data.xmat[b].reshape(3, 3)
            for c in FOOT_CORNERS:
                pts.append((data.xpos[b] + R @ c, b))
        return pts

    # -- one QP solve ------------------------------------------------------
    def command(self, data, ref, com_corr=np.zeros(3)):
        model = self.model
        nv = self.nv
        qd = data.qvel.copy()
        M = np.zeros((nv, nv))
        try:
            mujoco.mj_fullM(model, M, data.qM)
        except TypeError:
            mujoco.mj_fullM(model, data, M)
        h = data.qfrc_bias.copy()

        stance = ref["stance"]
        pts = self._contact_points(data, stance)
        ncp = len(pts)
        nf = 3 * ncp
        n = nv + nf                            # z = [qdd, f]

        P = np.zeros((n, n)); q = np.zeros(n)

        def add_ls(C, target, w):              # weighted ||C z - target||^2 over qdd block
            if w == 0:
                return
            P[:nv, :nv] += w * C.T @ C
            q[:nv] += -w * C.T @ target

        # CoM tracking, in the stance-foot frame (the lateral-balance variable).
        # Stance feet are planted (hard contact), so d^2/dt^2(com - p_stance) =
        # J_com qdd and only the error term is measured relative to the foot.
        com = (model.body_mass[1:, None] * data.xipos[1:]).sum(0) / model.body_mass[1:].sum()
        Jc = self._com_jac(data)
        vcom = Jc @ qd
        pstance = np.mean([data.xpos[self.foot_body[f]] for f in stance], axis=0)
        e_com = (com - pstance) - (ref["com"] - ref["pstance"])
        a_com = ref["acom"] + self.kp_com * (-e_com) + self.kd_com * (ref["vcom"] - vcom) + com_corr
        add_ls(Jc, a_com - self._bias("com", Jc, qd), self.w_com)

        # base attitude (angular dofs of the free joint are qd[3:6])
        rpy = quat_to_rpy(data.qpos[3:7])
        Jatt = np.zeros((3, nv)); Jatt[:, 3:6] = np.eye(3)
        a_att = self.kp_att * (ref["rpy"] - rpy) + self.kd_att * (ref["angvel"] - qd[3:6])
        add_ls(Jatt, a_att, self.w_att)

        # swing foot
        if ref["swing"] is not None:
            fb = self.foot_body[ref["swing"]]
            fpos = data.xpos[fb]
            Jsw = self._point_jac(data, fpos, fb)
            vsw = Jsw @ qd
            tgt = ref["lfoot"] if ref["swing"] == "left" else ref["rfoot"]
            vtgt = (tgt - self._prev.get("swtgt", tgt)) / SIM_DT
            self._prev["swtgt"] = tgt.copy()
            a_sw = self.kp_sw * (tgt - fpos) + self.kd_sw * (vtgt - vsw)
            add_ls(Jsw, a_sw - self._bias("sw", Jsw, qd), self.w_swing)

        # posture regularization toward the recorded joint reference
        # Drive posture toward the policy's joint TARGET (leads the recorded
        # position), reproducing the holding torque the recorded PD gait used.
        qj = data.qpos[self.qadr]; dqj = qd[self.dof]
        a_post = self.kp_j * (ref["qj_target"] - qj) - self.kd_j * dqj
        Cpost = np.zeros((12, nv)); Cpost[np.arange(12), self.dof] = 1.0
        add_ls(Cpost, a_post, self.w_post)

        # contact forces at the 4 corners of each stance foot (friction + support)
        Jcontact = np.zeros((nf, nv))
        for i, (pos, body) in enumerate(pts):
            Jcontact[3 * i:3 * i + 3] = self._point_jac(data, pos, body)

        # force regularization (keep forces small/even)
        P[nv:, nv:] += self.w_freg * np.eye(nf)

        # torque map: tau = M[act] qdd + h[act] - Jc^T[act] f = Atau z + h[act]
        Atau = np.zeros((12, n))
        Atau[:, :nv] = M[self.dof, :]
        Atau[:, nv:] = -Jcontact.T[self.dof, :]
        # Resolve the redundant contact-force null space by MINIMIZING joint
        # torque (the physical inverse-dynamics solution), not ||f||. Minimizing
        # ||f|| gave a non-physical force split and wildly wrong torques.
        P += self.w_tau * (Atau.T @ Atau)
        q += self.w_tau * (Atau.T @ h[self.dof])

        # ---- hard constraints ----
        A_rows = []; l = []; u = []
        # floating-base dynamics: (M qdd + h - Jcontact^T f)[0:6] = 0
        Abase = np.zeros((6, n))
        Abase[:, :nv] = M[:6, :]
        Abase[:, nv:] = -Jcontact.T[:6, :]
        A_rows.append(Abase); l.append(-h[:6]); u.append(-h[:6])

        # HARD rigid contact: each stance foot has zero 6-D spatial acceleration
        # (J_foot qdd + Jdot qd = 0), so the base cannot fall through planted feet.
        for foot in stance:
            Jf = self._site_jac6(data, self.foot_site[foot])
            Afoot = np.zeros((6, n)); Afoot[:, :nv] = Jf
            b = -self._bias(f"foot_{foot}", Jf, qd)
            A_rows.append(Afoot); l.append(b); u.append(b)

        # torque limits: tau in [tau_min, tau_max]
        A_rows.append(Atau); l.append(self.tau_min - h[self.dof]); u.append(self.tau_max - h[self.dof])

        # friction pyramid per contact: fz>=0, -mu fz <= fx <= mu fz, same for fy
        big = 1e6
        for i in range(ncp):
            base = nv + 3 * i
            for ax in (0, 1):                  # |f_ax| <= mu fz
                r1 = np.zeros(n); r1[base + ax] = 1.0; r1[base + 2] = -self.mu
                A_rows.append(r1[None, :]); l.append([-big]); u.append([0.0])
                r2 = np.zeros(n); r2[base + ax] = -1.0; r2[base + 2] = -self.mu
                A_rows.append(r2[None, :]); l.append([-big]); u.append([0.0])
            rz = np.zeros(n); rz[base + 2] = 1.0
            A_rows.append(rz[None, :]); l.append([0.0]); u.append([self.fz_max])

        A = np.vstack(A_rows)
        l = np.concatenate([np.atleast_1d(x) for x in l])
        u = np.concatenate([np.atleast_1d(x) for x in u])

        P = 0.5 * (P + P.T) + 1e-8 * np.eye(n)
        prob = osqp.OSQP()
        prob.setup(sp.csc_matrix(P), q, sp.csc_matrix(A), l, u,
                   verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=4000,
                   polish=False)
        res = prob.solve()
        self.last_status = res.info.status
        if res.x is None or not np.all(np.isfinite(res.x)):
            self.last_tau = np.zeros(12)
            return np.zeros(12), False
        z = res.x
        qdd = z[:nv]
        f = z[nv:]
        tau = (M[self.dof, :] @ qdd + h[self.dof] - Jcontact.T[self.dof, :] @ f)
        tau = np.clip(tau, self.tau_min, self.tau_max)
        self.last_tau = tau
        self.last_qdd = qdd
        self.last_f = f
        return tau, True


def run(duration=20.0, seed=0, settle=0.4, verbose=False):
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    ref = RecordedReference()
    wbc = WBC12(model)

    # Start at a recorded double-support instant so the feet are on the ground
    # and the initial MuJoCo state is physically consistent with the reference.
    ds = np.where((ref.contact[:, 0] == 1) & (ref.contact[:, 1] == 1) & (ref.t >= 0.6))[0]
    k0 = int(ds[0]) if len(ds) else ref.idx(0.6)
    t0 = k0 * ref.dt
    data.qpos[:3] = ref.base_pos[k0] if hasattr(ref, "base_pos") else [0, 0, 0.793]
    data.qpos[3:7] = ref.base_quat[k0]
    data.qpos[7:19] = ref.qj[k0]
    data.qvel[:3] = ref.base_linvel[k0]
    data.qvel[3:6] = ref.base_angvel[k0]
    data.qvel[6:18] = ref.dqj[k0]
    rng = np.random.default_rng(seed)
    data.qvel[6:] += rng.normal(0, 2e-4, size=model.nv - 6)
    mujoco.mj_forward(model, data)

    n = int(round((duration - t0) / SIM_DT))
    com_err = np.zeros((n, 3))
    fell_at = None
    nfail = 0
    for k in range(n):
        t = t0 + k * SIM_DT
        r = ref.sample(t)
        tau, ok = wbc.command(data, r)
        if not ok:
            nfail += 1
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        com = (model.body_mass[1:, None] * data.xipos[1:]).sum(0) / model.body_mass[1:].sum()
        com_err[k] = com - r["com"]
        up = 1 - 2 * (data.qpos[3] ** 2 + data.qpos[6] ** 2)
        if fell_at is None and (data.qpos[2] < 0.45 or up > -0.5):
            fell_at = t
            if verbose:
                print(f"  fell at {t:.2f}s (h={data.qpos[2]:.2f}, up={up:.2f})")
            break
    surv = duration if fell_at is None else fell_at
    used = k + 1
    rms = float(np.sqrt(np.mean(np.sum(com_err[:used, :2] ** 2, axis=1)))) * 1000
    return {"survived": surv, "fell": fell_at is not None, "com_rms_mm": rms,
            "qp_fail": nfail, "duration": duration}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    n_ok = 0
    for seed in args.seeds:
        r = run(args.duration, seed=seed, verbose=args.verbose)
        ok = (not r["fell"])
        n_ok += ok
        print(f"seed {seed}: fell={r['fell']} survived={r['survived']:5.2f}/{r['duration']:.0f}s "
              f"CoM_xy_RMS={r['com_rms_mm']:.1f}mm qp_fail={r['qp_fail']}", flush=True)
    print(f"\nWBC-only upright full {args.duration:.0f}s: {n_ok}/{len(args.seeds)}")


if __name__ == "__main__":
    main()
