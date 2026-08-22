"""
Experimental Level-1 centroidal balance controller for the toy biped.

This is a small self-contained GRF allocator, not a production humanoid walking
stack.  At each control tick it computes a desired centroidal wrench from CoM
and torso attitude feedback, solves a constrained contact-force QP, and maps
the resulting foot forces into joint torques through the foot Jacobians.
"""
from __future__ import annotations

import numpy as np
import osqp
import scipy.sparse as sp

from wbc_core import get_robot_com, get_site_jacobian


def _roll_pitch_from_xmat(xmat_flat):
    R = np.asarray(xmat_flat).reshape(3, 3)
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    return float(roll), float(pitch)


class Level1CentroidalBalance:
    """Instantaneous centroidal wrench QP with foot-force constraints."""

    def __init__(
        self,
        model,
        ids,
        foot_site_keys=("lfoot", "rfoot"),
        foot_names=("left", "right"),
        mu=0.8,
        fz_min=5.0,
        fz_max=650.0,
    ):
        self.model = model
        self.ids = ids
        self.foot_site_keys = foot_site_keys
        self.foot_names = foot_names
        self.mu = float(mu)
        self.fz_min = float(fz_min)
        self.fz_max = float(fz_max)
        self.mass = float(np.sum(model.body_mass[1:]))

        self.kp_com = np.array([120.0, 520.0, 900.0])
        self.kd_com = np.array([90.0, 180.0, 240.0])
        self.kp_rp = np.array([900.0, 520.0])
        self.kd_rp = np.array([120.0, 80.0])
        self.w_wrench = np.diag([1.0, 1.4, 2.0, 0.45, 0.30, 0.05])
        self.rho_force = 1e-4
        self._last_forces = {name: np.zeros(3) for name in foot_names}

    def solve(self, data, active=(True, True), com_ref=None, comd_ref=None):
        active = tuple(bool(x) for x in active)
        if not any(active):
            return np.zeros(0), {"active": active, "status": "no_contact"}

        com, comv = get_robot_com(self.model, data)
        if com_ref is None:
            com_ref = com.copy()
            com_ref[2] = 0.88
        if comd_ref is None:
            comd_ref = np.zeros(3)

        acc = self.kp_com * (com_ref - com) + self.kd_com * (comd_ref - comv)
        force_des = self.mass * (acc + np.array([0.0, 0.0, 9.81]))

        roll, pitch = _roll_pitch_from_xmat(data.xmat[self.ids["torso_body"]])
        omega = data.qvel[3:6]
        moment_des = np.array([
            -self.kp_rp[0] * roll - self.kd_rp[0] * omega[0],
            -self.kp_rp[1] * pitch - self.kd_rp[1] * omega[1],
            -8.0 * omega[2],
        ])
        wrench_des = np.concatenate([force_des, moment_des])

        foot_positions = []
        names = []
        for use, key, name in zip(active, self.foot_site_keys, self.foot_names):
            if use:
                foot_positions.append(data.site_xpos[self.ids[key]].copy())
                names.append(name)

        n = len(foot_positions)
        A = np.zeros((6, 3 * n))
        for i, p in enumerate(foot_positions):
            r = p - com
            A[:3, 3 * i:3 * i + 3] = np.eye(3)
            A[3:, 3 * i:3 * i + 3] = np.array([
                [0.0, -r[2], r[1]],
                [r[2], 0.0, -r[0]],
                [-r[1], r[0], 0.0],
            ])

        fnom = np.zeros(3 * n)
        fz_nom = self.mass * 9.81 / n
        for i, name in enumerate(names):
            fnom[3 * i:3 * i + 3] = self._last_forces.get(
                name, np.array([0.0, 0.0, fz_nom])
            )
            fnom[3 * i + 2] = max(fnom[3 * i + 2], fz_nom)

        H = A.T @ self.w_wrench @ A + self.rho_force * np.eye(3 * n)
        q = -(A.T @ self.w_wrench @ wrench_des + self.rho_force * fnom)

        rows = []
        lb = []
        ub = []
        for i in range(n):
            ix = 3 * i
            # fz bounds.
            row = np.zeros(3 * n)
            row[ix + 2] = 1.0
            rows.append(row); lb.append(self.fz_min); ub.append(self.fz_max)
            # Linearized friction pyramid: |fx|, |fy| <= mu fz.
            for axis in (0, 1):
                row = np.zeros(3 * n)
                row[ix + axis] = 1.0
                row[ix + 2] = -self.mu
                rows.append(row); lb.append(-np.inf); ub.append(0.0)
                row = np.zeros(3 * n)
                row[ix + axis] = -1.0
                row[ix + 2] = -self.mu
                rows.append(row); lb.append(-np.inf); ub.append(0.0)

        prob = osqp.OSQP()
        prob.setup(
            P=sp.triu(sp.csc_matrix(H), format="csc"),
            q=q,
            A=sp.csc_matrix(np.vstack(rows)),
            l=np.asarray(lb),
            u=np.asarray(ub),
            verbose=False,
            polishing=False,
            eps_abs=1e-5,
            eps_rel=1e-5,
            max_iter=200,
        )
        res = prob.solve(raise_error=False)
        if res.info.status_val not in (1, 2):
            f = fnom
            status = res.info.status
        else:
            f = np.asarray(res.x)
            status = res.info.status

        forces = {}
        for i, name in enumerate(names):
            forces[name] = f[3 * i:3 * i + 3].copy()
            self._last_forces[name] = forces[name]
        return f, {
            "active": active,
            "names": names,
            "forces": forces,
            "wrench_des": wrench_des,
            "wrench": A @ f,
            "status": status,
            "com": com,
            "com_ref": com_ref,
            "roll_pitch": np.array([roll, pitch]),
        }

    def torques(self, data, active=(True, True), com_ref=None, comd_ref=None):
        _, info = self.solve(data, active=active, com_ref=com_ref, comd_ref=comd_ref)
        tau = np.zeros(self.model.nv)
        for name, key in zip(self.foot_names, self.foot_site_keys):
            if name not in info.get("forces", {}):
                continue
            J = get_site_jacobian(self.model, data, self.ids[key])
            # MuJoCo qfrc_applied sign convention: a positive contact force on
            # the robot is represented as J^T f in generalized coordinates.
            tau += J.T @ info["forces"][name]
        return tau, info
