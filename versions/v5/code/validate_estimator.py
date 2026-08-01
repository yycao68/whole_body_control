"""Step V1 — external-wrench estimator vs ground truth.

Runs the policy (ID correction OFF, so the estimator is tested in isolation) and
logs the true applied force vs the estimated F_ext = m*c_ddot_xy - GRF_xy across:
nominal walking, turning, diagonal (lateral) walking, step-up, step-down, a known
transient impulse, and a known sustained force. Reports RMSE, peak error,
detection delay, decay time, and false-positive rate. Process noise = 0 so the
only "external force" is the deliberate one (isolates estimation from the
noise-as-force confound).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import mujoco
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stage2_id_on_policy as S

HERE = Path(__file__).resolve().parent
F_THRESH = S.IDResidual("id_mpc", 0.02).f_thresh   # 3.0 N, the controller's gate


def run_scenario(force_fn, cmd_fn, duration=8.0, scene=None, init_z=0.793, seed=0,
                 observer=False):
    """observer=False: F_ext = lowpass(m*c_ddot - GRF) (raw CoM-velocity diff).
    observer=True: momentum-residual observer  F_hat = K*(l - int(GRF + F_hat)),
    l = m*c_dot — no acceleration differentiation."""
    model = mujoco.MjModel.from_xml_path(str(scene or S.SCENE)); model.opt.timestep = S.SIM_DT
    data = mujoco.MjData(model)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    mass = float(model.body_mass[1:].sum())
    data.qpos[:3] = [0, 0, init_z]; data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:19] = S.DEFAULT_ANGLES
    rng = np.random.default_rng(seed)
    data.qvel[6:] += rng.normal(0, 2e-4, size=model.nv - 6)
    mujoco.mj_forward(model, data)
    policy = torch.jit.load(str(S.POLICY))

    def com_vel():
        Jc = np.zeros((3, model.nv)); mujoco.mj_jacSubtreeCom(model, data, Jc, pelvis)
        return Jc @ data.qvel

    n = int(round(duration / S.SIM_DT))
    action = np.zeros(S.NUM_ACTIONS, np.float32); target = S.DEFAULT_ANGLES.copy()
    obs = np.zeros(S.NUM_OBS, np.float32); counter = 0; settle = 0.5
    vcom_prev = None; f_ext = np.zeros(2); a_ext = 1.0 - np.exp(-2 * np.pi * 3.0 * S.SIM_DT)
    mom_int = np.zeros(2); K_obs = 20.0        # observer gain (~3 Hz bandwidth)
    T, Ftrue, Fest, fell = [], [], [], None
    for k in range(n):
        t = k * S.SIM_DT
        tau = S.pd_control(target if t >= settle else S.DEFAULT_ANGLES,
                           data.qpos[7:], S.KPS, data.qvel[6:], S.KDS)
        data.xfrc_applied[:] = 0.0
        f_applied = np.array(force_fn(t)) if t >= settle else np.zeros(2)
        data.xfrc_applied[pelvis, :2] = f_applied
        data.ctrl[:] = tau
        mujoco.mj_step(model, data); counter += 1
        vcom = com_vel()
        grf = S.grf_world(model, data)[:2]
        if observer:
            l = mass * vcom[:2]                 # linear momentum (no differentiation)
            f_ext = K_obs * (l - mom_int)
            mom_int += S.SIM_DT * (grf + f_ext)
        else:
            cddot = np.zeros(3) if vcom_prev is None else (vcom - vcom_prev) / S.SIM_DT
            vcom_prev = vcom.copy()
            f_ext = (1 - a_ext) * f_ext + a_ext * (mass * cddot[:2] - grf)
        if t >= settle and counter % S.CONTROL_DECIMATION == 0:
            qj = (data.qpos[7:] - S.DEFAULT_ANGLES) * S.DOF_POS_SCALE
            dqj = data.qvel[6:] * S.DOF_VEL_SCALE
            grav = S.gravity_orientation(data.qpos[3:7]); om = data.qvel[3:6] * S.ANG_VEL_SCALE
            ph = (counter * S.SIM_DT) % S.GAIT_PERIOD / S.GAIT_PERIOD
            cmd = np.array(cmd_fn(t))
            obs[:3] = om; obs[3:6] = grav; obs[6:9] = cmd * S.CMD_SCALE
            obs[9:21] = qj; obs[21:33] = dqj; obs[33:45] = action
            obs[45:47] = [np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)]
            action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
            target = action * S.ACTION_SCALE + S.DEFAULT_ANGLES
        T.append(t); Ftrue.append(f_applied.copy()); Fest.append(f_ext.copy())
        if fell is None and t >= settle and (data.qpos[2] < 0.45
                                             or S.gravity_orientation(data.qpos[3:7])[2] > -0.5):
            fell = t
    return (np.array(T), np.array(Ftrue), np.array(Fest), fell)


def metrics(T, Ftrue, Fest, has_force, onset=None, offset=None, fell=None):
    # evaluate only while UPRIGHT (a fall has genuinely-large unaccounted accel,
    # so the estimator "error" there is not an estimator flaw)
    hi = (fell - 0.2) if fell is not None else T[-1] + 1
    m = (T >= 0.6) & (T < hi)
    est_mag = np.linalg.norm(Fest, axis=1)
    err = np.linalg.norm(Fest - Ftrue, axis=1)
    out = {"rmse_N": float(np.sqrt(np.mean(err[m] ** 2))),
           "peak_err_N": float(err[m].max())}
    if not has_force:
        out["false_pos_rate"] = float(np.mean(est_mag[m] > F_THRESH)) if m.any() else float("nan")
    else:
        # detection delay: onset -> |Fest| first crosses threshold
        dd = None
        idx = np.where((T >= onset) & (est_mag > F_THRESH))[0]
        if len(idx): dd = float(T[idx[0]] - onset)
        out["detect_delay_s"] = dd
        if offset is not None:                       # transient: decay after offset
            dc = None
            after = np.where(T >= offset)[0]
            for j in after:
                if est_mag[j] < F_THRESH:
                    dc = float(T[j] - offset); break
            out["decay_s"] = dc
        else:                                        # sustained: track error during hold
            hold = (T >= onset + 0.5) & (T <= (T[-1] - 1.0))
            out["track_rmse_N"] = float(np.sqrt(np.mean(err[hold] ** 2)))
    return out


def main():
    step_up, iz_u = S.terrain_scene("up", 0.03)
    step_dn, iz_d = S.terrain_scene("down", 0.03)
    scenarios = {
        "nominal":      dict(force=lambda t: (0, 0), cmd=lambda t: (0.5, 0, 0), has_force=False),
        "turning":      dict(force=lambda t: (0, 0), cmd=lambda t: (0.3, 0, 0.4), has_force=False),
        "diagonal":     dict(force=lambda t: (0, 0), cmd=lambda t: (0.3, 0.3, 0), has_force=False),
        "step_up_30":   dict(force=lambda t: (0, 0), cmd=lambda t: (0.5, 0, 0), has_force=False,
                             scene=step_up, iz=iz_u),
        "step_down_30": dict(force=lambda t: (0, 0), cmd=lambda t: (0.5, 0, 0), has_force=False,
                             scene=step_dn, iz=iz_d),
        "impulse_120":  dict(force=lambda t: (0, 120 if 3.0 <= t < 3.15 else 0),
                             cmd=lambda t: (0.5, 0, 0), has_force=True, onset=3.0, offset=3.15),
        "sustained_12": dict(force=lambda t: (0, 12 if 3.0 <= t < 6.0 else 0),
                             cmd=lambda t: (0.5, 0, 0), has_force=True, onset=3.0),
    }
    results = {}; traces = {}
    def fmt(v): return float("nan") if v is None else v
    print(f"{'scenario':14s}{'fell':>6s}{'RMSE(N)':>9s}{'peak_err':>9s}{'FP_rate':>9s}"
          f"{'detect_s':>9s}{'decay_s':>9s}{'trk_rmse':>9s}")
    for name, cfg in scenarios.items():
        T, Ft, Fe, fell = run_scenario(cfg["force"], cfg["cmd"], scene=cfg.get("scene"),
                                       init_z=cfg.get("iz", 0.793))
        mt = metrics(T, Ft, Fe, cfg["has_force"], cfg.get("onset"), cfg.get("offset"), fell)
        mt["fell"] = None if fell is None else round(fell, 2)
        results[name] = mt; traces[name] = (T, Ft[:, 1], Fe[:, 1])
        print(f"{name:14s}{str(mt['fell']):>6s}{mt['rmse_N']:9.1f}{mt['peak_err_N']:9.1f}"
              f"{fmt(mt.get('false_pos_rate'))*100 if mt.get('false_pos_rate') is not None else float('nan'):8.1f}%"
              f"{fmt(mt.get('detect_delay_s')):9.2f}{fmt(mt.get('decay_s')):9.2f}"
              f"{fmt(mt.get('track_rmse_N')):9.1f}", flush=True)

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "estimator_validation.json").write_text(json.dumps(results, indent=2))

    keys = ["nominal", "step_down_30", "impulse_120", "sustained_12"]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6), sharey=False)
    for a, key in zip(ax, keys):
        T, ft, fe = traces[key]
        a.plot(T, ft, color="#4C4C4C", lw=1.6, label="true $F_y$")
        a.plot(T, fe, color="#D55E00", lw=1.4, label="est $\\hat F_y$")
        a.axhline(F_THRESH, color="#0072B2", ls=":", lw=1, label="gate")
        a.axhline(-F_THRESH, color="#0072B2", ls=":", lw=1)
        a.set_title(key); a.set_xlabel("time [s]"); a.grid(alpha=0.3)
    ax[0].set_ylabel("lateral force [N]"); ax[0].legend(fontsize=8)
    fig.suptitle("External-wrench estimator vs ground truth (ID off, no process noise)")
    fig.tight_layout(); fig.savefig(HERE / "figures" / "estimator_validation.png", dpi=150)
    print("saved figures/estimator_validation.png, results/estimator_validation.json")


if __name__ == "__main__":
    main()
