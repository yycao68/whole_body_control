"""
Verify contact-consistent interaction inertia.

The paper's key modeling claim is that the normalized interaction dynamics
share a constant A_d while the input channel changes through
Lambda_arm(q, contact). This script plots the task-space inertia obtained
from:
  - free-space inverse M^{-1},
  - double-support constrained inverse, and
  - right-foot-support constrained inverse
over an arm posture sweep.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from wbc_core import (
    _get_ids,
    get_contact_consistent_inverse,
    get_contact_jacobian,
    get_mass_matrix,
    get_site_jacobian,
    get_task_inertia,
)


MODEL_PATH = Path(__file__).with_name("biped.xml")
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def joint_qadr(model, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return model.jnt_qposadr[jid]


def lambda_for_contact(model, data, hand_site, foot_sites, mask):
    M = get_mass_matrix(model, data)
    J = get_site_jacobian(model, data, hand_site)
    if mask is None:
        Minv = np.linalg.inv(M + 1e-6 * np.eye(model.nv))
    else:
        Jc = get_contact_jacobian(model, data, foot_sites, mask)
        Minv = get_contact_consistent_inverse(M, Jc)
    return get_task_inertia(J, Minv)


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    ids = _get_ids(model)

    hand = ids["hand_site"]
    feet = [ids["left_foot_site"], ids["right_foot_site"]]
    qadr = joint_qadr(model, "right_shoulder_y")
    q0 = float(data.qpos[qadr])

    sweep = np.linspace(q0 - 0.45, q0 + 0.45, 45)
    cases = {
        "free-space": None,
        "double support": [True, True],
        "right support": [False, True],
    }
    diag_logs = {name: [] for name in cases}
    eig_logs = {name: [] for name in cases}

    for q in sweep:
        data.qpos[qadr] = q
        mujoco.mj_forward(model, data)
        for name, mask in cases.items():
            Lam = lambda_for_contact(model, data, hand, feet, mask)
            diag_logs[name].append(np.diag(Lam))
            eig_logs[name].append(np.linalg.eigvalsh(0.5 * (Lam + Lam.T)))

    for name in cases:
        diag_logs[name] = np.asarray(diag_logs[name])
        eig_logs[name] = np.asarray(eig_logs[name])

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
    x = sweep - q0
    colors = {
        "free-space": "#666666",
        "double support": "#1f77b4",
        "right support": "#d62728",
    }
    for name in cases:
        ax[0].plot(x, diag_logs[name][:, 0], color=colors[name], lw=2, label=f"{name}: x")
        ax[0].plot(x, diag_logs[name][:, 1], color=colors[name], lw=1.4, ls="--", label=f"{name}: y")
        ax[1].plot(x, eig_logs[name][:, -1], color=colors[name], lw=2, label=name)

    ax[0].set_xlabel("right shoulder-y offset [rad]")
    ax[0].set_ylabel(r"diag$(\Lambda_{\rm arm})$ [kg]")
    ax[0].set_title("Task inertia depends on contact and posture")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(fontsize=7, ncol=2)

    ax[1].set_xlabel("right shoulder-y offset [rad]")
    ax[1].set_ylabel(r"largest eig$(\Lambda_{\rm arm})$ [kg]")
    ax[1].set_title("Contact-consistent apparent inertia")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    out = OUT_DIR / "inertia_normalization.png"
    fig.savefig(out, dpi=180)

    mid = len(sweep) // 2
    print("Nominal Lambda_arm diag [kg]")
    for name in cases:
        print(f"  {name:15s}: {np.round(diag_logs[name][mid], 3)}")
    print(f"Figure saved to {out}")


if __name__ == "__main__":
    main()
