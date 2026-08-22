#!/usr/bin/env python3
"""Centroidal-model walking with the interaction-dynamics framework.

Walking is a sequence of contact-mode switches. This demo drives a forward gait
with the pieces already built:
  - CentroidalBalance  (g1_centroidal_balance): the §IV body port regulates the
    CoM to a walking reference, producing the ground-reaction wrench.
  - a divergent-component-of-motion (DCM) walking-pattern reference over planned
    footsteps keeps the ZMP inside the stance foot.
  - CentroidalDisturbanceObserver + ContactDetector (contact_detector): the
    d̂_com state DETECTS each foot touchdown from the landing transient and
    switches the contact mode — contact-mode changes are observed, not scheduled.

The controller is NOT told the touchdown times; it infers them. We report walking
distance, per-step detection latency vs the true schedule, and false positives,
and plot the CoM path over the footsteps and the detected contact timeline.

Run:  python3 g1_centroidal_walk.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from g1_centroidal_balance import CentroidalBalance
from contact_detector import CentroidalDisturbanceObserver, ContactDetector

G0, MASS, ZC = 9.81, 33.3, 0.88
DT = 1e-3
OMEGA = np.sqrt(G0 / ZC)

STEP_LEN, STEP_W = 0.25, 0.09      # forward stride, half stance width [m]
T_SS, T_DS = 0.6, 0.15             # single- / double-support durations [s]
N_STEPS = 6


def plan_footsteps():
    """Alternating forward footsteps; returns list of (x, y, which)."""
    steps = []
    x = 0.0
    side = 1                       # start stepping with the left foot (+y)
    for i in range(N_STEPS):
        x += STEP_LEN
        steps.append((x, side * STEP_W, "L" if side > 0 else "R"))
        side *= -1
    return steps


def build_reference(footsteps):
    """DCM walking pattern -> CoM reference + gait schedule (per-ms).
    Returns t, com_ref[N,2], phase list of (t0,t1,mode,stance_xy,touchdown_t)."""
    # ZMP sequence: start centered, then each stance foot in turn
    zmps = [np.array([0.0, 0.0])]
    for (x, y, _) in footsteps:
        zmps.append(np.array([x, y]))
    # DCM end-of-step targets = the next ZMP; backward recursion for step-initial DCM
    xi_ini = [None] * (len(zmps))
    xi_ini[-1] = zmps[-1].copy()
    for i in range(len(zmps) - 2, -1, -1):
        xi_ini[i] = zmps[i] + (xi_ini[i + 1] - zmps[i]) * np.exp(-OMEGA * T_SS)

    t, com_ref, phases = [], [], []
    x_com = zmps[0].copy(); v_com = np.zeros(2)
    tt = 0.0
    for i in range(len(footsteps)):
        p = zmps[i + 1]                              # stance foot ZMP this step
        xi0 = xi_ini[i + 1]
        t0 = tt
        n = int(T_SS / DT)
        for k in range(n):
            s = k * DT
            xi = p + (xi0 - p) * np.exp(OMEGA * s)   # DCM trajectory
            v_com = OMEGA * (xi - x_com)             # CoM follows DCM
            x_com = x_com + v_com * DT
            com_ref.append(x_com.copy()); t.append(tt); tt += DT
        phases.append((t0, tt, "SS", p.copy(), t0))  # touchdown at phase start
    return np.array(t), np.array(com_ref), phases


def main():
    footsteps = plan_footsteps()
    t, com_ref, phases = build_reference(footsteps)
    N = len(t)

    bal = CentroidalBalance(mass=MASS, wn=8.0, dt=DT)
    obs = CentroidalDisturbanceObserver(mass=MASS, dt=DT)
    det = ContactDetector(normal=(0, 0, 1), calib_steps=int(0.3 / DT), refractory=int(0.25 / DT))

    c = np.array([com_ref[0, 0], com_ref[0, 1], ZC]); cv = np.zeros(3)
    log_c = np.zeros((N, 2)); dhat_z = np.zeros(N)
    true_td = [ph[4] for ph in phases[1:]]           # touchdowns (each new stance)
    det_td = []

    for k in range(N):
        ref = np.array([com_ref[k, 0], com_ref[k, 1], ZC])
        F, _ = bal.command(c, cv, ref)
        # true CoM plant: m c̈ = F - m g + (touchdown impact)
        impact = np.zeros(3)
        for td in true_td:                           # modeled landing transient
            if 0.0 <= (t[k] - td) < 0.03:
                impact[2] = 0.45 * MASS * G0          # brief support-force onset
        cc = (F - MASS * np.array([0, 0, G0]) + impact) / MASS
        cv += cc * DT; c += cv * DT
        log_c[k] = c[:2]

        # detection: observer on CoM (fixed assumed support) -> d̂_z step at landing
        a_cmd = F / MASS - np.array([0, 0, G0])       # controller's assumed CoM accel
        d_hat = obs.step(c, cv, a_cmd)
        dhat_z[k] = d_hat[2]
        ev = det.update(d_hat)
        if ev == "touchdown":
            det_td.append(t[k])

    # metrics: match each detection to the nearest true touchdown
    lat = []
    for td in true_td:
        cand = [d - td for d in det_td if -0.05 <= d - td <= 0.2]
        if cand:
            lat.append(min(cand, key=abs) * 1e3)
    fp = len(det_td) - len(lat)
    dist = log_c[-1, 0] - log_c[0, 0]
    print(f"walked {dist:.2f} m in {N_STEPS} steps ({t[-1]:.1f} s)")
    print(f"touchdowns: {len(true_td)} true, {len(det_td)} detected")
    print(f"detection latency: mean {np.mean(lat):.1f} ms  (per-step {[f'{x:.0f}' for x in lat]})")
    print(f"false positives: {fp}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(log_c[:, 0], log_c[:, 1], "b", lw=1, label="CoM path")
    for (x, y, w) in footsteps:
        ax[0].add_patch(plt.Rectangle((x - 0.08, y - 0.04), 0.16, 0.08,
                        color="green" if w == "L" else "red", alpha=0.35))
        ax[0].text(x, y + 0.06, w, ha="center", fontsize=7)
    ax[0].set_xlabel("forward x [m]"); ax[0].set_ylabel("lateral y [m]")
    ax[0].set_title("CoM path over footsteps (L green / R red)")
    ax[0].axis("equal"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)

    ax[1].plot(t, dhat_z, "purple", lw=0.8, label="d̂_com (vertical) [N]")
    for td in true_td:
        ax[1].axvline(td, color="k", ls=":", lw=0.8)
    for d in det_td:
        ax[1].axvline(d, color="orange", ls="-", lw=1.0, alpha=0.7)
    ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("d̂_com,z [N]")
    ax[1].set_title("Touchdown detection (dotted=true, orange=detected)")
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("g1_centroidal_walk.png", dpi=150)
    print("figure -> g1_centroidal_walk.png")


if __name__ == "__main__":
    main()
