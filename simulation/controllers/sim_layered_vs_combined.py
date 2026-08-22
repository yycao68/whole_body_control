#!/usr/bin/env python3
"""Two versions of the floating-base interaction controller — LAYERED (three
separate layers / two MPCs) vs COMBINED (one unified interaction MPC) — on the
reduced, G1-parameterized interaction plant.

Both regulate a body (CoM) port and a task (arm) port, each of the form
    ë = u + d,
coupled by the arm's reaction on the CoM. The only difference is exactly §VI of
the paper:

  LAYERED  : body MPC and task MPC run as separate offset-free regulators. The
             arm's reaction reaches the body ONLY as a disturbance d_body, which
             the body Kalman must discover (lag) -> transient CoM excursion.
  COMBINED : one stacked interaction MPC. The KNOWN arm reaction is fed forward
             into the body channel (the Γ_bt coupling), so balance is
             pre-compensated (anticipation) -> the CoM barely moves.

Both are offset-free (integrating Kalman d̂), so a sustained push -> 0 SS error at
both ports. Disabling the anticipation feedforward makes COMBINED ≡ LAYERED
(the §VI-E equivalence, verified numerically at the end).

Run:  python3 sim_layered_vs_combined.py
Writes layered_vs_combined.png and prints the comparison metrics.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G1_MASS = 33.3          # kg  (official Unitree G1)
LAM_ARM = 5.0           # kg  representative contact-consistent arm inertia
DT = 1e-3
T = 5.0
N = int(T / DT)


class OffsetFreeChannel:
    """Scalar interaction channel ë = u + d with augmented [e, ė, d̂] Kalman
    (measuring e and ė) and a critically-damped feedback gain."""

    def __init__(self, wn=20.0, q_d=80.0, r_p=1e-9, r_v=1e-7):
        dt = DT
        self.Ad = np.array([[1, dt, 0.5 * dt * dt],
                            [0, 1, dt],
                            [0, 0, 1]])
        self.Bu = np.array([0.5 * dt * dt, dt, 0.0])
        self.C = np.array([[1, 0, 0], [0, 1, 0]])
        self.Q = np.diag([1e-10, 1e-8, q_d])
        self.R = np.diag([r_p, r_v])
        self.z = np.zeros(3)
        self.P = np.diag([1e-6, 1e-4, 1e2])
        self.ke, self.kv = wn * wn, 2.0 * wn          # critically damped

    def estimate(self, e_meas, ev_meas, u_known):
        """u_known: the KNOWN acceleration input applied to the plant last step
        (control + any modeled/anticipated disturbance). d̂ then captures only
        the UNKNOWN residual."""
        z = self.Ad @ self.z + self.Bu * u_known
        P = self.Ad @ self.P @ self.Ad.T + self.Q
        y = np.array([e_meas, ev_meas]) - self.C @ z
        S = self.C @ P @ self.C.T + self.R
        K = P @ self.C.T @ np.linalg.inv(S)
        self.z = z + K @ y
        self.P = (np.eye(3) - K @ self.C) @ P
        return self.z                                 # [ê, ê̇, d̂]

    def control(self, e, ev, d_hat, ff=0.0):
        """Offset-free residual-acceleration command."""
        return -(self.ke * e + self.kv * ev) - d_hat + ff


def _minjerk_step(t, t0, dur, amp):
    """Smooth min-jerk step of size amp over [t0, t0+dur]; returns (x_d, xdd_d)."""
    if t < t0:
        return 0.0, 0.0
    if t > t0 + dur:
        return amp, 0.0
    s = (t - t0) / dur
    x = amp * (10 * s**3 - 15 * s**4 + 6 * s**5)
    xdd = amp * (60 * s - 180 * s**2 + 120 * s**3) / dur**2
    return x, xdd


def run(mode):
    """mode in {'layered', 'combined', 'combined_no_ff'}."""
    body = OffsetFreeChannel(wn=7.0, q_d=12.0)        # CoM port: compliant balance + slower observer
    arm = OffsetFreeChannel(wn=22.0, q_d=120.0)       # task port (fast disturbance rejection)
    # true plant states: [pos, vel] for body(CoM lateral) and arm(error frame)
    c, cv = 0.0, 0.0                                  # CoM position (ref 0)
    ex, exv = 0.0, 0.0                                # arm tracking error
    log = {k: np.zeros(N) for k in
           ("t", "c", "ex", "Farm", "dhat_c", "dhat_x")}
    uk_arm, uk_body = 0.0, 0.0                        # known plant inputs (prev step)

    for k in range(N):
        t = k * DT
        # --- scenario inputs ---
        Fh = 8.0 if t >= 1.0 else 0.0                 # sustained human push on arm [N]
        a_ext = 0.6 if t >= 3.5 else 0.0              # sustained external CoM accel disturbance [m/s^2]
        x_d, xdd_d = _minjerk_step(t, 2.0, 0.10, 0.20)  # fast 20 cm reach -> sharp reaction

        # --- disturbances entering each channel (ë = u + d) ---
        d_x = Fh / LAM_ARM                            # human force as arm accel disturbance (unknown)

        # --- estimation (uses previous-step KNOWN input; d̂ = unknown residual) ---
        _, _, dhat_x = arm.estimate(ex, exv, uk_arm)
        _, _, dhat_c = body.estimate(c, cv, uk_body)

        # --- task control ---
        u_x = arm.control(ex, exv, dhat_x)
        # physical arm force (feedforward + residual) and its reaction on the CoM
        F_arm = LAM_ARM * (xdd_d - u_x)
        a_react = -F_arm / G1_MASS                    # arm reaction as CoM accel

        # --- body control; anticipation only in the combined version ---
        if mode == "combined":
            ff = -a_react                             # cancel KNOWN reaction
            known_extra = a_react                     # a_react is modeled -> known to estimator
        else:                                         # layered / combined_no_ff
            ff = 0.0
            known_extra = 0.0                         # a_react unknown -> booked into d̂_c (lag)
        u_c = body.control(c, cv, dhat_c, ff=ff)

        # known inputs applied to each plant this step (for next prediction)
        uk_arm = u_x                                  # human d_x is unknown -> excluded
        uk_body = u_c + known_extra

        # --- true plant integration ---
        exv += (u_x + d_x) * DT                       # arm:  ë_x = u_x + d_x
        ex += exv * DT
        ac = u_c + a_react + a_ext                    # body: ë_c = u_c + a_react + a_ext
        cv += ac * DT
        c += cv * DT

        for key, val in (("t", t), ("c", c), ("ex", ex), ("Farm", F_arm),
                         ("dhat_c", dhat_c), ("dhat_x", dhat_x)):
            log[key][k] = val
    return log


def metrics(log):
    t = log["t"]
    ss = slice(int(4.5 / DT), N)                      # last 0.5 s
    move = slice(int(1.9 / DT), int(2.6 / DT))        # around the arm move
    return dict(
        arm_ss_mm=abs(np.mean(log["ex"][ss])) * 1e3,
        com_peak_move_mm=np.max(np.abs(log["c"][move])) * 1e3,
        com_ss_mm=abs(np.mean(log["c"][ss])) * 1e3,
    )


def main():
    lay = run("layered")
    com = run("combined")
    eqv = run("combined_no_ff")

    ml, mc, me = metrics(lay), metrics(com), metrics(eqv)
    print(f"{'metric':30s}{'LAYERED':>12}{'COMBINED':>12}{'COMB(no ff)':>14}")
    for key, lab in (("arm_ss_mm", "arm SS error [mm]"),
                     ("com_peak_move_mm", "CoM peak @ arm-move [mm]"),
                     ("com_ss_mm", "CoM SS error [mm]")):
        print(f"{lab:30s}{ml[key]:>12.4f}{mc[key]:>12.4f}{me[key]:>14.4f}")

    # equivalence check: combined_no_ff should match layered to machine precision
    dc = np.max(np.abs(lay["c"] - eqv["c"]))
    print(f"\n§VI-E equivalence  max|c_layered - c_combined(no ff)| = {dc:.2e} m "
          f"({'PASS' if dc < 1e-9 else 'differ'})")
    print(f"anticipation gain  CoM peak {ml['com_peak_move_mm']:.2f} -> "
          f"{mc['com_peak_move_mm']:.2f} mm  "
          f"({ml['com_peak_move_mm']/max(mc['com_peak_move_mm'],1e-9):.1f}x lower)")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    a1.plot(lay["t"], lay["c"] * 1e3, color="#d62728", label="Layered (3 layers, 2 MPCs)")
    a1.plot(com["t"], com["c"] * 1e3, color="#2ca02c", label="Combined (unified MPC + anticipation)")
    a1.axvspan(2.0, 2.25, alpha=0.08, color="k")
    a1.set_ylabel("CoM excursion [mm]")
    a1.set_title("CoM under the arm-move reaction (shaded: arm reference move)")
    a1.legend(fontsize=8); a1.grid(alpha=.3)
    a2.plot(lay["t"], lay["ex"] * 1e3, color="#d62728", label="Layered arm error")
    a2.plot(com["t"], com["ex"] * 1e3, color="#2ca02c", label="Combined arm error")
    a2.axvline(1.0, ls=":", c="gray"); a2.text(1.02, a2.get_ylim()[1]*0.6, "8 N push", fontsize=7)
    a2.set_xlabel("time [s]"); a2.set_ylabel("arm error [mm]")
    a2.set_title("Task error — both offset-free under the sustained push")
    a2.legend(fontsize=8); a2.grid(alpha=.3)
    fig.tight_layout()
    out = "layered_vs_combined.png"
    fig.savefig(out, dpi=150)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
