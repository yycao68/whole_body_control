#!/usr/bin/env python3
"""Centroidal balance in interaction-dynamics form (the §IV body port), and why
the anticipation must flow through the centroidal WRENCH, not a hip angle.

The full-G1 test showed that feeding the raw arm force into a hip ANGLE
destabilizes balance (combined ~1000 mm, near-fall). The correct formulation:
the arm's known reaction enters the desired CoM ACCELERATION, is mapped to a
ground-reaction wrench  F = m(acc + g)  (allocated to the feet within friction
cones by level1_centroidal) — dimensionally consistent with the CoM feedback,
hence STABLE, and it cancels the CoM excursion.

Running the file validates this on a controlled centroidal sim (G1 params):
  reactive     : CoM feedback + offset-free d̂, no anticipation.
  wrench-antic : + arm reaction fed forward as CoM acceleration (§VI Γ_bt, in
                 wrench space) -> stable, excursion eliminated.
"""
import numpy as np

G0 = 9.81


class CentroidalBalance:
    """Interaction-dynamics balance: desired CoM acceleration -> GRF wrench
    F = m(acc + g). Offset-free via an accel-residual disturbance estimate; the
    arm reaction is anticipated in ACCELERATION space (well-posed)."""

    def __init__(self, mass, wn=6.0, dt=1e-3, d_lp=0.03):
        self.m, self.dt, self.L = mass, dt, d_lp
        self.ke, self.kv = wn * wn, 2.0 * wn
        self.d_hat = np.zeros(3)
        self.uk_prev = np.zeros(3)          # known accel input last step
        self.cv_prev = None

    def command(self, com, comv, com_ref, arm_reaction=None):
        """Returns net ground-reaction force F (3,) and CoM accel command.
        arm_reaction: known 3-D arm reaction force on the body (None => reactive)."""
        # offset-free disturbance estimate from the acceleration residual
        if self.cv_prev is not None:
            c_ddot = (comv - self.cv_prev) / self.dt
            self.d_hat = (1 - self.L) * self.d_hat + self.L * (c_ddot - self.uk_prev)
        self.cv_prev = comv.copy()

        a_react = (arm_reaction / self.m) if arm_reaction is not None else np.zeros(3)
        acc = self.ke * (com_ref - com) - self.kv * comv - self.d_hat - a_react
        self.uk_prev = acc + a_react        # known total input to the plant

        F = self.m * (acc + np.array([0.0, 0.0, G0]))   # net GRF wrench (linear)
        return F, acc


# ---------------------------------------------------------------------------
def _test():
    m, dt, T = 33.3, 1e-3, 6.0
    N = int(T / dt)

    def sim(anticipate):
        bal = CentroidalBalance(mass=m, dt=dt)
        c = np.array([0.0, 0.0, 0.88]); cv = np.zeros(3); ref = c.copy()
        log = np.zeros(N)
        for k in range(N):
            t = k * dt
            F_ext = np.array([0, 6.0, 0]) if t >= 1.0 else np.zeros(3)   # sustained CoM push
            F_arm = np.zeros(3)
            if 2.0 <= t <= 2.5:                                          # fast lateral arm reach
                F_arm = np.array([0, 40.0 * np.sin(np.pi * (t - 2.0) / 0.5), 0])
            reaction = -F_arm
            F, _ = bal.command(c, cv, ref, arm_reaction=(reaction if anticipate else None))
            # true CoM plant: m c̈ = F - m g + reaction + F_ext
            cc = (F - m * np.array([0, 0, G0]) + reaction + F_ext) / m
            cv = cv + cc * dt
            c = c + cv * dt
            log[k] = c[1]                                                # lateral CoM
        return log

    lay, com = sim(False), sim(True)
    reach = slice(int(1.9 / dt), int(3.1 / dt))
    ss = slice(int(5.5 / dt), N)
    print(f"{'metric':36s}{'REACTIVE':>12}{'WRENCH-ANTIC':>14}")
    print(f"{'lateral CoM peak @ arm reach [mm]':36s}"
          f"{np.max(np.abs(lay[reach]))*1e3:>12.2f}{np.max(np.abs(com[reach]))*1e3:>14.2f}")
    print(f"{'lateral CoM SS under 6 N push [mm]':36s}"
          f"{abs(np.mean(lay[ss]))*1e3:>12.3f}{abs(np.mean(com[ss]))*1e3:>14.3f}")
    print(f"{'max |CoM| (stability check) [mm]':36s}"
          f"{np.max(np.abs(lay))*1e3:>12.2f}{np.max(np.abs(com))*1e3:>14.2f}")
    print("\n-> anticipation via the centroidal WRENCH is stable (bounded CoM) and "
          "cancels the reach excursion, unlike the hip-angle feedforward that "
          "destabilized the full-G1 stance.")


if __name__ == "__main__":
    _test()
