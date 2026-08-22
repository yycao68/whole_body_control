#!/usr/bin/env python3
"""Sensor-free contact detection from the centroidal disturbance state.

Two reusable pieces for the reframed WBC case study (§VIII):

  CentroidalDisturbanceObserver — a Kalman filter on the CoM channel
      [p; v; d] per axis, where d is the *unmodeled* net force on the body.
      Driven by the measured CoM position and the controller's commanded
      net force; d̂ absorbs any force the assumed contact set does not
      explain — i.e. a new/removed contact.

  ContactDetector — projects the residual force d̂ onto candidate contact
      normals and flags touchdown/liftoff when the projection crosses a
      3σ threshold about its quiet-stance baseline.

Running this file executes a self-test on a controlled centroidal sim (no
MuJoCo): a step in the true unmodeled support force is recovered by the
observer and detected, reporting detection latency and false-positive rate.
The MuJoCo/G1 integration reuses these classes unchanged.
"""
import numpy as np

G0 = 9.81


class CentroidalDisturbanceObserver:
    """Per-axis Kalman on [p, v, d]; d = unmodeled body force (N)."""

    def __init__(self, mass, dt, q_d=8.0, r_p=1e-8, r_v=1e-6, p0_d=1e2):
        self.m, self.dt = float(mass), float(dt)
        m, dt_ = self.m, self.dt
        # per-axis augmented double integrator, d enters as accel d/m
        self.A = np.array([[1.0, dt_, dt_ * dt_ / (2 * m)],
                           [0.0, 1.0, dt_ / m],
                           [0.0, 0.0, 1.0]])
        self.B = np.array([dt_ * dt_ / 2.0, dt_, 0.0])   # commanded accel input
        self.C = np.array([[1.0, 0.0, 0.0],              # measure position AND velocity
                           [0.0, 1.0, 0.0]])
        self.Q = np.diag([1e-10, 1e-8, q_d])             # process noise (random-walk d)
        self.R = np.diag([r_p, r_v])
        self.z = np.zeros((3, 3))                        # [axis, (p,v,d)]
        self.P = np.stack([np.diag([1e-6, 1e-4, p0_d])] * 3)

    def step(self, p_meas, v_meas, a_cmd):
        """Measured CoM position and velocity; a_cmd[axis] the commanded CoM
        acceleration incl. gravity comp (F_cmd/m + g). Returns d̂ (3,) [N]."""
        d_hat = np.zeros(3)
        I3 = np.eye(3)
        for ax in range(3):
            z, P = self.z[ax], self.P[ax]
            z = self.A @ z + self.B * a_cmd[ax]
            P = self.A @ P @ self.A.T + self.Q
            y = np.array([p_meas[ax], v_meas[ax]]) - self.C @ z
            S = self.C @ P @ self.C.T + self.R
            K = P @ self.C.T @ np.linalg.inv(S)
            z = z + K @ y
            P = (I3 - K @ self.C) @ P
            self.z[ax], self.P[ax] = z, P
            d_hat[ax] = z[2]
        return d_hat


class ContactDetector:
    """3σ detector on the residual force projected onto a contact normal."""

    def __init__(self, normal=(0.0, 0.0, 1.0), n_sigma=3.0,
                 calib_steps=500, debounce=8, refractory=300):
        self.n = np.asarray(normal, float)
        self.n /= np.linalg.norm(self.n)
        self.n_sigma, self.debounce, self.refractory = n_sigma, debounce, refractory
        self.calib_steps = calib_steps
        self._buf, self._sig = [], 1.0
        self._level = 0.0        # current committed force plateau [N]
        self._cnt = self._sign = 0
        self._refr = 0
        self.k = 0

    def update(self, d_hat):
        """Change-point detector: flags a step of the projected residual force
        away from the current plateau. Returns 'touchdown'/'liftoff' or None."""
        proj = float(self.n @ d_hat)
        self.k += 1
        if self.k <= self.calib_steps:                  # quiet-stance calibration
            self._buf.append(proj)
            if self.k == self.calib_steps:
                self._level = float(np.mean(self._buf))
                self._sig = float(np.std(self._buf)) + 1e-9
            return None
        if self._refr > 0:                              # refractory: re-baseline at end
            self._refr -= 1
            if self._refr == 0:
                self._level = proj                      # commit new settled plateau
            return None
        dev = (proj - self._level) / self._sig
        sign = 1 if dev > self.n_sigma else (-1 if dev < -self.n_sigma else 0)
        if sign != 0 and sign == self._sign:
            self._cnt += 1
            if self._cnt >= self.debounce:
                self._cnt, self._sign = 0, 0
                self._refr = self.refractory
                return "touchdown" if sign > 0 else "liftoff"
        else:
            self._sign, self._cnt = sign, (1 if sign != 0 else 0)
        return None


# ---------------------------------------------------------------------------
# Self-test: controlled centroidal sim, no MuJoCo.
# ---------------------------------------------------------------------------
def _self_test():
    rng = np.random.default_rng(0)
    dt, m, T = 1e-3, 33.3, 12.0
    N = int(T / dt)
    # ground-truth unmodeled support force on z: 0, then +0.5*m*g at touchdown
    # (a foot taking half body weight), back to 0 at liftoff.
    F_step = 0.5 * m * G0                                # ≈163 N
    t_touch, t_lift = 4.0, 8.0
    d_true = np.where((np.arange(N) * dt >= t_touch) & (np.arange(N) * dt < t_lift),
                      F_step, 0.0)

    obs = CentroidalDisturbanceObserver(mass=m, dt=dt)
    det = ContactDetector(normal=(0, 0, 1), calib_steps=int(2.0 / dt))

    # simulate CoM z with the controller commanding a_cmd = g (gravity comp);
    # the true extra force d_true perturbs the actual acceleration.
    p, v = 0.88, 0.0
    a_cmd = np.array([0.0, 0.0, 0.0])                    # residual accel command ~0 (holding)
    meas_noise = 2e-4                                    # 0.2 mm CoM position noise
    events, dhat_log = [], np.zeros(N)
    for k in range(N):
        a_true = a_cmd[2] + d_true[k] / m               # d/m extra accel
        v += a_true * dt
        p += v * dt
        p_meas = np.array([0.0, 0.0, p + rng.normal(0, meas_noise)])
        v_meas = np.array([0.0, 0.0, v + rng.normal(0, 1e-3)])
        d_hat = obs.step(p_meas, v_meas, a_cmd)
        dhat_log[k] = d_hat[2]
        ev = det.update(d_hat)
        if ev:
            events.append((k * dt, ev))

    # metrics
    def latency(true_t, kind):
        for (tt, ev) in events:
            if ev == kind and tt >= true_t - 0.05:
                return (tt - true_t) * 1e3
        return None
    lat_td = latency(t_touch, "touchdown")
    lat_lo = latency(t_lift, "liftoff")
    fp = sum(1 for (tt, _) in events if tt < t_touch - 0.05)   # events before first true one
    steady = dhat_log[int(3.0 / dt):int(3.8 / dt)]             # settled after touchdown
    print(f"true F_step        = {F_step:.1f} N")
    print(f"d̂ settled (contact) = {np.mean(dhat_log[int(6.0/dt):int(7.8/dt)]):.1f} N "
          f"(should ≈ {F_step:.0f})")
    print(f"d̂ std (quiet)      = {np.std(dhat_log[int(0.5/dt):int(1.8/dt)]):.2f} N")
    print(f"detected events    = {events}")
    print(f"touchdown latency  = {lat_td:.1f} ms" if lat_td is not None else "touchdown MISSED")
    print(f"liftoff   latency  = {lat_lo:.1f} ms" if lat_lo is not None else "liftoff MISSED")
    print(f"false positives    = {fp} (over {int((t_touch-0.05))}s quiet)")


if __name__ == "__main__":
    _self_test()
