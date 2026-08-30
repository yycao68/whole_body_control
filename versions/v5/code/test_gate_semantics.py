"""Unit tests for the confidence-gate / capture-hold blend state machine.

Added after an external review (2026-08-30) found that the abstract's claim --
that the layer "engages ... only when the estimated wrench rises above its own
measured noise floor" -- does not hold for the *hold* path. These tests pin the
actual, implemented semantics so the paper text and the code cannot drift apart
again, including the adversarial case the review asked for.

Implemented semantics (what these tests assert):

  * capture is gated:  u_capture = -k_map * a_e * g_cap
  * hold is NOT gated: u_hold    = -kp_s*e_eff - ki_s*e_int
  * blend:             u = (1-p)*u_capture + p*u_hold

  so with f_thresh < ||F_hat|| < f_cap the hold weight p rises while the
  confidence gate g_cap stays closed, and the layer still commands a nonzero
  action. The gate protects the capture path only.
"""
import unittest

import numpy as np

from stage2_id_on_policy import IDResidual


Z = np.zeros(2)


def _calibrated(noise_n=6.0, steps=600, mode="wrench"):
    """Controller whose online wrench noise floor has been driven up.

    Feeds quiescent samples (|e| < deadband, so the sigma_f estimator updates)
    carrying `noise_n` of apparent wrench, which raises
    f_cap = f_floor + k_conf*sigma_f well above the persistence threshold
    f_thresh. That separation is what creates the adversarial band.
    """
    c = IDResidual("id_mpc", 0.02, mode=mode)
    for _ in range(steps):
        c.update(Z, Z, Z, Z, f_ext=np.array([noise_n, 0.0]))
    return c


def _f_cap(c):
    return c.f_floor + c.k_conf * np.sqrt(c.sig2_f)


class ConfidenceGateTests(unittest.TestCase):
    def test_noise_floor_self_calibrates_above_persistence_threshold(self):
        c = _calibrated()
        self.assertGreater(_f_cap(c), c.f_thresh,
                           "test premise: the adversarial band must be nonempty")

    def test_subthreshold_noise_never_latches_capture(self):
        """The property the gate exists for: no capture runaway on noise."""
        c = _calibrated()
        for _ in range(150):
            c.update(np.array([0.05, 0.0]), Z, Z, Z, f_ext=np.array([5.0, 0.0]))
        self.assertEqual(c.g_cap, 0.0)

    def test_confident_wrench_opens_the_capture_gate(self):
        c = _calibrated()
        big = _f_cap(c) + 5.0
        c.update(np.array([0.05, 0.0]), Z, Z, Z, f_ext=np.array([big, 0.0]))
        self.assertEqual(c.g_cap, 1.0)

    def test_adversarial_band_still_commands_via_the_ungated_hold_path(self):
        """f_thresh < ||F_hat|| < f_cap: gate closed, layer still acts.

        This is the review's adversarial case. It documents a real gap between
        the abstract's "only when" wording and the implementation: the assertion
        is that the action is NONZERO, i.e. the paper claim -- not the code --
        is what needs narrowing (or the blend needs gating wholesale).
        """
        c = _calibrated()
        f_mid = 0.5 * (c.f_thresh + _f_cap(c))
        self.assertLess(c.f_thresh, f_mid)
        self.assertLess(f_mid, _f_cap(c))

        u = None
        for _ in range(150):
            u = c.update(np.array([0.05, 0.0]), Z, Z, Z,
                         f_ext=np.array([f_mid, 0.0]))

        self.assertEqual(c.g_cap, 0.0, "capture gate must be closed here")
        self.assertGreater(c.last_p, 0.0, "hold weight rises above f_thresh")
        self.assertGreater(np.linalg.norm(u), 1e-9,
                           "hold path is ungated, so the layer still commands")

    def test_inside_deadband_the_layer_is_silent(self):
        """Guards the earlier false negative: a deviation inside the deadband
        zeroes e_eff and hence the hold term, which can masquerade as the gate
        working. Any adversarial test must use |e| > deadband."""
        c = _calibrated()
        f_mid = 0.5 * (c.f_thresh + _f_cap(c))
        u = None
        for _ in range(150):
            u = c.update(np.array([0.5 * c.deadband, 0.0]), Z, Z, Z,
                         f_ext=np.array([f_mid, 0.0]))
        np.testing.assert_allclose(u, np.zeros(2), atol=1e-12)

    def test_capture_term_is_gated_to_zero_when_gate_closed(self):
        """Isolate the capture path: in 'transient' mode there is no hold term,
        so a closed gate must produce exactly zero."""
        c = _calibrated(mode="transient")
        f_mid = 0.5 * (c.f_thresh + _f_cap(c))
        u = None
        for _ in range(150):
            u = c.update(np.array([0.05, 0.0]), Z, Z, Z,
                         f_ext=np.array([f_mid, 0.0]))
        self.assertEqual(c.g_cap, 0.0)
        np.testing.assert_allclose(u, np.zeros(2), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
