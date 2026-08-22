"""Regression tests for the normalized model stated in paper Section V."""

import unittest

import numpy as np

from impedance_mpc import ImpedanceMPC
from kalman import KalmanDisturbanceEstimator
import scenario_c_g1


class NormalizedModelTests(unittest.TestCase):
    def test_exact_zoh_double_integrator(self):
        dt = 1e-3
        mpc = ImpedanceMPC(N=4, dt=dt)
        expected = np.vstack([
            0.5 * dt**2 * np.eye(3),
            dt * np.eye(3),
        ])
        np.testing.assert_allclose(mpc.B_d, expected)

    def test_lifted_model_is_inertia_independent(self):
        mpc = ImpedanceMPC(N=4)
        a = mpc.precompute_mode("a", np.diag([1.0, 2.0, 3.0]))
        b = mpc.precompute_mode("b", np.diag([4.0, 5.0, 6.0]))
        np.testing.assert_allclose(a["B_d"], b["B_d"])
        np.testing.assert_allclose(a["Gamma"], b["Gamma"])
        np.testing.assert_allclose(a["H"], b["H"])

    def test_force_recovery_and_observer_input_coordinates(self):
        mpc = ImpedanceMPC(N=8, F_max=1e6)
        inertia = np.diag([1.2, 2.0, 3.0])
        force = mpc.solve(
            np.array([0.01, -0.02, 0.005, 0.0, 0.0, 0.0]),
            inertia,
            use_osqp=False,
        )
        np.testing.assert_allclose(force, inertia @ mpc.last_u)

        kf = KalmanDisturbanceEstimator(dt=mpc.dt)
        kf.set_mode(mpc.A_d, mpc.B_d)
        np.testing.assert_allclose(kf.B_ctrl[:6], mpc.B_d)

    def test_force_constraint_updates_with_inertia(self):
        mpc = ImpedanceMPC(N=4, F_max=0.01)
        x = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        for inertia in (
            np.diag([1.0, 2.0, 3.0]),
            np.array([[1.0, 0.1, 0.0], [0.1, 2.0, 0.2], [0.0, 0.2, 3.0]]),
        ):
            force = mpc.solve(x, inertia, use_osqp=True)
            self.assertLessEqual(np.max(np.abs(force)), mpc.F_max + 1e-6)

    def test_g1_benchmark_overrides_xml_timestep(self):
        model, _ = scenario_c_g1._make_robot()
        self.assertAlmostEqual(model.opt.timestep, scenario_c_g1.SIM_DT)


if __name__ == "__main__":
    unittest.main()
