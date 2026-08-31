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


class FeedforwardLawTests(unittest.TestCase):
    """Bind paper eq:ff_a, F_arm = Lambda_arm(p_ddot_d + u) + mu_arm, to code.

    External review found the implementation returned only `Lambda_arm @ u`,
    silently dropping both feedforward terms; nothing in the suite caught it.
    """

    def test_solve_returns_full_feedforward_law(self):
        mpc = ImpedanceMPC(N=8, F_max=1e6)
        inertia = np.array([[1.2, 0.1, 0.0],
                            [0.1, 2.0, 0.2],
                            [0.0, 0.2, 3.0]])
        p_ddot_d = np.array([0.3, -0.2, 0.7])
        mu_arm = np.array([-6.8, -0.1, 22.5])
        x = np.array([0.01, -0.02, 0.005, 0.0, 0.0, 0.0])
        force = mpc.solve(x, inertia, use_osqp=False,
                          p_ddot_d=p_ddot_d, mu_arm=mu_arm)
        np.testing.assert_allclose(
            force, inertia @ (mpc.last_u + p_ddot_d) + mu_arm, atol=1e-9)

    def test_feedforward_terms_actually_change_the_command(self):
        """Guards against the terms being accepted but ignored."""
        mpc = ImpedanceMPC(N=8, F_max=1e6)
        inertia = np.diag([1.2, 2.0, 3.0])
        x = np.array([0.01, -0.02, 0.005, 0.0, 0.0, 0.0])
        bare = mpc.solve(x, inertia, use_osqp=False)
        with_mu = mpc.solve(x, inertia, use_osqp=False,
                            mu_arm=np.array([5.0, -3.0, 8.0]))
        with_acc = mpc.solve(x, inertia, use_osqp=False,
                             p_ddot_d=np.array([0.5, 0.5, 0.5]))
        self.assertGreater(np.linalg.norm(with_mu - bare), 1.0)
        self.assertGreater(np.linalg.norm(with_acc - bare), 1.0)

    def test_omitted_feedforward_defaults_to_zero_not_garbage(self):
        mpc = ImpedanceMPC(N=8, F_max=1e6)
        inertia = np.diag([1.2, 2.0, 3.0])
        x = np.array([0.01, -0.02, 0.005, 0.0, 0.0, 0.0])
        a = mpc.solve(x, inertia, use_osqp=False)
        b = mpc.solve(x, inertia, use_osqp=False,
                      p_ddot_d=np.zeros(3), mu_arm=np.zeros(3))
        np.testing.assert_allclose(a, b)


class ConstraintTests(unittest.TestCase):
    """Paper eq:QP constrains the TOTAL force |F_ff + Lambda_arm u| <= F_max.

    Review found the QP constrained only |Lambda_arm u|, and that solver
    failure fell back to an unconstrained solution.
    """

    def test_qp_constrains_total_force_including_feedforward(self):
        F_max = 10.0
        mpc = ImpedanceMPC(N=4, F_max=F_max)
        inertia = np.diag([1.0, 2.0, 3.0])
        x = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        # A feedforward that already sits near the limit: if the box were
        # applied to Lambda_arm @ u alone, the total would blow past F_max.
        mu_arm = np.array([9.0, -9.0, 0.0])
        force = mpc.solve(x, inertia, use_osqp=True, mu_arm=mu_arm)
        self.assertLessEqual(np.max(np.abs(force)), F_max + 1e-6)

    def test_solver_failure_fails_safe(self):
        """A failed solve must not return an unconstrained solution."""
        mpc = ImpedanceMPC(N=4, F_max=10.0)
        inertia = np.diag([1.0, 2.0, 3.0])
        h_qp = np.zeros(mpc.N * mpc.nu)
        H = np.eye(mpc.N * mpc.nu)

        class _Failed:
            x = None
        original = mpc._osqp
        try:
            mpc._osqp = type('S', (), {
                'solve': lambda self_, raise_error=False: _Failed(),
                'update': lambda self_, **kw: None,
            })()
            mpc._osqp_nnz = None  # force reuse of the stub, not a rebuild
            u = mpc._solve_osqp(H, h_qp, mpc.N, mpc.nu, inertia,
                                np.zeros(3))
        finally:
            mpc._osqp = original
        np.testing.assert_allclose(u, np.zeros(mpc.nu))

    def test_qp_constrains_total_torque_at_every_horizon_stage(self):
        mpc = ImpedanceMPC(N=4, F_max=1e6)
        inertia = np.eye(3)
        torque_map = np.array([[3.0, 0.0, 0.0], [0.0, -2.0, 0.0]])
        torque_offset = np.array([4.0, -4.5])
        torque_min = np.array([-5.0, -5.0])
        torque_max = np.array([5.0, 5.0])
        mpc.solve(
            np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0]), inertia,
            use_osqp=True, torque_map=torque_map,
            torque_offset=torque_offset, torque_min=torque_min,
            torque_max=torque_max,
        )
        planned_torque = torque_offset + mpc.last_u_sequence @ torque_map.T
        self.assertTrue(np.all(planned_torque <= torque_max + 1e-5))
        self.assertTrue(np.all(planned_torque >= torque_min - 1e-5))


class ContactConsistencyTests(unittest.TestCase):
    """Paper Sec. IV's contact-consistent torque realization.

    Regression guard for the instability found in the second audit. The
    damping in this projector trades two properties against each other:

      contact_damp -> 0 : Pc is an EXACT oblique projector (idempotent,
                          eigenvalues exactly 0/1) but ||Pc||_2 ~ 28, high
                          enough to amplify the arm feedforward into a
                          divergent closed loop.
      larger damping    : ||Pc||_2 falls (12.6 at the shipped 0.1) and the
                          loop is stable, but Pc is no longer a true
                          projector -- its six contact-direction eigenvalues
                          drift from 0 toward 1 (0.65 at 0.1), so contact
                          decoupling is only approximate.

    These tests pin both ends of that trade so neither silently regresses.
    """

    def _stance(self):
        import scenario_a as sa
        from wbc_core import (get_mass_matrix, get_contact_jacobian, _get_ids)
        model, data = sa._make_robot()
        ids = _get_ids(model)
        sa._settle(model, data, ids, sa._precompute_arm_gravity(model))
        M = get_mass_matrix(model, data)
        Jc = get_contact_jacobian(
            model, data,
            [ids['left_foot_site'], ids['right_foot_site']], [True, True])
        return M, Jc

    def test_projector_is_exact_in_the_undamped_limit(self):
        """The formula itself is right: with damping removed, Pc @ Pc == Pc."""
        from wbc_core import get_contact_consistent_projector
        M, Jc = self._stance()
        Pc = get_contact_consistent_projector(M, Jc, contact_damp=1e-10)
        with np.errstate(all='ignore'):
            Pc_squared = Pc @ Pc
        self.assertTrue(np.all(np.isfinite(Pc_squared)))
        np.testing.assert_allclose(Pc, Pc_squared, atol=1e-6)
        eig = np.sort(np.linalg.eigvals(Pc).real)
        np.testing.assert_allclose(eig, np.round(eig), atol=1e-6)

    def test_shipped_damping_makes_the_projection_approximate(self):
        """Documents the honest cost of the stabilizing damping.

        This is not a nice property -- it is pinned so the paper's
        'approximate contact decoupling' caveat stays quantitatively true.
        """
        from wbc_core import get_contact_consistent_projector
        M, Jc = self._stance()
        Pc = get_contact_consistent_projector(M, Jc)
        eig = np.sort(np.linalg.eigvals(Pc).real)
        leaked = eig[(eig > 1e-6) & (eig < 1 - 1e-6)]
        self.assertEqual(len(leaked), 6)  # one per contact constraint row
        self.assertLess(leaked.max(), 0.8,
                        'contact directions have leaked too far toward 1; '
                        'the projection is no longer meaningfully projecting')

    def test_projector_operator_norm_is_bounded(self):
        """The check that would have caught the divergence directly."""
        from wbc_core import get_contact_consistent_projector
        M, Jc = self._stance()
        Pc = get_contact_consistent_projector(M, Jc)
        norm = np.linalg.norm(Pc, 2)
        # Measured 12.6 at the shipped contact_damp=0.1; 27.6 at the old
        # 1e-3 default, which diverged. 20 sits between the two.
        self.assertLess(norm, 20.0,
                        f'||Pc||_2 = {norm:.1f}: contact damping too small, '
                        'the arm feedforward will be amplified into '
                        'instability (see CODE_PAPER_AUDIT.md second audit)')

    def test_empty_contact_set_gives_identity(self):
        from wbc_core import get_contact_consistent_projector
        M, _ = self._stance()
        Pc = get_contact_consistent_projector(M, np.zeros((0, M.shape[0])))
        np.testing.assert_allclose(Pc, np.eye(M.shape[0]))


class EndToEndClaimTests(unittest.TestCase):
    """End-to-end assertions binding the manuscript's headline claims.

    Review: the suite "did not cover full scenarios, feedforward agreement,
    ... or finite trajectory behavior", and warned against claiming numerical
    robustness without tests that reject non-finite intermediates.
    """

    def test_scenario_a_states_stay_finite_under_strict_numpy(self):
        import scenario_a as sa
        old = np.seterr(all='raise')
        old_n = sa.N_RUN
        try:
            sa.N_RUN = 1200
            for name in ('D5 Proposed noKalman', 'D7 Proposed Full'):
                _, e = sa.run_controller(name, sa.CONTROLLERS[name])
                self.assertTrue(np.all(np.isfinite(e)), name)
                # Bounded, not merely finite: the divergence the second audit
                # found produced finite-but-runaway errors (>1000 mm).
                self.assertLess(np.abs(e).max() * 1000, 100.0, name)
        finally:
            np.seterr(**old)
            sa.N_RUN = old_n

    def test_observer_rejects_sustained_offset(self):
        """Paper's central claim: the Kalman path removes the 10 mm offset."""
        import scenario_a as sa
        old_n = sa.N_RUN
        try:
            sa.N_RUN = 3000
            _, e_pd = sa.run_controller('D1 SK05 PD',
                                        sa.CONTROLLERS['D1 SK05 PD'])
            _, e_full = sa.run_controller('D7 Proposed Full',
                                          sa.CONTROLLERS['D7 Proposed Full'])
        finally:
            sa.N_RUN = old_n
        ss = lambda e: np.sqrt(np.mean(np.sum(e[-500:]**2, axis=1))) * 1000
        # D1 sits at the analytical F/K_x = 8/800 = 10 mm impedance offset.
        self.assertAlmostEqual(ss(e_pd), 10.0, delta=2.0)
        # D7 must reject it by at least an order of magnitude.
        self.assertLess(ss(e_full), ss(e_pd) / 10.0)


class ReproducibilityTests(unittest.TestCase):
    """Review: 'the G1 experiment is not reproducible from this checkout'."""

    def test_g1_xml_contains_no_absolute_path(self):
        from pathlib import Path
        xml = Path(scenario_c_g1.MODEL_PATH).read_text()
        self.assertIn('@MENAGERIE_G1_ASSETS@', xml)
        self.assertNotIn('/Users/', xml)
        self.assertNotIn('/home/', xml)

    def test_g1_assets_resolve_and_load(self):
        assets = scenario_c_g1._resolve_menagerie_assets()
        self.assertTrue(assets.is_dir())
        model, _ = scenario_c_g1._make_robot()
        # Values the paper quotes for the Menagerie-derived G1.
        self.assertEqual(model.nu, 29)
        self.assertEqual(model.nv, 35)
        self.assertAlmostEqual(float(sum(model.body_mass)), 34.04, places=2)

    def test_bad_asset_override_raises_clear_error(self):
        import os
        old = os.environ.get('MENAGERIE_G1_ASSETS')
        os.environ['MENAGERIE_G1_ASSETS'] = '/nonexistent/g1/assets'
        try:
            with self.assertRaises(FileNotFoundError):
                scenario_c_g1._resolve_menagerie_assets()
        finally:
            if old is None:
                del os.environ['MENAGERIE_G1_ASSETS']
            else:
                os.environ['MENAGERIE_G1_ASSETS'] = old


if __name__ == "__main__":
    unittest.main()
