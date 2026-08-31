import numpy as np

from normalized_mpc import NormalizedMPC, affine_output_box_polytope


def test_affine_output_polytope_includes_full_offset():
    torque_map = np.array([[2.0], [-1.0]])
    torque_offset = np.array([4.0, -3.0])
    torque_min = np.array([-5.0, -5.0])
    torque_max = np.array([5.0, 5.0])

    H, h = affine_output_box_polytope(
        torque_map, torque_offset, torque_min, torque_max
    )

    np.testing.assert_allclose(H, np.array([[2.0], [-1.0], [-2.0], [1.0]]))
    np.testing.assert_allclose(h, np.array([1.0, 8.0, 9.0, 2.0]))
    assert np.all(H @ np.array([0.5]) <= h)
    assert np.any(H @ np.array([1.0]) > h)


def test_polytope_constrains_each_horizon_stage_and_uses_feasible_fallback():
    mpc = NormalizedMPC(
        dim=1, dt=0.1, horizon=4, q_pos=1.0, q_vel=1.0, r=0.1
    )
    H, h = affine_output_box_polytope(
        np.array([[1.0]]), np.array([0.75]), np.array([-1.0]), np.array([1.0])
    )
    mpc.update_input_polytope(H, h, fallback_input=np.array([0.25]))
    mpc.solve(np.array([4.0, 0.0]))

    assert mpc.last_u_sequence is not None
    assert mpc.last_u_sequence.shape == (4, 1)
    assert np.all(H @ mpc.last_u_sequence.T <= h[:, None] + 1e-6)


def test_polytope_projects_an_infeasible_fallback_command():
    mpc = NormalizedMPC(
        dim=1, dt=0.1, horizon=3, q_pos=1.0, q_vel=1.0, r=0.1
    )
    H = np.array([[2.0], [-1.0]])
    h = np.array([0.5, 0.25])
    mpc.update_input_polytope(
        H, h,
        fallback_input=np.array([1.0]),
    )

    np.testing.assert_allclose(mpc._poly_fallback, np.array([0.25]), atol=1e-5)
    assert np.all(H @ mpc._poly_fallback <= h + 1e-6)