# Code-Paper Consistency Audit

Audit date: 2026-07-09

## Confirmed fixes

- The QP now optimizes residual Cartesian acceleration with constant exact-ZOH
  matrices. The old force-input, inertia-dependent `B_d` implementation did
  not match Section V.
- The Kalman filter now predicts with acceleration input and estimates an
  acceleration disturbance. Scenario drivers pass `mpc.last_u`, not recovered
  force, to the estimator.
- Corrective force is recovered as `Lambda_arm @ u`; force constraints update
  with the current task inertia while the Hessian remains constant.
- OSQP now updates its constraint matrix values when task inertia changes.
- The Unitree G1 benchmark now overrides the XML's 2 ms timestep with 0.5 ms.
  Before this fix, each nominal 1 ms iteration advanced 4 ms of physics.
- The paper no longer says that the normalized `B_d`, lifted rollout, Hessian,
  or Kalman matrices switch with contact mode.
- The arm torque realization no longer adds the recovered MPC force twice.

## Verified results after correction

All entries below were rerun from the final source on the audit date.

| Experiment | Key result |
| --- | --- |
| Scenario A | D7 SS 0.079 mm; D5 SS 13.21 mm |
| Scenario B | D7 RMS 3.17 mm; D5 RMS 14.41 mm |
| Scenario C, corrected timing | D7 SS 1.589 mm; D5 SS 26.37 mm |
| Scenario E | D7 RMS 2.53 mm; D6 RMS 2.69 mm |
| Scenario F | D7 RMS 10.88 mm; D6 RMS 10.95 mm |
| Gain convergence | relative error 0.671 at N=20, 0.0246 at N=80 |

## Remaining limitations

- `get_contact_consistent_inverse` uses contact damping 0.1 and task-mobility
  eigenvalue clamping. Contact decoupling is therefore approximate in
  simulation; the exact theorem applies to the unregularized model.
- The simulation WBC uses joint PD balance and initial-pose arm gravity
  compensation rather than the full model feedforward in (18).
- Scenario F is a scheduled model switch, not physical single support: the
  left foot remains in floor contact for 90.4% of the nominal single-support
  interval.
- `level1_centroidal.py` is an instantaneous wrench QP, not centroidal MPC.
  Its balance-only Scenario G falls before the first support switch.
- The unconstrained fast path is used in reported experiments. Force-row code
  is tested, but active-limit experimental results are not reported.
- No hardware experiment or dynamic walking result is currently supported.

Run the regression audit with:

```bash
cd whole_body_control/versions/v2_strong/code
python3 -m unittest -v test_code_paper_consistency.py
```
