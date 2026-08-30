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

All entries below were rerun from the final source on 2026-08-30 (second
audit, see "Second audit" below). The 2026-07-09 values they replace are kept
in parentheses.

| Experiment | Key result |
| --- | --- |
| Scenario A | D7 SS 0.139 mm (was 0.079); D5 SS 20.19 mm (was 13.21) |
| Scenario B | D7 RMS 4.37 mm (was 3.17); D5 RMS 21.19 mm (was 14.41) |
| Scenario C, corrected timing | D7 SS 0.884 mm (was 1.589); D5 SS 20.50 mm (was 26.37) |
| Scenario E | D7 RMS 4.32 mm (was 2.53); D6 RMS 4.52 mm (was 2.69) |
| Scenario F | D7 RMS 12.04 mm (was 10.88); D6 RMS 12.11 mm (was 10.95) |
| Gain convergence | relative error 0.671 at N=20, 0.0246 at N=80 (unchanged) |

## Second audit (2026-08-30)

Prompted by an external code-vs-paper review. Two P0 findings, both real:

- **The arm feedforward law in (18) was not implemented.** `impedance_mpc.py`
  computed only `Lambda_arm @ u`, silently dropping both the `Lambda_arm *
  p_ddot_d` and the `mu_arm` (task-space Coriolis/gravity bias) terms. Now
  implemented and wired through all five scenario drivers. `p_ddot_d` remains
  zero in every reported scenario, but legitimately so: each holds a *fixed*
  target (world-frame in A/B/C/E, torso-relative in F, where the relative
  Jacobian already absorbs torso motion), so the desired acceleration really
  is zero rather than being dropped.
- **Scenarios E and F used a raw `J^T F` torque map**, not the contact-
  consistent realization the paper claims. Both now project through
  `get_contact_consistent_projector` before extracting the arm rows.

Implementing the feedforward exposed a latent numerical bug that made the
MPC controllers diverge (>1000 mm). Root cause: `get_contact_consistent_
projector`'s `contact_damp` default of 1e-3, whose operator norm
(`||Pc||_2 = 27.6` at the double-support stance) amplified the newly-added
`mu_arm` correction by up to ~80x in some directions. Sharp bifurcation at
`contact_damp <= 0.003` (diverges) vs `>= 0.003` (stable). Raising the
default to 0.1 fixes it with ~30x margin and produced the reruns above.

`Pc` is oblique rather than orthogonal, so being a projector does *not*
bound its gain. Measured at that stance:

| `contact_damp` | `\|\|Pc\|\|_2` | `\|\|Pc - Pc@Pc\|\|` | max leaked contact eigenvalue |
| --- | ---: | ---: | ---: |
| 1e-8 | 28.0 | 0.000 | 0.000 |
| 1e-3 (old) | 27.6 | 0.396 | 0.018 |
| 1e-1 (shipped) | 12.6 | 6.825 | 0.650 |
| 1.0 | 4.4 | 3.000 | 0.949 |

So the damping buys stability by making the projection substantially
**approximate**, not by repairing it: only in the undamped limit is `Pc` an
exact projector (idempotent, eigenvalues exactly 0/1), and at the shipped
0.1 the six contact-direction eigenvalues have drifted from 0 to 0.65. This
is the honest characterization and the reason the contact-decoupling claim
must stay qualified. (An earlier draft of this note and of commit `1f4817c`'s
message asserted `Pc` was "idempotent at any damping with eigenvalues exactly
0/1"; that was wrong -- it holds only as `contact_damp -> 0`.)

The feedforward law's own derivation was independently re-verified as correct
(`get_site_jacobian_dot` matches finite differences to 8.3e-8; `mu_arm`
re-derives from `M qddot + h = tau`), so no sign or formula was changed.
Both ends of the damping trade-off are now pinned by
`ContactConsistencyTests` so neither silently regresses.

Net effect on the paper's claims: D5 (no observer) degrades noticeably almost
everywhere, since it now carries the real velocity feedforward and its
approximation residual with nothing to absorb it. D6/D7 stay strong. This
strengthens rather than weakens the central claim that the disturbance
observer drives robustness.

Also corrected in this audit:

- Scenario F's double-support inertia diagonal was reported as
  `[1.11, 1.04, 2.31]` kg; the actual value is `[1.11, 1.04, 2.30]`. The
  accompanying "less than one percent" claim understated the true worst-case
  change (1.7%) and now reads "less than two percent".
- Scenario E's table (11.26 mm) and prose (13.28 mm) disagreed with each
  other; both are superseded by the single rerun value.
- The paper stated a uniform 1 kHz interaction-layer rate; Scenario F
  actually runs at 2 kHz (`CTRL_DT = 0.0005`). Now stated per scenario.
- The paper's 500 Hz Level-2 rate and its frozen-matrix argument describe the
  intended hardware deployment only: the simulations recompute `N_aug`,
  `Lambda_arm`, and `mu_arm` every interaction-layer step, so that
  approximation is not exercised by these benchmarks. Now stated explicitly.
- `g1_wbc.xml` hardcoded an absolute `meshdir` under one developer's home
  cache. It now carries a `@MENAGERIE_G1_ASSETS@` placeholder resolved at load
  time (see Reproduction below).

## Remaining limitations

- `get_contact_consistent_inverse` and `get_contact_consistent_projector` both
  use contact damping 0.1, plus task-mobility eigenvalue clamping. Contact
  decoupling is therefore approximate in simulation; the exact theorem applies
  to the unregularized model. The damping is not merely a conditioning
  convenience: as the second audit found, `Pc` is ill-conditioned as an
  oblique projector (`||Pc||_2 = 12.6` even at 0.1), the closed loop is
  genuinely unstable below ~0.003, and at the shipped 0.1 the contact
  directions leak by 0.65 (see the table above). Contact decoupling in the
  reported simulations is therefore *substantially* approximate, not
  marginally so; claims of contact non-interference should not be read as
  validated by these runs.
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

## Reproduction

Run the regression audit with:

```bash
cd whole_body_control/versions/v2_strong/code
python3 -m unittest -v test_code_paper_consistency.py
```

Reproduce the reported tables (each prints its own table; Scenario B takes a
few minutes):

```bash
python3 scenario_a.py          # Table III
python3 scenario_b.py          # Table IV
python3 scenario_c_g1.py       # Table V   (needs G1 meshes, see below)
python3 scenario_brace.py      # Table VI  (Scenario E)
python3 scenario_qstatic.py    # Table VII (Scenario F)
python3 verify_impedance_equivalence.py   # gain-convergence diagnostics
python3 verify_inertia_normalization.py   # task-inertia diagnostics
```

**Python dependencies:** `mujoco`, `numpy`, `scipy`, `matplotlib`, `osqp`.
(`playground_g1_bridge.py` additionally imports `jax` / `mujoco_playground`,
but it is not part of any reported result.)
Scenario C additionally needs the Unitree G1 meshes, resolved in this order:

1. `$MENAGERIE_G1_ASSETS`, if set — point it at the `assets/` directory of an
   existing `mujoco_menagerie/unitree_g1` checkout.
2. Otherwise the `robot_descriptions` package (`pip install
   robot_descriptions`), which downloads and caches `mujoco_menagerie`
   automatically on first use.

`g1_wbc.xml` ships with the `@MENAGERIE_G1_ASSETS@` placeholder and is patched
in memory at load time, so no absolute path is baked into the repository.

**Building the paper:**

```bash
cd whole_body_control/versions/v2_strong/arXiv
latexmk -pdf arxiv_main.tex
```

Requires a LaTeX distribution providing `IEEEtran` plus `amsmath`, `amssymb`,
`amsfonts`, `amsthm`, `graphicx`, `booktabs`, `bm`, `cite`, and `hyperref`.
These are all in TeX Live's `collection-latexrecommended`; a minimal install
may need `tlmgr install cite booktabs IEEEtran bm`. Verified building clean
(9 pages, 0 undefined references, 0 overfull boxes) on TeX Live 2026.
