# Whole-Body Control V3

V3 develops a dual interaction-dynamics architecture:

- Level 1: centroidal double-integrator MPC;
- Level 2: instantaneous whole-body physical interface;
- Level 3: task double-integrator MPC.

## Contents

- `wbc_ieee.md`: new manuscript draft.
- `BENCHMARK_PLAN.md`: detailed Unitree G1 comparison protocol.
- `code/`: self-contained v3 implementation package and local G1 model.

## Status

The formulation and benchmark are drafted. The v3 controller and benchmark
results are not yet implemented. Numerical results from v2 must not be
reported as v3 results.

## Immediate Milestone

Implement fixed-double-support S1 first and verify that both MPC layers use
the exact-ZOH normalized model. Contact detection and dynamic walking follow
only after S1 passes.
