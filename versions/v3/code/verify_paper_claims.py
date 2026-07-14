#!/usr/bin/env python3
"""Verify every quantitative claim the paper actually makes.

The previous verifier (``verify_v3_artifacts.py``) checks the H1-H6 hypothesis
suite, which the paper no longer contains; it is kept only as a legacy check on
those older artifacts.  This script checks the claims in the current text.

It is deliberately strict about the two things that have gone wrong before in
this project:

  * a number in the prose that no artifact supports, and
  * a claim ("sound", "one QP per cycle") that was only ever measured in the
    regime where it happens to hold.

Exit code 0 = every claim checked passes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

FAILURES: list[str] = []
CHECKS = 0


def check(cond: bool, claim: str, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  ok   {claim}")
    else:
        print(f"  FAIL {claim}" + (f"  [{detail}]" if detail else ""))
        FAILURES.append(claim)


def main() -> int:
    path = RESULTS / "multirate_benchmark.json"
    if not path.exists():
        print("missing results/multirate_benchmark.json -- run run_multirate_benchmarks.py")
        return 1
    d = json.loads(path.read_text())

    print("\n[E1] the 1 kHz loop solves exactly ONE whole-body QP per cycle")
    e1 = d["E1_realtime"]
    check(e1["whole_body_qp_solves_per_cycle"]["max"] == 1,
          "max whole-body QP solves per 1 kHz cycle == 1",
          str(e1["whole_body_qp_solves_per_cycle"]))
    check(e1["authority_kkt_ms"]["median_ms"] < 1.0,
          "authority (KKT) median < 1 ms",
          f"{e1['authority_kkt_ms']['median_ms']:.3f} ms")
    # The paper does NOT claim the Python prototype hits 1 kHz; it claims the
    # algorithmic content does.  Assert the honest version.
    check(e1["realization_cycle_ms"]["median_ms"] > 1.0,
          "prototype cycle EXCEEDS 1 ms (paper must not claim otherwise)",
          f"{e1['realization_cycle_ms']['median_ms']:.3f} ms")

    print("\n[E3] canonical (A,B,Hessian) invariant while H_k,h_k move")
    e3 = d["E3_occupancy"]
    check(e3["all_canonical_invariant"] is True, "canonical matrices bitwise invariant")
    rows = {r["scenario"]: r for r in e3["rows"]}
    nom = rows["double_support_nominal"]
    acc = rows["double_support_accel_ref"]
    check(acc["axis_extent_upper"][0] < 0.1 * nom["axis_extent_upper"][0],
          "a 0.8 m/s^2 reference feedforward consumes forward authority",
          f"{nom['axis_extent_upper'][0]} -> {acc['axis_extent_upper'][0]}")
    check(rows["single_support_left"]["valid"] is False,
          "unprepared single support yields an EMPTY set (nominal residual > tol)")

    print("\n[E5] 1-cell analytic is sound-but-conservative; continuation is exact")
    for r in d["E5_mapping_fidelity"]["rows"]:
        a, c = r["analytic"], r["continuation"]
        check(a["false_positive_rate"] == 0.0,
              f"{r['scenario']}: analytic false-positive rate == 0")
        check(a["false_negative_rate"] > 0.5,
              f"{r['scenario']}: analytic refuses >50% of feasible commands",
              str(a["false_negative_rate"]))
        check(c["false_positive_rate"] == 0.0,
              f"{r['scenario']}: CONTINUATION false-positive rate == 0")
        check(c["false_negative_rate"] == 0.0,
              f"{r['scenario']}: CONTINUATION false-negative rate == 0 (recovers the whole set)",
              str(c["false_negative_rate"]))
        check(c["whole_body_qp"] == 0,
              f"{r['scenario']}: continuation costs ZERO extra whole-body QP solves")
        check(r["speedup_continuation_over_oracle"] > 5,
              f"{r['scenario']}: continuation is >5x faster than the oracle",
              f"{r['speedup_continuation_over_oracle']}x")

    print("\n[E2] mapped authority reduces the realization residual in double support")
    e2 = d["E2_fixed_vs_mapped"]
    c1, c2 = e2["C1_fixed_box"], e2["C2_analytic_authority"]
    check(c2["median_residual"] < c1["median_residual"],
          "C2 median realization residual < C1",
          f"{c2['median_residual']:.3f} < {c1['median_residual']:.3f}")
    check(c2["rms_planar_error_mm"] > c1["rms_planar_error_mm"],
          "C2 tracking is WORSE (the honest cost; paper must not claim a free win)",
          f"{c2['rms_planar_error_mm']:.1f} > {c1['rms_planar_error_mm']:.1f} mm")

    print("\n[E4] speed vs tightness on the support transition")
    e4 = d["E4_contact_switch"]
    ex = e4["exact_authority"]; an = e4["analytic_authority"]
    co = e4["continuation_authority"]
    check(ex["fell"] is False, "exact-query authority sustains the transition")
    check(an["fell"] is True,
          "1-cell analytic authority does NOT (too conservative to balance on one foot)")
    check(co["fell"] is False,
          "CONTINUATION authority sustains the transition (the open problem, closed)")
    check(co["active_query_whole_body_qp_solves"] == 0,
          "continuation costs ZERO extra whole-body QP solves")
    check(ex["active_query_whole_body_qp_solves"] > 50,
          "exact oracle costs ~62 extra whole-body QP solves",
          str(ex["active_query_whole_body_qp_solves"]))
    check(co["active_query_ms_median"] < 0.25 * ex["active_query_ms_median"],
          "continuation is materially faster than the oracle",
          f"{co['active_query_ms_median']:.1f} vs {ex['active_query_ms_median']:.1f} ms")
    check(all(v["canonical_matrices_bitwise_invariant"] is True for v in (an, ex, co)),
          "canonical matrices invariant across the contact switch, all three sources")

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("\nFAILED CLAIMS:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("PASS: every claim the paper makes is supported by the artifacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
