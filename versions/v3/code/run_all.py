#!/usr/bin/env python3
"""Deterministic v3 benchmark entrypoint.

Runs the full reproducibility suite for the paper
"Interaction Dynamics for Floating-Base Whole-Body Manipulation" in a fixed
order, regenerating every artifact the paper cites, then runs
``verify_paper_claims.py``.



Usage:
    python3 run_all.py                # run everything, then verify
    python3 run_all.py --skip-verify  # regenerate artifacts only
    python3 run_all.py --keep-going   # do not stop on the first failure

Each step is executed as an isolated subprocess so a failure in one runner
cannot corrupt the interpreter state of the others. ``MPLCONFIGDIR`` is set to
a writable temp dir so Matplotlib never touches the user's home cache.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (label, [script, *args]) in deterministic execution order. The two torque
# invocations mirror the documented stand / stand_push smoke gates (H2-H5).
STEPS: list[tuple[str, list[str]]] = [
    # --- the paper's pipeline -------------------------------------------------
    ("E1-E5 multirate architecture benchmark", ["run_multirate_benchmarks.py"]),
    ("E4 support transition, exact-query authority",
     ["run_authority_transition.py", "--authority", "exact",
      "--trials", "4", "--seed", "100"]),
    ("E4 support transition, analytic KKT authority (expected negative)",
     ["run_authority_transition.py", "--authority", "analytic",
      "--trials", "4", "--seed", "100"]),

]




def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")
    return env


def run_step(label: str, argv: list[str]) -> bool:
    script = HERE / argv[0]
    if not script.exists():
        print(f"  ! missing script: {argv[0]}", flush=True)
        return False
    cmd = [sys.executable, str(script), *argv[1:]]
    print(f"\n==> {label}\n    {' '.join(cmd[1:])}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=HERE, env=_env())
    dt = time.time() - t0
    ok = result.returncode == 0
    print(f"    {'ok' if ok else 'FAILED'} ({dt:.1f}s)", flush=True)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-verify", action="store_true",
                        help="regenerate artifacts without running the verifier")
    parser.add_argument("--keep-going", action="store_true",
                        help="continue after a failing step instead of stopping")
    args = parser.parse_args()

    failures: list[str] = []
    for label, argv in STEPS:
        if not run_step(label, argv):
            failures.append(label)
            if not args.keep_going:
                print(f"\nStopping: '{label}' failed (use --keep-going to continue).")
                return 1

    if not args.skip_verify and not failures:
        if not run_step("Verify artifacts", ["verify_paper_claims.py"]):
            failures.append("Verify artifacts")

    print("\n" + "=" * 60)
    if failures:
        print("FAILED steps:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS: all v3 benchmarks regenerated and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
