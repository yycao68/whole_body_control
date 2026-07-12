#!/usr/bin/env python3
"""Deterministic v3 benchmark entrypoint.

Runs the full reproducibility suite for the paper
"Interaction Dynamics for Floating-Base Whole-Body Manipulation" in a fixed
order, regenerating every artifact under ``results/`` that the paper figures
and ``verify_v3_artifacts.py`` depend on, then runs the verifier.

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
    ("H1 multi-robot predictor invariance", ["run_h1_multirobot.py"]),
    ("H1/H2 equivalence + offset-free regulation", ["run_h1_h2.py"]),
    ("H3 arm-reaction preview", ["run_h3_coupling.py"]),
    ("H4 contact-event detection", ["run_h4_detection.py"]),
    ("H5 constraint recovery", ["run_h5_constraints.py"]),
    ("H6 interaction layer on a moving base", ["run_h6_onbase.py"]),
    ("Torque realizer smoke (stand)",
     ["run_g1_torque_realizer_benchmark.py", "--scenario", "stand",
      "--duration", "3.0", "--trials", "1", "--seed", "31"]),
    ("Torque realizer smoke (stand_push)",
     ["run_g1_torque_realizer_benchmark.py", "--scenario", "stand_push",
      "--duration", "3.0", "--trials", "3", "--seed", "21"]),
    ("Root-assisted walking visualization (Appendix A.1)",
     ["run_g1_root_assist_demo.py"]),
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
        if not run_step("Verify artifacts", ["verify_v3_artifacts.py"]):
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
