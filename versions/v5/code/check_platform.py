#!/usr/bin/env python3
"""Preflight check: can this checkout actually run the V5 experiments?

External review (2026-08-30) could not run the policy loop and hit the
prerequisites one at a time -- first `torch`, then the frozen reference, then
the meshes. This script reports every missing prerequisite at once, with the
exact command to fix each, and exits nonzero if anything is missing.

Run before any validation script:

    python3 check_platform.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MESH_DIR = HERE / "g1_description" / "meshes"
XML = HERE / "g1_description" / "g1_12dof.xml"
REF = HERE / "reference" / "frozen_walk_seed0.npz"
POLICY = HERE / "motion.pt"

# Third-party Python packages the publication path needs. Keep in sync with
# requirements.txt.
REQUIRED = ("numpy", "mujoco", "torch", "matplotlib", "scipy", "osqp")

# Expected identity of the regenerable frozen reference, so a stale or
# truncated copy is reported rather than silently used. See ASSETS.md.
REF_SHA256 = "ff5187be469ef53d499af984bc250272d2fb7bc14c35eb06d30cbc8db2ef1e33"
REF_BYTES = 3552081


def _missing_packages():
    return [p for p in REQUIRED if importlib.util.find_spec(p) is None]


def _referenced_meshes():
    """STL filenames the MJCF actually asks for."""
    if not XML.exists():
        return []
    import re
    return sorted(set(re.findall(r'file="([^"]+\.STL)"', XML.read_text())))


def main() -> int:
    problems = []

    missing = _missing_packages()
    if missing:
        problems.append(
            f"missing Python packages: {', '.join(missing)}\n"
            f"    pip install {' '.join(missing)}"
        )

    if not POLICY.exists():
        problems.append(f"missing frozen policy: {POLICY} (this one IS tracked in git)")

    wanted = _referenced_meshes()
    if wanted:
        env = os.environ.get("G1_MESH_DIR")
        mesh_dir = Path(env) if env else MESH_DIR
        absent = [m for m in wanted if not (mesh_dir / m).exists()]
        if absent:
            problems.append(
                f"missing {len(absent)}/{len(wanted)} G1 meshes in {mesh_dir}\n"
                f"    first missing: {absent[0]}\n"
                "    The repository gitignores *.STL, so a fresh clone has none.\n"
                "    These are the 12-DoF legs-only G1 meshes from unitree_rl_gym\n"
                "    (see FROZEN_PLATFORM.md); note 3 of them "
                "(torso_link_23dof_rev_1_0.STL and the two\n"
                "    *_wrist_roll_rubber_hand.STL) are NOT in mujoco_menagerie, so\n"
                "    robot_descriptions alone is not a sufficient source.\n"
                "    Copy g1_description/meshes/ from a unitree_rl_gym checkout, or\n"
                "    set $G1_MESH_DIR to a directory containing them."
            )

    if not REF.exists():
        problems.append(
            f"missing frozen nominal reference: {REF}\n"
            "    Gitignored (*.npz); regenerate once (needs meshes + torch):\n"
            "    python3 run_policy_walk.py --duration 20 --seeds 0 "
            "--save reference/frozen_walk_seed0.npz"
        )
    else:
        import hashlib
        got = hashlib.sha256(REF.read_bytes()).hexdigest()
        if got != REF_SHA256:
            problems.append(
                f"frozen reference does not match the recorded manifest: {REF}\n"
                f"    expected sha256 {REF_SHA256} ({REF_BYTES} bytes)\n"
                f"    found    sha256 {got} ({REF.stat().st_size} bytes)\n"
                "    Results generated against a different reference are not\n"
                "    comparable to the committed artifacts. Regenerate with the\n"
                "    command in ASSETS.md, or accept the difference knowingly."
            )

    if problems:
        print("PLATFORM INCOMPLETE -- the V5 experiments cannot run here.\n")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}\n")
        return 1

    print("PLATFORM OK: packages, policy, meshes, and frozen reference all present.")
    print("  meshes   :", MESH_DIR)
    print("  reference:", REF)
    print("  policy   :", POLICY)
    return 0


if __name__ == "__main__":
    sys.exit(main())
