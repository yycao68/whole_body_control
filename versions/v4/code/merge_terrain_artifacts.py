#!/usr/bin/env python3
"""Merge independently run terrain cells into the authoritative artifact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from run_uneven_ground_benchmark import CONTROLLERS, RESULTS, aggregate


TERRAINS = ("flat", "depression", "obstacle", "rough")
EXPECTED_SEEDS = set(range(4200, 4210))


def main() -> None:
    trials = []
    metadata = None
    sources = {}
    for terrain in TERRAINS:
        path = RESULTS / f"terrain_{terrain}_corrected.json"
        raw = path.read_bytes()
        data = json.loads(raw)
        rows = data["trials"]
        if {row["terrain"] for row in rows} != {terrain}:
            raise RuntimeError(f"{path.name}: mixed or wrong terrain")
        for controller in CONTROLLERS:
            cell = [row for row in rows if row["controller"] == controller]
            if len(cell) != 10 or {row["seed"] for row in cell} != EXPECTED_SEEDS:
                raise RuntimeError(f"{path.name}: incomplete {controller} cell")
        current = data["metadata"]
        if metadata is None:
            metadata = current
        elif current != metadata:
            raise RuntimeError(f"{path.name}: configuration differs from other terrains")
        sources[path.name] = hashlib.sha256(raw).hexdigest()
        trials.extend(rows)

    artifact = aggregate(trials)
    artifact["metadata"] = metadata
    artifact["source_artifacts_sha256"] = sources
    output = RESULTS / "uneven_ground_benchmark.json"
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"saved {output} with {len(trials)} validated trials")


if __name__ == "__main__":
    main()
