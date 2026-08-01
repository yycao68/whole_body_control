"""Re-run the sustained studies with the FLOOR-CORRECTED paired metric.

The raw `lat_offset` (CoM_y - recorded-ref_y) has a ~30 mm phase-drift floor
(robot desyncs from the fixed recorded reference under process noise). The paired
metric removes it exactly: for the SAME seed, run with force and with zero force
and take |offset_force - offset_noforce| — the sway/drift/reference cancel, leaving
the force-induced lateral drift. This supersedes the raw sustained numbers (V2/V4/
envelope/V5-sustained), which were floored.
"""
import json
from pathlib import Path
import numpy as np
import stage2_id_on_policy as S

HERE = Path(__file__).resolve().parent
ZERO = lambda t: (0.0, 0.0)


# Low process noise: at noise>=2 the capture path amplifies noise-induced lateral
# drift into runaway (>>the 8 N signal), contaminating the measurement; the
# hold-only path is immune. noise=1 gives seed variation with minimal runaway.
NOISE = 1


def paired_drift(kind, mode, force, seeds):
    """Median [IQR] floor-corrected force-induced lateral drift, over seeds."""
    d = []
    for s in seeds:
        rf = S.run(kind, push_n=force, push_t=3.0, push_dir=(0, 1), push_dur=3.0,
                   push_phase="time", duration=8.0, seed=s, process_noise=NOISE, id_mode=mode)
        r0 = S.run(kind, push_n=1.0, push_t=3.0, push_dir=(0, 1), push_dur=3.0,
                   push_phase="time", duration=8.0, seed=s, process_noise=NOISE, id_mode=mode,
                   force_override=ZERO)
        if not rf["fell"] and not r0["fell"]:
            d.append(abs(rf["lat_offset_mm"] - r0["lat_offset_mm"]))
    d = np.array(d)
    return (float(np.median(d)), float(np.percentile(d, 25)), float(np.percentile(d, 75)),
            len(d)) if len(d) else (float("nan"),) * 3 + (0,)


def main():
    out = {}
    CORE = [("policy", "policy", "policy"), ("capture-spec", "id_mpc", "transient"),
            ("hold-spec", "id_mpc", "sustained"), ("CoM-unified", "id_mpc", "unified"),
            ("wrench", "id_mpc", "wrench"), ("oracle", "id_mpc", "oracle")]
    seeds20 = list(range(2000, 2020))

    print("=== Floor-corrected sustained drift, median [IQR] mm (20 seeds) ===")
    print(f"{'controller':14s}{'8 N':>18s}{'12 N':>18s}")
    out["core"] = {}
    for label, kind, mode in CORE:
        m8, l8, h8, n8 = paired_drift(kind, mode, 8, seeds20)
        m12, l12, h12, n12 = paired_drift(kind, mode, 12, seeds20)
        out["core"][label] = {"8N": [m8, l8, h8, n8], "12N": [m12, l12, h12, n12]}
        print(f"{label:14s}{m8:8.0f} [{l8:.0f}-{h8:.0f}]{'':>2}{m12:8.0f} [{l12:.0f}-{h12:.0f}]", flush=True)

    print("\n=== Envelope: floor-corrected drift vs force (15 seeds) ===")
    seeds15 = list(range(2000, 2015))
    forces = [6, 10, 14, 18, 22]
    ENV = [("policy", "policy", "policy"), ("hold-spec", "id_mpc", "sustained"),
           ("wrench", "id_mpc", "wrench")]
    out["envelope"] = {"forces": forces}
    print(f"{'controller':12s}" + "".join(f"{f:>8}N" for f in forces))
    for label, kind, mode in ENV:
        row = [paired_drift(kind, mode, f, seeds15)[0] for f in forces]
        out["envelope"][label] = row
        print(f"{label:12s}" + "".join(f"{v:>8.0f} " for v in row), flush=True)

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "sustained_floorcorrected.json").write_text(json.dumps(out, indent=2))
    print("saved results/sustained_floorcorrected.json")


if __name__ == "__main__":
    main()
