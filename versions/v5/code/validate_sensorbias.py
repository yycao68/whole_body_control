"""Step V5 — sensor-bias / noise robustness.

The estimator subtracts measured foot force, so a horizontal foot-force bias
biases F_ext directly. Since sustained forces are only 8-12 N and f_thresh=3 N,
a few N of bias could corrupt detection. Sweep lateral GRF bias b_F and add
measurement noise; report nominal (does bias trigger spurious hold?), sustained
8 N offset, and transient 300 N latSS falls, for the wrench-unified controller.
20 seeds. (Nominal is protected by the |e|>deadband gate; this confirms it.)
"""
import json
from pathlib import Path
import numpy as np
import stage2_id_on_policy as S

HERE = Path(__file__).resolve().parent
SEEDS = list(range(2000, 2020))


def sustained_offset(bias, noise):
    v = []
    for s in SEEDS:
        r = S.run("id_mpc", push_n=8, push_t=3.0, push_dir=(0, 1), push_dur=3.0,
                  push_phase="time", duration=8.0, seed=s, process_noise=4,
                  id_mode="wrench", grf_bias=(0.0, bias), grf_noise=noise)
        if not r["fell"]:
            v.append(abs(r["lat_offset_mm"]))
    return float(np.median(v)) if v else float("nan")


def transient_falls(bias, noise):
    return sum(int(S.run("id_mpc", push_n=300, push_dir=(0, 1), push_phase="SS",
               seed=s, process_noise=6, id_mode="wrench",
               grf_bias=(0.0, bias), grf_noise=noise)["fell"]) for s in SEEDS)


def nominal_roll(bias, noise):
    # no applied force; does the bias falsely engage the controller (deadband gate)?
    rolls = []
    for s in SEEDS[:5]:
        r = S.run("id_mpc", push_n=0, duration=6.0, seed=s, process_noise=6,
                  id_mode="wrench", grf_bias=(0.0, bias), grf_noise=noise)
        rolls.append(r["peak_roll_deg"] if not r["fell"] else 90.0)
    return float(np.median(rolls))


def main():
    out = {"bias_sweep": {}, "noise": {}}
    print("Bias sweep (noise=0): lateral foot-force bias b_F [N]")
    print(f"{'b_F':>6}{'nominal_roll':>14}{'sustained8N_off':>18}{'transient300_falls':>20}")
    for b in (-5, -3, 0, 3, 5):
        nr = nominal_roll(b, 0.0); so = sustained_offset(b, 0.0); tf = transient_falls(b, 0.0)
        out["bias_sweep"][b] = {"nominal_roll": nr, "sustained8N_offset": so, "transient300_falls": tf}
        print(f"{b:>6}{nr:>12.1f}°{so:>16.0f}mm{tf:>16d}/{len(SEEDS)}", flush=True)

    print("\nMeasurement noise (bias=0): GRF noise std [N]")
    print(f"{'noise':>6}{'nominal_roll':>14}{'sustained8N_off':>18}{'transient300_falls':>20}")
    for nz in (0, 3, 6):
        nr = nominal_roll(0.0, nz); so = sustained_offset(0.0, nz); tf = transient_falls(0.0, nz)
        out["noise"][nz] = {"nominal_roll": nr, "sustained8N_offset": so, "transient300_falls": tf}
        print(f"{nz:>6}{nr:>12.1f}°{so:>16.0f}mm{tf:>16d}/{len(SEEDS)}", flush=True)

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "sensorbias_validation.json").write_text(json.dumps(out, indent=2))
    print("saved results/sensorbias_validation.json")


if __name__ == "__main__":
    main()
