"""Step V6 — momentum-observer reformulation vs raw finite-difference.

Re-runs the V1 scenarios with both estimators and compares the V1 metrics.
Goal: confirm the smoother momentum-residual observer (no CoM-accel diff) holds
or improves RMSE / detection / decay / false-positive.
"""
import json
from pathlib import Path
import numpy as np
import validate_estimator as V

HERE = Path(__file__).resolve().parent
step_up, iz_u = V.S.terrain_scene("up", 0.03)
step_dn, iz_d = V.S.terrain_scene("down", 0.03)
SCN = {
    "nominal":      dict(force=lambda t: (0, 0), cmd=lambda t: (0.5, 0, 0), has_force=False),
    "step_down_30": dict(force=lambda t: (0, 0), cmd=lambda t: (0.5, 0, 0), has_force=False,
                         scene=step_dn, iz=iz_d),
    "impulse_120":  dict(force=lambda t: (0, 120 if 3.0 <= t < 3.15 else 0),
                         cmd=lambda t: (0.5, 0, 0), has_force=True, onset=3.0, offset=3.15),
    "sustained_12": dict(force=lambda t: (0, 12 if 3.0 <= t < 6.0 else 0),
                         cmd=lambda t: (0.5, 0, 0), has_force=True, onset=3.0),
}


def main():
    out = {}
    print(f"{'scenario':14s}{'estimator':12s}{'RMSE':>8s}{'peak_err':>9s}"
          f"{'FP%':>7s}{'detect':>8s}{'decay':>8s}{'trk_rmse':>9s}")
    for name, cfg in SCN.items():
        for est, obs in (("finite-diff", False), ("observer", True)):
            T, Ft, Fe, fell = V.run_scenario(cfg["force"], cfg["cmd"], scene=cfg.get("scene"),
                                             init_z=cfg.get("iz", 0.793), observer=obs)
            m = V.metrics(T, Ft, Fe, cfg["has_force"], cfg.get("onset"), cfg.get("offset"), fell)
            out[f"{name}|{est}"] = m
            def g(k): return m.get(k)
            print(f"{name:14s}{est:12s}{m['rmse_N']:8.2f}{m['peak_err_N']:9.1f}"
                  f"{(g('false_pos_rate')*100 if g('false_pos_rate') is not None else float('nan')):6.1f}%"
                  f"{(g('detect_delay_s') if g('detect_delay_s') is not None else float('nan')):8.2f}"
                  f"{(g('decay_s') if g('decay_s') is not None else float('nan')):8.2f}"
                  f"{(g('track_rmse_N') if g('track_rmse_N') is not None else float('nan')):9.2f}", flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "observer_validation.json").write_text(json.dumps(out, indent=2))
    print("saved results/observer_validation.json")


if __name__ == "__main__":
    main()
