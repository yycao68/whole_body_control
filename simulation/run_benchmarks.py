"""
Run all benchmarks and produce the comparison tables from the IEEE paper.

Usage:
    python run_benchmarks.py [--scenario a|b|c|all] [--walking]

Results are saved to simulation/results/
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / 'simulation' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_scenario_a():
    from simulation.scenarios.scenario_a import run_all
    return run_all()


def run_scenario_b():
    from simulation.scenarios.scenario_b import run_all
    return run_all()


def run_scenario_c():
    # Paper Table C is the Unitree G1 official model (33.3 kg); scenario_c.py is the
    # R1-approximate (28.7 kg) mass-generalization study, run separately.
    from simulation.scenarios.scenario_c_g1 import run_all
    return run_all()


def run_walking():
    from simulation.walking_demo import run_walking_demo
    return run_walking_demo()


def print_paper_table(results_a, results_b):
    """Print Table III and Table IV in a format matching the IEEE paper."""
    print("\n" + "="*70)
    print("TABLE III: Scenario A — Fixed Stance, 8 N Step Disturbance")
    print("="*70)
    print(f"{'Controller':<28} {'RMS [mm]':>10} {'SS err [mm]':>12}")
    print("-"*54)
    for name, res in results_a.items():
        print(f"{name:<28} {res['rms']:>10.2f} {res['ss']:>12.3f}")
    d1_ss = results_a['D1 SK05 PD']['ss']
    d7_ss = results_a['D7 Proposed Full']['ss']
    print(f"\nImprovement D7 vs D1 (SS error): {d1_ss/max(d7_ss,0.001):.0f}x")

    print("\n" + "="*70)
    print("TABLE IV: Scenario B — Stance + 1 Hz Contact-Transition Shocks, Sustained 8 N pHRI")
    print("="*70)
    print(f"{'Controller':<28} {'RMS [mm]':>10} {'Peak@trans [mm]':>18}")
    print("-"*60)
    for name, res in results_b.items():
        print(f"{name:<28} {res['rms']:>10.2f} {res['peak']:>18.2f}")
    d1_rms = results_b['D1_SK05_PD']['rms']
    d7_rms = results_b['D7_Proposed_Full']['rms']
    d6_pk  = results_b['D6_Proposed_noInflat']['peak']
    d7_pk  = results_b['D7_Proposed_Full']['peak']
    # Note: D7/D6 ratio < 1 when inflation helps, > 1 when it hurts
    print(f"\nImprovement D7 vs D1 (RMS): {d1_rms/max(d7_rms,0.01):.1f}x")
    print(f"Covariance inflation D6→D7 (peak): {d6_pk/max(d7_pk,0.01):.1f}x")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', choices=['a', 'b', 'c', 'all'], default='all')
    parser.add_argument('--walking', action='store_true',
                        help='Also run walking demo and record video')
    args = parser.parse_args()

    results_a = results_b = results_c = None

    if args.scenario in ('a', 'all'):
        results_a = run_scenario_a()

    if args.scenario in ('b', 'all'):
        results_b = run_scenario_b()

    if args.scenario in ('c', 'all'):
        results_c = run_scenario_c()

    if results_a and results_b:
        print_paper_table(results_a, results_b)

    if results_c:
        print("\n" + "="*70)
        print("TABLE V: Scenario C — Unitree G1 (33.3 kg), Fixed Stance, 8 N Step")
        print("="*70)
        print(f"{'Controller':<28} {'RMS [mm]':>10} {'SS err [mm]':>12}")
        print("-"*54)
        for name, res in results_c.items():
            print(f"{name:<28} {res['rms']:>10.3f} {res['ss']:>12.4f}")
        d1 = results_c['D1 SK05 PD']['ss']
        d7 = results_c['D7 Proposed Full']['ss']
        print(f"\nImprovement D7 vs D1 (SS error): {d1/max(d7,0.001):.0f}x")

    if args.walking:
        print("\n--- Running walking demo (this may take a few minutes) ---")
        run_walking()

    print(f"\nAll results saved to {OUT_DIR}")
