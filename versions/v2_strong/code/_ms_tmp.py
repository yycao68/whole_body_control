import scenario_a as sa
print("=== D8 vs D4 ===", flush=True)
print("max |D4-D8| =", sa.verify_d8_matches_d4(), "mm", flush=True)
print("\n=== 10-seed jitter ===", flush=True)
sa.run_multiseed(n_seeds=10, base_seed=4300)
