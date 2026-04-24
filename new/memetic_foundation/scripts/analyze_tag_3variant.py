"""
analyze_tag_3variant.py
Analyze results from run_tag_3variant_suite.sh

Usage:
    python3.9 -m new.memetic_foundation.scripts.analyze_tag_3variant
"""
import json
import os
import glob
import numpy as np

BASE_DIR = "new/memetic_foundation/checkpoints/tag_3variant"
VARIANTS = ["baseline", "memory_only", "commnet"]
SEEDS = [1, 2, 3, 4, 5]

def load_results(variant, seed):
    run_dir = os.path.join(BASE_DIR, f"{variant}_seed{seed}")
    # Find results json
    jsons = glob.glob(os.path.join(run_dir, "**", "*results.json"), recursive=True)
    if not jsons:
        return None
    with open(jsons[0]) as f:
        return json.load(f)

def load_log_rewards(variant, seed):
    """Parse reward from training log."""
    log_path = os.path.join(BASE_DIR, f"{variant}_seed{seed}.log")
    if not os.path.exists(log_path):
        return None, None
    rewards = []
    win_rates = []
    entropies = []
    with open(log_path) as f:
        for line in f:
            if line.startswith("Step") and "reward=" in line:
                try:
                    parts = {k.strip(): v.strip() for k, v in
                             (p.split("=") for p in line.split("|") if "=" in p)}
                    rew = float(parts.get("reward", "nan").replace(",", ""))
                    wr = float(parts.get("win_rate", "0%").replace("%", ""))
                    ent = float(parts.get("entropy", "nan"))
                    rewards.append(rew)
                    win_rates.append(wr)
                    entropies.append(ent)
                except Exception:
                    pass
    return rewards, win_rates, entropies

print("=" * 60)
print("Simple Tag 3-Variant Suite Results")
print("Task: predator-prey (3 predators, 1 prey)")
print("Params: ~92k each | ent_coef=0.3 | 200k steps")
print("=" * 60)

all_data = {}

for variant in VARIANTS:
    final_rewards = []
    final_wrs = []
    final_entropies = []
    n_done = 0
    n_collapsed = 0  # entropy < 0.5 in final 10 evals

    for seed in SEEDS:
        rewards, win_rates, entropies = load_log_rewards(variant, seed)
        if rewards is None or len(rewards) == 0:
            continue
        n_done += 1
        final_rew = np.mean(rewards[-5:]) if len(rewards) >= 5 else rewards[-1]
        final_wr = np.mean(win_rates[-5:]) if len(win_rates) >= 5 else win_rates[-1]
        final_ent = np.mean(entropies[-5:]) if len(entropies) >= 5 else entropies[-1]
        final_rewards.append(final_rew)
        final_wrs.append(final_wr)
        final_entropies.append(final_ent)
        if final_ent < 0.5:
            n_collapsed += 1

    all_data[variant] = {
        "rewards": final_rewards,
        "win_rates": final_wrs,
        "entropies": final_entropies,
        "n_done": n_done,
        "n_collapsed": n_collapsed,
    }

# Print table
print(f"\n{'Variant':<16} {'Seeds':>5} {'Mean Rew':>10} {'Median Rew':>11} {'Win Rate%':>10} {'Entropy':>8} {'Collapsed':>10}")
print("-" * 75)

for variant in VARIANTS:
    d = all_data[variant]
    if d["n_done"] == 0:
        print(f"{variant:<16} {'(no data)':>50}")
        continue
    mean_r = np.mean(d["rewards"]) if d["rewards"] else float("nan")
    med_r = np.median(d["rewards"]) if d["rewards"] else float("nan")
    mean_wr = np.mean(d["win_rates"]) if d["win_rates"] else float("nan")
    mean_ent = np.mean(d["entropies"]) if d["entropies"] else float("nan")
    print(f"{variant:<16} {d['n_done']:>5} {mean_r:>10.1f} {med_r:>11.1f} "
          f"{mean_wr:>10.1f} {mean_ent:>8.3f} {d['n_collapsed']:>5}/{d['n_done']}")

# Per-seed breakdown
print("\n--- Per-seed final reward ---")
print(f"{'Variant':<16}", end="")
for s in SEEDS:
    print(f"  seed{s}", end="")
print()
print("-" * 55)
for variant in VARIANTS:
    d = all_data[variant]
    print(f"{variant:<16}", end="")
    for i, s in enumerate(SEEDS):
        rewards, _, _ = load_log_rewards(variant, s)
        if rewards and len(rewards) > 0:
            v = np.mean(rewards[-5:]) if len(rewards) >= 5 else rewards[-1]
            print(f"  {v:>5.0f}", end="")
        else:
            print(f"  {'---':>5}", end="")
    print()

# Entropy trajectory (first/mid/last)
print("\n--- Entropy at eval 1 / mid / final ---")
print(f"{'Variant':<16} {'seed':>5}  {'E@1':>6}  {'E@mid':>6}  {'E@fin':>6}")
print("-" * 50)
for variant in VARIANTS:
    for seed in SEEDS:
        _, _, entropies = load_log_rewards(variant, seed)
        if entropies and len(entropies) >= 3:
            e1 = entropies[0]
            emid = entropies[len(entropies)//2]
            efin = entropies[-1]
            flag = " *** COLLAPSE" if efin < 0.5 else ""
            print(f"{variant:<16} {seed:>5}  {e1:>6.3f}  {emid:>6.3f}  {efin:>6.3f}{flag}")

print("\nNote: Reward for predators (higher = more catches).")
print("Win rate = episode fraction where prey was tagged >= 1 time.")
