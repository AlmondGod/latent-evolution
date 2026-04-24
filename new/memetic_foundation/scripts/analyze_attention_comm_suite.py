"""
analyze_attention_comm_suite.py — Analyze attention communication architecture comparison.

Compares 5 variants across N={3,5,8}:
  baseline             — no memory, no comm
  memory_only          — GRU only
  ic3net               — binary gate + gate entropy, comm→GRU
  attention_integrated — soft attention, comm→GRU (no gate entropy)
  attention_separated  — soft attention, GRU=pure obs, actor=[u;h;c]

Key scientific questions:
  1. Does attention_integrated beat memory_only? (comm benefit without gate instability)
  2. Does attention_separated beat attention_integrated? (pure personal meme + social ctx)
  3. Does either attention variant beat baseline? (is comm ever helpful?)
  4. Does catastrophic failure rate drop vs ic3net?

Usage:
    /opt/homebrew/bin/python3.9 -m new.memetic_foundation.scripts.analyze_attention_comm_suite
"""

import os, re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "checkpoints/attention_comm_suite"
TOTAL_STEPS = 400000
CATASTROPHIC_DIST = 10.0  # dist threshold for catastrophic failure
N_VALUES = [3, 5, 8]

VARIANTS = {
    "baseline":             "baseline",
    "memory_only":          "memory_only",
    "ic3net":               "ic3net",
    "attn_integrated":      "attn_integrated",
    "attn_separated":       "attn_separated",
}

VARIANT_LABELS = {
    "baseline":         "Baseline",
    "memory_only":      "Memory Only",
    "ic3net":           "IC3Net",
    "attn_integrated":  "Attn Integrated",
    "attn_separated":   "Attn Separated",
}

COLORS = {
    "baseline":         "#888888",
    "memory_only":      "#2196F3",
    "ic3net":           "#F44336",
    "attn_integrated":  "#4CAF50",
    "attn_separated":   "#FF9800",
}


def parse_eval_dist(log_path):
    result = {}
    try:
        with open(log_path, errors="replace") as f:
            for line in f:
                m = re.search(r'\[Eval\] Step (\d+) \| .* dist=([-\d.]+)', line)
                if m:
                    result[int(m.group(1))] = float(m.group(2))
    except Exception:
        pass
    return result


def parse_eval_reward(log_path):
    result = {}
    try:
        with open(log_path, errors="replace") as f:
            for line in f:
                m = re.search(r'\[Eval\] Step (\d+) \| reward=([-\d.]+)', line)
                if m:
                    result[int(m.group(1))] = float(m.group(2))
    except Exception:
        pass
    return result


def collect_results():
    """
    Returns: dict[variant_key][N] = {
        'dists': list of final dists (one per seed),
        'rewards': list of final rewards (one per seed),
        'curves': list of {step: dist} dicts,
        'n_seeds': int,
        'n_catastrophic': int,
    }
    """
    results = {}
    for vk in VARIANTS:
        results[vk] = {}
        for n in N_VALUES:
            prefix = f"{VARIANTS[vk]}_n{n}"
            logs = sorted(glob.glob(f"{BASE_DIR}/{prefix}_seed*.log"))
            dists, rewards, curves = [], [], []
            for log in logs:
                d = parse_eval_dist(log)
                r = parse_eval_reward(log)
                if not d:
                    continue
                curves.append(d)
                # final = last eval step
                final_step = max(d.keys())
                dists.append(d[final_step])
                if r:
                    final_r_step = max(r.keys())
                    rewards.append(r[final_r_step])
            n_cat = sum(1 for d in dists if d > CATASTROPHIC_DIST)
            results[vk][n] = {
                "dists": dists,
                "rewards": rewards,
                "curves": curves,
                "n_seeds": len(dists),
                "n_catastrophic": n_cat,
            }
    return results


def print_summary(results):
    print("\n" + "="*80)
    print("ATTENTION COMMUNICATION SUITE — RESULTS SUMMARY")
    print("="*80)
    print(f"{'Variant':<22} {'N':>3}  {'Seeds':>6}  {'Mean dist':>10}  {'Std':>8}  {'Catast%':>8}  {'Mean rew':>10}")
    print("-"*80)

    for n in N_VALUES:
        print(f"\n--- N={n} ---")
        for vk, vlabel in VARIANT_LABELS.items():
            r = results[vk].get(n, {})
            seeds = r.get("n_seeds", 0)
            if seeds == 0:
                print(f"  {vlabel:<20} {n:>3}  {'N/A':>6}")
                continue
            dists = np.array(r["dists"])
            mean_d = np.mean(dists)
            std_d = np.std(dists)
            cat_pct = 100 * r["n_catastrophic"] / seeds
            rews = r.get("rewards", [])
            mean_r = np.mean(rews) if rews else float("nan")
            print(f"  {vlabel:<20} {n:>3}  {seeds:>6}  {mean_d:>10.2f}  {std_d:>8.2f}  {cat_pct:>7.0f}%  {mean_r:>10.1f}")


def print_comparisons(results):
    print("\n" + "="*80)
    print("KEY COMPARISONS (mean dist, lower is better)")
    print("="*80)

    for n in N_VALUES:
        print(f"\n--- N={n} ---")
        base_dists = results["baseline"].get(n, {}).get("dists", [])
        memo_dists = results["memory_only"].get(n, {}).get("dists", [])
        ic3_dists = results["ic3net"].get(n, {}).get("dists", [])
        ai_dists = results["attn_integrated"].get(n, {}).get("dists", [])
        as_dists = results["attn_separated"].get(n, {}).get("dists", [])

        base_mean = np.mean(base_dists) if base_dists else float("nan")
        memo_mean = np.mean(memo_dists) if memo_dists else float("nan")
        ic3_mean = np.mean(ic3_dists) if ic3_dists else float("nan")
        ai_mean = np.mean(ai_dists) if ai_dists else float("nan")
        as_mean = np.mean(as_dists) if as_dists else float("nan")

        # vs baseline
        if not np.isnan(base_mean):
            for vk, vm, vlabel in [
                ("memory_only",     memo_mean, "memory_only"),
                ("ic3net",          ic3_mean,  "ic3net"),
                ("attn_integrated", ai_mean,   "attn_integrated"),
                ("attn_separated",  as_mean,   "attn_separated"),
            ]:
                if not np.isnan(vm):
                    ratio = base_mean / vm if vm > 0 else float("inf")
                    direction = "BETTER" if vm < base_mean else "worse"
                    print(f"  {vlabel:<22} vs baseline: {vm:.2f} vs {base_mean:.2f}  ({ratio:.1f}x) [{direction}]")

        # attention vs ic3net
        if not np.isnan(ic3_mean):
            for vm, vlabel in [(ai_mean, "attn_integrated"), (as_mean, "attn_separated")]:
                if not np.isnan(vm):
                    ratio = ic3_mean / vm if vm > 0 else float("inf")
                    direction = "BETTER" if vm < ic3_mean else "worse"
                    print(f"  {vlabel:<22} vs ic3net:   {vm:.2f} vs {ic3_mean:.2f}  ({ratio:.1f}x) [{direction}]")

        # attention_separated vs attention_integrated
        if not np.isnan(ai_mean) and not np.isnan(as_mean):
            ratio = ai_mean / as_mean if as_mean > 0 else float("inf")
            direction = "BETTER" if as_mean < ai_mean else "worse"
            print(f"  attn_separated vs attn_integrated: {as_mean:.2f} vs {ai_mean:.2f}  ({ratio:.1f}x) [{direction}]")


def plot_learning_curves(results, n, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for vk, vlabel in VARIANT_LABELS.items():
        r = results[vk].get(n, {})
        curves = r.get("curves", [])
        if not curves:
            continue
        # Gather all eval steps
        all_steps = sorted(set(s for c in curves for s in c.keys()))
        # For each step, compute mean and std over seeds that have that step
        means, stds = [], []
        for s in all_steps:
            vals = [c[s] for c in curves if s in c]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means = np.array(means)
        stds = np.array(stds)
        color = COLORS[vk]
        ax.plot(all_steps, means, label=vlabel, color=color, linewidth=2)
        ax.fill_between(all_steps, means - stds, means + stds, alpha=0.15, color=color)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Coverage Distance (lower is better)")
    ax.set_title(f"Attention Comm Suite — N={n} — Coverage Distance")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved: {out_path}")


def plot_catastrophic_rates(results, out_path):
    fig, axes = plt.subplots(1, len(N_VALUES), figsize=(14, 5))
    x = np.arange(len(VARIANTS))
    labels = list(VARIANT_LABELS.values())
    colors = [COLORS[vk] for vk in VARIANTS]

    for ax, n in zip(axes, N_VALUES):
        rates = []
        for vk in VARIANTS:
            r = results[vk].get(n, {})
            seeds = r.get("n_seeds", 0)
            cat = r.get("n_catastrophic", 0)
            rates.append(100 * cat / seeds if seeds > 0 else 0)
        bars = ax.bar(x, rates, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Catastrophic Failure Rate (%)")
        ax.set_title(f"N={n}")
        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.text(bar.get_x() + bar.get_width()/2, rate + 1, f"{rate:.0f}%",
                        ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Catastrophic Failure Rate (dist > {CATASTROPHIC_DIST}) by Variant and N")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved: {out_path}")


def plot_final_dist_bars(results, out_path):
    fig, axes = plt.subplots(1, len(N_VALUES), figsize=(14, 5))
    x = np.arange(len(VARIANTS))
    labels = list(VARIANT_LABELS.values())
    colors = [COLORS[vk] for vk in VARIANTS]

    for ax, n in zip(axes, N_VALUES):
        means, stds = [], []
        for vk in VARIANTS:
            r = results[vk].get(n, {})
            dists = r.get("dists", [])
            means.append(np.mean(dists) if dists else 0)
            stds.append(np.std(dists) if dists else 0)
        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Final Mean Coverage Distance")
        ax.set_title(f"N={n}")

    fig.suptitle("Final Coverage Distance (lower is better) — Attention Comm Suite")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    results = collect_results()
    print_summary(results)
    print_comparisons(results)

    out_dir = "results/attention_comm_suite"
    os.makedirs(out_dir, exist_ok=True)

    # Learning curves per N
    for n in N_VALUES:
        plot_learning_curves(results, n, f"{out_dir}/curves_n{n}.png")

    # Catastrophic failure rates
    plot_catastrophic_rates(results, f"{out_dir}/catastrophic_rates.png")

    # Final dist bars
    plot_final_dist_bars(results, f"{out_dir}/final_dist_bars.png")

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
