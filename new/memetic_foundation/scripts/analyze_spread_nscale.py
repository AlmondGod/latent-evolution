"""
analyze_spread_nscale.py — Analyze simple_spread partial obs N-scaling results.

Tests the Scalable Memetics Hypothesis:
  H: Under partial observability with structural role asymmetry (simple_spread),
     the advantage of memory+comm over baseline GROWS with the number of agents N.

Key metrics:
  1. Reward gap: (memory_only - baseline) / |baseline|
  2. Memory diversity: pairwise cosine similarity between agent h states
  3. Landmark specialization: fraction of agents consistently covering same landmark
  4. Training stability: variance across seeds

Usage:
    python3.9 -m new.memetic_foundation.scripts.analyze_spread_nscale
"""

import os, re, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def parse_eval_rewards(log_path):
    """Parse [Eval] lines from log, return dict of {step: reward}."""
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


def parse_eval_dist(log_path):
    """Parse [Eval] lines, return dict of {step: dist}."""
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


def collect_results(base_dir, total_steps=400000):
    """
    Collect all results from the spread partial obs experiment.

    Returns: dict[variant][N] = {
        'final_rewards': [r1, r2, ...],    # per seed, at last eval
        'peak_rewards': [r1, r2, ...],     # per seed, best eval ever
        'all_evals': [[step_rewards], ...] # per seed, full curve
        'final_dists': [d1, d2, ...],      # per seed, final dist
    }
    """
    variants = ["baseline", "memory_only", "full_gated"]
    Ns = [3, 5, 8]

    data = {v: {n: {"final_rewards": [], "peak_rewards": [],
                    "all_evals": [], "final_dists": []} for n in Ns}
            for v in variants}

    for variant in variants:
        for N in Ns:
            seed_logs = sorted(glob.glob(
                os.path.join(base_dir, f"{variant}_n{N}_seed*.log")
            ))
            if not seed_logs:
                continue

            for lf in seed_logs:
                evals = parse_eval_rewards(lf)
                dists = parse_eval_dist(lf)
                if not evals:
                    continue

                steps = sorted(evals.keys())
                rewards = [evals[s] for s in steps]

                # Final = last eval point
                final_rew = rewards[-1]
                peak_rew = max(rewards)
                final_step = max(steps)
                final_dist = dists.get(final_step, 1.0)

                data[variant][N]["final_rewards"].append(final_rew)
                data[variant][N]["peak_rewards"].append(peak_rew)
                data[variant][N]["all_evals"].append(list(zip(steps, rewards)))
                data[variant][N]["final_dists"].append(final_dist)

    return data


# ---------------------------------------------------------------------------
# Scalability analysis
# ---------------------------------------------------------------------------

def compute_reward_gap(data, Ns):
    """
    Compute relative reward advantage of memory_only and full_gated over baseline.

    rel_advantage = (variant_mean - baseline_mean) / |baseline_mean|
    """
    gaps = {}
    for variant in ["memory_only", "full_gated"]:
        gaps[variant] = {}
        for N in Ns:
            baseline_r = data["baseline"][N]["final_rewards"]
            variant_r = data[variant][N]["final_rewards"]
            if not baseline_r or not variant_r:
                continue

            b_mean = np.mean(baseline_r)
            v_mean = np.mean(variant_r)
            b_std = np.std(baseline_r)
            v_std = np.std(variant_r)

            rel_adv = (v_mean - b_mean) / (abs(b_mean) + 1e-8)
            # Bootstrap CI for the gap
            n_boot = 1000
            boot_gaps = []
            for _ in range(n_boot):
                b_boot = np.random.choice(baseline_r, size=len(baseline_r), replace=True)
                v_boot = np.random.choice(variant_r, size=len(variant_r), replace=True)
                b_m = np.mean(b_boot)
                v_m = np.mean(v_boot)
                boot_gaps.append((v_m - b_m) / (abs(b_m) + 1e-8))
            ci_lo, ci_hi = np.percentile(boot_gaps, [5, 95])

            gaps[variant][N] = {
                "baseline_mean": float(b_mean),
                "baseline_std": float(b_std),
                "variant_mean": float(v_mean),
                "variant_std": float(v_std),
                "relative_advantage": float(rel_adv),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
                "n_seeds_baseline": len(baseline_r),
                "n_seeds_variant": len(variant_r),
            }

    return gaps


def check_scalability_hypothesis(gaps):
    """
    Test whether the memory advantage grows with N.

    Method: linear regression of relative advantage vs N.
    Positive slope = advantage grows with N (supports hypothesis).
    """
    print("\n" + "=" * 65)
    print("SCALABLE MEMETICS HYPOTHESIS TEST")
    print("H: memory advantage grows with N under partial obs + spread")
    print("=" * 65)

    for variant, variant_gaps in gaps.items():
        Ns_avail = sorted(variant_gaps.keys())
        if len(Ns_avail) < 2:
            print(f"\n{variant}: insufficient N values ({Ns_avail})")
            continue

        adv_by_N = [(N, variant_gaps[N]["relative_advantage"],
                     variant_gaps[N]["ci_lo"], variant_gaps[N]["ci_hi"])
                    for N in Ns_avail]

        print(f"\n{variant}:")
        for N, adv, ci_lo, ci_hi in adv_by_N:
            print(f"  N={N}: advantage={adv:+.2%}  [90% CI: {ci_lo:+.2%}, {ci_hi:+.2%}]")

        # Linear regression
        Ns_arr = np.array([x[0] for x in adv_by_N], dtype=float)
        advs_arr = np.array([x[1] for x in adv_by_N])
        if len(Ns_arr) >= 2:
            slope, intercept = np.polyfit(Ns_arr, advs_arr, deg=1)
            print(f"\n  Linear trend: slope = {slope:+.4f} per agent")
            if slope > 0.01:
                print(f"  ✓ HYPOTHESIS SUPPORTED: advantage grows by {slope:.1%} per agent")
            elif slope > 0:
                print(f"  ~ Weak positive trend (slope={slope:.4f})")
            else:
                print(f"  ✗ HYPOTHESIS REJECTED: advantage SHRINKS with N (slope={slope:.4f})")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_reward_summary(data, gaps, Ns, out_dir):
    """
    Three-panel figure:
      Left:  Mean final reward per variant per N (bar chart)
      Center: Relative advantage over baseline per N (line chart)
      Right:  Training curves at N=3 and N=8
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Scalable Memetics: Simple Spread + Partial Obs (obs_radius=0.5)",
                 fontsize=13, fontweight="bold")

    colors = {"baseline": "#888888", "memory_only": "#2196F3", "full_gated": "#E91E63"}
    x = np.arange(len(Ns))
    bar_width = 0.28

    # Panel 1: Bar chart of final rewards
    ax = axes[0]
    for i, (variant, color) in enumerate(colors.items()):
        means, stds = [], []
        for N in Ns:
            r = data[variant][N]["final_rewards"]
            means.append(np.mean(r) if r else 0)
            stds.append(np.std(r) if r else 0)
        ax.bar(x + i * bar_width, means, bar_width, label=variant,
               color=color, alpha=0.8, yerr=stds, capsize=4)

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.set_ylabel("Final Reward (higher = better)")
    ax.set_title("Final Reward by Variant & N\n(simple_spread + partial obs)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5)

    # Panel 2: Relative advantage over baseline
    ax = axes[1]
    for variant, color in [("memory_only", "#2196F3"), ("full_gated", "#E91E63")]:
        Ns_avail = sorted(gaps[variant].keys())
        if not Ns_avail:
            continue
        adv = [gaps[variant][n]["relative_advantage"] for n in Ns_avail]
        ci_lo = [gaps[variant][n]["ci_lo"] for n in Ns_avail]
        ci_hi = [gaps[variant][n]["ci_hi"] for n in Ns_avail]
        ax.plot(Ns_avail, adv, "o-", color=color, label=variant, linewidth=2, markersize=8)
        ax.fill_between(Ns_avail, ci_lo, ci_hi, alpha=0.2, color=color)

    ax.axhline(0, color="k", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Number of Agents (N)")
    ax.set_ylabel("Relative Advantage over Baseline")
    ax.set_title("Memory Advantage vs N\n(Scalable Memetics Test)")
    ax.set_xticks(Ns)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Mark the slope direction
    for variant, color in [("memory_only", "#2196F3"), ("full_gated", "#E91E63")]:
        Ns_avail = sorted(gaps[variant].keys())
        if len(Ns_avail) >= 2:
            adv = [gaps[variant][n]["relative_advantage"] for n in Ns_avail]
            slope, intercept = np.polyfit(Ns_avail, adv, 1)
            x_fit = np.array([min(Ns_avail), max(Ns_avail)], dtype=float)
            ax.plot(x_fit, slope * x_fit + intercept, "--", color=color, alpha=0.5, linewidth=1.5)

    # Panel 3: Learning curves for extreme Ns
    ax = axes[2]
    Ns_to_plot = [Ns[0], Ns[-1]] if len(Ns) >= 2 else Ns
    linestyles = {Ns_to_plot[0]: "-", Ns_to_plot[-1]: "--"} if len(Ns_to_plot) >= 2 else {}
    for variant, color in colors.items():
        for N in Ns_to_plot:
            all_evals = data[variant][N]["all_evals"]
            if not all_evals:
                continue
            # Find max step
            max_step = max(s for seed_evals in all_evals for s, r in seed_evals)
            # Build average curve
            step_rewards = {}
            for seed_evals in all_evals:
                for s, r in seed_evals:
                    step_rewards.setdefault(s, []).append(r)
            steps = sorted(step_rewards.keys())
            means = [np.mean(step_rewards[s]) for s in steps]
            ls = linestyles.get(N, "-")
            ax.plot(steps, means, ls, color=color,
                    label=f"{variant} N={N}", alpha=0.8, linewidth=1.5)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Team Reward")
    ax.set_title(f"Learning Curves\n(N={Ns_to_plot[0]} solid, N={Ns_to_plot[-1]} dashed)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "spread_nscale_analysis.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
    plt.close()


def plot_dist_curves(data, Ns, out_dir):
    """Plot mean landmark distance over training (lower = better coverage)."""
    fig, axes = plt.subplots(1, len(Ns), figsize=(5 * len(Ns), 5))
    if len(Ns) == 1:
        axes = [axes]

    colors = {"baseline": "#888888", "memory_only": "#2196F3", "full_gated": "#E91E63"}
    fig.suptitle("Mean Landmark Distance (lower = better)\nSimple Spread + Partial Obs",
                 fontsize=12)

    for ax, N in zip(axes, Ns):
        for variant, color in colors.items():
            all_evals = data[variant][N]["all_evals"]
            if not all_evals:
                continue
            step_rewards = {}
            for seed_evals in all_evals:
                for s, r in seed_evals:
                    step_rewards.setdefault(s, []).append(r)
            steps = sorted(step_rewards.keys())
            means = [np.mean(step_rewards[s]) for s in steps]
            ax.plot(steps, means, color=color, label=variant, linewidth=2)

        ax.set_title(f"N={N}")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Team Reward (proxy for coverage)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "spread_coverage_curves.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Coverage plot saved to: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def print_full_table(data, Ns):
    """Print final results table."""
    print("\n" + "=" * 75)
    print("SIMPLE SPREAD PARTIAL OBS: FINAL RESULTS TABLE")
    print("(simple_spread_v2, obs_radius=0.5, 400k steps, LR annealing)")
    print("=" * 75)
    print(f"{'Variant':15s} | {'N=3':>20s} | {'N=5':>20s} | {'N=8':>20s}")
    print("-" * 75)
    for variant in ["baseline", "memory_only", "full_gated"]:
        row = f"{variant:15s}"
        for N in Ns:
            r = data[variant][N]["final_rewards"]
            if r:
                row += f" | {np.mean(r):8.0f} ± {np.std(r):.0f} (n={len(r)})"
            else:
                row += f" | {'—':>18s}"
        print(row)
    print("=" * 75)

    # Distance table
    print(f"\n{'Variant':15s} | {'N=3 dist':>12s} | {'N=5 dist':>12s} | {'N=8 dist':>12s}")
    print("-" * 60)
    for variant in ["baseline", "memory_only", "full_gated"]:
        row = f"{variant:15s}"
        for N in Ns:
            d = data[variant][N]["final_dists"]
            if d:
                row += f" | {np.mean(d):6.3f} ± {np.std(d):.3f}"
            else:
                row += f" | {'—':>12s}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    base_dir = "checkpoints/mpe_spread_partial_obs"
    out_dir = "plots"
    Ns = [3, 5, 8]

    print(f"Analyzing spread partial obs results in: {base_dir}")
    data = collect_results(base_dir)

    # Check what we have
    print("\nAvailable results:")
    for variant in ["baseline", "memory_only", "full_gated"]:
        for N in Ns:
            n_seeds = len(data[variant][N]["final_rewards"])
            if n_seeds > 0:
                r = data[variant][N]["final_rewards"]
                print(f"  {variant}_n{N}: {n_seeds} seeds, "
                      f"mean={np.mean(r):.0f}±{np.std(r):.0f}")

    # Full table
    print_full_table(data, Ns)

    # Compute reward gaps
    gaps = compute_reward_gap(data, Ns)

    # Test scalability hypothesis
    check_scalability_hypothesis(gaps)

    # Plot
    plot_reward_summary(data, gaps, Ns, out_dir)

    # Save JSON summary
    summary = {}
    for variant in ["baseline", "memory_only", "full_gated"]:
        summary[variant] = {}
        for N in Ns:
            r = data[variant][N]["final_rewards"]
            d = data[variant][N]["final_dists"]
            if r:
                summary[variant][str(N)] = {
                    "mean_reward": float(np.mean(r)),
                    "std_reward": float(np.std(r)),
                    "n_seeds": len(r),
                    "mean_dist": float(np.mean(d)) if d else None,
                }
    summary["gaps"] = {v: {str(k): w for k, w in gaps[v].items()}
                       for v in gaps}

    os.makedirs("results", exist_ok=True)
    with open("results/spread_nscale_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nJSON summary saved to: results/spread_nscale_summary.json")


if __name__ == "__main__":
    main()
