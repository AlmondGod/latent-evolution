"""
analyze_partial_obs_nscale.py
-----------------------------
Analyzes the partial-obs N-scaling experiment to answer the core
memetic hypothesis:

  Does memory+comm advantage grow with N under partial observability?

Usage:
    python3.9 -m new.memetic_foundation.scripts.analyze_partial_obs_nscale

Produces:
  plots/partial_obs_nscale.png  — main figure
  plots/partial_obs_nscale_table.txt  — LaTeX-ready results table
"""

import json, os, glob, sys
import numpy as np

# Optional matplotlib import
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

BASE_PARTIAL = "checkpoints/mpe_partial_obs_nscale"
BASE_FULL    = "checkpoints/mpe_tag_nscale"  # full observability (already run)


def load_results(directory_pattern: str) -> list[float]:
    """Load final_mean_reward from all matching checkpoint dirs."""
    rewards = []
    for path in sorted(glob.glob(directory_pattern)):
        if not os.path.isdir(path):
            continue
        jsons = glob.glob(os.path.join(path, "**/*results.json"), recursive=True)
        if jsons:
            try:
                with open(jsons[-1]) as f:
                    d = json.load(f)
                r = d.get("final_mean_reward")
                if r is not None:
                    rewards.append(float(r))
            except Exception:
                pass
    return rewards


def summarize(rewards: list[float]) -> tuple[float, float, int]:
    if not rewards:
        return float("nan"), float("nan"), 0
    return float(np.mean(rewards)), float(np.std(rewards)), len(rewards)


def compute_advantage(memory_rewards, baseline_rewards):
    """Relative advantage: (memory - baseline) / |baseline| * 100 (%)"""
    if not memory_rewards or not baseline_rewards:
        return float("nan"), float("nan")
    m = np.mean(memory_rewards)
    b = np.mean(baseline_rewards)
    # Use bootstrap for CI
    rng = np.random.default_rng(42)
    advantages = []
    n_boot = 10000
    for _ in range(n_boot):
        m_boot = np.mean(rng.choice(memory_rewards, size=len(memory_rewards), replace=True))
        b_boot = np.mean(rng.choice(baseline_rewards, size=len(baseline_rewards), replace=True))
        if abs(b_boot) > 1e-6:
            advantages.append((m_boot - b_boot) / abs(b_boot) * 100)
    if advantages:
        return float(np.mean(advantages)), float(np.std(advantages))
    return float((m - b) / max(abs(b), 1) * 100), float("nan")


def main():
    ns = [3, 5, 8]
    variants = ["baseline", "memory_only", "full_gated"]

    # ---- collect all data ----
    data_partial = {}  # (variant, N) -> [rewards]
    data_full    = {}  # (variant, N) -> [rewards]

    for N in ns:
        for variant in variants:
            # partial obs
            pattern = os.path.join(BASE_PARTIAL, f"{variant}_n{N}_seed*")
            data_partial[(variant, N)] = load_results(pattern)
            # full obs
            pattern = os.path.join(BASE_FULL, f"{variant}_n{N}_seed*")
            data_full[(variant, N)] = load_results(pattern)

    # ---- print results table ----
    print("\n" + "=" * 70)
    print("PARTIAL OBS N-SCALING RESULTS")
    print("=" * 70)
    print(f"{'Variant':20s} {'N=3':>12} {'N=5':>12} {'N=8':>12}")
    print("-" * 70)
    for variant in variants:
        row = f"{variant:20s}"
        for N in ns:
            m, s, n = summarize(data_partial[(variant, N)])
            row += f"  {m:6.1f}±{s:5.1f}({n})"
        print(row)

    print("\n" + "=" * 70)
    print("FULL OBS N-SCALING RESULTS (reference)")
    print("=" * 70)
    print(f"{'Variant':20s} {'N=3':>12} {'N=5':>12} {'N=8':>12}")
    print("-" * 70)
    for variant in variants:
        row = f"{variant:20s}"
        for N in ns:
            m, s, n = summarize(data_full[(variant, N)])
            row += f"  {m:6.1f}±{s:5.1f}({n})"
        print(row)

    # ---- relative advantage: memory_only vs baseline ----
    print("\n" + "=" * 70)
    print("RELATIVE ADVANTAGE OF memory_only OVER baseline (%)")
    print("(positive = memory helps)")
    print("=" * 70)
    print(f"{'Condition':25s}  {'N=3':>12} {'N=5':>12} {'N=8':>12}")
    print("-" * 70)

    adv_partial, adv_full = {}, {}
    for N in ns:
        adv_partial[N] = compute_advantage(
            data_partial[("memory_only", N)], data_partial[("baseline", N)]
        )
        adv_full[N] = compute_advantage(
            data_full[("memory_only", N)], data_full[("baseline", N)]
        )

    row_partial = f"{'Partial obs (r=0.5)':25s}"
    row_full    = f"{'Full obs':25s}"
    for N in ns:
        a, e = adv_partial[N]
        row_partial += f"  {a:+7.1f}%±{e:4.1f}" if not np.isnan(a) else "    N/A       "
        a, e = adv_full[N]
        row_full += f"  {a:+7.1f}%±{e:4.1f}" if not np.isnan(a) else "    N/A       "
    print(row_partial)
    print(row_full)

    # ---- check the scaling hypothesis ----
    print("\n" + "=" * 70)
    print("SCALING HYPOTHESIS TEST")
    print("=" * 70)
    adv_partial_vals = [adv_partial[N][0] for N in ns if not np.isnan(adv_partial[N][0])]
    adv_full_vals    = [adv_full[N][0]    for N in ns if not np.isnan(adv_full[N][0])]

    if len(adv_partial_vals) >= 2:
        trend_partial = adv_partial_vals[-1] - adv_partial_vals[0]
        print(f"  Partial obs advantage trend (N=3→N=8): {trend_partial:+.1f}%")
        if trend_partial > 5:
            print("  HYPOTHESIS SUPPORTED: memory advantage GROWS with N under partial obs")
        elif trend_partial < -5:
            print("  HYPOTHESIS REJECTED: memory advantage SHRINKS with N under partial obs")
        else:
            print("  INCONCLUSIVE: advantage is flat across N under partial obs")

    if len(adv_full_vals) >= 2:
        trend_full = adv_full_vals[-1] - adv_full_vals[0]
        print(f"  Full obs advantage trend (N=3→N=8):    {trend_full:+.1f}%")
        print(f"  (expected ~0 — full obs = homogeneous agents = no memory benefit)")

    # ---- plot ----
    if not HAS_PLOT:
        print("\n(matplotlib not available, skipping plots)")
        return

    os.makedirs("plots", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Memetic Foundation: N-Agent Scaling Study", fontsize=13, fontweight="bold")

    colors = {"baseline": "#888888", "memory_only": "#2196F3", "full_gated": "#FF5722"}
    markers = {"baseline": "o", "memory_only": "s", "full_gated": "^"}

    for ax_idx, (data_dict, title, obs_label) in enumerate([
        (data_partial, "Partial Observability (r=0.5)", "partial"),
        (data_full,    "Full Observability",            "full"),
    ]):
        ax = axes[ax_idx]
        for variant in variants:
            means, stds, ns_valid = [], [], []
            for N in ns:
                m, s, n = summarize(data_dict[(variant, N)])
                if not np.isnan(m):
                    means.append(m)
                    stds.append(s)
                    ns_valid.append(N)
            if means:
                ax.plot(ns_valid, means, marker=markers[variant], color=colors[variant],
                        linewidth=2, markersize=8, label=variant, zorder=3)
                ax.fill_between(ns_valid,
                                [m - s for m, s in zip(means, stds)],
                                [m + s for m, s in zip(means, stds)],
                                alpha=0.15, color=colors[variant])

        ax.set_xlabel("Number of Adversaries (N)", fontsize=11)
        ax.set_ylabel("Mean Reward (↑ better)", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(ns)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "plots/partial_obs_nscale.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out}")

    # ---- advantage scaling plot ----
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.set_title("Memory Advantage vs N\n(memory_only − baseline) / |baseline| × 100%",
                  fontsize=11)

    for adv_dict, label, color, ls in [
        (adv_partial, "Partial obs (r=0.5)", "#2196F3", "-"),
        (adv_full,    "Full obs",            "#888888", "--"),
    ]:
        xs, ys, errs = [], [], []
        for N in ns:
            a, e = adv_dict[N]
            if not np.isnan(a):
                xs.append(N)
                ys.append(a)
                errs.append(e if not np.isnan(e) else 0)
        if xs:
            ax2.plot(xs, ys, marker="o", linewidth=2, markersize=8,
                     color=color, linestyle=ls, label=label)
            ax2.fill_between(xs, [y - e for y, e in zip(ys, errs)],
                             [y + e for y, e in zip(ys, errs)],
                             alpha=0.2, color=color)

    ax2.axhline(0, color="black", linewidth=1, linestyle=":")
    ax2.set_xlabel("N (number of adversaries)", fontsize=11)
    ax2.set_ylabel("Relative memory advantage (%)", fontsize=11)
    ax2.set_xticks(ns)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = "plots/memory_advantage_vs_N.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Advantage figure saved: {out2}")

    # ---- LaTeX table ----
    table_path = "plots/partial_obs_nscale_table.txt"
    with open(table_path, "w") as f:
        f.write("% Auto-generated by analyze_partial_obs_nscale.py\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\begin{tabular}{l|ccc|ccc}\n\\hline\n")
        f.write(" & \\multicolumn{3}{c|}{Partial obs (r=0.5)} & \\multicolumn{3}{c}{Full obs} \\\\\n")
        f.write("Variant & N=3 & N=5 & N=8 & N=3 & N=5 & N=8 \\\\\n\\hline\n")
        for variant in variants:
            row = variant.replace("_", "\\_") + " & "
            cells = []
            for (data_dict) in [data_partial, data_full]:
                for N in ns:
                    m, s, n = summarize(data_dict[(variant, N)])
                    if np.isnan(m):
                        cells.append("—")
                    else:
                        cells.append(f"{m:.0f}$\\pm${s:.0f}")
            row += " & ".join(cells) + " \\\\\n"
            f.write(row)
        f.write("\\hline\n\\end{tabular}\n")
        f.write("\\caption{Mean reward \\pm std across seeds on simple\\_tag with N adversaries.}\n")
        f.write("\\end{table}\n")
    print(f"LaTeX table saved: {table_path}")


if __name__ == "__main__":
    main()
