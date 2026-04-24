"""
tag_nscale_analysis.py
======================
Parse re-eval results from tag_nscale suite and generate scaling plots.

Metrics:
  - mean_reward: cumulative catch reward per eval episode (higher = better predators)

Variants:
  memory_only, memory_only_persistent, commnet_persistent, commnet
N values: 3, 6, 9, 12, 15
Seeds: 5 per cell
"""

import os
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
BASE = Path("checkpoints/tag_nscale")
OUT_DIR = Path("new/memetic_foundation/analysis/tag_nscale_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["memory_only", "memory_only_persistent", "commnet_persistent", "commnet"]
VARIANT_LABELS = {
    "memory_only":            "Memory only",
    "memory_only_persistent": "Memory + persistence",
    "commnet_persistent":     "CommNet + persistence",
    "commnet":                "CommNet (no persist)",
}
COLORS = {
    "memory_only":            "#2196F3",
    "memory_only_persistent": "#4CAF50",
    "commnet_persistent":     "#FF9800",
    "commnet":                "#F44336",
}
NS = [3, 6, 9, 12, 15]
SEEDS = [1, 2, 3, 4, 5]

# ── Parsing ──────────────────────────────────────────────────────────────────
def parse_reeval(run_dir: Path):
    """Extract mean_reward from reeval.log; fall back to training log if missing."""
    reeval_log = run_dir / "reeval.log"
    if reeval_log.exists():
        txt = reeval_log.read_text()
        m = re.search(r"Mean reward:\s*([\d.]+)", txt)
        if m:
            return float(m.group(1))

    # Fallback: last eval line from training log
    log_path = Path(str(run_dir) + ".log")
    if log_path.exists():
        lines = [l for l in log_path.read_text().splitlines() if "[Eval]" in l]
        if lines:
            m = re.search(r"reward=([\d.]+)", lines[-1])
            if m:
                return float(m.group(1))
    return None


def collect_data():
    """Returns dict[variant][N] -> list of rewards (one per seed)."""
    data = {v: {N: [] for N in NS} for v in VARIANTS}
    missing = []
    for v in VARIANTS:
        for N in NS:
            for s in SEEDS:
                run_dir = BASE / f"{v}_n{N}_seed{s}"
                r = parse_reeval(run_dir)
                if r is not None:
                    data[v][N].append(r)
                else:
                    missing.append(f"{v}_n{N}_seed{s}")
    if missing:
        print(f"WARNING: missing data for {len(missing)} runs: {missing[:5]}...")
    return data


# ── Plotting ─────────────────────────────────────────────────────────────────
def plot_scaling(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Simple Tag: Predator Reward vs N Predators", fontsize=14, fontweight="bold")

    ax_mean, ax_cv = axes

    for v in VARIANTS:
        means, sems, medians, cvs = [], [], [], []
        for N in NS:
            vals = data[v][N]
            if vals:
                arr = np.array(vals)
                means.append(np.mean(arr))
                sems.append(np.std(arr) / np.sqrt(len(arr)))
                medians.append(np.median(arr))
                cvs.append(100 * np.std(arr) / np.mean(arr) if np.mean(arr) > 0 else 0)
            else:
                means.append(np.nan); sems.append(np.nan)
                medians.append(np.nan); cvs.append(np.nan)

        label = VARIANT_LABELS[v]
        c = COLORS[v]
        ns_arr = np.array(NS, dtype=float)

        ax_mean.plot(ns_arr, means, "o-", color=c, label=label, linewidth=2, markersize=6)
        ax_mean.fill_between(ns_arr,
                             np.array(means) - np.array(sems),
                             np.array(means) + np.array(sems),
                             alpha=0.15, color=c)

        ax_cv.plot(ns_arr, cvs, "s--", color=c, label=label, linewidth=1.5, markersize=5)

    ax_mean.set_xlabel("N predators")
    ax_mean.set_ylabel("Mean reward (100 eval episodes)")
    ax_mean.set_title("Mean ± SEM reward")
    ax_mean.legend(fontsize=9)
    ax_mean.grid(True, alpha=0.3)
    ax_mean.set_xticks(NS)

    ax_cv.set_xlabel("N predators")
    ax_cv.set_ylabel("Coefficient of variation (%)")
    ax_cv.set_title("Reward variance (CV%)")
    ax_cv.legend(fontsize=9)
    ax_cv.grid(True, alpha=0.3)
    ax_cv.set_xticks(NS)
    ax_cv.axhline(50, color="gray", linestyle=":", linewidth=1, label="50% CV")

    plt.tight_layout()
    out = OUT_DIR / "fig1_tag_scaling.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_commnet_collapse(data):
    """Highlight commnet no-persistence vs others at each N."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for v in VARIANTS:
        means, sems = [], []
        for N in NS:
            vals = data[v][N]
            if vals:
                arr = np.array(vals)
                means.append(np.mean(arr))
                sems.append(np.std(arr) / np.sqrt(len(arr)))
            else:
                means.append(np.nan); sems.append(np.nan)

        lw = 3 if v == "commnet" else 1.5
        ls = "-" if v == "commnet" else "--"
        ax.plot(NS, means, marker="o", color=COLORS[v],
                label=VARIANT_LABELS[v], linewidth=lw, linestyle=ls, markersize=7)
        ax.fill_between(NS,
                        np.array(means) - np.array(sems),
                        np.array(means) + np.array(sems),
                        alpha=0.12, color=COLORS[v])

    ax.set_xlabel("N predators", fontsize=12)
    ax.set_ylabel("Mean reward (100 episodes)", fontsize=12)
    ax.set_title("Simple Tag: CommNet collapse without persistent memory", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(NS)
    plt.tight_layout()
    out = OUT_DIR / "fig2_commnet_collapse.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_seed_scatter(data):
    """Show individual seed rewards to expose bimodal distributions."""
    fig, axes = plt.subplots(1, len(NS), figsize=(16, 4), sharey=False)
    fig.suptitle("Simple Tag: Per-seed reward at final checkpoint", fontsize=13)

    for ax, N in zip(axes, NS):
        for i, v in enumerate(VARIANTS):
            vals = data[v][N]
            if vals:
                x = [i + 0.1 * (s - 3) for s in range(len(vals))]
                ax.scatter(x, vals, color=COLORS[v], alpha=0.8, s=40, zorder=3)
                ax.hlines(np.mean(vals), i - 0.2, i + 0.2, color=COLORS[v], linewidth=2)
        ax.set_title(f"N={N}")
        ax.set_xticks(range(len(VARIANTS)))
        ax.set_xticklabels(["MO", "MOP", "CP", "C"], fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Reward (100 episodes)")
    plt.tight_layout()
    out = OUT_DIR / "fig3_seed_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def print_table(data):
    print("\n" + "="*75)
    print("Tag N-Scale Results (mean ± std, 5 seeds × 100 eval episodes)")
    print("="*75)
    print(f"{'N':>4} | {'Variant':<26} | {'Mean':>7} | {'Std':>7} | {'CV%':>5} | Seeds")
    print("-"*75)
    for N in NS:
        for v in VARIANTS:
            vals = data[v][N]
            if vals:
                arr = np.array(vals)
                m, s = np.mean(arr), np.std(arr)
                cv = 100 * s / m if m > 0 else 0
                vals_str = " ".join(f"{x:.0f}" for x in vals)
                print(f"{N:>4} | {v:<26} | {m:>7.1f} | {s:>7.1f} | {cv:>4.0f}% | {vals_str}")
        print()


def save_json(data):
    out = {}
    for v in VARIANTS:
        out[v] = {}
        for N in NS:
            vals = data[v][N]
            if vals:
                arr = np.array(vals)
                out[v][str(N)] = {
                    "seeds": vals,
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "sem": float(np.std(arr) / np.sqrt(len(arr))),
                }
    path = OUT_DIR / "tag_nscale_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Collecting re-eval results...")
    data = collect_data()
    print_table(data)
    save_json(data)
    plot_scaling(data)
    plot_commnet_collapse(data)
    plot_seed_scatter(data)
    print("\nDone. Plots saved to:", OUT_DIR)
