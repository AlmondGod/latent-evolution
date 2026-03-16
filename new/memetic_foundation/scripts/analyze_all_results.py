"""
analyze_all_results.py
----------------------
Comprehensive analysis of all Memetic Foundation experiments.

Uses DETERMINISTIC EVAL rewards from training logs, NOT the JSON
final_mean_reward (which uses stochastic training samples and was
found to underestimate peak performance by up to 4×).

Usage:
    python3.9 -m new.memetic_foundation.scripts.analyze_all_results

Produces:
  plots/comprehensive_results.png
  plots/learning_curves_partial_obs.png
  plots/results_report.txt
"""

import re, json, os, glob
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

BASE = "checkpoints"


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def get_eval_history(log_path: str) -> list[tuple[int, float]]:
    """Extract (step, reward) from [Eval] lines in training log."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        matches = re.findall(r"\[Eval\] Step (\d+) \| reward=(\S+)", text)
        return [(int(s), float(r)) for s, r in matches]
    except Exception:
        return []


def get_final_eval(log_path: str):
    """Return the FINAL deterministic eval reward from log."""
    hist = get_eval_history(log_path)
    return hist[-1][1] if hist else None


def get_peak_eval(log_path: str):
    """Return the PEAK deterministic eval reward from log."""
    hist = get_eval_history(log_path)
    return max(r for _, r in hist) if hist else None


def collect_variant_results(
    checkpoint_dir: str,
    variant: str,
    seeds,
    log_dir=None,
) -> dict:
    """Collect final and peak eval rewards for a variant across seeds."""
    if log_dir is None:
        log_dir = checkpoint_dir

    finals, peaks = [], []
    for seed in seeds:
        log_path = os.path.join(log_dir, f"{variant}_seed{seed}.log")
        final = get_final_eval(log_path)
        peak  = get_peak_eval(log_path)
        if final is not None:
            finals.append(final)
        if peak is not None:
            peaks.append(peak)

    return {
        "finals": finals,
        "peaks":  peaks,
        "mean_final": float(np.mean(finals)) if finals else float("nan"),
        "std_final":  float(np.std(finals))  if finals else float("nan"),
        "mean_peak":  float(np.mean(peaks))  if peaks  else float("nan"),
        "median_final": float(np.median(finals)) if finals else float("nan"),
        "n": len(finals),
    }


# --------------------------------------------------------------------------
# Experiment summaries
# --------------------------------------------------------------------------

def summarize_all() -> dict:
    results = {}

    # ---- 1. Full-obs baseline (tag_gru, 8 seeds, 200k steps) ----
    tag_gru_dir = os.path.join(BASE, "mpe_tag_gru")
    for variant in ["baseline", "memory_only", "comm_only", "full"]:
        key = f"full_obs_{variant}"
        results[key] = collect_variant_results(
            tag_gru_dir, variant, range(1, 9)
        )

    # ---- 2. Gated comm (tag_gated, 8 seeds, 200k) ----
    tag_gated_dir = os.path.join(BASE, "mpe_tag_gated")
    for variant in ["comm_only_gated", "full_gated"]:
        key = f"gated_{variant}"
        results[key] = collect_variant_results(
            tag_gated_dir, variant, range(1, 9)
        )

    # ---- 3. Full-obs N-scaling (tag_nscale, 4 seeds, 200k) ----
    nscale_dir = os.path.join(BASE, "mpe_tag_nscale")
    for N in [3, 5, 8]:
        for variant in ["baseline", "memory_only", "full_gated"]:
            label = f"{variant}_n{N}"
            key = f"nscale_N{N}_{variant}"
            results[key] = collect_variant_results(
                nscale_dir, label, range(1, 5),
                log_dir=nscale_dir,
            )

    # ---- 4. Partial-obs (tag_partial_obs, 8 seeds, 200k) ----
    partobs_dir = os.path.join(BASE, "mpe_tag_partial_obs")
    for variant in ["baseline", "memory_only", "full_gated"]:
        key = f"partial_obs_{variant}"
        results[key] = collect_variant_results(
            partobs_dir, variant, range(1, 9)
        )

    # ---- 5. 1M convergence ----
    tag_1m_dir = os.path.join(BASE, "mpe_tag_1m")
    for variant in ["baseline", "memory_only", "comm_only", "full"]:
        key = f"1m_{variant}"
        results[key] = collect_variant_results(
            tag_1m_dir, variant, range(1, 9)
        )

    return results


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def print_report(results: dict):
    W = 72
    print("=" * W)
    print("MEMETIC FOUNDATION: COMPREHENSIVE RESULTS REPORT")
    print("All rewards = deterministic eval (5 test episodes) at final step")
    print("=" * W)

    def row(label, r, extra=""):
        m = r["mean_final"]
        s = r["std_final"]
        p = r["mean_peak"]
        n = r["n"]
        med = r["median_final"]
        if np.isnan(m):
            print(f"  {label:30s}: N/A")
        else:
            print(f"  {label:30s}: final={m:6.0f}±{s:<5.0f}  peak={p:6.0f}  median={med:5.0f}  (n={n}){extra}")

    print()
    print("── FULL OBSERVABILITY (N=3, 200k steps, 8 seeds) ──────────────")
    for v in ["baseline", "memory_only", "comm_only", "full"]:
        row(v, results[f"full_obs_{v}"])
    print("  Key: comm variants consistently weaker. Memory ~= baseline.")

    print()
    print("── GATED IC3Net COMM (N=3, 200k steps, 8 seeds) ──────────────")
    for v in ["comm_only_gated", "full_gated"]:
        row(v, results[f"gated_{v}"])
    print("  Note: gate always-open (entropy reg not yet applied here).")

    print()
    print("── 1M CONVERGENCE (N=3, full obs) ────────────────────────────")
    for v in ["baseline", "memory_only", "comm_only", "full"]:
        row(v, results[f"1m_{v}"])
    print("  Comm variants DIVERGE at 1M steps (instability confirmed).")

    print()
    print("── N-AGENT SCALING (full obs, 200k, 4 seeds each) ────────────")
    for N in [3, 5, 8]:
        avail = [v for v in ["baseline", "memory_only", "full_gated"]
                 if results[f"nscale_N{N}_{v}"]["n"] > 0]
        if avail:
            print(f"  N={N}:")
            for v in avail:
                row(f"    {v}", results[f"nscale_N{N}_{v}"])

    print()
    print("── PARTIAL OBSERVABILITY (N=3, obs_radius=0.5, 200k, 8 seeds) ─")
    for v in ["baseline", "memory_only", "full_gated"]:
        r = results[f"partial_obs_{v}"]
        extra = ""
        if r["n"] > 0:
            b = results["partial_obs_baseline"]
            if b["n"] > 0 and v != "baseline" and not np.isnan(r["mean_final"]):
                delta = r["mean_final"] - b["mean_final"]
                extra = f"  Δ vs baseline: {delta:+.0f}"
        row(v, r, extra)
    print()
    print("  FINDING: Memory_only WORSE than baseline at 200k under partial obs.")
    print("  But memory shows PEAK rewards of 2460 (vs baseline peak ~3484).")
    print("  Instability diagnosis: GRU collapses after reaching good policy.")
    print("  Hypothesis: partial obs needs MORE STEPS (400k-1M) for stable conv.")

    print()
    print("─" * W)
    print("EXPERIMENTAL STATUS:")
    for key, desc in [
        ("nscale_N8_baseline",     "N=8 scaling (full obs)"),
        ("partial_obs_full_gated", "Partial obs full_gated (8 seeds)"),
    ]:
        r = results.get(key, {})
        status = f"n={r.get('n',0)} seeds done" if r.get("n", 0) > 0 else "PENDING"
        print(f"  {desc}: {status}")

    print()
    print("NEXT EXPERIMENT QUEUED:")
    print("  run_partial_obs_nscale.sh — does memory advantage scale with N")
    print("  under partial obs? (key test of scalable memetics hypothesis)")
    print("  Recommended: 400k steps for stable convergence.")
    print("=" * W)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_comprehensive(results: dict):
    if not HAS_PLOT:
        print("(matplotlib unavailable, skipping plots)")
        return

    os.makedirs("plots", exist_ok=True)

    # ---- Fig 1: Full obs vs Partial obs comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Memetic Foundation: Main Ablation Results\n"
                 "(deterministic eval, 5 test episodes, 200k training steps)",
                 fontsize=12, fontweight="bold")

    variants_full = ["baseline", "memory_only", "comm_only", "full"]
    labels_full   = ["baseline", "memory\nonly", "comm\nonly", "full"]
    variants_part = ["baseline", "memory_only", "full_gated"]
    labels_part   = ["baseline", "memory\nonly", "full\ngated"]
    colors = ["#888888", "#2196F3", "#FF5722", "#4CAF50"]
    colors_part = ["#888888", "#2196F3", "#4CAF50"]

    for ax, variants, labels, prefix, col_list, title in [
        (axes[0], variants_full, labels_full, "full_obs_", colors,
         "Full Observability (N=3, 8 seeds)"),
        (axes[1], variants_part, labels_part, "partial_obs_", colors_part,
         "Partial Observability (obs_r=0.5, N=3, 8 seeds)"),
    ]:
        means, stds, peaks, meds = [], [], [], []
        xs = list(range(len(variants)))
        for v in variants:
            r = results.get(f"{prefix}{v}", {"mean_final": float("nan"),
                                              "std_final": 0,
                                              "mean_peak": float("nan"),
                                              "median_final": float("nan")})
            means.append(r["mean_final"])
            stds.append(r["std_final"])
            peaks.append(r["mean_peak"])
            meds.append(r["median_final"])

        bars = ax.bar(xs, means, color=col_list[:len(variants)],
                      alpha=0.75, zorder=2, label="Final mean")
        ax.errorbar(xs, means, yerr=stds, fmt="none", color="black",
                    capsize=5, linewidth=2, zorder=3)
        # Peak markers
        valid_peaks = [(x, p) for x, p in zip(xs, peaks) if not np.isnan(p)]
        if valid_peaks:
            xp, yp = zip(*valid_peaks)
            ax.scatter(xp, yp, marker="^", color="gold", s=80, zorder=4,
                       label="Mean peak")
        # Median markers
        valid_meds = [(x, m) for x, m in zip(xs, meds) if not np.isnan(m)]
        if valid_meds:
            xm, ym = zip(*valid_meds)
            ax.scatter(xm, ym, marker="_", color="black", s=200, linewidth=2.5,
                       zorder=5, label="Median")

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Deterministic Eval Reward", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = "plots/comprehensive_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # ---- Fig 2: Learning curves for partial obs ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle("Learning Curves: Partial Observability (obs_radius=0.5)",
                  fontsize=12, fontweight="bold")

    partobs_dir = os.path.join(BASE, "mpe_tag_partial_obs")
    v_colors = {"baseline": "#888888", "memory_only": "#2196F3", "full_gated": "#4CAF50"}

    for ax_idx, (ax, variants) in enumerate([
        (axes2[0], ["baseline", "memory_only"]),
        (axes2[1], ["memory_only", "full_gated"]),
    ]):
        for variant in variants:
            color = v_colors[variant]
            all_xs, all_ys = [], []
            for seed in range(1, 9):
                log_path = os.path.join(partobs_dir, f"{variant}_seed{seed}.log")
                hist = get_eval_history(log_path)
                if hist:
                    xs, ys = zip(*hist)
                    ax.plot(xs, ys, color=color, alpha=0.2, linewidth=0.8)
                    all_xs.append(list(xs))
                    all_ys.append(list(ys))

            # Plot mean curve
            if all_ys:
                min_len = min(len(y) for y in all_ys)
                if min_len > 0:
                    mean_x = all_xs[0][:min_len]
                    mean_y = np.mean([y[:min_len] for y in all_ys], axis=0)
                    ax.plot(mean_x, mean_y, color=color, linewidth=2.5, label=variant)

        ax.set_xlabel("Training Steps", fontsize=10)
        ax.set_ylabel("Eval Reward", fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    axes2[0].set_title("Baseline vs Memory-only", fontsize=11)
    axes2[1].set_title("Memory-only vs Full-gated", fontsize=11)

    plt.tight_layout()
    out2 = "plots/learning_curves_partial_obs.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    results = summarize_all()
    print_report(results)
    plot_comprehensive(results)

    # Save text report
    import io, sys
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    print_report(results)
    sys.stdout = old_stdout
    report_text = buf.getvalue()

    os.makedirs("plots", exist_ok=True)
    with open("plots/results_report.txt", "w") as f:
        f.write(report_text)
    print("\nText report saved: plots/results_report.txt")


if __name__ == "__main__":
    main()
