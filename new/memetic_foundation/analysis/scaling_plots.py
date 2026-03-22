"""
scaling_plots.py — Scaling analysis plots for memetic dynamics paper.

Generates:
  1. Performance vs N
  2. Key meme metrics vs N — including null baseline for silhouette
  3. Phase transition / commnet collapse figure

Usage:
  python -m new.memetic_foundation.analysis.scaling_plots
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Output dir ────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), "../../scaling_analysis")
os.makedirs(out_dir, exist_ok=True)

# ── N values ──────────────────────────────────────────────────────────────────
Ns        = [3, 5, 7, 10, 12, 15, 17, 20]
Ns_comm   = [3, 5, 10, 20]   # commnet (no persistence) only run at these N

# ── Performance data ──────────────────────────────────────────────────────────
# Median final dist across 5 seeds (lower = better).  None = not measured.
perf_dist = {
    "memory_only":            [0.580, 0.640, 0.670, 0.630, 0.580, 0.570, 0.420, 0.860],
    "commnet_persistent":     [0.600, 0.740, 0.500, 0.630, 1.080, 0.430, 0.550, 0.460],
    "memory_only_persistent": [0.600, 0.580, 0.710, 0.630, 0.580, 0.550, 0.500, 0.430],
    # commnet (no persistence): most seeds collapsed (dist ~70); median reflects failure
    "commnet": {3: 0.63, 5: 66.85, 10: 77.81, 20: 31.4},
}
perf_rew = {
    "memory_only":            [ -603,  -886, -1045, -1439, -1601, -1808, -1444, -2528],
    "commnet_persistent":     [ -567,  -974,  -929, -1527, -2661, -1461, -1853, -1994],
    "memory_only_persistent": [ -628,  -871, -1195, -1599, -1491, -1693, -1830, -1809],
    "commnet": {3: -659, 5: -28797, 10: -73706, 20: -52567},
}

# ── Null baseline silhouette (per-seed mean, 5k sample cap) ───────────────────
# real_sil ≈ null_sil everywhere → clusters are GRU geometry artifacts
real_sil = {
    "memory_only":            [0.217, 0.199, 0.244, 0.332, 0.252, 0.230, 0.328, 0.440],
    "commnet_persistent":     [0.242, 0.199, 0.213, 0.409, 0.352, 0.246, 0.315, 0.344],
    "memory_only_persistent": [0.189, 0.239, 0.208, 0.317, 0.247, 0.271, 0.313, 0.305],
    "commnet": {3: 0.396, 5: 0.387, 10: 0.457, 20: 0.315},
}
null_sil = {
    "memory_only":            [0.215, 0.202, 0.244, 0.331, 0.249, 0.231, 0.329, 0.450],
    "commnet_persistent":     [0.239, 0.200, 0.210, 0.409, 0.354, 0.242, 0.313, 0.341],
    "memory_only_persistent": [0.188, 0.228, 0.214, 0.316, 0.249, 0.273, 0.311, 0.305],
    "commnet": {3: 0.396, 5: 0.375, 10: 0.455, 20: 0.314},
}

# ── Q5 delta silhouette ───────────────────────────────────────────────────────
q5_dsil = {
    "memory_only":            [0.289, 0.176, 0.192, 0.413, 0.924, 0.855, 0.236, 0.260],
    "commnet_persistent":     [0.405, 0.211, 0.237, 0.976, 0.967, 0.191, 0.292, 0.963],
    "memory_only_persistent": [0.301, 0.950, 0.313, 0.980, 0.259, 0.926, 0.211, 0.944],
}

# ── Style ─────────────────────────────────────────────────────────────────────
COLORS = {
    "memory_only":            "#4C72B0",
    "commnet_persistent":     "#DD8452",
    "memory_only_persistent": "#55A868",
    "commnet":                "#C44E52",
}
LABELS = {
    "memory_only":            "Memory Only (episodic)",
    "commnet_persistent":     "CommNet + Persistent",
    "memory_only_persistent": "Memory Persistent",
    "commnet":                "CommNet (episodic) ✗",
}
MARKERS = {
    "memory_only":            "o",
    "commnet_persistent":     "s",
    "memory_only_persistent": "^",
    "commnet":                "X",
}


def _xs_ys(data, variant, Ns_list):
    """Extract non-None xs/ys; handles both list and dict formats."""
    d = data[variant]
    if isinstance(d, dict):
        xs = [n for n in Ns_list if n in d and d[n] is not None]
        ys = [d[n] for n in xs]
    else:
        xs = [n for n, v in zip(Ns_list, d) if v is not None]
        ys = [v for v in d if v is not None]
    return xs, ys


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — Performance
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Task Performance vs. Number of Agents", fontsize=13, fontweight="bold")

core_variants = ["memory_only", "commnet_persistent", "memory_only_persistent"]
all_variants  = core_variants + ["commnet"]

for ax, data, ylabel, title, hi in [
    (axes[0], perf_dist, "Median Distance to Landmark", "Final Distance  (↓ better)", False),
    (axes[1], perf_rew,  "Median Episode Reward",        "Final Reward  (↑ better)",  True),
]:
    for v in all_variants:
        xs, ys = _xs_ys(data, v, Ns)
        lw  = 1.5 if v == "commnet" else 2.0
        ls  = "--" if v == "commnet" else "-"
        ax.plot(xs, ys, marker=MARKERS[v], color=COLORS[v], label=LABELS[v],
                linewidth=lw, linestyle=ls, markersize=8)
    ax.set_xticks(Ns)
    ax.set_xlabel("Number of Agents (N)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
           bbox_to_anchor=(0.5, -0.06))
plt.tight_layout(rect=[0, 0.10, 1, 1])
fig.savefig(os.path.join(out_dir, "fig1_performance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig1_performance.png")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — Null baseline: real sil vs shuffled sil
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Silhouette Score: Real vs. Shuffled Null Baseline\n"
             "Real ≈ Null everywhere → hidden-state clusters are GRU geometry artifacts, not memes",
             fontsize=12, fontweight="bold")

variant_labels = {
    "memory_only":            "Mem Only",
    "commnet_persistent":     "CommNet+Persist",
    "memory_only_persistent": "Mem Persist",
}

for ax, v in zip(axes, core_variants):
    xs_r, ys_r = _xs_ys(real_sil, v, Ns)
    xs_n, ys_n = _xs_ys(null_sil, v, Ns)
    above = [r - n for r, n in zip(ys_r, ys_n)]

    ax.plot(xs_r, ys_r, marker="o", color=COLORS[v], linewidth=2,
            markersize=7, label="Real sil")
    ax.plot(xs_n, ys_n, marker="o", linestyle="--", color="gray", linewidth=1.5,
            markersize=5, label="Shuffled null")
    ax.fill_between(xs_r, ys_r, ys_n, alpha=0.15, color=COLORS[v])

    ax2 = ax.twinx()
    ax2.bar(xs_r, above, width=0.6, alpha=0.25, color=COLORS[v], label="Above null")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax2.set_ylabel("Real − Null", fontsize=8, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)
    ax2.set_ylim(-0.05, 0.15)

    ax.set_xticks(Ns)
    ax.set_xlabel("N", fontsize=10)
    ax.set_ylabel("Silhouette Score", fontsize=10)
    ax.set_title(variant_labels[v], fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig2_null_baseline.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2_null_baseline.png")


# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — CommNet collapse + delta sil
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Key Findings: CommNet Collapse Without Persistence  &  Delta Silhouette Trends",
             fontsize=12, fontweight="bold")

# Left: distance highlighting commnet collapse
ax = axes[0]
for v in all_variants:
    xs, ys = _xs_ys(perf_dist, v, Ns)
    lw = 1.5 if v == "commnet" else 2.0
    ls = "--" if v == "commnet" else "-"
    ax.plot(xs, ys, marker=MARKERS[v], color=COLORS[v], label=LABELS[v],
            linewidth=lw, linestyle=ls, markersize=8)
ax.set_yscale("log")
ax.set_xticks(Ns)
ax.set_xlabel("Number of Agents (N)", fontsize=11)
ax.set_ylabel("Median Final Distance (log scale)", fontsize=10)
ax.set_title("CommNet without persistence collapses at N≥5\n(log scale reveals divergence)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which="both")

# Right: delta silhouette (note: also near null — same caveat applies)
ax = axes[1]
for v in core_variants:
    xs, ys = _xs_ys(q5_dsil, v, Ns)
    ax.plot(xs, ys, marker=MARKERS[v], color=COLORS[v], label=LABELS[v],
            linewidth=2, markersize=8)
ax.set_xticks(Ns)
ax.set_xlabel("Number of Agents (N)", fontsize=11)
ax.set_ylabel("Delta Silhouette", fontsize=11)
ax.set_title("Q5 Mutation Structure vs N\n(interpret cautiously — null baseline not computed)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig3_findings.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3_findings.png")
print(f"\nAll figures saved to: {os.path.abspath(out_dir)}")
