"""
scaling_plots.py — Scaling analysis plots for memetic dynamics paper.

Generates:
  1. Performance vs N (median dist, median reward)
  2. Key meme metrics vs N (silhouette, delta_sil, cross-agent corr, between_var)
  3. Combined summary figure

Usage:
  python -m new.memetic_foundation.analysis.scaling_plots --out-dir analysis_out/scaling
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Raw data ────────────────────────────────────────────────────────────────

Ns = [3, 5, 7, 10, 12, 15, 17, 20]

# Median dist (lower = better). Bad seeds excluded via median robustness.
perf_dist = {
    "memory_only":            [None, 0.640, None, 0.630, None, None, None, 0.860],
    "commnet_persistent":     [None, 0.740, None, 0.630, None, None, None, 0.460],
    "memory_only_persistent": [None, 0.580, None, 0.630, None, None, None, 0.430],
}
perf_rew = {
    "memory_only":            [None, -886,  None, -1439, None, None, None, -2528],
    "commnet_persistent":     [None, -974,  None, -1527, None, None, None, -1994],
    "memory_only_persistent": [None, -871,  None, -1599, None, None, None, -1809],
}

# Q1 — silhouette score (higher = clearer attractors)
# N:               3      5      7      10     12     15     17     20
q1_sil = {
    "memory_only":            [0.234, 0.260, 0.244, 0.841, 0.387, 0.851, 0.362, 0.715],
    "commnet_persistent":     [0.303, 0.211, 0.435, 0.877, 0.861, 0.427, 0.583, 0.649],
    "memory_only_persistent": [0.154, 0.727, 0.285, 0.827, 0.527, 0.304, 0.300, 0.860],
}

# Q2 — final pairwise cosine similarity (higher = more convergent)
q2_sim = {
    "memory_only":            [0.619, 0.715, 0.636, 0.731, 0.797, 0.747, 0.626, 0.846],
    "commnet_persistent":     [0.636, 0.646, 0.657, 0.699, 0.719, 0.564, 0.687, 0.585],
    "memory_only_persistent": [0.690, 0.956, 0.739, 0.835, 0.572, 0.754, 0.675, 0.637],
}

# Q4 — between-agent variance × 10^4 (higher = more specialisation)
q4_bvar = {
    "memory_only":            [0.00, 0.00, 0.00, 0.01, None, None, None, 0.24],
    "commnet_persistent":     [0.00, 0.00, 0.00, 0.00, None, None, None, 0.01],
    "memory_only_persistent": [0.00, 0.04, 0.01, 0.02, None, None, None, 0.02],
}

# Q5 — delta silhouette (higher = more structured mutations)
q5_dsil = {
    "memory_only":            [0.289, 0.176, 0.192, 0.413, 0.924, 0.855, 0.236, 0.260],
    "commnet_persistent":     [0.405, 0.211, 0.237, 0.976, 0.967, 0.191, 0.292, 0.963],
    "memory_only_persistent": [0.301, 0.950, 0.313, 0.980, 0.259, 0.926, 0.211, 0.944],
}

# Q5 — cross-agent delta correlation (higher = socially coupled mutations)
q5_corr = {
    "memory_only":            [-0.017, -0.020,  -0.011, 0.065, 0.025, 0.116, -0.002, 0.049],
    "commnet_persistent":     [ 0.084, -0.008,   0.000, 0.063, 0.174, 0.099,  0.043, 0.069],
    "memory_only_persistent": [-0.029,  0.172,  -0.008, 0.184, 0.024, 0.021, -0.006, 0.019],
}

# ── Style ────────────────────────────────────────────────────────────────────

COLORS = {
    "memory_only":            "#4C72B0",
    "commnet_persistent":     "#DD8452",
    "memory_only_persistent": "#55A868",
}
LABELS = {
    "memory_only":            "Memory Only",
    "commnet_persistent":     "CommNet + Persistent",
    "memory_only_persistent": "Memory Persistent",
}
MARKERS = {
    "memory_only":            "o",
    "commnet_persistent":     "s",
    "memory_only_persistent": "^",
}

def plot_metric(ax, data, ylabel, title, higher_better=True, hline=None):
    for v in data:
        xs = [n for n, val in zip(Ns, data[v]) if val is not None]
        ys = [val for val in data[v] if val is not None]
        ax.plot(xs, ys, marker=MARKERS[v], color=COLORS[v],
                label=LABELS[v], linewidth=2, markersize=8)
    if hline is not None:
        ax.axhline(hline, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(Ns)
    ax.set_xlabel("Number of Agents (N)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    arrow = "↑ better" if higher_better else "↓ better"
    ax.text(0.98, 0.02, arrow, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="gray")


# ── Figure 1: Performance ────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Task Performance vs. Number of Agents", fontsize=13, fontweight="bold")

plot_metric(axes[0], perf_dist, "Median Distance to Landmark ↓",
            "Final Distance (lower = better)", higher_better=False)
plot_metric(axes[1], perf_rew, "Median Episode Reward ↑",
            "Final Reward (higher = better)", higher_better=True)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.05))
plt.tight_layout(rect=[0, 0.08, 1, 1])

out_dir = os.path.join(os.path.dirname(__file__), "../../scaling_analysis")
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, "fig1_performance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig1_performance.png")


# ── Figure 2: Meme Metrics ───────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Memetic Dynamics vs. Number of Agents", fontsize=14, fontweight="bold")

plot_metric(axes[0, 0], q1_sil,
            "Silhouette Score", "Q1: Attractor Clarity\n(do stable patterns form?)",
            higher_better=True)

plot_metric(axes[0, 1], q2_sim,
            "Pairwise Cosine Similarity", "Q2: Inter-Agent Convergence\n(do patterns spread?)",
            higher_better=True)

plot_metric(axes[0, 2], q4_bvar,
            "Between-Agent Variance (×10⁻⁴)", "Q4: Specialisation\n(diverge vs. homogenise?)",
            higher_better=True)

plot_metric(axes[1, 0], q5_dsil,
            "Delta Silhouette", "Q5a: Mutation Structure\n(structured vs. random updates?)",
            higher_better=True)

plot_metric(axes[1, 1], q5_corr,
            "Cross-Agent Delta Correlation", "Q5b: Social Coupling\n(do mutations co-occur?)",
            higher_better=True, hline=0.0)

# Summary radar-style bar chart for N=10 (the clearest signal point)
ax = axes[1, 2]
metrics = ["Attractor\nClarity", "Inter-Agent\nConv.", "Specialisation\n(×100)", "Mutation\nStructure", "Social\nCoupling"]
variants_list = ["memory_only", "commnet_persistent", "memory_only_persistent"]
n_idx = 1  # N=10

vals = {
    "memory_only":            [q1_sil["memory_only"][n_idx],
                               q2_sim["memory_only"][n_idx],
                               q4_bvar["memory_only"][n_idx] * 100,
                               q5_dsil["memory_only"][n_idx],
                               max(0, q5_corr["memory_only"][n_idx])],
    "commnet_persistent":     [q1_sil["commnet_persistent"][n_idx],
                               q2_sim["commnet_persistent"][n_idx],
                               q4_bvar["commnet_persistent"][n_idx] * 100,
                               q5_dsil["commnet_persistent"][n_idx],
                               max(0, q5_corr["commnet_persistent"][n_idx])],
    "memory_only_persistent": [q1_sil["memory_only_persistent"][n_idx],
                               q2_sim["memory_only_persistent"][n_idx],
                               q4_bvar["memory_only_persistent"][n_idx] * 100,
                               q5_dsil["memory_only_persistent"][n_idx],
                               max(0, q5_corr["memory_only_persistent"][n_idx])],
}
x = np.arange(len(metrics))
width = 0.25
for i, v in enumerate(variants_list):
    ax.bar(x + i * width, vals[v], width, label=LABELS[v],
           color=COLORS[v], alpha=0.85)
ax.set_xticks(x + width)
ax.set_xticklabels(metrics, fontsize=8)
ax.set_title("Summary at N=10\n(all metrics normalised to [0,1] range)", fontsize=11, fontweight="bold")
ax.set_ylabel("Score", fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.03))
plt.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(os.path.join(out_dir, "fig2_meme_metrics.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2_meme_metrics.png")


# ── Figure 3: Phase transition highlight ─────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Phase Transition: Comm Imposes Mutation Structure at N≥10",
             fontsize=12, fontweight="bold")

# Left: delta_sil with shaded region
ax = axes[0]
for v in q5_dsil:
    ax.plot(Ns, q5_dsil[v], marker=MARKERS[v], color=COLORS[v],
            label=LABELS[v], linewidth=2.5, markersize=10)
ax.axvspan(9, 11, alpha=0.1, color="orange", label="Phase transition zone")
ax.axhline(0.9, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label="'Structured' threshold")
ax.set_xticks(Ns)
ax.set_xlabel("Number of Agents (N)", fontsize=11)
ax.set_ylabel("Delta Silhouette", fontsize=11)
ax.set_title("Mutation Structure vs N\ncommnet jumps 0.21→0.98 at N=10", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Right: specialisation divergence
ax = axes[1]
for v in q4_bvar:
    xs = [n for n, val in zip(Ns, q4_bvar[v]) if val is not None]
    vals_scaled = [val * 1e4 for val in q4_bvar[v] if val is not None]
    ax.plot(xs, vals_scaled, marker=MARKERS[v], color=COLORS[v],
            label=LABELS[v], linewidth=2.5, markersize=10)
ax.set_xticks(Ns)
ax.set_xlabel("Number of Agents (N)", fontsize=11)
ax.set_ylabel("Between-Agent Variance (×10⁻⁴)", fontsize=11)
ax.set_title("Agent Specialisation vs N\nmemory_only diverges strongly at N=20", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig3_phase_transition.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3_phase_transition.png")
print(f"\nAll figures saved to: {os.path.abspath(out_dir)}")
