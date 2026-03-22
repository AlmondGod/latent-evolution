"""
null_analysis.py — Compute silhouette vs shuffled-null baseline across all runs.

For every (N, variant, seed), loads last 3 probe files, computes:
  - actual_sil   : real silhouette on PCA-reduced h
  - shuffled_sil : mean sil after row-wise permutation (null distribution)
  - above_null   : actual_sil - shuffled_sil  (> 0 = genuine structure)
  - ratio        : actual_sil / shuffled_sil   (> 1 = genuine structure)

Then plots above_null vs N for all variants.

Usage:
    python -m new.memetic_foundation.analysis.null_analysis
"""

from __future__ import annotations

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

Ns = [3, 5, 7, 10, 12, 15, 17, 20]
VARIANTS = ["memory_only", "commnet_persistent", "memory_only_persistent", "commnet"]
BASE_DIRS = {
    3:  "new/memetic_foundation/checkpoints/meme_analysis_n3",
    5:  "new/memetic_foundation/checkpoints/meme_analysis",
    7:  "new/memetic_foundation/checkpoints/meme_analysis_n7",
    10: "new/memetic_foundation/checkpoints/meme_analysis_n10",
    12: "new/memetic_foundation/checkpoints/meme_analysis_n12",
    15: "new/memetic_foundation/checkpoints/meme_analysis_n15",
    17: "new/memetic_foundation/checkpoints/meme_analysis_n17",
    20: "new/memetic_foundation/checkpoints/meme_analysis_n20",
}

# Variant → log prefix mapping (how seed dirs are named on disk)
VARIANT_PREFIXES = {
    "memory_only":            "memory_only_seed",
    "commnet_persistent":     "commnet_persistent_seed",
    "memory_only_persistent": "memory_only_persistent_seed",
    "commnet":                "commnet_seed",
}

COLORS  = {"memory_only": "#1f77b4", "commnet_persistent": "#d62728",
           "memory_only_persistent": "#2ca02c", "commnet": "#ff7f0e"}
MARKERS = {"memory_only": "o", "commnet_persistent": "s",
           "memory_only_persistent": "^", "commnet": "D"}
LABELS  = {"memory_only": "memory_only", "commnet_persistent": "commnet_persistent",
           "memory_only_persistent": "memory_only_persistent", "commnet": "commnet"}

MAX_SAMPLES   = 2000   # cap for speed; still representative
N_SHUFFLES    = 5      # fewer shuffles, still gives stable mean
N_LAST_PROBES = 1      # just the final probe (most trained, fastest)

OUT_DIR = "new/scaling_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_sil(h_pool: np.ndarray, k: int = 4) -> float | None:
    if h_pool.shape[0] < k * 2:
        return None
    if h_pool.shape[0] > MAX_SAMPLES:
        idx = np.random.choice(h_pool.shape[0], MAX_SAMPLES, replace=False)
        h_pool = h_pool[idx]
    scaler = StandardScaler()
    h_s = scaler.fit_transform(h_pool)
    pca  = PCA(n_components=min(10, h_pool.shape[1], h_pool.shape[0] - 1))
    h_p  = pca.fit_transform(h_s)
    km   = KMeans(n_clusters=k, n_init=5, random_state=42)
    lbl  = km.fit_predict(h_p)
    if len(np.unique(lbl)) < 2:
        return None
    return float(silhouette_score(h_p, lbl)), h_p, lbl, k


def compute_shuffled_sil(h_p: np.ndarray, k: int,
                          n_shuffles: int = N_SHUFFLES) -> tuple[float, float]:
    """Return (mean, std) of sil scores on column-permuted h_p."""
    rng = np.random.default_rng(1)
    scores = []
    for _ in range(n_shuffles):
        h_shuf = h_p.copy()
        for col in range(h_shuf.shape[1]):
            h_shuf[:, col] = rng.permutation(h_shuf[:, col])
        km = KMeans(n_clusters=k, n_init=5, random_state=42)
        lbl = km.fit_predict(h_shuf)
        if len(np.unique(lbl)) < 2:
            continue
        scores.append(
            silhouette_score(h_shuf, lbl, sample_size=min(MAX_SAMPLES, len(h_shuf)))
        )
    if not scores:
        return 0.0, 0.0
    return float(np.mean(scores)), float(np.std(scores))


def load_seed(bdir: str, prefix: str, seed: int) -> np.ndarray | None:
    """Load and pool h from last N_LAST_PROBES probe files for one seed."""
    probe_dirs = glob.glob(f"{bdir}/{prefix}{seed}/*/probes")
    if not probe_dirs:
        return None
    files = sorted(glob.glob(f"{probe_dirs[0]}/probe_*.npz"))
    if not files:
        return None
    files = files[-N_LAST_PROBES:]
    chunks = []
    for f in files:
        d = np.load(f)
        h = d["h"]          # (T, A, H)
        T, A, H = h.shape
        chunks.append(h.reshape(T * A, H))
    return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

np.random.seed(0)

# results[variant][n] = list of (actual_sil, shuffled_mean, above_null) per seed
results: dict[str, dict[int, list]] = {v: {n: [] for n in Ns} for v in VARIANTS}

print(f"{'N':>4} {'variant':<28} {'actual_sil':>11} {'shuffled_sil':>13} "
      f"{'above_null':>11} {'ratio':>7}  n_seeds")

for n in Ns:
    bdir = BASE_DIRS[n]
    for v in VARIANTS:
        prefix = VARIANT_PREFIXES[v]
        seed_rows = []
        for seed in range(1, 6):
            h_pool = load_seed(bdir, prefix, seed)
            if h_pool is None:
                continue
            out = compute_sil(h_pool)
            if out is None:
                continue
            actual, h_p, lbl, k = out
            shuf_mean, shuf_std = compute_shuffled_sil(h_p, k)
            above = actual - shuf_mean
            ratio = actual / shuf_mean if shuf_mean > 0 else float("inf")
            seed_rows.append((actual, shuf_mean, above, ratio))

        results[v][n] = seed_rows
        if seed_rows:
            arr = np.array(seed_rows)
            print(f"{n:>4} {v:<28} "
                  f"{np.mean(arr[:,0]):>7.3f}±{np.std(arr[:,0]):.3f}  "
                  f"{np.mean(arr[:,1]):>7.3f}±{np.std(arr[:,1]):.3f}  "
                  f"{np.mean(arr[:,2]):>+7.3f}  "
                  f"{np.mean(arr[:,3]):>7.2f}x  "
                  f"n={len(seed_rows)}")
        else:
            print(f"{n:>4} {v:<28}  NO DATA")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Silhouette vs Shuffled Null Baseline", fontsize=13)

for v in VARIANTS:
    xs_actual, ys_actual, err_actual = [], [], []
    xs_null,   ys_null,   err_null   = [], [], []
    xs_above,  ys_above,  err_above  = [], [], []

    for n in Ns:
        rows = results[v][n]
        if not rows:
            continue
        arr = np.array(rows)
        xs_actual.append(n); ys_actual.append(np.mean(arr[:, 0])); err_actual.append(np.std(arr[:, 0]))
        xs_null.append(n);   ys_null.append(np.mean(arr[:, 1]));   err_null.append(np.std(arr[:, 1]))
        xs_above.append(n);  ys_above.append(np.mean(arr[:, 2]));  err_above.append(np.std(arr[:, 2]))

    kw = dict(color=COLORS[v], marker=MARKERS[v], linewidth=1.8, markersize=6)

    # Panel 1: actual sil (solid) vs shuffled sil (dashed)
    axes[0].errorbar(xs_actual, ys_actual, yerr=err_actual, label=LABELS[v], **kw)
    axes[0].errorbar(xs_null,   ys_null,   yerr=err_null,
                     color=COLORS[v], marker=MARKERS[v], linewidth=1.0,
                     markersize=4, linestyle="--", alpha=0.5)

    # Panel 2: above-null (actual - shuffled)
    axes[1].errorbar(xs_above, ys_above, yerr=err_above, label=LABELS[v], **kw)

    # Panel 3: ratio (actual / shuffled)
    xs_ratio, ys_ratio, err_ratio = [], [], []
    for n in Ns:
        rows = results[v][n]
        if not rows:
            continue
        arr = np.array(rows)
        xs_ratio.append(n); ys_ratio.append(np.mean(arr[:, 3])); err_ratio.append(np.std(arr[:, 3]))
    axes[2].errorbar(xs_ratio, ys_ratio, yerr=err_ratio, label=LABELS[v], **kw)

axes[0].set_title("Actual sil (solid) vs Null sil (dashed)")
axes[0].set_xlabel("N agents"); axes[0].set_ylabel("Silhouette score")
axes[0].axhline(0, color="k", linewidth=0.5)
axes[0].legend(fontsize=8)

axes[1].set_title("Above-null  (actual − shuffled)")
axes[1].set_xlabel("N agents"); axes[1].set_ylabel("Δ silhouette")
axes[1].axhline(0, color="k", linewidth=1.0, linestyle="--")
axes[1].legend(fontsize=8)

axes[2].set_title("Ratio  (actual / shuffled)")
axes[2].set_xlabel("N agents"); axes[2].set_ylabel("Sil ratio")
axes[2].axhline(1, color="k", linewidth=1.0, linestyle="--")
axes[2].legend(fontsize=8)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "fig_null_baseline.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out_path}")
