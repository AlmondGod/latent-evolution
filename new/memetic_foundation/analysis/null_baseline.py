"""
null_baseline.py — Compute real vs shuffled silhouette scores across all probe data.

For each (N, variant, seed): compute actual sil, then shuffle h vectors randomly
across agents and timesteps and recompute sil. The shuffled score is the null
distribution — if actual ≈ shuffled, clusters are a GRU geometry artifact.

Usage:
  python -m new.memetic_foundation.analysis.null_baseline
"""
import os, sys, glob, numpy as np
os.chdir(os.path.join(os.path.dirname(__file__), "../../.."))

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

MAX_SAMPLES = 5000
N_SHUFFLES  = 5      # average over multiple shuffles for stable null estimate
N_CLUSTERS  = 4
RNG         = np.random.default_rng(0)

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

VARIANTS = [
    "memory_only",
    "commnet_persistent",
    "memory_only_persistent",
    "commnet",           # new — no persistence
]

Ns = [3, 5, 7, 10, 12, 15, 17, 20]


def _sil(h: np.ndarray):
    """Silhouette on up to MAX_SAMPLES rows of h."""
    if h.shape[0] < N_CLUSTERS * 2:
        return None
    if h.shape[0] > MAX_SAMPLES:
        idx = RNG.choice(h.shape[0], MAX_SAMPLES, replace=False)
        h = h[idx]
    scaler = StandardScaler()
    h_s = scaler.fit_transform(h)
    n_comp = min(8, h.shape[1], h.shape[0] - 1)
    h_p = PCA(n_components=n_comp, random_state=0).fit_transform(h_s)
    lbl = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=5).fit_predict(h_p)
    if len(np.unique(lbl)) < 2:
        return None
    return float(silhouette_score(h_p, lbl))


def _shuffled_sil(h: np.ndarray):
    """Average sil after randomly permuting all rows (destroys all structure)."""
    scores = []
    for _ in range(N_SHUFFLES):
        h_shuf = h[RNG.permutation(h.shape[0])]
        s = _sil(h_shuf)
        if s is not None:
            scores.append(s)
    return float(np.mean(scores)) if scores else None


def load_h_pool(probe_dir: str, n_last: int = 3):
    """Load last n_last probes and return (flat h pool, list of h arrays)."""
    files = sorted(glob.glob(f"{probe_dir}/probe_*.npz"))
    if not files:
        return None, None
    files = files[-n_last:]
    all_h, all_h_seq = [], []
    for f in files:
        d = np.load(f)
        h = d["h"]          # (T, A, H)
        T, A, H = h.shape
        all_h.append(h.reshape(T * A, H))
        all_h_seq.append(h)
    return np.concatenate(all_h, axis=0), all_h_seq


# ── Main loop ─────────────────────────────────────────────────────────────────

print(f"\n{'N':>4} {'variant':<28} {'real_sil':>9} {'null_sil':>9} {'above_null':>11} {'n':>3}")
print("-" * 70)

results = {}   # (N, variant) -> dict

for n in Ns:
    bdir = BASE_DIRS[n]
    for v in VARIANTS:
        seed_real, seed_null = [], []
        for seed in range(1, 6):
            probe_dirs = glob.glob(f"{bdir}/{v}_seed{seed}/*/probes")
            if not probe_dirs:
                continue
            h_pool, _ = load_h_pool(probe_dirs[0])
            if h_pool is None:
                continue
            r = _sil(h_pool)
            null = _shuffled_sil(h_pool)
            if r is not None:
                seed_real.append(r)
            if null is not None:
                seed_null.append(null)

        if seed_real and seed_null:
            real_mu  = np.mean(seed_real)
            null_mu  = np.mean(seed_null)
            above    = real_mu - null_mu
            n_seeds  = len(seed_real)
            print(f"{n:>4} {v:<28} {real_mu:>9.3f} {null_mu:>9.3f} {above:>+11.3f}  {n_seeds}")
            results[(n, v)] = dict(real=real_mu, real_std=np.std(seed_real),
                                   null=null_mu, null_std=np.std(seed_null),
                                   above=above, n=n_seeds)
        else:
            print(f"{n:>4} {v:<28}  {'NO DATA':>9}")

# ── Summary: above-null table ──────────────────────────────────────────────────

print("\n\n=== ABOVE-NULL SILHOUETTE (real - shuffled) ===")
print(f"{'variant':<28}", end="")
for n in Ns:
    print(f"  N={n:>2}", end="")
print()
for v in VARIANTS:
    print(f"{v:<28}", end="")
    for n in Ns:
        key = (n, v)
        if key in results:
            print(f"  {results[key]['above']:>+5.2f}", end="")
        else:
            print(f"  {'--':>5}", end="")
    print()

print("\n=== NULL SIL (shuffled baseline) ===")
print(f"{'variant':<28}", end="")
for n in Ns:
    print(f"  N={n:>2}", end="")
print()
for v in VARIANTS:
    print(f"{v:<28}", end="")
    for n in Ns:
        key = (n, v)
        if key in results:
            print(f"  {results[key]['null']:>+5.2f}", end="")
        else:
            print(f"  {'--':>5}", end="")
    print()
