"""
meme_analysis.py — Memetic dynamics analysis of probe rollout data.

Answers four questions:
  Q1. PERSISTENCE  — Do stable hidden-state attractors form and lock in?
  Q2. TRANSMISSION — Do patterns spread between agents after communication?
  Q3. SELECTION    — Do high-reward hidden-state patterns dominate over time?
  Q4. STRUCTURE    — Do agents specialise (diverge) or homogenise (converge)?
  Q5. MUTATION     — Do new patterns spawn from existing ones in a structured way?

Usage:
  python -m new.memetic_foundation.analysis.meme_analysis \
      --probe-dirs checkpoints/mem_only/probes checkpoints/commnet_p/probes \
      --labels memory_only commnet_persistent \
      --out-dir analysis_out
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_probes(probe_dir: str) -> List[dict]:
    """Load all .npz probe files from a directory, sorted by training_step."""
    paths = sorted(Path(probe_dir).glob("probe_*.npz"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not paths:
        raise FileNotFoundError(f"No probe files found in {probe_dir}")
    probes = []
    for p in paths:
        d = np.load(str(p))
        probes.append({k: d[k] for k in d.files})
    return probes


def stack_h(probes: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack hidden states across all probes.

    Returns:
      h_all   : (N_total, hidden_dim)  — all h vectors flattened across time × agents
      steps   : (N_total,)             — training_step each row came from
      agents  : (N_total,)             — agent index each row came from
    """
    h_list, step_list, agent_list = [], [], []
    for probe in probes:
        h   = probe["h"]        # (T, n_agents, hidden_dim)
        T, A, D = h.shape
        step = int(probe["training_step"])
        for a in range(A):
            h_list.append(h[:, a, :])           # (T, D)
            step_list.append(np.full(T, step))
            agent_list.append(np.full(T, a))
    return (np.concatenate(h_list, axis=0),
            np.concatenate(step_list),
            np.concatenate(agent_list))


# ---------------------------------------------------------------------------
# Q1 — PERSISTENCE: attractors and cluster stability
# ---------------------------------------------------------------------------

def q1_persistence(probes: List[dict], label: str, out_dir: str) -> dict:
    """PCA + k-means clustering. Track cluster membership over training time."""
    h_all, steps, agents = stack_h(probes)
    n_agents = int(agents.max()) + 1

    # PCA to 2D for visualisation, keep 10D for clustering
    pca2  = PCA(n_components=2)
    pca10 = PCA(n_components=min(10, h_all.shape[1]))
    h2  = pca2.fit_transform(h_all)
    h10 = pca10.fit_transform(h_all)

    # Choose k via silhouette (search k=2..8)
    best_k, best_score = 2, -1
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels_k = km.fit_predict(h10)
        sc = silhouette_score(h10, labels_k, sample_size=min(5000, len(h10)))
        if sc > best_score:
            best_score, best_k = sc, k

    km_best = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    cluster_labels = km_best.fit_predict(h10)

    # PCA scatter coloured by cluster
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    unique_steps = np.unique(steps)
    sc = axes[0].scatter(h2[:, 0], h2[:, 1], c=cluster_labels,
                         cmap="tab10", alpha=0.3, s=4)
    axes[0].set_title(f"[{label}] PCA of h — k={best_k} clusters (sil={best_score:.3f})")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    plt.colorbar(sc, ax=axes[0], label="cluster")

    # Cluster prevalence over training time
    cluster_counts = np.zeros((len(unique_steps), best_k))
    for i, s in enumerate(unique_steps):
        mask = steps == s
        for c in range(best_k):
            cluster_counts[i, c] = (cluster_labels[mask] == c).sum()
    cluster_frac = cluster_counts / cluster_counts.sum(axis=1, keepdims=True)
    for c in range(best_k):
        axes[1].plot(unique_steps, cluster_frac[:, c], label=f"cluster {c}")
    axes[1].set_title(f"[{label}] Cluster prevalence over training")
    axes[1].set_xlabel("Training step"); axes[1].set_ylabel("Fraction of timesteps")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"q1_persistence_{label}.png"), dpi=120)
    plt.close()

    # Per-agent cluster entropy (low = locked in, high = wandering)
    per_agent_entropy = {}
    for a in range(n_agents):
        mask = agents == a
        cl = cluster_labels[mask]
        counts = np.bincount(cl, minlength=best_k).astype(float)
        probs = counts / counts.sum()
        ent = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
        per_agent_entropy[a] = float(ent)

    return {"best_k": best_k, "silhouette": best_score,
            "per_agent_entropy": per_agent_entropy,
            "cluster_labels": cluster_labels, "h10": h10,
            "steps": steps, "agents": agents}


# ---------------------------------------------------------------------------
# Q2 — TRANSMISSION: do patterns spread between agents after comm?
# ---------------------------------------------------------------------------

def q2_transmission(probes: List[dict], label: str, out_dir: str) -> dict:
    """Pairwise cosine similarity between agents' h over training time.

    For commnet_persistent: similarity should *increase* over training as
    agents share patterns via communication.
    For memory_only: similarity should stay flat or drift randomly.
    """
    unique_steps = sorted({int(p["training_step"]) for p in probes})
    mean_sim_over_time = []
    std_sim_over_time  = []

    for probe in probes:
        h = probe["h"]      # (T, n_agents, hidden_dim)
        T, A, D = h.shape
        # Flatten time: compute mean pairwise cosine sim at each step
        sims = []
        for t in range(T):
            h_t = normalize(h[t], norm="l2")  # (A, D) normalised
            sim_matrix = h_t @ h_t.T           # (A, A)
            # Off-diagonal only
            idx = np.triu_indices(A, k=1)
            sims.extend(sim_matrix[idx].tolist())
        mean_sim_over_time.append(np.mean(sims))
        std_sim_over_time.append(np.std(sims))

    # Also: does receiving m_bar make agent more similar to sender?
    # Proxy: correlation between ||m_bar_i|| and pairwise h sim
    m_norms, h_sims = [], []
    for probe in probes:
        h   = probe["h"]      # (T, A, D)
        m   = probe["m_bar"]  # (T, A, D)
        T, A, D = h.shape
        for t in range(T):
            h_t = normalize(h[t], norm="l2")
            sim_mat = h_t @ h_t.T
            idx = np.triu_indices(A, k=1)
            mean_pair_sim = sim_mat[idx].mean()
            mean_m_norm   = np.linalg.norm(m[t], axis=-1).mean()
            h_sims.append(mean_pair_sim)
            m_norms.append(mean_m_norm)

    corr = float(np.corrcoef(m_norms, h_sims)[0, 1]) if len(m_norms) > 2 else 0.0

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(unique_steps, mean_sim_over_time, marker="o")
    axes[0].fill_between(unique_steps,
                         np.array(mean_sim_over_time) - np.array(std_sim_over_time),
                         np.array(mean_sim_over_time) + np.array(std_sim_over_time),
                         alpha=0.2)
    axes[0].set_title(f"[{label}] Mean pairwise cosine similarity of h over training")
    axes[0].set_xlabel("Training step"); axes[0].set_ylabel("Cosine similarity")

    axes[1].scatter(m_norms, h_sims, alpha=0.2, s=4)
    axes[1].set_title(f"[{label}] ||m_bar|| vs pairwise h similarity\n(corr={corr:.3f})")
    axes[1].set_xlabel("Mean ||m_bar||"); axes[1].set_ylabel("Pairwise h cosine sim")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"q2_transmission_{label}.png"), dpi=120)
    plt.close()

    return {"mean_sim_over_time": mean_sim_over_time,
            "m_h_correlation": corr}


# ---------------------------------------------------------------------------
# Q3 — SELECTION: do high-reward clusters dominate over time?
# ---------------------------------------------------------------------------

def q3_selection(probes: List[dict], q1_result: dict,
                 label: str, out_dir: str) -> dict:
    """Correlate cluster membership with reward, track winning cluster over time."""
    cluster_labels = q1_result["cluster_labels"]
    steps          = q1_result["steps"]
    agents         = q1_result["agents"]
    best_k         = q1_result["best_k"]

    # Flatten rewards to match h_all row ordering
    rewards_flat = []
    for probe in probes:
        rew = probe["rewards"]  # (T, n_agents)
        T, A = rew.shape
        for a in range(A):
            rewards_flat.extend(rew[:, a].tolist())
    rewards_flat = np.array(rewards_flat)

    # Mean reward per cluster
    cluster_mean_rew = {}
    for c in range(best_k):
        mask = cluster_labels == c
        cluster_mean_rew[c] = float(rewards_flat[mask].mean()) if mask.sum() > 0 else 0.0

    # Best cluster by reward
    best_cluster = max(cluster_mean_rew, key=cluster_mean_rew.get)

    # Prevalence of best cluster over training time
    unique_steps = np.unique(steps)
    best_cluster_frac = []
    for s in unique_steps:
        mask = steps == s
        frac = (cluster_labels[mask] == best_cluster).mean()
        best_cluster_frac.append(frac)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    clusters = list(cluster_mean_rew.keys())
    mean_rews = [cluster_mean_rew[c] for c in clusters]
    axes[0].bar([f"C{c}" for c in clusters], mean_rews,
                color=plt.cm.tab10(np.linspace(0, 1, best_k)))
    axes[0].set_title(f"[{label}] Mean reward per cluster")
    axes[0].set_xlabel("Cluster"); axes[0].set_ylabel("Mean reward")

    axes[1].plot(unique_steps, best_cluster_frac, marker="o", color="green")
    axes[1].set_title(f"[{label}] Prevalence of best cluster (C{best_cluster}) over training")
    axes[1].set_xlabel("Training step"); axes[1].set_ylabel("Fraction")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"q3_selection_{label}.png"), dpi=120)
    plt.close()

    return {"cluster_mean_reward": cluster_mean_rew,
            "best_cluster": best_cluster,
            "best_cluster_prevalence": best_cluster_frac}


# ---------------------------------------------------------------------------
# Q4 — STRUCTURE: specialisation vs homogenisation
# ---------------------------------------------------------------------------

def q4_structure(probes: List[dict], label: str, out_dir: str) -> dict:
    """Track within-agent h variance vs between-agent h variance over time.

    High between-agent variance → specialisation (agents diverge)
    Low between-agent variance  → homogenisation (shared meme dominates)
    """
    unique_steps = sorted({int(p["training_step"]) for p in probes})
    within_var, between_var = [], []

    for probe in probes:
        h = probe["h"]  # (T, n_agents, D)
        T, A, D = h.shape
        # Between-agent: variance of per-agent means
        agent_means = h.mean(axis=0)   # (A, D)
        b_var = agent_means.var(axis=0).mean()
        # Within-agent: mean of per-agent variances across time
        w_var = np.array([h[:, a, :].var(axis=0).mean() for a in range(A)]).mean()
        between_var.append(float(b_var))
        within_var.append(float(w_var))

    # Per-agent PCA trajectory (final probe only)
    last_probe = probes[-1]
    h_last = last_probe["h"]  # (T, A, D)
    T, A, D = h_last.shape
    pca2 = PCA(n_components=2)
    h_flat = h_last.reshape(T * A, D)
    h_2d = pca2.fit_transform(h_flat).reshape(T, A, 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for a in range(A):
        axes[0].plot(h_2d[:, a, 0], h_2d[:, a, 1],
                     alpha=0.6, label=f"agent {a}", marker=".", markersize=2)
    axes[0].set_title(f"[{label}] Per-agent h trajectory (PCA, final probe)")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].legend(fontsize=7)

    axes[1].plot(unique_steps, within_var,  label="within-agent var",  marker="o")
    axes[1].plot(unique_steps, between_var, label="between-agent var", marker="s")
    axes[1].set_title(f"[{label}] Specialisation vs homogenisation over training")
    axes[1].set_xlabel("Training step"); axes[1].set_ylabel("Variance of h")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"q4_structure_{label}.png"), dpi=120)
    plt.close()

    return {"within_var": within_var, "between_var": between_var}


# ---------------------------------------------------------------------------
# Q5 — MUTATION: do new patterns spawn from old ones in a structured way?
# ---------------------------------------------------------------------------

def q5_mutation(probes: List[dict], q1_result: dict,
                label: str, out_dir: str) -> dict:
    """Analyse h_{t} → h_{t+1} transitions within episodes.

    Two signals:
    (a) Delta magnitude: ||h_{t+1} - h_t|| — how big are 'mutations'?
    (b) Delta structure: are deltas clustered (structured) or isotropic (random)?
    (c) Cross-agent delta correlation: does agent A's delta predict agent B's?
         If yes → mutations are socially driven (comm-induced), not just env-driven.
    """
    delta_magnitudes = []
    delta_vecs = []
    cross_agent_corrs = []

    for probe in probes:
        h    = probe["h"]       # (T, A, D)
        ep   = probe["episode"] # (T,)
        T, A, D = h.shape

        for t in range(T - 1):
            if ep[t] != ep[t + 1]:
                continue  # skip episode boundaries (memory reset)
            delta = h[t + 1] - h[t]   # (A, D)
            delta_magnitudes.append(np.linalg.norm(delta, axis=-1).mean())
            delta_vecs.append(delta.mean(axis=0))  # mean over agents

            # Cross-agent: does agent i's delta correlate with agent j's?
            if A > 1:
                corrs = []
                for i in range(A):
                    for j in range(i + 1, A):
                        c = float(np.corrcoef(delta[i], delta[j])[0, 1])
                        if not np.isnan(c):
                            corrs.append(c)
                if corrs:
                    cross_agent_corrs.append(np.mean(corrs))

    delta_vecs = np.array(delta_vecs) if delta_vecs else np.zeros((1, 1))

    # Cluster deltas to see if mutations are structured
    best_k_d, best_sc_d = 2, -1
    if len(delta_vecs) > 20:
        pca_d = PCA(n_components=min(5, delta_vecs.shape[1]))
        dv_low = pca_d.fit_transform(delta_vecs)
        for k in range(2, 6):
            km = KMeans(n_clusters=k, n_init=5, random_state=42)
            lbl = km.fit_predict(dv_low)
            if len(np.unique(lbl)) < 2:
                continue
            sc  = silhouette_score(dv_low, lbl, sample_size=min(2000, len(dv_low)))
            if sc > best_sc_d:
                best_sc_d, best_k_d = sc, k
    else:
        dv_low = delta_vecs
        best_k_d = 1

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(delta_magnitudes, bins=40, color="steelblue", edgecolor="white")
    axes[0].set_title(f"[{label}] Distribution of ||delta_h|| (mutation size)")
    axes[0].set_xlabel("||h_{t+1} - h_t||"); axes[0].set_ylabel("Count")

    if delta_vecs.shape[1] >= 2:
        pca2d = PCA(n_components=2).fit_transform(delta_vecs)
        axes[1].scatter(pca2d[:, 0], pca2d[:, 1], alpha=0.2, s=3)
        axes[1].set_title(f"[{label}] PCA of delta_h vectors\n"
                          f"(k={best_k_d} clusters, sil={best_sc_d:.3f})")
        axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    else:
        axes[1].text(0.5, 0.5, "insufficient data", ha="center", va="center")

    if cross_agent_corrs:
        axes[2].hist(cross_agent_corrs, bins=30, color="darkorange", edgecolor="white")
        mean_c = np.mean(cross_agent_corrs)
        axes[2].axvline(mean_c, color="red", linestyle="--",
                        label=f"mean={mean_c:.3f}")
        axes[2].set_title(f"[{label}] Cross-agent delta correlation\n"
                          "(>0 = socially coupled mutations)")
        axes[2].set_xlabel("Pearson corr(delta_i, delta_j)")
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "no cross-agent data", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"q5_mutation_{label}.png"), dpi=120)
    plt.close()

    return {
        "mean_delta_magnitude": float(np.mean(delta_magnitudes)) if delta_magnitudes else 0.0,
        "delta_cluster_silhouette": best_sc_d,
        "mean_cross_agent_corr": float(np.mean(cross_agent_corrs)) if cross_agent_corrs else 0.0,
    }


# ---------------------------------------------------------------------------
# Side-by-side comparison plots
# ---------------------------------------------------------------------------

def plot_comparison(results: dict, metric_key: str, ylabel: str,
                    title: str, out_dir: str, filename: str):
    """Generic comparison plot across variants."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, res in results.items():
        vals = res.get(metric_key, [])
        if vals:
            ax.plot(range(len(vals)), vals, marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel("Probe index (training time →)")
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(all_results: dict):
    print("\n" + "=" * 60)
    print("MEME ANALYSIS SUMMARY")
    print("=" * 60)
    for label, res in all_results.items():
        print(f"\n--- {label} ---")
        q1 = res.get("q1", {})
        q2 = res.get("q2", {})
        q3 = res.get("q3", {})
        q4 = res.get("q4", {})
        q5 = res.get("q5", {})

        print(f"  Q1 Persistence:   k={q1.get('best_k','?')}  "
              f"silhouette={q1.get('silhouette', 0):.3f}  "
              f"per-agent entropy={q1.get('per_agent_entropy', {})}")
        print(f"  Q2 Transmission:  final sim={q2.get('mean_sim_over_time', [0])[-1]:.3f}  "
              f"m_bar↔h_sim corr={q2.get('m_h_correlation', 0):.3f}")
        print(f"  Q3 Selection:     best cluster mean_rew="
              f"{max(q3.get('cluster_mean_reward', {0: 0}).values()):.3f}  "
              f"final prevalence={q3.get('best_cluster_prevalence', [0])[-1]:.3f}")
        print(f"  Q4 Structure:     final between_var={q4.get('between_var', [0])[-1]:.4f}  "
              f"final within_var={q4.get('within_var', [0])[-1]:.4f}")
        print(f"  Q5 Mutation:      mean ||delta||={q5.get('mean_delta_magnitude', 0):.4f}  "
              f"delta_sil={q5.get('delta_cluster_silhouette', 0):.3f}  "
              f"cross-agent corr={q5.get('mean_cross_agent_corr', 0):.3f}")
    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Meme analysis on probe rollout data")
    parser.add_argument("--probe-dirs", nargs="+", required=True,
                        help="Directories containing probe_*.npz files")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Labels for each probe dir (same order)")
    parser.add_argument("--out-dir", default="meme_analysis_out",
                        help="Output directory for plots and summary")
    args = parser.parse_args()

    assert len(args.probe_dirs) == len(args.labels), \
        "--probe-dirs and --labels must have same length"

    os.makedirs(args.out_dir, exist_ok=True)

    all_results = {}
    for probe_dir, label in zip(args.probe_dirs, args.labels):
        print(f"\nLoading probes: {label} from {probe_dir}")
        probes = load_probes(probe_dir)
        print(f"  {len(probes)} probe checkpoints, "
              f"steps {int(probes[0]['training_step'])}..{int(probes[-1]['training_step'])}")

        res = {}
        print("  Q1: persistence...")
        res["q1"] = q1_persistence(probes, label, args.out_dir)
        print("  Q2: transmission...")
        res["q2"] = q2_transmission(probes, label, args.out_dir)
        print("  Q3: selection...")
        res["q3"] = q3_selection(probes, res["q1"], label, args.out_dir)
        print("  Q4: structure...")
        res["q4"] = q4_structure(probes, label, args.out_dir)
        print("  Q5: mutation...")
        res["q5"] = q5_mutation(probes, res["q1"], label, args.out_dir)
        all_results[label] = res

    # Cross-variant comparison plots
    plot_comparison(
        {l: r["q2"] for l, r in all_results.items()},
        "mean_sim_over_time", "Cosine similarity",
        "Q2: Pairwise h similarity over training (transmission signal)",
        args.out_dir, "compare_transmission.png"
    )
    plot_comparison(
        {l: r["q4"] for l, r in all_results.items()},
        "between_var", "Between-agent h variance",
        "Q4: Between-agent variance (specialisation signal)",
        args.out_dir, "compare_specialisation.png"
    )

    print_summary(all_results)


if __name__ == "__main__":
    main()
