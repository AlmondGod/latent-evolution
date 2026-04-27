#!/usr/bin/env python3
# RALE alignment analysis from per-step latent probes.
# Builds shared k-means clusters across all (method, seed) probes, then
# computes x, F, Q, R per cluster and reports weighted Spearman + isotonic
# mismatch A_iso, with shuffle null and episode-bootstrap CIs.

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RALE alignment analysis")
    parser.add_argument("--probe-root", type=Path, required=True,
                        help="Dir containing {method}/seed{seed}/probe.npz files.")
    parser.add_argument("--methods", type=str, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature", choices=["z", "h"], default="z",
                        help="Latent to cluster: z (memetic) or h (control).")
    parser.add_argument("--k-list", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--horizon", type=int, default=10,
                        help="Future-reward horizon for Y_i^t.")
    parser.add_argument("--target", choices=["future", "episode"], default="future",
                        help="Y_i^t target: 'future' = sum next H rewards; 'episode' = episode return.")
    parser.add_argument("--cluster-cap", type=int, default=5000,
                        help="Max rows per (method, seed) used for fitting shared centroids.")
    parser.add_argument("--cluster-scope", choices=["shared", "per_method", "per_seed"],
                        default="shared",
                        help="shared = one k-means over all (method, seed); "
                             "per_method = one per method (pooled across seeds); "
                             "per_seed = one per (method, seed). Affects cluster IDs and validity filter.")
    parser.add_argument("--bootstrap-iters", type=int, default=200)
    parser.add_argument("--shuffle-iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-cluster-frac", type=float, default=0.005,
                        help="Drop clusters whose pooled occupancy is below this fraction.")
    return parser.parse_args()


def load_probe(path: Path) -> dict:
    raw = np.load(path, allow_pickle=True)
    out = {k: raw[k] for k in raw.files if k != "meta"}
    out["meta"] = json.loads(str(raw["meta"]))
    return out


def collect_probes(
    probe_root: Path, methods: list[str], seeds: list[int]
) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for m in methods:
        for s in seeds:
            p = probe_root / m / f"seed{s}" / "probe.npz"
            if not p.exists():
                raise FileNotFoundError(p)
            out[(m, s)] = load_probe(p)
    return out


def select_feature(data: dict, feature: str) -> tuple[np.ndarray, np.ndarray]:
    # Returns (feat_t, feat_t1) for chosen feature. For h, feat_t1 is built by
    # shifting within (episode, agent).
    if feature == "z":
        return data["z_t"], data["z_t1"]
    h = data["h_t"]
    ep = data["episode_id"]
    aid = data["agent_id"]
    t = data["timestep"]
    n = h.shape[0]
    h1 = np.zeros_like(h)
    # Sort by (ep, aid, t) to make "next" lookup trivial.
    order = np.lexsort((t, aid, ep))
    h_sorted = h[order]
    ep_s = ep[order]
    aid_s = aid[order]
    t_s = t[order]
    h1_sorted = np.zeros_like(h_sorted)
    same_traj = (ep_s[1:] == ep_s[:-1]) & (aid_s[1:] == aid_s[:-1]) & (t_s[1:] == t_s[:-1] + 1)
    h1_sorted[:-1] = np.where(same_traj[:, None], h_sorted[1:], h_sorted[:-1])
    # When there's no valid next-step, leave equal to h (will be excluded via done mask anyway).
    # Map back to original order.
    inv = np.empty_like(order)
    inv[order] = np.arange(n)
    h1[:] = h1_sorted[inv]
    return h, h1


def compute_future_reward(
    reward_t: np.ndarray,
    episode_id: np.ndarray,
    timestep: np.ndarray,
    agent_id: np.ndarray,
    horizon: int,
) -> np.ndarray:
    # For each row, sum reward over the next H timesteps within the same (episode, agent).
    # Rewards are shared across agents but we still compute per-agent for the (episode, agent, t) key.
    n = reward_t.shape[0]
    Y = np.zeros(n, dtype=np.float32)
    order = np.lexsort((timestep, agent_id, episode_id))
    r_sorted = reward_t[order]
    ep_s = episode_id[order]
    aid_s = agent_id[order]
    t_s = timestep[order]
    # Walk per trajectory contiguous block.
    i = 0
    while i < n:
        j = i
        while j < n and ep_s[j] == ep_s[i] and aid_s[j] == aid_s[i]:
            j += 1
        traj_r = r_sorted[i:j]
        L = traj_r.shape[0]
        traj_y = np.zeros(L, dtype=np.float32)
        for k in range(L):
            end = min(k + horizon, L)
            traj_y[k] = traj_r[k:end].sum()
        Y[order[i:j]] = traj_y
        i = j
    return Y


def _fit_kmeans_on(X: np.ndarray, k: int, rng: np.random.Generator) -> KMeans:
    km = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(0, 2**31 - 1)))
    km.fit(X.astype(np.float64))
    return km


def _subsample(feat: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    n = feat.shape[0]
    if n <= cap:
        return feat
    idx = rng.choice(n, cap, replace=False)
    return feat[idx]


def fit_kmeans_dicts(
    probes: dict[tuple[str, int], dict],
    feature: str,
    k: int,
    cluster_cap: int,
    rng: np.random.Generator,
    scope: str,
) -> dict[tuple[str, int], KMeans]:
    # Returns one KMeans per (method, seed). For shared/per_method scopes,
    # multiple keys share the same fitted object (so cluster IDs align).
    out: dict[tuple[str, int], KMeans] = {}
    if scope == "shared":
        pooled = [_subsample(select_feature(d, feature)[0], cluster_cap, rng)
                  for d in probes.values()]
        km = _fit_kmeans_on(np.concatenate(pooled, axis=0), k, rng)
        for key in probes.keys():
            out[key] = km
        return out
    if scope == "per_method":
        by_method: dict[str, list[np.ndarray]] = defaultdict(list)
        for (m, _), d in probes.items():
            by_method[m].append(_subsample(select_feature(d, feature)[0], cluster_cap, rng))
        method_km = {m: _fit_kmeans_on(np.concatenate(xs, axis=0), k, rng)
                     for m, xs in by_method.items()}
        for (m, s) in probes.keys():
            out[(m, s)] = method_km[m]
        return out
    # per_seed
    for key, d in probes.items():
        feat_t, _ = select_feature(d, feature)
        out[key] = _fit_kmeans_on(_subsample(feat_t, cluster_cap, rng), k, rng)
    return out


def assign_clusters(km: KMeans, X: np.ndarray) -> np.ndarray:
    return km.predict(X.astype(np.float64))


def weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    w = w / max(w.sum(), 1e-12)
    mx = float((w * x).sum())
    my = float((w * y).sum())
    cov = float((w * (x - mx) * (y - my)).sum())
    vx = float((w * (x - mx) ** 2).sum())
    vy = float((w * (y - my) ** 2).sum())
    denom = float(np.sqrt(max(vx, 0.0) * max(vy, 0.0)))
    if denom < 1e-12:
        return 0.0
    return cov / denom


def weighted_spearman(F: np.ndarray, R: np.ndarray, w: np.ndarray) -> float:
    return weighted_pearson(rankdata(F), rankdata(R), w)


def isotonic_mismatch(F: np.ndarray, R: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    iso = IsotonicRegression(increasing=True)
    iso.fit(F, R, sample_weight=w)
    R_hat = iso.predict(F)
    A = float((w * (R - R_hat) ** 2).sum())
    return A, R_hat


def compute_metrics_for_run(
    cluster_t: np.ndarray,
    cluster_t1: np.ndarray,
    Y: np.ndarray,
    done_t: np.ndarray,
    K: int,
    occ_keep: np.ndarray,
) -> dict:
    # x_k from cluster_t.
    counts = np.bincount(cluster_t, minlength=K).astype(np.float64)
    x = counts / max(counts.sum(), 1.0)

    # F_k = E[Y | cluster_t = k].
    F = np.zeros(K, dtype=np.float64)
    for k in range(K):
        mask = cluster_t == k
        if mask.any():
            F[k] = float(Y[mask].mean())

    # Q: only over non-terminal rows.
    nonterm = done_t == 0
    Q = np.zeros((K, K), dtype=np.float64)
    if nonterm.any():
        ct = cluster_t[nonterm]
        ct1 = cluster_t1[nonterm]
        for k in range(K):
            mk = ct == k
            row_total = float(mk.sum())
            if row_total > 0.0:
                bins = np.bincount(ct1[mk], minlength=K).astype(np.float64)
                Q[k] = bins / row_total

    # R_j = (1 / x_j) * sum_k x_k F_k Q_kj.
    weighted = (x[:, None] * F[:, None]) * Q
    num = weighted.sum(axis=0)
    R = np.zeros(K, dtype=np.float64)
    for j in range(K):
        if x[j] > 0.0:
            R[j] = num[j] / x[j]

    # Restrict to occupied + kept clusters.
    keep = occ_keep & (x > 0.0)
    Fk = F[keep]
    Rk = R[keep]
    xk = x[keep]
    if Fk.size < 3:
        return {"x": x, "F": F, "Q": Q, "R": R, "keep": keep,
                "weighted_spearman": float("nan"), "A_iso": float("nan"),
                "n_clusters_kept": int(keep.sum())}

    ws = weighted_spearman(Fk, Rk, xk)
    A_iso, _ = isotonic_mismatch(Fk, Rk, xk)
    return {
        "x": x, "F": F, "Q": Q, "R": R, "keep": keep,
        "weighted_spearman": ws,
        "A_iso": A_iso,
        "n_clusters_kept": int(keep.sum()),
    }


def shuffle_null(
    cluster_t: np.ndarray,
    cluster_t1: np.ndarray,
    Y: np.ndarray,
    done_t: np.ndarray,
    K: int,
    occ_keep: np.ndarray,
    iters: int,
    rng: np.random.Generator,
) -> dict:
    # Null: break the link between cluster identity and reward by shuffling Y.
    spearmans = np.zeros(iters, dtype=np.float64)
    A_isos = np.zeros(iters, dtype=np.float64)
    n = Y.shape[0]
    for i in range(iters):
        Y_shuf = Y[rng.permutation(n)]
        m = compute_metrics_for_run(
            cluster_t=cluster_t, cluster_t1=cluster_t1, Y=Y_shuf,
            done_t=done_t, K=K, occ_keep=occ_keep,
        )
        spearmans[i] = m["weighted_spearman"]
        A_isos[i] = m["A_iso"]
    return {
        "weighted_spearman_mean": float(np.nanmean(spearmans)),
        "weighted_spearman_p95": float(np.nanpercentile(spearmans, 95)),
        "A_iso_mean": float(np.nanmean(A_isos)),
        "A_iso_p05": float(np.nanpercentile(A_isos, 5)),
    }


def bootstrap_ci(
    episode_id: np.ndarray,
    cluster_t: np.ndarray,
    cluster_t1: np.ndarray,
    Y: np.ndarray,
    done_t: np.ndarray,
    K: int,
    occ_keep: np.ndarray,
    iters: int,
    rng: np.random.Generator,
) -> dict:
    unique_eps = np.unique(episode_id)
    spearmans = np.zeros(iters, dtype=np.float64)
    A_isos = np.zeros(iters, dtype=np.float64)
    for i in range(iters):
        sampled = rng.choice(unique_eps, size=unique_eps.size, replace=True)
        # Build mask via concatenating per-episode index sets.
        masks = [episode_id == e for e in sampled]
        idx = np.concatenate([np.where(mask)[0] for mask in masks])
        m = compute_metrics_for_run(
            cluster_t=cluster_t[idx],
            cluster_t1=cluster_t1[idx],
            Y=Y[idx],
            done_t=done_t[idx],
            K=K,
            occ_keep=occ_keep,
        )
        spearmans[i] = m["weighted_spearman"]
        A_isos[i] = m["A_iso"]
    return {
        "weighted_spearman_lo": float(np.nanpercentile(spearmans, 2.5)),
        "weighted_spearman_hi": float(np.nanpercentile(spearmans, 97.5)),
        "A_iso_lo": float(np.nanpercentile(A_isos, 2.5)),
        "A_iso_hi": float(np.nanpercentile(A_isos, 97.5)),
    }


def episode_prevalence_corr(
    cluster_t: np.ndarray,
    episode_id: np.ndarray,
    episode_return: np.ndarray,
    K: int,
) -> dict:
    eps = np.unique(episode_id)
    G = np.array([float(episode_return[episode_id == e][0]) for e in eps], dtype=np.float64)
    out = {}
    for k in range(K):
        x_e = np.array([(cluster_t[episode_id == e] == k).mean() for e in eps], dtype=np.float64)
        if x_e.std() < 1e-12 or G.std() < 1e-12:
            pearson = float("nan")
            spearman = float("nan")
        else:
            pearson = float(np.corrcoef(x_e, G)[0, 1])
            spearman = weighted_pearson(rankdata(x_e), rankdata(G), np.ones_like(x_e))
        out[int(k)] = {"pearson": pearson, "spearman": spearman}
    return out


def run_pipeline(
    probes: dict[tuple[str, int], dict],
    feature: str,
    K: int,
    horizon: int,
    target: str,
    cluster_cap: int,
    min_cluster_frac: float,
    bootstrap_iters: int,
    shuffle_iters: int,
    rng: np.random.Generator,
    cluster_scope: str = "shared",
) -> dict:
    kms = fit_kmeans_dicts(
        probes=probes, feature=feature, k=K,
        cluster_cap=cluster_cap, rng=rng, scope=cluster_scope,
    )

    # Per-(method, seed) occupancy: a cluster counts as "kept" for that probe
    # if its own occupancy is above the floor. With shared scope this collapses
    # back to a single shared mask (occupancy is computed on the same dict).
    occ_keep_by_key: dict[tuple[str, int], np.ndarray] = {}
    occ_frac_by_key: dict[tuple[str, int], np.ndarray] = {}
    if cluster_scope == "shared":
        pooled_counts = np.zeros(K, dtype=np.float64)
        for key, data in probes.items():
            feat_t, _ = select_feature(data, feature)
            labels = assign_clusters(kms[key], feat_t)
            pooled_counts += np.bincount(labels, minlength=K).astype(np.float64)
        pooled_total = max(pooled_counts.sum(), 1.0)
        shared_frac = pooled_counts / pooled_total
        shared_keep = shared_frac >= min_cluster_frac
        for key in probes.keys():
            occ_frac_by_key[key] = shared_frac
            occ_keep_by_key[key] = shared_keep
    else:
        for key, data in probes.items():
            feat_t, _ = select_feature(data, feature)
            labels = assign_clusters(kms[key], feat_t)
            counts = np.bincount(labels, minlength=K).astype(np.float64)
            frac = counts / max(counts.sum(), 1.0)
            occ_frac_by_key[key] = frac
            occ_keep_by_key[key] = frac >= min_cluster_frac

    rows = []
    cluster_assignments = {}
    for (method, seed), data in probes.items():
        feat_t, feat_t1 = select_feature(data, feature)
        km = kms[(method, seed)]
        c_t = assign_clusters(km, feat_t)
        c_t1 = assign_clusters(km, feat_t1)
        occ_keep = occ_keep_by_key[(method, seed)]
        cluster_assignments[(method, seed)] = (c_t, c_t1)

        if target == "future":
            Y = compute_future_reward(
                reward_t=data["reward_t"],
                episode_id=data["episode_id"],
                timestep=data["timestep"],
                agent_id=data["agent_id"],
                horizon=horizon,
            )
        else:
            Y = data["episode_return"].astype(np.float32)

        m = compute_metrics_for_run(
            cluster_t=c_t, cluster_t1=c_t1, Y=Y,
            done_t=data["done_t"].astype(np.int8),
            K=K, occ_keep=occ_keep,
        )
        null = shuffle_null(
            cluster_t=c_t, cluster_t1=c_t1, Y=Y,
            done_t=data["done_t"].astype(np.int8),
            K=K, occ_keep=occ_keep,
            iters=shuffle_iters, rng=rng,
        )
        boot = bootstrap_ci(
            episode_id=data["episode_id"], cluster_t=c_t, cluster_t1=c_t1, Y=Y,
            done_t=data["done_t"].astype(np.int8),
            K=K, occ_keep=occ_keep,
            iters=bootstrap_iters, rng=rng,
        )
        ep_corr = episode_prevalence_corr(
            cluster_t=c_t, episode_id=data["episode_id"],
            episode_return=data["episode_return"], K=K,
        )

        rows.append({
            "method": method, "seed": int(seed), "K": int(K),
            "weighted_spearman": m["weighted_spearman"],
            "A_iso": m["A_iso"],
            "n_clusters_kept": m["n_clusters_kept"],
            "x_k": m["x"].tolist(),
            "F_k": m["F"].tolist(),
            "R_k": m["R"].tolist(),
            "Q": m["Q"].tolist(),
            "keep_mask": m["keep"].astype(int).tolist(),
            "shuffle_null": null,
            "bootstrap": boot,
            "episode_corr": ep_corr,
        })

    return {
        "K": int(K),
        "feature": feature,
        "cluster_scope": cluster_scope,
        "centroids_by_key": {f"{m}__seed{s}": kms[(m, s)].cluster_centers_.tolist()
                              for (m, s) in probes.keys()},
        "occupancy_by_key": {f"{m}__seed{s}": occ_frac_by_key[(m, s)].tolist()
                              for (m, s) in probes.keys()},
        "occ_keep_by_key": {f"{m}__seed{s}": occ_keep_by_key[(m, s)].astype(int).tolist()
                             for (m, s) in probes.keys()},
        "rows": rows,
        "cluster_assignments": {f"{m}__seed{s}": (c_t.tolist(), c_t1.tolist())
                                 for (m, s), (c_t, c_t1) in cluster_assignments.items()},
    }


def aggregate_method(rows: list[dict], methods: list[str]) -> dict:
    out: dict[str, dict] = {}
    for m in methods:
        ws = np.array([r["weighted_spearman"] for r in rows if r["method"] == m], dtype=float)
        ai = np.array([r["A_iso"] for r in rows if r["method"] == m], dtype=float)
        n_total = int(ws.size)
        n_valid = int(np.sum(~np.isnan(ws)))
        ws_clean = ws[~np.isnan(ws)]
        ai_clean = ai[~np.isnan(ai)]
        shuf_ws = np.array([r["shuffle_null"]["weighted_spearman_mean"]
                            for r in rows if r["method"] == m], dtype=float)
        shuf_ai = np.array([r["shuffle_null"]["A_iso_mean"]
                            for r in rows if r["method"] == m], dtype=float)
        ncl = [int(r["n_clusters_kept"]) for r in rows if r["method"] == m]
        out[m] = {
            "n_seeds_total": n_total,
            "n_seeds_valid": n_valid,
            "n_clusters_kept_per_seed": ncl,
            "weighted_spearman_mean": float(ws_clean.mean()) if ws_clean.size else float("nan"),
            "weighted_spearman_var": float(ws_clean.var(ddof=1)) if ws_clean.size > 1 else float("nan"),
            "weighted_spearman_std": float(ws_clean.std(ddof=1)) if ws_clean.size > 1 else float("nan"),
            "weighted_spearman_sem": float(ws_clean.std(ddof=1) / np.sqrt(ws_clean.size)) if ws_clean.size > 1 else float("nan"),
            "A_iso_mean": float(ai_clean.mean()) if ai_clean.size else float("nan"),
            "A_iso_var": float(ai_clean.var(ddof=1)) if ai_clean.size > 1 else float("nan"),
            "A_iso_std": float(ai_clean.std(ddof=1)) if ai_clean.size > 1 else float("nan"),
            "A_iso_sem": float(ai_clean.std(ddof=1) / np.sqrt(ai_clean.size)) if ai_clean.size > 1 else float("nan"),
            "shuffle_spearman_mean": float(np.nanmean(shuf_ws)) if np.any(~np.isnan(shuf_ws)) else float("nan"),
            "shuffle_spearman_std": float(np.nanstd(shuf_ws, ddof=1)) if np.sum(~np.isnan(shuf_ws)) > 1 else float("nan"),
            "shuffle_A_iso_mean": float(np.nanmean(shuf_ai)) if np.any(~np.isnan(shuf_ai)) else float("nan"),
        }
    return out


def plot_F_vs_R(rows: list[dict], methods: list[str], K: int, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.5), sharey=False)
    if len(methods) == 1:
        axes = [axes]
    for ax, m in zip(axes, methods):
        method_rows = [r for r in rows if r["method"] == m]
        for r in method_rows:
            keep = np.asarray(r["keep_mask"]).astype(bool)
            x = np.asarray(r["x_k"])[keep]
            F = np.asarray(r["F_k"])[keep]
            R = np.asarray(r["R_k"])[keep]
            sizes = 50 + 800 * (x / max(x.max(), 1e-9))
            ax.scatter(F, R, s=sizes, alpha=0.5, label=f"seed {r['seed']}")
        ax.set_title(f"{m} (K={K}): F vs R")
        ax.set_xlabel("F_k (host fitness)")
        ax.set_ylabel("R_k (replicative fitness)")
        ax.grid(alpha=0.3)
        ax.legend()
        # Diagonal reference
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", alpha=0.4, lw=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_Q_heatmap(rows: list[dict], method: str, K: int, out_path: Path) -> None:
    method_rows = [r for r in rows if r["method"] == method]
    n = len(method_rows)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, method_rows):
        Q = np.asarray(r["Q"])
        im = ax.imshow(Q, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(f"{method} seed{r['seed']} Q (K={K})")
        ax.set_xlabel("next cluster j")
        ax.set_ylabel("from cluster k")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_episode_corr(probes: dict[tuple[str, int], dict], rows: list[dict], methods: list[str], K: int, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4))
    if len(methods) == 1:
        axes = [axes]
    for ax, m in zip(axes, methods):
        method_rows = [r for r in rows if r["method"] == m]
        for r in method_rows:
            keep = np.asarray(r["keep_mask"]).astype(bool)
            corrs = [r["episode_corr"][str(k)]["spearman"] if str(k) in r["episode_corr"] else r["episode_corr"][k]["spearman"]
                     for k in range(K)]
            corrs_kept = [c for c, kk in zip(corrs, keep) if kk]
            ax.plot(np.arange(len(corrs_kept)), corrs_kept, "o", alpha=0.6, label=f"seed {r['seed']}")
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_title(f"{m} (K={K}): episode prevalence vs return Spearman")
        ax.set_xlabel("kept cluster index")
        ax.set_ylabel("Spearman corr per cluster")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_robustness_K(per_K_aggs: dict[int, dict], methods: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    Ks = sorted(per_K_aggs.keys())
    for m in methods:
        ws_means = [per_K_aggs[k]["aggregate"][m]["weighted_spearman_mean"] for k in Ks]
        ws_sems = [per_K_aggs[k]["aggregate"][m]["weighted_spearman_sem"] for k in Ks]
        ai_means = [per_K_aggs[k]["aggregate"][m]["A_iso_mean"] for k in Ks]
        ai_sems = [per_K_aggs[k]["aggregate"][m]["A_iso_sem"] for k in Ks]
        ws_sems = [0.0 if (s is None or (isinstance(s, float) and np.isnan(s))) else s for s in ws_sems]
        ai_sems = [0.0 if (s is None or (isinstance(s, float) and np.isnan(s))) else s for s in ai_sems]
        axes[0].errorbar(Ks, ws_means, yerr=ws_sems, marker="o", label=m, capsize=3)
        axes[1].errorbar(Ks, ai_means, yerr=ai_sems, marker="o", label=m, capsize=3)
    axes[0].axhline(0.0, color="k", lw=0.5)
    axes[0].set_xlabel("K (number of clusters)")
    axes[0].set_ylabel("weighted Spearman (F, R)")
    axes[0].set_title("Alignment vs K")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel("K (number of clusters)")
    axes[1].set_ylabel("A_iso (isotonic mismatch)")
    axes[1].set_title("Mismatch vs K (lower = more aligned)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_summary(args: argparse.Namespace, per_K: dict[int, dict], methods: list[str], out_path: Path) -> None:
    Ks = sorted(per_K.keys())
    lines = []
    lines.append(f"# RALE alignment summary ({args.feature})")
    lines.append("")
    lines.append(f"- Probe root: `{args.probe_root}`")
    lines.append(f"- Methods: `{methods}`")
    lines.append(f"- Seeds: `{args.seeds}`")
    lines.append(f"- K values: `{Ks}`")
    lines.append(f"- Y target: `{args.target}` (H={args.horizon})")
    lines.append(f"- Bootstrap iters: {args.bootstrap_iters}; Shuffle iters: {args.shuffle_iters}")
    lines.append("")
    lines.append("## Aggregate (mean ± SEM across seeds with >=3 occupied clusters)")
    lines.append("")
    lines.append("| K | method | n_valid/n_total | clusters/seed | weighted Spearman | A_iso | shuffle Spearman | shuffle A_iso |")
    lines.append("|--:|:------:|:---------------:|:-------------:|:------------------:|:------:|:------------------:|:---------------:|")
    for K in Ks:
        for m in methods:
            agg = per_K[K]["aggregate"][m]
            ncl_str = ",".join(str(x) for x in agg["n_clusters_kept_per_seed"])
            lines.append(
                f"| {K} | {m} | {agg['n_seeds_valid']}/{agg['n_seeds_total']} | {ncl_str} | "
                f"{agg['weighted_spearman_mean']:+.3f} ± {agg['weighted_spearman_sem']:.3f} | "
                f"{agg['A_iso_mean']:.3f} ± {agg['A_iso_sem']:.3f} | "
                f"{agg['shuffle_spearman_mean']:+.3f} | {agg['shuffle_A_iso_mean']:.3f} |"
            )
    lines.append("")
    lines.append(
        "Notes: `n_valid` = seeds where ≥3 clusters were occupied at >=`min_cluster_frac` "
        "(weighted Spearman / A_iso are undefined when fewer than 3 clusters are kept). "
        "`clusters/seed` shows the per-seed count of kept clusters. "
        "If all seeds collapse to <3 clusters, the latent space is effectively a point and the "
        "method is not using the meme channel meaningfully."
    )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    probes = collect_probes(args.probe_root, args.methods, args.seeds)
    print(json.dumps({"loaded_probes": [f"{m}/seed{s}" for (m, s) in probes.keys()]}), flush=True)

    per_K: dict[int, dict] = {}
    for K in args.k_list:
        result = run_pipeline(
            probes=probes, feature=args.feature, K=K,
            horizon=args.horizon, target=args.target,
            cluster_cap=args.cluster_cap,
            min_cluster_frac=args.min_cluster_frac,
            bootstrap_iters=args.bootstrap_iters,
            shuffle_iters=args.shuffle_iters,
            rng=rng,
            cluster_scope=args.cluster_scope,
        )
        agg = aggregate_method(result["rows"], args.methods)
        per_K[K] = {**result, "aggregate": agg}
        print(json.dumps({"K": K, "aggregate": agg}), flush=True)

    # Save tables.
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    tables_dir.mkdir(exist_ok=True, parents=True)
    figures_dir.mkdir(exist_ok=True, parents=True)

    seed_table_lines = ["method,seed,K,weighted_spearman,A_iso,n_clusters_kept,boot_ws_lo,boot_ws_hi,boot_aiso_lo,boot_aiso_hi"]
    for K, payload in per_K.items():
        for r in payload["rows"]:
            seed_table_lines.append(
                f"{r['method']},{r['seed']},{K},"
                f"{r['weighted_spearman']:.6f},{r['A_iso']:.6f},{r['n_clusters_kept']},"
                f"{r['bootstrap']['weighted_spearman_lo']:.6f},{r['bootstrap']['weighted_spearman_hi']:.6f},"
                f"{r['bootstrap']['A_iso_lo']:.6f},{r['bootstrap']['A_iso_hi']:.6f}"
            )
    (tables_dir / "seed_metrics.csv").write_text("\n".join(seed_table_lines) + "\n")

    method_table_lines = ["K,method,n_seeds_valid,n_seeds_total,clusters_per_seed,"
                          "ws_mean,ws_var,ws_std,ws_sem,"
                          "aiso_mean,aiso_var,aiso_std,aiso_sem,"
                          "shuffle_ws,shuffle_ws_std,shuffle_aiso"]
    for K, payload in per_K.items():
        for m in args.methods:
            a = payload["aggregate"][m]
            ncl = ";".join(str(x) for x in a["n_clusters_kept_per_seed"])
            method_table_lines.append(
                f"{K},{m},{a['n_seeds_valid']},{a['n_seeds_total']},{ncl},"
                f"{a['weighted_spearman_mean']:.6f},{a['weighted_spearman_var']:.6f},"
                f"{a['weighted_spearman_std']:.6f},{a['weighted_spearman_sem']:.6f},"
                f"{a['A_iso_mean']:.6f},{a['A_iso_var']:.6f},"
                f"{a['A_iso_std']:.6f},{a['A_iso_sem']:.6f},"
                f"{a['shuffle_spearman_mean']:.6f},{a['shuffle_spearman_std']:.6f},"
                f"{a['shuffle_A_iso_mean']:.6f}"
            )
    (tables_dir / "method_aggregate.csv").write_text("\n".join(method_table_lines) + "\n")

    # Save full per-K detail JSON without bulky cluster_assignments arrays.
    full_payload = {}
    for K, payload in per_K.items():
        full_payload[str(K)] = {kk: vv for kk, vv in payload.items() if kk != "cluster_assignments"}
    (args.output_dir / "rale_full.json").write_text(json.dumps(full_payload, indent=2))

    # Figures.
    for K, payload in per_K.items():
        plot_F_vs_R(payload["rows"], args.methods, K, figures_dir / f"F_vs_R_K{K}.png")
        for m in args.methods:
            plot_Q_heatmap(payload["rows"], m, K, figures_dir / f"Q_heatmap_{m}_K{K}.png")
        plot_episode_corr(probes, payload["rows"], args.methods, K, figures_dir / f"episode_corr_K{K}.png")

    plot_robustness_K(per_K, args.methods, figures_dir / "robustness_K.png")

    write_summary(args, per_K, args.methods, args.output_dir / "summary.md")
    print(json.dumps({"saved_summary": str(args.output_dir / "summary.md")}), flush=True)


if __name__ == "__main__":
    main()
