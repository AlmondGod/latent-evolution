"""
analyze_meme_diversity.py
--------------------------
Measures the DIVERSITY of agent memory states (GRU hidden states h)
across conditions to test the memetic diversity hypothesis:

  Hypothesis: Under partial observability, agents develop more diverse
  "memes" (distinct memory states) than under full observability.
  Information asymmetry forces role specialization.

Metrics:
  1. Pairwise cosine similarity between agent hidden states
     (lower = more diverse memes)
  2. Effective dimensionality of memory space (entropy of PCA eigenvalues)
  3. Memory utilization rate (||h||_mean vs ||h||_zero_init)
  4. Memory update rate (||h_t - h_{t-1}||_mean across episode)

Usage:
    python3.9 -m new.memetic_foundation.scripts.analyze_meme_diversity

Produces:
  plots/meme_diversity.png
  plots/meme_diversity_report.txt
"""

from __future__ import annotations

import os
import glob
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))))

from new.memetic_foundation.models.agent_network import MemeticFoundationAC
from new.memetic_foundation.training.mpe_wrapper import MPEWrapper


BASE = "checkpoints"
N_EVAL_EPISODES = 20
N_AGENTS = 3


def load_model_from_checkpoint(ckpt_path: str, use_memory: bool = True,
                                 use_gate: bool = True) -> MemeticFoundationAC | None:
    """Load a trained model from checkpoint."""
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        env_info = state.get("env_info", {})
        policy_sd = state["policy_state_dict"]

        model = MemeticFoundationAC(
            obs_dim=env_info.get("obs_shape", 16),
            state_dim=env_info.get("state_shape", 62),
            n_actions=env_info.get("n_actions", 5),
            n_agents=env_info.get("n_agents", N_AGENTS),
            use_memory=use_memory,
            use_comm=False,  # memory_only for clean diversity measurement
            use_gate=use_gate,
        )
        model.load_state_dict(policy_sd, strict=False)
        model.eval()
        return model
    except Exception as e:
        print(f"  [WARN] Could not load {ckpt_path}: {e}")
        return None


def collect_memory_trajectories(
    model: MemeticFoundationAC,
    env: MPEWrapper,
    n_episodes: int = N_EVAL_EPISODES,
    obs_radius: float | None = None,
) -> dict:
    """Roll out the model and collect GRU hidden states per agent per step."""
    all_h_per_agent = [[] for _ in range(N_AGENTS)]  # h[t] per agent
    step_rewards = []

    for ep in range(n_episodes):
        env.reset()
        model.reset_memory()
        ep_reward = 0.0

        for step in range(100):  # max_cycles
            obs_list = env.get_obs()
            obs_tensor = torch.tensor(
                np.stack(obs_list), dtype=torch.float32
            )

            with torch.no_grad():
                result = model.forward_step(obs_tensor, deterministic=True)
                actions = result["actions"].numpy().tolist()
                h = result.get("h")

                if h is not None:
                    for agent_idx in range(N_AGENTS):
                        all_h_per_agent[agent_idx].append(
                            h[agent_idx].numpy()
                        )

            reward, done, _ = env.step(actions)
            ep_reward += reward
            if done:
                break

        step_rewards.append(ep_reward)

    return {
        "h_per_agent": [np.array(h) for h in all_h_per_agent],  # (T, mem_dim)
        "mean_reward": float(np.mean(step_rewards)),
    }


def compute_diversity_metrics(h_per_agent: list[np.ndarray]) -> dict:
    """Compute pairwise cosine similarity and other diversity metrics."""
    N = len(h_per_agent)
    if N == 0 or len(h_per_agent[0]) == 0:
        return {}

    T = min(len(h) for h in h_per_agent)
    if T == 0:
        return {}

    # Trim to common length
    h_matrix = np.stack([h[:T] for h in h_per_agent])  # (N, T, mem_dim)
    mem_dim = h_matrix.shape[-1]

    # --- Pairwise cosine similarity (lower = more diverse) ---
    cosine_sims = []
    for i in range(N):
        for j in range(i + 1, N):
            hi = h_matrix[i]  # (T, mem_dim)
            hj = h_matrix[j]
            norms_i = np.linalg.norm(hi, axis=-1, keepdims=True) + 1e-8
            norms_j = np.linalg.norm(hj, axis=-1, keepdims=True) + 1e-8
            sim = np.sum((hi / norms_i) * (hj / norms_j), axis=-1)  # (T,)
            cosine_sims.append(float(np.mean(sim)))
    mean_cosine_sim = float(np.mean(cosine_sims)) if cosine_sims else float("nan")

    # --- Memory utilization: ||h|| / sqrt(mem_dim) ---
    all_norms = np.linalg.norm(h_matrix, axis=-1)  # (N, T)
    mean_norm = float(np.mean(all_norms))
    norm_ratio = mean_norm / np.sqrt(mem_dim)

    # --- Memory update rate: mean ||h_t - h_{t-1}|| ---
    deltas = np.diff(h_matrix, axis=1)  # (N, T-1, mem_dim)
    if deltas.size > 0:
        delta_norms = np.linalg.norm(deltas, axis=-1)  # (N, T-1)
        mean_delta = float(np.mean(delta_norms))
    else:
        mean_delta = 0.0

    # --- Effective dimensionality via PCA on joint h ---
    h_flat = h_matrix.reshape(-1, mem_dim)  # (N*T, mem_dim)
    cov = np.cov(h_flat.T)  # (mem_dim, mem_dim)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 0]
    if len(eigvals) > 0:
        p = eigvals / eigvals.sum()
        entropy = -np.sum(p * np.log(p + 1e-12))
        effective_dim = float(np.exp(entropy))
    else:
        effective_dim = 0.0

    # --- Inter-agent variance (how different are agents' mean states?) ---
    mean_per_agent = h_matrix.mean(axis=1)  # (N, mem_dim)
    inter_agent_std = float(np.std(mean_per_agent, axis=0).mean())

    return {
        "mean_cosine_similarity": mean_cosine_sim,  # lower = more diverse
        "pairwise_sims": cosine_sims,
        "mean_memory_norm": mean_norm,
        "norm_utilization": norm_ratio,
        "mean_memory_delta": mean_delta,   # higher = more active memory
        "effective_dim": effective_dim,    # higher = uses more dimensions
        "inter_agent_std": inter_agent_std,  # higher = agents more different
    }


def run_analysis():
    """Main analysis: compare memory diversity across conditions."""
    results = {}

    # ---- Full observability: memory_only, 8 seeds ----
    print("Loading full-obs memory_only checkpoints...")
    full_obs_metrics = []
    env_full = MPEWrapper("simple_tag_v2", num_adversaries=N_AGENTS,
                          obs_radius=None)
    for seed in range(1, 9):
        ckpt_dir = glob.glob(
            f"{BASE}/mpe_tag_gru/memory_only_seed{seed}/**/*latest.pt",
            recursive=True
        )
        if not ckpt_dir:
            continue
        model = load_model_from_checkpoint(ckpt_dir[-1], use_memory=True)
        if model is None:
            continue
        traj = collect_memory_trajectories(model, env_full, n_episodes=10)
        if traj["h_per_agent"][0].size > 0:
            metrics = compute_diversity_metrics(traj["h_per_agent"])
            metrics["reward"] = traj["mean_reward"]
            full_obs_metrics.append(metrics)
            print(f"  Seed {seed}: cosine_sim={metrics['mean_cosine_similarity']:.3f} "
                  f"eff_dim={metrics['effective_dim']:.1f} "
                  f"reward={metrics['reward']:.0f}")
    env_full.close()
    results["full_obs"] = full_obs_metrics

    # ---- Partial observability: memory_only, 8 seeds ----
    print("Loading partial-obs memory_only checkpoints...")
    partial_obs_metrics = []
    env_part = MPEWrapper("simple_tag_v2", num_adversaries=N_AGENTS,
                          obs_radius=0.5)
    for seed in range(1, 9):
        ckpt_dir = glob.glob(
            f"{BASE}/mpe_tag_partial_obs/memory_only_seed{seed}/**/*latest.pt",
            recursive=True
        )
        if not ckpt_dir:
            continue
        model = load_model_from_checkpoint(ckpt_dir[-1], use_memory=True)
        if model is None:
            continue
        traj = collect_memory_trajectories(model, env_part, n_episodes=10,
                                           obs_radius=0.5)
        if traj["h_per_agent"][0].size > 0:
            metrics = compute_diversity_metrics(traj["h_per_agent"])
            metrics["reward"] = traj["mean_reward"]
            partial_obs_metrics.append(metrics)
            print(f"  Seed {seed}: cosine_sim={metrics['mean_cosine_similarity']:.3f} "
                  f"eff_dim={metrics['effective_dim']:.1f} "
                  f"reward={metrics['reward']:.0f}")
    env_part.close()
    results["partial_obs"] = partial_obs_metrics

    return results


def print_diversity_report(results: dict):
    W = 68
    print("\n" + "=" * W)
    print("MEME DIVERSITY ANALYSIS")
    print("Comparing agent memory state diversity under full vs partial obs")
    print("=" * W)

    metrics_to_show = [
        ("mean_cosine_similarity", "Cosine sim (↓ = more diverse)"),
        ("effective_dim",          "Effective mem dimensions (↑ = richer)"),
        ("mean_memory_norm",       "Memory norm (↑ = more active)"),
        ("mean_memory_delta",      "Memory update rate (↑ = more dynamic)"),
        ("inter_agent_std",        "Inter-agent std (↑ = more specialized)"),
        ("reward",                 "Mean episode reward"),
    ]

    for key, data in results.items():
        if not data:
            continue
        print(f"\n{'─' * W}")
        print(f"{'Full Observability' if key == 'full_obs' else 'Partial Obs (r=0.5)':^{W}}")
        print(f"  n = {len(data)} seeds")
        print()
        for metric, label in metrics_to_show:
            vals = [d.get(metric, float("nan")) for d in data
                    if metric in d and not np.isnan(d[metric])]
            if vals:
                print(f"  {label:42s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # ---- Hypothesis test ----
    fo = results.get("full_obs", [])
    po = results.get("partial_obs", [])
    if fo and po:
        print("\n" + "─" * W)
        print("HYPOTHESIS TEST: Partial obs → more diverse memes?")
        print()
        for metric, label in metrics_to_show[:4]:
            fo_vals = [d[metric] for d in fo if metric in d]
            po_vals = [d[metric] for d in po if metric in d]
            if fo_vals and po_vals:
                fo_m = np.mean(fo_vals)
                po_m = np.mean(po_vals)
                direction = "↑" if po_m > fo_m else "↓"
                change = (po_m - fo_m) / (abs(fo_m) + 1e-8) * 100
                print(f"  {label[:40]:40s}: "
                      f"full={fo_m:.3f}  partial={po_m:.3f}  "
                      f"Δ={change:+.0f}%  {direction}")

        # Key prediction: partial obs should have lower cosine similarity
        fo_sim = np.mean([d.get("mean_cosine_similarity", np.nan)
                          for d in fo if "mean_cosine_similarity" in d])
        po_sim = np.mean([d.get("mean_cosine_similarity", np.nan)
                          for d in po if "mean_cosine_similarity" in d])

        if not np.isnan(fo_sim) and not np.isnan(po_sim):
            print()
            if po_sim < fo_sim - 0.05:
                print("  ✓ SUPPORTED: Partial obs agents have MORE DIVERSE memory states")
                print(f"    Cosine sim: full={fo_sim:.3f} → partial={po_sim:.3f} (Δ={po_sim-fo_sim:+.3f})")
            elif po_sim > fo_sim + 0.05:
                print("  ✗ UNEXPECTED: Partial obs agents have LESS DIVERSE memory states")
                print(f"    Cosine sim: full={fo_sim:.3f} → partial={po_sim:.3f} (Δ={po_sim-fo_sim:+.3f})")
            else:
                print(f"  ~ INCONCLUSIVE: cosine sim change is small ({po_sim-fo_sim:+.3f})")

    print("=" * W)


def plot_diversity(results: dict):
    if not HAS_PLOT:
        return

    os.makedirs("plots", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Meme Diversity: Full Obs vs Partial Obs (memory_only)",
                 fontsize=13, fontweight="bold")

    metrics_to_plot = [
        ("mean_cosine_similarity", "Cosine Similarity\n(↓ = more diverse memes)", False),
        ("effective_dim",          "Effective Dimensionality\n(↑ = richer representations)", True),
        ("mean_memory_delta",      "Memory Update Rate\n(↑ = more dynamic)", True),
        ("inter_agent_std",        "Inter-Agent Std\n(↑ = more specialization)", True),
    ]

    colors = {"full_obs": "#888888", "partial_obs": "#2196F3"}
    labels = {"full_obs": "Full Obs", "partial_obs": "Partial Obs (r=0.5)"}

    for ax, (metric, ylabel, higher_better) in zip(axes.flatten(), metrics_to_plot):
        for key, data in results.items():
            vals = [d.get(metric, np.nan) for d in data if metric in d]
            vals = [v for v in vals if not np.isnan(v)]
            if not vals:
                continue
            xs = list(range(1, len(vals) + 1))
            ax.scatter(xs, vals, color=colors[key], alpha=0.7, s=60,
                       label=labels[key], zorder=3)
            ax.axhline(np.mean(vals), color=colors[key], linewidth=2,
                       linestyle="--", alpha=0.8)

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Seed", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "plots/meme_diversity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def main():
    results = run_analysis()
    print_diversity_report(results)
    plot_diversity(results)

    # Save report
    import io
    buf = io.StringIO()
    old_out = sys.stdout
    sys.stdout = buf
    print_diversity_report(results)
    sys.stdout = old_out

    os.makedirs("plots", exist_ok=True)
    with open("plots/meme_diversity_report.txt", "w") as f:
        f.write(buf.getvalue())
    print("Saved: plots/meme_diversity_report.txt")


if __name__ == "__main__":
    main()
