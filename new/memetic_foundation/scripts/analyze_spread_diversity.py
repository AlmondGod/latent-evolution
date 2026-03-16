"""
analyze_spread_diversity.py — Meme diversity analysis for simple_spread checkpoints.

Tests the key sub-hypothesis: agents develop MORE DIVERSE GRU hidden states under
simple_spread + partial obs than under simple_tag + partial obs.

The reasoning:
  - In simple_tag (partial obs): agents mostly see NOTHING (prey outside obs_radius).
    → GRU states converge because all agents get similar zero-input updates
    → Partial obs creates TEMPORAL not STRUCTURAL asymmetry

  - In simple_spread (partial obs): agents must cover DIFFERENT landmarks.
    → Agents near different landmarks develop different GRU encoding
    → Each agent accumulates a unique "landmark fingerprint" over episodes
    → Partial obs creates STRUCTURAL asymmetry through landmark specialization

If the hypothesis holds: spread diversity > tag diversity (already shown inverted,
but the KEY is that spread diversity should GROW with N and training steps).

Usage:
    python3.9 -m new.memetic_foundation.scripts.analyze_spread_diversity \
        --spread-dir checkpoints/mpe_spread_partial_obs \
        --tag-dir checkpoints/mpe_tag_partial_obs  [optional comparison]
"""

import sys, os, glob, json, argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from new.memetic_foundation.models.agent_network import MemeticFoundationAC
from new.memetic_foundation.training.mpe_wrapper import MPEWrapper


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------

def collect_hidden_states(policy, env, n_steps=200, n_episodes=5):
    """
    Collect GRU hidden states across multiple episodes.

    Returns h_all: array of shape (total_steps, n_agents, mem_dim)
    """
    all_h = []
    policy.eval()

    for ep in range(n_episodes):
        policy.reset_memory()
        obs, _ = env.reset()
        h_ep = []

        for step in range(n_steps):
            obs_t = torch.FloatTensor(np.array(env.get_obs()))
            with torch.no_grad():
                out = policy.forward_step(obs_t, deterministic=True)

            h = out.get("h")
            if h is not None:
                h_ep.append(h.detach().cpu().numpy())  # (N, mem_dim)

            actions = out["actions"].tolist()
            rew, done, info = env.step(actions)
            policy.detach_memory()
            if done:
                break

        if h_ep:
            all_h.extend(h_ep)

    return np.array(all_h) if all_h else None  # (T, N, mem_dim)


def compute_diversity_metrics(h_all):
    """
    Compute multiple diversity metrics from hidden state trajectories.

    h_all: (T, N, mem_dim)

    Returns:
      - pairwise_cosine_sim: mean cosine similarity between agent pairs
      - effective_dim: effective dimensionality via PCA entropy
      - update_rate: mean per-step |Δh| (how much states change)
      - inter_agent_std: std of h across agents (per dim, then mean)
      - h_norm: mean L2 norm of hidden states
    """
    T, N, D = h_all.shape

    # 1. Pairwise cosine similarity (lower = more diverse)
    cos_sims = []
    for t in range(T):
        for i in range(N):
            for j in range(i+1, N):
                hi = h_all[t, i]
                hj = h_all[t, j]
                norm_i = np.linalg.norm(hi) + 1e-8
                norm_j = np.linalg.norm(hj) + 1e-8
                cos_sims.append(float(np.dot(hi, hj) / (norm_i * norm_j)))
    mean_cos_sim = float(np.mean(cos_sims))

    # 2. Effective dimensionality (entropy of PCA eigenvalue spectrum)
    # Flatten to (T*N, D)
    h_flat = h_all.reshape(-1, D)
    # Center
    h_centered = h_flat - h_flat.mean(axis=0)
    # SVD
    try:
        _, s, _ = np.linalg.svd(h_centered, full_matrices=False)
        eigenvalues = s**2
        total = eigenvalues.sum() + 1e-12
        probs = eigenvalues / total
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        effective_dim = float(np.exp(entropy))  # convert to effective count
    except Exception:
        effective_dim = 1.0

    # 3. Update rate: mean per-step |Δh| per agent
    if T > 1:
        deltas = np.linalg.norm(np.diff(h_all, axis=0), axis=-1)  # (T-1, N)
        update_rate = float(deltas.mean())
    else:
        update_rate = 0.0

    # 4. Inter-agent std: std across agents at each step
    inter_agent_std = float(h_all.std(axis=1).mean())  # mean over time and dims

    # 5. H norm
    h_norm = float(np.linalg.norm(h_all, axis=-1).mean())

    return {
        "pairwise_cosine_sim": mean_cos_sim,
        "effective_dim": effective_dim,
        "update_rate": update_rate,
        "inter_agent_std": inter_agent_std,
        "h_norm": h_norm,
        "n_steps": T,
        "n_agents": N,
        "mem_dim": D,
    }


def compute_temporal_specialization(h_all, n_clusters=None):
    """
    Measure whether agents develop persistently different hidden states
    ("temporal specialization" = agents maintain distinct roles over time).

    Method: cluster the hidden states of each agent across time.
    If agents are specialized, their mean states should be far apart.

    Returns:
      - between_agent_var: variance explained by agent identity
      - total_var: total variance
      - specialization_ratio: between / total
    """
    T, N, D = h_all.shape

    # Mean state per agent
    agent_means = h_all.mean(axis=0)  # (N, D)

    # Between-agent variance
    grand_mean = agent_means.mean(axis=0)  # (D,)
    between_var = np.mean(np.sum((agent_means - grand_mean)**2, axis=-1))

    # Total variance
    all_flat = h_all.reshape(-1, D)
    total_var = np.mean(np.sum((all_flat - all_flat.mean(axis=0))**2, axis=-1))

    specialization_ratio = between_var / (total_var + 1e-8)

    return {
        "between_agent_var": float(between_var),
        "total_var": float(total_var),
        "specialization_ratio": float(specialization_ratio),
    }


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------

def load_policy(ckpt_dir, n_agents, obs_shape, n_actions, state_shape,
                use_memory=True, use_comm=False):
    """Load trained policy from a checkpoint directory."""
    pt_files = sorted(
        glob.glob(os.path.join(ckpt_dir, "*.pt")) +
        glob.glob(os.path.join(ckpt_dir, "**/*.pt"), recursive=True)
    )
    if not pt_files:
        return None
    ckpt_path = pt_files[-1]

    policy = MemeticFoundationAC(
        obs_dim=obs_shape, state_dim=state_shape,
        n_actions=n_actions, n_agents=n_agents,
        use_memory=use_memory, use_comm=use_comm,
    )

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in ckpt:
            policy.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif "policy" in ckpt:
            policy.load_state_dict(ckpt["policy"], strict=False)
        else:
            policy.load_state_dict(ckpt, strict=False)
    except Exception as e:
        print(f"  Warning: {e}")
        return None

    policy.eval()
    return policy


# ---------------------------------------------------------------------------
# Per-variant diversity sweep
# ---------------------------------------------------------------------------

def analyze_variant_diversity(ckpt_base_dir, variant, N_list, scenario, obs_radius,
                               n_steps=150, n_episodes=5):
    """Analyze memory diversity for a specific variant across N values."""
    results = {}
    for N in N_list:
        seed_dirs = sorted(glob.glob(
            os.path.join(ckpt_base_dir, f"{variant}_n{N}_seed*")
        ))
        if not seed_dirs:
            continue

        # Build env
        env = MPEWrapper(
            scenario_name=scenario,
            num_adversaries=N, max_cycles=n_steps,
            obs_radius=obs_radius, N=N,
        )
        info = env.get_env_info()

        all_metrics = []
        for sd in seed_dirs[:4]:  # analyze up to 4 seeds
            policy = load_policy(
                sd, N, info["obs_shape"], info["n_actions"],
                info["state_shape"], use_memory=True, use_comm=False,
            )
            if policy is None or not policy.use_memory:
                continue

            h_all = collect_hidden_states(policy, env, n_steps=n_steps,
                                           n_episodes=n_episodes)
            if h_all is None:
                continue

            metrics = compute_diversity_metrics(h_all)
            spec = compute_temporal_specialization(h_all)
            metrics.update(spec)
            all_metrics.append(metrics)

        env.close()

        if all_metrics:
            # Average across seeds
            avg_metrics = {}
            for key in all_metrics[0]:
                if isinstance(all_metrics[0][key], (int, float)):
                    avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))
                    avg_metrics[key + "_std"] = float(np.std([m[key] for m in all_metrics]))
            results[N] = avg_metrics
            print(f"  {variant}_n{N}: cos_sim={avg_metrics['pairwise_cosine_sim']:.3f}, "
                  f"eff_dim={avg_metrics['effective_dim']:.1f}, "
                  f"update_rate={avg_metrics['update_rate']:.4f}, "
                  f"spec_ratio={avg_metrics['specialization_ratio']:.3f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Meme diversity analysis for spread")
    parser.add_argument("--spread-dir", type=str,
                        default="checkpoints/mpe_spread_partial_obs")
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--obs-radius", type=float, default=0.5)
    args = parser.parse_args()

    N_list = [3, 5, 8]
    scenario = "simple_spread_v2"
    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 65)
    print("Meme Diversity Analysis: Simple Spread + Partial Obs")
    print(f"Obs radius: {args.obs_radius}, Steps: {args.n_steps}, Episodes: {args.n_episodes}")
    print("=" * 65)

    # Analyze memory_only variant across N
    variants_to_analyze = ["memory_only", "full_gated"]
    all_diversity = {}

    for variant in variants_to_analyze:
        print(f"\n--- {variant} ---")
        diversity = analyze_variant_diversity(
            args.spread_dir, variant, N_list, scenario, args.obs_radius,
            n_steps=args.n_steps, n_episodes=args.n_episodes,
        )
        all_diversity[variant] = diversity

    if not any(all_diversity.values()):
        print("\nNo checkpoints found yet. Run after experiment completes.")
        return

    # Plot diversity metrics vs N
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Meme Diversity vs N: Simple Spread + Partial Obs\n"
                 "(H: diversity grows with N due to structural role asymmetry)",
                 fontsize=12)

    metrics_to_plot = [
        ("pairwise_cosine_sim", "Pairwise Cosine Similarity (↓ = more diverse)", True),
        ("effective_dim", "Effective Dimensionality (↑ = richer representations)", False),
        ("specialization_ratio", "Specialization Ratio (↑ = more distinct roles)", False),
        ("update_rate", "Memory Update Rate (↑ = more dynamic)", False),
    ]

    colors = {"memory_only": "#2196F3", "full_gated": "#E91E63"}

    for ax, (metric, ylabel, lower_is_better) in zip(axes.flat, metrics_to_plot):
        for variant in variants_to_analyze:
            div = all_diversity[variant]
            Ns_avail = sorted(div.keys())
            if not Ns_avail:
                continue
            vals = [div[N][metric] for N in Ns_avail]
            stds = [div[N].get(metric + "_std", 0) for N in Ns_avail]
            ax.plot(Ns_avail, vals, "o-", color=colors.get(variant, "k"),
                    label=variant, linewidth=2, markersize=8)
            ax.fill_between(Ns_avail,
                            [v - s for v, s in zip(vals, stds)],
                            [v + s for v, s in zip(vals, stds)],
                            alpha=0.2, color=colors.get(variant, "k"))

        ax.set_xlabel("Number of Agents (N)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(N_list)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Mark trend direction
        for variant in variants_to_analyze:
            div = all_diversity[variant]
            Ns_avail = sorted(div.keys())
            if len(Ns_avail) >= 2:
                vals = [div[N][metric] for N in Ns_avail]
                slope, intercept = np.polyfit(Ns_avail, vals, 1)
                if lower_is_better:
                    if slope < -0.01:
                        ax.text(0.05, 0.95, "↓ as N grows ✓", transform=ax.transAxes,
                                fontsize=9, color="green", va="top")
                else:
                    if slope > 0.01:
                        ax.text(0.05, 0.95, "↑ as N grows ✓", transform=ax.transAxes,
                                fontsize=9, color="green", va="top")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "spread_diversity_analysis.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nDiversity plot saved to: {out_path}")
    plt.close()

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/spread_diversity.json", "w") as f:
        json.dump(all_diversity, f, indent=2, default=str)
    print("Results saved to: results/spread_diversity.json")

    # Print summary table
    print("\n" + "=" * 65)
    print("DIVERSITY SUMMARY TABLE")
    print("=" * 65)
    for variant in variants_to_analyze:
        div = all_diversity[variant]
        if not div:
            continue
        print(f"\n{variant}:")
        print(f"  {'N':>4}  {'cos_sim':>8}  {'eff_dim':>8}  {'spec_ratio':>10}  {'update_rate':>12}")
        for N in sorted(div.keys()):
            m = div[N]
            print(f"  {N:>4}  {m['pairwise_cosine_sim']:>8.3f}  "
                  f"{m['effective_dim']:>8.1f}  "
                  f"{m['specialization_ratio']:>10.3f}  "
                  f"{m['update_rate']:>12.4f}")


if __name__ == "__main__":
    main()
