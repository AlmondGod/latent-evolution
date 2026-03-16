"""
debug_memory_content.py — Deep diagnostic of memory cell representations.

Runs a trained (or untrained) model for several episodes and logs:
1. Per-cell norm trajectories over time
2. Cell diversity (pairwise cosine similarity between cells)
3. Write gate activation values (how much is actually being written?)
4. Read attention entropy (is the agent reading one cell or spreading evenly?)
5. Memory-action correlation (does memory content predict action choice?)
6. Cell usage patterns (are all K cells used, or do they collapse?)
7. Memory content PCA/t-SNE to visualize if cells encode distinct concepts
"""

import sys, os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from new.memetic_foundation.models.agent_network import MemeticFoundationAC
from new.memetic_foundation.training.mpe_wrapper import MPEWrapper


def run_diagnostic(model_path=None, n_episodes=50, scenario='simple_tag_v2', variant='full'):
    """Run deep memory diagnostics."""

    env = MPEWrapper(scenario=scenario)
    info = env.get_env_info()
    obs_dim = info['obs_shape']
    state_dim = info['state_shape']
    n_actions = info['n_actions']
    n_agents = info['n_agents']

    use_mem = variant in ('full', 'memory_only')
    use_comm = variant in ('full', 'comm_only')

    model = MemeticFoundationAC(
        obs_dim, state_dim, n_actions, n_agents,
        hidden_dim=128, mem_dim=128, comm_dim=128, n_mem_cells=8,
        use_memory=use_mem, use_comm=use_comm, mem_decay=0.005,
    )

    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded model from {model_path}")
    else:
        print("Running with UNTRAINED model for baseline comparison")

    model.eval()

    # --- Data collection ---
    cell_norms_over_time = []        # (T, n_agents, K)
    cell_diversity_over_time = []     # (T,) avg pairwise cosine sim
    read_attn_entropy_over_time = []  # (T, n_agents)
    write_gate_values = []            # (T, n_agents)
    memory_delta_norms = []           # (T, n_agents)
    actions_taken = []                # (T, n_agents)
    memory_snapshots = []             # (T, n_agents, K, d) — subsample
    cell_usage_counts = np.zeros((n_agents, 8))  # which cells get max attention

    step_count = 0

    for ep in range(n_episodes):
        env.reset()
        model.reset_memory()
        done = False

        while not done:
            obs_list = env.get_obs()
            obs = torch.tensor(np.array(obs_list), dtype=torch.float32)
            avail = torch.ones(n_agents, n_actions)

            with torch.no_grad():
                # Manual forward to extract internals
                u = model.encode(obs)

                if model.use_memory and model.memory is not None:
                    mem_state = model.memory()  # (N, K, d)
                    z, read_attn = model.memory_reader(u, mem_state)

                    # 1. Cell norms
                    norms = mem_state.norm(dim=-1).numpy()  # (N, K)
                    cell_norms_over_time.append(norms)

                    # 2. Cell diversity: pairwise cosine sim between cells
                    # For each agent, compute avg pairwise cosine sim of their K cells
                    mem_normed = F.normalize(mem_state, dim=-1)  # (N, K, d)
                    sim_matrix = torch.einsum('nid,njd->nij', mem_normed, mem_normed)  # (N, K, K)
                    # Mask diagonal
                    K = sim_matrix.shape[1]
                    mask = ~torch.eye(K, dtype=torch.bool).unsqueeze(0).expand(n_agents, -1, -1)
                    avg_sim = sim_matrix[mask].reshape(n_agents, -1).mean().item()
                    cell_diversity_over_time.append(avg_sim)

                    # 3. Read attention entropy
                    if read_attn is not None:
                        entropy = -(read_attn * (read_attn + 1e-10).log()).sum(dim=-1).numpy()
                        read_attn_entropy_over_time.append(entropy)

                        # Cell usage: which cell gets argmax attention
                        top_cells = read_attn.argmax(dim=-1).numpy()
                        for a in range(n_agents):
                            cell_usage_counts[a, top_cells[a]] += 1

                    # 4. Snapshot memory (subsample every 10 steps)
                    if step_count % 10 == 0:
                        memory_snapshots.append(mem_state.numpy().copy())

                    # Get pre-write state for delta computation
                    pre_write_mem = mem_state.clone()

                # Act
                actions, log_probs, logits = model.act(u, z, avail)
                actions_taken.append(actions.numpy())

                # Communicate + Write
                m_bar, _ = model.send_messages(u, z)

                if model.memory_writer is not None and model.memory is not None:
                    mem_before = model.memory().clone()
                    model.write_memory(u, z, m_bar)
                    mem_after = model.memory()

                    delta = (mem_after - mem_before).norm(dim=-1).mean(dim=-1).numpy()
                    memory_delta_norms.append(delta)

                    # Extract write gate value (approximate from delta magnitude)
                    write_gate_values.append(delta)

            # Step env
            reward, terminated, info_env = env.step(actions.numpy().tolist())
            done = terminated
            step_count += 1

    # --- Analysis ---
    print(f"\n{'='*60}")
    print(f"MEMORY DIAGNOSTIC REPORT ({variant}, {n_episodes} episodes, {step_count} steps)")
    print(f"{'='*60}\n")

    if not cell_norms_over_time:
        print("No memory data collected (variant has no memory)")
        return

    cell_norms = np.array(cell_norms_over_time)  # (T, N, K)

    # 1. Overall cell norm statistics
    print("1. CELL NORM STATISTICS")
    print(f"   Mean cell norm:  {cell_norms.mean():.4f}")
    print(f"   Std cell norm:   {cell_norms.std():.4f}")
    print(f"   Min cell norm:   {cell_norms.min():.4f}")
    print(f"   Max cell norm:   {cell_norms.max():.4f}")

    # Per-cell averages
    avg_per_cell = cell_norms.mean(axis=(0, 1))  # (K,)
    print(f"   Per-cell avg norms: {[f'{x:.3f}' for x in avg_per_cell]}")

    # 2. Cell diversity
    diversity = np.array(cell_diversity_over_time)
    print(f"\n2. CELL DIVERSITY (avg pairwise cosine similarity)")
    print(f"   Mean similarity: {diversity.mean():.4f}")
    print(f"   If ~1.0: cells are COLLAPSED (all same direction) = BAD")
    print(f"   If ~0.0: cells are ORTHOGONAL (diverse) = GOOD")
    print(f"   If < 0:  cells are ANTI-CORRELATED = INTERESTING")

    # 3. Read attention entropy
    if read_attn_entropy_over_time:
        read_entropy = np.array(read_attn_entropy_over_time)  # (T, N)
        max_entropy = np.log(8)  # K=8
        print(f"\n3. READ ATTENTION ENTROPY")
        print(f"   Mean entropy:    {read_entropy.mean():.4f} / {max_entropy:.4f} (max)")
        print(f"   Entropy ratio:   {read_entropy.mean()/max_entropy:.2%}")
        print(f"   If ~100%: uniform attention = memory not specialized")
        print(f"   If <50%:  focused attention = cells encode distinct info")

    # 4. Cell usage distribution
    print(f"\n4. CELL USAGE (argmax of read attention)")
    usage_pcts = cell_usage_counts / cell_usage_counts.sum(axis=1, keepdims=True) * 100
    print(f"   Per-agent cell usage %:")
    for a in range(n_agents):
        print(f"   Agent {a}: {[f'{x:.1f}%' for x in usage_pcts[a]]}")
    uniform_pct = 100.0 / 8
    print(f"   (Uniform would be {uniform_pct:.1f}% each)")

    # 5. Memory delta norms (write magnitude)
    if memory_delta_norms:
        deltas = np.array(memory_delta_norms)
        print(f"\n5. MEMORY WRITE MAGNITUDE")
        print(f"   Mean delta norm: {deltas.mean():.6f}")
        print(f"   Std delta norm:  {deltas.std():.6f}")
        print(f"   If ~0: memory is NOT being written to = writes are dead")
        print(f"   If large relative to cell norm: memory is volatile")
        ratio = deltas.mean() / max(cell_norms.mean(), 1e-10)
        print(f"   Delta/Norm ratio: {ratio:.4f}")

    # 6. Memory-Action Mutual Information (approximate)
    if actions_taken and memory_snapshots:
        print(f"\n6. MEMORY-ACTION CORRELATION")
        # Simple test: cluster memory states and check if actions differ across clusters
        all_actions = np.concatenate(actions_taken)  # (T*N,)
        # Use memory norms as a simple 1D feature
        all_norms = cell_norms.reshape(-1, 8)  # (T*N, K)
        # Check if high-norm-cell-0 agents take different actions than low-norm
        median_norm = np.median(all_norms[:, 0])
        high_mask = all_norms[:, 0] > median_norm
        actions_high = all_actions[high_mask[:len(all_actions)]]
        actions_low = all_actions[~high_mask[:len(all_actions)]]
        if len(actions_high) > 0 and len(actions_low) > 0:
            from collections import Counter
            dist_high = Counter(actions_high.tolist())
            dist_low = Counter(actions_low.tolist())
            print(f"   Action dist (high cell-0 norm): {dict(sorted(dist_high.items()))}")
            print(f"   Action dist (low cell-0 norm):  {dict(sorted(dist_low.items()))}")
            # KL divergence approximation
            all_acts = set(list(dist_high.keys()) + list(dist_low.keys()))
            p = np.array([dist_high.get(a, 1) for a in sorted(all_acts)], dtype=float)
            q = np.array([dist_low.get(a, 1) for a in sorted(all_acts)], dtype=float)
            p /= p.sum()
            q /= q.sum()
            kl = (p * np.log(p / q)).sum()
            print(f"   KL divergence: {kl:.4f}")
            print(f"   If ~0: memory content does NOT influence actions")
            print(f"   If >0.1: memory content influences action selection")

    # --- Plotting ---
    out_dir = '/Users/almondgod/Repositories/memeplex-capstone/plots'
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Memory Diagnostic: {variant} ({n_episodes} eps)', fontsize=14)

    # Plot 1: Cell norms over time (agent 0)
    ax = axes[0, 0]
    for k in range(min(8, cell_norms.shape[2])):
        ax.plot(cell_norms[:, 0, k], alpha=0.7, label=f'Cell {k}')
    ax.set_title('Cell Norms Over Time (Agent 0)')
    ax.set_xlabel('Step')
    ax.set_ylabel('L2 Norm')
    ax.legend(fontsize=6)

    # Plot 2: Cell diversity over time
    ax = axes[0, 1]
    window = min(50, len(diversity))
    if len(diversity) > window:
        smoothed = np.convolve(diversity, np.ones(window)/window, mode='valid')
        ax.plot(smoothed)
    else:
        ax.plot(diversity)
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Orthogonal')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Collapsed')
    ax.set_title('Cell Diversity (Avg Pairwise Cosine Sim)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Cosine Similarity')
    ax.legend()

    # Plot 3: Read attention entropy
    ax = axes[0, 2]
    if read_attn_entropy_over_time:
        ent = np.array(read_attn_entropy_over_time)
        ax.plot(ent.mean(axis=1), alpha=0.7, label='Mean across agents')
        ax.axhline(y=np.log(8), color='red', linestyle='--', alpha=0.5, label='Max (uniform)')
        ax.set_title('Read Attention Entropy')
        ax.set_xlabel('Step')
        ax.set_ylabel('Entropy (nats)')
        ax.legend()

    # Plot 4: Memory delta norms
    ax = axes[1, 0]
    if memory_delta_norms:
        d = np.array(memory_delta_norms)
        ax.plot(d.mean(axis=1) if d.ndim > 1 else d, alpha=0.7)
        ax.set_title('Memory Write Magnitude')
        ax.set_xlabel('Step')
        ax.set_ylabel('Delta Norm')

    # Plot 5: Cell usage heatmap
    ax = axes[1, 1]
    im = ax.imshow(usage_pcts, aspect='auto', cmap='YlOrRd')
    ax.set_title('Cell Usage % (argmax read attn)')
    ax.set_xlabel('Cell Index')
    ax.set_ylabel('Agent')
    plt.colorbar(im, ax=ax)

    # Plot 6: Memory content PCA (snapshots)
    ax = axes[1, 2]
    if memory_snapshots:
        snaps = np.array(memory_snapshots)  # (T_sub, N, K, d)
        # Flatten all cells across time and agents
        all_cells = snaps.reshape(-1, snaps.shape[-1])  # (T*N*K, d)
        # Simple PCA via SVD
        centered = all_cells - all_cells.mean(axis=0)
        if len(centered) > 10:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            pca2d = U[:, :2] * S[:2]
            # Color by cell index
            n_points = snaps.shape[0] * snaps.shape[1]
            cell_ids = np.tile(np.arange(snaps.shape[2]), n_points)
            scatter = ax.scatter(pca2d[:, 0], pca2d[:, 1], c=cell_ids,
                               cmap='Set1', alpha=0.3, s=5)
            ax.set_title('Memory PCA (colored by cell index)')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            plt.colorbar(scatter, ax=ax, label='Cell Index')

            # Print explained variance
            var_explained = S[:5]**2 / (S**2).sum()
            print(f"\n7. MEMORY PCA")
            print(f"   Top-5 explained variance: {[f'{v:.1%}' for v in var_explained]}")
            print(f"   If PC1 dominates: cells live on a line = low diversity")
            print(f"   If spread: cells use multiple dimensions = rich representation")

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f'memory_diagnostic_{variant}.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlots saved to {plot_path}")

    return {
        'cell_norm_mean': float(cell_norms.mean()),
        'cell_diversity': float(diversity.mean()),
        'read_entropy_ratio': float(read_entropy.mean() / np.log(8)) if read_attn_entropy_over_time else 0,
        'write_delta_mean': float(np.array(memory_delta_norms).mean()) if memory_delta_norms else 0,
    }


if __name__ == '__main__':
    import glob

    # Find trained models
    results = {}

    for variant in ['memory_only', 'full']:
        # Try to find a trained model
        patterns = [
            f'checkpoints/mpe_tag_decay/{variant}_seed1/**/*.pt',
            f'checkpoints/mpe_spread_decay/{variant}_seed1/**/*.pt',
            f'checkpoints/memfound_{variant}_*/*.pt',
        ]
        model_path = None
        for pat in patterns:
            files = glob.glob(pat, recursive=True)
            if files:
                model_path = files[0]
                break

        print(f"\n{'#'*60}")
        print(f"# VARIANT: {variant}")
        print(f"# Model: {model_path or 'UNTRAINED'}")
        print(f"{'#'*60}")

        r = run_diagnostic(model_path=model_path, n_episodes=50,
                          scenario='simple_tag_v2', variant=variant)
        if r:
            results[variant] = r

    # Also run untrained for comparison
    print(f"\n{'#'*60}")
    print(f"# VARIANT: full (UNTRAINED BASELINE)")
    print(f"{'#'*60}")
    r_untrained = run_diagnostic(model_path=None, n_episodes=20,
                                scenario='simple_tag_v2', variant='full')
    if r_untrained:
        results['full_untrained'] = r_untrained

    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Variant':<20s} {'Norm':>8s} {'Diversity':>10s} {'ReadEnt%':>10s} {'WriteDelta':>12s}")
    for name, r in results.items():
        print(f"{name:<20s} {r['cell_norm_mean']:>8.4f} {r['cell_diversity']:>10.4f} "
              f"{r['read_entropy_ratio']:>9.1%} {r['write_delta_mean']:>12.6f}")
