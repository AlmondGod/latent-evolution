"""
role_stability_probe.py — Test whether landmark role assignments are stable
across independent episodes (with memory reset between episodes).

Two hypotheses:
1. INTRA-EPISODE stability: within a single episode, does each agent consistently
   cover the same landmark? (Already tested by meme_content_probe specialization metric)

2. INTER-EPISODE stability: across different episodes (with h reset to h_0),
   does agent_i tend to cover the same landmark? If h_0 encodes no role information
   (shared zeros-initialized), roles should be assigned randomly per episode.

3. ROLE-LOCK duration: how many steps into an episode does it take for agents
   to "commit" to a stable role? (Onset of specialization)

This probes the TEMPORAL STRUCTURE of meme formation.

Usage:
    python3.9 -m new.memetic_foundation.scripts.role_stability_probe \
        --ckpt-dir checkpoints/mpe_spread_partial_obs \
        --n-agents 5 --n-episodes 30 --obs-radius 0.5
"""

import sys, os, glob, json, argparse
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from new.memetic_foundation.models.agent_network import MemeticFoundationAC
from new.memetic_foundation.training.mpe_wrapper import MPEWrapper


def get_landmark_assignments(obs_list, n_landmarks):
    """From observations, compute which landmark each agent is nearest to."""
    N = len(obs_list)
    assignments = []

    # obs format: [vel(2), pos(2), landmark_rel(N_lm*2), other_agents_rel((N-1)*2), ...]
    landmark_start = 4
    for i, obs in enumerate(obs_list):
        lm_dists = []
        for l in range(n_landmarks):
            idx = landmark_start + l * 2
            rel = obs[idx:idx+2]
            dist = float(np.linalg.norm(rel))
            # If masked (obs_radius), dist=0 means either at landmark or masked
            # Use raw norm as proxy (when masked, all zeros → dist=0 for all)
            lm_dists.append(dist)
        # Assign to nearest visible landmark
        assignments.append(int(np.argmin(lm_dists)))
    return assignments


def compute_role_lock_onset(episode_assignments, min_stable_steps=10):
    """
    Find the earliest step at which each agent's role becomes stable.

    Stability: an agent is 'locked' when it maintains the same assignment
    for `min_stable_steps` consecutive steps.

    Returns: onset_steps (list of N onset steps, or None if never locked)
    """
    N = len(episode_assignments[0])
    T = len(episode_assignments)

    onset_steps = [None] * N
    for agent_i in range(N):
        agent_traj = [episode_assignments[t][agent_i] for t in range(T)]
        for t in range(T - min_stable_steps):
            window = agent_traj[t:t + min_stable_steps]
            if len(set(window)) == 1:  # all same role
                onset_steps[agent_i] = t
                break
    return onset_steps


def load_best_policy(ckpt_base_dir, variant, N, obs_shape, n_actions, state_shape):
    """Load the best-performing (lowest dist) seed."""
    # Find seed dirs for this variant+N
    seed_dirs = sorted(glob.glob(os.path.join(ckpt_base_dir, f"{variant}_n{N}_seed*")))
    if not seed_dirs:
        return None, None

    # Read final dist from logs to find best seed
    best_dir = None
    best_dist = float('inf')
    for sd in seed_dirs:
        log_path = sd + ".log"
        if not os.path.exists(log_path):
            continue
        try:
            with open(log_path) as f:
                lines = f.readlines()
            eval_lines = [l for l in lines if '[Eval]' in l]
            if eval_lines:
                last = eval_lines[-1]
                dist_str = last.split("dist=")[1].strip()
                dist = float(dist_str)
                if dist < best_dist:
                    best_dist = dist
                    best_dir = sd
        except Exception:
            continue

    if best_dir is None:
        best_dir = seed_dirs[0]

    # Detect variant
    use_mem = "memory_only" in variant or "full_gated" in variant
    use_comm = "full_gated" in variant

    policy = MemeticFoundationAC(
        obs_dim=obs_shape, state_dim=state_shape,
        n_actions=n_actions, n_agents=N,
        use_memory=use_mem, use_comm=use_comm,
    )

    pt_files = sorted(
        glob.glob(os.path.join(best_dir, "*.pt")) +
        glob.glob(os.path.join(best_dir, "**/*.pt"), recursive=True)
    )
    if not pt_files:
        return None, None

    ckpt = torch.load(pt_files[-1], map_location="cpu")
    if "model_state_dict" in ckpt:
        policy.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif "policy" in ckpt:
        policy.load_state_dict(ckpt["policy"], strict=False)
    else:
        policy.load_state_dict(ckpt, strict=False)
    policy.eval()

    return policy, best_dist


def probe_role_stability(policy, env, n_episodes=30, n_steps=100):
    """
    Run n_episodes with policy.reset_memory() between each.
    Measure:
      - intra-episode role consistency (per-agent consistency within episode)
      - inter-episode role assignment (does agent 0 always cover landmark 0?)
      - role-lock onset (how fast does specialization emerge)

    Returns: dict of metrics
    """
    N = env.num_agents
    n_landmarks = env.num_landmarks
    policy.eval()

    # Collect per-episode final role assignments
    # (which landmark each agent covers in the last 20% of steps)
    episode_final_roles = []  # (n_episodes, N) → which landmark each agent covered

    # Collect intra-episode consistency
    intra_consistencies = []

    # Collect role-lock onset times
    lock_onsets = []

    for ep in range(n_episodes):
        policy.reset_memory()
        obs, _ = env.reset()

        ep_assignments = []  # (T, N)

        for step in range(n_steps):
            obs_list = env.get_obs()
            obs_t = torch.FloatTensor(np.array(obs_list))

            with torch.no_grad():
                out = policy.forward_step(obs_t, deterministic=True)

            actions = out["actions"].tolist()
            assignments = get_landmark_assignments(obs_list, n_landmarks)
            ep_assignments.append(assignments)

            rew, done, info = env.step(actions)
            policy.detach_memory()
            if done:
                break

        # 1. Final role: majority assignment in last 20% of steps
        T = len(ep_assignments)
        final_window = ep_assignments[int(T * 0.8):]
        final_roles = []
        for agent_i in range(N):
            agent_traj = [final_window[t][agent_i] for t in range(len(final_window))]
            if agent_traj:
                counts = np.bincount(agent_traj, minlength=n_landmarks)
                final_roles.append(int(np.argmax(counts)))
            else:
                final_roles.append(-1)
        episode_final_roles.append(final_roles)

        # 2. Intra-episode consistency: for each agent, how consistent is its role?
        for agent_i in range(N):
            agent_traj = [ep_assignments[t][agent_i] for t in range(T)]
            if not agent_traj:
                continue
            counts = np.bincount(agent_traj, minlength=n_landmarks)
            majority_frac = float(counts.max()) / len(agent_traj)
            intra_consistencies.append(majority_frac)

        # 3. Role-lock onset
        onsets = compute_role_lock_onset(ep_assignments, min_stable_steps=10)
        lock_onsets.extend([o for o in onsets if o is not None])

    # Inter-episode stability: does agent_i tend to cover the same landmark?
    episode_final_roles = np.array(episode_final_roles)  # (n_ep, N)
    inter_consistencies = []
    for agent_i in range(N):
        role_traj = episode_final_roles[:, agent_i]
        role_traj = role_traj[role_traj >= 0]
        if len(role_traj) == 0:
            continue
        counts = np.bincount(role_traj, minlength=n_landmarks)
        majority_frac = float(counts.max()) / len(role_traj)
        inter_consistencies.append(majority_frac)

    # Role coverage: are all N landmarks covered in most episodes?
    # (coverage diversity: 1.0 = all N landmarks covered by different agents)
    coverage_diversities = []
    for ep_roles in episode_final_roles:
        unique_roles = len(set(ep_roles))
        coverage_diversities.append(unique_roles / n_landmarks)

    return {
        "intra_episode_consistency": float(np.mean(intra_consistencies)),
        "inter_episode_consistency": float(np.mean(inter_consistencies)),
        "coverage_diversity": float(np.mean(coverage_diversities)),
        "mean_lock_onset_steps": float(np.mean(lock_onsets)) if lock_onsets else None,
        "fraction_agents_locked": float(len(lock_onsets) / (n_episodes * N)),
        "n_episodes": n_episodes,
        "n_agents": N,
        "n_landmarks": n_landmarks,
    }


def main():
    parser = argparse.ArgumentParser(description="Role stability probe")
    parser.add_argument("--ckpt-dir", type=str,
                        default="checkpoints/mpe_spread_partial_obs")
    parser.add_argument("--n-agents", type=int, default=5)
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--obs-radius", type=float, default=0.5)
    parser.add_argument("--scenario", type=str, default="simple_spread_v2")
    args = parser.parse_args()

    N = args.n_agents
    print("=" * 65)
    print(f"Role Stability Probe: {args.scenario}, N={N}")
    print(f"Measuring: intra-episode, inter-episode, lock-onset")
    print("=" * 65)

    env = MPEWrapper(
        scenario_name=args.scenario,
        num_adversaries=N, max_cycles=args.n_steps,
        obs_radius=args.obs_radius, N=N,
    )
    info = env.get_env_info()

    results = {}
    for variant in ["baseline", "memory_only", "full_gated"]:
        print(f"\n--- {variant}_n{N} ---")
        policy, best_dist = load_best_policy(
            args.ckpt_dir, variant, N, info["obs_shape"],
            info["n_actions"], info["state_shape"]
        )
        if policy is None:
            print("  No checkpoint found")
            continue

        print(f"  Best seed dist: {best_dist:.3f}")
        metrics = probe_role_stability(policy, env, args.n_episodes, args.n_steps)
        results[variant] = metrics

        print(f"  Intra-episode consistency: {metrics['intra_episode_consistency']:.3f} "
              f"(fraction of steps in majority role)")
        print(f"  Inter-episode consistency: {metrics['inter_episode_consistency']:.3f} "
              f"(does agent always cover same landmark?)")
        print(f"  Coverage diversity: {metrics['coverage_diversity']:.3f} "
              f"(fraction of landmarks covered by distinct agents)")
        if metrics['mean_lock_onset_steps'] is not None:
            print(f"  Role-lock onset: {metrics['mean_lock_onset_steps']:.1f} steps "
                  f"(fraction locked: {metrics['fraction_agents_locked']:.2f})")
        else:
            print(f"  Role-lock onset: N/A (no agents locked within episode)")

    env.close()

    # Save results
    os.makedirs("results", exist_ok=True)
    out_path = f"results/role_stability_n{N}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    # Print comparison table
    print("\n" + "=" * 65)
    print("ROLE STABILITY COMPARISON")
    print("=" * 65)
    print(f"{'Metric':<30} {'baseline':>12} {'memory_only':>13} {'full_gated':>12}")
    print("-" * 70)

    metrics_to_show = [
        ("intra_episode_consistency", "Intra-ep consistency"),
        ("inter_episode_consistency", "Inter-ep consistency"),
        ("coverage_diversity", "Coverage diversity"),
        ("fraction_agents_locked", "Fraction locked"),
    ]

    for key, label in metrics_to_show:
        vals = {v: results[v].get(key, float('nan')) for v in results}
        print(f"{label:<30} {vals.get('baseline', float('nan')):>12.3f} "
              f"{vals.get('memory_only', float('nan')):>13.3f} "
              f"{vals.get('full_gated', float('nan')):>12.3f}")


if __name__ == "__main__":
    main()
