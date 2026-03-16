"""
comm_probe.py — Causal test of meme transmission in simple_spread.

This probe answers the key scientific question for scalable memetics:
  "Does communication causally enable agents to act on information
   they cannot directly observe?"

The test:
  1. Normal condition:   full_gated agent with partial obs (obs_radius=0.5)
  2. Silent condition:   same agent, communication zeroed out (intervene_comm_silence)
  3. Blind+comm:         one agent sees NO landmarks (obs_radius=0), but receives messages
  4. Blind+silent:       one agent blind AND communication silenced (double-blind control)

If memory/comm is genuinely transmitting "memes" (role representations):
  - Normal > Silent     (comm helps coordination)
  - Blind+comm > Blind+silent   (comm carries useful landmark info to blind agent)
  - The blind agent should still move toward its assigned landmark based on comms alone

Usage:
    python3.9 -m new.memetic_foundation.scripts.comm_probe \
        --checkpoint checkpoints/mpe_spread_partial_obs/full_gated_n3_seed1 \
        --n-agents 3 --n-episodes 20

Scalability test:
    Run for N=3, N=5, N=8 — communication value should grow with N.
"""

import sys, os, glob, json, argparse, copy
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from new.memetic_foundation.models.agent_network import MemeticFoundationAC
from new.memetic_foundation.training.mpe_wrapper import MPEWrapper


# ---------------------------------------------------------------------------
# Evaluation engine
# ---------------------------------------------------------------------------

def run_probe_episode(policy, env, n_steps, condition):
    """
    Run one episode under a specific intervention condition.

    condition: dict with keys:
        - comm_silence (bool): zero out all outgoing messages
        - blind_agent (int or None): this agent's observations are zeroed
        - obs_radius_override (float or None): override env obs_radius for blind agent
    """
    policy.eval()
    policy.reset_memory()

    obs_list = env.get_obs()  # list of (obs_dim,) arrays
    ep_reward = 0.0
    n_agents = env.num_agents

    # Track landmark coverage per step
    coverage_history = []
    blind_agent_moves = []  # displacement of blind agent per step

    obs, _ = env.reset()
    policy.reset_memory()

    prev_positions = None

    for step in range(n_steps):
        obs_list = env.get_obs()

        # Apply blind agent condition: zero their observation
        if condition.get("blind_agent") is not None:
            ba = condition["blind_agent"]
            obs_list = list(obs_list)
            obs_list[ba] = np.zeros_like(obs_list[ba])

        obs_t = torch.FloatTensor(np.array(obs_list))

        with torch.no_grad():
            out = policy.forward_step(
                obs_t,
                deterministic=True,
                intervene_comm_silence=condition.get("comm_silence", False),
            )

        actions = out["actions"].tolist()
        rew, done, info = env.step(actions)
        ep_reward += rew

        # Track blind agent behavior
        if condition.get("blind_agent") is not None and out.get("h") is not None:
            ba = condition["blind_agent"]
            h = out["h"].detach().cpu().numpy()
            blind_agent_moves.append(h[ba])  # track hidden state evolution

        # Track coverage
        if "min_dist" in info:
            coverage_history.append(info.get("min_dist", 1.0))

        if done:
            break

    policy.detach_memory()

    return {
        "reward": ep_reward,
        "mean_dist": float(np.mean(coverage_history)) if coverage_history else 1.0,
        "final_dist": coverage_history[-1] if coverage_history else 1.0,
        "blind_h_trajectory": np.array(blind_agent_moves) if blind_agent_moves else None,
    }


def run_condition_n_episodes(policy, env, n_steps, condition, n_episodes=20, label=""):
    """Run multiple episodes for a condition and return statistics."""
    rewards, dists = [], []
    for ep in range(n_episodes):
        result = run_probe_episode(policy, env, n_steps, condition)
        rewards.append(result["reward"])
        dists.append(result["mean_dist"])

    r = np.array(rewards)
    d = np.array(dists)
    print(f"  {label:30s}: reward={r.mean():.1f}±{r.std():.1f}  "
          f"dist={d.mean():.3f}±{d.std():.3f}  (n={n_episodes})")
    return {"rewards": rewards, "dists": dists, "label": label}


# ---------------------------------------------------------------------------
# Blind-agent activation analysis
# ---------------------------------------------------------------------------

def analyze_blind_agent_sensitivity(policy, env, n_steps=100):
    """
    Run one episode where a single agent is blind (zero obs) but receives comms.
    Track whether the blind agent's hidden state h shows non-trivial dynamics
    (i.e., is it being influenced by received messages?).

    Compare:
      A) Blind + comm active: h should evolve based on received messages
      B) Blind + comm silent: h should show minimal evolution (no signal)
    """
    if not (policy.use_memory and policy.use_comm):
        print("  Skipping blind agent analysis (requires full_gated variant)")
        return None

    print("\n--- Blind Agent Activation Analysis ---")

    conditions = [
        {"blind_agent": 0, "comm_silence": False, "label": "blind_comm_active"},
        {"blind_agent": 0, "comm_silence": True,  "label": "blind_comm_silent"},
    ]

    h_trajectories = {}
    for cond in conditions:
        policy.reset_memory()
        h_seq = []

        obs, _ = env.reset()
        for step in range(n_steps):
            obs_list = env.get_obs()
            obs_list = list(obs_list)
            obs_list[0] = np.zeros_like(obs_list[0])  # blind agent 0
            obs_t = torch.FloatTensor(np.array(obs_list))

            with torch.no_grad():
                out = policy.forward_step(
                    obs_t,
                    deterministic=True,
                    intervene_comm_silence=cond["comm_silence"],
                )

            h = out.get("h")
            if h is not None:
                h_seq.append(h[0].detach().cpu().numpy())  # agent 0's hidden state

            actions = out["actions"].tolist()
            rew, done, info = env.step(actions)
            if done:
                break
            policy.detach_memory()

        h_trajectories[cond["label"]] = np.array(h_seq) if h_seq else None

    # Compute dynamics
    for label, h_seq in h_trajectories.items():
        if h_seq is None or len(h_seq) < 2:
            continue
        deltas = np.linalg.norm(np.diff(h_seq, axis=0), axis=1)
        print(f"  {label}: mean |Δh| = {deltas.mean():.4f} ± {deltas.std():.4f}")

    # Key metric: does comm increase h dynamics for blind agent?
    if all(v is not None for v in h_trajectories.values()):
        h_comm = h_trajectories["blind_comm_active"]
        h_sil = h_trajectories["blind_comm_silent"]
        if len(h_comm) > 1 and len(h_sil) > 1:
            delta_comm = np.linalg.norm(np.diff(h_comm, axis=0), axis=1).mean()
            delta_sil = np.linalg.norm(np.diff(h_sil, axis=0), axis=1).mean()
            ratio = delta_comm / (delta_sil + 1e-8)
            print(f"\n  RESULT: comm/silent h-dynamics ratio = {ratio:.2f}x")
            if ratio > 1.5:
                print("  ✓ Communication causally drives blind agent's hidden state!")
            elif ratio > 1.1:
                print("  ~ Mild communication influence on hidden state")
            else:
                print("  ✗ Communication shows no effect on blind agent's hidden state")
            return ratio
    return None


# ---------------------------------------------------------------------------
# Landmark specialization probe
# ---------------------------------------------------------------------------

def measure_landmark_specialization(policy, env, n_steps, n_episodes=10):
    """
    Measure whether agents consistently cover the same landmark across episodes.

    For each episode, determine which agent ends up closest to each landmark.
    If specialization is happening, agent i should be nearest to landmark i
    consistently across episodes.

    Returns: specialization_score (0 = random, 1 = perfect specialization)
    """
    N = env.num_agents
    assignment_matrix = np.zeros((N, N))  # [agent, landmark] co-occurrence

    for ep in range(n_episodes):
        policy.reset_memory()
        obs, _ = env.reset()

        # Track per-agent, per-landmark proximity over episode
        agent_landmark_proximity = np.zeros((N, N))  # [agent, landmark]

        for step in range(n_steps):
            obs_list = env.get_obs()
            obs_t = torch.FloatTensor(np.array(obs_list))

            with torch.no_grad():
                out = policy.forward_step(obs_t, deterministic=True)

            actions = out["actions"].tolist()
            rew, done, info = env.step(actions)

            # Parse agent/landmark positions from obs
            obs0 = env.get_obs()[0]
            n_landmarks = env.num_landmarks
            ld_start = 4
            landmarks_rel = obs0[ld_start:ld_start + n_landmarks * 2].reshape(n_landmarks, 2)
            other_start = ld_start + n_landmarks * 2
            others_rel = obs0[other_start:other_start + (N-1)*2].reshape(N-1, 2)

            # Positions relative to agent 0
            agent_positions = [np.zeros(2)] + [others_rel[i] for i in range(N-1)]
            landmark_positions = landmarks_rel

            for a_idx in range(N):
                for l_idx in range(n_landmarks):
                    d = np.linalg.norm(np.array(agent_positions[a_idx]) - landmark_positions[l_idx])
                    agent_landmark_proximity[a_idx, l_idx] += float(1.0 / (d + 0.01))

            if done:
                break
            policy.detach_memory()

        # Assign each agent to its most-covered landmark this episode
        assignments = np.argmax(agent_landmark_proximity, axis=1)
        for a_idx in range(N):
            assignment_matrix[a_idx, assignments[a_idx]] += 1

    # Specialization = fraction of episodes with consistent assignment
    # Perfect: each agent always assigned same landmark (diagonal dominant)
    row_max = assignment_matrix.max(axis=1)  # most-common landmark for each agent
    total = assignment_matrix.sum(axis=1)
    consistency = (row_max / (total + 1e-8)).mean()

    # Check uniqueness: are different agents assigned to different landmarks?
    most_common_assignments = assignment_matrix.argmax(axis=1)
    n_unique = len(set(most_common_assignments))
    uniqueness = n_unique / N  # 1.0 = all different landmarks

    specialization_score = consistency * uniqueness

    print(f"\n--- Landmark Specialization ---")
    print(f"  Assignment consistency: {consistency:.2%}")
    print(f"  Assignment uniqueness:  {uniqueness:.2%}")
    print(f"  Specialization score:   {specialization_score:.2%}")
    print(f"  Agent→Landmark modal assignments: {most_common_assignments.tolist()}")

    return {
        "consistency": float(consistency),
        "uniqueness": float(uniqueness),
        "specialization_score": float(specialization_score),
        "assignment_matrix": assignment_matrix.tolist(),
    }


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_policy_from_dir(ckpt_dir, n_agents, obs_shape, n_actions, state_shape,
                          use_memory=True, use_comm=True):
    """Load a trained policy from checkpoint directory."""
    # Find .pt files
    pt_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")) +
                      glob.glob(os.path.join(ckpt_dir, "**/*.pt"), recursive=True))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {ckpt_dir}")

    ckpt_path = pt_files[-1]  # last checkpoint = most trained
    print(f"  Loading: {ckpt_path}")

    policy = MemeticFoundationAC(
        obs_dim=obs_shape,
        state_dim=state_shape,
        n_actions=n_actions,
        n_agents=n_agents,
        use_memory=use_memory,
        use_comm=use_comm,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        policy.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif "policy" in ckpt:
        policy.load_state_dict(ckpt["policy"], strict=False)
    else:
        policy.load_state_dict(ckpt, strict=False)

    policy.eval()
    return policy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Communication probe for scalable memetics")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="checkpoints/mpe_spread_partial_obs",
                        help="Base directory with variant checkpoints")
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--obs-radius", type=float, default=0.5)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--scenario", type=str, default="simple_spread_v2")
    args = parser.parse_args()

    N = args.n_agents
    print("=" * 65)
    print(f"Communication Probe: {args.scenario}, N={N}, obs_radius={args.obs_radius}")
    print("=" * 65)

    # Build env for info
    env = MPEWrapper(
        scenario_name=args.scenario,
        num_adversaries=N,
        max_cycles=args.n_steps,
        obs_radius=args.obs_radius,
        N=N,
    )
    info = env.get_env_info()
    obs_shape = info["obs_shape"]
    n_actions = info["n_actions"]
    state_shape = info["state_shape"]

    # Find checkpoints
    ckpt_base = args.checkpoint_dir

    variants = [
        ("baseline",    False, False),
        ("memory_only", True,  False),
        ("full_gated",  True,  True),
    ]

    all_results = {}

    for variant_name, use_mem, use_comm in variants:
        # Find best seed checkpoint for this variant and N
        seed_dirs = sorted(glob.glob(
            os.path.join(ckpt_base, f"{variant_name}_n{N}_seed*")
        ))
        if not seed_dirs:
            print(f"\n[{variant_name}_n{N}] No checkpoints found, skipping")
            continue

        # Pick best seed based on log file
        best_dir = seed_dirs[-1]
        for sd in seed_dirs:
            log_path = sd + ".log"
            if os.path.exists(log_path):
                import re
                content = open(log_path, errors="replace").read()
                evals = re.findall(r"\[Eval\] Step \d+ \| reward=([-\d.]+)", content)
                if evals:
                    peak = max(float(e) for e in evals)
                    if peak > -400:  # reasonable threshold
                        best_dir = sd
                        break

        print(f"\n{'='*40}")
        print(f"Variant: {variant_name}_n{N}")
        print(f"{'='*40}")

        try:
            policy = load_policy_from_dir(best_dir, N, obs_shape, n_actions,
                                           state_shape, use_memory=use_mem,
                                           use_comm=use_comm)
        except Exception as e:
            print(f"  Could not load policy: {e}")
            continue

        variant_results = {}

        # --- Main probe conditions ---
        print("\n--- Performance under intervention conditions ---")

        # 1. Normal
        cond_normal = {"comm_silence": False, "blind_agent": None}
        r_normal = run_condition_n_episodes(
            policy, env, args.n_steps, cond_normal,
            n_episodes=args.n_episodes, label="Normal"
        )
        variant_results["normal"] = r_normal

        # 2. Comm silenced
        if use_comm:
            cond_silent = {"comm_silence": True, "blind_agent": None}
            r_silent = run_condition_n_episodes(
                policy, env, args.n_steps, cond_silent,
                n_episodes=args.n_episodes, label="Comm silenced"
            )
            variant_results["comm_silent"] = r_silent

            # Comm effect: how much does silencing hurt?
            normal_r = np.mean(r_normal["rewards"])
            silent_r = np.mean(r_silent["rewards"])
            comm_effect = (normal_r - silent_r) / (abs(normal_r) + 1e-8)
            print(f"\n  *** Comm causal effect: {comm_effect:.1%} reward change from silencing ***")
            variant_results["comm_causal_effect"] = float(comm_effect)

        # 3. Blind agent (agent 0 sees nothing, but can receive comms)
        if use_comm:
            cond_blind_comm = {"comm_silence": False, "blind_agent": 0}
            r_blind_comm = run_condition_n_episodes(
                policy, env, args.n_steps, cond_blind_comm,
                n_episodes=args.n_episodes, label="Agent 0 blind + comm active"
            )
            variant_results["blind_comm"] = r_blind_comm

            # 4. Blind + silenced (hardest condition, double-blind control)
            cond_blind_sil = {"comm_silence": True, "blind_agent": 0}
            r_blind_sil = run_condition_n_episodes(
                policy, env, args.n_steps, cond_blind_sil,
                n_episodes=args.n_episodes, label="Agent 0 blind + comm silenced"
            )
            variant_results["blind_silent"] = r_blind_sil

            # Blind communication effect
            bc_r = np.mean(r_blind_comm["rewards"])
            bs_r = np.mean(r_blind_sil["rewards"])
            blind_comm_effect = (bc_r - bs_r) / (abs(bs_r) + 1e-8)
            print(f"\n  *** Blind-agent comm effect: {blind_comm_effect:.1%} "
                  f"(comm helps blind agent navigate) ***")
            variant_results["blind_comm_effect"] = float(blind_comm_effect)

        # --- Hidden state analysis ---
        if use_mem and use_comm:
            ratio = analyze_blind_agent_sensitivity(policy, env, n_steps=args.n_steps)
            if ratio is not None:
                variant_results["blind_h_dynamics_ratio"] = float(ratio)

        # --- Landmark specialization ---
        spec = measure_landmark_specialization(policy, env, args.n_steps,
                                               n_episodes=args.n_episodes // 2 + 1)
        variant_results["specialization"] = spec

        all_results[f"{variant_name}_n{N}"] = variant_results

    # --- N-scaling summary ---
    print("\n" + "=" * 65)
    print("SCALABLE MEMETICS PROBE SUMMARY")
    print("=" * 65)
    print(f"\nN={N} results:")
    for variant_key, res in all_results.items():
        if "comm_causal_effect" in res:
            print(f"  {variant_key}: comm_effect={res['comm_causal_effect']:.1%}, "
                  f"specialization={res['specialization']['specialization_score']:.1%}")

    # Save results
    os.makedirs("results", exist_ok=True)
    out_path = f"results/comm_probe_n{N}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    env.close()


if __name__ == "__main__":
    main()
