"""
meme_content_probe.py — Linear probing to measure meme content in messages.

The "meme" hypothesis claims that agent GRU hidden states encode their role
(landmark assignment), and that this role information is TRANSMITTED via the
communication channel. This probe provides direct evidence via linear regression.

Methodology:
1. Run full_gated policy for N episodes; collect:
   - m_bar: the outgoing message for each agent at each step (N, comm_dim)
   - h: the hidden state for each agent at each step (N, mem_dim)
   - landmark_assignment: which landmark each agent is nearest to (N,)
2. Fit linear classifiers:
   - message → landmark_assignment (does message encode role?)
   - h → landmark_assignment (does hidden state encode role?)
   - h[j] + m_bar[i] → behavior_j (does receiving meme change action?)
3. Compare accuracy to:
   - Random baseline: 1/N (chance)
   - Full obs oracle: train on full-obs agent directly

If message accuracy >> 1/N: communication carries role information ("meme content")
If h accuracy >> message accuracy: hidden state has richer meme content than message

Usage:
    python3.9 -m new.memetic_foundation.scripts.meme_content_probe \
        --checkpoint-dir checkpoints/mpe_spread_partial_obs \
        --n-agents 3 --n-episodes 50 --obs-radius 0.5
"""

import sys, os, glob, json, argparse
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from new.memetic_foundation.models.agent_network import MemeticFoundationAC
from new.memetic_foundation.training.mpe_wrapper import MPEWrapper


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_meme_data(policy, env, n_episodes=50, n_steps=100):
    """
    Collect hidden states, messages, and landmark assignments from eval episodes.

    Returns:
        messages: array (total_steps * N, comm_dim) — outgoing messages
        hidden_states: array (total_steps * N, mem_dim) — GRU hidden states
        landmark_assignments: array (total_steps * N,) — which landmark each agent is nearest
        agent_ids: array (total_steps * N,) — which agent slot (for cross-agent analysis)
    """
    all_messages = []
    all_hidden = []
    all_assignments = []
    all_agent_ids = []

    N = env.num_agents
    policy.eval()

    for ep in range(n_episodes):
        policy.reset_memory()
        obs, _ = env.reset()
        ep_msgs, ep_h, ep_assign, ep_ids = [], [], [], []

        for step in range(n_steps):
            obs_list = env.get_obs()
            obs_t = torch.FloatTensor(np.array(obs_list))

            with torch.no_grad():
                out = policy.forward_step(obs_t, deterministic=True)

            h = out.get("h")
            m_bar = out.get("m_bar")
            actions = out["actions"].tolist()

            # Compute landmark assignments from obs
            obs0 = obs_list[0]
            n_landmarks = env.num_landmarks
            ld_start = 4
            landmarks_rel = obs0[ld_start:ld_start + n_landmarks * 2].reshape(n_landmarks, 2)
            other_start = ld_start + n_landmarks * 2
            others_rel = obs0[other_start:other_start + (N-1)*2].reshape(N-1, 2)

            # Positions relative to agent 0
            agent_positions = [np.zeros(2)] + [others_rel[i] for i in range(N-1)]
            agent_positions = np.array(agent_positions)

            for a_idx in range(N):
                if h is not None:
                    h_np = h[a_idx].detach().cpu().numpy()
                else:
                    h_np = None

                if m_bar is not None:
                    msg_np = m_bar[a_idx].detach().cpu().numpy()
                else:
                    msg_np = None

                # Landmark assignment: closest landmark to this agent
                dists = [np.linalg.norm(agent_positions[a_idx] - landmarks_rel[l])
                         for l in range(n_landmarks)]
                assignment = int(np.argmin(dists))

                if h_np is not None:
                    ep_h.append(h_np)
                if msg_np is not None:
                    ep_msgs.append(msg_np)
                ep_assign.append(assignment)
                ep_ids.append(a_idx)

            rew, done, info = env.step(actions)
            policy.detach_memory()
            if done:
                break

        all_hidden.extend(ep_h)
        all_messages.extend(ep_msgs)
        all_assignments.extend(ep_assign)
        all_agent_ids.extend(ep_ids)

    return {
        "hidden_states": np.array(all_hidden) if all_hidden else None,
        "messages": np.array(all_messages) if all_messages else None,
        "assignments": np.array(all_assignments),
        "agent_ids": np.array(all_agent_ids),
    }


# ---------------------------------------------------------------------------
# Linear probing
# ---------------------------------------------------------------------------

def linear_probe(features, targets, n_classes, n_splits=5, verbose=True):
    """
    Train a linear classifier (logistic regression) to predict targets from features.

    Returns:
        accuracy: mean cross-validated accuracy
        chance_level: 1 / n_classes
        above_chance: accuracy / chance_level
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  sklearn not available, using numpy-based linear probe")
        return _numpy_probe(features, targets, n_classes)

    if len(features) < 20:
        return None

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    y = targets

    # Logistic regression (linear only)
    clf = LogisticRegression(max_iter=1000, C=1.0, multi_class="auto",
                              solver="lbfgs", random_state=42)

    try:
        scores = cross_val_score(clf, X, y, cv=min(n_splits, 5), scoring="accuracy")
        acc = float(scores.mean())
    except Exception as e:
        if verbose:
            print(f"  Warning: cross-val failed ({e}), using single-split")
        clf.fit(X[:len(X)//2], y[:len(X)//2])
        acc = float((clf.predict(X[len(X)//2:]) == y[len(X)//2:]).mean())

    chance = 1.0 / n_classes
    return {
        "accuracy": acc,
        "chance_level": chance,
        "above_chance_ratio": acc / chance,
        "n_samples": len(features),
        "n_classes": n_classes,
    }


def _numpy_probe(features, targets, n_classes):
    """Fallback: one-vs-rest logistic regression via gradient descent."""
    n = len(features)
    if n < 10:
        return None

    X = features - features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    X = X / std

    split = n * 4 // 5
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = targets[:split], targets[split:]

    d = X.shape[1]
    W = np.zeros((n_classes, d))

    # Gradient descent
    for _ in range(200):
        for c in range(n_classes):
            y_c = (y_train == c).astype(float)
            logit = X_train @ W[c]
            pred = 1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10)))
            W[c] -= 0.01 * (X_train.T @ (pred - y_c)) / len(y_train)

    # Predict
    scores = X_test @ W.T  # (n_test, n_classes)
    preds = scores.argmax(axis=1)
    acc = float((preds == y_test).mean())
    chance = 1.0 / n_classes

    return {
        "accuracy": acc,
        "chance_level": chance,
        "above_chance_ratio": acc / chance,
        "n_samples": n,
        "n_classes": n_classes,
    }


# ---------------------------------------------------------------------------
# Alignment analysis: does message sender's role predict receiver's behavior?
# ---------------------------------------------------------------------------

def analyze_behavior_alignment(policy, env, n_episodes=20, n_steps=100):
    """
    Test whether the receiver (agent j) changes behavior when sender (agent i)
    sends a message about its landmark coverage.

    Method:
    1. Normal episode: collect each agent's action at each step
    2. Silenced episode: same policy, but comm zeroed out
    3. Compare: do action distributions differ between normal and silenced?
       If yes, messages are changing behavior → causal transmission.

    Returns: action_divergence (KL divergence between normal and silenced action probs)
    """
    if not policy.use_comm:
        return None

    N = env.num_agents
    policy.eval()

    # Collect action distributions under both conditions
    normal_actions = {i: [] for i in range(N)}
    silent_actions = {i: [] for i in range(N)}

    for ep in range(n_episodes):
        for condition in ["normal", "silent"]:
            policy.reset_memory()
            obs, _ = env.reset()

            for step in range(n_steps):
                obs_t = torch.FloatTensor(np.array(env.get_obs()))
                with torch.no_grad():
                    out = policy.forward_step(
                        obs_t, deterministic=False,
                        intervene_comm_silence=(condition == "silent"),
                    )

                logits = out["logits"].detach().cpu().numpy()
                actions = out["actions"].tolist()
                for a_idx in range(N):
                    if condition == "normal":
                        normal_actions[a_idx].append(logits[a_idx])
                    else:
                        silent_actions[a_idx].append(logits[a_idx])

                rew, done, info = env.step(actions)
                policy.detach_memory()
                if done:
                    break

    # Compute action divergence per agent
    divergences = []
    for a_idx in range(N):
        if not normal_actions[a_idx] or not silent_actions[a_idx]:
            continue
        n_arr = np.array(normal_actions[a_idx])
        s_arr = np.array(silent_actions[a_idx])
        # Softmax to get distributions
        n_probs = np.exp(n_arr - n_arr.max(axis=1, keepdims=True))
        n_probs /= n_probs.sum(axis=1, keepdims=True)
        s_probs = np.exp(s_arr - s_arr.max(axis=1, keepdims=True))
        s_probs /= s_probs.sum(axis=1, keepdims=True)
        # Mean KL divergence (normal || silent)
        kl = np.mean(np.sum(n_probs * np.log(n_probs / (s_probs + 1e-12) + 1e-12), axis=1))
        divergences.append(float(kl))

    return {
        "mean_kl_divergence": float(np.mean(divergences)) if divergences else 0.0,
        "per_agent_kl": divergences,
    }


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_policy(ckpt_dir, n_agents, obs_shape, n_actions, state_shape):
    """Load the best checkpoint from a directory."""
    pt_files = sorted(
        glob.glob(os.path.join(ckpt_dir, "*.pt")) +
        glob.glob(os.path.join(ckpt_dir, "**/*.pt"), recursive=True)
    )
    if not pt_files:
        return None
    ckpt_path = pt_files[-1]

    # Detect variant from directory name
    dir_name = os.path.basename(ckpt_dir)
    use_mem = "memory_only" in dir_name or "full_gated" in dir_name
    use_comm = "full_gated" in dir_name

    policy = MemeticFoundationAC(
        obs_dim=obs_shape, state_dim=state_shape,
        n_actions=n_actions, n_agents=n_agents,
        use_memory=use_mem, use_comm=use_comm,
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
    parser = argparse.ArgumentParser(description="Meme content probe via linear regression")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="checkpoints/mpe_spread_partial_obs")
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--obs-radius", type=float, default=0.5)
    parser.add_argument("--scenario", type=str, default="simple_spread_v2")
    args = parser.parse_args()

    N = args.n_agents
    print("=" * 65)
    print(f"Meme Content Probe: {args.scenario}, N={N}")
    print(f"{'='*65}")
    print(f"Task: can a linear classifier decode landmark assignment from messages?")
    print(f"Chance level: {1/N:.1%}")
    print("=" * 65)

    # Build env
    env = MPEWrapper(
        scenario_name=args.scenario,
        num_adversaries=N, max_cycles=args.n_steps,
        obs_radius=args.obs_radius, N=N,
    )
    info = env.get_env_info()

    results = {}
    variants_to_test = ["full_gated"]  # must have both memory and comm

    for variant in variants_to_test:
        seed_dirs = sorted(glob.glob(
            os.path.join(args.checkpoint_dir, f"{variant}_n{N}_seed*")
        ))
        if not seed_dirs:
            print(f"\n{variant}: no checkpoints found in {args.checkpoint_dir}")
            continue

        print(f"\n{'='*45}")
        print(f"Variant: {variant}_n{N}")
        print(f"{'='*45}")

        # Use best-performing seed
        best_dir = seed_dirs[0]
        policy = load_policy(best_dir, N, info["obs_shape"], info["n_actions"],
                             info["state_shape"])
        if policy is None:
            print(f"  No .pt file found")
            continue
        if not (policy.use_memory and policy.use_comm):
            print(f"  Variant doesn't have both memory and comm, skipping meme probe")
            continue

        print(f"\n--- Collecting meme data ({args.n_episodes} episodes) ---")
        data = collect_meme_data(policy, env, n_episodes=args.n_episodes,
                                 n_steps=args.n_steps)

        variant_results = {}

        # --- Probe 1: Message → Landmark Assignment ---
        if data["messages"] is not None:
            print("\n--- Probe 1: Message → Landmark Assignment ---")
            print(f"  Testing: does message encode which landmark sender is covering?")
            probe_result = linear_probe(
                data["messages"], data["assignments"],
                n_classes=N, verbose=True
            )
            if probe_result:
                acc = probe_result["accuracy"]
                ratio = probe_result["above_chance_ratio"]
                print(f"  Accuracy: {acc:.1%}  (chance: {probe_result['chance_level']:.1%}, "
                      f"{ratio:.2f}x above chance)")
                if ratio > 2.0:
                    print(f"  ✓ STRONG meme content: messages encode landmark roles {ratio:.1f}x above chance!")
                elif ratio > 1.5:
                    print(f"  ~ Moderate meme content: {ratio:.1f}x above chance")
                else:
                    print(f"  ✗ Weak meme content: {ratio:.1f}x above chance (near random)")
                variant_results["message_probe"] = probe_result

        # --- Probe 2: Hidden State → Landmark Assignment ---
        if data["hidden_states"] is not None:
            print("\n--- Probe 2: Hidden State → Landmark Assignment ---")
            print(f"  Testing: does GRU hidden state encode landmark role?")
            probe_h = linear_probe(
                data["hidden_states"], data["assignments"],
                n_classes=N, verbose=True
            )
            if probe_h:
                acc = probe_h["accuracy"]
                ratio = probe_h["above_chance_ratio"]
                print(f"  Accuracy: {acc:.1%}  (chance: {probe_h['chance_level']:.1%}, "
                      f"{ratio:.2f}x above chance)")
                if ratio > 2.0:
                    print(f"  ✓ Hidden state strongly encodes landmark role!")
                variant_results["hidden_probe"] = probe_h

        # --- Probe 3: Agent ID → Landmark Assignment (specialization baseline) ---
        if len(set(data["assignments"])) > 1:
            print("\n--- Probe 3: Agent ID → Landmark Assignment (specialization) ---")
            print(f"  Testing: do different agents consistently cover different landmarks?")
            # Use one-hot agent ID as features
            agent_onehot = np.eye(N)[data["agent_ids"]]
            probe_id = linear_probe(
                agent_onehot, data["assignments"],
                n_classes=N, verbose=True
            )
            if probe_id:
                acc = probe_id["accuracy"]
                ratio = probe_id["above_chance_ratio"]
                print(f"  Agent-landmark consistency: {acc:.1%}  (chance: {1/N:.1%}, "
                      f"{ratio:.2f}x above chance)")
                if ratio > 1.5:
                    print(f"  ✓ Agents show consistent landmark specialization!")
                else:
                    print(f"  ✗ Agents show no consistent landmark assignment (no specialization)")
                variant_results["agent_specialization"] = probe_id

        # --- Analysis 4: Behavior Alignment ---
        print("\n--- Analysis 4: Behavior Alignment (comm → action KL) ---")
        print(f"  Testing: does communication change action distributions?")
        align = analyze_behavior_alignment(policy, env, n_episodes=10,
                                           n_steps=args.n_steps)
        if align:
            kl = align["mean_kl_divergence"]
            print(f"  Mean action KL (normal vs silenced): {kl:.4f}")
            if kl > 0.1:
                print(f"  ✓ Communication significantly alters action distributions!")
            elif kl > 0.01:
                print(f"  ~ Moderate action distribution change from communication")
            else:
                print(f"  ✗ Communication has minimal effect on action distributions")
            variant_results["behavior_alignment"] = align

        results[f"{variant}_n{N}"] = variant_results

    # Save results
    os.makedirs("results", exist_ok=True)
    out_path = f"results/meme_content_probe_n{N}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    # Summary
    print("\n" + "=" * 65)
    print("MEME CONTENT PROBE SUMMARY")
    print("=" * 65)
    for key, res in results.items():
        print(f"\n{key}:")
        if "message_probe" in res:
            mp = res["message_probe"]
            print(f"  Message→Role accuracy: {mp['accuracy']:.1%}  "
                  f"({mp['above_chance_ratio']:.2f}x chance)")
        if "hidden_probe" in res:
            hp = res["hidden_probe"]
            print(f"  HiddenState→Role accuracy: {hp['accuracy']:.1%}  "
                  f"({hp['above_chance_ratio']:.2f}x chance)")
        if "agent_specialization" in res:
            sp = res["agent_specialization"]
            print(f"  AgentID→Role consistency: {sp['accuracy']:.1%}  "
                  f"({sp['above_chance_ratio']:.2f}x chance)")
        if "behavior_alignment" in res:
            ba = res["behavior_alignment"]
            print(f"  Comm→Action KL divergence: {ba['mean_kl_divergence']:.4f}")

    env.close()


if __name__ == "__main__":
    main()
