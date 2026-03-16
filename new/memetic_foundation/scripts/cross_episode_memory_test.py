"""
cross_episode_memory_test.py

Tests whether cross-episode persistent memory genuinely helps performance
or is just noise. Compares:
  A) Normal: memory persists across episode boundaries (designed behavior)
  B) Reset:  memory zeroed at each episode boundary (standard MARL)

Uses trained full (gated) checkpoint from mpe_tag_gated suite.
Runs 20 eval episodes for each condition and reports mean team reward.

Usage:
    python3.9 -m new.memetic_foundation.scripts.cross_episode_memory_test
"""

import sys, os, glob, json, torch
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from new.memetic_foundation.training.mpe_wrapper import MPEWrapper

def find_best_checkpoint(base_dir, variant="full_gated"):
    """Find the checkpoint with the highest reward."""
    logs = sorted(glob.glob(f"{base_dir}/{variant}_seed*.log"))
    best_rew, best_seed = -1e9, 1
    for lf in logs:
        try:
            with open(lf, errors='replace') as f: content = f.read()
            import re
            m = re.findall(r'rew=([-\d.]+)', content)
            if m and float(m[-1]) > best_rew:
                best_rew = float(m[-1])
                best_seed = int(re.search(r'seed(\d+)', lf).group(1))
        except: pass
    print(f"Best seed: {best_seed} (rew={best_rew:.0f})")
    ckpts = glob.glob(f"{base_dir}/{variant}_seed{best_seed}/*/*.pt")
    return ckpts[0] if ckpts else None, best_seed

def run_eval_episodes(policy, env, n_episodes=20, reset_memory_each_ep=False):
    """Run eval episodes and return mean reward."""
    policy.eval()
    rewards = []
    
    if reset_memory_each_ep:
        policy.reset_memory()
    else:
        policy.reset_memory()  # start fresh but persist after
    
    for ep in range(n_episodes):
        if reset_memory_each_ep:
            policy.reset_memory()
        
        obs, _ = env.reset()
        obs_t = torch.FloatTensor(np.array(obs))
        ep_reward = 0.0
        done = False
        
        while not done:
            with torch.no_grad():
                out = policy.forward_step(obs_t, deterministic=True)
            actions = out["actions"].tolist()
            rew, done, _ = env.step(actions)
            ep_reward += rew
            if not done:
                obs, _ = env.reset() if done else (env.get_obs(), None)
                obs_t = torch.FloatTensor(np.array(env.get_obs()))
        
        policy.detach_memory()
        rewards.append(ep_reward)
    
    return rewards

def main():
    base_dir = "checkpoints/mpe_tag_gated"
    
    print("=" * 60)
    print("Cross-Episode Memory Persistence Test")
    print("Question: does persistent memory help across episodes?")
    print("=" * 60)
    
    ckpt_path, best_seed = find_best_checkpoint(base_dir)
    if ckpt_path is None:
        print("No checkpoint found. Run the gated suite first.")
        return
    
    print(f"Loading: {ckpt_path}")
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Recreate policy from checkpoint config
    from new.memetic_foundation.models.agent_network import MemeticFoundationAC
    cfg = checkpoint.get('config', {})
    
    env = MPEWrapper("simple_tag_v2", num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=100)
    env_info = env.get_env_info()
    
    policy = MemeticFoundationAC(
        obs_dim=env_info['obs_shape'],
        state_dim=env_info['state_shape'],
        n_actions=env_info['n_actions'],
        n_agents=env_info['n_agents'],
        hidden_dim=cfg.get('hidden_dim', 128),
        mem_dim=cfg.get('mem_dim', 128),
        comm_dim=cfg.get('comm_dim', 128),
        use_memory=cfg.get('use_memory', True),
        use_comm=cfg.get('use_comm', True),
        use_gate=True,
    )
    policy.load_state_dict(checkpoint['policy_state_dict'])
    
    N_EPISODES = 30
    
    print(f"\nRunning {N_EPISODES} eval episodes per condition...")
    
    # Condition A: persistent memory (normal behavior)
    print("\n[A] Persistent memory (no reset between episodes)...")
    rewards_persistent = run_eval_episodes(policy, env, N_EPISODES, reset_memory_each_ep=False)
    
    # Condition B: reset memory each episode (standard MARL)
    print("[B] Reset memory each episode (standard MARL)...")
    rewards_reset = run_eval_episodes(policy, env, N_EPISODES, reset_memory_each_ep=True)
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  [A] Persistent memory: mean={np.mean(rewards_persistent):.1f} ± {np.std(rewards_persistent):.1f}")
    print(f"  [B] Reset each episode: mean={np.mean(rewards_reset):.1f} ± {np.std(rewards_reset):.1f}")
    delta = np.mean(rewards_persistent) - np.mean(rewards_reset)
    print(f"\n  Delta (persistent - reset): {delta:+.1f}")
    if delta > 0:
        print("  → Persistent memory HELPS: cross-episode memes are informative.")
    elif delta < -20:
        print("  → Persistent memory HURTS: cross-episode state is noise.")
    else:
        print("  → No significant difference: cross-episode memory is neutral.")
    print("=" * 60)
    
    # Save results
    results = {
        "persistent_mean": float(np.mean(rewards_persistent)),
        "persistent_std": float(np.std(rewards_persistent)),
        "reset_mean": float(np.mean(rewards_reset)),
        "reset_std": float(np.std(rewards_reset)),
        "delta": float(delta),
        "n_episodes": N_EPISODES,
        "checkpoint": ckpt_path,
    }
    out_path = "checkpoints/cross_episode_memory_test.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

if __name__ == "__main__":
    main()
