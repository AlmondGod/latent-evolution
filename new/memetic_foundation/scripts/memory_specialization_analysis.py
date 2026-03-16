"""
memory_specialization_analysis.py

Analyzes whether agents develop *differentiated* memory states (specialization)
or converge to similar states (homogenization) after training.

Key metrics:
  - Mean pairwise cosine similarity between agent hidden states (lower = more specialized)
  - Memory norm trajectory over an episode (does it grow/shrink/stabilize?)
  - Memory delta norm (how much does memory change per step?)
  - Gate open fraction over episode (do agents learn when to communicate?)

Run against the best full_gated checkpoint and compare to memory_only.

Usage:
    python3.9 new/memetic_foundation/scripts/memory_specialization_analysis.py
"""

import sys, os, glob, torch, json
import numpy as np
import torch.nn.functional as F
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from new.memetic_foundation.training.mpe_wrapper import MPEWrapper
from new.memetic_foundation.models.agent_network import MemeticFoundationAC


def load_best_policy(base_dir, variant, use_gate=False, use_memory=True, use_comm=True):
    logs = sorted(glob.glob(f"{base_dir}/{variant}_seed*.log"))
    best_rew, best_seed = -1e9, 1
    import re
    for lf in logs:
        try:
            with open(lf, errors='replace') as f: content = f.read()
            m = re.findall(r'rew=([-\d.]+)', content)
            if m and float(m[-1]) > best_rew:
                best_rew = float(m[-1])
                best_seed = int(re.search(r'seed(\d+)', lf).group(1))
        except: pass
    
    ckpts = glob.glob(f"{base_dir}/{variant}_seed{best_seed}/*/*latest*.pt")
    if not ckpts:
        ckpts = glob.glob(f"{base_dir}/{variant}_seed{best_seed}/*/*.pt")
    if not ckpts:
        return None, best_rew
    
    ckpt = torch.load(ckpts[0], map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    
    env = MPEWrapper("simple_tag_v2", num_good=1, num_adversaries=3, max_cycles=100)
    info = env.get_env_info()
    
    policy = MemeticFoundationAC(
        obs_dim=info['obs_shape'], state_dim=info['state_shape'],
        n_actions=info['n_actions'], n_agents=info['n_agents'],
        hidden_dim=cfg.get('hidden_dim', 128),
        mem_dim=cfg.get('mem_dim', 128),
        comm_dim=cfg.get('comm_dim', 128),
        use_memory=use_memory, use_comm=use_comm, use_gate=use_gate,
    )
    policy.load_state_dict(ckpt['policy_state_dict'])
    policy.eval()
    env.close()
    return policy, best_rew


def analyze_episode(policy, env, n_steps=100):
    """Run one episode and track memory states over time."""
    policy.reset_memory()
    obs, _ = env.reset()
    obs_t = torch.FloatTensor(np.array(obs))
    
    mem_norms, mem_deltas, gate_fracs, cosine_sims = [], [], [], []
    h_prev = None
    
    for step in range(n_steps):
        with torch.no_grad():
            out = policy.forward_step(obs_t, deterministic=True)
        
        h = out.get('h')
        if h is not None:
            h_np = h.detach().numpy()
            # Memory norm
            mem_norms.append(float(np.linalg.norm(h_np)))
            # Memory delta from previous
            if h_prev is not None:
                mem_deltas.append(float(np.linalg.norm(h_np - h_prev)))
            h_prev = h_np.copy()
            # Pairwise cosine similarity between agent states
            N = h_np.shape[0]
            sims = []
            for i in range(N):
                for j in range(i+1, N):
                    hi = h_np[i] / (np.linalg.norm(h_np[i]) + 1e-8)
                    hj = h_np[j] / (np.linalg.norm(h_np[j]) + 1e-8)
                    sims.append(float(np.dot(hi, hj)))
            if sims:
                cosine_sims.append(np.mean(sims))
        
        # Gate open fraction
        if policy.use_gate and policy.comm is not None and policy.comm.gate is not None and h is not None:
            gate_val = policy.comm.gate(h.detach())
            gate_fracs.append(float(gate_val.mean().item()))
        
        actions = out["actions"].tolist()
        rew, done, _ = env.step(actions)
        if done:
            break
        obs_t = torch.FloatTensor(np.array(env.get_obs()))
        policy.detach_memory()
    
    return {
        "mem_norms": mem_norms,
        "mem_deltas": mem_deltas,
        "cosine_sims": cosine_sims,
        "gate_fracs": gate_fracs,
    }


def main():
    print("=" * 65)
    print("Memory Specialization Analysis")
    print("=" * 65)
    
    configs = [
        ("memory_only", "checkpoints/mpe_tag_gru",    False, True,  False),
        ("full_gated",  "checkpoints/mpe_tag_gated",  True,  True,  True),
    ]
    
    all_results = {}
    N_EPISODES = 10
    
    for name, base_dir, use_gate, use_memory, use_comm in configs:
        policy, best_rew = load_best_policy(base_dir, name, use_gate, use_memory, use_comm)
        if policy is None:
            print(f"  {name}: no checkpoint found")
            continue
        
        print(f"\n[{name}] (best seed rew={best_rew:.0f})")
        env = MPEWrapper("simple_tag_v2", num_good=1, num_adversaries=3, max_cycles=100)
        
        ep_results = [analyze_episode(policy, env) for _ in range(N_EPISODES)]
        
        def agg(key):
            vals = [v for ep in ep_results for v in ep[key]]
            return (np.mean(vals), np.std(vals)) if vals else (0, 0)
        
        mn_mean, mn_std = agg("mem_norms")
        md_mean, md_std = agg("mem_deltas")
        cs_mean, cs_std = agg("cosine_sims")
        gf_mean, gf_std = agg("gate_fracs")
        
        print(f"  Memory norm:           {mn_mean:.3f} ± {mn_std:.3f}")
        print(f"  Memory delta/step:     {md_mean:.3f} ± {md_std:.3f}  (how much memory changes)")
        print(f"  Agent cosine sim:      {cs_mean:.3f} ± {cs_std:.3f}  (1=identical, 0=orthogonal)")
        if gf_mean > 0:
            print(f"  Gate open fraction:    {gf_mean:.3f} ± {gf_std:.3f}  (1=always talking)")
        
        if cs_mean > 0.7:
            print(f"  → HIGH similarity: agents homogenize (no role specialization)")
        elif cs_mean < 0.3:
            print(f"  → LOW similarity: agents differentiate (role specialization!)")
        else:
            print(f"  → MODERATE similarity: partial specialization")
        
        all_results[name] = {
            "mem_norm_mean": float(mn_mean), "mem_norm_std": float(mn_std),
            "mem_delta_mean": float(md_mean), "mem_delta_std": float(md_std),
            "cosine_sim_mean": float(cs_mean), "cosine_sim_std": float(cs_std),
            "gate_open_mean": float(gf_mean),
            "best_seed_rew": float(best_rew),
        }
        env.close()
    
    out_path = "checkpoints/memory_specialization_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
