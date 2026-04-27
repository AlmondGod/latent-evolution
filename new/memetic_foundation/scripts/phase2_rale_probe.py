#!/usr/bin/env python3
# Per-step latent probe for the RALE (reward-aligned latent evolution) analysis.
# Runs deterministic eval rollouts for one (method, seed) pair and logs
# z_t, z_t1, h_t, reward, episode metadata as a single .npz.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from new.memetic_foundation.models.frozen_attention_hu_actor import (
    FrozenAttentionHUActorBackbone,
)
from new.memetic_foundation.modules.memetic_adapter import MemeticCommAdapter
from new.memetic_foundation.training.rollout_utils import (
    make_smacv2_env,
    tensorize_avail,
    tensorize_obs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RALE latent probe")
    parser.add_argument("--backbone-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, default=None,
                        help="Adapter .pt path. Omit for the no-adapter base control.")
    parser.add_argument("--method", type=str, required=True,
                        help="Label for this run (e.g. alec, rl, base).")
    parser.add_argument("--save-path", type=Path, required=True,
                        help="Output .npz file path.")
    parser.add_argument("--env", choices=["smacv2"], default="smacv2",
                        help="Only smacv2 is supported by this probe.")
    parser.add_argument("--n-agents", type=int, required=True)
    parser.add_argument("--n-enemies", type=int, default=None)
    parser.add_argument("--race", choices=["terran", "protoss", "zerg"], default="terran")
    parser.add_argument("--episode-steps", type=int, default=200)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--z-dim", type=int, default=16)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--disable-z", action="store_true")
    parser.add_argument("--dense-state-update", action="store_true")
    parser.add_argument("--stochastic-actions", action="store_true")
    return parser.parse_args()


def load_adapter(
    backbone: FrozenAttentionHUActorBackbone,
    adapter_path: Path,
    device: torch.device,
) -> MemeticCommAdapter:
    payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    z_dim = int(cfg.get("z_dim", 16))
    rank = int(cfg.get("rank", 4))
    disable_z = bool(cfg.get("disable_z", False))
    dense_state_update = bool(cfg.get("dense_state_update", False))
    adapter = MemeticCommAdapter(
        h_dim=backbone.mem_dim,
        u_dim=backbone.enc_dim,
        z_dim=0 if disable_z else z_dim,
        attn_dim=backbone.attn_dim,
        rank=rank,
        use_z=not disable_z,
        dense_state_update=dense_state_update,
    ).to(device)
    theta = payload.get("theta")
    if theta is None:
        theta = payload.get("candidate_theta")
    if theta is None:
        raise RuntimeError(f"No theta found in adapter payload: {adapter_path}")
    adapter.load_genotype(theta.to(device=device, dtype=torch.float32))
    adapter.eval()
    return adapter


def run_episodes(
    env,
    backbone: FrozenAttentionHUActorBackbone,
    adapter: Optional[MemeticCommAdapter],
    episodes: int,
    deterministic: bool,
    device: torch.device,
) -> dict[str, np.ndarray]:
    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    h_dim = backbone.mem_dim
    z_dim = adapter.z_dim if adapter is not None and adapter.use_z else 0

    z_t_list: list[np.ndarray] = []
    z_t1_list: list[np.ndarray] = []
    h_t_list: list[np.ndarray] = []
    reward_list: list[float] = []
    episode_id_list: list[int] = []
    timestep_list: list[int] = []
    agent_id_list: list[int] = []
    done_list: list[int] = []
    ep_return_list: list[float] = []
    ep_win_list: list[int] = []

    for ep_idx in range(episodes):
        env.reset()
        backbone.reset_memory()
        z_state = adapter.initial_state(n_agents=n_agents, device=device) if adapter is not None else None

        ep_z_t: list[np.ndarray] = []
        ep_z_t1: list[np.ndarray] = []
        ep_h_t: list[np.ndarray] = []
        ep_reward: list[float] = []
        ep_t: list[int] = []
        ep_aid: list[int] = []
        terminated = False
        info: dict = {}
        ep_return = 0.0
        step_index = 0

        while not terminated:
            obs_t = tensorize_obs(env.get_obs(), device=device)
            avail_t = tensorize_avail(env, env_info=env_info, device=device)
            with torch.no_grad():
                out = backbone.step_with_adapter(
                    obs=obs_t,
                    avail_actions=avail_t,
                    adapter=adapter,
                    z=z_state,
                    deterministic=deterministic,
                )
            # Capture z_t (pre-step) and h_t (current memory).
            if z_state is not None:
                z_t_np = z_state.detach().cpu().numpy()
            else:
                z_t_np = np.zeros((n_agents, max(z_dim, 1)), dtype=np.float32)
            h_t_np = out["h"].detach().cpu().numpy()
            z_next = out["z_next"]
            if z_next is not None:
                z_t1_np = z_next.detach().cpu().numpy()
            else:
                z_t1_np = np.zeros_like(z_t_np)

            actions = out["actions"].detach().cpu().numpy().tolist()
            reward, terminated, info = env.step(actions)
            ep_return += float(reward)

            for aid in range(n_agents):
                ep_z_t.append(z_t_np[aid])
                ep_z_t1.append(z_t1_np[aid])
                ep_h_t.append(h_t_np[aid])
                ep_reward.append(float(reward))
                ep_t.append(step_index)
                ep_aid.append(aid)

            z_state = z_next.detach() if z_next is not None else None
            step_index += 1

        # Fill per-episode columns now that we know length and outcome.
        ep_len = len(ep_t)
        win_int = int(info.get("battle_won", -1)) if isinstance(info, dict) else -1

        # Mark the final timestep rows as terminal: no valid t+1 transition.
        max_t = step_index - 1
        for k in range(ep_len):
            done_list.append(1 if ep_t[k] == max_t else 0)

        z_t_list.append(np.asarray(ep_z_t, dtype=np.float32))
        z_t1_list.append(np.asarray(ep_z_t1, dtype=np.float32))
        h_t_list.append(np.asarray(ep_h_t, dtype=np.float32))
        reward_list.extend(ep_reward)
        timestep_list.extend(ep_t)
        agent_id_list.extend(ep_aid)
        episode_id_list.extend([ep_idx] * ep_len)
        ep_return_list.extend([ep_return] * ep_len)
        ep_win_list.extend([win_int] * ep_len)

    return {
        "z_t": np.concatenate(z_t_list, axis=0),
        "z_t1": np.concatenate(z_t1_list, axis=0),
        "h_t": np.concatenate(h_t_list, axis=0),
        "reward_t": np.asarray(reward_list, dtype=np.float32),
        "episode_id": np.asarray(episode_id_list, dtype=np.int32),
        "timestep": np.asarray(timestep_list, dtype=np.int32),
        "agent_id": np.asarray(agent_id_list, dtype=np.int32),
        "done_t": np.asarray(done_list, dtype=np.int8),
        "episode_return": np.asarray(ep_return_list, dtype=np.float32),
        "episode_win": np.asarray(ep_win_list, dtype=np.int8),
    }


def main() -> None:
    args = parse_args()
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env = make_smacv2_env(race=args.race, n_units=args.n_agents, n_enemies=args.n_enemies)
    env_info = env.get_env_info()
    device = torch.device(args.device)

    backbone = FrozenAttentionHUActorBackbone.from_checkpoint(
        checkpoint_path=str(args.backbone_path),
        obs_dim=env_info["obs_shape"],
        state_dim=env_info["state_shape"],
        n_actions=env_info["n_actions"],
        n_agents=env_info["n_agents"],
        map_location=str(device),
    ).to(device)
    backbone.freeze()

    adapter: Optional[MemeticCommAdapter] = None
    if args.adapter_path is not None:
        adapter = load_adapter(backbone, args.adapter_path, device=device)

    data = run_episodes(
        env=env,
        backbone=backbone,
        adapter=adapter,
        episodes=args.episodes,
        deterministic=not args.stochastic_actions,
        device=device,
    )

    meta = {
        "method": args.method,
        "seed": args.seed,
        "env": args.env,
        "n_agents": env_info["n_agents"],
        "z_dim": int(adapter.z_dim) if adapter is not None and adapter.use_z else 0,
        "h_dim": int(backbone.mem_dim),
        "episodes": args.episodes,
        "deterministic": not args.stochastic_actions,
        "backbone_path": str(args.backbone_path),
        "adapter_path": "" if args.adapter_path is None else str(args.adapter_path),
    }

    np.savez_compressed(
        args.save_path,
        meta=np.asarray(json.dumps(meta), dtype=object),
        **data,
    )
    print(json.dumps({
        "saved": str(args.save_path),
        "rows": int(data["z_t"].shape[0]),
        "episodes": int(args.episodes),
        "n_agents": int(env_info["n_agents"]),
        "z_dim": meta["z_dim"],
        "h_dim": meta["h_dim"],
        "method": args.method,
        "seed": args.seed,
    }), flush=True)
    env.close()


if __name__ == "__main__":
    main()
