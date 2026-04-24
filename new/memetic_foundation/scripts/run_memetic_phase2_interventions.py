#!/opt/homebrew/bin/python3.9
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
from new.memetic_foundation.scripts.run_memetic_selection_phase2 import (
    create_phase2_env,
    evaluate_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Posthoc baseline/silence/shift evaluation for Phase-2 adapters"
    )
    parser.add_argument("--load-path", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--env", choices=["smacv2", "mpe", "lbf", "rware", "vmas"], default="mpe")
    parser.add_argument("--mpe-scenario", type=str, default="simple_spread_v2")
    parser.add_argument("--vmas-scenario", choices=["transport", "discovery"], default="transport")
    parser.add_argument("--n-agents", type=int, required=True)
    parser.add_argument("--n-enemies", type=int, default=None)
    parser.add_argument("--race", choices=["terran", "protoss", "zerg"], default="terran")
    parser.add_argument("--obs-radius", type=float, default=0.5)
    parser.add_argument("--episode-steps", type=int, default=200)
    parser.add_argument("--lbf-size", type=int, default=6)
    parser.add_argument("--lbf-foods", type=int, default=2)
    parser.add_argument("--lbf-sight", type=int, default=2)
    parser.add_argument("--lbf-max-steps", type=int, default=50)
    parser.add_argument("--lbf-no-coop", action="store_true")
    parser.add_argument("--rware-rows", type=int, default=2)
    parser.add_argument("--rware-cols", type=int, default=3)
    parser.add_argument("--rware-height", type=int, default=8)
    parser.add_argument("--rware-requests", type=int, default=2)
    parser.add_argument("--rware-sensor-range", type=int, default=1)
    parser.add_argument("--vmas-packages", type=int, default=1)
    parser.add_argument("--vmas-targets", type=int, default=4)
    parser.add_argument("--vmas-agents-per-target", type=int, default=1)
    parser.add_argument("--vmas-targets-respawn", action="store_true")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=128)
    parser.add_argument("--z-dim", type=int, default=16)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--persistent-z", action="store_true")
    parser.add_argument("--persistent-memory", action="store_true")
    return parser.parse_args()


def load_adapter(
    adapter_path: Optional[Path],
    backbone: FrozenAttentionHUActorBackbone,
    device: torch.device,
    z_dim: int,
    rank: int,
) -> Optional[MemeticCommAdapter]:
    if adapter_path is None:
        return None
    payload = torch.load(adapter_path, map_location=str(device))
    theta = payload.get("theta")
    if theta is None:
        return None
    cfg = payload.get("config", {})
    adapter = MemeticCommAdapter(
        h_dim=backbone.mem_dim,
        u_dim=backbone.enc_dim,
        z_dim=int(cfg.get("z_dim", z_dim)),
        attn_dim=backbone.attn_dim,
        rank=int(cfg.get("rank", rank)),
    ).to(device)
    adapter.load_genotype(theta.to(device=device, dtype=torch.float32))
    return adapter


def main() -> None:
    args = parse_args()
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env = create_phase2_env(args)
    env_info = env.get_env_info()
    device = torch.device(args.device)

    backbone = FrozenAttentionHUActorBackbone.from_checkpoint(
        checkpoint_path=str(args.load_path),
        obs_dim=env_info["obs_shape"],
        state_dim=env_info["state_shape"],
        n_actions=env_info["n_actions"],
        n_agents=env_info["n_agents"],
        map_location=str(device),
    ).to(device)
    backbone.freeze()
    backbone.reset_memory()

    adapter = load_adapter(args.adapter_path, backbone, device, args.z_dim, args.rank)

    rng = np.random.RandomState(args.seed)
    episode_seeds = [int(rng.randint(0, 2**31 - 1)) for _ in range(args.eval_episodes)]

    baseline = evaluate_candidate(
        env=env,
        backbone=backbone,
        adapter=adapter,
        episodes=args.eval_episodes,
        deterministic=True,
        persistent_z=args.persistent_z,
        persistent_memory=args.persistent_memory,
        episode_seeds=episode_seeds,
    )
    silence = evaluate_candidate(
        env=env,
        backbone=backbone,
        adapter=adapter,
        episodes=args.eval_episodes,
        deterministic=True,
        persistent_z=args.persistent_z,
        persistent_memory=args.persistent_memory,
        intervene_comm_silence=True,
        episode_seeds=episode_seeds,
    )
    shift = evaluate_candidate(
        env=env,
        backbone=backbone,
        adapter=adapter,
        episodes=args.eval_episodes,
        deterministic=True,
        persistent_z=args.persistent_z,
        persistent_memory=args.persistent_memory,
        intervene_comm_shift=True,
        episode_seeds=episode_seeds,
    )

    result = {
        "label": args.label,
        "checkpoint": str(args.load_path),
        "adapter_path": None if args.adapter_path is None else str(args.adapter_path),
        "seed": args.seed,
        "eval_episodes": args.eval_episodes,
        "baseline": baseline,
        "silence": silence,
        "shift": shift,
        "delta_reward_silence": float(silence["mean_reward"] - baseline["mean_reward"]),
        "delta_reward_shift": float(shift["mean_reward"] - baseline["mean_reward"]),
        "delta_min_dist_silence": (
            float(silence["mean_min_dist"] - baseline["mean_min_dist"])
            if "mean_min_dist" in baseline and "mean_min_dist" in silence
            else None
        ),
        "delta_min_dist_shift": (
            float(shift["mean_min_dist"] - baseline["mean_min_dist"])
            if "mean_min_dist" in baseline and "mean_min_dist" in shift
            else None
        ),
    }
    args.save_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    env.close()


if __name__ == "__main__":
    main()
