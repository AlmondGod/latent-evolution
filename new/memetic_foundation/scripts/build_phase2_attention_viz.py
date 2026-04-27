#!/opt/homebrew/bin/python3.9
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from new.memetic_foundation.models.frozen_attention_hu_actor import (
    FrozenAttentionHUActorBackbone,
)
from new.memetic_foundation.scripts.attention_viz_utils import (
    collect_attention_stats,
    episode_seed_list,
    load_adapter,
    render_mean_figure,
    render_residual_figure,
    render_sample_figure,
    serialize_example,
    uniform_offdiag_attention,
)
from new.memetic_foundation.training.vmas_wrapper import VMASWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ES-vs-RL communication attention visualizations from phase-2 adapters."
    )
    parser.add_argument(
        "--backbone-root",
        type=Path,
        default=Path(
            "/Users/almondgod/Repositories/memeplex-capstone/new/memetic_foundation/checkpoints/vmas_phase1_discovery_50k_n12/discovery_n12/attention_hu_actor"
        ),
    )
    parser.add_argument(
        "--es-root",
        type=Path,
        default=Path(
            "/Users/almondgod/Repositories/memeplex-capstone/results/vmas_phase2_linear_a_discovery_50k_n12/linear_a/n12/es"
        ),
    )
    parser.add_argument(
        "--rl-root",
        type=Path,
        default=Path(
            "/Users/almondgod/Repositories/memeplex-capstone/results/vmas_phase2_linear_a_discovery_50k_n12/linear_a/n12/rl"
        ),
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(
            "/Users/almondgod/Repositories/memeplex-capstone/results/vmas_phase2_attention_linear_a_discovery_n12"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--n-agents", type=int, default=12)
    parser.add_argument("--episode-steps", type=int, default=200)
    parser.add_argument("--vmas-targets", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def make_env(n_agents: int, episode_steps: int, vmas_targets: int, device: str) -> VMASWrapper:
    return VMASWrapper(
        scenario_name="discovery",
        n_agents=n_agents,
        max_steps=episode_steps,
        device=device,
        n_targets=vmas_targets,
        agents_per_target=1,
        targets_respawn=False,
        shared_reward=True,
    )


def build_backbone(env: VMASWrapper, checkpoint_path: Path, device: torch.device) -> FrozenAttentionHUActorBackbone:
    env_info = env.get_env_info()
    backbone = FrozenAttentionHUActorBackbone.from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        obs_dim=env_info["obs_shape"],
        state_dim=env_info["state_shape"],
        n_actions=env_info["n_actions"],
        n_agents=env_info["n_agents"],
        map_location=str(device),
    )
    return backbone.to(device).freeze()


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    per_seed: dict[str, dict[str, dict]] = {"es": {}, "rl": {}}
    for seed in args.seeds:
        env = make_env(
            n_agents=args.n_agents,
            episode_steps=args.episode_steps,
            vmas_targets=args.vmas_targets,
            device=args.device,
        )
        backbone = build_backbone(
            env=env,
            checkpoint_path=args.backbone_root / f"seed{seed}" / "memfound_full_attention_hu_actor_best.pt",
            device=device,
        )
        es_adapter = load_adapter(backbone, args.es_root / f"seed{seed}" / "best_adapter.pt", device=device)
        rl_adapter = load_adapter(backbone, args.rl_root / f"seed{seed}" / "best_adapter.pt", device=device)

        seeds = episode_seed_list(seed=seed, episodes=args.episodes)
        per_seed["es"][str(seed)] = collect_attention_stats(
            env=env,
            backbone=backbone,
            adapter=es_adapter,
            episodes=args.episodes,
            episode_seeds=seeds,
            device=device,
        )
        per_seed["rl"][str(seed)] = collect_attention_stats(
            env=env,
            backbone=backbone,
            adapter=rl_adapter,
            episodes=args.episodes,
            episode_seeds=seeds,
            device=device,
        )
        env.close()

    es_mean = np.mean([per_seed["es"][str(s)]["mean_attn"] for s in args.seeds], axis=0)
    rl_mean = np.mean([per_seed["rl"][str(s)]["mean_attn"] for s in args.seeds], axis=0)
    es_sender = np.mean([per_seed["es"][str(s)]["mean_sender_influence"] for s in args.seeds], axis=0)
    rl_sender = np.mean([per_seed["rl"][str(s)]["mean_sender_influence"] for s in args.seeds], axis=0)
    es_entropy = float(np.mean([per_seed["es"][str(s)]["mean_receiver_entropy"] for s in args.seeds]))
    rl_entropy = float(np.mean([per_seed["rl"][str(s)]["mean_receiver_entropy"] for s in args.seeds]))
    es_reward = float(np.mean([per_seed["es"][str(s)]["mean_reward"] for s in args.seeds]))
    rl_reward = float(np.mean([per_seed["rl"][str(s)]["mean_reward"] for s in args.seeds]))
    baseline = uniform_offdiag_attention(args.n_agents)

    render_mean_figure(
        a_mean=es_mean,
        b_mean=rl_mean,
        a_sender=es_sender,
        b_sender=rl_sender,
        a_entropy=es_entropy,
        b_entropy=rl_entropy,
        label_a="ES",
        label_b="RL",
        suptitle="Phase-2 Communication Attention: ES vs RL\nVMAS Discovery, N=12, linear_a",
        save_path=args.save_dir / "attention_comparison.png",
    )
    render_residual_figure(
        a_mean=es_mean,
        b_mean=rl_mean,
        a_sender=es_sender,
        b_sender=rl_sender,
        baseline=baseline,
        label_a="ES",
        label_b="RL",
        suptitle="Phase-2 Communication Attention Residuals\nUniform baseline = 1/(N-1) on off-diagonals",
        save_path=args.save_dir / "attention_residual_comparison.png",
    )
    es_representative_episode = min(
        [per_seed["es"][str(s)]["most_focused_episode"] for s in args.seeds],
        key=lambda item: float(item["mean_receiver_entropy"]),
    )
    rl_representative_episode = min(
        [per_seed["rl"][str(s)]["most_focused_episode"] for s in args.seeds],
        key=lambda item: float(item["mean_receiver_entropy"]),
    )
    es_representative_step = min(
        [per_seed["es"][str(s)]["most_focused_step"] for s in args.seeds],
        key=lambda item: float(item["mean_receiver_entropy"]),
    )
    rl_representative_step = min(
        [per_seed["rl"][str(s)]["most_focused_step"] for s in args.seeds],
        key=lambda item: float(item["mean_receiver_entropy"]),
    )
    render_sample_figure(
        a_episode=es_representative_episode,
        b_episode=rl_representative_episode,
        a_step=es_representative_step,
        b_step=rl_representative_step,
        baseline=baseline,
        label_a="ES",
        label_b="RL",
        suptitle="Representative Communication Structure\nResiduals relative to uniform off-diagonal attention",
        save_path=args.save_dir / "attention_sample_diagnostics.png",
    )

    summary = {
        "episodes_per_seed": args.episodes,
        "seeds": args.seeds,
        "uniform_offdiag_baseline": baseline.tolist(),
        "es_mean_reward": es_reward,
        "rl_mean_reward": rl_reward,
        "es_mean_receiver_entropy": es_entropy,
        "rl_mean_receiver_entropy": rl_entropy,
        "es_mean_attn": es_mean.tolist(),
        "rl_mean_attn": rl_mean.tolist(),
        "es_mean_sender_influence": es_sender.tolist(),
        "rl_mean_sender_influence": rl_sender.tolist(),
        "es_representative_episode": serialize_example(es_representative_episode),
        "rl_representative_episode": serialize_example(rl_representative_episode),
        "es_representative_step": serialize_example(es_representative_step),
        "rl_representative_step": serialize_example(rl_representative_step),
        "per_seed": {
            method: {
                seed: {
                    "mean_reward": float(stats["mean_reward"]),
                    "mean_receiver_entropy": float(stats["mean_receiver_entropy"]),
                    "mean_attn": np.asarray(stats["mean_attn"]).tolist(),
                    "mean_sender_influence": np.asarray(stats["mean_sender_influence"]).tolist(),
                    "most_focused_episode": serialize_example(stats["most_focused_episode"]),
                    "most_focused_step": serialize_example(stats["most_focused_step"]),
                }
                for seed, stats in seed_map.items()
            }
            for method, seed_map in per_seed.items()
        },
    }
    (args.save_dir / "attention_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                "saved_plot": str(args.save_dir / "attention_comparison.png"),
                "saved_residual_plot": str(args.save_dir / "attention_residual_comparison.png"),
                "saved_sample_plot": str(args.save_dir / "attention_sample_diagnostics.png"),
                "saved_summary": str(args.save_dir / "attention_summary.json"),
                "es_mean_reward": es_reward,
                "rl_mean_reward": rl_reward,
                "es_mean_receiver_entropy": es_entropy,
                "rl_mean_receiver_entropy": rl_entropy,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
