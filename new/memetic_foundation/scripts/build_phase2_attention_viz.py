#!/opt/homebrew/bin/python3.9
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from new.memetic_foundation.models.frozen_attention_hu_actor import (
    FrozenAttentionHUActorBackbone,
)
from new.memetic_foundation.modules.memetic_adapter import MemeticCommAdapter
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


def tensorize_obs(obs_list, device: torch.device) -> torch.Tensor:
    return torch.tensor(np.asarray(obs_list, dtype=np.float32), device=device)


def tensorize_avail(env, env_info: dict, device: torch.device) -> torch.Tensor:
    avail = np.zeros((env_info["n_agents"], env_info["n_actions"]), dtype=np.float32)
    for aid in range(env_info["n_agents"]):
        avail[aid] = np.asarray(env.get_avail_agent_actions(aid), dtype=np.float32)
    return torch.tensor(avail, device=device)


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


def load_adapter(backbone: FrozenAttentionHUActorBackbone, adapter_path: Path, device: torch.device) -> MemeticCommAdapter:
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


def episode_seed_list(seed: int, episodes: int) -> list[int]:
    rng = np.random.default_rng(10000 + int(seed))
    return rng.integers(low=0, high=2**31 - 1, size=episodes, dtype=np.int64).tolist()


def mean_row_entropy(attn: np.ndarray) -> float:
    return float((-(attn * np.log(np.clip(attn, 1e-8, 1.0))).sum(axis=1)).mean())


def uniform_offdiag_attention(n_agents: int) -> np.ndarray:
    baseline = np.zeros((n_agents, n_agents), dtype=np.float64)
    if n_agents <= 1:
        return baseline
    baseline[:] = 1.0 / (n_agents - 1)
    np.fill_diagonal(baseline, 0.0)
    return baseline


def serialize_example(example: dict[str, np.ndarray | float | int]) -> dict[str, object]:
    return {
        "episode_seed": int(example["episode_seed"]),
        "episode_index": int(example["episode_index"]),
        "step_index": int(example["step_index"]) if "step_index" in example else None,
        "mean_receiver_entropy": float(example["mean_receiver_entropy"]),
        "attn": np.asarray(example["attn"]).tolist(),
    }


def collect_attention_stats(
    env: VMASWrapper,
    backbone: FrozenAttentionHUActorBackbone,
    adapter: MemeticCommAdapter,
    episodes: int,
    episode_seeds: list[int],
    device: torch.device,
) -> dict[str, np.ndarray | float | list]:
    env_info = env.get_env_info()
    episode_mats = []
    episode_rewards = []
    episode_focus = []
    episode_sender_influence = []
    most_focused_episode = None
    most_focused_step = None

    for episode_index, ep_seed in enumerate(episode_seeds[:episodes]):
        env.reset(seed=int(ep_seed))
        backbone.reset_memory()
        z_state = adapter.initial_state(n_agents=backbone.n_agents, device=device)

        terminated = False
        ep_reward = 0.0
        ep_sum = np.zeros((backbone.n_agents, backbone.n_agents), dtype=np.float64)
        ep_steps = 0

        while not terminated:
            obs_t = tensorize_obs(env.get_obs(), device=device)
            avail_t = tensorize_avail(env, env_info=env_info, device=device)
            with torch.no_grad():
                out = backbone.step_with_adapter(
                    obs=obs_t,
                    avail_actions=avail_t,
                    adapter=adapter,
                    z=z_state,
                    deterministic=True,
                )
            z_state = out["z_next"]
            attn = out["comm_attn"].detach().cpu().numpy()
            step_entropy = mean_row_entropy(attn)
            if most_focused_step is None or step_entropy < float(most_focused_step["mean_receiver_entropy"]):
                most_focused_step = {
                    "episode_seed": int(ep_seed),
                    "episode_index": episode_index,
                    "step_index": ep_steps,
                    "mean_receiver_entropy": step_entropy,
                    "attn": attn.copy(),
                }
            ep_sum += attn
            ep_steps += 1
            actions = out["actions"].cpu().numpy().tolist()
            reward, terminated, _info = env.step(actions)
            ep_reward += float(reward)

        ep_mean = ep_sum / max(ep_steps, 1)
        episode_mats.append(ep_mean)
        episode_rewards.append(ep_reward)
        ep_entropy = mean_row_entropy(ep_mean)
        episode_focus.append(ep_entropy)
        episode_sender_influence.append(ep_mean.mean(axis=0))
        if most_focused_episode is None or ep_entropy < float(most_focused_episode["mean_receiver_entropy"]):
            most_focused_episode = {
                "episode_seed": int(ep_seed),
                "episode_index": episode_index,
                "mean_receiver_entropy": ep_entropy,
                "attn": ep_mean.copy(),
            }

    mean_attn = np.mean(np.stack(episode_mats, axis=0), axis=0)
    mean_sender_influence = np.mean(np.stack(episode_sender_influence, axis=0), axis=0)
    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "mean_attn": mean_attn,
        "mean_receiver_entropy": float(np.mean(episode_focus)),
        "mean_sender_influence": mean_sender_influence,
        "per_episode_rewards": episode_rewards,
        "most_focused_episode": most_focused_episode,
        "most_focused_step": most_focused_step,
    }


def render_mean_figure(
    es_mean: np.ndarray,
    rl_mean: np.ndarray,
    es_sender: np.ndarray,
    rl_sender: np.ndarray,
    es_entropy: float,
    rl_entropy: float,
    save_path: Path,
) -> None:
    diff = es_mean - rl_mean
    vmax = max(float(es_mean.max()), float(rl_mean.max()), 1e-6)
    dmax = max(abs(float(diff.min())), abs(float(diff.max())), 1e-6)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    im0 = axes[0, 0].imshow(es_mean, cmap="RdBu_r", vmin=0.0, vmax=vmax)
    axes[0, 0].set_title(f"ES Mean Attention\nEntropy={es_entropy:.3f}")
    axes[0, 0].set_xlabel("Sender")
    axes[0, 0].set_ylabel("Receiver")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(rl_mean, cmap="RdBu_r", vmin=0.0, vmax=vmax)
    axes[0, 1].set_title(f"RL Mean Attention\nEntropy={rl_entropy:.3f}")
    axes[0, 1].set_xlabel("Sender")
    axes[0, 1].set_ylabel("Receiver")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
    axes[1, 0].set_title("ES - RL Attention Difference")
    axes[1, 0].set_xlabel("Sender")
    axes[1, 0].set_ylabel("Receiver")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    x = np.arange(es_sender.shape[0])
    axes[1, 1].plot(x, es_sender, marker="o", label="ES")
    axes[1, 1].plot(x, rl_sender, marker="o", label="RL")
    axes[1, 1].set_title("Mean Sender Influence")
    axes[1, 1].set_xlabel("Agent Index")
    axes[1, 1].set_ylabel("Mean Attention Received")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.2)

    fig.suptitle("Phase-2 Communication Attention: ES vs RL\nVMAS Discovery, N=12, linear_a", fontsize=14)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def render_residual_figure(
    es_mean: np.ndarray,
    rl_mean: np.ndarray,
    es_sender: np.ndarray,
    rl_sender: np.ndarray,
    baseline: np.ndarray,
    save_path: Path,
) -> None:
    es_resid = es_mean - baseline
    rl_resid = rl_mean - baseline
    diff = es_mean - rl_mean
    rmax = max(
        abs(float(es_resid.min())),
        abs(float(es_resid.max())),
        abs(float(rl_resid.min())),
        abs(float(rl_resid.max())),
        abs(float(diff.min())),
        abs(float(diff.max())),
        1e-6,
    )
    baseline_sender = float(baseline.mean(axis=0)[0])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    im0 = axes[0, 0].imshow(es_resid, cmap="RdBu_r", vmin=-rmax, vmax=rmax)
    axes[0, 0].set_title("ES Mean Attention - Uniform")
    axes[0, 0].set_xlabel("Sender")
    axes[0, 0].set_ylabel("Receiver")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(rl_resid, cmap="RdBu_r", vmin=-rmax, vmax=rmax)
    axes[0, 1].set_title("RL Mean Attention - Uniform")
    axes[0, 1].set_xlabel("Sender")
    axes[0, 1].set_ylabel("Receiver")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(diff, cmap="RdBu_r", vmin=-rmax, vmax=rmax)
    axes[1, 0].set_title("ES - RL Attention Difference")
    axes[1, 0].set_xlabel("Sender")
    axes[1, 0].set_ylabel("Receiver")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    x = np.arange(es_sender.shape[0])
    axes[1, 1].plot(x, es_sender, marker="o", color="#b2182b", label="ES")
    axes[1, 1].plot(x, rl_sender, marker="o", color="#2166ac", label="RL")
    axes[1, 1].axhline(baseline_sender, color="0.5", linestyle="--", linewidth=1, label="Uniform baseline")
    axes[1, 1].set_title("Mean Sender Influence")
    axes[1, 1].set_xlabel("Agent Index")
    axes[1, 1].set_ylabel("Mean Attention Received")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.2)

    fig.suptitle(
        "Phase-2 Communication Attention Residuals\nUniform baseline = 1/(N-1) on off-diagonals",
        fontsize=14,
    )
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def render_sample_figure(
    es_episode: dict[str, np.ndarray | float | int],
    rl_episode: dict[str, np.ndarray | float | int],
    es_step: dict[str, np.ndarray | float | int],
    rl_step: dict[str, np.ndarray | float | int],
    baseline: np.ndarray,
    save_path: Path,
) -> None:
    es_episode_resid = np.asarray(es_episode["attn"]) - baseline
    rl_episode_resid = np.asarray(rl_episode["attn"]) - baseline
    es_step_resid = np.asarray(es_step["attn"]) - baseline
    rl_step_resid = np.asarray(rl_step["attn"]) - baseline
    vmax = max(
        abs(float(es_episode_resid.min())),
        abs(float(es_episode_resid.max())),
        abs(float(rl_episode_resid.min())),
        abs(float(rl_episode_resid.max())),
        abs(float(es_step_resid.min())),
        abs(float(es_step_resid.max())),
        abs(float(rl_step_resid.min())),
        abs(float(rl_step_resid.max())),
        1e-6,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    panels = [
        (
            axes[0, 0],
            es_episode_resid,
            f"ES Most Focused Episode Mean\nseed={int(es_episode['episode_seed'])} ep={int(es_episode['episode_index']) + 1} H={float(es_episode['mean_receiver_entropy']):.3f}",
        ),
        (
            axes[0, 1],
            rl_episode_resid,
            f"RL Most Focused Episode Mean\nseed={int(rl_episode['episode_seed'])} ep={int(rl_episode['episode_index']) + 1} H={float(rl_episode['mean_receiver_entropy']):.3f}",
        ),
        (
            axes[1, 0],
            es_step_resid,
            f"ES Most Focused Single Step\nseed={int(es_step['episode_seed'])} ep={int(es_step['episode_index']) + 1} step={int(es_step['step_index']) + 1} H={float(es_step['mean_receiver_entropy']):.3f}",
        ),
        (
            axes[1, 1],
            rl_step_resid,
            f"RL Most Focused Single Step\nseed={int(rl_step['episode_seed'])} ep={int(rl_step['episode_index']) + 1} step={int(rl_step['step_index']) + 1} H={float(rl_step['mean_receiver_entropy']):.3f}",
        ),
    ]

    for ax, mat, title in panels:
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Sender")
        ax.set_ylabel("Receiver")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Representative Communication Structure\nResiduals relative to uniform off-diagonal attention",
        fontsize=14,
    )
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    per_seed = {"es": {}, "rl": {}}
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
        es_mean=es_mean,
        rl_mean=rl_mean,
        es_sender=es_sender,
        rl_sender=rl_sender,
        es_entropy=es_entropy,
        rl_entropy=rl_entropy,
        save_path=args.save_dir / "attention_comparison.png",
    )
    render_residual_figure(
        es_mean=es_mean,
        rl_mean=rl_mean,
        es_sender=es_sender,
        rl_sender=rl_sender,
        baseline=baseline,
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
        es_episode=es_representative_episode,
        rl_episode=rl_representative_episode,
        es_step=es_representative_step,
        rl_step=rl_representative_step,
        baseline=baseline,
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
