from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from new.memetic_foundation.models.frozen_attention_hu_actor import (
    FrozenAttentionHUActorBackbone,
)
from new.memetic_foundation.modules.memetic_adapter import MemeticCommAdapter
from new.memetic_foundation.training.env_reset_utils import reset_env


def tensorize_obs(obs_list, device: torch.device) -> torch.Tensor:
    return torch.tensor(np.asarray(obs_list, dtype=np.float32), device=device)


def tensorize_avail(env, env_info: dict, device: torch.device) -> torch.Tensor:
    avail = np.zeros((env_info["n_agents"], env_info["n_actions"]), dtype=np.float32)
    for aid in range(env_info["n_agents"]):
        avail[aid] = np.asarray(env.get_avail_agent_actions(aid), dtype=np.float32)
    return torch.tensor(avail, device=device)


def mean_row_entropy(attn: np.ndarray) -> float:
    return float((-(attn * np.log(np.clip(attn, 1e-8, 1.0))).sum(axis=1)).mean())


def uniform_offdiag_attention(n_agents: int) -> np.ndarray:
    baseline = np.zeros((n_agents, n_agents), dtype=np.float64)
    if n_agents <= 1:
        return baseline
    baseline[:] = 1.0 / (n_agents - 1)
    np.fill_diagonal(baseline, 0.0)
    return baseline


def smac_role_ids_from_env(env, env_info: dict) -> np.ndarray:
    n_agents = int(env_info["n_agents"])
    cap_dim = int(env_info["cap_shape"])
    cap = np.asarray(env.get_capabilities(), dtype=np.float32).reshape(n_agents, cap_dim)
    return np.argmax(cap, axis=1).astype(np.int64)


def pool_attn_to_role_matrix(attn: np.ndarray, roles: np.ndarray, n_roles: int) -> np.ndarray:
    n = int(attn.shape[0])
    out = np.zeros((n_roles, n_roles), dtype=np.float64)
    for tr in range(n_roles):
        receivers = [i for i in range(n) if int(roles[i]) == tr]
        if not receivers:
            continue
        block = np.zeros((len(receivers), n_roles), dtype=np.float64)
        for ri, i in enumerate(receivers):
            for ts in range(n_roles):
                mass = 0.0
                for j in range(n):
                    if j == i:
                        continue
                    if int(roles[j]) == ts:
                        mass += float(attn[i, j])
                block[ri, ts] = mass
        out[tr] = block.mean(axis=0)
    return out


def uniform_role_attention(n_roles: int) -> np.ndarray:
    v = np.full((n_roles, n_roles), 1.0 / float(n_roles), dtype=np.float64)
    return v


def serialize_example(example: dict) -> dict:
    return {
        "episode_seed": int(example["episode_seed"]),
        "episode_index": int(example["episode_index"]),
        "step_index": int(example["step_index"]) if example.get("step_index") is not None else None,
        "mean_receiver_entropy": float(example["mean_receiver_entropy"]),
        "attn": np.asarray(example["attn"]).tolist(),
    }


def episode_seed_list(seed: int, episodes: int) -> list[int]:
    rng = np.random.default_rng(10000 + int(seed))
    return rng.integers(low=0, high=2**31 - 1, size=episodes, dtype=np.int64).tolist()


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


def collect_attention_stats(
    env,
    backbone: FrozenAttentionHUActorBackbone,
    adapter: MemeticCommAdapter,
    episodes: int,
    episode_seeds: list[int],
    device: torch.device,
) -> dict:
    env_info = env.get_env_info()
    episode_mats = []
    episode_rewards = []
    episode_focus = []
    episode_sender_influence = []
    most_focused_episode: Optional[dict] = None
    most_focused_step: Optional[dict] = None

    for episode_index, ep_seed in enumerate(episode_seeds[:episodes]):
        reset_env(env, seed=int(ep_seed))
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


def collect_attention_stats_by_smac_role(
    env,
    backbone: FrozenAttentionHUActorBackbone,
    adapter: MemeticCommAdapter,
    episodes: int,
    episode_seeds: list[int],
    device: torch.device,
) -> dict:
    env_info = env.get_env_info()
    n_roles = int(env_info["cap_shape"])
    episode_mats = []
    episode_rewards = []
    episode_focus = []
    episode_sender_influence = []
    most_focused_episode: Optional[dict] = None
    most_focused_step: Optional[dict] = None

    for episode_index, ep_seed in enumerate(episode_seeds[:episodes]):
        reset_env(env, seed=int(ep_seed))
        backbone.reset_memory()
        z_state = adapter.initial_state(n_agents=backbone.n_agents, device=device)

        terminated = False
        ep_reward = 0.0
        ep_sum = np.zeros((n_roles, n_roles), dtype=np.float64)
        ep_steps = 0

        while not terminated:
            roles = smac_role_ids_from_env(env, env_info)
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
            role_attn = pool_attn_to_role_matrix(attn, roles, n_roles)
            step_entropy = mean_row_entropy(role_attn)
            if most_focused_step is None or step_entropy < float(most_focused_step["mean_receiver_entropy"]):
                most_focused_step = {
                    "episode_seed": int(ep_seed),
                    "episode_index": episode_index,
                    "step_index": ep_steps,
                    "mean_receiver_entropy": step_entropy,
                    "attn": role_attn.copy(),
                }
            ep_sum += role_attn
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
        "n_roles": n_roles,
    }


def render_mean_figure(
    a_mean: np.ndarray,
    b_mean: np.ndarray,
    a_sender: np.ndarray,
    b_sender: np.ndarray,
    a_entropy: float,
    b_entropy: float,
    label_a: str,
    label_b: str,
    suptitle: str,
    save_path: Path,
    axis_labels: Optional[list[str]] = None,
    influence_xlabel: str = "Agent Index",
) -> None:
    diff = a_mean - b_mean
    vmax = max(float(a_mean.max()), float(b_mean.max()), 1e-6)
    dmax = max(abs(float(diff.min())), abs(float(diff.max())), 1e-6)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    im0 = axes[0, 0].imshow(a_mean, cmap="RdBu_r", vmin=0.0, vmax=vmax)
    axes[0, 0].set_title(f"{label_a} Mean Attention\nEntropy={a_entropy:.3f}")
    axes[0, 0].set_xlabel("Sender role" if axis_labels else "Sender")
    axes[0, 0].set_ylabel("Receiver role" if axis_labels else "Receiver")
    if axis_labels is not None:
        axes[0, 0].set_xticks(range(len(axis_labels)))
        axes[0, 0].set_xticklabels(axis_labels, rotation=35, ha="right")
        axes[0, 0].set_yticks(range(len(axis_labels)))
        axes[0, 0].set_yticklabels(axis_labels)
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(b_mean, cmap="RdBu_r", vmin=0.0, vmax=vmax)
    axes[0, 1].set_title(f"{label_b} Mean Attention\nEntropy={b_entropy:.3f}")
    axes[0, 1].set_xlabel("Sender role" if axis_labels else "Sender")
    axes[0, 1].set_ylabel("Receiver role" if axis_labels else "Receiver")
    if axis_labels is not None:
        axes[0, 1].set_xticks(range(len(axis_labels)))
        axes[0, 1].set_xticklabels(axis_labels, rotation=35, ha="right")
        axes[0, 1].set_yticks(range(len(axis_labels)))
        axes[0, 1].set_yticklabels(axis_labels)
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
    axes[1, 0].set_title(f"{label_a} - {label_b} Attention Difference")
    axes[1, 0].set_xlabel("Sender role" if axis_labels else "Sender")
    axes[1, 0].set_ylabel("Receiver role" if axis_labels else "Receiver")
    if axis_labels is not None:
        axes[1, 0].set_xticks(range(len(axis_labels)))
        axes[1, 0].set_xticklabels(axis_labels, rotation=35, ha="right")
        axes[1, 0].set_yticks(range(len(axis_labels)))
        axes[1, 0].set_yticklabels(axis_labels)
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    x = np.arange(a_sender.shape[0])
    axes[1, 1].plot(x, a_sender, marker="o", color="#b2182b", label=label_a)
    axes[1, 1].plot(x, b_sender, marker="o", color="#2166ac", label=label_b)
    axes[1, 1].set_title("Mean Sender Influence")
    axes[1, 1].set_xlabel(influence_xlabel)
    axes[1, 1].set_ylabel("Mean Attention Received")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.2)

    fig.suptitle(suptitle, fontsize=14)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def render_residual_figure(
    a_mean: np.ndarray,
    b_mean: np.ndarray,
    a_sender: np.ndarray,
    b_sender: np.ndarray,
    baseline: np.ndarray,
    label_a: str,
    label_b: str,
    suptitle: str,
    save_path: Path,
    axis_labels: Optional[list[str]] = None,
    influence_xlabel: str = "Agent Index",
) -> None:
    a_resid = a_mean - baseline
    b_resid = b_mean - baseline
    diff = a_mean - b_mean
    rmax = max(
        abs(float(a_resid.min())),
        abs(float(a_resid.max())),
        abs(float(b_resid.min())),
        abs(float(b_resid.max())),
        abs(float(diff.min())),
        abs(float(diff.max())),
        1e-6,
    )
    baseline_sender = float(baseline.mean(axis=0)[0]) if baseline.shape[0] > 0 else 0.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    im0 = axes[0, 0].imshow(a_resid, cmap="RdBu_r", vmin=-rmax, vmax=rmax)
    axes[0, 0].set_title(f"{label_a} Mean Attention - Uniform")
    axes[0, 0].set_xlabel("Sender role" if axis_labels else "Sender")
    axes[0, 0].set_ylabel("Receiver role" if axis_labels else "Receiver")
    if axis_labels is not None:
        axes[0, 0].set_xticks(range(len(axis_labels)))
        axes[0, 0].set_xticklabels(axis_labels, rotation=35, ha="right")
        axes[0, 0].set_yticks(range(len(axis_labels)))
        axes[0, 0].set_yticklabels(axis_labels)
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(b_resid, cmap="RdBu_r", vmin=-rmax, vmax=rmax)
    axes[0, 1].set_title(f"{label_b} Mean Attention - Uniform")
    axes[0, 1].set_xlabel("Sender role" if axis_labels else "Sender")
    axes[0, 1].set_ylabel("Receiver role" if axis_labels else "Receiver")
    if axis_labels is not None:
        axes[0, 1].set_xticks(range(len(axis_labels)))
        axes[0, 1].set_xticklabels(axis_labels, rotation=35, ha="right")
        axes[0, 1].set_yticks(range(len(axis_labels)))
        axes[0, 1].set_yticklabels(axis_labels)
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(diff, cmap="RdBu_r", vmin=-rmax, vmax=rmax)
    axes[1, 0].set_title(f"{label_a} - {label_b} Attention Difference")
    axes[1, 0].set_xlabel("Sender role" if axis_labels else "Sender")
    axes[1, 0].set_ylabel("Receiver role" if axis_labels else "Receiver")
    if axis_labels is not None:
        axes[1, 0].set_xticks(range(len(axis_labels)))
        axes[1, 0].set_xticklabels(axis_labels, rotation=35, ha="right")
        axes[1, 0].set_yticks(range(len(axis_labels)))
        axes[1, 0].set_yticklabels(axis_labels)
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    x = np.arange(a_sender.shape[0])
    axes[1, 1].plot(x, a_sender, marker="o", color="#b2182b", label=label_a)
    axes[1, 1].plot(x, b_sender, marker="o", color="#2166ac", label=label_b)
    axes[1, 1].axhline(baseline_sender, color="0.5", linestyle="--", linewidth=1, label="Uniform baseline")
    axes[1, 1].set_title("Mean Sender Influence")
    axes[1, 1].set_xlabel(influence_xlabel)
    axes[1, 1].set_ylabel("Mean Attention Received")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.2)

    fig.suptitle(suptitle, fontsize=14)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def render_sample_figure(
    a_episode: dict,
    b_episode: dict,
    a_step: dict,
    b_step: dict,
    baseline: np.ndarray,
    label_a: str,
    label_b: str,
    suptitle: str,
    save_path: Path,
    axis_labels: Optional[list[str]] = None,
) -> None:
    a_episode_resid = np.asarray(a_episode["attn"]) - baseline
    b_episode_resid = np.asarray(b_episode["attn"]) - baseline
    a_step_resid = np.asarray(a_step["attn"]) - baseline
    b_step_resid = np.asarray(b_step["attn"]) - baseline
    vmax = max(
        abs(float(a_episode_resid.min())),
        abs(float(a_episode_resid.max())),
        abs(float(b_episode_resid.min())),
        abs(float(b_episode_resid.max())),
        abs(float(a_step_resid.min())),
        abs(float(a_step_resid.max())),
        abs(float(b_step_resid.min())),
        abs(float(b_step_resid.max())),
        1e-6,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    panels = [
        (
            axes[0, 0],
            a_episode_resid,
            f"{label_a} Most Focused Episode Mean\nseed={int(a_episode['episode_seed'])} ep={int(a_episode['episode_index']) + 1} H={float(a_episode['mean_receiver_entropy']):.3f}",
        ),
        (
            axes[0, 1],
            b_episode_resid,
            f"{label_b} Most Focused Episode Mean\nseed={int(b_episode['episode_seed'])} ep={int(b_episode['episode_index']) + 1} H={float(b_episode['mean_receiver_entropy']):.3f}",
        ),
        (
            axes[1, 0],
            a_step_resid,
            f"{label_a} Most Focused Single Step\nseed={int(a_step['episode_seed'])} ep={int(a_step['episode_index']) + 1} step={int(a_step['step_index']) + 1} H={float(a_step['mean_receiver_entropy']):.3f}",
        ),
        (
            axes[1, 1],
            b_step_resid,
            f"{label_b} Most Focused Single Step\nseed={int(b_step['episode_seed'])} ep={int(b_step['episode_index']) + 1} step={int(b_step['step_index']) + 1} H={float(b_step['mean_receiver_entropy']):.3f}",
        ),
    ]

    for ax, mat, title in panels:
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Sender role" if axis_labels else "Sender")
        ax.set_ylabel("Receiver role" if axis_labels else "Receiver")
        if axis_labels is not None:
            ax.set_xticks(range(len(axis_labels)))
            ax.set_xticklabels(axis_labels, rotation=35, ha="right")
            ax.set_yticks(range(len(axis_labels)))
            ax.set_yticklabels(axis_labels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(suptitle, fontsize=14)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
