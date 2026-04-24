#!/opt/homebrew/bin/python3.9
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from new.memetic_foundation.models.frozen_attention_hu_actor import (
    FrozenAttentionHUActorBackbone,
)
from new.memetic_foundation.modules.memetic_adapter import MemeticCommAdapter
from new.memetic_foundation.training.vmas_wrapper import VMASWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe whether phase-2 communication is informative or effectively uniform."
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
    parser.add_argument("--save-name", type=str, default="comm_probe_summary.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--n-agents", type=int, default=12)
    parser.add_argument("--episode-steps", type=int, default=200)
    parser.add_argument("--vmas-targets", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["base", "es", "rl"],
        choices=["base", "es", "rl"],
    )
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


def uniform_offdiag_attention(n_agents: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    baseline = torch.zeros((n_agents, n_agents), device=device, dtype=dtype)
    if n_agents > 1:
        baseline.fill_(1.0 / float(n_agents - 1))
        baseline.fill_diagonal_(0.0)
    return baseline


def mean_policy_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    logp = torch.log_softmax(logits_p, dim=-1)
    logq = torch.log_softmax(logits_q, dim=-1)
    p = logp.exp()
    return float((p * (logp - logq)).sum(dim=-1).mean().item())


def row_offdiag_std(raw_scores: torch.Tensor) -> float:
    n_agents = raw_scores.shape[0]
    eye = torch.eye(n_agents, device=raw_scores.device, dtype=torch.bool)
    zeroed = raw_scores.masked_fill(eye, 0.0)
    denom = max(n_agents - 1, 1)
    row_mean = zeroed.sum(dim=-1) / float(denom)
    centered = (zeroed - row_mean.unsqueeze(-1)).masked_fill(eye, 0.0)
    row_var = centered.pow(2).sum(dim=-1) / float(denom)
    return float(row_var.sqrt().mean().item())


def pairwise_cosine_summary(x: torch.Tensor) -> dict[str, float]:
    n_agents = x.shape[0]
    if n_agents <= 1:
        return {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0}
    x_norm = torch.nn.functional.normalize(x, dim=-1)
    sim = torch.matmul(x_norm, x_norm.transpose(-2, -1))
    mask = ~torch.eye(n_agents, device=x.device, dtype=torch.bool)
    vals = sim[mask]
    return {
        "mean": float(vals.mean().item()),
        "std": float(vals.std(unbiased=False).item()),
        "min": float(vals.min().item()),
        "max": float(vals.max().item()),
    }


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def serialize_event(event: Optional[dict]) -> Optional[dict]:
    if event is None:
        return None
    out = dict(event)
    if "attn" in out:
        out["attn"] = np.asarray(out["attn"]).tolist()
    if "raw_scores" in out:
        out["raw_scores"] = np.asarray(out["raw_scores"]).tolist()
    return out


def compute_step_metrics(
    backbone: FrozenAttentionHUActorBackbone,
    adapter: Optional[MemeticCommAdapter],
    out: dict[str, torch.Tensor],
    avail_actions: torch.Tensor,
    temperatures: list[float],
) -> dict[str, object]:
    n_agents = out["comm_attn"].shape[0]
    device = out["comm_attn"].device
    baseline = uniform_offdiag_attention(n_agents=n_agents, device=device, dtype=out["comm_attn"].dtype)
    raw_scores = out["raw_scores"]
    attn = out["comm_attn"]
    eye = torch.eye(n_agents, device=device, dtype=torch.bool)

    row_max = raw_scores.masked_fill(eye, float("-inf")).max(dim=-1).values
    row_min = raw_scores.masked_fill(eye, float("inf")).min(dim=-1).values
    entropy = -(attn * torch.log(attn.clamp_min(1e-8))).sum(dim=-1)
    top1 = attn.max(dim=-1).values
    kl_to_uniform = math.log(max(n_agents - 1, 1)) - entropy
    l1_to_uniform = (attn - baseline).abs().sum(dim=-1)

    silence_c = torch.zeros_like(out["c_unscaled"])
    logits_silence = backbone.actor_logits_from_parts(
        u=out["u"],
        h=out["h"],
        c=silence_c,
        avail_actions=avail_actions,
    )

    uniform_latent = torch.matmul(baseline, out["v"])
    uniform_c = backbone.attn_comm_hu.out_proj(uniform_latent)
    if adapter is not None:
        uniform_c = uniform_c + adapter.o_delta(uniform_latent)
    logits_uniform = backbone.actor_logits_from_parts(
        u=out["u"],
        h=out["h"],
        c=uniform_c,
        avail_actions=avail_actions,
    )

    actual_actions = torch.argmax(out["logits"], dim=-1)
    silence_actions = torch.argmax(logits_silence, dim=-1)
    uniform_actions = torch.argmax(logits_uniform, dim=-1)

    temp_stats: dict[str, dict[str, float]] = {}
    for temp in temperatures:
        temp_attn = torch.softmax(out["masked_scores"] / temp, dim=-1)
        temp_attn = torch.nan_to_num(temp_attn, nan=0.0)
        temp_entropy = -(temp_attn * torch.log(temp_attn.clamp_min(1e-8))).sum(dim=-1)
        temp_stats[str(temp)] = {
            "top1_mass": float(temp_attn.max(dim=-1).values.mean().item()),
            "entropy": float(temp_entropy.mean().item()),
            "kl_to_uniform": float((math.log(max(n_agents - 1, 1)) - temp_entropy).mean().item()),
        }

    q_cos = pairwise_cosine_summary(out["q"])
    k_cos = pairwise_cosine_summary(out["k"])

    metrics = {
        "comm_scale": float(torch.sigmoid(backbone.comm_scale_logit).item()),
        "u_norm_mean": float(out["u"].norm(dim=-1).mean().item()),
        "h_norm_mean": float(out["h"].norm(dim=-1).mean().item()),
        "c_unscaled_norm_mean": float(out["c_unscaled"].norm(dim=-1).mean().item()),
        "c_scaled_norm_mean": float(out["c"].norm(dim=-1).mean().item()),
        "raw_score_range": float((row_max - row_min).mean().item()),
        "raw_score_std": row_offdiag_std(raw_scores),
        "attention_top1_mass": float(top1.mean().item()),
        "attention_entropy": float(entropy.mean().item()),
        "attention_kl_to_uniform": float(kl_to_uniform.mean().item()),
        "attention_l1_to_uniform": float(l1_to_uniform.mean().item()),
        "q_cosine_mean": q_cos["mean"],
        "q_cosine_std": q_cos["std"],
        "k_cosine_mean": k_cos["mean"],
        "k_cosine_std": k_cos["std"],
        "action_change_rate_silence": float((actual_actions != silence_actions).float().mean().item()),
        "action_change_rate_uniform": float((actual_actions != uniform_actions).float().mean().item()),
        "policy_kl_silence": mean_policy_kl(out["logits"], logits_silence),
        "policy_kl_uniform": mean_policy_kl(out["logits"], logits_uniform),
        "logit_delta_l2_silence": float((out["logits"] - logits_silence).norm(dim=-1).mean().item()),
        "logit_delta_l2_uniform": float((out["logits"] - logits_uniform).norm(dim=-1).mean().item()),
    }
    return {
        "metrics": metrics,
        "temperature_stats": temp_stats,
        "attn": out["comm_attn"].detach().cpu().numpy(),
        "raw_scores": out["raw_scores"].detach().cpu().numpy(),
    }


def probe_method(
    method_name: str,
    env: VMASWrapper,
    backbone: FrozenAttentionHUActorBackbone,
    adapter: Optional[MemeticCommAdapter],
    episodes: int,
    episode_seeds: list[int],
    device: torch.device,
    temperatures: list[float],
) -> dict[str, object]:
    env_info = env.get_env_info()
    metric_lists: dict[str, list[float]] = defaultdict(list)
    temp_metric_lists: dict[str, dict[str, list[float]]] = {
        str(temp): defaultdict(list) for temp in temperatures
    }
    most_sensitive = None
    most_nonuniform = None
    total_steps = 0

    for episode_index, ep_seed in enumerate(episode_seeds[:episodes], start=1):
        env.reset(seed=int(ep_seed))
        backbone.reset_memory()
        z_state = adapter.initial_state(n_agents=backbone.n_agents, device=device) if adapter is not None else None
        done = False
        step_index = 0

        while not done:
            obs_t = tensorize_obs(env.get_obs(), device=device)
            avail_t = tensorize_avail(env, env_info=env_info, device=device)
            with torch.no_grad():
                out = backbone.step_with_adapter(
                    obs=obs_t,
                    avail_actions=avail_t,
                    adapter=adapter,
                    z=z_state,
                    deterministic=True,
                    return_diagnostics=True,
                )
            z_state = out["z_next"]
            if z_state is not None:
                z_state = z_state.detach()

            step_summary = compute_step_metrics(
                backbone=backbone,
                adapter=adapter,
                out=out,
                avail_actions=avail_t,
                temperatures=temperatures,
            )
            for key, value in step_summary["metrics"].items():
                metric_lists[key].append(value)
            for temp, vals in step_summary["temperature_stats"].items():
                for key, value in vals.items():
                    temp_metric_lists[temp][key].append(value)

            event = {
                "method": method_name,
                "episode_seed": int(ep_seed),
                "episode_index": int(episode_index),
                "step_index": int(step_index + 1),
                **step_summary["metrics"],
                "attn": step_summary["attn"],
                "raw_scores": step_summary["raw_scores"],
            }
            if most_sensitive is None or event["policy_kl_silence"] > most_sensitive["policy_kl_silence"]:
                most_sensitive = event
            if most_nonuniform is None or event["attention_kl_to_uniform"] > most_nonuniform["attention_kl_to_uniform"]:
                most_nonuniform = event

            actions = out["actions"].detach().cpu().numpy().tolist()
            reward, terminated, _info = env.step(actions)
            _ = reward
            if isinstance(terminated, dict):
                done = any(terminated.values())
            elif isinstance(terminated, list):
                done = any(terminated)
            else:
                done = bool(terminated)
            step_index += 1
            total_steps += 1

    summary = {
        "method": method_name,
        "num_steps": total_steps,
        "metrics": {key: summarize(values) for key, values in metric_lists.items()},
        "temperature_sweep": {
            temp: {key: summarize(values) for key, values in metric_map.items()}
            for temp, metric_map in temp_metric_lists.items()
        },
        "most_sensitive_step": serialize_event(most_sensitive),
        "most_nonuniform_step": serialize_event(most_nonuniform),
    }
    return summary


def print_method_summary(summary: dict[str, object]) -> None:
    metrics = summary["metrics"]
    print(
        json.dumps(
            {
                "method": summary["method"],
                "num_steps": summary["num_steps"],
                "attention_top1_mass_mean": metrics["attention_top1_mass"]["mean"],
                "attention_kl_to_uniform_mean": metrics["attention_kl_to_uniform"]["mean"],
                "raw_score_range_mean": metrics["raw_score_range"]["mean"],
                "q_cosine_mean": metrics["q_cosine_mean"]["mean"],
                "k_cosine_mean": metrics["k_cosine_mean"]["mean"],
                "comm_scale_mean": metrics["comm_scale"]["mean"],
                "action_change_rate_silence_mean": metrics["action_change_rate_silence"]["mean"],
                "policy_kl_silence_mean": metrics["policy_kl_silence"]["mean"],
                "action_change_rate_uniform_mean": metrics["action_change_rate_uniform"]["mean"],
                "policy_kl_uniform_mean": metrics["policy_kl_uniform"]["mean"],
            }
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    all_results = {
        "config": {
            "episodes_per_seed": args.episodes,
            "seeds": args.seeds,
            "n_agents": args.n_agents,
            "episode_steps": args.episode_steps,
            "temperatures": args.temperatures,
            "methods": args.methods,
        },
        "methods": {},
    }

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
        seed_episode_seeds = episode_seed_list(seed=seed, episodes=args.episodes)

        method_adapters: dict[str, Optional[MemeticCommAdapter]] = {
            "base": None,
            "es": es_adapter,
            "rl": rl_adapter,
        }
        for method_name in args.methods:
            method_seed_key = f"{method_name}_seed{seed}"
            summary = probe_method(
                method_name=method_name,
                env=env,
                backbone=backbone,
                adapter=method_adapters[method_name],
                episodes=args.episodes,
                episode_seeds=seed_episode_seeds,
                device=device,
                temperatures=args.temperatures,
            )
            all_results["methods"][method_seed_key] = summary
            print_method_summary(summary)
        env.close()

    merged: dict[str, list[dict[str, object]]] = defaultdict(list)
    for summary in all_results["methods"].values():
        merged[summary["method"]].append(summary)

    aggregate = {}
    for method_name, method_summaries in merged.items():
        merged_metrics: dict[str, list[float]] = defaultdict(list)
        merged_temp_metrics: dict[str, dict[str, list[float]]] = {
            str(temp): defaultdict(list) for temp in args.temperatures
        }
        for summary in method_summaries:
            for metric_name, stats in summary["metrics"].items():
                merged_metrics[metric_name].append(float(stats["mean"]))
            for temp, temp_stats in summary["temperature_sweep"].items():
                for metric_name, stats in temp_stats.items():
                    merged_temp_metrics[temp][metric_name].append(float(stats["mean"]))
        aggregate[method_name] = {
            "seed_count": len(method_summaries),
            "metrics": {key: summarize(values) for key, values in merged_metrics.items()},
            "temperature_sweep": {
                temp: {key: summarize(values) for key, values in metric_map.items()}
                for temp, metric_map in merged_temp_metrics.items()
            },
        }

    all_results["aggregate"] = aggregate
    save_path = args.save_dir / args.save_name
    save_path.write_text(json.dumps(all_results, indent=2))
    print(json.dumps({"saved_summary": str(save_path)}), flush=True)


if __name__ == "__main__":
    main()
