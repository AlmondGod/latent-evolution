#!/opt/homebrew/bin/python3.9
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from new.memetic_foundation.models.frozen_attention_hu_actor import (
    FrozenAttentionHUActorBackbone,
)
from new.memetic_foundation.modules.memetic_adapter import MemeticCommAdapter
from new.memetic_foundation.scripts.run_memetic_selection_phase2 import (
    create_phase2_env,
    evaluate_candidate,
)
from new.memetic_foundation.training.env_reset_utils import reset_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-2 RL baseline over a frozen attention_hu_actor checkpoint"
    )
    parser.add_argument("--load-path", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
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
    parser.add_argument("--z-dim", type=int, default=16)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--disable-z", action="store_true")
    parser.add_argument("--dense-state-update", action="store_true")
    parser.add_argument("--train-transitions", type=int, default=768000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--eval-interval-transitions", type=int, default=38400)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--elite-archive-size", type=int, default=8)
    parser.add_argument("--final-eval-episodes", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--common-random-numbers", action="store_true")
    parser.add_argument("--persistent-z", action="store_true")
    parser.add_argument("--persistent-memory", action="store_true")
    parser.add_argument("--wallclock-seconds", type=float, default=None,
                        help="If set, exit the training loop when this wall-clock budget elapses.")
    parser.add_argument("--warm-init-std", type=float, default=0.0,
                        help="If >0, perturb LowRankDelta.up.weight (zero-init by default) "
                             "and state-cell biases with N(0, std). Breaks the dead-feature "
                             "trap that keeps the meme state cell at z=0 under MAPPO.")
    return parser.parse_args()


def warm_init_adapter(adapter: MemeticCommAdapter, std: float) -> None:
    # Symmetry-breaking init for params that are zero-init by design
    # (LowRankDelta.up.weight, MemeticStateCell biases). Without this, MAPPO has
    # no gradient signal into the state cell and z stays identically 0 forever.
    if std <= 0.0:
        return
    with torch.no_grad():
        for delta in (adapter.q_delta, adapter.k_delta, adapter.v_delta, adapter.o_delta):
            delta.up.weight.normal_(mean=0.0, std=std)
        if adapter.state_cell is not None:
            adapter.state_cell.candidate.up.weight.normal_(mean=0.0, std=std)
            adapter.state_cell.update_gate.up.weight.normal_(mean=0.0, std=std)
            adapter.state_cell.candidate_bias.normal_(mean=0.0, std=std)
            adapter.state_cell.update_gate_bias.normal_(mean=0.0, std=std)


def tensorize_obs(obs_list, device: torch.device) -> torch.Tensor:
    return torch.tensor(np.asarray(obs_list, dtype=np.float32), device=device)


def tensorize_avail(env, env_info: dict, device: torch.device) -> torch.Tensor:
    avail = np.zeros((env_info["n_agents"], env_info["n_actions"]), dtype=np.float32)
    for aid in range(env_info["n_agents"]):
        avail[aid] = np.asarray(env.get_avail_agent_actions(aid), dtype=np.float32)
    return torch.tensor(avail, device=device)


def update_eval_archive(
    archive: list[dict],
    adapter: MemeticCommAdapter,
    metrics: dict[str, float],
    archive_size: int,
    transitions: int,
) -> list[dict]:
    if archive_size <= 0:
        return archive
    archive.append(
        {
            "transitions": int(transitions),
            "fitness": float(metrics["mean_reward"]),
            "theta": adapter.flatten_genotype().detach().cpu().numpy().copy(),
            "metrics": dict(metrics),
        }
    )
    archive.sort(key=lambda item: item["fitness"], reverse=True)
    return archive[:archive_size]


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

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

    adapter = MemeticCommAdapter(
        h_dim=backbone.mem_dim,
        u_dim=backbone.enc_dim,
        z_dim=0 if args.disable_z else args.z_dim,
        attn_dim=backbone.attn_dim,
        rank=args.rank,
        use_z=not args.disable_z,
        dense_state_update=args.dense_state_update,
    ).to(device)
    warm_init_adapter(adapter, args.warm_init_std)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)

    rng = np.random.RandomState(args.seed)

    def sample_episode_seeds(num_episodes: int) -> Optional[list[int]]:
        if not args.common_random_numbers:
            return None
        return [int(rng.randint(0, 2**31 - 1)) for _ in range(num_episodes)]

    def scalar_value(state_np: np.ndarray) -> torch.Tensor:
        state_t = torch.tensor(state_np, dtype=torch.float32, device=device)
        state_rep = state_t.unsqueeze(0).expand(env_info["n_agents"], -1)
        return backbone.get_value(state_rep).mean()

    base_metrics = evaluate_candidate(
        env=env,
        backbone=backbone,
        adapter=None,
        episodes=args.final_eval_episodes,
        deterministic=True,
        persistent_z=args.persistent_z,
        persistent_memory=args.persistent_memory,
        episode_seeds=sample_episode_seeds(args.final_eval_episodes),
    )
    print(json.dumps({"base_checkpoint_metrics": base_metrics}), flush=True)

    history = []
    eval_archive: list[dict] = []
    t0 = time.time()
    transitions = 0
    next_eval_at = args.eval_interval_transitions
    update_idx = 0

    reset_env(env, seed=int(rng.randint(0, 2**31 - 1)))
    backbone.reset_memory()
    z_state = None
    episode_reward = 0.0

    while transitions < args.train_transitions:
        if args.wallclock_seconds is not None and (time.time() - t0) >= args.wallclock_seconds:
            print(json.dumps({"event": "wallclock_budget_exhausted", "elapsed": time.time() - t0, "transitions": transitions}), flush=True)
            break
        backbone.detach_memory()
        if z_state is not None:
            z_state = z_state.detach()

        log_probs = []
        entropies = []
        rewards = []
        dones = []
        values = []

        rollout_steps = min(args.rollout_steps, args.train_transitions - transitions)
        for _ in range(rollout_steps):
            obs_t = tensorize_obs(env.get_obs(), device=device)
            avail_t = tensorize_avail(env, env_info=env_info, device=device)
            state_np = np.asarray(env.get_state(), dtype=np.float32)

            step_out = backbone.step_with_adapter(
                obs=obs_t,
                avail_actions=avail_t,
                adapter=adapter,
                z=z_state,
                deterministic=False,
            )
            z_state = step_out["z_next"]
            if z_state is not None:
                z_state = z_state.detach()
            dist = Categorical(logits=step_out["logits"])
            entropy = dist.entropy().mean()
            log_prob = step_out["log_probs"].mean()
            value = scalar_value(state_np).detach()

            actions = step_out["actions"].detach().cpu().numpy().tolist()
            reward, terminated, info = env.step(actions)
            if isinstance(info, list) and len(info) > 0:
                info = info[0]

            if isinstance(terminated, dict):
                done = any(terminated.values())
            elif isinstance(terminated, list):
                done = any(terminated)
            else:
                done = bool(terminated)

            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(float(reward))
            dones.append(float(done))
            values.append(value)

            transitions += 1
            episode_reward += float(reward)

            if done:
                episode_reward = 0.0
                reset_env(env, seed=int(rng.randint(0, 2**31 - 1)))
                if not args.persistent_memory:
                    backbone.reset_memory()
                if not args.persistent_z:
                    z_state = None

            if transitions >= args.train_transitions:
                break

        if dones and dones[-1] >= 1.0:
            next_value = torch.tensor(0.0, device=device)
        else:
            next_state_np = np.asarray(env.get_state(), dtype=np.float32)
            next_value = scalar_value(next_state_np).detach()

        adv = []
        gae = torch.tensor(0.0, device=device)
        values_t = torch.stack(values).to(device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device)
        for t in reversed(range(len(rewards))):
            next_nonterminal = 1.0 - dones_t[t]
            next_val = next_value if t == len(rewards) - 1 else values_t[t + 1]
            delta = rewards_t[t] + args.gamma * next_val * next_nonterminal - values_t[t]
            gae = delta + args.gamma * args.gae_lambda * next_nonterminal * gae
            adv.append(gae)
        adv_t = torch.stack(list(reversed(adv)))
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        logp_t = torch.stack(log_probs)
        ent_t = torch.stack(entropies)

        loss = -(logp_t * adv_t.detach()).mean() - args.ent_coef * ent_t.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite RL phase-2 loss at update {update_idx + 1}")
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(adapter.parameters(), args.max_grad_norm)
        for name, param in adapter.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise RuntimeError(f"Non-finite gradient in adapter parameter {name}")
        optimizer.step()
        for name, param in adapter.named_parameters():
            if not torch.isfinite(param).all():
                raise RuntimeError(f"Non-finite adapter parameter after step: {name}")

        update_idx += 1
        history.append(
            {
                "update": update_idx,
                "transitions": transitions,
                "loss": float(loss.item()),
                "adv_mean": float(adv_t.mean().item()),
                "adv_std": float(adv_t.std().item()),
                "theta_norm": float(adapter.flatten_genotype().norm().item()),
            }
        )

        if transitions >= next_eval_at or transitions >= args.train_transitions:
            eval_metrics = evaluate_candidate(
                env=env,
                backbone=backbone,
                adapter=adapter,
                episodes=args.eval_episodes,
                deterministic=True,
                persistent_z=args.persistent_z,
                persistent_memory=args.persistent_memory,
                episode_seeds=sample_episode_seeds(args.eval_episodes),
            )
            eval_archive = update_eval_archive(
                archive=eval_archive,
                adapter=adapter,
                metrics=eval_metrics,
                archive_size=args.elite_archive_size,
                transitions=transitions,
            )
            history[-1]["eval_metrics"] = eval_metrics
            print(
                json.dumps(
                    {
                        "update": update_idx,
                        "transitions": transitions,
                        "loss": history[-1]["loss"],
                        "eval_reward": eval_metrics["mean_reward"],
                        "eval_min_dist": eval_metrics.get("mean_min_dist"),
                        "archive_size": len(eval_archive),
                    }
                ),
                flush=True,
            )
            next_eval_at += args.eval_interval_transitions

    final_pool = eval_archive if eval_archive else [
        {
            "transitions": transitions,
            "fitness": float("-inf"),
            "theta": adapter.flatten_genotype().detach().cpu().numpy().copy(),
            "metrics": None,
        }
    ]
    final_episode_seeds = sample_episode_seeds(args.final_eval_episodes)
    final_heldout_rows = []
    for entry in final_pool:
        adapter.load_genotype(torch.tensor(entry["theta"], dtype=torch.float32, device=device))
        final_metrics = evaluate_candidate(
            env=env,
            backbone=backbone,
            adapter=adapter,
            episodes=args.final_eval_episodes,
            deterministic=True,
            persistent_z=args.persistent_z,
            persistent_memory=args.persistent_memory,
            episode_seeds=final_episode_seeds,
        )
        final_metrics["fitness"] = final_metrics["mean_reward"]
        final_metrics["transitions"] = int(entry["transitions"])
        final_metrics["search_fitness"] = float(entry["fitness"])
        final_heldout_rows.append(final_metrics)

    final_best = max(final_heldout_rows, key=lambda row: row["fitness"])
    selected_entry = next(
        entry for entry in final_pool
        if int(entry["transitions"]) == int(final_best["transitions"])
        and float(entry["fitness"]) == float(final_best["search_fitness"])
    )
    adapter.load_genotype(torch.tensor(selected_entry["theta"], dtype=torch.float32, device=device))

    result = {
        "checkpoint": str(args.load_path),
        "env": args.env,
        "n_agents": args.n_agents,
        "seed": args.seed,
        "z_dim": args.z_dim,
        "rank": args.rank,
        "disable_z": args.disable_z,
        "dense_state_update": args.dense_state_update,
        "train_transitions": args.train_transitions,
        "rollout_steps": args.rollout_steps,
        "eval_interval_transitions": args.eval_interval_transitions,
        "eval_episodes": args.eval_episodes,
        "elite_archive_size": args.elite_archive_size,
        "final_eval_episodes": args.final_eval_episodes,
        "lr": args.lr,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "ent_coef": args.ent_coef,
        "base_checkpoint_metrics": base_metrics,
        "history": history,
        "final_heldout_rows": final_heldout_rows,
        "final_best_metrics": final_best,
        "elapsed_seconds": time.time() - t0,
    }
    (args.save_dir / "phase2_rl_history.json").write_text(json.dumps(result, indent=2))
    torch.save(
        {
            "theta": torch.tensor(selected_entry["theta"], dtype=torch.float32),
            "config": {
                "z_dim": args.z_dim,
                "rank": args.rank,
                "disable_z": args.disable_z,
                "dense_state_update": args.dense_state_update,
                "train_transitions": args.train_transitions,
                "rollout_steps": args.rollout_steps,
                "eval_interval_transitions": args.eval_interval_transitions,
                "eval_episodes": args.eval_episodes,
                "elite_archive_size": args.elite_archive_size,
                "final_eval_episodes": args.final_eval_episodes,
                "lr": args.lr,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "ent_coef": args.ent_coef,
                "warm_init_std": args.warm_init_std,
            },
        },
        args.save_dir / "best_adapter.pt",
    )
    print(
        json.dumps(
            {
                "saved": str(args.save_dir / "phase2_rl_history.json"),
                "base_reward": base_metrics["mean_reward"],
                "final_best_reward": final_best["mean_reward"],
                "final_best_fitness": final_best["fitness"],
            }
        ),
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
