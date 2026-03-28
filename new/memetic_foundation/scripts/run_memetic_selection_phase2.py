#!/opt/homebrew/bin/python3.9
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from new.memetic_foundation.models.frozen_attention_hu_actor import (
    FrozenAttentionHUActorBackbone,
)
from new.memetic_foundation.modules.memetic_adapter import MemeticCommAdapter
from new.memetic_foundation.training.env_utils import make_env
from new.memetic_foundation.training.mpe_wrapper import MPEWrapper
from new.memetic_foundation.training.openai_es import OpenAIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-2 memetic selection over a frozen attention_hu_actor checkpoint")
    parser.add_argument("--load-path", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--env", choices=["smacv2", "mpe", "lbf"], default="mpe")
    parser.add_argument("--mpe-scenario", type=str, default="simple_spread_v2")
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
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--z-dim", type=int, default=16)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--final-eval-episodes", type=int, default=16)
    parser.add_argument("--es-sigma", type=float, default=0.05)
    parser.add_argument("--es-lr", type=float, default=0.02)
    parser.add_argument("--fitness-comm-gain", type=float, default=0.0)
    parser.add_argument("--persistent-z", action="store_true")
    parser.add_argument("--persistent-memory", action="store_true")
    parser.add_argument("--stochastic-actions", action="store_true")
    return parser.parse_args()


def create_phase2_env(args: argparse.Namespace):
    if args.env == "mpe":
        return MPEWrapper(
            scenario_name=args.mpe_scenario,
            num_adversaries=args.n_agents,
            max_cycles=args.episode_steps,
            obs_radius=args.obs_radius,
            N=args.n_agents,
        )
    if args.env == "smacv2":
        return make_env(
            race=args.race,
            n_units=args.n_agents,
            n_enemies=args.n_enemies if args.n_enemies is not None else args.n_agents,
            render=False,
        )
    if args.env == "lbf":
        try:
            from new.memetic_foundation.training.lbf_wrapper import LBFWrapper
        except ImportError as exc:
            raise RuntimeError("LBF wrapper is not available in this worktree") from exc
        return LBFWrapper(
            size=args.lbf_size,
            players=args.n_agents,
            foods=args.lbf_foods,
            sight=args.lbf_sight,
            max_episode_steps=args.lbf_max_steps,
            force_coop=not args.lbf_no_coop,
        )
    raise ValueError(f"Unsupported env: {args.env}")


def tensorize_obs(obs_list, device: torch.device) -> torch.Tensor:
    return torch.tensor(np.asarray(obs_list, dtype=np.float32), device=device)


def tensorize_avail(env, env_info: dict, device: torch.device) -> torch.Tensor:
    avail = np.zeros((env_info["n_agents"], env_info["n_actions"]), dtype=np.float32)
    for aid in range(env_info["n_agents"]):
        avail[aid] = np.asarray(env.get_avail_agent_actions(aid), dtype=np.float32)
    return torch.tensor(avail, device=device)


def summarize_rollout_metrics(episode_metrics: list[dict]) -> dict[str, float]:
    mean_reward = float(np.mean([m["reward"] for m in episode_metrics])) if episode_metrics else 0.0
    wins = [m["won"] for m in episode_metrics if m["won"] is not None]
    success = [m["success"] for m in episode_metrics if m["success"] is not None]
    min_dist = [m["min_dist"] for m in episode_metrics if m["min_dist"] is not None]
    collisions = [m["collisions"] for m in episode_metrics if m["collisions"] is not None]
    z_norm = [m["z_norm"] for m in episode_metrics if m["z_norm"] is not None]
    comm_norm = [m["comm_norm"] for m in episode_metrics if m["comm_norm"] is not None]
    out = {"mean_reward": mean_reward}
    if wins:
        out["win_rate"] = float(np.mean(wins))
    if success:
        out["success_rate"] = float(np.mean(success))
    if min_dist:
        out["mean_min_dist"] = float(np.mean(min_dist))
    if collisions:
        out["mean_collisions"] = float(np.mean(collisions))
    if z_norm:
        out["mean_z_norm"] = float(np.mean(z_norm))
    if comm_norm:
        out["mean_comm_norm"] = float(np.mean(comm_norm))
    return out


def evaluate_candidate(
    env,
    backbone: FrozenAttentionHUActorBackbone,
    adapter: Optional[MemeticCommAdapter],
    episodes: int,
    deterministic: bool,
    persistent_z: bool,
    persistent_memory: bool,
    intervene_comm_silence: bool = False,
    intervene_comm_shift: bool = False,
) -> dict[str, float]:
    device = next(backbone.parameters()).device
    env_info = env.get_env_info()
    episode_metrics = []
    z_state = None

    for _ in range(episodes):
        env.reset()
        if not persistent_memory:
            backbone.reset_memory()
        if adapter is not None and (not persistent_z or z_state is None):
            z_state = adapter.initial_state(n_agents=backbone.n_agents, device=device)

        terminated = False
        ep_reward = 0.0
        step_min_dists = []
        step_collisions = []
        z_norms = []
        comm_norms = []
        info = {}

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
                    intervene_comm_silence=intervene_comm_silence,
                    intervene_comm_shift=intervene_comm_shift,
                )
            actions = out["actions"].cpu().numpy().tolist()
            z_state = out["z_next"] if adapter is not None else z_state
            reward, terminated, info = env.step(actions)
            ep_reward += reward
            if "min_dist" in info:
                step_min_dists.append(float(info["min_dist"]))
            if "collisions" in info:
                step_collisions.append(float(info["collisions"]))
            if adapter is not None and z_state is not None:
                z_norms.append(float(z_state.norm(dim=-1).mean().item()))
            comm_norms.append(float(out["c"].norm(dim=-1).mean().item()))

        episode_metrics.append(
            {
                "reward": ep_reward,
                "won": float(info["battle_won"]) if "battle_won" in info else None,
                "success": float(info["success"]) if "success" in info else None,
                "min_dist": float(np.mean(step_min_dists)) if step_min_dists else None,
                "collisions": float(np.mean(step_collisions)) if step_collisions else None,
                "z_norm": float(np.mean(z_norms)) if z_norms else None,
                "comm_norm": float(np.mean(comm_norms)) if comm_norms else None,
            }
        )

    return summarize_rollout_metrics(episode_metrics)


def add_fitness(
    baseline_metrics: dict[str, float],
    silence_metrics: Optional[dict[str, float]],
    comm_gain_coef: float,
) -> dict[str, float]:
    out = dict(baseline_metrics)
    reward = baseline_metrics["mean_reward"]
    comm_gain = 0.0
    if silence_metrics is not None:
        comm_gain = reward - silence_metrics["mean_reward"]
        out["silence_mean_reward"] = silence_metrics["mean_reward"]
        if "win_rate" in silence_metrics:
            out["silence_win_rate"] = silence_metrics["win_rate"]
        if "success_rate" in silence_metrics:
            out["silence_success_rate"] = silence_metrics["success_rate"]
        if "mean_min_dist" in silence_metrics:
            out["silence_mean_min_dist"] = silence_metrics["mean_min_dist"]
    out["comm_gain"] = comm_gain
    out["fitness"] = reward + comm_gain_coef * comm_gain
    return out


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
        z_dim=args.z_dim,
        attn_dim=backbone.attn_dim,
        rank=args.rank,
    ).to(device)
    theta0 = adapter.flatten_genotype().cpu().numpy()
    es = OpenAIES(theta0=theta0, sigma=args.es_sigma, lr=args.es_lr, antithetic=True)

    history = []
    t0 = time.time()

    base_metrics = evaluate_candidate(
        env=env,
        backbone=backbone,
        adapter=None,
        episodes=args.final_eval_episodes,
        deterministic=not args.stochastic_actions,
        persistent_z=args.persistent_z,
        persistent_memory=args.persistent_memory,
    )
    print(json.dumps({"base_checkpoint_metrics": base_metrics}), flush=True)

    best = {
        "fitness": float("-inf"),
        "generation": -1,
        "theta": theta0.copy(),
        "metrics": None,
    }

    for generation in range(args.generations):
        candidates, noises = es.ask(args.population_size)
        fitnesses = []
        generation_rows = []

        for cand_idx, theta in enumerate(candidates):
            adapter.load_genotype(torch.tensor(theta, dtype=torch.float32, device=device))
            baseline = evaluate_candidate(
                env=env,
                backbone=backbone,
                adapter=adapter,
                episodes=args.eval_episodes,
                deterministic=not args.stochastic_actions,
                persistent_z=args.persistent_z,
                persistent_memory=args.persistent_memory,
            )
            silence = None
            if args.fitness_comm_gain > 0.0:
                silence = evaluate_candidate(
                    env=env,
                    backbone=backbone,
                    adapter=adapter,
                    episodes=args.eval_episodes,
                    deterministic=not args.stochastic_actions,
                    persistent_z=args.persistent_z,
                    persistent_memory=args.persistent_memory,
                    intervene_comm_silence=True,
                )
            scored = add_fitness(baseline, silence, args.fitness_comm_gain)
            scored["candidate"] = cand_idx
            generation_rows.append(scored)
            fitnesses.append(scored["fitness"])

            if scored["fitness"] > best["fitness"]:
                best = {
                    "fitness": scored["fitness"],
                    "generation": generation,
                    "theta": theta.copy(),
                    "metrics": dict(scored),
                }

        es_stats = es.tell(noises, np.asarray(fitnesses, dtype=np.float64))
        gen_summary = {
            "generation": generation,
            "es": es_stats,
            "best_candidate_fitness": float(np.max(fitnesses)),
            "mean_candidate_fitness": float(np.mean(fitnesses)),
            "rows": generation_rows,
        }
        history.append(gen_summary)
        print(json.dumps({"generation": generation, "best_fitness": gen_summary["best_candidate_fitness"], "mean_fitness": gen_summary["mean_candidate_fitness"], **es_stats}), flush=True)

    adapter.load_genotype(torch.tensor(best["theta"], dtype=torch.float32, device=device))
    final_best = evaluate_candidate(
        env=env,
        backbone=backbone,
        adapter=adapter,
        episodes=args.final_eval_episodes,
        deterministic=not args.stochastic_actions,
        persistent_z=args.persistent_z,
        persistent_memory=args.persistent_memory,
    )
    final_best_scored = add_fitness(
        final_best,
        evaluate_candidate(
            env=env,
            backbone=backbone,
            adapter=adapter,
            episodes=args.final_eval_episodes,
            deterministic=not args.stochastic_actions,
            persistent_z=args.persistent_z,
            persistent_memory=args.persistent_memory,
            intervene_comm_silence=True,
        ) if args.fitness_comm_gain > 0.0 else None,
        args.fitness_comm_gain,
    )

    result = {
        "checkpoint": str(args.load_path),
        "env": args.env,
        "n_agents": args.n_agents,
        "seed": args.seed,
        "z_dim": args.z_dim,
        "rank": args.rank,
        "population_size": args.population_size,
        "generations": args.generations,
        "eval_episodes": args.eval_episodes,
        "final_eval_episodes": args.final_eval_episodes,
        "fitness_comm_gain": args.fitness_comm_gain,
        "base_checkpoint_metrics": base_metrics,
        "best_generation": best["generation"],
        "best_generation_metrics": best["metrics"],
        "final_best_metrics": final_best_scored,
        "elapsed_seconds": time.time() - t0,
        "history": history,
    }

    (args.save_dir / "phase2_history.json").write_text(json.dumps(result, indent=2))
    torch.save(
        {
            "theta": torch.tensor(best["theta"], dtype=torch.float32),
            "config": {
                "z_dim": args.z_dim,
                "rank": args.rank,
                "population_size": args.population_size,
                "generations": args.generations,
                "fitness_comm_gain": args.fitness_comm_gain,
            },
        },
        args.save_dir / "best_adapter.pt",
    )
    print(json.dumps({"saved": str(args.save_dir / 'phase2_history.json'), "best_generation": best["generation"], "base_reward": base_metrics["mean_reward"], "final_best_reward": final_best_scored["mean_reward"], "final_best_fitness": final_best_scored["fitness"]}), flush=True)
    env.close()


if __name__ == "__main__":
    main()
