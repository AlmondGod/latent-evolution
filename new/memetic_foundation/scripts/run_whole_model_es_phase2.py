# Whole-model OpenAI-ES on FrozenAttentionHUActorBackbone (adapter=None).
# Default (Arm C): random init, wall-clock matched to Phase-2 adapter runs.
# Optional: --init-theta-path loads a prior best_theta.npy (e.g. Arm C output) for a continuation run.

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from new.memetic_foundation.models.frozen_attention_hu_actor import (
    FrozenAttentionHUActorBackbone,
)
from new.memetic_foundation.training.openai_es import OpenAIES
from new.memetic_foundation.scripts.run_memetic_selection_phase2 import (
    create_phase2_env,
    evaluate_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Whole-model OpenAI-ES (no memetic adapter)")
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--env", choices=["smacv2", "mpe", "vmas"], default="smacv2")
    parser.add_argument("--mpe-scenario", type=str, default="simple_spread_v2")
    parser.add_argument("--vmas-scenario", choices=["transport", "discovery"], default="transport")
    parser.add_argument("--n-agents", type=int, required=True)
    parser.add_argument("--n-enemies", type=int, default=None)
    parser.add_argument("--race", choices=["terran", "protoss", "zerg"], default="terran")
    parser.add_argument("--obs-radius", type=float, default=0.5)
    parser.add_argument("--episode-steps", type=int, default=200)
    parser.add_argument("--vmas-packages", type=int, default=1)
    parser.add_argument("--vmas-targets", type=int, default=4)
    parser.add_argument("--vmas-agents-per-target", type=int, default=1)
    parser.add_argument("--vmas-targets-respawn", action="store_true")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--final-eval-episodes", type=int, default=32)
    parser.add_argument("--es-sigma", type=float, default=0.05)
    parser.add_argument("--es-lr", type=float, default=0.02)
    parser.add_argument("--wallclock-seconds", type=float, required=True)
    parser.add_argument("--max-generations", type=int, default=10_000)
    parser.add_argument("--stochastic-actions", action="store_true")
    parser.add_argument(
        "--init-theta-path",
        type=Path,
        default=None,
        help="Optional .npy vector (e.g. arm_c_es/best_theta.npy) to seed ES mean and initial best.",
    )
    parser.add_argument(
        "--arm-tag",
        type=str,
        default="C_whole_model_es",
        help="Label written into results.json under key 'arm'.",
    )
    return parser.parse_args()


def flatten_params(model: torch.nn.Module) -> np.ndarray:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).cpu().numpy()


def load_params(model: torch.nn.Module, theta: np.ndarray) -> None:
    theta_t = torch.tensor(theta, dtype=torch.float32, device=next(model.parameters()).device)
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.data.copy_(theta_t[offset:offset + n].view_as(p))
            offset += n


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

    backbone = FrozenAttentionHUActorBackbone(
        obs_dim=env_info["obs_shape"],
        state_dim=env_info["state_shape"],
        n_actions=env_info["n_actions"],
        n_agents=env_info["n_agents"],
    ).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    expected_dim = flatten_params(backbone).size
    if args.init_theta_path is not None:
        loaded = np.load(args.init_theta_path, mmap_mode=None)
        theta0 = np.asarray(loaded, dtype=np.float64).reshape(-1)
        if theta0.size != expected_dim:
            raise RuntimeError(
                f"init theta dim {theta0.size} != backbone flat dim {expected_dim} ({args.init_theta_path})"
            )
        load_params(backbone, theta0)
    else:
        theta0 = flatten_params(backbone)

    print(json.dumps({"arm": args.arm_tag, "genotype_dim": int(theta0.size), "init_from_file": str(args.init_theta_path) if args.init_theta_path else None}), flush=True)
    es = OpenAIES(theta0=theta0, sigma=args.es_sigma, lr=args.es_lr, antithetic=True)

    rng = np.random.RandomState(args.seed)
    history: list[dict] = []

    def episode_seeds(n: int) -> list[int]:
        return [int(rng.randint(0, 2**31 - 1)) for _ in range(n)]

    def fitness_of(theta: np.ndarray, eps: int, ep_seeds: Optional[list[int]]) -> dict:
        load_params(backbone, theta)
        backbone.reset_memory()
        return evaluate_candidate(
            env=env,
            backbone=backbone,
            adapter=None,
            episodes=eps,
            deterministic=not args.stochastic_actions,
            persistent_z=False,
            persistent_memory=False,
            episode_seeds=ep_seeds,
        )

    t0 = time.time()
    transitions_consumed = 0
    if args.init_theta_path is not None:
        warm_seeds = episode_seeds(args.eval_episodes)
        warm_metrics = fitness_of(theta0, args.eval_episodes, warm_seeds)
        transitions_consumed += args.eval_episodes * int(env_info.get("episode_limit", args.episode_steps))
        best = {
            "fitness": float(warm_metrics["mean_reward"]),
            "generation": -1,
            "theta": theta0.copy(),
            "metrics": dict(warm_metrics),
        }
        warm_start_eval = dict(warm_metrics)
    else:
        best = {"fitness": float("-inf"), "generation": -1, "theta": theta0.copy(), "metrics": None}
        warm_start_eval = None

    print(json.dumps({"event": "start", "wallclock_budget": args.wallclock_seconds, "pop": args.population_size, "ep_per_cand": args.eval_episodes}), flush=True)

    for generation in range(args.max_generations):
        elapsed = time.time() - t0
        if elapsed >= args.wallclock_seconds:
            print(json.dumps({"event": "budget_exhausted", "elapsed": elapsed, "generation": generation}), flush=True)
            break

        candidates, noises = es.ask(args.population_size)
        seeds_for_gen = episode_seeds(args.eval_episodes)
        fitnesses = []
        gen_t0 = time.time()
        for cand_idx, theta in enumerate(candidates):
            metrics = fitness_of(theta, args.eval_episodes, seeds_for_gen)
            fitnesses.append(metrics["mean_reward"])
            if metrics["mean_reward"] > best["fitness"]:
                best = {
                    "fitness": float(metrics["mean_reward"]),
                    "generation": generation,
                    "theta": theta.copy(),
                    "metrics": dict(metrics),
                }
            transitions_consumed += args.eval_episodes * int(env_info.get("episode_limit", args.episode_steps))
        es_stats = es.tell(noises, np.asarray(fitnesses))
        gen_secs = time.time() - gen_t0
        row = {
            "generation": generation,
            "elapsed_secs": time.time() - t0,
            "gen_secs": gen_secs,
            "fitness_mean": es_stats["fitness_mean"],
            "fitness_max": es_stats["fitness_max"],
            "best_so_far": best["fitness"],
            "transitions_so_far": transitions_consumed,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    # Load best theta into backbone for final eval.
    load_params(backbone, best["theta"])
    backbone.reset_memory()
    final_seeds = episode_seeds(args.final_eval_episodes)
    final_metrics = evaluate_candidate(
        env=env,
        backbone=backbone,
        adapter=None,
        episodes=args.final_eval_episodes,
        deterministic=not args.stochastic_actions,
        persistent_z=False,
        persistent_memory=False,
        episode_seeds=final_seeds,
    )

    out = {
        "arm": args.arm_tag,
        "seed": args.seed,
        "env": args.env,
        "n_agents": args.n_agents,
        "n_enemies": args.n_enemies if args.n_enemies is not None else args.n_agents,
        "wallclock_budget": args.wallclock_seconds,
        "wallclock_elapsed": time.time() - t0,
        "generations_completed": len(history),
        "transitions_consumed": transitions_consumed,
        "init_theta_path": str(args.init_theta_path) if args.init_theta_path else None,
        "warm_start_eval": warm_start_eval,
        "best_during_search": best["metrics"],
        "best_generation": best["generation"],
        "final_eval": final_metrics,
        "history": history,
    }
    with open(args.save_dir / "results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    np.save(args.save_dir / "best_theta.npy", best["theta"])
    print(json.dumps({"final": {k: v for k, v in out.items() if k != "history"}}, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
