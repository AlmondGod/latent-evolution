#!/opt/homebrew/bin/python3.9
"""Restored Phase-1 training/eval runner for attention_hu_actor-style variants."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from new.memetic_foundation.run import create_env
from new.memetic_foundation.training.trainer_phase1 import MemeticFoundationTrainerPhase1


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["memory_only", "mean_u", "attention_u", "attention_hu", "attention_hu_actor"],
        required=True,
    )
    parser.add_argument("--env", choices=["smacv2", "mpe", "rware", "vmas"], default="mpe")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mpe-scenario", type=str, default="simple_spread_v2")
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--race", choices=["terran", "protoss", "zerg"], default="terran")
    parser.add_argument("--n-enemies", type=int, default=None)
    parser.add_argument("--obs-radius", type=float, default=0.5)
    parser.add_argument("--obs-radius-curriculum", action="store_true")
    parser.add_argument("--obs-curriculum-steps", type=int, default=100_000)
    parser.add_argument("--rware-rows", type=int, default=2)
    parser.add_argument("--rware-cols", type=int, default=3)
    parser.add_argument("--rware-height", type=int, default=8)
    parser.add_argument("--rware-requests", type=int, default=2)
    parser.add_argument("--rware-sensor-range", type=int, default=1)
    parser.add_argument("--vmas-scenario", choices=["transport", "discovery"], default="transport")
    parser.add_argument("--vmas-packages", type=int, default=1)
    parser.add_argument("--vmas-targets", type=int, default=4)
    parser.add_argument("--vmas-agents-per-target", type=int, default=1)
    parser.add_argument("--vmas-targets-respawn", action="store_true")
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--rollout-steps", type=int, default=400)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--eval-snapshot-episodes", type=int, default=16)
    parser.add_argument("--probe-interval", type=int, default=0)
    parser.add_argument("--probe-episodes", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--mem-dim", type=int, default=128)
    parser.add_argument("--comm-dim", type=int, default=128)
    parser.add_argument("--n-mem-cells", type=int, default=8)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--print-every", type=int, default=10_000)
    return parser.parse_args()


def curriculum_radius(total_steps: int, target_radius: float, curriculum_steps: int = 100_000) -> float:
    if total_steps >= curriculum_steps:
        return target_radius
    frac_done = total_steps / float(curriculum_steps)
    return 100.0 + frac_done * (target_radius - 100.0)


def variant_config(variant: str) -> dict:
    if variant == "memory_only":
        return {
            "use_memory": True,
            "use_comm": False,
            "comm_mode": "commnet",
            "checkpoint_name": "memfound_memory_only_latest.pt",
        }
    if variant == "mean_u":
        return {
            "use_memory": True,
            "use_comm": True,
            "comm_mode": "commnet_u",
            "checkpoint_name": "memfound_full_commnet_u_latest.pt",
        }
    if variant == "attention_u":
        return {
            "use_memory": True,
            "use_comm": True,
            "comm_mode": "attention_u",
            "checkpoint_name": "memfound_full_attention_u_latest.pt",
        }
    if variant == "attention_hu":
        return {
            "use_memory": True,
            "use_comm": True,
            "comm_mode": "attention_hu",
            "checkpoint_name": "memfound_full_attention_hu_latest.pt",
        }
    if variant == "attention_hu_actor":
        return {
            "use_memory": True,
            "use_comm": True,
            "comm_mode": "attention_hu_actor",
            "checkpoint_name": "memfound_full_attention_hu_actor_latest.pt",
        }
    raise ValueError(f"Unsupported variant: {variant}")


def best_checkpoint_name(latest_name: str) -> str:
    if latest_name.endswith("_latest.pt"):
        return latest_name.replace("_latest.pt", "_best.pt")
    if latest_name.endswith(".pt"):
        return latest_name[:-3] + "_best.pt"
    return latest_name + "_best"


def is_better_eval(candidate: Dict[str, Optional[float]], incumbent: Optional[Dict[str, Optional[float]]]) -> bool:
    if incumbent is None:
        return True
    cand_dist = candidate.get("eval_min_dist")
    inc_dist = incumbent.get("eval_min_dist")
    if cand_dist is not None and inc_dist is not None and cand_dist != inc_dist:
        return cand_dist < inc_dist
    cand_success = candidate.get("eval_success_rate")
    inc_success = incumbent.get("eval_success_rate")
    if cand_success is not None and inc_success is not None and cand_success != inc_success:
        return cand_success > inc_success
    cand_win = candidate.get("eval_win_rate")
    inc_win = incumbent.get("eval_win_rate")
    if cand_win is not None and inc_win is not None and cand_win != inc_win:
        return cand_win > inc_win
    cand_reward = candidate.get("eval_mean_reward")
    inc_reward = incumbent.get("eval_mean_reward")
    if cand_reward is not None and inc_reward is not None and cand_reward != inc_reward:
        return cand_reward > inc_reward
    return candidate.get("step", 0) < incumbent.get("step", 0)


def main() -> None:
    args = build_args()
    config = variant_config(args.variant)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.save_dir.mkdir(parents=True, exist_ok=True)

    env_args = argparse.Namespace(
        env=args.env,
        mpe_scenario=args.mpe_scenario,
        n_adversaries=args.n_agents,
        obs_radius=args.obs_radius,
        rollout_steps=args.rollout_steps,
        race=args.race,
        n_units=args.n_agents,
        n_enemies=args.n_enemies if args.n_enemies is not None else args.n_agents,
        rware_rows=args.rware_rows,
        rware_cols=args.rware_cols,
        rware_height=args.rware_height,
        rware_requests=args.rware_requests,
        rware_sensor_range=args.rware_sensor_range,
        vmas_scenario=args.vmas_scenario,
        vmas_packages=args.vmas_packages,
        vmas_targets=args.vmas_targets,
        vmas_agents_per_target=args.vmas_agents_per_target,
        vmas_targets_respawn=args.vmas_targets_respawn,
    )
    env = create_env(env_args, render=False)
    trainer = MemeticFoundationTrainerPhase1(
        env=env,
        device=args.device,
        lr=5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=10.0,
        update_epochs=5,
        num_mini_batches=1,
        hidden_dim=args.hidden_dim,
        mem_dim=args.mem_dim,
        comm_dim=args.comm_dim,
        n_mem_cells=args.n_mem_cells,
        use_memory=config["use_memory"],
        use_comm=config["use_comm"],
        use_gate=True,
        mem_decay=0.005,
        comm_mode=config["comm_mode"],
        param_eq=False,
        persistent_memory=False,
    )

    total_steps = 0
    n_iters = args.total_steps // args.rollout_steps
    log = []
    eval_log = []
    best_eval = None
    latest_ckpt_path = args.save_dir / config["checkpoint_name"]
    best_ckpt_path = args.save_dir / best_checkpoint_name(config["checkpoint_name"])
    eval_ckpt_dir = args.save_dir / "eval_checkpoints"
    t0 = time.time()

    for iteration in range(n_iters):
        frac = 1.0 - (iteration / max(n_iters, 1))
        trainer.anneal_lr(frac)
        if args.env == "mpe" and args.obs_radius_curriculum and hasattr(env, "set_obs_radius"):
            env.set_obs_radius(curriculum_radius(total_steps, args.obs_radius, args.obs_curriculum_steps))
        elif args.env == "mpe" and hasattr(env, "set_obs_radius"):
            env.set_obs_radius(args.obs_radius)

        buffer, last_values, stats = trainer.collect_rollout(args.rollout_steps)
        update_stats = trainer.update(buffer, last_values)
        buffer.clear()
        total_steps += args.rollout_steps

        row = {
            "step": total_steps,
            "train_mean_reward": stats.get("mean_reward", 0.0),
            "train_success_rate": stats.get("success_rate"),
            "train_min_dist": stats.get("min_dist"),
            "train_collisions": stats.get("collisions"),
            "memory_norm": update_stats.get("memory_norm", 0.0),
            "message_in_norm": update_stats.get("message_in_norm", 0.0),
            "comm_scale": update_stats.get("comm_scale", 0.0),
            "pg_loss": update_stats.get("pg_loss", 0.0),
            "vf_loss": update_stats.get("vf_loss", 0.0),
            "entropy": update_stats.get("entropy", 0.0),
        }
        log.append(row)
        if args.probe_interval > 0 and total_steps % args.probe_interval == 0:
            trainer.probe_rollout(
                n_episodes=args.probe_episodes,
                training_step=total_steps,
                save_dir=str(args.save_dir / "probes"),
            )
        if args.eval_interval > 0 and total_steps % args.eval_interval == 0:
            eval_stats = trainer.evaluate(test_episodes=args.eval_snapshot_episodes, deterministic=True)
            eval_ckpt_dir.mkdir(parents=True, exist_ok=True)
            eval_ckpt_path = eval_ckpt_dir / f"{config['checkpoint_name'][:-3]}_step{total_steps:08d}.pt"
            trainer.save(str(eval_ckpt_path))
            eval_row = {
                "step": total_steps,
                "eval_mean_reward": eval_stats.get("mean_reward"),
                "eval_win_rate": eval_stats.get("win_rate"),
                "eval_success_rate": eval_stats.get("success_rate"),
                "eval_min_dist": eval_stats.get("min_dist"),
                "eval_collisions": eval_stats.get("collisions"),
                "checkpoint_path": str(eval_ckpt_path),
            }
            eval_log.append(eval_row)
            if is_better_eval(eval_row, best_eval):
                trainer.save(str(best_ckpt_path))
                best_eval = dict(eval_row)
                best_eval["best_checkpoint_path"] = str(best_ckpt_path)
            print(json.dumps({"eval": eval_row}), flush=True)
        if total_steps % args.print_every == 0 or total_steps == args.total_steps:
            print(json.dumps(row), flush=True)

    eval_stats = trainer.evaluate(test_episodes=args.eval_episodes, deterministic=True)
    trainer.save(str(latest_ckpt_path))

    result = {
        "variant": args.variant,
        "seed": args.seed,
        "mpe_scenario": args.mpe_scenario if args.env == "mpe" else None,
        "vmas_scenario": args.vmas_scenario if args.env == "vmas" else None,
        "env": args.env,
        "n_agents": args.n_agents,
        "race": args.race if args.env == "smacv2" else None,
        "n_enemies": (args.n_enemies if args.n_enemies is not None else args.n_agents) if args.env == "smacv2" else None,
        "total_steps": total_steps,
        "eval_episodes": args.eval_episodes,
        "eval_mean_reward": eval_stats.get("mean_reward"),
        "eval_win_rate": eval_stats.get("win_rate"),
        "eval_success_rate": eval_stats.get("success_rate"),
        "eval_min_dist": eval_stats.get("min_dist"),
        "eval_collisions": eval_stats.get("collisions"),
        "eval_log": eval_log,
        "latest_checkpoint_path": str(latest_ckpt_path),
        "best_eval_step": best_eval.get("step") if best_eval else None,
        "best_eval_mean_reward": best_eval.get("eval_mean_reward") if best_eval else None,
        "best_eval_win_rate": best_eval.get("eval_win_rate") if best_eval else None,
        "best_eval_success_rate": best_eval.get("eval_success_rate") if best_eval else None,
        "best_eval_min_dist": best_eval.get("eval_min_dist") if best_eval else None,
        "best_eval_collisions": best_eval.get("eval_collisions") if best_eval else None,
        "best_checkpoint_path": best_eval.get("best_checkpoint_path") if best_eval else None,
        "final_comm_scale": log[-1]["comm_scale"] if log else None,
        "final_message_in_norm": log[-1]["message_in_norm"] if log else None,
        "final_memory_norm": log[-1]["memory_norm"] if log else None,
        "wall_clock_seconds": time.time() - t0,
        "log": log,
    }
    with open(args.save_dir / "direct_perf_results.json", "w") as handle:
        json.dump(result, handle, indent=2)

    env.close()
    summary = {
        "variant": result["variant"],
        "seed": result["seed"],
        "eval_mean_reward": result["eval_mean_reward"],
        "eval_win_rate": result["eval_win_rate"],
        "eval_success_rate": result["eval_success_rate"],
        "eval_min_dist": result["eval_min_dist"],
        "eval_collisions": result["eval_collisions"],
        "best_eval_step": result["best_eval_step"],
        "best_eval_mean_reward": result["best_eval_mean_reward"],
        "best_eval_win_rate": result["best_eval_win_rate"],
        "best_eval_min_dist": result["best_eval_min_dist"],
        "final_comm_scale": result["final_comm_scale"],
        "final_message_in_norm": result["final_message_in_norm"],
        "final_memory_norm": result["final_memory_norm"],
        "wall_clock_seconds": result["wall_clock_seconds"],
    }
    print(json.dumps({"final": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
