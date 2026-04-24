#!/opt/homebrew/bin/python3.9
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed-A and scaled-A Phase-2 budget matrices"
    )
    parser.add_argument("--ns", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument(
        "--team-steps-budget",
        type=int,
        default=48_000,
        help="Per-parallel-team env-step budget. Total transitions = E * team_steps_budget.",
    )
    parser.add_argument(
        "--fixed-total-agents",
        type=int,
        default=128,
        help="Target A for the fixed-A regime, where A = N * E.",
    )
    parser.add_argument(
        "--linear-parallel-teams",
        type=int,
        default=16,
        help="Fixed E for the linear-A regime. Then A grows linearly with N.",
    )
    parser.add_argument(
        "--search-scale-factor",
        type=int,
        default=2,
        help="For search-scaled regime, E = search_scale_factor * N, so A grows as N^2.",
    )
    parser.add_argument("--episode-steps", type=int, default=200)
    parser.add_argument("--es-eval-episodes", type=int, default=4)
    parser.add_argument("--es-elite-k", type=int, default=4)
    parser.add_argument("--es-elite-eval-episodes", type=int, default=32)
    parser.add_argument("--es-final-eval-episodes", type=int, default=64)
    parser.add_argument("--es-min-generations", type=int, default=1)
    parser.add_argument("--rl-rollout-steps", type=int, default=128)
    parser.add_argument("--rl-final-eval-episodes", type=int, default=64)
    return parser.parse_args()


def es_generation_transition_cost(
    episode_steps: int,
    population_size: int,
    eval_episodes: int,
    elite_k: int,
    elite_eval_episodes: int,
) -> int:
    return episode_steps * (
        population_size * eval_episodes + elite_k * elite_eval_episodes
    )


def make_row(
    regime: str,
    n_agents: int,
    parallel_teams: int,
    args: argparse.Namespace,
) -> dict:
    total_active_agents = n_agents * parallel_teams
    transition_budget = parallel_teams * args.team_steps_budget

    es_population = parallel_teams
    es_gen_cost = es_generation_transition_cost(
        episode_steps=args.episode_steps,
        population_size=es_population,
        eval_episodes=args.es_eval_episodes,
        elite_k=args.es_elite_k,
        elite_eval_episodes=args.es_elite_eval_episodes,
    )
    es_generations = max(args.es_min_generations, transition_budget // max(es_gen_cost, 1))
    es_realized_transitions = es_generations * es_gen_cost

    rl_eval_interval = max(args.rl_rollout_steps, transition_budget // 20)

    return {
        "regime": regime,
        "n_agents": n_agents,
        "parallel_teams": parallel_teams,
        "total_active_agents": total_active_agents,
        "team_steps_budget": args.team_steps_budget,
        "transition_budget": transition_budget,
        "es": {
            "population_size": es_population,
            "generations": es_generations,
            "eval_episodes": args.es_eval_episodes,
            "elite_k": args.es_elite_k,
            "elite_eval_episodes": args.es_elite_eval_episodes,
            "final_eval_episodes": args.es_final_eval_episodes,
            "generation_transition_cost": es_gen_cost,
            "realized_transition_budget": es_realized_transitions,
        },
        "rl": {
            "train_transitions": transition_budget,
            "rollout_steps": args.rl_rollout_steps,
            "eval_interval_transitions": rl_eval_interval,
            "final_eval_episodes": args.rl_final_eval_episodes,
        },
    }


def rows_to_markdown(rows: list[dict]) -> str:
    lines = [
        "# Phase 2 Agent-Budget Matrix",
        "",
        "This matrix defines the resource protocol for Phase 2 `ES-Comm` vs `RL-Comm` comparisons.",
        "Here `N` is agents per team, `E` is parallel teams, and `A = N * E` is the total active agent budget.",
        "",
        "| Regime | N | E | A=N*E | Transition Budget | ES Pop | ES Gens | RL Transitions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {regime} | {n_agents} | {parallel_teams} | {total_active_agents} | {transition_budget} | {es_pop} | {es_gens} | {rl_transitions} |".format(
                regime=row["regime"],
                n_agents=row["n_agents"],
                parallel_teams=row["parallel_teams"],
                total_active_agents=row["total_active_agents"],
                transition_budget=row["transition_budget"],
                es_pop=row["es"]["population_size"],
                es_gens=row["es"]["generations"],
                rl_transitions=row["rl"]["train_transitions"],
            )
        )
    lines.extend(
        [
            "",
            "Regimes:",
            "- `fixed_a`: hold `A` constant, so larger `N` means fewer parallel teams `E = floor(A / N)`.",
            "- `linear_a`: hold `E` constant, so total active agents `A` grows linearly with `N`.",
            "- `search_scaled`: set `E ∝ N`, so total active agents `A` grows quadratically with `N` and ES gets a larger search population as teams grow.",
            "",
            "Important: with the current runners this is an agent-equivalent budget protocol, not a literal wall-clock-equal parallel implementation, because the frozen backbone is not vectorized across multiple teams.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for n in args.ns:
        rows.append(
            make_row(
                regime="fixed_a",
                n_agents=n,
                parallel_teams=max(1, args.fixed_total_agents // n),
                args=args,
            )
        )
        rows.append(
            make_row(
                regime="linear_a",
                n_agents=n,
                parallel_teams=args.linear_parallel_teams,
                args=args,
            )
        )
        rows.append(
            make_row(
                regime="search_scaled",
                n_agents=n,
                parallel_teams=max(1, args.search_scale_factor * n),
                args=args,
            )
        )

    rows.sort(key=lambda row: (row["regime"], row["n_agents"]))

    json_path = args.save_dir / "phase2_agent_budget_matrix.json"
    md_path = args.save_dir / "phase2_agent_budget_matrix.md"
    json_path.write_text(json.dumps({"rows": rows}, indent=2))
    md_path.write_text(rows_to_markdown(rows))
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}))


if __name__ == "__main__":
    main()
