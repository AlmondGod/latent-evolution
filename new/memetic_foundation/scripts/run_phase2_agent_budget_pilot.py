#!/opt/homebrew/bin/python3.9
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Phase-2 ES/RL runs for a chosen agent-budget matrix regime"
    )
    parser.add_argument("--matrix-json", type=Path, required=True)
    parser.add_argument("--regime", choices=["fixed_a", "linear_a", "search_scaled"], required=True)
    parser.add_argument("--backbone-root", type=Path, required=True)
    parser.add_argument("--save-root", type=Path, required=True)
    parser.add_argument("--ns", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env", choices=["vmas", "mpe", "rware", "lbf", "smacv2"], default="vmas")
    parser.add_argument("--vmas-scenario", choices=["transport", "discovery"], default="transport")
    parser.add_argument("--vmas-packages", type=int, default=1)
    parser.add_argument("--vmas-targets", type=int, default=4)
    parser.add_argument("--episode-steps", type=int, default=200)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--python-bin", type=str, default="/opt/homebrew/bin/python3.9")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_rows(matrix_json: Path) -> dict[tuple[str, int], dict]:
    payload = json.loads(matrix_json.read_text())
    return {(row["regime"], int(row["n_agents"])): row for row in payload["rows"]}


def backbone_path(backbone_root: Path, n_agents: int, seed: int, env: str, vmas_scenario: str) -> Path:
    scenario_prefix = "transport"
    if env == "vmas":
        scenario_prefix = vmas_scenario
    return (
        backbone_root
        / f"{scenario_prefix}_n{n_agents}"
        / "attention_hu_actor"
        / f"seed{seed}"
        / "memfound_full_attention_hu_actor_best.pt"
    )


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.matrix_json)
    args.save_root.mkdir(parents=True, exist_ok=True)

    es_script = Path("/Users/almondgod/Repositories/memeplex-capstone/new/memetic_foundation/scripts/run_memetic_selection_phase2.py")
    rl_script = Path("/Users/almondgod/Repositories/memeplex-capstone/new/memetic_foundation/scripts/run_memetic_rl_phase2.py")

    for n_agents in args.ns:
        row = rows[(args.regime, n_agents)]
        ckpt = backbone_path(
            args.backbone_root,
            n_agents=n_agents,
            seed=args.seed,
            env=args.env,
            vmas_scenario=args.vmas_scenario,
        )
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing backbone checkpoint: {ckpt}")

        common = [
            "--load-path", str(ckpt),
            "--env", args.env,
            "--n-agents", str(n_agents),
            "--episode-steps", str(args.episode_steps),
            "--seed", str(args.seed),
            "--device", args.device,
        ]
        if args.env == "vmas":
            common.extend(
                [
                    "--vmas-scenario", args.vmas_scenario,
                    "--vmas-packages", str(args.vmas_packages),
                    "--vmas-targets", str(args.vmas_targets),
                ]
            )

        es_save = args.save_root / args.regime / f"n{n_agents}" / "es" / f"seed{args.seed}"
        es_cmd = [
            args.python_bin, str(es_script),
            *common,
            "--save-dir", str(es_save),
            "--population-size", str(row["es"]["population_size"]),
            "--generations", str(row["es"]["generations"]),
            "--eval-episodes", str(row["es"]["eval_episodes"]),
            "--elite-k", str(row["es"]["elite_k"]),
            "--elite-eval-episodes", str(row["es"]["elite_eval_episodes"]),
            "--elite-archive-size", "8",
            "--final-eval-episodes", str(row["es"]["final_eval_episodes"]),
            "--fitness-comm-gain", "0.0",
            "--common-random-numbers",
        ]
        run_cmd(es_cmd, args.dry_run)

        rl_save = args.save_root / args.regime / f"n{n_agents}" / "rl" / f"seed{args.seed}"
        rl_cmd = [
            args.python_bin, str(rl_script),
            *common,
            "--save-dir", str(rl_save),
            "--train-transitions", str(row["rl"]["train_transitions"]),
            "--rollout-steps", str(row["rl"]["rollout_steps"]),
            "--eval-interval-transitions", str(row["rl"]["eval_interval_transitions"]),
            "--eval-episodes", "32",
            "--elite-archive-size", "8",
            "--final-eval-episodes", str(row["rl"]["final_eval_episodes"]),
            "--common-random-numbers",
        ]
        run_cmd(rl_cmd, args.dry_run)


if __name__ == "__main__":
    main()
