#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


METHODS = {
    "ppo": {
        "label": "PPO",
        "base_key": "ppo",
        "run_key_prefixes": ("ppo_seed", "ppo"),
    },
    "maddpg": {
        "label": "MADDPG",
        "base_key": "maddpg",
        "run_key_prefixes": ("maddpg",),
    },
    "method_i": {
        "label": "LA-PPO",
        "base_key": "method_i",
        "run_key_prefixes": ("method_i_seed", "method_i"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a step-matched MPE2 summary across seed 42 and any completed reruns."
    )
    parser.add_argument(
        "--base-results",
        type=Path,
        default=Path("/Users/almondgod/Repositories/memeplex-capstone/old/experiment_results.json"),
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("/Users/almondgod/Repositories/memeplex-capstone/results/mpe2_step_matched_seed_sweep"),
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("/Users/almondgod/Repositories/memeplex-capstone/results/mpe2_step_matched_summary"),
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
    )
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def stderr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return float(math.sqrt(variance) / math.sqrt(len(values)))


def parse_seed_from_name(name: str) -> int | None:
    match = re.search(r"seed(\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def row_from_payload(method_name: str, payload: dict, seed: int, source: str) -> dict:
    return {
        "seed": seed,
        "source": source,
        "final_elo_predator": float(payload["final_elo_predator"]),
        "training_time_s": float(payload["training_time_s"]),
        "total_env_steps": int(payload.get("total_env_steps", 0)),
        "description": payload.get("description", ""),
        "method_name": method_name,
    }


def find_method_payload(data: dict, method_name: str) -> dict | None:
    prefixes = METHODS[method_name]["run_key_prefixes"]

    for prefix in prefixes:
        for key, value in data.items():
            if isinstance(value, dict) and key.startswith(prefix):
                return value

    for value in data.values():
        if not isinstance(value, dict):
            continue
        if "final_elo_predator" in value and "training_time_s" in value:
            description = str(value.get("description", "")).lower()
            if method_name == "method_i" and "method i" in description:
                return value
            if method_name == "ppo" and "ppo" in description:
                return value
            if method_name == "maddpg" and "maddpg" in description:
                return value

    return None


def collect_rows(args: argparse.Namespace) -> dict[str, list[dict]]:
    rows_by_method: dict[str, dict[int, dict]] = {method: {} for method in METHODS}

    base_data = load_json(args.base_results)
    for method_name, meta in METHODS.items():
        payload = base_data.get(meta["base_key"])
        if not isinstance(payload, dict):
            continue
        rows_by_method[method_name][42] = row_from_payload(
            method_name=method_name,
            payload=payload,
            seed=42,
            source=str(args.base_results),
        )

    if args.sweep_dir.exists():
        for method_name in METHODS:
            method_dir = args.sweep_dir / method_name
            if not method_dir.exists():
                continue

            for seed_dir in sorted(path for path in method_dir.iterdir() if path.is_dir()):
                seed = parse_seed_from_name(seed_dir.name)
                results_path = seed_dir / "results.json"
                if seed is None or not results_path.exists():
                    continue

                data = load_json(results_path)
                payload = find_method_payload(data, method_name)
                if payload is None:
                    continue

                rows_by_method[method_name][seed] = row_from_payload(
                    method_name=method_name,
                    payload=payload,
                    seed=seed,
                    source=str(results_path),
                )

    return {
        method_name: [rows_by_method[method_name][seed] for seed in sorted(rows_by_method[method_name])]
        for method_name in METHODS
    }


def summarize_rows(rows_by_method: dict[str, list[dict]], expected_seeds: list[int]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    expected_seed_set = set(expected_seeds)

    for method_name, rows in rows_by_method.items():
        elo_values = [row["final_elo_predator"] for row in rows]
        time_values = [row["training_time_s"] for row in rows]
        completed_seeds = sorted(row["seed"] for row in rows)
        completed_seed_set = set(completed_seeds)

        summary[method_name] = {
            "label": METHODS[method_name]["label"],
            "n_seeds": len(rows),
            "completed_seeds": completed_seeds,
            "missing_seeds": sorted(expected_seed_set - completed_seed_set),
            "predator_elo_mean": mean(elo_values) if elo_values else None,
            "predator_elo_se": stderr(elo_values) if elo_values else None,
            "training_time_mean_s": mean(time_values) if time_values else None,
            "training_time_se_s": stderr(time_values) if time_values else None,
            "rows": rows,
        }

    return summary


def fmt_value(mean_value: float | None, se_value: float | None, digits: int, n_seeds: int) -> str:
    if mean_value is None:
        return "--"
    if n_seeds < 2 or se_value is None:
        return f"{mean_value:.{digits}f}"
    return f"{mean_value:.{digits}f} ± {se_value:.{digits}f}"


def latex_value(mean_value: float | None, se_value: float | None, digits: int, n_seeds: int) -> str:
    if mean_value is None:
        return "--"
    if n_seeds < 2 or se_value is None:
        return f"${mean_value:.{digits}f}$"
    return rf"${mean_value:.{digits}f} \pm {se_value:.{digits}f}$"


def build_markdown(summary: dict[str, dict], expected_seeds: list[int]) -> str:
    lines = [
        "# MPE2 Step-Matched Summary",
        "",
        "Step-matched comparison at 200k environment steps on `simple_tag_v3`.",
        f"Expected seeds: {', '.join(str(seed) for seed in expected_seeds)}.",
        "",
        "| Method | Completed Seeds | Missing Seeds | Predator Elo | Seeds |",
        "| --- | --- | --- | ---: | ---: |",
    ]

    for method_name in ["ppo", "maddpg", "method_i"]:
        item = summary[method_name]
        completed = ", ".join(str(seed) for seed in item["completed_seeds"]) or "--"
        missing = ", ".join(str(seed) for seed in item["missing_seeds"]) or "--"
        lines.append(
            f"| {item['label']} | {completed} | {missing} | "
            f"{fmt_value(item['predator_elo_mean'], item['predator_elo_se'], 1, item['n_seeds'])} | "
            f"{item['n_seeds']} |"
        )

    lines.extend(
        [
            "",
            "Per-seed sources:",
            "",
        ]
    )

    for method_name in ["ppo", "maddpg", "method_i"]:
        item = summary[method_name]
        for row in item["rows"]:
            lines.append(
                f"- {item['label']} seed {row['seed']}: Elo {row['final_elo_predator']:.2f}, "
                f"source `{row['source']}`"
            )

    return "\n".join(lines)


def build_latex_table(summary: dict[str, dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (
            r"\caption{LA-PPO step-matched comparison at 200k environment steps. Values are mean "
            r"$\pm$ standard error when more than one seed is available; the final column reports "
            r"the seed count.}"
        ),
        r"\label{tab:lappo_step_matched}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & Predator Elo & Seeds \\",
        r"\midrule",
    ]

    for method_name in ["ppo", "maddpg", "method_i"]:
        item = summary[method_name]
        lines.append(
            rf"{item['label']} & "
            rf"{latex_value(item['predator_elo_mean'], item['predator_elo_se'], 1, item['n_seeds'])} & "
            rf"{item['n_seeds']} \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def build_json_payload(summary: dict[str, dict], expected_seeds: list[int]) -> dict:
    return {
        "environment": "simple_tag_v3",
        "total_env_steps": 200000,
        "expected_seeds": expected_seeds,
        "methods": summary,
    }


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    rows_by_method = collect_rows(args)
    summary = summarize_rows(rows_by_method, args.expected_seeds)

    markdown_path = args.save_dir / "mpe2_step_matched_summary.md"
    json_path = args.save_dir / "mpe2_step_matched_summary.json"
    table_path = args.save_dir / "mpe2_step_matched_table.tex"

    markdown_path.write_text(build_markdown(summary, args.expected_seeds))
    json_path.write_text(json.dumps(build_json_payload(summary, args.expected_seeds), indent=2))
    table_path.write_text(build_latex_table(summary))

    print(
        json.dumps(
            {
                "markdown": str(markdown_path),
                "json": str(json_path),
                "table_tex": str(table_path),
            }
        )
    )


if __name__ == "__main__":
    main()
