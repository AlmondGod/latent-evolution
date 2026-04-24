from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/Users/almondgod/Repositories/memeplex-capstone")
RESULTS_DIR = ROOT / "results" / "vmas_phase2_ablations_n12"
MARKDOWN_PATH = RESULTS_DIR / "ablation_tables.md"
JSON_PATH = RESULTS_DIR / "ablation_tables.json"


@dataclass(frozen=True)
class VariantSpec:
    label: str
    method: str
    path_template: str
    kind: str
    notes: str = ""
    parallel_teams: int | None = None
    total_active_agents: int | None = None
    es_population: int | None = None
    es_generations: int | None = None
    rl_train_transitions: int | None = None


RESOURCE_VARIANTS = [
    VariantSpec(
        label="linear_a",
        method="ES",
        path_template="results/vmas_phase2_linear_a_discovery_50k_n12/linear_a/n12/es/seed{seed}/phase2_history.json",
        kind="es",
        notes="with z, low-rank update",
        parallel_teams=16,
        total_active_agents=192,
        es_population=16,
        es_generations=20,
    ),
    VariantSpec(
        label="linear_a",
        method="RL",
        path_template="results/vmas_phase2_linear_a_discovery_50k_n12/linear_a/n12/rl/seed{seed}/phase2_rl_history.json",
        kind="rl",
        notes="with z, low-rank update",
        parallel_teams=16,
        total_active_agents=192,
        rl_train_transitions=768_000,
    ),
    VariantSpec(
        label="fixed_a",
        method="ES",
        path_template="results/vmas_phase2_fixed_a_discovery_50k_n12/fixed_a/n12/es/seed{seed}/phase2_history.json",
        kind="es",
        notes="with z, low-rank update",
        parallel_teams=10,
        total_active_agents=120,
        es_population=10,
        es_generations=14,
    ),
    VariantSpec(
        label="fixed_a",
        method="RL",
        path_template="results/vmas_phase2_fixed_a_discovery_50k_n12/fixed_a/n12/rl/seed{seed}/phase2_rl_history.json",
        kind="rl",
        notes="with z, low-rank update",
        parallel_teams=10,
        total_active_agents=120,
        rl_train_transitions=480_000,
    ),
]


ARCH_VARIANTS = [
    VariantSpec(
        label="with_z_low_rank",
        method="ES",
        path_template="results/vmas_phase2_linear_a_discovery_50k_n12/linear_a/n12/es/seed{seed}/phase2_history.json",
        kind="es",
        notes="linear_a baseline",
    ),
    VariantSpec(
        label="without_z",
        method="ES",
        path_template="results/vmas_phase2_linear_a_discovery_50k_n12_noz_es/seed{seed}/phase2_history.json",
        kind="es",
        notes="disable_z=True",
    ),
    VariantSpec(
        label="with_z_dense_update",
        method="ES",
        path_template="results/vmas_phase2_linear_a_discovery_50k_n12_dense_es/seed{seed}/phase2_history.json",
        kind="es",
        notes="dense_state_update=True",
    ),
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def extract_metrics(data: dict[str, Any], kind: str) -> tuple[float, float]:
    baseline = float(data["base_checkpoint_metrics"]["mean_reward"])
    if kind == "es":
        final = float(data["final_selected_metrics"]["mean_reward"])
    else:
        final = float(data["final_best_metrics"]["mean_reward"])
    return baseline, final


def build_variant_row(spec: VariantSpec) -> dict[str, Any]:
    seeds: dict[str, Any] = {}
    deltas = []
    baselines = []
    finals = []
    for seed in (1, 2, 3):
        path = ROOT / spec.path_template.format(seed=seed)
        data = load_json(path)
        baseline, final = extract_metrics(data, spec.kind)
        delta = final - baseline
        seeds[str(seed)] = {
            "baseline_reward": baseline,
            "final_reward": final,
            "delta_reward": delta,
            "path": str(path),
        }
        baselines.append(baseline)
        finals.append(final)
        deltas.append(delta)
    return {
        "label": spec.label,
        "method": spec.method,
        "notes": spec.notes,
        "parallel_teams": spec.parallel_teams,
        "total_active_agents": spec.total_active_agents,
        "es_population": spec.es_population,
        "es_generations": spec.es_generations,
        "rl_train_transitions": spec.rl_train_transitions,
        "seed_results": seeds,
        "mean_baseline_reward": mean(baselines),
        "mean_final_reward": mean(finals),
        "mean_delta_reward": mean(deltas),
        "improved_seeds": sum(delta > 0 for delta in deltas),
    }


def fmt(x: float) -> str:
    return f"{x:+.4f}"


def render_resource_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Regime | Method | E | A | Budget | Seed 1 Δ | Seed 2 Δ | Seed 3 Δ | Mean Δ | Improved |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        budget = (
            f"pop {row['es_population']}, gen {row['es_generations']}"
            if row["method"] == "ES"
            else f"{row['rl_train_transitions']:,} transitions"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    row["method"],
                    str(row["parallel_teams"]),
                    str(row["total_active_agents"]),
                    budget,
                    fmt(row["seed_results"]["1"]["delta_reward"]),
                    fmt(row["seed_results"]["2"]["delta_reward"]),
                    fmt(row["seed_results"]["3"]["delta_reward"]),
                    fmt(row["mean_delta_reward"]),
                    str(row["improved_seeds"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_arch_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Variant | Seed 1 Δ | Seed 2 Δ | Seed 3 Δ | Mean Δ | Improved | Notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    fmt(row["seed_results"]["1"]["delta_reward"]),
                    fmt(row["seed_results"]["2"]["delta_reward"]),
                    fmt(row["seed_results"]["3"]["delta_reward"]),
                    fmt(row["mean_delta_reward"]),
                    str(row["improved_seeds"]),
                    row["notes"],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_baseline_context(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Variant | Mean Baseline Reward | Mean Final Reward |",
        "| --- | ---: | ---: |",
    ]
    for row in rows:
        label = f"{row['label']} ({row['method']})"
        lines.append(
            f"| {label} | {row['mean_baseline_reward']:.4f} | {row['mean_final_reward']:.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    resource_rows = [build_variant_row(spec) for spec in RESOURCE_VARIANTS]
    arch_rows = [build_variant_row(spec) for spec in ARCH_VARIANTS]

    payload = {
        "env": "vmas_discovery",
        "n_agents": 12,
        "resource_ablation": resource_rows,
        "architecture_ablation": arch_rows,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2))

    markdown = "\n".join(
        [
            "# VMAS Discovery N=12 Ablation Tables",
            "",
            "These tables summarize reward deltas over the frozen Phase-1 backbone.",
            "",
            "## Resource Ablation",
            "",
            render_resource_table(resource_rows),
            "",
            "## Architecture Ablation",
            "",
            render_arch_table(arch_rows),
            "",
            "## Mean Reward Context",
            "",
            render_baseline_context(resource_rows + arch_rows),
            "",
        ]
    )
    MARKDOWN_PATH.write_text(markdown)


if __name__ == "__main__":
    main()
