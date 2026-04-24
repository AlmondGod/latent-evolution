#!/opt/homebrew/bin/python3.9
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact Phase-2 ES vs RL summary artifacts")
    parser.add_argument(
        "--intervention-summary",
        type=Path,
        default=Path("/Users/almondgod/Repositories/memeplex-capstone/results/memetic_phase2_interventions_eval128/intervention_summary.json"),
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("/Users/almondgod/Repositories/memeplex-capstone/results/memetic_phase2_summary"),
    )
    return parser.parse_args()


def fmt_signed(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def fmt_pm(mean_value: float, se_value: float, digits: int) -> str:
    return f"{mean_value:.{digits}f} ± {se_value:.{digits}f}"


def stderr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def method_rows(summary: dict, method: str) -> list[dict]:
    return [row for row in summary["rows"] if row["method"] == method]


def method_stats(summary: dict, method: str) -> dict[str, float]:
    rows = method_rows(summary, method)
    rewards = [row["baseline"]["mean_reward"] for row in rows]
    min_dists = [row["baseline"]["mean_min_dist"] for row in rows]
    return {
        "n_seeds": len(rows),
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_se": stderr(rewards),
        "min_dist_mean": float(np.mean(min_dists)) if min_dists else 0.0,
        "min_dist_se": stderr(min_dists),
    }


def build_markdown(summary: dict) -> str:
    frozen = summary["by_method"]["frozen"]
    es = summary["by_method"]["es"]
    rl = summary["by_method"]["rl"]

    rows = []
    for method, label, stats in [("frozen", "Frozen", frozen), ("es", "ALEC", es), ("rl", "MAPPO adapter", rl)]:
        seeded = method_stats(summary, method)
        reward_gain = stats["baseline_reward_mean"] - frozen["baseline_reward_mean"]
        min_dist_gain = stats["baseline_min_dist_mean"] - frozen["baseline_min_dist_mean"]
        rows.append(
            "| {label} | {reward} | {min_dist} | {seeds} | {reward_gain} | {min_dist_gain} | {silence} | {shift} |".format(
                label=label,
                reward=fmt_pm(seeded["reward_mean"], seeded["reward_se"], 2),
                min_dist=fmt_pm(seeded["min_dist_mean"], seeded["min_dist_se"], 3),
                seeds=seeded["n_seeds"],
                reward_gain=fmt_signed(reward_gain, 2),
                min_dist_gain=fmt_signed(min_dist_gain, 3),
                silence=fmt_signed(stats["delta_reward_silence_mean"], 2),
                shift=fmt_signed(stats["delta_reward_shift_mean"], 2),
            )
        )

    return "\n".join(
        [
            "# Phase 2 Summary",
            "",
            "Comparison on MPE `simple_spread_v2`, `N=8`, using the 128-episode paired intervention evaluation.",
            "",
            "| Method | Mean Reward ± SE | Mean min_dist ± SE | Seeds | Reward vs Frozen | min_dist vs Frozen | Silence dReward | Shift dReward |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "More negative `Silence dReward` / `Shift dReward` means the intervention hurt more, which indicates the policy is relying more on communication.",
        ]
    )


def build_latex_table(summary: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{ALEC phase-2 performance on MPE. Values are mean $\pm$ standard error over seeds; the final column reports the seed count.}",
        r"\label{tab:alec_phase2_mpe}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Mean Reward & Mean min\_dist & Seeds \\",
        r"\midrule",
    ]

    for method, label in [("frozen", "Frozen"), ("rl", "MAPPO adapter"), ("es", "ALEC")]:
        stats = method_stats(summary, method)
        lines.append(
            rf"{label} & ${stats['reward_mean']:.2f} \pm {stats['reward_se']:.2f}$ & "
            rf"${stats['min_dist_mean']:.3f} \pm {stats['min_dist_se']:.3f}$ & "
            rf"{stats['n_seeds']} \\"
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


def build_figure(summary: dict, save_path: Path) -> None:
    methods = ["frozen", "es", "rl"]
    labels = ["Frozen", "ES", "RL"]
    colors = ["#8a8f98", "#1f6feb", "#d97706"]

    reward_means = [summary["by_method"][m]["baseline_reward_mean"] for m in methods]
    min_dist_means = [summary["by_method"][m]["baseline_min_dist_mean"] for m in methods]
    silence_deltas = [summary["by_method"][m]["delta_reward_silence_mean"] for m in methods]
    shift_deltas = [summary["by_method"][m]["delta_reward_shift_mean"] for m in methods]

    x = np.arange(len(labels))
    width = 0.36

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.bar(x, reward_means, color=colors)
    ax.set_xticks(x, labels)
    ax.set_title("Capability")
    ax.set_ylabel("Mean Reward")
    for i, (reward, min_dist) in enumerate(zip(reward_means, min_dist_means)):
        ax.text(i, reward + 8, f"{reward:.1f}\nmd {min_dist:.3f}", ha="center", va="bottom", fontsize=9)

    ax2 = axes[1]
    ax2.bar(x - width / 2, silence_deltas, width=width, label="Silence", color="#c2410c")
    ax2.bar(x + width / 2, shift_deltas, width=width, label="Shift", color="#0f766e")
    ax2.axhline(0.0, color="#444", linewidth=1)
    ax2.set_xticks(x, labels)
    ax2.set_title("Posthoc Comm Interventions")
    ax2.set_ylabel("Reward Delta vs Baseline")
    ax2.legend(frameon=False)
    for i, (sil, shf) in enumerate(zip(silence_deltas, shift_deltas)):
        ax2.text(i - width / 2, sil - 1.5 if sil < 0 else sil + 1.5, f"{sil:.1f}", ha="center", va="top" if sil < 0 else "bottom", fontsize=8)
        ax2.text(i + width / 2, shf - 1.5 if shf < 0 else shf + 1.5, f"{shf:.1f}", ha="center", va="top" if shf < 0 else "bottom", fontsize=8)

    fig.suptitle("Phase 2: Matched-Budget ES vs RL on Communication Adaptation", fontsize=13)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.intervention_summary.read_text())

    md = build_markdown(summary)
    (args.save_dir / "phase2_summary.md").write_text(md)
    build_figure(summary, args.save_dir / "phase2_summary.png")
    (args.save_dir / "phase2_summary.json").write_text(json.dumps(summary, indent=2))
    (args.save_dir / "phase2_summary_table.tex").write_text(build_latex_table(summary))

    print(json.dumps({
        "markdown": str(args.save_dir / "phase2_summary.md"),
        "figure": str(args.save_dir / "phase2_summary.png"),
        "json": str(args.save_dir / "phase2_summary.json"),
        "table_tex": str(args.save_dir / "phase2_summary_table.tex"),
    }))


if __name__ == "__main__":
    main()
