#!/opt/homebrew/bin/python3.9
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SAVE_DIR = Path("/Users/almondgod/Repositories/memeplex-capstone/results/vmas_phase2_scaling")

TRANSPORT_SEED1_ROOT = Path(
    "/Users/almondgod/Repositories/memeplex-capstone/results/vmas_phase2_linear_a_transport_50k_seed1_partial/linear_a"
)
TRANSPORT_REPEAT_ROOT = Path(
    "/Users/almondgod/Repositories/memeplex-capstone/results/vmas_phase2_linear_a_transport_50k_repeat/linear_a"
)
TRANSPORT_N12_ROOT = Path(
    "/Users/almondgod/Repositories/memeplex-capstone/results/vmas_phase2_linear_a_transport_50k_n12/linear_a"
)
DISCOVERY_ROOT = Path(
    "/Users/almondgod/Repositories/memeplex-capstone/results/vmas_phase2_linear_a_discovery_50k/linear_a"
)
DISCOVERY_N12_ROOT = Path(
    "/Users/almondgod/Repositories/memeplex-capstone/results/vmas_phase2_linear_a_discovery_50k_n12/linear_a"
)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return float(math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1)))


def stderr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(stdev(values) / math.sqrt(len(values)))


def transport_root_for(n_agents: int, seed: int) -> Path:
    if n_agents == 12:
        return TRANSPORT_N12_ROOT
    if seed == 1:
        return TRANSPORT_SEED1_ROOT
    return TRANSPORT_REPEAT_ROOT


def discovery_root_for(n_agents: int, seed: int) -> Path:
    if n_agents == 12:
        return DISCOVERY_N12_ROOT
    return DISCOVERY_ROOT


def load_es(path: Path) -> dict:
    data = json.loads(path.read_text())
    return {
        "base_reward": float(data["base_checkpoint_metrics"]["mean_reward"]),
        "final_reward": float(data["final_selected_metrics"]["mean_reward"]),
    }


def load_rl(path: Path) -> dict:
    data = json.loads(path.read_text())
    return {
        "base_reward": float(data["base_checkpoint_metrics"]["mean_reward"]),
        "final_reward": float(data["final_best_metrics"]["mean_reward"]),
    }


def collect_env_rows(env_name: str, ns: list[int], seeds: list[int]) -> dict:
    loader_by_method = {"es": load_es, "rl": load_rl}
    out: dict[str, dict[int, dict[str, float | list[dict[str, float]]]]] = {"es": {}, "rl": {}}

    for n_agents in ns:
        for method in ["es", "rl"]:
            per_seed = []
            for seed in seeds:
                if env_name == "transport":
                    root = transport_root_for(n_agents, seed)
                else:
                    root = discovery_root_for(n_agents, seed)
                filename = "phase2_history.json" if method == "es" else "phase2_rl_history.json"
                path = root / f"n{n_agents}" / method / f"seed{seed}" / filename
                if not path.exists():
                    continue
                row = loader_by_method[method](path)
                row["delta_reward"] = row["final_reward"] - row["base_reward"]
                row["seed"] = seed
                per_seed.append(row)

            if not per_seed:
                continue

            deltas = [row["delta_reward"] for row in per_seed]
            finals = [row["final_reward"] for row in per_seed]
            bases = [row["base_reward"] for row in per_seed]
            out[method][n_agents] = {
                "n_seeds": len(per_seed),
                "delta_reward_mean": mean(deltas),
                "delta_reward_std": stdev(deltas),
                "delta_reward_se": stderr(deltas),
                "final_reward_mean": mean(finals),
                "final_reward_se": stderr(finals),
                "base_reward_mean": mean(bases),
                "base_reward_se": stderr(bases),
                "per_seed": per_seed,
            }
    return out


def build_markdown(transport: dict, discovery: dict) -> str:
    lines = [
        "# VMAS Phase 2 Scaling",
        "",
        "Reward gain is measured as `final adapted reward - method-paired frozen backbone reward` for each seed.",
        "",
    ]
    for env_name, rows in [("Transport", transport), ("Discovery", discovery)]:
        lines.append(f"## {env_name}")
        lines.append("")
        lines.append("| N | Method | Seeds | Mean Reward Gain ± SE | Mean Final Reward ± SE | Mean Frozen Reward ± SE |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        all_ns = sorted(set(rows["es"].keys()) | set(rows["rl"].keys()))
        for n_agents in all_ns:
            for method, label in [("es", "ALEC"), ("rl", "MAPPO")]:
                if n_agents not in rows[method]:
                    continue
                item = rows[method][n_agents]
                lines.append(
                    f"| {n_agents} | {label} | {item['n_seeds']} | "
                    f"{item['delta_reward_mean']:+.4f} ± {item['delta_reward_se']:.4f} | "
                    f"{item['final_reward_mean']:.4f} ± {item['final_reward_se']:.4f} | "
                    f"{item['base_reward_mean']:.4f} ± {item['base_reward_se']:.4f} |"
                )
        lines.append("")
    return "\n".join(lines)


def latex_pm(mean_value: float, se_value: float) -> str:
    return rf"${mean_value:.4f} \pm {se_value:.4f}$"


def method_gain(rows: dict, n_agents: int, method: str) -> float:
    baseline = rows[method][n_agents]["base_reward_mean"]
    adapted = rows[method][n_agents]["final_reward_mean"]
    return adapted - baseline


def build_section_patch(transport: dict, discovery: dict) -> str:
    d_gain = {n: {m: method_gain(discovery, n, m) for m in ["es", "rl"]} for n in [2, 4, 8, 12]}
    t_gain = {n: {m: method_gain(transport, n, m) for m in ["es", "rl"]} for n in [2, 4, 8, 12]}

    lines = [
        r"\subsection{ALEC Scaling in VMAS}",
        "",
        (
            r"\paragraph{Experimental Setup} We evaluate post-training adaptation in VMAS Discovery "
            r"and Transport across three seeds at team sizes $N \in \{2,4,8,12\}$. A backbone is "
            r"trained first, after which either ALEC or MAPPO is used for phase-2 adaptation under the "
            r"same budget."
        ),
        "",
        (
            r"\paragraph{VMAS Discovery shows positive gains for both methods, with ALEC slightly "
            r"stronger at larger team sizes.} Reward gain over each method's paired frozen "
            r"checkpoint is positive for both methods at all tested $N$. In VMAS Discovery, ALEC is "
            rf"slightly stronger at $N=4$ ($+{d_gain[4]['es']:.4f}$ vs.\ $+{d_gain[4]['rl']:.4f}$), "
            rf"$N=8$ ($+{d_gain[8]['es']:.4f}$ vs.\ $+{d_gain[8]['rl']:.4f}$), and "
            rf"$N=12$ ($+{d_gain[12]['es']:.4f}$ vs.\ $+{d_gain[12]['rl']:.4f}$), while MAPPO is "
            rf"stronger only at $N=2$ ($+{d_gain[2]['rl']:.4f}$ vs.\ $+{d_gain[2]['es']:.4f}$). "
            r"VMAS Transport is noisier: ALEC is better at "
            rf"$N=2$ ($+{t_gain[2]['es']:.4f}$ vs.\ $+{t_gain[2]['rl']:.4f}$), "
            rf"$N=4$ ($+{t_gain[4]['es']:.4f}$ vs.\ $+{t_gain[4]['rl']:.4f}$), and "
            rf"$N=12$ ($+{t_gain[12]['es']:.4f}$ vs.\ $+{t_gain[12]['rl']:.4f}$), while MAPPO is "
            rf"slightly better at $N=8$ ($+{t_gain[8]['rl']:.4f}$ vs.\ $+{t_gain[8]['es']:.4f}$)."
        ),
        "",
        r"\begin{table}[t]",
        r"\centering",
        (
            r"\caption{VMAS Discovery mean reward by team size $N$. For each method, the paired "
            r"frozen checkpoint mean is reported alongside the phase-2 adapted mean. Values are "
            r"mean $\pm$ standard error; the final column reports the seed count.}"
        ),
        r"\label{tab:vmas_discovery_rewards}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"$N$ & ALEC Base & ALEC & MAPPO Base & MAPPO & Seeds \\",
        r"\midrule",
    ]

    for n_agents in [2, 4, 8, 12]:
        base = discovery["es"][n_agents]
        es = discovery["es"][n_agents]
        rl = discovery["rl"][n_agents]
        lines.append(
            rf"{n_agents} & {latex_pm(base['base_reward_mean'], base['base_reward_se'])} & "
            rf"{latex_pm(es['final_reward_mean'], es['final_reward_se'])} & "
            rf"{latex_pm(rl['base_reward_mean'], rl['base_reward_se'])} & "
            rf"{latex_pm(rl['final_reward_mean'], rl['final_reward_se'])} & "
            rf"{base['n_seeds']} \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
            r"\subsection{VMAS Scaling Transport Results}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            (
            r"\caption{VMAS Transport mean reward by team size $N$. For each method, the paired "
            r"frozen checkpoint mean is reported alongside the phase-2 adapted mean. Values are "
            r"mean $\pm$ standard error; the final column reports the seed count.}"
            ),
            r"\label{tab:vmas_transport_rewards}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"$N$ & ALEC Base & ALEC & MAPPO Base & MAPPO & Seeds \\",
            r"\midrule",
        ]
    )

    for n_agents in [2, 4, 8, 12]:
        base = transport["es"][n_agents]
        es = transport["es"][n_agents]
        rl = transport["rl"][n_agents]
        lines.append(
            rf"{n_agents} & {latex_pm(base['base_reward_mean'], base['base_reward_se'])} & "
            rf"{latex_pm(es['final_reward_mean'], es['final_reward_se'])} & "
            rf"{latex_pm(rl['base_reward_mean'], rl['base_reward_se'])} & "
            rf"{latex_pm(rl['final_reward_mean'], rl['final_reward_se'])} & "
            rf"{base['n_seeds']} \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def build_plot(transport: dict, discovery: dict, save_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )

    colors = {"es": "#1f6feb", "rl": "#d97706"}
    labels = {"es": "ES", "rl": "RL"}

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True, constrained_layout=True)

    for ax, env_name, rows in [
        (axes[0], "VMAS Transport", transport),
        (axes[1], "VMAS Discovery", discovery),
    ]:
        for method in ["es", "rl"]:
            xs = sorted(rows[method].keys())
            ys = [rows[method][x]["delta_reward_mean"] for x in xs]
            errs = [rows[method][x]["delta_reward_std"] for x in xs]
            ax.errorbar(
                xs,
                ys,
                yerr=errs,
                color=colors[method],
                marker="o",
                linewidth=2.0,
                capsize=4,
                label=labels[method],
            )
        ax.axhline(0.0, color="#666", linewidth=1, linestyle="--")
        ax.set_title(env_name)
        ax.set_xlabel("Team Size N")
        ax.set_xticks(sorted(set(rows["es"].keys()) | set(rows["rl"].keys())))
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Reward Gain Over Frozen Backbone")
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle("Phase 2 Scaling on Positive-Sum VMAS Tasks", fontsize=13)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    transport = collect_env_rows("transport", ns=[2, 4, 8, 12], seeds=[1, 2, 3])
    discovery = collect_env_rows("discovery", ns=[2, 4, 8, 12], seeds=[1, 2, 3])

    summary = {"transport": transport, "discovery": discovery}
    (SAVE_DIR / "vmas_phase2_scaling.json").write_text(json.dumps(summary, indent=2))
    (SAVE_DIR / "vmas_phase2_scaling.md").write_text(build_markdown(transport, discovery))
    (SAVE_DIR / "vmas_phase2_scaling_section53.tex").write_text(build_section_patch(transport, discovery))
    build_plot(transport, discovery, SAVE_DIR / "vmas_phase2_scaling.png")

    print(
        json.dumps(
            {
                "figure": str(SAVE_DIR / "vmas_phase2_scaling.png"),
                "markdown": str(SAVE_DIR / "vmas_phase2_scaling.md"),
                "json": str(SAVE_DIR / "vmas_phase2_scaling.json"),
                "section53_tex": str(SAVE_DIR / "vmas_phase2_scaling_section53.tex"),
            }
        )
    )


if __name__ == "__main__":
    main()
