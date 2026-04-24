#!/opt/homebrew/bin/python3.9
from __future__ import annotations

import json
import math
import re
from pathlib import Path


ROOT = Path("/Users/almondgod/Repositories/memeplex-capstone")
SAVE_DIR = ROOT / "results" / "ablation_memory_lowrank_section"

PARTIAL_OBS_DIR = ROOT / "checkpoints" / "mpe_spread_partial_obs"

TRANSPORT_SEED1_ROOT = ROOT / "results" / "vmas_phase2_linear_a_transport_50k_seed1_partial" / "linear_a"
TRANSPORT_REPEAT_ROOT = ROOT / "results" / "vmas_phase2_linear_a_transport_50k_repeat" / "linear_a"
TRANSPORT_N12_ROOT = ROOT / "results" / "vmas_phase2_linear_a_transport_50k_n12" / "linear_a"
DISCOVERY_ROOT = ROOT / "results" / "vmas_phase2_linear_a_discovery_50k" / "linear_a"
DISCOVERY_N12_ROOT = ROOT / "results" / "vmas_phase2_linear_a_discovery_50k_n12" / "linear_a"


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return float(math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1)))


def stderr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(stdev(values) / math.sqrt(len(values)))


def fmt_pm(value: float, se: float, digits: int) -> str:
    return f"{value:.{digits}f} ± {se:.{digits}f}"


def latex_pm(value: float, se: float, digits: int) -> str:
    return rf"${value:.{digits}f} \pm {se:.{digits}f}$"


def parse_partial_obs_log(path: Path) -> tuple[float, float]:
    pattern = re.compile(r"\[Eval\] Step (\d+) \| reward=([-\d.]+).* dist=([-\d.]+)")
    last_reward = None
    last_dist = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                last_reward = float(match.group(2))
                last_dist = float(match.group(3))
    if last_reward is None or last_dist is None:
        raise RuntimeError(f"No eval lines found in {path}")
    return last_reward, last_dist


def collect_partial_obs_scaling() -> dict[str, dict[int, dict[str, float]]]:
    variants = ["baseline", "memory_only", "full_gated"]
    ns = [3, 5, 8]
    out: dict[str, dict[int, dict[str, float]]] = {}
    for variant in variants:
        out[variant] = {}
        for n_agents in ns:
            rewards = []
            dists = []
            for log_path in sorted(PARTIAL_OBS_DIR.glob(f"{variant}_n{n_agents}_seed*.log")):
                reward, dist = parse_partial_obs_log(log_path)
                rewards.append(reward)
                dists.append(dist)
            if not rewards:
                continue
            out[variant][n_agents] = {
                "n_seeds": len(rewards),
                "reward_mean": mean(rewards),
                "reward_se": stderr(rewards),
                "dist_mean": mean(dists),
                "dist_se": stderr(dists),
            }
    return out


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


def load_elapsed(path: Path) -> float:
    data = json.loads(path.read_text())
    return float(data["elapsed_seconds"])


def collect_wallclock_groups() -> dict[str, dict[str, float]]:
    grouped: dict[str, tuple[list[float], list[float]]] = {
        "MPE": ([], []),
        "VMAS Discovery": ([], []),
        "VMAS Transport": ([], []),
    }

    es_times, rl_times = grouped["MPE"]
    for seed in [1, 2, 3]:
        es_times.append(
            load_elapsed(
                ROOT
                / "results"
                / f"memetic_phase2_lowrankz_rewardonly_mpe_n8_seed{seed}_g20_p16_archive8_final64"
                / "phase2_history.json"
            )
        )
        rl_times.append(
            load_elapsed(
                ROOT
                / "results"
                / f"memetic_phase2_rl_seed{seed}_budget768k_archive8_final64"
                / "phase2_rl_history.json"
            )
        )

    for env_name, label in [("discovery", "VMAS Discovery"), ("transport", "VMAS Transport")]:
        es_times, rl_times = grouped[label]
        for n_agents in [2, 4, 8, 12]:
            for seed in [1, 2, 3]:
                root = transport_root_for(n_agents, seed) if env_name == "transport" else discovery_root_for(n_agents, seed)
                es_times.append(load_elapsed(root / f"n{n_agents}" / "es" / f"seed{seed}" / "phase2_history.json"))
                rl_times.append(load_elapsed(root / f"n{n_agents}" / "rl" / f"seed{seed}" / "phase2_rl_history.json"))

    out: dict[str, dict[str, float]] = {}
    overall_es: list[float] = []
    overall_rl: list[float] = []
    for label, (es_times, rl_times) in grouped.items():
        overall_es.extend(es_times)
        overall_rl.extend(rl_times)
        out[label] = {
            "n_pairs": len(es_times),
            "es_mean": mean(es_times),
            "es_se": stderr(es_times),
            "rl_mean": mean(rl_times),
            "rl_se": stderr(rl_times),
            "es_faster": sum(es < rl for es, rl in zip(es_times, rl_times)),
        }
    out["Overall"] = {
        "n_pairs": len(overall_es),
        "es_mean": mean(overall_es),
        "es_se": stderr(overall_es),
        "rl_mean": mean(overall_rl),
        "rl_se": stderr(overall_rl),
        "es_faster": sum(es < rl for es, rl in zip(overall_es, overall_rl)),
    }
    return out


def collect_lowrank_ablation() -> list[dict[str, float | int | str]]:
    variants = [
        (
            "with $z$, low-rank update",
            ROOT / "results" / "vmas_phase2_linear_a_discovery_50k_n12" / "linear_a" / "n12" / "es",
        ),
        (
            "without $z$",
            ROOT / "results" / "vmas_phase2_linear_a_discovery_50k_n12_noz_es",
        ),
        (
            "with $z$, dense update",
            ROOT / "results" / "vmas_phase2_linear_a_discovery_50k_n12_dense_es",
        ),
    ]

    rows = []
    for label, base_dir in variants:
        baselines = []
        finals = []
        deltas = []
        times = []
        for seed in [1, 2, 3]:
            if base_dir.name == "es":
                path = base_dir / f"seed{seed}" / "phase2_history.json"
            else:
                path = base_dir / f"seed{seed}" / "phase2_history.json"
            data = json.loads(path.read_text())
            baseline = float(data["base_checkpoint_metrics"]["mean_reward"])
            final = float(data["final_selected_metrics"]["mean_reward"])
            baselines.append(baseline)
            finals.append(final)
            deltas.append(final - baseline)
            times.append(float(data["elapsed_seconds"]))
        rows.append(
            {
                "label": label,
                "n_seeds": len(deltas),
                "baseline_mean": mean(baselines),
                "baseline_se": stderr(baselines),
                "final_mean": mean(finals),
                "final_se": stderr(finals),
                "delta_mean": mean(deltas),
                "delta_se": stderr(deltas),
                "time_mean": mean(times),
                "time_se": stderr(times),
                "improved": sum(delta > 0 for delta in deltas),
            }
        )
    return rows


def build_markdown(memory_rows: dict, wallclock_rows: dict, lowrank_rows: list[dict]) -> str:
    lines = [
        "# Ablation Summary",
        "",
        "## Paper-ready text",
        "",
        (
            "Persistent memory is the most stable simple-spread variant as team size grows. "
            "On partial-observation MPE simple_spread, `memory_only` keeps final mean coverage distance "
            "low at every tested team size (`0.678 ± 0.059`, `0.695 ± 0.180`, `0.613 ± 0.065` at "
            "`N=3,5,8`; `n=6` each), while the memory-free baseline rises to "
            "`9.815 ± 9.203`, `16.325 ± 12.717`, and `13.425 ± 12.945`. The SGD-trained "
            "`full_gated` memory+communication variant is less stable still, reaching "
            "`10.225 ± 9.575`, `38.275 ± 16.793`, and `26.127 ± 23.589` "
            "(`n=6,6,3`). Final rewards show the same pattern: `memory_only` remains between "
            "`-646.90 ± 31.55` and `-1200.81 ± 80.29`, whereas baseline and full_gated are both "
            "much worse and much more variable."
        ),
        "",
        (
            "On a single Apple M3 Pro MacBook Pro (12 CPU cores, 36 GB unified memory; CPU execution), "
            "ALEC is faster overall than matched RL adaptation but not in every VMAS transport setting. "
            "Across the 27 paired phase-2 runs in the current artifact set, ALEC averages "
            f"`{wallclock_rows['Overall']['es_mean']:.1f} ± {wallclock_rows['Overall']['es_se']:.1f}` seconds "
            f"versus `{wallclock_rows['Overall']['rl_mean']:.1f} ± {wallclock_rows['Overall']['rl_se']:.1f}` for RL, "
            f"and is faster in `{wallclock_rows['Overall']['es_faster']}/{wallclock_rows['Overall']['n_pairs']}` pairings."
        ),
        "",
        (
            "Low-rank state updates are slightly better and more stable than the dense alternative on VMAS "
            "Discovery at `N=12`. The standard low-rank `z` update reaches reward gain "
            f"`{lowrank_rows[0]['delta_mean']:+.4f} ± {lowrank_rows[0]['delta_se']:.4f}` and improves in "
            f"`{lowrank_rows[0]['improved']}/{lowrank_rows[0]['n_seeds']}` seeds, compared with "
            f"`{lowrank_rows[2]['delta_mean']:+.4f} ± {lowrank_rows[2]['delta_se']:.4f}` and "
            f"`{lowrank_rows[2]['improved']}/{lowrank_rows[2]['n_seeds']}` for dense updates and "
            f"`{lowrank_rows[1]['delta_mean']:+.4f} ± {lowrank_rows[1]['delta_se']:.4f}` and "
            f"`{lowrank_rows[1]['improved']}/{lowrank_rows[1]['n_seeds']}` with `z` disabled. "
            "It is also materially cheaper wall-clock."
        ),
        "",
        "## Memory scaling table",
        "",
        "| N | Baseline Reward | Memory-Only Reward | Memory+Comm Reward | Baseline mean_dist | Memory-Only mean_dist | Memory+Comm mean_dist | Seeds (B/M/F) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for n_agents in [3, 5, 8]:
        baseline = memory_rows["baseline"][n_agents]
        memory = memory_rows["memory_only"][n_agents]
        full = memory_rows["full_gated"][n_agents]
        lines.append(
            f"| {n_agents} | "
            f"{fmt_pm(baseline['reward_mean'], baseline['reward_se'], 2)} | "
            f"{fmt_pm(memory['reward_mean'], memory['reward_se'], 2)} | "
            f"{fmt_pm(full['reward_mean'], full['reward_se'], 2)} | "
            f"{fmt_pm(baseline['dist_mean'], baseline['dist_se'], 3)} | "
            f"{fmt_pm(memory['dist_mean'], memory['dist_se'], 3)} | "
            f"{fmt_pm(full['dist_mean'], full['dist_se'], 3)} | "
            f"{baseline['n_seeds']}/{memory['n_seeds']}/{full['n_seeds']} |"
        )

    lines.extend(
        [
            "",
            "## Phase-2 wall-clock table",
            "",
            "| Setting | ALEC Time (s) | MAPPO Time (s) | Paired Runs | ALEC Faster |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label in ["MPE", "VMAS Discovery", "VMAS Transport", "Overall"]:
        row = wallclock_rows[label]
        lines.append(
            f"| {label} | "
            f"{fmt_pm(row['es_mean'], row['es_se'], 1)} | "
            f"{fmt_pm(row['rl_mean'], row['rl_se'], 1)} | "
            f"{row['n_pairs']} | "
            f"{row['es_faster']}/{row['n_pairs']} |"
        )

    lines.extend(
        [
            "",
            "## Low-rank ablation table",
            "",
            "| Variant | Baseline Reward | Final Reward | Reward Gain | Time (s) | Improved | Seeds |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in lowrank_rows:
        lines.append(
            f"| {row['label']} | "
            f"{fmt_pm(row['baseline_mean'], row['baseline_se'], 4)} | "
            f"{fmt_pm(row['final_mean'], row['final_se'], 4)} | "
            f"{row['delta_mean']:+.4f} ± {row['delta_se']:.4f} | "
            f"{fmt_pm(row['time_mean'], row['time_se'], 1)} | "
            f"{row['improved']}/{row['n_seeds']} | "
            f"{row['n_seeds']} |"
        )

    lines.append("")
    return "\n".join(lines)


def build_tex(memory_rows: dict, wallclock_rows: dict, lowrank_rows: list[dict]) -> str:
    lines = [
        r"\subsection{Ablations on Memory and Low-rank updates}",
        "",
        (
            r"\paragraph{Persistent memory helps scalability.} On partial-observation MPE "
            r"simple\_spread, the memory-only variant is the most stable option as team size grows. "
            r"It keeps final mean coverage distance low at every tested $N$ "
            r"($0.678 \pm 0.059$, $0.695 \pm 0.180$, $0.613 \pm 0.065$ at $N=3,5,8$; $n=6$ each), "
            r"whereas the memory-free baseline rises to $9.815 \pm 9.203$, $16.325 \pm 12.717$, and "
            r"$13.425 \pm 12.945$. The SGD-trained memory+communication variant is less stable still, "
            r"reaching $10.225 \pm 9.575$, $38.275 \pm 16.793$, and $26.127 \pm 23.589$ "
            r"($n=6,6,3$). Final rewards show the same pattern: memory-only stays between "
            r"$-646.90 \pm 31.55$ and $-1200.81 \pm 80.29$, while the baseline and "
            r"memory+communication variants are both worse and far more variable."
        ),
        "",
        (
            r"\paragraph{Improved wall-clock behavior shows for ALEC relative to matched RL adaptation.} "
            r"On a single Apple M3 Pro MacBook Pro (12 CPU cores, 36\,GB unified memory; CPU execution), "
            r"ALEC is faster overall than matched RL adaptation, though not in every VMAS Transport "
            r"setting. Across the 27 paired phase-2 runs in the current artifact set, ALEC averaged "
            rf"${wallclock_rows['Overall']['es_mean']:.1f} \pm {wallclock_rows['Overall']['es_se']:.1f}$ "
            r"seconds versus "
            rf"${wallclock_rows['Overall']['rl_mean']:.1f} \pm {wallclock_rows['Overall']['rl_se']:.1f}$ "
            r"for RL, and ALEC finished faster in "
            rf"{wallclock_rows['Overall']['es_faster']}/{wallclock_rows['Overall']['n_pairs']} pairings."
        ),
        "",
        (
            r"\paragraph{Low-rank state updates appear slightly better and more stable than dense alternatives.} "
            r"On VMAS Discovery at $N=12$, the standard low-rank $z$ update achieves reward gain "
            rf"${lowrank_rows[0]['delta_mean']:+.4f} \pm {lowrank_rows[0]['delta_se']:.4f}$ and "
            rf"improves in {lowrank_rows[0]['improved']}/{lowrank_rows[0]['n_seeds']} seeds, compared "
            r"with "
            rf"${lowrank_rows[2]['delta_mean']:+.4f} \pm {lowrank_rows[2]['delta_se']:.4f}$ and "
            rf"{lowrank_rows[2]['improved']}/{lowrank_rows[2]['n_seeds']} seeds for dense updates, and "
            rf"${lowrank_rows[1]['delta_mean']:+.4f} \pm {lowrank_rows[1]['delta_se']:.4f}$ and "
            rf"{lowrank_rows[1]['improved']}/{lowrank_rows[1]['n_seeds']} seeds with $z$ disabled. "
            r"The low-rank variant is also materially cheaper wall-clock."
        ),
        "",
        r"\begin{table}[t]",
        r"\centering",
        (
            r"\caption{Partial-observation MPE simple\_spread scaling ablation. Values are mean $\pm$ "
            r"standard error over seeds; lower mean\_dist is better. The final column reports seeds as "
            r"baseline / memory-only / memory+communication. The memory+communication row at $N=8$ "
            r"uses 3 seeds; all other entries use 6.}"
        ),
        r"\label{tab:mpe_memory_scaling_ablation}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"$N$ & Baseline Reward & Memory-Only Reward & Memory+Comm Reward & Baseline mean\_dist & Memory-Only mean\_dist & Memory+Comm mean\_dist & Seeds \\",
        r"\midrule",
    ]

    for n_agents in [3, 5, 8]:
        baseline = memory_rows["baseline"][n_agents]
        memory = memory_rows["memory_only"][n_agents]
        full = memory_rows["full_gated"][n_agents]
        lines.append(
            rf"{n_agents} & "
            rf"{latex_pm(baseline['reward_mean'], baseline['reward_se'], 2)} & "
            rf"{latex_pm(memory['reward_mean'], memory['reward_se'], 2)} & "
            rf"{latex_pm(full['reward_mean'], full['reward_se'], 2)} & "
            rf"{latex_pm(baseline['dist_mean'], baseline['dist_se'], 3)} & "
            rf"{latex_pm(memory['dist_mean'], memory['dist_se'], 3)} & "
            rf"{latex_pm(full['dist_mean'], full['dist_se'], 3)} & "
            rf"{baseline['n_seeds']}/{memory['n_seeds']}/{full['n_seeds']} \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            (
                r"\caption{Matched phase-2 wall-clock by domain. Values are mean $\pm$ standard error "
                r"in seconds over paired runs on a single Apple M3 Pro MacBook Pro (12 CPU cores, "
                r"36\,GB unified memory) using CPU execution.}"
            ),
            r"\label{tab:phase2_wallclock_ablation}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Setting & ALEC Time (s) & MAPPO Time (s) & Pairs & ALEC Faster \\",
            r"\midrule",
        ]
    )

    for label in ["MPE", "VMAS Discovery", "VMAS Transport", "Overall"]:
        row = wallclock_rows[label]
        lines.append(
            rf"{label} & "
            rf"{latex_pm(row['es_mean'], row['es_se'], 1)} & "
            rf"{latex_pm(row['rl_mean'], row['rl_se'], 1)} & "
            rf"{row['n_pairs']} & "
            rf"{row['es_faster']}/{row['n_pairs']} \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{VMAS Discovery $N=12$ state-update ablation for ALEC. Values are mean $\pm$ standard error over seeds.}",
            r"\label{tab:vmas_lowrank_ablation}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Variant & Baseline Reward & Final Reward & Reward Gain & Time (s) & Improved & Seeds \\",
            r"\midrule",
        ]
    )

    for row in lowrank_rows:
        lines.append(
            rf"{row['label']} & "
            rf"{latex_pm(row['baseline_mean'], row['baseline_se'], 4)} & "
            rf"{latex_pm(row['final_mean'], row['final_se'], 4)} & "
            rf"{latex_pm(row['delta_mean'], row['delta_se'], 4)} & "
            rf"{latex_pm(row['time_mean'], row['time_se'], 1)} & "
            rf"{row['improved']}/{row['n_seeds']} & "
            rf"{row['n_seeds']} \\"
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


def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    memory_rows = collect_partial_obs_scaling()
    wallclock_rows = collect_wallclock_groups()
    lowrank_rows = collect_lowrank_ablation()

    payload = {
        "memory_scaling": memory_rows,
        "wallclock": wallclock_rows,
        "lowrank_ablation": lowrank_rows,
    }
    (SAVE_DIR / "ablation_summary.json").write_text(json.dumps(payload, indent=2))
    (SAVE_DIR / "ablation_summary.md").write_text(build_markdown(memory_rows, wallclock_rows, lowrank_rows))
    (SAVE_DIR / "ablation_section.tex").write_text(build_tex(memory_rows, wallclock_rows, lowrank_rows))

    print(
        json.dumps(
            {
                "json": str(SAVE_DIR / "ablation_summary.json"),
                "markdown": str(SAVE_DIR / "ablation_summary.md"),
                "tex": str(SAVE_DIR / "ablation_section.tex"),
            }
        )
    )


if __name__ == "__main__":
    main()
