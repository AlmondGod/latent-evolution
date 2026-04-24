#!/opt/homebrew/bin/python3.9
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Section 5.4 alignment between search-time fitness and held-out reward."
    )
    parser.add_argument(
        "--alec-glob",
        type=str,
        default="results/memetic_phase2_lowrankz_rewardonly_mpe_n8_seed*_g20_p16_archive8_final64/phase2_history.json",
        help="Glob for ALEC phase-2 history files.",
    )
    parser.add_argument(
        "--rl-glob",
        type=str,
        default="results/memetic_phase2_rl_seed*_budget768k_archive8_final64/phase2_rl_history.json",
        help="Glob for matched-budget RL phase-2 history files.",
    )
    parser.add_argument(
        "--heldout-key",
        type=str,
        default="mean_reward",
        help="Held-out metric key inside each final candidate row.",
    )
    parser.add_argument(
        "--pair-key",
        choices=["seed", "path"],
        default="seed",
        help="How to pair ALEC and RL runs for significance testing.",
    )
    parser.add_argument(
        "--permutation-samples",
        type=int,
        default=200000,
        help="Monte Carlo samples for the paired permutation test when exact enumeration is too large.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for Monte Carlo permutation testing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase2_alignment_section54"),
        help="Directory where JSON/Markdown summaries will be written.",
    )
    return parser.parse_args()


def average_ranks(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    while i < order.size:
        j = i
        while j + 1 < order.size and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(x: Iterable[float], y: Iterable[float]) -> float:
    rx = average_ranks(x)
    ry = average_ranks(y)
    rx_centered = rx - rx.mean()
    ry_centered = ry - ry.mean()
    denom = math.sqrt(float(np.dot(rx_centered, rx_centered) * np.dot(ry_centered, ry_centered)))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(rx_centered, ry_centered) / denom)


def standard_error(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size < 2:
        return 0.0
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def mean_of(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def exact_two_sided_sign_flip_pvalue(differences: np.ndarray) -> float:
    observed = abs(float(differences.mean()))
    count = 0
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=differences.size):
        stat = abs(float((differences * np.asarray(signs, dtype=np.float64)).mean()))
        count += 1
        if stat >= observed - 1e-12:
            extreme += 1
    return float(extreme / count)


def paired_permutation_pvalue(
    differences: Iterable[float],
    permutation_samples: int,
    rng: np.random.RandomState,
) -> tuple[float, str]:
    diffs = np.asarray(list(differences), dtype=np.float64)
    if diffs.size == 0:
        return float("nan"), "none"
    if np.allclose(diffs, 0.0):
        return 1.0, "degenerate"
    if diffs.size <= 20:
        return exact_two_sided_sign_flip_pvalue(diffs), "exact_sign_flip"

    observed = abs(float(diffs.mean()))
    flips = rng.choice((-1.0, 1.0), size=(permutation_samples, diffs.size))
    stats = np.abs((flips * diffs[None, :]).mean(axis=1))
    extreme = int(np.sum(stats >= observed - 1e-12))
    pvalue = float((extreme + 1) / (permutation_samples + 1))
    return pvalue, "monte_carlo_sign_flip"


def exact_mcnemar_pvalue(left_success: Iterable[bool], right_success: Iterable[bool]) -> tuple[float, int, int]:
    left_arr = np.asarray(list(left_success), dtype=bool)
    right_arr = np.asarray(list(right_success), dtype=bool)
    only_left = int(np.sum(left_arr & ~right_arr))
    only_right = int(np.sum(~left_arr & right_arr))
    discordant = only_left + only_right
    if discordant == 0:
        return 1.0, only_left, only_right

    tail = 0.0
    threshold = min(only_left, only_right)
    for k in range(threshold + 1):
        tail += math.comb(discordant, k)
    pvalue = min(1.0, 2.0 * tail / (2 ** discordant))
    return float(pvalue), only_left, only_right


def percent(value: float) -> float:
    return 100.0 * value


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def format_percent(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{percent(value):.1f}"


def load_run(path: Path, heldout_key: str, pair_key: str) -> dict:
    payload = json.loads(path.read_text())
    if "final_heldout_rows" not in payload:
        raise ValueError(f"{path} does not contain final_heldout_rows")

    rows = payload["final_heldout_rows"]
    if not rows:
        raise ValueError(f"{path} has an empty final_heldout_rows list")

    if any(heldout_key not in row for row in rows):
        raise KeyError(f"{path} is missing held-out key '{heldout_key}' in one or more rows")

    search_scores = np.asarray([row["search_fitness"] for row in rows], dtype=np.float64)
    heldout_scores = np.asarray([row[heldout_key] for row in rows], dtype=np.float64)
    top_search = int(np.argmax(search_scores))
    top_heldout = int(np.argmax(heldout_scores))

    run_id = payload.get("seed")
    if pair_key == "path":
        run_id = str(path.parent)
    if run_id is None:
        raise KeyError(f"{path} does not expose a usable pairing key")

    return {
        "id": str(run_id),
        "path": str(path),
        "seed": payload.get("seed"),
        "n_candidates": int(len(rows)),
        "spearman": spearman_correlation(search_scores, heldout_scores),
        "top_match": bool(top_search == top_heldout),
        "top_search_index": top_search,
        "top_heldout_index": top_heldout,
        "top_search_score": float(search_scores[top_search]),
        "top_heldout_score": float(heldout_scores[top_heldout]),
    }


def load_runs(glob_pattern: str, heldout_key: str, pair_key: str) -> dict[str, dict]:
    runs = {}
    for path in sorted(Path().glob(glob_pattern)):
        run = load_run(path=path, heldout_key=heldout_key, pair_key=pair_key)
        if run["id"] in runs:
            raise ValueError(f"Duplicate pairing key '{run['id']}' for pattern {glob_pattern}")
        runs[run["id"]] = run
    if not runs:
        raise FileNotFoundError(f"No runs matched {glob_pattern}")
    return runs


def summarize_method(runs: list[dict]) -> dict:
    spearman_values = [run["spearman"] for run in runs]
    top_match_values = [float(run["top_match"]) for run in runs]
    return {
        "n_runs": len(runs),
        "mean_spearman": mean_of(spearman_values),
        "se_spearman": standard_error(spearman_values),
        "mean_top_match": mean_of(top_match_values),
        "se_top_match": standard_error(top_match_values),
    }


def build_paper_sentence(alec_summary: dict, rl_summary: dict, corr_test: dict, top_test: dict) -> str:
    return (
        "The mean within-run Spearman correlation between search fitness and held-out reward was "
        f"${format_float(alec_summary['mean_spearman'])} \\pm {format_float(alec_summary['se_spearman'])}$ "
        f"for ALEC versus ${format_float(rl_summary['mean_spearman'])} \\pm {format_float(rl_summary['se_spearman'])}$ "
        f"for the matched-budget RL baseline (mean $\\pm$ s.e.m.; paired sign-flip test $p={format_float(corr_test['pvalue'])}$). "
        "The top search-time candidate remained the top held-out candidate in "
        f"${format_percent(alec_summary['mean_top_match'])}\\% \\pm {format_percent(alec_summary['se_top_match'])}\\%$ "
        f"versus ${format_percent(rl_summary['mean_top_match'])}\\% \\pm {format_percent(rl_summary['se_top_match'])}\\%$ of runs "
        f"(paired exact McNemar test $p={format_float(top_test['pvalue'])}$)."
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)

    alec_runs = load_runs(args.alec_glob, heldout_key=args.heldout_key, pair_key=args.pair_key)
    rl_runs = load_runs(args.rl_glob, heldout_key=args.heldout_key, pair_key=args.pair_key)

    paired_ids = sorted(set(alec_runs) & set(rl_runs), key=lambda item: str(item))
    if not paired_ids:
        raise RuntimeError("No paired ALEC/RL runs were found after applying the pairing key.")

    missing_alec = sorted(set(rl_runs) - set(alec_runs))
    missing_rl = sorted(set(alec_runs) - set(rl_runs))

    paired_alec = [alec_runs[run_id] for run_id in paired_ids]
    paired_rl = [rl_runs[run_id] for run_id in paired_ids]

    alec_summary = summarize_method(paired_alec)
    rl_summary = summarize_method(paired_rl)

    spearman_diffs = np.asarray(
        [alec_runs[run_id]["spearman"] - rl_runs[run_id]["spearman"] for run_id in paired_ids],
        dtype=np.float64,
    )
    corr_pvalue, corr_method = paired_permutation_pvalue(
        differences=spearman_diffs,
        permutation_samples=args.permutation_samples,
        rng=rng,
    )
    top_pvalue, only_alec, only_rl = exact_mcnemar_pvalue(
        left_success=[alec_runs[run_id]["top_match"] for run_id in paired_ids],
        right_success=[rl_runs[run_id]["top_match"] for run_id in paired_ids],
    )

    summary = {
        "config": {
            "alec_glob": args.alec_glob,
            "rl_glob": args.rl_glob,
            "heldout_key": args.heldout_key,
            "pair_key": args.pair_key,
            "permutation_samples": args.permutation_samples,
            "seed": args.seed,
        },
        "paired_ids": paired_ids,
        "unpaired": {
            "missing_alec": missing_alec,
            "missing_rl": missing_rl,
        },
        "alec_runs": paired_alec,
        "rl_runs": paired_rl,
        "summary": {
            "alec": alec_summary,
            "rl": rl_summary,
            "paired_spearman_test": {
                "method": corr_method,
                "pvalue": corr_pvalue,
                "mean_difference": mean_of(spearman_diffs),
                "se_difference": standard_error(spearman_diffs),
            },
            "paired_top_match_test": {
                "method": "exact_mcnemar",
                "pvalue": top_pvalue,
                "alec_only_successes": only_alec,
                "rl_only_successes": only_rl,
            },
        },
    }
    summary["paper_sentence"] = build_paper_sentence(
        alec_summary=alec_summary,
        rl_summary=rl_summary,
        corr_test=summary["summary"]["paired_spearman_test"],
        top_test=summary["summary"]["paired_top_match_test"],
    )

    json_path = args.output_dir / "alignment_summary.json"
    md_path = args.output_dir / "alignment_summary.md"

    json_path.write_text(json.dumps(summary, indent=2))
    md_lines = [
        "# Section 5.4 Alignment Summary",
        "",
        f"- Paired runs: {len(paired_ids)}",
        f"- ALEC mean Spearman: {format_float(alec_summary['mean_spearman'])} +/- {format_float(alec_summary['se_spearman'])}",
        f"- RL mean Spearman: {format_float(rl_summary['mean_spearman'])} +/- {format_float(rl_summary['se_spearman'])}",
        (
            "- Paired Spearman test: "
            f"{summary['summary']['paired_spearman_test']['method']} "
            f"(p={format_float(summary['summary']['paired_spearman_test']['pvalue'])})"
        ),
        f"- ALEC top-match rate: {format_percent(alec_summary['mean_top_match'])}% +/- {format_percent(alec_summary['se_top_match'])}%",
        f"- RL top-match rate: {format_percent(rl_summary['mean_top_match'])}% +/- {format_percent(rl_summary['se_top_match'])}%",
        (
            "- Paired top-match test: exact McNemar "
            f"(p={format_float(summary['summary']['paired_top_match_test']['pvalue'])}, "
            f"ALEC-only={only_alec}, RL-only={only_rl})"
        ),
        "",
        "Paper-ready sentence:",
        "",
        summary["paper_sentence"],
        "",
    ]
    md_path.write_text("\n".join(md_lines))

    print(json.dumps({"saved_json": str(json_path), "saved_md": str(md_path)}))
    print(summary["paper_sentence"])


if __name__ == "__main__":
    main()
