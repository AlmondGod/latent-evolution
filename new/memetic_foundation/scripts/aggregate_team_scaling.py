#!/usr/bin/env python3
# Aggregate the 3-arm Phase-2 results across team sizes (N=2/4/8 vs 4 enemies).
# Produces a single summary table + mean/delta over the Phase-1 paired baseline.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_summary(out_dir: Path) -> dict[str, Any] | None:
    p = out_dir / "summary_3arm.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def compute_deltas(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary["per_seed"]
    deltas = {"A_alec": [], "B_mappo": [], "C_es": []}
    for r in rows:
        base = r["phase1_base_reward"]
        if base is None:
            continue
        for arm in deltas.keys():
            v = r.get(f"{arm.split('_')[0]}_{arm.split('_')[1]}_reward")
            if v is None:
                continue
            deltas[arm].append(v - base)
    return {
        arm: {
            "mean_delta": float(np.mean(vs)) if vs else None,
            "se_delta": float(np.std(vs) / np.sqrt(len(vs))) if len(vs) > 1 else None,
            "per_seed_delta": vs,
        }
        for arm, vs in deltas.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n2-dir", type=Path, default=Path("results/smacv2_phase2_3arm_2v4"))
    parser.add_argument("--n4-dir", type=Path, default=Path("results/smacv2_phase2_3arm_4v4"))
    parser.add_argument("--n8-dir", type=Path, default=Path("results/smacv2_phase2_3arm_8v4"))
    parser.add_argument("--out", type=Path, default=Path("results/smacv2_team_scaling_summary.json"))
    args = parser.parse_args()

    rollups: dict[str, dict[str, Any]] = {}
    for n, d in [(2, args.n2_dir), (4, args.n4_dir), (8, args.n8_dir)]:
        s = load_summary(d)
        if s is None:
            print(f"[warn] missing summary for N={n} at {d}")
            continue
        base_per_seed = [r["phase1_base_reward"] for r in s["per_seed"]]
        base_mean = float(np.mean([b for b in base_per_seed if b is not None])) if base_per_seed else None
        rollups[f"N={n}"] = {
            "base_mean": base_mean,
            "abs_mean": s["mean_reward"],
            "deltas": compute_deltas(s),
            "per_seed": s["per_seed"],
        }

    # Pretty-print a table
    print()
    print("=" * 88)
    print("SMACv2 Terran team-scaling (n_enemies=4 fixed). 30 min/arm/seed wall-clock budget.")
    print("=" * 88)
    print(f"{'N':>3} {'base':>9} {'A(ALEC)':>9} {'B(MAPPO)':>10} {'C(ES)':>9} {'ΔA':>9} {'ΔB':>9} {'ΔC':>9}")
    print("-" * 88)
    for n_label in ["N=2", "N=4", "N=8"]:
        if n_label not in rollups:
            continue
        r = rollups[n_label]
        b = r["base_mean"]
        am = r["abs_mean"]
        d = r["deltas"]
        print(f"{n_label:>3} {b:>9.3f} {am['A_alec']:>9.3f} {am['B_mappo']:>10.3f} "
              f"{am['C_es']:>9.3f} "
              f"{d['A_alec']['mean_delta']:>+9.3f} "
              f"{d['B_mappo']['mean_delta']:>+9.3f} "
              f"{d['C_es']['mean_delta']:>+9.3f}")
    print("=" * 88)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rollups, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
