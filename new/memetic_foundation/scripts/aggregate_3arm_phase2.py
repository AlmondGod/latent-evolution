# Aggregate Phase-2 3-arm time-matched results into a single summary table.
# Reads results.json (Arm C) and the Arm A / Arm B run JSONs from the orchestrator output dir.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_arm_a(path: Path) -> dict[str, Any] | None:
    # ALEC writes phase2_history.json with final_best_metrics / final_selected_metrics.
    p = path / "phase2_history.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return {"src": str(p), **data}


def load_arm_b(path: Path) -> dict[str, Any] | None:
    # MAPPO Phase-2 writes phase2_rl_history.json with final_best_metrics.
    p = path / "phase2_rl_history.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return {"src": str(p), **data}


def load_arm_c(path: Path) -> dict[str, Any] | None:
    rj = path / "results.json"
    if not rj.exists():
        return None
    with open(rj) as f:
        return json.load(f)


def summarize_seed(out_dir: Path, seed: int) -> dict[str, Any]:
    seed_dir = out_dir / f"seed{seed}"
    return {
        "seed": seed,
        "arm_a_alec": load_arm_a(seed_dir / "arm_a_alec"),
        "arm_b_mappo": load_arm_b(seed_dir / "arm_b_mappo"),
        "arm_c_es": load_arm_c(seed_dir / "arm_c_es"),
        "arm_d_es": load_arm_c(seed_dir / "arm_d_es_pretrain_matched"),
    }


def extract_final_reward(arm_payload: Any, key_paths: list[list[str]]) -> float | None:
    if not isinstance(arm_payload, dict):
        return None
    for path in key_paths:
        cur: Any = arm_payload
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, (int, float)):
            return float(cur)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    rows = []
    for s in args.seeds:
        rows.append(summarize_seed(args.out_dir, s))

    table = []
    for row in rows:
        seed = row["seed"]
        a_reward = extract_final_reward(row["arm_a_alec"], [
            ["final_selected_metrics", "mean_reward"],
            ["final_best_metrics", "mean_reward"],
        ])
        a_base_reward = extract_final_reward(row["arm_a_alec"], [
            ["base_checkpoint_metrics", "mean_reward"],
        ])
        b_reward = extract_final_reward(row["arm_b_mappo"], [
            ["final_best_metrics", "mean_reward"],
        ])
        b_base_reward = extract_final_reward(row["arm_b_mappo"], [
            ["base_checkpoint_metrics", "mean_reward"],
        ])
        c_reward = extract_final_reward(row["arm_c_es"], [
            ["final_eval", "mean_reward"],
        ])
        c_gens = (row["arm_c_es"] or {}).get("generations_completed")
        c_trans = (row["arm_c_es"] or {}).get("transitions_consumed")
        c_wall = (row["arm_c_es"] or {}).get("wallclock_budget")
        d_reward = extract_final_reward(row["arm_d_es"], [
            ["final_eval", "mean_reward"],
        ])
        d_gens = (row["arm_d_es"] or {}).get("generations_completed")
        d_trans = (row["arm_d_es"] or {}).get("transitions_consumed")
        d_wall = (row["arm_d_es"] or {}).get("wallclock_budget")
        c_plus_d_wall = None
        if c_wall is not None and d_wall is not None:
            c_plus_d_wall = float(c_wall) + float(d_wall)
        table.append({
            "seed": seed,
            "phase1_base_reward": a_base_reward if a_base_reward is not None else b_base_reward,
            "A_alec_reward": a_reward,
            "B_mappo_reward": b_reward,
            "C_es_reward": c_reward,
            "C_generations": c_gens,
            "C_transitions": c_trans,
            "C_wallclock_budget": c_wall,
            "D_es_reward": d_reward,
            "D_generations": d_gens,
            "D_transitions": d_trans,
            "D_wallclock_budget": d_wall,
            "C_plus_D_wallclock_budget": c_plus_d_wall,
        })

    mean_d = None
    if any(r["D_es_reward"] is not None for r in table):
        mean_d = float(np.mean([r["D_es_reward"] for r in table if r["D_es_reward"] is not None]))
    summary = {
        "per_seed": table,
        "mean_reward": {
            "A_alec": float(np.mean([r["A_alec_reward"] for r in table if r["A_alec_reward"] is not None])) if any(r["A_alec_reward"] is not None for r in table) else None,
            "B_mappo": float(np.mean([r["B_mappo_reward"] for r in table if r["B_mappo_reward"] is not None])) if any(r["B_mappo_reward"] is not None for r in table) else None,
            "C_es": float(np.mean([r["C_es_reward"] for r in table if r["C_es_reward"] is not None])) if any(r["C_es_reward"] is not None for r in table) else None,
            "D_es_pretrain_matched": mean_d,
        },
    }
    out_path = args.out_dir / "summary_3arm.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
