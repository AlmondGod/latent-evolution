from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from new.memetic_foundation.models.frozen_attention_hu_actor import (
    FrozenAttentionHUActorBackbone,
)
from new.memetic_foundation.scripts.attention_viz_utils import (
    collect_attention_stats,
    collect_attention_stats_by_smac_role,
    episode_seed_list,
    load_adapter,
    render_mean_figure,
    render_residual_figure,
    render_sample_figure,
    serialize_example,
    uniform_offdiag_attention,
    uniform_role_attention,
)
from new.memetic_foundation.training.env_utils import RACE_CONFIGS, make_env

LABEL_A = "ALEC"
LABEL_B = "MAPPO"
ARM_A = "arm_a_alec"
ARM_B = "arm_b_mappo"


def race_role_labels(race: str) -> list[str]:
    names = RACE_CONFIGS[race.lower()]["unit_types"]
    return [n.replace("_", " ").title() for n in names]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ALEC-vs-MAPPO communication attention visualizations on SMACv2."
    )
    parser.add_argument(
        "--phase1-root",
        type=Path,
        default=Path("/workspace/latent-evolution/results"),
        help="Folder containing smacv2_phase1_attn_<NvE> directories.",
    )
    parser.add_argument(
        "--phase2-root",
        type=Path,
        default=Path("/workspace/latent-evolution/results"),
        help="Folder containing smacv2_phase2_3arm_<NvE> directories.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("/workspace/latent-evolution/results/smacv2_phase2_attention_viz"),
    )
    parser.add_argument("--sizes", type=str, nargs="+", default=["2v4", "4v4", "8v4"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--race", choices=["terran", "protoss", "zerg"], default="terran")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--sc2-path", type=str, default="/workspace/StarCraftII")
    parser.add_argument(
        "--pooling",
        choices=["unit_type", "slot"],
        default="unit_type",
        help="unit_type: pool attention to receiver-role × sender-role (SMAC capabilities). "
        "slot: raw agent index matrix (mixes identities across random teams).",
    )
    return parser.parse_args()


def parse_size(size: str) -> tuple[int, int]:
    n_units, n_enemies = size.split("v")
    return int(n_units), int(n_enemies)


def build_backbone(env, checkpoint_path: Path, device: torch.device) -> FrozenAttentionHUActorBackbone:
    info = env.get_env_info()
    backbone = FrozenAttentionHUActorBackbone.from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        obs_dim=info["obs_shape"],
        state_dim=info["state_shape"],
        n_actions=info["n_actions"],
        n_agents=info["n_agents"],
        map_location=str(device),
    )
    return backbone.to(device).freeze()


def aggregate(per_seed: dict, method: str, seeds: list[int], key: str) -> np.ndarray:
    return np.mean([np.asarray(per_seed[method][str(s)][key]) for s in seeds], axis=0)


def best_example(per_seed: dict, method: str, seeds: list[int], key: str) -> dict:
    return min(
        [per_seed[method][str(s)][key] for s in seeds],
        key=lambda item: float(item["mean_receiver_entropy"]),
    )


def render_size_figures(
    size: str,
    n_units: int,
    per_seed: dict,
    seeds: list[int],
    save_dir: Path,
    pooling: str,
    race: str,
    role_labels: list[str],
) -> dict:
    a_mean = aggregate(per_seed, "alec", seeds, "mean_attn")
    b_mean = aggregate(per_seed, "mappo", seeds, "mean_attn")
    a_sender = aggregate(per_seed, "alec", seeds, "mean_sender_influence")
    b_sender = aggregate(per_seed, "mappo", seeds, "mean_sender_influence")
    a_entropy = float(np.mean([per_seed["alec"][str(s)]["mean_receiver_entropy"] for s in seeds]))
    b_entropy = float(np.mean([per_seed["mappo"][str(s)]["mean_receiver_entropy"] for s in seeds]))
    a_reward = float(np.mean([per_seed["alec"][str(s)]["mean_reward"] for s in seeds]))
    b_reward = float(np.mean([per_seed["mappo"][str(s)]["mean_reward"] for s in seeds]))
    if pooling == "unit_type":
        baseline = uniform_role_attention(len(role_labels))
        axis_labels = role_labels
        influence_xlabel = "Sender role"
        race_tag = race.title()
        attn_note = "receiver type × sender type (pooled)"
    else:
        baseline = uniform_offdiag_attention(n_units)
        axis_labels = None
        influence_xlabel = "Agent index"
        race_tag = race.title()
        attn_note = "agent slot index"

    render_mean_figure(
        a_mean=a_mean,
        b_mean=b_mean,
        a_sender=a_sender,
        b_sender=b_sender,
        a_entropy=a_entropy,
        b_entropy=b_entropy,
        label_a=LABEL_A,
        label_b=LABEL_B,
        suptitle=(
            f"Phase-2 Communication Attention: {LABEL_A} vs {LABEL_B}\n"
            f"SMACv2 {race_tag} {size}, {attn_note}, mean over {len(seeds)} seeds"
        ),
        save_path=save_dir / "attention_comparison.png",
        axis_labels=axis_labels,
        influence_xlabel=influence_xlabel,
    )
    render_residual_figure(
        a_mean=a_mean,
        b_mean=b_mean,
        a_sender=a_sender,
        b_sender=b_sender,
        baseline=baseline,
        label_a=LABEL_A,
        label_b=LABEL_B,
        suptitle=(
            f"Phase-2 Attention Residuals vs Uniform: {LABEL_A} vs {LABEL_B}\n"
            f"SMACv2 {race_tag} {size}, {attn_note}"
        ),
        save_path=save_dir / "attention_residual_comparison.png",
        axis_labels=axis_labels,
        influence_xlabel=influence_xlabel,
    )
    a_rep_ep = best_example(per_seed, "alec", seeds, "most_focused_episode")
    b_rep_ep = best_example(per_seed, "mappo", seeds, "most_focused_episode")
    a_rep_step = best_example(per_seed, "alec", seeds, "most_focused_step")
    b_rep_step = best_example(per_seed, "mappo", seeds, "most_focused_step")
    render_sample_figure(
        a_episode=a_rep_ep,
        b_episode=b_rep_ep,
        a_step=a_rep_step,
        b_step=b_rep_step,
        baseline=baseline,
        label_a=LABEL_A,
        label_b=LABEL_B,
        suptitle=(
            f"Most Focused Communication Structure (residuals vs uniform)\n"
            f"SMACv2 {race_tag} {size}, {attn_note}"
        ),
        save_path=save_dir / "attention_sample_diagnostics.png",
        axis_labels=axis_labels,
    )

    return {
        "n_units": n_units,
        "size": size,
        "pooling": pooling,
        "alec_mean_attn": a_mean,
        "mappo_mean_attn": b_mean,
        "alec_mean_sender": a_sender,
        "mappo_mean_sender": b_sender,
        "alec_entropy": a_entropy,
        "mappo_entropy": b_entropy,
        "alec_reward": a_reward,
        "mappo_reward": b_reward,
        "baseline": baseline,
        "role_labels": role_labels if pooling == "unit_type" else None,
    }


def _apply_axis_labels(ax, axis_labels: list[str]) -> None:
    ax.set_xticks(range(len(axis_labels)))
    ax.set_xticklabels(axis_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(axis_labels)))
    ax.set_yticklabels(axis_labels)


def render_combined_grid(per_size: list[dict], save_path: Path, label_a: str, label_b: str) -> None:
    n_rows = len(per_size)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 4.2 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = axes[None, :]
    for row, item in enumerate(per_size):
        a_mean = item["alec_mean_attn"]
        b_mean = item["mappo_mean_attn"]
        diff = a_mean - b_mean
        vmax = max(float(a_mean.max()), float(b_mean.max()), 1e-6)
        dmax = max(abs(float(diff.min())), abs(float(diff.max())), 1e-6)
        axis_labels = item.get("role_labels")
        sender_lbl = "Sender role" if axis_labels else "Sender"
        recv_lbl = "Receiver role" if axis_labels else "Receiver"

        im0 = axes[row, 0].imshow(a_mean, cmap="RdBu_r", vmin=0.0, vmax=vmax)
        axes[row, 0].set_title(f"{label_a} ({item['size']})\nH={item['alec_entropy']:.3f}, R={item['alec_reward']:.2f}")
        axes[row, 0].set_xlabel(sender_lbl)
        axes[row, 0].set_ylabel(recv_lbl)
        if axis_labels:
            _apply_axis_labels(axes[row, 0], axis_labels)
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        im1 = axes[row, 1].imshow(b_mean, cmap="RdBu_r", vmin=0.0, vmax=vmax)
        axes[row, 1].set_title(f"{label_b} ({item['size']})\nH={item['mappo_entropy']:.3f}, R={item['mappo_reward']:.2f}")
        axes[row, 1].set_xlabel(sender_lbl)
        axes[row, 1].set_ylabel(recv_lbl)
        if axis_labels:
            _apply_axis_labels(axes[row, 1], axis_labels)
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        im2 = axes[row, 2].imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
        axes[row, 2].set_title(f"{label_a} - {label_b} ({item['size']})")
        axes[row, 2].set_xlabel(sender_lbl)
        axes[row, 2].set_ylabel(recv_lbl)
        if axis_labels:
            _apply_axis_labels(axes[row, 2], axis_labels)
        fig.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

    pooling = per_size[0]["pooling"] if per_size else "unit_type"
    subtitle = (
        "receiver type × sender type (capabilities)"
        if pooling == "unit_type"
        else "agent slot index"
    )
    fig.suptitle(
        f"SMACv2 Phase-2 Attention: {label_a} vs {label_b} ({subtitle})",
        fontsize=15,
    )
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def serialize_per_seed(per_seed: dict) -> dict:
    out: dict = {}
    for method, seed_map in per_seed.items():
        out[method] = {}
        for seed, stats in seed_map.items():
            row = {
                "mean_reward": float(stats["mean_reward"]),
                "mean_receiver_entropy": float(stats["mean_receiver_entropy"]),
                "mean_attn": np.asarray(stats["mean_attn"]).tolist(),
                "mean_sender_influence": np.asarray(stats["mean_sender_influence"]).tolist(),
                "most_focused_episode": serialize_example(stats["most_focused_episode"]),
                "most_focused_step": serialize_example(stats["most_focused_step"]),
            }
            if "n_roles" in stats:
                row["n_roles"] = int(stats["n_roles"])
            out[method][seed] = row
    return out


def main() -> None:
    args = parse_args()
    if args.sc2_path:
        os.environ["SC2PATH"] = args.sc2_path
    args.save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    role_labels = race_role_labels(args.race)
    collect = (
        collect_attention_stats_by_smac_role
        if args.pooling == "unit_type"
        else collect_attention_stats
    )

    per_size_data = []
    full_summary = {"sizes": {}, "pooling": args.pooling, "race": args.race, "role_labels": role_labels}

    for size in args.sizes:
        n_units, n_enemies = parse_size(size)
        size_save_dir = args.save_dir / size
        size_save_dir.mkdir(parents=True, exist_ok=True)
        phase1_dir = args.phase1_root / f"smacv2_phase1_attn_{size}"
        phase2_dir = args.phase2_root / f"smacv2_phase2_3arm_{size}"

        per_seed: dict[str, dict[str, dict]] = {"alec": {}, "mappo": {}}
        for seed in args.seeds:
            print(f"[size={size} seed={seed}] starting", flush=True)
            env = make_env(race=args.race, n_units=n_units, n_enemies=n_enemies, render=False)
            env.reset()
            backbone = build_backbone(
                env=env,
                checkpoint_path=phase1_dir / f"seed{seed}" / "memfound_full_attention_hu_actor_latest.pt",
                device=device,
            )
            alec_adapter = load_adapter(
                backbone, phase2_dir / f"seed{seed}" / ARM_A / "best_adapter.pt", device=device
            )
            mappo_adapter = load_adapter(
                backbone, phase2_dir / f"seed{seed}" / ARM_B / "best_adapter.pt", device=device
            )

            ep_seeds = episode_seed_list(seed=seed, episodes=args.episodes)
            per_seed["alec"][str(seed)] = collect(
                env=env,
                backbone=backbone,
                adapter=alec_adapter,
                episodes=args.episodes,
                episode_seeds=ep_seeds,
                device=device,
            )
            per_seed["mappo"][str(seed)] = collect(
                env=env,
                backbone=backbone,
                adapter=mappo_adapter,
                episodes=args.episodes,
                episode_seeds=ep_seeds,
                device=device,
            )
            env.close()
            print(
                f"[size={size} seed={seed}] alec_R={per_seed['alec'][str(seed)]['mean_reward']:.2f} "
                f"mappo_R={per_seed['mappo'][str(seed)]['mean_reward']:.2f}",
                flush=True,
            )

        size_data = render_size_figures(
            size=size,
            n_units=n_units,
            per_seed=per_seed,
            seeds=args.seeds,
            save_dir=size_save_dir,
            pooling=args.pooling,
            race=args.race,
            role_labels=role_labels,
        )
        per_size_data.append(size_data)
        full_summary["sizes"][size] = {
            "pooling": args.pooling,
            "n_units": n_units,
            "n_enemies": n_enemies,
            "seeds": args.seeds,
            "episodes_per_seed": args.episodes,
            "role_labels": role_labels if args.pooling == "unit_type" else None,
            "alec_mean_reward": size_data["alec_reward"],
            "mappo_mean_reward": size_data["mappo_reward"],
            "alec_mean_receiver_entropy": size_data["alec_entropy"],
            "mappo_mean_receiver_entropy": size_data["mappo_entropy"],
            "alec_mean_attn": size_data["alec_mean_attn"].tolist(),
            "mappo_mean_attn": size_data["mappo_mean_attn"].tolist(),
            "alec_mean_sender": size_data["alec_mean_sender"].tolist(),
            "mappo_mean_sender": size_data["mappo_mean_sender"].tolist(),
            "baseline": size_data["baseline"].tolist(),
            "per_seed": serialize_per_seed(per_seed),
        }

    render_combined_grid(
        per_size=per_size_data,
        save_path=args.save_dir / "attention_grid_alec_vs_mappo.png",
        label_a=LABEL_A,
        label_b=LABEL_B,
    )
    (args.save_dir / "attention_summary.json").write_text(json.dumps(full_summary, indent=2))
    print(
        json.dumps(
            {
                "saved_grid": str(args.save_dir / "attention_grid_alec_vs_mappo.png"),
                "saved_summary": str(args.save_dir / "attention_summary.json"),
                "per_size": [str(args.save_dir / s) for s in args.sizes],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
