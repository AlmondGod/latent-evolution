#!/usr/bin/env python3
# Aggregate RALE per_seed (per (method, seed) k-means) results across N in
# {2,4,8} and K in {8,12,16}, and produce the paper figure: weighted Spearman
# + A_iso vs K with N as separate series. Error bars are sample std across
# seeds. Shuffle nulls are drawn as dotted lines per N/method.

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results" / "phase2_rale_alignment"


def parse_aggregate(path: Path) -> dict:
    rows: dict = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (int(r["K"]), r["method"])

            def _f(name: str) -> float:
                v = r[name]
                return float(v) if v not in ("", "nan") else float("nan")

            rows[key] = {
                "n_valid": int(r["n_seeds_valid"]),
                "n_total": int(r["n_seeds_total"]),
                "ws_mean": _f("ws_mean"),
                "ws_var": _f("ws_var"),
                "ws_std": _f("ws_std"),
                "aiso_mean": _f("aiso_mean"),
                "aiso_var": _f("aiso_var"),
                "aiso_std": _f("aiso_std"),
                "shuffle_ws": _f("shuffle_ws"),
                "shuffle_ws_std": _f("shuffle_ws_std"),
            }
    return rows


def load_per_seed(n: int) -> dict:
    return parse_aggregate(RES / f"smacv2_{n}v4_warm" / "analysis_z_per_seed" / "tables" / "method_aggregate.csv")


def main() -> None:
    Ns = [2, 4, 8]
    Ks = [8, 12, 16, 20, 24, 32, 64, 128]
    data = {n: load_per_seed(n) for n in Ns}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    color = {2: "tab:blue", 4: "tab:orange", 8: "tab:green"}
    K_jitter = {"alec": -0.25, "rl_warm": 0.25}

    for ax, metric, ylabel, ylim in (
        (axes[0], "ws_mean",   r"Weighted Spearman $\rho_w(F, R)$",        (0.0, 1.05)),
        (axes[1], "aiso_mean", r"Isotonic mismatch $A_{\mathrm{iso}}$ (lower = aligned)", (0.0, 0.40)),
    ):
        for n in Ns:
            ys = [data[n][(K, "alec")][metric] for K in Ks]
            yerrs = [data[n][(K, "alec")][metric.replace("_mean", "_std")] for K in Ks]
            xs = [K + K_jitter["alec"] for K in Ks]
            ax.errorbar(
                xs, ys, yerr=yerrs,
                marker="o", linestyle="-", color=color[n],
                label=f"ALEC N={n}v4" if metric == "ws_mean" else None,
                capsize=3, lw=1.6, markersize=6,
            )
        for n in Ns:
            ys = [data[n][(K, "rl_warm")][metric] for K in Ks]
            yerrs = [data[n][(K, "rl_warm")][metric.replace("_mean", "_std")] for K in Ks]
            xs = [K + K_jitter["rl_warm"] for K in Ks]
            ax.errorbar(
                xs, ys, yerr=yerrs,
                marker="^", linestyle="--", color=color[n],
                label=f"MAPPO N={n}v4" if metric == "ws_mean" else None,
                capsize=3, lw=1.4, markersize=7, alpha=0.9,
            )
        if metric == "ws_mean":
            for n in Ns:
                shuf_alec = [data[n][(K, "alec")]["shuffle_ws"] for K in Ks]
                ax.plot(Ks, shuf_alec, color=color[n], alpha=0.45, lw=1.0, ls=":")
            ax.text(
                0.02, 0.04, "dotted = shuffle null (ALEC)",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8, alpha=0.7,
            )
            ax.axhline(0.0, color="gray", lw=0.7, ls="-", alpha=0.4)
        ax.set_xlabel("K (number of clusters per seed)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(Ks)
        ax.set_xlim(Ks[0] - 2, Ks[-1] + 2)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Replication--reward alignment")
    axes[1].set_title("Isotonic mismatch")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="lower right", fontsize=7, ncol=2, frameon=True)

    fig.suptitle(
        "Per-(method, seed) RALE alignment vs cluster count $K$ and team size $N$ "
        "(SMACv2 Terran, 3 seeds $\\times$ 25 episodes; error bars = std)"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_dir = RES / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "rale_alignment_K_N.pdf"
    out_png = out_dir / "rale_alignment_K_N.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=180)
    print(f"saved: {out_pdf}")
    print(f"saved: {out_png}")


if __name__ == "__main__":
    main()
