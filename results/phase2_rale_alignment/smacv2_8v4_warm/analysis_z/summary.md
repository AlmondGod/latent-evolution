# RALE alignment summary (z)

- Probe root: `results/phase2_rale_alignment/smacv2_8v4_warm/probes`
- Methods: `['alec', 'rl_warm']`
- Seeds: `[42, 43, 44]`
- K values: `[8, 12, 16]`
- Y target: `future` (H=10)
- Bootstrap iters: 200; Shuffle iters: 50

## Aggregate (mean ± SEM across seeds with >=3 occupied clusters)

| K | method | n_valid/n_total | clusters/seed | weighted Spearman | A_iso | shuffle Spearman | shuffle A_iso |
|--:|:------:|:---------------:|:-------------:|:------------------:|:------:|:------------------:|:---------------:|
| 8 | alec | 1/3 | 2,2,3 | +1.000 ± nan | 0.000 ± nan | -0.165 | 0.704 |
| 8 | rl_warm | 0/3 | 2,2,2 | +nan ± nan | nan ± nan | +nan | nan |
| 12 | alec | 3/3 | 3,3,5 | +0.384 ± 0.466 | 0.049 ± 0.032 | +0.285 | 0.514 |
| 12 | rl_warm | 0/3 | 2,2,2 | +nan ± nan | nan ± nan | +nan | nan |
| 16 | alec | 3/3 | 4,3,7 | +0.443 ± 0.494 | 0.038 ± 0.025 | +0.403 | 0.550 |
| 16 | rl_warm | 1/3 | 3,2,2 | +1.000 ± nan | 0.000 ± nan | -0.111 | 1.084 |

Notes: `n_valid` = seeds where ≥3 clusters were occupied at >=`min_cluster_frac` (weighted Spearman / A_iso are undefined when fewer than 3 clusters are kept). `clusters/seed` shows the per-seed count of kept clusters. If all seeds collapse to <3 clusters, the latent space is effectively a point and the method is not using the meme channel meaningfully.
