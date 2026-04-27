# RALE alignment summary (h)

- Probe root: `results/phase2_rale_alignment/smacv2_4v4_warm/probes`
- Methods: `['alec', 'rl_warm']`
- Seeds: `[42, 43, 44]`
- K values: `[8, 12, 16]`
- Y target: `future` (H=10)
- Bootstrap iters: 200; Shuffle iters: 50

## Aggregate (mean ± SEM across seeds with >=3 occupied clusters)

| K | method | n_valid/n_total | clusters/seed | weighted Spearman | A_iso | shuffle Spearman | shuffle A_iso |
|--:|:------:|:---------------:|:-------------:|:------------------:|:------:|:------------------:|:---------------:|
| 8 | alec | 3/3 | 4,3,3 | +0.833 ± 0.167 | 0.000 ± 0.000 | +0.589 | 0.001 |
| 8 | rl_warm | 3/3 | 4,3,3 | +0.922 ± 0.078 | 0.000 ± 0.000 | +0.565 | 0.000 |
| 12 | alec | 3/3 | 5,5,4 | +1.000 ± 0.000 | 0.000 ± 0.000 | +0.626 | 0.002 |
| 12 | rl_warm | 3/3 | 5,5,4 | +1.000 ± 0.000 | 0.000 ± 0.000 | +0.704 | 0.001 |
| 16 | alec | 3/3 | 6,7,5 | +0.870 ± 0.130 | 0.001 ± 0.001 | +0.522 | 0.003 |
| 16 | rl_warm | 3/3 | 6,7,5 | +0.978 ± 0.022 | 0.000 ± 0.000 | +0.660 | 0.001 |

Notes: `n_valid` = seeds where ≥3 clusters were occupied at >=`min_cluster_frac` (weighted Spearman / A_iso are undefined when fewer than 3 clusters are kept). `clusters/seed` shows the per-seed count of kept clusters. If all seeds collapse to <3 clusters, the latent space is effectively a point and the method is not using the meme channel meaningfully.
