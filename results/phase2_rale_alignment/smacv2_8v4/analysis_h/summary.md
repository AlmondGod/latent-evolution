# RALE alignment summary (h)

- Probe root: `results/phase2_rale_alignment/smacv2_8v4/probes`
- Methods: `['alec', 'rl']`
- Seeds: `[42, 43, 44]`
- K values: `[8, 12, 16]`
- Y target: `future` (H=10)
- Bootstrap iters: 200; Shuffle iters: 50

## Aggregate (mean ± SEM across seeds with >=3 occupied clusters)

| K | method | n_valid/n_total | clusters/seed | weighted Spearman | A_iso | shuffle Spearman | shuffle A_iso |
|--:|:------:|:---------------:|:-------------:|:------------------:|:------:|:------------------:|:---------------:|
| 8 | alec | 3/3 | 4,3,3 | +0.854 ± 0.146 | 0.051 ± 0.051 | +0.275 | 0.045 |
| 8 | rl | 3/3 | 4,3,3 | +1.000 ± 0.000 | 0.000 ± 0.000 | +0.235 | 0.008 |
| 12 | alec | 3/3 | 7,3,4 | +0.970 ± 0.030 | 0.004 ± 0.004 | +0.431 | 0.013 |
| 12 | rl | 3/3 | 7,3,4 | +0.983 ± 0.017 | 0.001 ± 0.001 | +0.438 | 0.004 |
| 16 | alec | 3/3 | 6,5,7 | +0.798 ± 0.048 | 0.024 ± 0.013 | +0.424 | 0.031 |
| 16 | rl | 3/3 | 6,5,7 | +0.988 ± 0.012 | 0.000 ± 0.000 | +0.582 | 0.005 |

Notes: `n_valid` = seeds where ≥3 clusters were occupied at >=`min_cluster_frac` (weighted Spearman / A_iso are undefined when fewer than 3 clusters are kept). `clusters/seed` shows the per-seed count of kept clusters. If all seeds collapse to <3 clusters, the latent space is effectively a point and the method is not using the meme channel meaningfully.
