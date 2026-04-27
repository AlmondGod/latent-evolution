# RALE alignment summary (h)

- Probe root: `results/phase2_rale_alignment/smacv2_2v4/probes`
- Methods: `['alec', 'rl']`
- Seeds: `[42, 43, 44]`
- K values: `[8, 12, 16]`
- Y target: `future` (H=10)
- Bootstrap iters: 200; Shuffle iters: 50

## Aggregate (mean ± SEM across seeds with >=3 occupied clusters)

| K | method | n_valid/n_total | clusters/seed | weighted Spearman | A_iso | shuffle Spearman | shuffle A_iso |
|--:|:------:|:---------------:|:-------------:|:------------------:|:------:|:------------------:|:---------------:|
| 8 | alec | 3/3 | 3,3,4 | +1.000 ± 0.000 | 0.000 ± 0.000 | +0.305 | 0.003 |
| 8 | rl | 3/3 | 3,3,4 | +1.000 ± 0.000 | 0.000 ± 0.000 | +0.316 | 0.003 |
| 12 | alec | 3/3 | 5,4,5 | +0.965 ± 0.019 | 0.000 ± 0.000 | +0.508 | 0.004 |
| 12 | rl | 3/3 | 5,4,5 | +0.957 ± 0.043 | 0.000 ± 0.000 | +0.475 | 0.006 |
| 16 | alec | 3/3 | 5,5,8 | +1.000 ± 0.000 | 0.000 ± 0.000 | +0.455 | 0.005 |
| 16 | rl | 3/3 | 5,5,8 | +0.994 ± 0.006 | 0.000 ± 0.000 | +0.465 | 0.006 |

Notes: `n_valid` = seeds where ≥3 clusters were occupied at >=`min_cluster_frac` (weighted Spearman / A_iso are undefined when fewer than 3 clusters are kept). `clusters/seed` shows the per-seed count of kept clusters. If all seeds collapse to <3 clusters, the latent space is effectively a point and the method is not using the meme channel meaningfully.
