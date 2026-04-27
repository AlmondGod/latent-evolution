# RALE alignment summary (h)

- Probe root: `results/phase2_rale_alignment/smacv2_4v4/probes`
- Methods: `['alec', 'rl']`
- Seeds: `[42, 43, 44]`
- K values: `[8, 12, 16]`
- Y target: `future` (H=10)
- Bootstrap iters: 200; Shuffle iters: 50

## Aggregate (mean ± SEM across seeds with >=3 occupied clusters)

| K | method | n_valid/n_total | clusters/seed | weighted Spearman | A_iso | shuffle Spearman | shuffle A_iso |
|--:|:------:|:---------------:|:-------------:|:------------------:|:------:|:------------------:|:---------------:|
| 8 | alec | 3/3 | 3,3,4 | +1.000 ± 0.000 | 0.000 ± 0.000 | +0.596 | 0.000 |
| 8 | rl | 3/3 | 3,3,4 | +1.000 ± 0.000 | 0.000 ± 0.000 | +0.578 | 0.000 |
| 12 | alec | 3/3 | 5,4,5 | +0.930 ± 0.039 | 0.000 ± 0.000 | +0.527 | 0.001 |
| 12 | rl | 3/3 | 5,4,5 | +1.000 ± 0.000 | 0.000 ± 0.000 | +0.630 | 0.000 |
| 16 | alec | 3/3 | 6,6,6 | +1.000 ± 0.000 | 0.000 ± 0.000 | +0.725 | 0.001 |
| 16 | rl | 3/3 | 6,6,6 | +1.000 ± 0.000 | 0.000 ± 0.000 | +0.770 | 0.000 |

Notes: `n_valid` = seeds where ≥3 clusters were occupied at >=`min_cluster_frac` (weighted Spearman / A_iso are undefined when fewer than 3 clusters are kept). `clusters/seed` shows the per-seed count of kept clusters. If all seeds collapse to <3 clusters, the latent space is effectively a point and the method is not using the meme channel meaningfully.
