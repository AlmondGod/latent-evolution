# RALE alignment summary (z)

- Probe root: `results/phase2_rale_alignment/smacv2_8v4/probes`
- Methods: `['alec', 'rl']`
- Seeds: `[42, 43, 44]`
- K values: `[8, 12, 16]`
- Y target: `future` (H=10)
- Bootstrap iters: 200; Shuffle iters: 50

## Aggregate (mean ± SEM across seeds with >=3 occupied clusters)

| K | method | n_valid/n_total | clusters/seed | weighted Spearman | A_iso | shuffle Spearman | shuffle A_iso |
|--:|:------:|:---------------:|:-------------:|:------------------:|:------:|:------------------:|:---------------:|
| 8 | alec | 2/3 | 2,3,4 | +1.000 ± 0.000 | 0.000 ± 0.000 | +0.067 | 0.446 |
| 8 | rl | 0/3 | 1,1,1 | +nan ± nan | nan ± nan | +nan | nan |
| 12 | alec | 3/3 | 4,4,5 | +0.980 ± 0.020 | 0.043 ± 0.043 | +0.230 | 0.593 |
| 12 | rl | 0/3 | 1,1,1 | +nan ± nan | nan ± nan | +nan | nan |
| 16 | alec | 3/3 | 5,5,7 | +0.975 ± 0.012 | 0.025 ± 0.022 | +0.297 | 0.640 |
| 16 | rl | 0/3 | 1,1,1 | +nan ± nan | nan ± nan | +nan | nan |

Notes: `n_valid` = seeds where ≥3 clusters were occupied at >=`min_cluster_frac` (weighted Spearman / A_iso are undefined when fewer than 3 clusters are kept). `clusters/seed` shows the per-seed count of kept clusters. If all seeds collapse to <3 clusters, the latent space is effectively a point and the method is not using the meme channel meaningfully.
