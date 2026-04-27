# RALE alignment summary (z)

- Probe root: `results/phase2_rale_alignment/smacv2_2v4/probes`
- Methods: `['alec', 'rl']`
- Seeds: `[42, 43, 44]`
- K values: `[8, 12, 16]`
- Y target: `future` (H=10)
- Bootstrap iters: 200; Shuffle iters: 50

## Aggregate (mean ± SEM across seeds with >=3 occupied clusters)

| K | method | n_valid/n_total | clusters/seed | weighted Spearman | A_iso | shuffle Spearman | shuffle A_iso |
|--:|:------:|:---------------:|:-------------:|:------------------:|:------:|:------------------:|:---------------:|
| 8 | alec | 2/3 | 4,2,4 | +0.717 ± 0.216 | 0.061 ± 0.026 | +0.363 | 0.036 |
| 8 | rl | 0/3 | 1,1,1 | +nan ± nan | nan ± nan | +nan | nan |
| 12 | alec | 3/3 | 6,3,5 | +0.846 ± 0.057 | 0.046 ± 0.014 | +0.351 | 0.036 |
| 12 | rl | 0/3 | 1,1,1 | +nan ± nan | nan ± nan | +nan | nan |
| 16 | alec | 3/3 | 8,5,5 | +0.892 ± 0.039 | 0.046 ± 0.014 | +0.462 | 0.039 |
| 16 | rl | 0/3 | 1,1,1 | +nan ± nan | nan ± nan | +nan | nan |

Notes: `n_valid` = seeds where ≥3 clusters were occupied at >=`min_cluster_frac` (weighted Spearman / A_iso are undefined when fewer than 3 clusters are kept). `clusters/seed` shows the per-seed count of kept clusters. If all seeds collapse to <3 clusters, the latent space is effectively a point and the method is not using the meme channel meaningfully.
