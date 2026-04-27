# RALE alignment summary (z)

- Probe root: `results/phase2_rale_alignment/smacv2_2v4_warm/probes`
- Methods: `['alec', 'rl_warm']`
- Seeds: `[42, 43, 44]`
- K values: `[8, 12, 16]`
- Y target: `future` (H=10)
- Bootstrap iters: 200; Shuffle iters: 50

## Aggregate (mean ± SEM across seeds with >=3 occupied clusters)

| K | method | n_valid/n_total | clusters/seed | weighted Spearman | A_iso | shuffle Spearman | shuffle A_iso |
|--:|:------:|:---------------:|:-------------:|:------------------:|:------:|:------------------:|:---------------:|
| 8 | alec | 1/3 | 3,2,2 | +0.840 ± nan | 0.029 ± nan | +0.219 | 0.022 |
| 8 | rl_warm | 0/3 | 2,2,2 | +nan ± nan | nan ± nan | +nan | nan |
| 12 | alec | 3/3 | 4,3,4 | +0.875 ± 0.021 | 0.025 ± 0.004 | +0.347 | 0.027 |
| 12 | rl_warm | 0/3 | 2,2,2 | +nan ± nan | nan ± nan | +nan | nan |
| 16 | alec | 3/3 | 7,3,4 | +0.873 ± 0.018 | 0.032 ± 0.006 | +0.406 | 0.028 |
| 16 | rl_warm | 1/3 | 3,2,2 | +0.331 ± nan | 0.072 ± nan | +0.067 | 0.037 |

Notes: `n_valid` = seeds where ≥3 clusters were occupied at >=`min_cluster_frac` (weighted Spearman / A_iso are undefined when fewer than 3 clusters are kept). `clusters/seed` shows the per-seed count of kept clusters. If all seeds collapse to <3 clusters, the latent space is effectively a point and the method is not using the meme channel meaningfully.
