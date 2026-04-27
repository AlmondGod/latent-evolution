# RALE alignment summary (z)

- Probe root: `results/phase2_rale_alignment/smacv2_4v4_warm/probes`
- Methods: `['alec', 'rl_warm']`
- Seeds: `[42, 43, 44]`
- K values: `[8, 12, 16]`
- Y target: `future` (H=10)
- Bootstrap iters: 200; Shuffle iters: 50

## Aggregate (mean ± SEM across seeds with >=3 occupied clusters)

| K | method | n_valid/n_total | clusters/seed | weighted Spearman | A_iso | shuffle Spearman | shuffle A_iso |
|--:|:------:|:---------------:|:-------------:|:------------------:|:------:|:------------------:|:---------------:|
| 8 | alec | 1/3 | 2,2,3 | +0.894 ± nan | 0.043 ± nan | +0.099 | 0.069 |
| 8 | rl_warm | 0/3 | 2,2,2 | +nan ± nan | nan ± nan | +nan | nan |
| 12 | alec | 2/3 | 2,3,5 | +0.934 ± 0.066 | 0.052 ± 0.052 | +0.254 | 0.118 |
| 12 | rl_warm | 1/3 | 2,3,2 | +0.892 ± nan | 0.053 ± nan | +0.599 | 0.049 |
| 16 | alec | 3/3 | 3,3,7 | +0.771 ± 0.213 | 0.047 ± 0.030 | +0.457 | 0.108 |
| 16 | rl_warm | 1/3 | 2,3,2 | +0.892 ± nan | 0.053 ± nan | +0.590 | 0.078 |

Notes: `n_valid` = seeds where ≥3 clusters were occupied at >=`min_cluster_frac` (weighted Spearman / A_iso are undefined when fewer than 3 clusters are kept). `clusters/seed` shows the per-seed count of kept clusters. If all seeds collapse to <3 clusters, the latent space is effectively a point and the method is not using the meme channel meaningfully.
