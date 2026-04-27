# RALE alignment summary (h)

- Probe root: `results/phase2_rale_alignment/smacv2_8v4_warm/probes`
- Methods: `['alec', 'rl_warm']`
- Seeds: `[42, 43, 44]`
- K values: `[8, 12, 16]`
- Y target: `future` (H=10)
- Bootstrap iters: 200; Shuffle iters: 50

## Aggregate (mean ± SEM across seeds with >=3 occupied clusters)

| K | method | n_valid/n_total | clusters/seed | weighted Spearman | A_iso | shuffle Spearman | shuffle A_iso |
|--:|:------:|:---------------:|:-------------:|:------------------:|:------:|:------------------:|:---------------:|
| 8 | alec | 3/3 | 3,4,3 | +1.000 ± 0.000 | 0.000 ± 0.000 | +0.288 | 0.008 |
| 8 | rl_warm | 3/3 | 3,4,3 | +0.892 ± 0.108 | 0.000 ± 0.000 | +0.418 | 0.005 |
| 12 | alec | 3/3 | 5,5,4 | +1.000 ± 0.000 | 0.000 ± 0.000 | +0.399 | 0.019 |
| 12 | rl_warm | 3/3 | 5,5,4 | +0.910 ± 0.051 | 0.062 ± 0.049 | +0.440 | 0.065 |
| 16 | alec | 3/3 | 6,6,6 | +1.000 ± 0.000 | 0.000 ± 0.000 | +0.514 | 0.032 |
| 16 | rl_warm | 3/3 | 6,6,6 | +0.947 ± 0.028 | 0.057 ± 0.057 | +0.453 | 0.116 |

Notes: `n_valid` = seeds where ≥3 clusters were occupied at >=`min_cluster_frac` (weighted Spearman / A_iso are undefined when fewer than 3 clusters are kept). `clusters/seed` shows the per-seed count of kept clusters. If all seeds collapse to <3 clusters, the latent space is effectively a point and the method is not using the meme channel meaningfully.
