# RALE alignment summary (h)

- Probe root: `results/phase2_rale_alignment/smacv2_2v4_warm/probes`
- Methods: `['alec', 'rl_warm']`
- Seeds: `[42, 43, 44]`
- K values: `[8, 12, 16]`
- Y target: `future` (H=10)
- Bootstrap iters: 200; Shuffle iters: 50

## Aggregate (mean ± SEM across seeds with >=3 occupied clusters)

| K | method | n_valid/n_total | clusters/seed | weighted Spearman | A_iso | shuffle Spearman | shuffle A_iso |
|--:|:------:|:---------------:|:-------------:|:------------------:|:------:|:------------------:|:---------------:|
| 8 | alec | 3/3 | 3,4,3 | +1.000 ± 0.000 | 0.000 ± 0.000 | +0.444 | 0.003 |
| 8 | rl_warm | 3/3 | 3,4,3 | +1.000 ± 0.000 | 0.000 ± 0.000 | +0.457 | 0.006 |
| 12 | alec | 3/3 | 5,4,5 | +0.941 ± 0.059 | 0.000 ± 0.000 | +0.507 | 0.003 |
| 12 | rl_warm | 3/3 | 5,4,5 | +0.971 ± 0.029 | 0.000 ± 0.000 | +0.476 | 0.005 |
| 16 | alec | 3/3 | 7,6,5 | +0.921 ± 0.053 | 0.000 ± 0.000 | +0.515 | 0.002 |
| 16 | rl_warm | 3/3 | 7,6,5 | +0.973 ± 0.027 | 0.000 ± 0.000 | +0.501 | 0.003 |

Notes: `n_valid` = seeds where ≥3 clusters were occupied at >=`min_cluster_frac` (weighted Spearman / A_iso are undefined when fewer than 3 clusters are kept). `clusters/seed` shows the per-seed count of kept clusters. If all seeds collapse to <3 clusters, the latent space is effectively a point and the method is not using the meme channel meaningfully.
