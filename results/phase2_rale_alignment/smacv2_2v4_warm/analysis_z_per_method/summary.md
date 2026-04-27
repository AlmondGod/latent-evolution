# RALE alignment summary (z)

- Probe root: `results/phase2_rale_alignment/smacv2_2v4_warm/probes`
- Methods: `['alec', 'rl_warm']`
- Seeds: `[42, 43, 44]`
- K values: `[8, 12, 16, 20, 24, 32, 64, 128]`
- Y target: `future` (H=10)
- Bootstrap iters: 200; Shuffle iters: 50

## Aggregate (mean ± SEM across seeds with >=3 occupied clusters)

| K | method | n_valid/n_total | clusters/seed | weighted Spearman | A_iso | shuffle Spearman | shuffle A_iso |
|--:|:------:|:---------------:|:-------------:|:------------------:|:------:|:------------------:|:---------------:|
| 8 | alec | 3/3 | 4,3,3 | +0.863 ± 0.023 | 0.023 ± 0.005 | +0.323 | 0.022 |
| 8 | rl_warm | 3/3 | 3,4,3 | +0.164 ± 0.503 | 0.082 ± 0.021 | -0.080 | 0.063 |
| 12 | alec | 3/3 | 6,3,5 | +0.862 ± 0.029 | 0.031 ± 0.003 | +0.410 | 0.029 |
| 12 | rl_warm | 3/3 | 5,5,4 | +0.773 ± 0.143 | 0.066 ± 0.017 | +0.211 | 0.060 |
| 16 | alec | 3/3 | 8,4,6 | +0.744 ± 0.065 | 0.030 ± 0.007 | +0.493 | 0.030 |
| 16 | rl_warm | 3/3 | 6,6,6 | +0.785 ± 0.081 | 0.075 ± 0.011 | +0.298 | 0.070 |
| 20 | alec | 3/3 | 9,6,7 | +0.855 ± 0.038 | 0.029 ± 0.007 | +0.601 | 0.032 |
| 20 | rl_warm | 3/3 | 8,7,7 | +0.813 ± 0.046 | 0.076 ± 0.005 | +0.390 | 0.070 |
| 24 | alec | 3/3 | 12,5,9 | +0.859 ± 0.033 | 0.028 ± 0.008 | +0.537 | 0.033 |
| 24 | rl_warm | 3/3 | 10,8,8 | +0.836 ± 0.033 | 0.081 ± 0.011 | +0.451 | 0.074 |
| 32 | alec | 3/3 | 15,8,11 | +0.878 ± 0.023 | 0.032 ± 0.005 | +0.660 | 0.035 |
| 32 | rl_warm | 3/3 | 14,9,11 | +0.872 ± 0.025 | 0.073 ± 0.010 | +0.498 | 0.077 |
| 64 | alec | 3/3 | 33,15,18 | +0.895 ± 0.010 | 0.031 ± 0.003 | +0.739 | 0.039 |
| 64 | rl_warm | 3/3 | 30,18,18 | +0.887 ± 0.015 | 0.077 ± 0.012 | +0.613 | 0.084 |
| 128 | alec | 3/3 | 64,31,35 | +0.887 ± 0.010 | 0.033 ± 0.003 | +0.815 | 0.036 |
| 128 | rl_warm | 3/3 | 53,39,38 | +0.868 ± 0.012 | 0.081 ± 0.012 | +0.673 | 0.092 |

Notes: `n_valid` = seeds where ≥3 clusters were occupied at >=`min_cluster_frac` (weighted Spearman / A_iso are undefined when fewer than 3 clusters are kept). `clusters/seed` shows the per-seed count of kept clusters. If all seeds collapse to <3 clusters, the latent space is effectively a point and the method is not using the meme channel meaningfully.
