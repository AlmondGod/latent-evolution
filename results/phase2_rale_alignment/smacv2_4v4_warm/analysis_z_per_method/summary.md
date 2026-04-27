# RALE alignment summary (z)

- Probe root: `results/phase2_rale_alignment/smacv2_4v4_warm/probes`
- Methods: `['alec', 'rl_warm']`
- Seeds: `[42, 43, 44]`
- K values: `[8, 12, 16, 20, 24, 32, 64, 128]`
- Y target: `future` (H=10)
- Bootstrap iters: 200; Shuffle iters: 50

## Aggregate (mean ± SEM across seeds with >=3 occupied clusters)

| K | method | n_valid/n_total | clusters/seed | weighted Spearman | A_iso | shuffle Spearman | shuffle A_iso |
|--:|:------:|:---------------:|:-------------:|:------------------:|:------:|:------------------:|:---------------:|
| 8 | alec | 2/3 | 2,3,5 | +0.966 ± 0.034 | 0.039 ± 0.039 | +0.365 | 0.116 |
| 8 | rl_warm | 3/3 | 3,4,3 | +0.896 ± 0.054 | 0.034 ± 0.030 | +0.255 | 0.115 |
| 12 | alec | 3/3 | 3,3,8 | +0.767 ± 0.212 | 0.055 ± 0.030 | +0.519 | 0.097 |
| 12 | rl_warm | 3/3 | 4,6,4 | +0.922 ± 0.017 | 0.024 ± 0.021 | +0.272 | 0.112 |
| 16 | alec | 3/3 | 4,5,9 | +0.799 ± 0.157 | 0.098 ± 0.021 | +0.529 | 0.117 |
| 16 | rl_warm | 3/3 | 5,8,5 | +0.949 ± 0.019 | 0.057 ± 0.033 | +0.372 | 0.111 |
| 20 | alec | 3/3 | 5,6,11 | +0.955 ± 0.011 | 0.073 ± 0.017 | +0.486 | 0.113 |
| 20 | rl_warm | 3/3 | 6,9,7 | +0.814 ± 0.114 | 0.063 ± 0.022 | +0.491 | 0.122 |
| 24 | alec | 3/3 | 5,7,14 | +0.950 ± 0.009 | 0.096 ± 0.034 | +0.494 | 0.120 |
| 24 | rl_warm | 3/3 | 7,11,8 | +0.898 ± 0.055 | 0.062 ± 0.024 | +0.582 | 0.133 |
| 32 | alec | 3/3 | 6,8,20 | +0.919 ± 0.018 | 0.106 ± 0.026 | +0.558 | 0.117 |
| 32 | rl_warm | 3/3 | 12,13,9 | +0.872 ± 0.062 | 0.059 ± 0.017 | +0.667 | 0.126 |
| 64 | alec | 3/3 | 10,16,39 | +0.909 ± 0.037 | 0.112 ± 0.025 | +0.717 | 0.136 |
| 64 | rl_warm | 3/3 | 22,27,17 | +0.940 ± 0.011 | 0.103 ± 0.028 | +0.724 | 0.145 |
| 128 | alec | 3/3 | 16,34,75 | +0.940 ± 0.013 | 0.116 ± 0.023 | +0.810 | 0.144 |
| 128 | rl_warm | 3/3 | 43,52,33 | +0.947 ± 0.013 | 0.111 ± 0.030 | +0.767 | 0.163 |

Notes: `n_valid` = seeds where ≥3 clusters were occupied at >=`min_cluster_frac` (weighted Spearman / A_iso are undefined when fewer than 3 clusters are kept). `clusters/seed` shows the per-seed count of kept clusters. If all seeds collapse to <3 clusters, the latent space is effectively a point and the method is not using the meme channel meaningfully.
