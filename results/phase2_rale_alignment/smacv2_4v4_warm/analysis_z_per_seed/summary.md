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
| 8 | alec | 3/3 | 8,8,8 | +0.907 ± 0.011 | 0.110 ± 0.027 | +0.587 | 0.112 |
| 8 | rl_warm | 3/3 | 8,8,8 | +0.920 ± 0.009 | 0.062 ± 0.028 | +0.506 | 0.143 |
| 12 | alec | 3/3 | 12,12,12 | +0.908 ± 0.015 | 0.115 ± 0.020 | +0.685 | 0.125 |
| 12 | rl_warm | 3/3 | 12,12,12 | +0.931 ± 0.014 | 0.054 ± 0.021 | +0.622 | 0.141 |
| 16 | alec | 3/3 | 16,16,16 | +0.935 ± 0.012 | 0.104 ± 0.011 | +0.731 | 0.141 |
| 16 | rl_warm | 3/3 | 16,16,16 | +0.939 ± 0.012 | 0.095 ± 0.029 | +0.697 | 0.141 |
| 20 | alec | 3/3 | 20,20,20 | +0.926 ± 0.005 | 0.113 ± 0.021 | +0.751 | 0.141 |
| 20 | rl_warm | 3/3 | 20,20,20 | +0.948 ± 0.007 | 0.100 ± 0.031 | +0.714 | 0.145 |
| 24 | alec | 3/3 | 24,24,24 | +0.935 ± 0.010 | 0.116 ± 0.023 | +0.791 | 0.144 |
| 24 | rl_warm | 3/3 | 24,24,24 | +0.945 ± 0.013 | 0.111 ± 0.041 | +0.762 | 0.152 |
| 32 | alec | 3/3 | 32,32,32 | +0.932 ± 0.016 | 0.125 ± 0.027 | +0.813 | 0.145 |
| 32 | rl_warm | 3/3 | 32,32,32 | +0.942 ± 0.015 | 0.113 ± 0.037 | +0.785 | 0.156 |
| 64 | alec | 3/3 | 60,63,63 | +0.941 ± 0.013 | 0.110 ± 0.025 | +0.854 | 0.153 |
| 64 | rl_warm | 3/3 | 62,64,64 | +0.957 ± 0.009 | 0.108 ± 0.036 | +0.803 | 0.169 |
| 128 | alec | 3/3 | 93,95,101 | +0.942 ± 0.019 | 0.107 ± 0.021 | +0.875 | 0.152 |
| 128 | rl_warm | 3/3 | 100,101,104 | +0.944 ± 0.011 | 0.118 ± 0.038 | +0.800 | 0.177 |

Notes: `n_valid` = seeds where ≥3 clusters were occupied at >=`min_cluster_frac` (weighted Spearman / A_iso are undefined when fewer than 3 clusters are kept). `clusters/seed` shows the per-seed count of kept clusters. If all seeds collapse to <3 clusters, the latent space is effectively a point and the method is not using the meme channel meaningfully.
