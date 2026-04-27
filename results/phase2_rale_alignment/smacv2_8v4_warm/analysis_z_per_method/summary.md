# RALE alignment summary (z)

- Probe root: `results/phase2_rale_alignment/smacv2_8v4_warm/probes`
- Methods: `['alec', 'rl_warm']`
- Seeds: `[42, 43, 44]`
- K values: `[8, 12, 16, 20, 24, 32, 64, 128]`
- Y target: `future` (H=10)
- Bootstrap iters: 200; Shuffle iters: 50

## Aggregate (mean ± SEM across seeds with >=3 occupied clusters)

| K | method | n_valid/n_total | clusters/seed | weighted Spearman | A_iso | shuffle Spearman | shuffle A_iso |
|--:|:------:|:---------------:|:-------------:|:------------------:|:------:|:------------------:|:---------------:|
| 8 | alec | 3/3 | 3,3,4 | +0.450 ± 0.499 | 0.045 ± 0.035 | +0.162 | 0.501 |
| 8 | rl_warm | 3/3 | 3,4,3 | +1.000 ± 0.000 | 0.000 ± 0.000 | -0.021 | 0.779 |
| 12 | alec | 3/3 | 4,3,7 | +0.433 ± 0.489 | 0.040 ± 0.024 | +0.449 | 0.538 |
| 12 | rl_warm | 3/3 | 4,5,5 | +0.856 ± 0.076 | 0.074 ± 0.041 | +0.089 | 0.857 |
| 16 | alec | 3/3 | 5,5,8 | +0.960 ± 0.002 | 0.160 ± 0.053 | +0.386 | 0.560 |
| 16 | rl_warm | 3/3 | 5,7,6 | +0.809 ± 0.090 | 0.060 ± 0.033 | +0.157 | 0.883 |
| 20 | alec | 3/3 | 5,5,12 | +0.953 ± 0.004 | 0.168 ± 0.055 | +0.420 | 0.620 |
| 20 | rl_warm | 3/3 | 7,8,7 | +0.893 ± 0.067 | 0.063 ± 0.035 | +0.185 | 0.894 |
| 24 | alec | 3/3 | 7,6,13 | +0.954 ± 0.019 | 0.156 ± 0.033 | +0.442 | 0.590 |
| 24 | rl_warm | 3/3 | 8,10,8 | +0.876 ± 0.082 | 0.064 ± 0.031 | +0.219 | 0.935 |
| 32 | alec | 3/3 | 8,8,18 | +0.941 ± 0.038 | 0.155 ± 0.032 | +0.462 | 0.707 |
| 32 | rl_warm | 3/3 | 11,14,9 | +0.887 ± 0.038 | 0.032 ± 0.008 | +0.309 | 1.023 |
| 64 | alec | 3/3 | 13,17,35 | +0.968 ± 0.008 | 0.147 ± 0.029 | +0.632 | 0.747 |
| 64 | rl_warm | 3/3 | 22,24,20 | +0.941 ± 0.016 | 0.089 ± 0.017 | +0.419 | 1.062 |
| 128 | alec | 3/3 | 29,31,67 | +0.963 ± 0.005 | 0.243 ± 0.049 | +0.701 | 0.777 |
| 128 | rl_warm | 3/3 | 43,52,33 | +0.944 ± 0.010 | 0.200 ± 0.049 | +0.473 | 1.170 |

Notes: `n_valid` = seeds where ≥3 clusters were occupied at >=`min_cluster_frac` (weighted Spearman / A_iso are undefined when fewer than 3 clusters are kept). `clusters/seed` shows the per-seed count of kept clusters. If all seeds collapse to <3 clusters, the latent space is effectively a point and the method is not using the meme channel meaningfully.
