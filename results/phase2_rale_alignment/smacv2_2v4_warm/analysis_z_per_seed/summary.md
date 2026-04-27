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
| 8 | alec | 3/3 | 8,8,8 | +0.800 ± 0.010 | 0.039 ± 0.003 | +0.614 | 0.031 |
| 8 | rl_warm | 3/3 | 8,8,8 | +0.864 ± 0.044 | 0.075 ± 0.018 | +0.408 | 0.080 |
| 12 | alec | 3/3 | 12,12,12 | +0.886 ± 0.008 | 0.029 ± 0.003 | +0.691 | 0.035 |
| 12 | rl_warm | 3/3 | 12,12,12 | +0.879 ± 0.042 | 0.074 ± 0.009 | +0.529 | 0.078 |
| 16 | alec | 3/3 | 16,16,16 | +0.904 ± 0.005 | 0.030 ± 0.002 | +0.721 | 0.037 |
| 16 | rl_warm | 3/3 | 16,16,16 | +0.882 ± 0.029 | 0.072 ± 0.009 | +0.584 | 0.081 |
| 20 | alec | 3/3 | 20,20,20 | +0.884 ± 0.011 | 0.034 ± 0.002 | +0.748 | 0.036 |
| 20 | rl_warm | 3/3 | 20,20,20 | +0.851 ± 0.028 | 0.081 ± 0.010 | +0.630 | 0.086 |
| 24 | alec | 3/3 | 24,24,24 | +0.883 ± 0.015 | 0.032 ± 0.002 | +0.760 | 0.038 |
| 24 | rl_warm | 3/3 | 24,24,24 | +0.853 ± 0.032 | 0.086 ± 0.009 | +0.636 | 0.086 |
| 32 | alec | 3/3 | 32,32,32 | +0.877 ± 0.019 | 0.035 ± 0.005 | +0.781 | 0.038 |
| 32 | rl_warm | 3/3 | 32,32,32 | +0.866 ± 0.012 | 0.082 ± 0.008 | +0.665 | 0.089 |
| 64 | alec | 3/3 | 63,62,64 | +0.879 ± 0.012 | 0.036 ± 0.003 | +0.809 | 0.041 |
| 64 | rl_warm | 3/3 | 63,58,60 | +0.878 ± 0.007 | 0.082 ± 0.010 | +0.697 | 0.092 |
| 128 | alec | 3/3 | 91,94,103 | +0.877 ± 0.017 | 0.036 ± 0.003 | +0.825 | 0.038 |
| 128 | rl_warm | 3/3 | 94,77,91 | +0.869 ± 0.006 | 0.074 ± 0.012 | +0.638 | 0.099 |

Notes: `n_valid` = seeds where ≥3 clusters were occupied at >=`min_cluster_frac` (weighted Spearman / A_iso are undefined when fewer than 3 clusters are kept). `clusters/seed` shows the per-seed count of kept clusters. If all seeds collapse to <3 clusters, the latent space is effectively a point and the method is not using the meme channel meaningfully.
