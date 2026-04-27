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
| 8 | alec | 3/3 | 8,8,8 | +0.961 ± 0.018 | 0.145 ± 0.032 | +0.367 | 0.618 |
| 8 | rl_warm | 3/3 | 8,8,8 | +0.855 ± 0.081 | 0.084 ± 0.043 | +0.173 | 0.942 |
| 12 | alec | 3/3 | 12,12,12 | +0.955 ± 0.006 | 0.137 ± 0.031 | +0.524 | 0.665 |
| 12 | rl_warm | 3/3 | 12,12,12 | +0.888 ± 0.054 | 0.084 ± 0.043 | +0.310 | 1.048 |
| 16 | alec | 3/3 | 16,16,16 | +0.959 ± 0.006 | 0.157 ± 0.047 | +0.587 | 0.690 |
| 16 | rl_warm | 3/3 | 16,16,16 | +0.908 ± 0.014 | 0.084 ± 0.020 | +0.326 | 1.087 |
| 20 | alec | 3/3 | 20,20,20 | +0.970 ± 0.007 | 0.138 ± 0.017 | +0.624 | 0.727 |
| 20 | rl_warm | 3/3 | 20,20,20 | +0.907 ± 0.039 | 0.082 ± 0.014 | +0.403 | 1.102 |
| 24 | alec | 3/3 | 24,24,23 | +0.968 ± 0.008 | 0.177 ± 0.033 | +0.647 | 0.719 |
| 24 | rl_warm | 3/3 | 24,24,24 | +0.921 ± 0.015 | 0.086 ± 0.017 | +0.397 | 1.157 |
| 32 | alec | 3/3 | 32,32,31 | +0.971 ± 0.005 | 0.186 ± 0.024 | +0.676 | 0.779 |
| 32 | rl_warm | 3/3 | 31,32,32 | +0.936 ± 0.023 | 0.124 ± 0.036 | +0.450 | 1.161 |
| 64 | alec | 3/3 | 59,62,59 | +0.975 ± 0.002 | 0.238 ± 0.050 | +0.748 | 0.793 |
| 64 | rl_warm | 3/3 | 57,61,60 | +0.930 ± 0.014 | 0.247 ± 0.042 | +0.513 | 1.189 |
| 128 | alec | 3/3 | 83,99,91 | +0.965 ± 0.004 | 0.304 ± 0.078 | +0.765 | 0.789 |
| 128 | rl_warm | 3/3 | 91,97,98 | +0.935 ± 0.009 | 0.282 ± 0.060 | +0.522 | 1.202 |

Notes: `n_valid` = seeds where ≥3 clusters were occupied at >=`min_cluster_frac` (weighted Spearman / A_iso are undefined when fewer than 3 clusters are kept). `clusters/seed` shows the per-seed count of kept clusters. If all seeds collapse to <3 clusters, the latent space is effectively a point and the method is not using the meme channel meaningfully.
