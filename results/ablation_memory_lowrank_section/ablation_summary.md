# Ablation Summary

## Paper-ready text

Persistent memory is the most stable simple-spread variant as team size grows. On partial-observation MPE simple_spread, `memory_only` keeps final mean coverage distance low at every tested team size (`0.678 ± 0.059`, `0.695 ± 0.180`, `0.613 ± 0.065` at `N=3,5,8`; `n=6` each), while the memory-free baseline rises to `9.815 ± 9.203`, `16.325 ± 12.717`, and `13.425 ± 12.945`. The SGD-trained `full_gated` memory+communication variant is less stable still, reaching `10.225 ± 9.575`, `38.275 ± 16.793`, and `26.127 ± 23.589` (`n=6,6,3`). Final rewards show the same pattern: `memory_only` remains between `-646.90 ± 31.55` and `-1200.81 ± 80.29`, whereas baseline and full_gated are both much worse and much more variable.

On a single Apple M3 Pro MacBook Pro (12 CPU cores, 36 GB unified memory; CPU execution), ALEC is faster overall than matched RL adaptation but not in every VMAS transport setting. Across the 27 paired phase-2 runs in the current artifact set, ALEC averages `3601.0 ± 654.6` seconds versus `4493.5 ± 599.0` for RL, and is faster in `22/27` pairings.

Low-rank state updates are slightly better and more stable than the dense alternative on VMAS Discovery at `N=12`. The standard low-rank `z` update reaches reward gain `+0.0885 ± 0.0138` and improves in `3/3` seeds, compared with `+0.0755 ± 0.0478` and `2/3` for dense updates and `+0.0547 ± 0.0430` and `2/3` with `z` disabled. It is also materially cheaper wall-clock.

## Memory scaling table

| N | Baseline Reward | Memory-Only Reward | Memory+Comm Reward | Baseline mean_dist | Memory-Only mean_dist | Memory+Comm mean_dist | Seeds (B/M/F) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | -3257.73 ± 2634.42 | -646.89 ± 31.55 | -3997.40 ± 3371.98 | 9.815 ± 9.203 | 0.678 ± 0.059 | 10.225 ± 9.575 | 6/6/6 |
| 5 | -8577.72 ± 6199.96 | -868.27 ± 158.39 | -18768.44 ± 7951.74 | 16.325 ± 12.717 | 0.695 ± 0.180 | 38.275 ± 16.793 | 6/6/6 |
| 8 | -11124.43 ± 10070.12 | -1200.81 ± 80.29 | -21864.15 ± 18150.99 | 13.425 ± 12.945 | 0.613 ± 0.065 | 26.127 ± 23.589 | 6/6/3 |

## Phase-2 wall-clock table

| Setting | ALEC Time (s) | MAPPO Time (s) | Paired Runs | ALEC Faster |
| --- | ---: | ---: | ---: | ---: |
| MPE | 1156.0 ± 6.5 | 1312.9 ± 17.0 | 3 | 3/3 |
| VMAS Discovery | 2792.0 ± 478.8 | 5359.0 ± 1127.6 | 12 | 12/12 |
| VMAS Transport | 5021.4 ± 1292.5 | 4423.2 ± 581.4 | 12 | 7/12 |
| Overall | 3601.0 ± 654.6 | 4493.5 ± 599.0 | 27 | 22/27 |

## Low-rank ablation table

| Variant | Baseline Reward | Final Reward | Reward Gain | Time (s) | Improved | Seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| with $z$, low-rank update | 1.4792 ± 0.0114 | 1.5677 ± 0.0213 | +0.0885 ± 0.0138 | 5267.8 ± 170.8 | 3/3 | 3 |
| without $z$ | 1.5078 ± 0.0798 | 1.5625 ± 0.0413 | +0.0547 ± 0.0430 | 7920.8 ± 1026.8 | 2/3 | 3 |
| with $z$, dense update | 1.5234 ± 0.0502 | 1.5990 ± 0.0026 | +0.0755 ± 0.0478 | 16163.8 ± 634.0 | 2/3 | 3 |
