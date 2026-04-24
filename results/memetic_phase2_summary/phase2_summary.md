# Phase 2 Summary

Comparison on MPE `simple_spread_v2`, `N=8`, using the 128-episode paired intervention evaluation.

| Method | Mean Reward ± SE | Mean min_dist ± SE | Seeds | Reward vs Frozen | min_dist vs Frozen | Silence dReward | Shift dReward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen | -449.20 ± 42.45 | 0.428 ± 0.047 | 3 | +0.00 | +0.000 | -9.63 | +5.98 |
| ALEC | -430.84 ± 24.79 | 0.407 ± 0.027 | 3 | +18.36 | -0.021 | -25.85 | -8.19 |
| MAPPO adapter | -438.93 ± 32.85 | 0.415 ± 0.035 | 3 | +10.27 | -0.012 | -16.80 | -6.44 |

More negative `Silence dReward` / `Shift dReward` means the intervention hurt more, which indicates the policy is relying more on communication.