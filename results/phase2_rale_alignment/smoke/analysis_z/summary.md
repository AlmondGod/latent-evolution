# RALE alignment summary (z)

- Probe root: `results/phase2_rale_alignment/smoke/probes`
- Methods: `['alec', 'rl']`
- Seeds: `[42]`
- K values: `[8]`
- Y target: `future` (H=10)
- Bootstrap iters: 30; Shuffle iters: 20

## Aggregate (mean ± SEM across seeds)

| K | method | weighted Spearman | A_iso | shuffle Spearman | shuffle A_iso |
|--:|:------:|:------------------:|:------:|:------------------:|:---------------:|
| 8 | alec | +0.917 ± nan | 0.232 ± nan | +0.735 | 0.167 |
| 8 | rl | +nan ± nan | nan ± nan | +nan | nan |
