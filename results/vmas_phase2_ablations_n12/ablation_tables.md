# VMAS Discovery N=12 Ablation Tables

These tables summarize reward deltas over the frozen Phase-1 backbone.

## Resource Ablation


| Regime   | Method | E   | A   | Budget              | Seed 1 Δ | Seed 2 Δ | Seed 3 Δ | Mean Δ  | Improved |
| -------- | ------ | --- | --- | ------------------- | -------- | -------- | -------- | ------- | -------- |
| linear_a | ES     | 16  | 192 | pop 16, gen 20      | +0.1094  | +0.0938  | +0.0625  | +0.0885 | 3        |
| linear_a | RL     | 16  | 192 | 768,000 transitions | +0.0781  | +0.1562  | +0.0234  | +0.0859 | 3        |
| fixed_a  | ES     | 10  | 120 | pop 10, gen 14      | +0.1562  | +0.2266  | +0.0391  | +0.1406 | 3        |
| fixed_a  | RL     | 10  | 120 | 480,000 transitions | +0.0156  | +0.1875  | +0.2031  | +0.1354 | 3        |


## Architecture Ablation


| Variant             | Seed 1 Δ | Seed 2 Δ | Seed 3 Δ | Mean Δ  | Improved | Notes                   |
| ------------------- | -------- | -------- | -------- | ------- | -------- | ----------------------- |
| with_z_low_rank     | +0.1094  | +0.0938  | +0.0625  | +0.0885 | 3        | linear_a baseline       |
| without_z           | -0.0312  | +0.0938  | +0.1016  | +0.0547 | 2        | disable_z=True          |
| with_z_dense_update | +0.0000  | +0.0625  | +0.1641  | +0.0755 | 2        | dense_state_update=True |


## Mean Reward Context


| Variant                  | Mean Baseline Reward | Mean Final Reward |
| ------------------------ | -------------------- | ----------------- |
| linear_a (ES)            | 1.4792               | 1.5677            |
| linear_a (RL)            | 1.4661               | 1.5521            |
| fixed_a (ES)             | 1.4349               | 1.5755            |
| fixed_a (RL)             | 1.4609               | 1.5964            |
| with_z_low_rank (ES)     | 1.4792               | 1.5677            |
| without_z (ES)           | 1.5078               | 1.5625            |
| with_z_dense_update (ES) | 1.5234               | 1.5990            |


