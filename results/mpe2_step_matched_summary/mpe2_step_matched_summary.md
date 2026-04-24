# MPE2 Step-Matched Summary

Step-matched comparison at 200k environment steps on `simple_tag_v3`.
Expected seeds: 42, 43, 44, 45, 46.

| Method | Completed Seeds | Missing Seeds | Predator Elo | Seeds |
| --- | --- | --- | ---: | ---: |
| PPO | 42, 43, 44, 45, 46 | -- | 1230.3 ± 14.5 | 5 |
| MADDPG | 42, 43, 44, 45, 46 | -- | 1195.1 ± 4.9 | 5 |
| LA-PPO | 42, 43, 44, 45, 46 | -- | 1192.3 ± 12.1 | 5 |

Per-seed sources:

- PPO seed 42: Elo 1193.95, source `/Users/almondgod/Repositories/memeplex-capstone/old/experiment_results.json`
- PPO seed 43: Elo 1234.97, source `/Users/almondgod/Repositories/memeplex-capstone/results/mpe2_step_matched_seed_sweep/ppo/seed43/results.json`
- PPO seed 44: Elo 1268.84, source `/Users/almondgod/Repositories/memeplex-capstone/results/mpe2_step_matched_seed_sweep/ppo/seed44/results.json`
- PPO seed 45: Elo 1200.77, source `/Users/almondgod/Repositories/memeplex-capstone/results/mpe2_step_matched_seed_sweep/ppo/seed45/results.json`
- PPO seed 46: Elo 1252.90, source `/Users/almondgod/Repositories/memeplex-capstone/results/mpe2_step_matched_seed_sweep/ppo/seed46/results.json`
- MADDPG seed 42: Elo 1198.40, source `/Users/almondgod/Repositories/memeplex-capstone/old/experiment_results.json`
- MADDPG seed 43: Elo 1199.51, source `/Users/almondgod/Repositories/memeplex-capstone/results/mpe2_step_matched_seed_sweep/maddpg/seed43/results.json`
- MADDPG seed 44: Elo 1193.04, source `/Users/almondgod/Repositories/memeplex-capstone/results/mpe2_step_matched_seed_sweep/maddpg/seed44/results.json`
- MADDPG seed 45: Elo 1206.66, source `/Users/almondgod/Repositories/memeplex-capstone/results/mpe2_step_matched_seed_sweep/maddpg/seed45/results.json`
- MADDPG seed 46: Elo 1177.71, source `/Users/almondgod/Repositories/memeplex-capstone/results/mpe2_step_matched_seed_sweep/maddpg/seed46/results.json`
- LA-PPO seed 42: Elo 1201.91, source `/Users/almondgod/Repositories/memeplex-capstone/old/experiment_results.json`
- LA-PPO seed 43: Elo 1193.62, source `/Users/almondgod/Repositories/memeplex-capstone/results/mpe2_step_matched_seed_sweep/method_i/seed43/results.json`
- LA-PPO seed 44: Elo 1179.52, source `/Users/almondgod/Repositories/memeplex-capstone/results/mpe2_step_matched_seed_sweep/method_i/seed44/results.json`
- LA-PPO seed 45: Elo 1156.62, source `/Users/almondgod/Repositories/memeplex-capstone/results/mpe2_step_matched_seed_sweep/method_i/seed45/results.json`
- LA-PPO seed 46: Elo 1229.89, source `/Users/almondgod/Repositories/memeplex-capstone/results/mpe2_step_matched_seed_sweep/method_i/seed46/results.json`