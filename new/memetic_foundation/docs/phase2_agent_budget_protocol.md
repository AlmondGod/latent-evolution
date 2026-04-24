# Phase 2 Agent-Budget Protocol

This protocol defines how to compare `ES-Comm` and `RL-Comm` under explicit agent-resource budgets.

## Quantities

- `N`: agents inside one cooperating team or environment
- `E`: parallel teams or candidate evaluations
- `A = N * E`: total active agent budget

The goal is to separate:

1. sample efficiency under a fixed transition budget
2. resource efficiency under a fixed total active-agent budget
3. scaling behavior when larger deployed populations allow larger memetic search populations

## Regimes

### Fixed-A

Hold `A` constant and vary `N`.

- `E = floor(A / N)`
- Larger teams get fewer parallel teams
- This is the conservative resource-fair comparison

Use this to answer:

- Does `ES-Comm` still beat `RL-Comm` when the total deployed agent budget is fixed?

### Linear-A

Hold `E` constant and vary `N`.

- `A = N * E`
- Total active agents grow linearly with team size

Use this to answer:

- If the system simply gets bigger, does `ES-Comm` benefit more than `RL-Comm` from the increased deployed population?

### Search-Scaled

Let `E` grow with `N`.

- Example: `E = kN`
- Then `A = kN^2`

This is the strongest memetic-search regime because:

- larger teams
- larger deployed populations
- larger ES search populations

Use this to answer:

- Can explicit memetic search exploit larger deployed populations better than gradient adaptation?

## Operationalization with Current Runners

The current Phase 2 runners are:

- [run_memetic_selection_phase2.py](/Users/almondgod/Repositories/memeplex-capstone/new/memetic_foundation/scripts/run_memetic_selection_phase2.py)
- [run_memetic_rl_phase2.py](/Users/almondgod/Repositories/memeplex-capstone/new/memetic_foundation/scripts/run_memetic_rl_phase2.py)

These are not literally vectorized across `E` parallel teams. So for now this is an **agent-equivalent budget protocol**, not a literal equal-wall-clock parallel benchmark.

We approximate equal deployed-agent budget by deriving runner settings from `E`.

### RL-Comm

Choose a per-team step budget `B_team`.

- `train_transitions = E * B_team`

Interpretation:

- if `E` parallel teams each contribute `B_team` steps over the same adaptation window,
- then the total transition budget is `E * B_team`

### ES-Comm

Use:

- `population_size = E`

Then one ES generation costs approximately:

- `episode_steps * (population_size * eval_episodes + elite_k * elite_eval_episodes)`

Choose the number of generations so the realized ES transition budget is as close as possible to the RL transition budget for the same row.

## Current Limitation

Positive-sum Phase 2 on VMAS also requires pretrained `attention_hu_actor` backbones for each `N`.

At the moment:

- the Phase 2 adaptation path is working
- but the checked-in Phase 1 `attention_hu_actor` training source is missing from the live worktree
- and the current [agent_network.py](/Users/almondgod/Repositories/memeplex-capstone/new/memetic_foundation/models/agent_network.py) no longer exposes the exact `attention_hu_actor` training path used to create the existing MPE checkpoints

So the next implementation step is:

1. restore or rebuild the Phase 1 `attention_hu_actor` trainer
2. train VMAS transport/discovery backbones for `N = 2, 4, 8`
3. run the Phase 2 budget matrix on top of those checkpoints

## Output Helper

Use [build_phase2_agent_budget_matrix.py](/Users/almondgod/Repositories/memeplex-capstone/new/memetic_foundation/scripts/build_phase2_agent_budget_matrix.py) to generate concrete `fixed_a`, `linear_a`, and `search_scaled` rows before launching runs.
