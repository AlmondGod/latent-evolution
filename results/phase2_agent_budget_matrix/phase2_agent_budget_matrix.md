# Phase 2 Agent-Budget Matrix

This matrix defines the resource protocol for Phase 2 `ES-Comm` vs `RL-Comm` comparisons.
Here `N` is agents per team, `E` is parallel teams, and `A = N * E` is the total active agent budget.

| Regime | N | E | A=N*E | Transition Budget | ES Pop | ES Gens | RL Transitions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_a | 2 | 64 | 128 | 3072000 | 64 | 40 | 3072000 |
| fixed_a | 4 | 32 | 128 | 1536000 | 32 | 30 | 1536000 |
| fixed_a | 8 | 16 | 128 | 768000 | 16 | 20 | 768000 |
| linear_a | 2 | 16 | 32 | 768000 | 16 | 20 | 768000 |
| linear_a | 4 | 16 | 64 | 768000 | 16 | 20 | 768000 |
| linear_a | 8 | 16 | 128 | 768000 | 16 | 20 | 768000 |
| search_scaled | 2 | 4 | 8 | 192000 | 4 | 6 | 192000 |
| search_scaled | 4 | 8 | 32 | 384000 | 8 | 12 | 384000 |
| search_scaled | 8 | 16 | 128 | 768000 | 16 | 20 | 768000 |

Regimes:
- `fixed_a`: hold `A` constant, so larger `N` means fewer parallel teams `E = floor(A / N)`.
- `linear_a`: hold `E` constant, so total active agents `A` grows linearly with `N`.
- `search_scaled`: set `E ∝ N`, so total active agents `A` grows quadratically with `N` and ES gets a larger search population as teams grow.

Important: with the current runners this is an agent-equivalent budget protocol, not a literal wall-clock-equal parallel implementation, because the frozen backbone is not vectorized across multiple teams.