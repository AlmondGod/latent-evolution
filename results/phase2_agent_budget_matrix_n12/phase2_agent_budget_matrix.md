# Phase 2 Agent-Budget Matrix

This matrix defines the resource protocol for Phase 2 `ES-Comm` vs `RL-Comm` comparisons.
Here `N` is agents per team, `E` is parallel teams, and `A = N * E` is the total active agent budget.

| Regime | N | E | A=N*E | Transition Budget | ES Pop | ES Gens | RL Transitions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_a | 12 | 10 | 120 | 480000 | 10 | 14 | 480000 |
| linear_a | 12 | 16 | 192 | 768000 | 16 | 20 | 768000 |
| search_scaled | 12 | 24 | 288 | 1152000 | 24 | 25 | 1152000 |

Regimes:
- `fixed_a`: hold `A` constant, so larger `N` means fewer parallel teams `E = floor(A / N)`.
- `linear_a`: hold `E` constant, so total active agents `A` grows linearly with `N`.
- `search_scaled`: set `E ∝ N`, so total active agents `A` grows quadratically with `N` and ES gets a larger search population as teams grow.

Important: with the current runners this is an agent-equivalent budget protocol, not a literal wall-clock-equal parallel implementation, because the frozen backbone is not vectorized across multiple teams.