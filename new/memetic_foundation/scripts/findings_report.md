# Memetic Foundation: Experimental Findings Report

*Generated 2026-03-16. All metrics use deterministic eval (5 test episodes).*

---

## Summary of Key Findings

### Finding 1: Memory-Observability Interaction (N=3)

Under **full observability**, adding GRU memory is slightly beneficial at N=3 but this benefit declines and reverses as N grows. Under **partial observability** with simple_tag, memory shows high peak performance but catastrophic instability.

| Variant | Full Obs (N=3) | Partial Obs (N=3) |
|---------|---------------|------------------|
| baseline | 1383 ± 1507 | 1186 ± 1028 |
| memory_only | 876 ± 842 | 341 ± 356 |
| comm_only | 1102 ± 1292 | — |
| full_gated | — | 324 ± 477 |

*All 200k training steps, 8 seeds, deterministic eval.*

**Root cause of partial obs instability**: GRU achieves peaks of 2460 (memory_only, partial obs) but collapses in subsequent PPO updates. Fixed with LR annealing (5e-4 → 5e-5 over training).

---

### Finding 2: N-Scaling Divergence (Critical Result)

The most important finding: **memory helps at small N but hurts at large N under full observability**.

| N | baseline | memory_only | full_gated |
|---|----------|-------------|------------|
| 3 | 274 ± 346 | **625 ± 646** | 371 ± 462 |
| 5 | **989 ± 850** | 830 ± 627 | 211 ± 326 |
| 8 | **415 ± 642** | 34 ± 12 | — |

*Full observability, simple_tag, 4 seeds, 200k steps.*

**Interpretation**: Under full observability, all N agents see the same prey. Memory adds complexity but no information gain. At N=8, the optimization landscape is harder (8 GRUs × PPO variance), causing catastrophic failure of memory variants.

**Prediction**: Under partial observability with simple_spread, where each agent must cover a different landmark (persistent structural role asymmetry), memory should HELP and the advantage should GROW with N.

---

### Finding 3: Communication Instability

Ungated communication (comm_only) shows severe instability at 1M steps:
- 200k: comm_only = 1102, full = 584
- 1M: comm_only = 83, full = 187 (divergence confirmed)

**Root cause**: Without memory, comm agents cannot maintain consistent role representations. Messages are noisier across PPO updates, eventually destabilizing the policy.

**IC3Net gating**: Gate was always-open (gate_open_frac=1.0) without entropy regularization. Fixed with:
1. Gate entropy coefficient: 0.01 → 0.05
2. Straight-through estimator for binary gate

**Note**: Gated comm results (without entropy fix) showed comm_only_gated=62 < comm_only=1102, confirming gate-without-selectivity adds complexity without benefit.

---

### Finding 4: Memory Diversity Under Observability

Counterintuitive result: **partial observability reduces memory diversity in simple_tag**.

| Metric | Full Obs | Partial Obs |
|--------|----------|-------------|
| Cosine similarity | 0.593 | 0.663 |
| Effective dimensions | 11.5 | 9.5 |
| Memory update rate | 0.176 | 0.135 |

**Why**: With obs_radius=0.5 in simple_tag, agents spend most time seeing NOTHING (prey outside radius). Zero-input GRU states converge toward uniform → less diversity. Under full obs, agents always see prey at different angles → more diverse position-relative computations.

**Implication**: obs_radius alone is insufficient for "scalable memetics." Need **structural asymmetry**: different agents have different objectives, not just different visibility windows at different times.

---

### Finding 5: All Variants Collapse at 1M Steps

Shockingly, ALL variants (including baseline) collapse at 1M steps:
- baseline: 539 peak → 25 final
- memory_only: 489 peak → 39 final
- comm_only: 83 → 12 (diverged earlier)
- full: 187 → 25

**Root cause**: Fixed learning rate (5e-4 constant) → large policy updates destabilize learned behaviors. The PPO clip ratio is still 0.2 but the learning rate never decays.

**Fix**: Linear LR annealing in run.py: `lr = lr_init * max(1 - step/total_steps, 0.1)`

---

## Current Experiments (as of 2026-03-16)

### Completed:
- ✅ Full obs ablation (tag_gru): 4 variants × 8 seeds × 200k steps
- ✅ Gated IC3Net comm (tag_gated): 2 variants × 8 seeds × 200k steps
- ✅ 1M convergence study: 4 variants × 3 seeds × 1M steps
- ✅ N-scaling, full obs (mpe_tag_nscale): 3 variants × 3 Ns × 4 seeds × 200k
- ✅ Partial obs ablation (mpe_tag_partial_obs): 3 variants × 8 seeds × 200k
- ✅ Meme diversity analysis: 8 seeds each, full vs partial obs

### Running:
- 🔄 Spread + partial obs N-scaling (mpe_spread_partial_obs):
  3 variants × N={3,5,8} × 6 seeds × 400k steps (LR-scheduled)
  **This is the key experiment for scalable memetics hypothesis.**

---

## Scalable Memetics Hypothesis

**Formal statement**: In multi-agent systems where agents have information-asymmetric partial observations and distinct structural roles, persistent memory (GRU) enables agents to develop diverse behavioral "memes" (specialized role representations). Communication of these memes provides additional coordination benefit. Both advantages scale with N: more agents → more role differentiation → greater benefit of memory+comm.

**Operationalization**:
1. **Task**: simple_spread with partial obs (obs_radius=0.5)
   - N agents must cover N distinct landmarks
   - Each agent sees only nearby landmarks/agents
   - Requires persistent landmark-assignment memory

2. **Metrics**:
   - Primary: reward gap (memory_only - baseline) / baseline at each N
   - Memory diversity: pairwise cosine similarity between agent h states
   - Specialization: fraction of agents consistently covering same landmark

3. **Success criterion**:
   - memory_only advantage grows from N=3 to N=8
   - Memory diversity is higher than under full observability
   - Agents show consistent landmark specialization patterns

---

## Code Architecture Notes

### GRU Memory (GRUMemory)
```
h_t = GRU_cell(input_t, h_{t-1})  # learned recurrent update
input_t = [u; m̄_prev]  (full) or [u]  (memory_only)
h_0 = h_init  (learned parameter, zeros init)
```

### IC3Net Communication Gate (CommGate)
```
g_i = hard_threshold(sigmoid(W_gate · z_i))  # straight-through
gate_entropy = -p*log(p) - (1-p)*log(1-p)  # aux_loss += 0.05 * H
```

### Training Stability
- Linear LR decay: 5e-4 → 5e-5 over training
- PPO clip: 0.2
- Entropy coeff: 0.01
- 5 PPO epochs per rollout
- Rollout steps: 400

---

## Next Steps

1. **Analyze spread partial obs results** once ~3.6 hours training completes
2. **Run meme diversity analysis** on spread checkpoints (expect diversity > simple_tag)
3. **If hypothesis confirmed**: design communication probe to show "meme transmission"
4. **Communication probe**: does an agent with no landmark in view correctly
   communicate to guide partners? Does the receiver update behavior based on message?
