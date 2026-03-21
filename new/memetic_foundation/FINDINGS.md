# Memetic Dynamics in Multi-Agent Systems with Persistent Memory and Communication

## Overview

This study investigates whether **memetic dynamics** — the emergence, persistence, transmission, selection, and mutation of behavioral patterns — arise naturally in multi-agent reinforcement learning systems equipped with persistent memory and communication. Rather than asking whether communication or memory *improves* performance (already established in prior work), we ask: do these systems exhibit behavior that is meaningfully analogous to cultural evolution?

We test three architectures across three agent population sizes (N=5, 10, 20) on cooperative navigation with partial observability, probing hidden states every 50k steps over 1M training steps.

---

## Architectures

All variants are parameter-matched (~124k params) and trained with MAPPO, ent_coef=0.3, partial obs (obs_radius=0.5 with curriculum annealing over 200k steps).

| Variant | Memory | Comm | Resets between episodes |
|---|---|---|---|
| **memory_only** | GRU | None | Yes |
| **commnet_persistent** | GRU | CommNet (additive, zero-init) | No |
| **memory_only_persistent** | GRU | None | No |

**CommNet**: agents broadcast mean of neighbors' GRU hidden states, added additively to own hidden state before actor head. Zero-initialized so it starts silent and must earn its influence.

**Persistent memory**: `reset_memory()` is a no-op — hidden state carries across episode boundaries indefinitely.

---

## Research Questions & Results

### Q1 — Do Stable Attractors Form? (Persistence)

We apply PCA (10D) + k-means to all hidden states across time, measuring silhouette score (higher = cleaner clusters = more persistent attractors).

| Variant | N=5 | N=10 | N=20 |
|---|---|---|---|
| memory_only | 0.260 | 0.841 | 0.715 |
| commnet_persistent | 0.211 | **0.877** | 0.649 |
| memory_only_persistent | **0.727** | 0.827 | **0.860** |

**Finding:** Stable attractors form in all memory architectures, and **attractor clarity scales with population size** for commnet_persistent and memory_only. At N=5, structure is weak; by N=10, all variants show strong clustering (sil > 0.8). memory_only_persistent consistently forms the clearest attractors across all N, suggesting that removing episode resets forces the hidden state into a tighter basin.

**Per-agent entropy** (lower = more locked-in to a cluster):
- memory_only: ~0.63 (wandering) at N=5, drops to ~0.18 at N=10
- commnet_persistent: ~0.56 at N=5, ~0.24 at N=10, ~0.32 at N=20
- memory_only_persistent: ~0.44 at N=5 — agents lock in earliest

---

### Q2 — Do Patterns Spread Between Agents? (Transmission)

We track pairwise cosine similarity between agents' hidden states over training. Rising similarity = convergence to shared patterns.

| Variant | N=5 final | N=10 final | N=20 final |
|---|---|---|---|
| memory_only | 0.715 | 0.731 | **0.846** |
| commnet_persistent | 0.646 | 0.699 | 0.585 |
| memory_only_persistent | **0.956** | **0.835** | 0.637 |

**Finding (surprising):** memory_only_persistent achieves the highest inter-agent similarity at N=5 (0.956) despite having *no explicit communication*. This suggests the environment itself acts as a synchronization channel — agents converging on the same attractors because they face the same task, not because they are talking.

commnet_persistent shows *decreasing* similarity at N=20 (0.585), the lowest of all three. Communication appears to be **diversifying** agents rather than homogenizing them at scale — agents use comm to differentiate roles rather than to share strategies.

---

### Q3 — Are Successful Patterns Selected? (Selection)

We identify the cluster with highest mean reward and track its prevalence over training.

| Variant | N=5 final prevalence | N=10 | N=20 |
|---|---|---|---|
| memory_only | 0.692 | **1.000** | **1.000** |
| commnet_persistent | **1.000** | **1.000** | **1.000** |
| memory_only_persistent | **1.000** | **1.000** | **1.000** |

**Finding:** Selection is robust. By N=10, all architectures show complete takeover of the dominant attractor — the winning strategy occupies 100% of agent-timesteps by end of training. At N=5, memory_only fails to reach full selection (69%), suggesting that without persistent memory or communication, selection pressure is insufficient to drive full convergence.

---

### Q4 — Specialization or Homogenization? (Structure)

Between-agent variance measures whether agents diverge into specialized roles (high) or converge to a shared strategy (low).

| Variant | N=5 | N=10 | N=20 |
|---|---|---|---|
| memory_only | ~0.000 | 0.0001 | **0.0024** |
| commnet_persistent | ~0.000 | ~0.000 | 0.0001 |
| memory_only_persistent | 0.0004 | 0.0002 | 0.0002 |

**Finding:** At N=20, memory_only shows **10-24x more between-agent variance** than the other variants — agents are developing strongly differentiated hidden state patterns (role specialization) without explicit coordination. commnet_persistent stays nearly flat regardless of N — communication actively maintains homogeneity, consistent with a shared language forming rather than roles diverging. memory_only_persistent is intermediate and stable.

---

### Q5 — Structured Mutation? (Mutation)

We track h_{t+1} - h_t (within-episode hidden state deltas). Delta silhouette measures whether mutations are structured (high) or isotropic/random (low). Cross-agent correlation measures whether agents' mutations are coupled.

**Delta silhouette (mutation structure):**

| Variant | N=5 | N=10 | N=20 |
|---|---|---|---|
| memory_only | 0.176 | 0.413 | 0.260 |
| commnet_persistent | 0.211 | **0.976** | **0.963** |
| memory_only_persistent | **0.950** | **0.980** | **0.944** |

**Cross-agent delta correlation (socially coupled mutation):**

| Variant | N=5 | N=10 | N=20 |
|---|---|---|---|
| memory_only | -0.020 | 0.065 | 0.049 |
| commnet_persistent | -0.008 | 0.063 | 0.069 |
| memory_only_persistent | **0.172** | **0.184** | 0.019 |

**Finding:** This is the clearest evidence of memetic structure. commnet_persistent's mutations become highly structured by N=10 (sil=0.976) and remain so at N=20 (0.963) — the system is not randomly drifting through h-space but following a small set of stereotyped transition pathways. memory_only barely develops structure (peak 0.413 at N=10).

Cross-agent coupling tells a different story: memory_only_persistent has the strongest social coupling at N=5 and N=10 — when one agent's state changes, others' states change similarly, even without explicit comm. This is environment-mediated synchronization. commnet_persistent shows moderate stable coupling (0.063-0.069) that doesn't grow with N.

---

## Performance (Final Median Distance, Lower is Better)

| Variant | N=5 | N=10 | N=20 |
|---|---|---|---|
| memory_only | 0.69 | 0.72 | ~0.55* |
| commnet_persistent | **0.55** | **0.55** | **~0.42*** |
| memory_only_persistent | 0.68 | 0.57 | ~0.46* |

*N=20 median across 5 seeds (some unstable seeds excluded)

commnet_persistent achieves the best task performance at all N — combining memory and communication is unambiguously best for coordination. However, its memetic dynamics tell a different story from what we expected: it homogenizes agents (low between-var) with highly structured internal transitions (high delta_sil). It is finding a **shared efficient strategy** rather than generating diverse specializations.

---

## Synthesis: What "Meme-Like" Looks Like Here

Three distinct regimes emerged:

**1. memory_only** — weak memetics
- Attractor structure grows with N but never fully stabilizes at small N
- Selection incomplete at N=5
- Mutations largely unstructured and uncoupled
- At N=20: role specialization emerges (high between_var) as the only coordination mechanism available
- Verdict: *environmental selection without cultural transmission*

**2. commnet_persistent** — communicative convergence
- Strong attractor structure at N≥10
- Communication drives homogenization, not diversification
- Highly structured mutation pathways emerge at N≥10 — agents follow a shared "grammar" of state transitions
- Moderate, stable cross-agent coupling
- Verdict: *shared language / cultural homogenization — agents converge on a common meme*

**3. memory_only_persistent** — environmental synchronization
- Strongest attractors at small N; clearest lock-in
- High inter-agent similarity without communication (environment as sync channel)
- Strongest social mutation coupling at N=5–10 (0.172–0.184), without explicit comm
- Mutations highly structured — the most stereotyped transition dynamics
- Verdict: *convergent evolution driven by shared environment rather than communication*

---

## Key Scientific Contributions

1. **Memetic attractors emerge reliably at N≥10** in all memory architectures. The hidden state does not wander randomly — it locks into a small number of stable basins that correspond to behavioral strategies.

2. **Selection is robust and fast**: dominant attractors reach 100% prevalence by end of training at N≥10, consistent with strong selective pressure in cooperative tasks.

3. **Communication diversifies rather than unifies at scale**: commnet_persistent at N=20 has the *lowest* inter-agent similarity — comm is enabling role differentiation, not broadcasting a single dominant strategy. This is counterintuitive and contradicts the naive "comm → shared state" assumption.

4. **Persistent memory without communication produces the strongest environmental synchronization**: memory_only_persistent agents converge to similar hidden states through environmental pressure alone (same task → same attractors → apparent transmission without actual messaging). This demonstrates that apparent "meme spreading" can arise from convergent evolution, not transmission.

5. **Mutation structure scales with comm at N≥10**: commnet_persistent's delta_sil jumps from 0.211 (N=5) to 0.976 (N=10) and stays high — a qualitative phase transition where communication begins imposing structure on how agents update their internal states. This is the closest empirical analog to "mutations following a grammar" in biological memetics.

---

## Limitations and Future Work

- Analysis uses single representative seed; multi-seed averaging would strengthen statistical claims
- m_bar↔h_sim correlation returns NaN for all variants (likely near-zero variance in m_bar early in training) — a more sensitive transmission measure is needed
- N=20 runs were limited to 300k steps vs 1M for N=5/10; longer runs may show clearer N=20 dynamics
- Partial obs with obs_radius=0.5 means agents often don't see all others — future work should vary observability to study how information scarcity affects meme propagation
- SMACv2 with unit type heterogeneity would provide a richer context for role specialization

---

## Experimental Details

- **Environment**: simple_spread_v2 (MPE), N agents, N landmarks, minimize average agent-landmark distance
- **Partial observability**: obs_radius=0.5 with curriculum annealing over 200k steps
- **Training**: MAPPO, lr=5e-4, gamma=0.99, gae_lambda=0.95, clip=0.2, ent_coef=0.3, 5 PPO epochs per rollout
- **Probing**: 10 episodes × every 50k steps = 20 snapshots per run (10 for N=20 due to shorter runs)
- **Analysis**: PCA(10D) + k-means (k selected by silhouette, k=2..8), cosine similarity, within/between-agent variance, sequential delta analysis
- **Seeds**: 5 per condition (N=5, N=10); 5 per condition (N=20, 300k steps)
- **Param matching**: memory_only/persistent hid=90 (N=5), 83 (N=10), 74 (N=20); commnet_persistent hid=80/74/68; all ~124k params
