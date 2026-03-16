# Theoretical Framework: Scalable Memes in Multi-Agent Systems

*Analysis of why GRU memory prevents catastrophic failure and enables scalable coordination*

---

## 1. The Symmetry-Breaking Problem

In **simple_spread with partial observability**, all N agents begin with identical:
- Random initialization weights
- Learned initial hidden state h₀
- Same policy network

To achieve optimal coverage, agents must **specialize**: each should cover a different landmark.
But starting from identical initialization, what breaks the symmetry?

**Answer**: Random exploration. In the first few hundred steps, stochastic actions push agents
to different positions. The first agent near landmark 0 gets positive reward for proximity.
The key question is: **can the agent REMEMBER this experience?**

- **Without memory**: PPO updates the shared policy, but the agent has no persistent state.
  On the next episode reset, the agent starts from h₀ and has "forgotten" which landmark it covers.

- **With GRU memory**: The hidden state h accumulates a "landmark fingerprint":
  ```
  h_{t+1} = GRU(obs_t, h_t)
  ```
  After repeated visits to landmark 0, h encodes the position, direction, and identity
  of landmark 0. This creates a stable attractor that persists across episodes.

---

## 2. Why Memory Converts Permanent Catastrophes to Transient Ones

**Baseline catastrophic failure mechanism** (observed at all N):
1. Random initialization pushes some agents far from all landmarks
2. At large distances, obs_radius=0.5 means ALL observations are zero
3. With zero input, the policy π(a|o=0) = uniform (no gradient from rewards=~0)
4. Random actions in unbounded space → agents drift further away
5. Failure is **permanent** (absorbing state): agent stuck in zero-observation regime

**Memory_only N=8 observed behavior** (the key insight from experimental data):
```
Step 20k:  dist=78.13  ← enters catastrophic state (same trigger as baseline)
Step 80k:  dist=0.91   ← RECOVERED
Step 100k: dist=14.32  ← partial drift
Step 120k: dist=0.48   ← RECOVERED again
Step 180k: dist=0.40   ← stable coverage achieved
```

**Why memory converts permanent → transient failure**:
1. When obs=0, GRU update: h_t = GRU(0, h_{t-1}) = r⊙h_{t-1} + ... (partial decay)
2. h decays toward 0 but doesn't instantly collapse
3. Policy π(a|h≈ε, o=0) with small but nonzero h can maintain weak directional preference
4. This weak preference is enough to bias random exploration toward previously observed space
5. With PPO learning: the policy progressively learns that h≈ε → "explore toward landmarks"
6. Eventually, the exploring agent enters a radius where it can observe a landmark
7. Once a landmark is observed, normal learning resumes → agent recovers

**Critical distinction**:
- Baseline: catastrophic states are ABSORBING (π(a|o=0) provides no escape gradient)
- Memory_only: catastrophic states are TRANSIENT (h provides a weak restoring bias)

**Mathematical insight**: The GRU reservoir effect: even with zero input, h maintains
information about past trajectories. The policy can use this "trace" to bias exploration
toward historically informative regions. This is fundamentally different from a stateless policy.

---

## 3. Observed Catastrophic Failure Rates

**Observed catastrophic rates** (baseline, dist > 50):
| N | Catastrophic rate | Mean dist (all seeds) |
|---|-------------------|-----------------------|
| 3 | 1/6 = 16.7% | 9.82 |
| 5 | 1/6 = 16.7% | 16.33 |
| 8 | 1/6 = 16.7% (preliminary) | ~13 |

**Surprising finding**: The catastrophic rate is CONSTANT (~1/6) across N=3,5,8.
The initial prediction of increasing catastrophic rate was based on partial data (seeds 1-3 at
N=5 showed 2/3, which was later resolved to 1/6 with 6 complete seeds).

**What DOES grow with N**: The SEVERITY of non-catastrophic failures and partial catastrophes.
- N=5 showed a "partial catastrophe" at dist=16.45 (no equivalent at N=3 or N=8)
- The mean dist for baseline grows with N partly due to these partial failures
- The reward magnitude of catastrophic failures grows with N (more agents failing to
  cover more landmarks → exponentially larger negative reward)

**Memory_only's immunity**: With N=3,5 data (12 seeds total), 0 catastrophic failures.
The GRU restoration mechanism is N-independent: each agent independently maintains a
homing signal regardless of what other agents do.

---

## 4. The Meme Mechanism: Hidden State as Role Representation

Probe results show that GRU hidden states encode landmark roles with high accuracy:

| N | h→role accuracy | above-chance | interpretation |
|---|-----------------|--------------|----------------|
| 3 | 96.8% | 2.90x | h encodes landmark assignment |
| 5 | 91.1% | 4.56x | h encodes role with greater differentiation |

The **above-chance ratio grows** from N=3 to N=5 because:
- With more agents and more landmarks, the learned role representations become MORE distinct
- Each of 5 roles requires a more unique h fingerprint than each of 3 roles
- The GRU develops richer representational geometry at larger N

This is the core "meme" property: **the hidden state is a learned, persistent unit of
role specialization that becomes more structured as the task scales**.

---

## 5. Communication as Meme Transmission

When communication is added (full_gated), agents transmit their h-derived messages to others.
Probe results:

| N | message→role accuracy | above-chance |
|---|----------------------|--------------|
| 3 | 97.1% | 2.91x |
| 5 | 88.6% | 4.43x |

Messages carry role information that scales similarly to hidden states. The IC3Net gate
learns to selectively transmit role-relevant information.

**Why communication helps** (in theory): If agent A is far from its landmark and agent B
has covered agent A's landmark temporarily, agent B's message can inform agent A to continue
moving toward its assigned landmark rather than trying to cover agent B's landmark.

**Why communication hurts at scale** (observed): Gate entropy regularization forces each of
N gates to learn selectivity simultaneously. With N=8, this is 8 independent gate-learning
problems. The joint optimization is harder, leading to the observed 50%+ catastrophic rates.

**Trade-off**: full_gated has richer representation (eff_dim 5.3 at N=5 vs 3.0 for memory_only)
but is less stable (50% vs 0% catastrophic at N=5).

---

## 6. N=8 Predictions (Theory-Driven)

Based on the observed patterns:

**Baseline N=8**:
- Expected catastrophic rate: ~50% (extrapolating from 17%→33% trend)
- Expected mean dist (all seeds): ~30+ (driven by ~50% catastrophic failures at dist=78+)
- Good seeds: dist ~0.5-1.5 (similar to N=5 good seeds)

**Memory_only N=8**:
- Expected catastrophic rate: **0%** (GRU restoration mechanism is N-independent)
- Expected mean dist: ~0.7-1.0 (slightly worse than N=5 due to coordination complexity)
- Expected memory advantage: ~30-40x over baseline

**Full_gated N=8**:
- Expected catastrophic rate: ~67%+ (scaling from 17%→50% trend)
- Converged seeds will show richest meme content (eff_dim ~8+, 6.5x above-chance probing)

**If memory_only maintains 0% catastrophic failures at N=8**: This definitively establishes
that GRU memory provides a N-invariant protection against catastrophic failures, making it
the critical component for scalable multi-agent coordination.

---

## 7. Implications for Scalable Memetics

**Definition**: A multi-agent system exhibits "scalable memetics" if:
1. Agents develop distinct behavioral specializations ("memes")
2. These specializations are transmitted via communication
3. The benefit of (1) and (2) grows with N

**Our results support (1) and (3)** for memory_only:
- Hidden states encode roles (96.8% at N=3, 91.1% at N=5 — robust at scale)
- Memory advantage grows (14.5x at N=3, 23.5x at N=5, predicted ~35x at N=8)
- Memory diversity grows (eff_dim: 1.3→3.0 from N=3→5)

**Result (2) is complicated**: full_gated communicates memes effectively (4.4x above-chance
at N=5) but the communication system is unstable at scale. The communication benefit exists
but the training instability dominates.

**Future direction**: A communication architecture that avoids gate-entropy instability while
preserving meme transmission would allow all three properties to scale together. Possible
approaches:
- Fixed (non-learned) communication topology based on nearest-neighbor proximity
- Gradually reducing gate entropy coefficient with N (smaller regularization at larger N)
- Attention-based communication with stable softmax (no binary gate)
- Emergent division of labor via role tokens (add per-agent learned role embeddings)
