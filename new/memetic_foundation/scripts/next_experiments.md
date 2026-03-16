# Next Experiments for Scalable Memetics

*Generated 2026-03-16 — based on findings from spread partial obs N-scaling experiment*

---

## Problem: Bimodal Training Distribution

**Observed**: ~15-25% of seeds fail catastrophically in simple_spread partial obs:
- Good seeds: reward ≈ -471 to -726, dist ≈ 0.45-0.72
- Failed seeds: reward ≈ -8000 to -23000, dist ≈ 27-78

**Root cause**: When all agents initialize far from all landmarks (under obs_radius=0.5),
they receive zero observations. Random policy pushes them further away → value function
corrupted by extreme negative rewards → training never recovers.

**Two complementary fixes:**

### Fix A: Curriculum Obs Radius (Recommended)
Train with obs_radius starting large (full obs), annealing to 0.5:
```
obs_radius(t) = obs_radius_final + (obs_radius_init - obs_radius_final) * max(0, 1 - t/warmup_steps)
obs_radius_init = None (full obs)  →  obs_radius_final = 0.5
warmup_steps = 100000 (first 25% of training)
```
- Agents learn basic coverage behavior under full obs
- Gradually lose visibility, forcing memory-based retention
- By step 100k, they've already developed useful policies
- Memory now has something to work with from the start

### Fix B: Initial Position Clamping
Force agents to start within 1.5 obs_radius of some landmark:
```python
# In MPEWrapper.reset(): reject initializations where all landmarks are > 1.0 * obs_radius away
```
- Ensures agents always have at least one landmark in view at episode start
- More robust to bad random seeds
- Doesn't change the structural challenge after step 1

### Fix C: Entropy-Bounded Policy
Add KL penalty to keep policy from going too far from uniform in first 20k steps:
```
entropy_coeff = entropy_coeff_max * max(0, 1 - step/warmup_steps)
```
- Keeps policy exploratory early on → prevents premature convergence on bad trajectories
- Currently entropy_coeff = 0.01 fixed; could start at 0.05 and decay to 0.01

---

## Experiment 2: Symmetry Breaking for Specialization

**Problem**: With full parameter sharing, agents start identical and rely on observation
differences alone to specialize. Under obs_radius=0.5, early observations are often zero
for all agents → no specialization pressure early.

**Design**: Add a small per-agent "role token" to the observation:
```
obs_with_role = [obs; one_hot(agent_id, N)]  # N-dimensional one-hot
```
This doesn't hardcode roles — it just gives the GRU a signal to condition on.
The GRU can learn to USE the role token differently for each slot.

**Expected outcome**: Role tokens → stronger specialization → higher diversity metrics
**Ablation**: compare role_token=True vs role_token=False within memory_only variant

**Why this supports memetics**: The role token acts as an initial "seed meme" — a
minimal distinguishing signal that the GRU amplifies into a full role representation.
The role token is NOT the meme; the GRU hidden state is the meme.

---

## Experiment 3: Communication Causality at Scale

**After** we have good full_gated checkpoints from spread N={3,5,8}:

Run the comm_probe.py script to measure:
1. Reward loss from comm silence (how much does silencing comms hurt?)
2. Blind agent performance with vs without comm
3. Hidden state dynamics of blind agent

**Key prediction**: comm_causal_effect GROWS with N under partial obs + spread
- N=3: ~10-20% reward drop from silencing
- N=5: ~20-35% reward drop from silencing
- N=8: ~35-50% reward drop from silencing

This would show that communication value grows super-linearly with team size — the
"scalable memetics" claim in its strongest form.

---

## Experiment 4: Meme Transmission Dissection

The "meme" in scalable memetics should be:
1. **Encoded** in agent i's GRU hidden state h_i
2. **Transmitted** through the communication channel (message m̄_i)
3. **Decoded** by agent j into an actionable representation

To test this, we need to show the MESSAGE CONTENT correlates with:
- Which landmark agent i is covering (its "role")
- How agent j responds upon receiving the message

**Method: Linear probing**
1. Collect (message, agent_state, landmark_assignment) tuples from eval episodes
2. Train linear classifier: message → landmark_id
3. High accuracy = message encodes landmark assignment (content of meme)

If messages are informative about landmark assignments, we have direct evidence
that agents are transmitting their role ("meme") via communication.

```python
# Script: scripts/meme_content_probe.py
# 1. Run full_gated policy for N episodes, collect m_bar and landmark_assignments
# 2. Train logistic regression: m_bar → argmin_l(dist_to_landmark_l)
# 3. Report: how well does the message predict the sender's assigned landmark?
```

---

## Experiment 5: Long Training Convergence (800k steps)

**Motivation**: All variants plateau or diverge by 400k steps. What if we train longer?
- With LR schedule (5e-4 → 5e-5 over 800k), training should remain stable
- Memory_only might need more time to develop stable roles
- Full_gated might need more comm rounds to develop cooperative comm protocols

**Design**: Run top-3 seeds for each variant × N at 800k steps.
Select top-3 based on 400k results (exclude catastrophic failures).

---

## Priority Queue

1. **Immediate**: Run curr_obs_radius experiment (Fix A) to eliminate bimodal problem
   - 3 variants × N=3 × 4 seeds × 400k steps (18 runs)
   - Expected ~20-30 minutes

2. **After spread N-scaling completes**: Run comm_probe on full_gated checkpoints
   - Expected 10-20 minutes per N

3. **After comm_probe**: Run meme_content_probe for linear probing
   - Expected 5-10 minutes

4. **Design**: Add role_token option to run.py and agent_network.py
   - ~2-3 hours implementation + 30 min training

5. **Design**: Implement curriculum obs_radius in run.py and mpe_wrapper.py
   - ~1 hour implementation

---

## Success Criteria for Scalable Memetics

**Minimum viable result (supporting H)**:
- memory_only advantage grows from N=3 to N=5 or N=8
- OR: meme diversity grows with N in spread
- OR: comm_causal_effect grows with N

**Strong result (confirming H)**:
- All three of the above hold
- Linear probing shows message → landmark assignment with >70% accuracy

**Publication-quality result**:
- All of above + curriculum obs_radius eliminates bimodal distribution
- Role tokens show specialization 3-5x faster than no tokens
- Meme transmission probe shows direct causal path: h_i → m̄_i → behavior_j
