# Autonomous Experiment Plan: Scalable Memetics Between Agents
## Scientific Question
Can we demonstrate that *patterns in messages become self-sustaining and propagating*
through agent populations via persistent memory — the core memetic claim?

## Experiment 1: Memory Content Diversity Analysis (post-gating)
**Question:** After gating, do agents develop *different* memory states (specialization),
or do they converge to similar states (homogenization)?
**Why it matters:** True memetics requires differentiated memory — agent A carries
"meme X", agent B carries "meme Y". If all memories collapse to the same vector,
there's no information propagation, just shared averaging.
**Method:** After training, run 5 eval episodes. At each step, compute pairwise
cosine similarity between all agent hidden states. Track over episode.
- High similarity early → homogenization (comm is forcing convergence)
- Low similarity, stable → specialization (agents develop roles)
- Low early, high late → convergence through experience

## Experiment 2: Message Persistence / Meme Lifespan Test
**Question:** How long does a specific communication pattern persist in an agent's
memory after the sender has gone silent (or after the originating observation disappears)?
**Method:** 
1. Train full/gated variant to convergence
2. Run eval episode, record h_t at each step
3. At step T_silence: zero out all comm (intervene_comm_silence=True)
4. Measure: how many steps until h decays to baseline (||h_t - h_0|| < ε)?
**Prediction:** With mem_decay=0.005 and h=GRU, the decay is ~200 steps to
reach 1/e of original magnitude. But *learned* memory should decay slower
for "important" patterns (GRU gates will resist overwriting salient state).

## Experiment 3: N-Agent Scaling (3→5→8→12 predators on simple_tag variants)
**Question:** Does the advantage of memory+comm grow with N?
**Hypothesis:** At N=3, individual agents can observe the prey directly.
At N=8, agents at the periphery cannot directly coordinate with all others —
memory of received messages becomes load-bearing.
**Method:** Create simple_tag variants with N=3,5,8 adversaries. Run 4-way
ablation on each. If the gap (full - baseline) grows with N, this is direct
evidence of scalable memetics.
**Implementation:** MPEWrapper already supports num_adversaries parameter.

## Experiment 4: Targeted Message Tracing (which agent influences whom)
**Question:** Does the comm attention matrix (who attends to whom) form stable
patterns? Do "leader" agents (high out-degree in attn) emerge?
**Method:** Log comm_attn matrices during eval episodes across all steps.
Compute per-agent: mean attention received (influence) vs sent (informativeness).
Visualize as directed influence graph. Look for:
- Consistent asymmetry → role specialization
- Uniform attention → no meaningful communication
- Context-dependent patterns → adaptive communication

## Experiment 5: Cross-Episode Memory Persistence Test (the key memetic claim)
**Question:** Does memory from episode N affect behavior in episode N+1 in a 
useful way, or is cross-episode state just noise?
**Method:** 
1. Train full variant (memory persists across episodes by design)
2. Run two-episode sequences: episode 1 with normal prey, episode 2 with
   same prey but different start positions
3. Compare: performance with persistent memory vs reset memory at ep boundary
4. If persistent > reset: cross-episode memes are genuinely informative
**This is the most direct test of the memetic hypothesis.**
