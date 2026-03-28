# MeMEtIC Phase-2 Design

## Goal

Part 1 established that standard differentiable multi-agent communication can
show memetic-like properties:

- persistent latent structure
- causal transmission through communication
- optimization-mediated reinforcement of useful latent regimes

What Part 1 did **not** provide was explicit selection over heritable
communication variants. Part 2 adds that outer-loop selection.

## High-Level Method

`MeMEtIC` is a two-timescale method:

1. **Phase 1: Backbone learning**
   Train a competent multi-agent policy with MAPPO.
   The backbone includes:
   - observation encoder
   - GRU task memory `h`
   - actor/critic trunk

2. **Phase 2: Online memetic evolution**
   Freeze most of the backbone and evolve a small module that controls:
   - communication transmission
   - a persistent memetic state `z`

This gives a cleaner split:

- `h` = slow task competence
- `z` = fast socially-shaped latent culture
- evolvable adapter params = transmissibility rules

## Why Not Evolve the Whole Network?

Evolving the full policy is expensive and muddles the scientific question. If
everything changes, then "selection" is just another form of noisy policy
search.

Instead, we evolve only a compact substrate on top of a learned backbone:

- low-rank communication adapters
- memetic-state update rule
- optional communication scale or routing temperature

This makes the outer loop much closer to explicit selection over
communication/memetic mechanisms rather than full-agent re-learning.

## Memetic State

The key addition is a persistent latent `z_i^t` for each agent. Unlike the base
GRU memory `h_i^t`, which is optimized during Phase 1 for general task
competence, `z_i^t` is meant to carry socially-shaped, higher-variance cultural
state in Phase 2.

Suggested dynamics:

- `q_i^t = W_q [h_i^t ; z_i^t]`
- `c_i^t = Attn(q_i^t, K(u^t), V(u^t))`
- `z_i^{t+1} = (1 - eta) z_i^t + eta * tanh(W_z [z_i^t ; c_i^t ; h_i^t])`

Actor input can then use:

- `[h_i^t ; z_i^t ; c_i^t]`

The current scaffold for this lives in:

- [memetic_adapter.py](/Users/almondgod/Repositories/memeplex-capstone/new/memetic_foundation/modules/memetic_adapter.py)

## What Gets Evolved

Recommended genotype:

- low-rank deltas on communication projections `Q/K/V/O`
- memetic-state update parameters `W_z`
- scalar communication scale
- optional routing sparsity/temperature

Not recommended:

- full GRU weights
- full actor or critic
- the entire encoder

Those larger edits are more likely to destabilize the backbone than to reveal
clean memetic selection effects.

## Optimizer Choice

### Primary recommendation: OpenAI-ES over the adapter genotype

Use ES when:

- the genotype is moderate in size
- fitness is noisy
- we want a simple post-MAPPO outer loop

The initial scaffold is:

- [openai_es.py](/Users/almondgod/Repositories/memeplex-capstone/new/memetic_foundation/training/openai_es.py)

### CMA-ES

Only use CMA-ES if the genotype is compressed to a small latent code
(`16-64` dims). CMA-ES is not a good default for thousands of raw adapter
parameters.

### PBT

Use PBT if Phase 2 should still contain gradient updates. That gives a more
explicit Lamarckian story:

- agents learn within a generation
- elites reproduce across generations

## Fitness

Reward alone is not enough. If fitness is only task return, evolution can learn
to ignore communication entirely.

Recommended fitness:

`F = R + lambda * G + mu * P - beta * C`

Where:

- `R`: task reward
- `G`: causal communication gain
  - e.g. baseline reward minus silence/shift reward
- `P`: persistence score on `z`
  - dwell time, self-transition, low transition entropy
- `C`: message cost or instability penalty

The smallest viable version is:

`F = R + lambda * G`

## Minimal Experiment Ladder

### Step 1

Freeze a pretrained `attention_hu_actor` backbone and evolve only the low-rank
communication adapter.

This tests whether explicit selection can improve communication without touching
the core task policy.

### Step 2

Add persistent memetic state `z` and evolve both:

- communication adapter
- `z` update

This is the first genuinely memetic Phase-2 variant.

### Step 3

If needed, compress the genotype and try CMA-ES.

## Evaluation

Phase 2 should be judged on three axes:

1. **Performance**
   - reward
   - success / win rate
   - `min_dist` where applicable

2. **Communication utility**
   - silence intervention drop
   - shift intervention drop

3. **Memetic properties**
   measured on `z`, not just on `h`
   - self-transition probability
   - dwell time
   - transition entropy
   - latent clustering quality

## Current Recommendation

The best near-term implementation path is:

1. start from pretrained `attention_hu_actor`
2. freeze the backbone
3. add `z in R^16`
4. add low-rank communication adapters
5. evolve the adapter with OpenAI-ES first
6. score with reward plus causal communication gain

That is the lowest-risk way to make Part 2 concrete without throwing away the
stable gains from Part 1.
