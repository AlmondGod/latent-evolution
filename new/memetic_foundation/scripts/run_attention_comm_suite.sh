#!/bin/bash
# run_attention_comm_suite.sh
#
# Compares two attention-based communication architectures against baselines:
#
#   attention_integrated  — GRU gets [obs; m̄_prev]:
#                           social info merges INTO personal memory each step.
#                           h encodes both self-experience and received memes.
#
#   attention_separated   — GRU gets only obs; actor gets [u; h; c]:
#                           h is a pure individual meme; c is social context.
#                           Personal and social memory interact only at the actor.
#
# Both use soft attention (no IC3Net gate, no gate entropy loss).
# The scientific question: does routing comm INTO the GRU (so social info
# shapes the meme directly) beat keeping them separate (so personal meme
# stays "pure" and social info is a separate input)?
#
# Comparators included:
#   baseline     — no memory, no comm
#   memory_only  — GRU, no comm  (best from prior experiments)
#   ic3net       — current full_gated (binary gate, comm→GRU)
#
# Design: 5 variants × N={3,5,8} × 4 seeds × 400k steps = 60 runs
# Running 3 seeds in parallel per batch.
#
# Parameter note: attention_separated has ~10% fewer params (GRU input shrinks).
# This is noted but not adjusted — the architectural difference is the focus.

export PYTHONPATH="$(pwd):$PYTHONPATH"

ENV="mpe"
SCENARIO="simple_spread_v2"
TOTAL_STEPS=400000
OBS_RADIUS=0.5
SAVE_DIR="checkpoints/attention_comm_suite"
N_SEEDS=4

mkdir -p "$SAVE_DIR"

echo "============================================================"
echo "Attention Communication Architecture Comparison"
echo "============================================================"
echo "Variants:"
echo "  baseline             — no memory, no comm"
echo "  memory_only          — GRU only (best prior result)"
echo "  ic3net               — binary gate, comm→GRU (prior full_gated)"
echo "  attention_integrated — soft attention, comm→GRU (no gate entropy)"
echo "  attention_separated  — soft attention, GRU=pure obs, actor=[u;h;c]"
echo "N: 3, 5, 8 | Seeds: ${N_SEEDS} | Steps: ${TOTAL_STEPS}"
echo "Obs radius: ${OBS_RADIUS} (partial observability)"
echo "------------------------------------------------------------"

run_batch() {
    local LABEL=$1
    local FLAGS=$2
    local N_AGENTS=$3
    local SEED_START=$4
    local SEED_END=$5

    for SEED in $(seq $SEED_START $SEED_END); do
        LOG="$SAVE_DIR/${LABEL}_seed${SEED}.log"
        python3.9 -m new.memetic_foundation.run --mode train \
            --env "$ENV" \
            --mpe-scenario "$SCENARIO" \
            --total-steps "$TOTAL_STEPS" \
            --seed "$SEED" \
            --n-adversaries "$N_AGENTS" \
            --obs-radius "$OBS_RADIUS" \
            --save-dir "$SAVE_DIR/${LABEL}_seed${SEED}" \
            $FLAGS > "$LOG" 2>&1 &
    done
    wait
}

HALF=$((N_SEEDS / 2))
SECOND=$((HALF + 1))

for N_AGENTS in 3 5 8; do
    echo ""
    echo "=== N=${N_AGENTS} ==="

    echo "  baseline_n${N_AGENTS}"
    run_batch "baseline_n${N_AGENTS}"             "--no-memory --no-comm"          $N_AGENTS 1 $HALF
    run_batch "baseline_n${N_AGENTS}"             "--no-memory --no-comm"          $N_AGENTS $SECOND $N_SEEDS

    echo "  memory_only_n${N_AGENTS}"
    run_batch "memory_only_n${N_AGENTS}"          "--no-comm"                      $N_AGENTS 1 $HALF
    run_batch "memory_only_n${N_AGENTS}"          "--no-comm"                      $N_AGENTS $SECOND $N_SEEDS

    echo "  ic3net_n${N_AGENTS}"
    run_batch "ic3net_n${N_AGENTS}"              "--comm-mode ic3net"              $N_AGENTS 1 $HALF
    run_batch "ic3net_n${N_AGENTS}"              "--comm-mode ic3net"              $N_AGENTS $SECOND $N_SEEDS

    echo "  attention_integrated_n${N_AGENTS}"
    run_batch "attn_integrated_n${N_AGENTS}"     "--comm-mode attention_integrated" $N_AGENTS 1 $HALF
    run_batch "attn_integrated_n${N_AGENTS}"     "--comm-mode attention_integrated" $N_AGENTS $SECOND $N_SEEDS

    echo "  attention_separated_n${N_AGENTS}"
    run_batch "attn_separated_n${N_AGENTS}"      "--comm-mode attention_separated"  $N_AGENTS 1 $HALF
    run_batch "attn_separated_n${N_AGENTS}"      "--comm-mode attention_separated"  $N_AGENTS $SECOND $N_SEEDS
done

echo ""
echo "============================================================"
echo "All runs complete. Results in: ${SAVE_DIR}"
echo ""
echo "Key questions:"
echo "  1. Does attention_integrated beat memory_only? (comm benefit with no gate instability)"
echo "  2. Does attention_separated beat attention_integrated? (pure personal meme + social ctx)"
echo "  3. Does either attention variant beat baseline? (full arch vs no arch)"
echo "  4. Does catastrophic failure rate change with attention vs ic3net?"
echo "============================================================"
