#!/bin/bash
# run_tag_gated_suite.sh
# Reruns only comm_only and full with IC3Net gating on simple_tag.
# 8 seeds × 2 variants × 200k steps — 4 seeds in parallel per batch.

export PYTHONPATH="$(pwd):$PYTHONPATH"

ENV="mpe"
SCENARIO="simple_tag_v2"
TOTAL_STEPS=200000
SAVE_DIR="checkpoints/mpe_tag_gated"
mkdir -p "$SAVE_DIR"

echo "=================================================="
echo "Memetic Foundation Simple Tag — Gated Comm Suite"
echo "=================================================="
echo "Variants: comm_only (gated), full (gated)"
echo "Seeds: 8   |   Steps: ${TOTAL_STEPS}"
echo "--------------------------------------------------"

run_variant() {
    local LABEL=$1
    local FLAGS=$2
    echo ""
    echo ">>> Variant: ${LABEL}"
    for SEED in 1 2 3 4; do
        python3.9 -m new.memetic_foundation.run --mode train \
            --env "$ENV" --mpe-scenario "$SCENARIO" \
            --total-steps "$TOTAL_STEPS" --seed "$SEED" \
            --save-dir "$SAVE_DIR/${LABEL}_seed${SEED}" \
            $FLAGS > "$SAVE_DIR/${LABEL}_seed${SEED}.log" 2>&1 &
    done
    wait
    echo "  Batch 1 (seeds 1-4) done."
    for SEED in 5 6 7 8; do
        python3.9 -m new.memetic_foundation.run --mode train \
            --env "$ENV" --mpe-scenario "$SCENARIO" \
            --total-steps "$TOTAL_STEPS" --seed "$SEED" \
            --save-dir "$SAVE_DIR/${LABEL}_seed${SEED}" \
            $FLAGS > "$SAVE_DIR/${LABEL}_seed${SEED}.log" 2>&1 &
    done
    wait
    echo "  Batch 2 (seeds 5-8) done."
    echo "  Done: ${LABEL}"
    echo "--------------------------------------------------"
}

run_variant "comm_only_gated" "--no-memory"
run_variant "full_gated"      ""

echo ""
echo "=================================================="
echo "All 16 runs complete. Results in: ${SAVE_DIR}"
echo "=================================================="
