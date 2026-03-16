#!/bin/bash
# run_simple_spread_suite.sh
#
# 32 runs: 4 ablation variants * 8 random seeds * 200k steps on MPE Simple Spread
# Runs 4 seeds in parallel (2 batches of 4) per variant to avoid CPU contention.

export PYTHONPATH="$(pwd):$PYTHONPATH"

ENV="mpe"
SCENARIO="simple_spread_v2"
TOTAL_STEPS=200000

SAVE_DIR="checkpoints/mpe_spread_ablations"
mkdir -p "$SAVE_DIR"

echo "=================================================="
echo "Memetic Foundation MPE Simple Spread Suite"
echo "=================================================="
echo "Total Runs: 32 (4 variants * 8 seeds)"
echo "Steps per Run: ${TOTAL_STEPS}"
echo "--------------------------------------------------"

run_variant() {
    local VARIANT=$1
    local FLAGS=$2
    echo ""
    echo ">>> Variant: ${VARIANT}"

    # Batch 1: seeds 1-4
    for SEED in 1 2 3 4; do
        RUN_DIR="${SAVE_DIR}/${VARIANT}_seed${SEED}"
        LOG_FILE="${SAVE_DIR}/${VARIANT}_seed${SEED}.log"
        python3.9 -m new.memetic_foundation.run --mode train \
            --env ${ENV} --mpe-scenario ${SCENARIO} \
            --total-steps ${TOTAL_STEPS} --seed ${SEED} \
            --save-dir "${RUN_DIR}" \
            ${FLAGS} \
            > "${LOG_FILE}" 2>&1 &
    done
    echo "  Batch 1 (seeds 1-4) launched..."
    wait

    # Batch 2: seeds 5-8
    for SEED in 5 6 7 8; do
        RUN_DIR="${SAVE_DIR}/${VARIANT}_seed${SEED}"
        LOG_FILE="${SAVE_DIR}/${VARIANT}_seed${SEED}.log"
        python3.9 -m new.memetic_foundation.run --mode train \
            --env ${ENV} --mpe-scenario ${SCENARIO} \
            --total-steps ${TOTAL_STEPS} --seed ${SEED} \
            --save-dir "${RUN_DIR}" \
            ${FLAGS} \
            > "${LOG_FILE}" 2>&1 &
    done
    echo "  Batch 2 (seeds 5-8) launched..."
    wait

    echo "  Done: ${VARIANT}"
    echo "--------------------------------------------------"
}

run_variant "baseline"     "--no-memory --no-comm"
run_variant "memory_only"  "--no-comm"
run_variant "comm_only"    "--no-memory"
run_variant "full"         ""

echo "=================================================="
echo "All 32 Simple Spread Ablation Runs Complete!"
echo "=================================================="
