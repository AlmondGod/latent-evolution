#!/bin/bash
# run_mem_decay_suite.sh
#
# 16 runs: 2 variants (memory_only, full) * 8 seeds * 200k steps
# Tests memory decay stabilization on MPE Simple Spread

export PYTHONPATH="$(pwd):$PYTHONPATH"

ENV="mpe"
SCENARIO="simple_spread_v2"
TOTAL_STEPS=200000

SAVE_DIR="checkpoints/mpe_spread_decay"
mkdir -p "$SAVE_DIR"

echo "=================================================="
echo "Memory Decay Ablation Suite (mem_only + full)"
echo "=================================================="
echo "Total Runs: 16 (2 variants * 8 seeds)"
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

run_variant "memory_only"  "--no-comm"
run_variant "full"         ""

echo "=================================================="
echo "All 16 Memory Decay Runs Complete!"
echo "=================================================="
