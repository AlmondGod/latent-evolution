#!/bin/bash
# run_reference_suite.sh
#
# simple_reference_v2 sanity check for CommNet.
# 4 variants × 5 seeds = 20 runs. Fixed 2-agent task.
# Checks: does CommNet (with/without persistence) learn better coordination
# than no-comm baselines?
#
# All 20 seeds run in parallel.

export PYTHONPATH="$(pwd):$PYTHONPATH"

ENV="mpe"
SCENARIO="simple_reference_v2"
TOTAL_STEPS=400000
SAVE_DIR="checkpoints/reference_suite"
mkdir -p "$SAVE_DIR"

echo "=================================================="
echo "Simple Reference v2 — CommNet Sanity Check"
echo "=================================================="
echo "Variants: memory_only, memory_only_persistent, commnet_persistent, commnet"
echo "Seeds: 5 | Steps: ${TOTAL_STEPS}"
echo "Total: 20 runs"
echo "--------------------------------------------------"

for SEED in 1 2 3 4 5; do

    # memory_only
    python3.9 -m new.memetic_foundation.run --mode train \
        --env "$ENV" --mpe-scenario "$SCENARIO" \
        --total-steps "$TOTAL_STEPS" --seed "$SEED" \
        --no-comm --cpu \
        --save-dir "${SAVE_DIR}/memory_only_seed${SEED}" \
        > "${SAVE_DIR}/memory_only_seed${SEED}.log" 2>&1 &

    # memory_only_persistent
    python3.9 -m new.memetic_foundation.run --mode train \
        --env "$ENV" --mpe-scenario "$SCENARIO" \
        --total-steps "$TOTAL_STEPS" --seed "$SEED" \
        --no-comm --persistent-memory --cpu \
        --save-dir "${SAVE_DIR}/memory_only_persistent_seed${SEED}" \
        > "${SAVE_DIR}/memory_only_persistent_seed${SEED}.log" 2>&1 &

    # commnet_persistent
    python3.9 -m new.memetic_foundation.run --mode train \
        --env "$ENV" --mpe-scenario "$SCENARIO" \
        --total-steps "$TOTAL_STEPS" --seed "$SEED" \
        --comm-mode commnet --persistent-memory --cpu \
        --save-dir "${SAVE_DIR}/commnet_persistent_seed${SEED}" \
        > "${SAVE_DIR}/commnet_persistent_seed${SEED}.log" 2>&1 &

    # commnet (no persistence)
    python3.9 -m new.memetic_foundation.run --mode train \
        --env "$ENV" --mpe-scenario "$SCENARIO" \
        --total-steps "$TOTAL_STEPS" --seed "$SEED" \
        --comm-mode commnet --cpu \
        --save-dir "${SAVE_DIR}/commnet_seed${SEED}" \
        > "${SAVE_DIR}/commnet_seed${SEED}.log" 2>&1 &

done

echo "Launched 20 reference suite jobs — waiting..."
wait
echo ""
echo "=================================================="
echo "All 20 runs complete. Results in: ${SAVE_DIR}"
echo "=================================================="
