#!/bin/bash
# run_tag_nscale_suite.sh
#
# Simple_tag N-scaling: 4 memetic variants × N={3,6,9,12,15} × 5 seeds = 100 runs
#
# Policy controls N predators chasing 1 heuristic prey (2 obstacles).
# Goal: does memory/comm advantage change with N in an adversarial coordination task?
#
# Variants (same 4 as spread N-scaling):
#   memory_only          — GRU episodic, no comm
#   memory_only_persistent — GRU cross-episode, no comm
#   commnet_persistent   — CommNet + cross-episode GRU
#   commnet              — CommNet episodic (no persistence) — expected to collapse at N≥6
#
# 5 seeds run in parallel per (variant × N) cell.
# CPU-only: --cpu flag (MPS dispatch overhead hurts small models).

export PYTHONPATH="$(pwd):$PYTHONPATH"

ENV="mpe"
SCENARIO="simple_tag_v2"
TOTAL_STEPS=400000
SAVE_DIR="checkpoints/tag_nscale"
mkdir -p "$SAVE_DIR"

echo "=================================================="
echo "Simple Tag N-Scaling Suite"
echo "=================================================="
echo "Variants: memory_only, memory_only_persistent, commnet_persistent, commnet"
echo "N:        3, 6, 9, 12, 15"
echo "Seeds:    5 per cell | Steps: ${TOTAL_STEPS}"
echo "Total:    100 runs"
echo "--------------------------------------------------"

run_cell() {
    local VARIANT=$1
    local N=$2
    local LABEL="${VARIANT}_n${N}"
    echo ""
    echo ">>> ${LABEL} (N=${N} predators)"

    for SEED in 1 2 3 4 5; do
        LOG="$SAVE_DIR/${LABEL}_seed${SEED}.log"
        RUN_DIR="$SAVE_DIR/${LABEL}_seed${SEED}"

        case "$VARIANT" in
            memory_only)
                python3.9 -m new.memetic_foundation.run --mode train \
                    --env "$ENV" --mpe-scenario "$SCENARIO" \
                    --total-steps "$TOTAL_STEPS" --seed "$SEED" \
                    --n-adversaries "$N" \
                    --no-comm \
                    --cpu \
                    --save-dir "$RUN_DIR" > "$LOG" 2>&1 &
                ;;
            memory_only_persistent)
                python3.9 -m new.memetic_foundation.run --mode train \
                    --env "$ENV" --mpe-scenario "$SCENARIO" \
                    --total-steps "$TOTAL_STEPS" --seed "$SEED" \
                    --n-adversaries "$N" \
                    --no-comm --persistent-memory \
                    --cpu \
                    --save-dir "$RUN_DIR" > "$LOG" 2>&1 &
                ;;
            commnet_persistent)
                python3.9 -m new.memetic_foundation.run --mode train \
                    --env "$ENV" --mpe-scenario "$SCENARIO" \
                    --total-steps "$TOTAL_STEPS" --seed "$SEED" \
                    --n-adversaries "$N" \
                    --comm-mode commnet --persistent-memory \
                    --cpu \
                    --save-dir "$RUN_DIR" > "$LOG" 2>&1 &
                ;;
            commnet)
                python3.9 -m new.memetic_foundation.run --mode train \
                    --env "$ENV" --mpe-scenario "$SCENARIO" \
                    --total-steps "$TOTAL_STEPS" --seed "$SEED" \
                    --n-adversaries "$N" \
                    --comm-mode commnet \
                    --cpu \
                    --save-dir "$RUN_DIR" > "$LOG" 2>&1 &
                ;;
        esac
    done
    wait
    echo "  Done: ${LABEL}"
}

for N in 3 6 9 12 15; do
    for VARIANT in memory_only memory_only_persistent commnet_persistent commnet; do
        run_cell "$VARIANT" "$N"
    done
done

echo ""
echo "=================================================="
echo "All 100 runs complete. Results in: ${SAVE_DIR}"
echo "=================================================="
