#!/bin/bash
# run_1m_convergence.sh — 3 seeds x 4 variants x 1M steps (convergence study)
export PYTHONPATH="$(pwd):$PYTHONPATH"

ENV="mpe"
SCENARIO="simple_spread_v2"
TOTAL_STEPS=1000000
SAVE_DIR="checkpoints/mpe_spread_1m"
mkdir -p "$SAVE_DIR"

echo "=================================================="
echo "Memetic Foundation 1M-step Convergence Study"
echo "=================================================="
echo "Seeds: 3 per variant   |   Steps: ${TOTAL_STEPS}"
echo "--------------------------------------------------"

launch() {
    local LABEL=$1
    local FLAGS=$2
    for SEED in 1 2 3; do
        LOG="$SAVE_DIR/${LABEL}_seed${SEED}.log"
        python3.9 -m new.memetic_foundation.run --mode train \
            --env "$ENV" \
            --mpe-scenario "$SCENARIO" \
            --total-steps "$TOTAL_STEPS" \
            --seed "$SEED" \
            --save-dir "$SAVE_DIR/${LABEL}_seed${SEED}" \
            $FLAGS \
            > "$LOG" 2>&1 &
        echo "  Launched ${LABEL} seed${SEED} (PID $!)"
    done
}

launch "baseline"    "--no-memory --no-comm"
launch "memory_only" "--no-comm"
launch "comm_only"   "--no-memory"
launch "full"        ""

echo ""
echo "All 12 runs launched. Waiting..."
wait
echo "=================================================="
echo "All done. Results in: $SAVE_DIR"
echo "=================================================="
