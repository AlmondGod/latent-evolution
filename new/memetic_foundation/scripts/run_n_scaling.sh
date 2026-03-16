#!/bin/bash
# run_n_scaling.sh
# Tests whether memory+comm advantage grows with N agents.
# N=3, N=5, N=8 predators on simple_tag variants.
# 3 variants x 3 Ns x 4 seeds = 36 runs. 4 parallel per batch.
# Variants: baseline, memory_only, full_gated
# (comm_only skipped — gating without memory shown to fail)

export PYTHONPATH="$(pwd):$PYTHONPATH"

ENV="mpe"
SCENARIO="simple_tag_v2"
TOTAL_STEPS=200000
SAVE_DIR="checkpoints/mpe_tag_nscale"
mkdir -p "$SAVE_DIR"

echo "=================================================="
echo "Memetic Foundation N-Agent Scaling Study"
echo "=================================================="
echo "N: 3, 5, 8 adversaries on simple_tag"
echo "Variants: baseline, memory_only, full_gated"
echo "Seeds: 4 per cell   |   Steps: ${TOTAL_STEPS}"
echo "--------------------------------------------------"

get_flags() {
    case "$1" in
        baseline)    echo "--no-memory --no-comm" ;;
        memory_only) echo "--no-comm" ;;
        full_gated)  echo "" ;;
    esac
}

for N_ADV in 3 5 8; do
    for VARIANT in baseline memory_only full_gated; do
        FLAGS=$(get_flags "$VARIANT")
        LABEL="${VARIANT}_n${N_ADV}"
        echo ""
        echo ">>> ${LABEL} (N=${N_ADV})"
        for SEED in 1 2 3 4; do
            LOG="$SAVE_DIR/${LABEL}_seed${SEED}.log"
            python3.9 -m new.memetic_foundation.run --mode train \
                --env "$ENV" \
                --mpe-scenario "$SCENARIO" \
                --total-steps "$TOTAL_STEPS" \
                --seed "$SEED" \
                --save-dir "$SAVE_DIR/${LABEL}_seed${SEED}" \
                --n-adversaries "$N_ADV" \
                $FLAGS \
                > "$LOG" 2>&1 &
        done
        wait
        echo "  Done: ${LABEL}"
    done
done

echo ""
echo "=================================================="
echo "All 36 runs complete. Results in: ${SAVE_DIR}"
echo "=================================================="
