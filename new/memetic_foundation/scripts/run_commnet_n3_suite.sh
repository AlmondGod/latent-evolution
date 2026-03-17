#!/bin/bash
# run_commnet_n3_suite.sh
#
# 16 runs: 4 variants * 4 seeds * 400k steps on MPE Simple Spread (N=3 fixed)
# Variants: baseline, memory_only, commnet, commnet_sep
# param_eq=False (all variants use enc_dim=hidden_dim=128, no widening)
# Zero-init comm weights so commnet/commnet_sep start identical to memory_only.

cd /Users/almondgod/Repositories/memeplex-capstone
export PYTHONPATH="$(pwd):$PYTHONPATH"

ENV="mpe"
SCENARIO="simple_spread_v2"
TOTAL_STEPS=400000
N_AGENTS=3  # default for simple_spread_v2 via --n-adversaries

SAVE_DIR="new/memetic_foundation/checkpoints/commnet_n3"
mkdir -p "$SAVE_DIR"

echo "=================================================="
echo "CommNet N=3 Ablation Suite"
echo "=================================================="
echo "Variants: baseline, memory_only, commnet, commnet_sep"
echo "Seeds: 1-4 | Steps: ${TOTAL_STEPS} | N=${N_AGENTS}"
echo "--------------------------------------------------"

run_variant() {
    local VARIANT=$1
    local FLAGS=$2
    echo ""
    echo ">>> Variant: ${VARIANT}"

    for SEED in 1 2 3 4; do
        RUN_DIR="${SAVE_DIR}/${VARIANT}_seed${SEED}"
        LOG_FILE="${SAVE_DIR}/${VARIANT}_seed${SEED}.log"
        /opt/homebrew/bin/python3.9 -m new.memetic_foundation.run --mode train \
            --env ${ENV} --mpe-scenario ${SCENARIO} \
            --total-steps ${TOTAL_STEPS} --seed ${SEED} \
            --n-adversaries ${N_AGENTS} \
            --save-dir "${RUN_DIR}" \
            --no-param-eq \
            ${FLAGS} \
            > "${LOG_FILE}" 2>&1 &
        echo "  Launched seed ${SEED} -> ${LOG_FILE}"
    done
    echo "  Waiting for all 4 seeds..."
    wait
    echo "  Done: ${VARIANT}"
    echo "--------------------------------------------------"
}

run_variant "baseline"     "--no-memory --no-comm"
run_variant "memory_only"  "--no-comm"
run_variant "commnet"      "--comm-mode commnet"
run_variant "commnet_sep"  "--comm-mode commnet_sep"

echo "=================================================="
echo "All 16 CommNet N=3 Runs Complete!"
echo "=================================================="
