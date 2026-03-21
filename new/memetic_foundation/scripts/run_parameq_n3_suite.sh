#!/bin/bash
# N=3, param_eq=True (default), 5 seeds, 400k steps — fair comparison
cd /Users/almondgod/Repositories/memeplex-capstone
export PYTHONPATH="$(pwd):$PYTHONPATH"

ENV="mpe"
SCENARIO="simple_spread_v2"
TOTAL_STEPS=400000
N_AGENTS=3
SAVE_DIR="new/memetic_foundation/checkpoints/parameq_n3"
PY=/opt/homebrew/bin/python3.9

mkdir -p "$SAVE_DIR"

run_variant() {
    local VARIANT=$1
    local FLAGS=$2
    echo ">>> Variant: ${VARIANT}"
    for SEED in 1 2 3 4 5; do
        RUN_DIR="${SAVE_DIR}/${VARIANT}_seed${SEED}"
        LOG_FILE="${SAVE_DIR}/${VARIANT}_seed${SEED}.log"
        $PY -m new.memetic_foundation.run --mode train \
            --env ${ENV} --mpe-scenario ${SCENARIO} \
            --total-steps ${TOTAL_STEPS} --seed ${SEED} \
            --n-adversaries ${N_AGENTS} \
            --save-dir "${RUN_DIR}" \
            ${FLAGS} \
            > "${LOG_FILE}" 2>&1 &
        echo "  Launched ${VARIANT} seed${SEED} (PID $!)"
    done
}

# Flags: default is memory+comm ON, so use --no-memory/--no-comm to disable
# param_eq=True is default (no --no-param-eq flag)
run_variant "baseline"     "--no-memory --no-comm --comm-mode commnet"
run_variant "memory_only"  "--no-comm --comm-mode commnet"
run_variant "commnet"      "--comm-mode commnet"
run_variant "commnet_sep"  "--comm-mode commnet_sep"

echo ""
echo "All 20 runs launched. Logs: $SAVE_DIR/"
wait
echo "Suite complete."
