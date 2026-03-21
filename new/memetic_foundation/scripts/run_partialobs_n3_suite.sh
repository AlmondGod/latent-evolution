#!/bin/bash
# N=3, partial obs (radius=0.5), param_eq=True, 5 seeds, 400k steps
cd /Users/almondgod/Repositories/memeplex-capstone
export PYTHONPATH="$(pwd):$PYTHONPATH"

ENV="mpe"
SCENARIO="simple_spread_v2"
TOTAL_STEPS=400000
N_AGENTS=3
OBS_RADIUS=0.5
SAVE_DIR="new/memetic_foundation/checkpoints/partialobs_n3"
PY=/opt/homebrew/bin/python3.9

mkdir -p "$SAVE_DIR"

run_variant() {
    local VARIANT=$1
    local FLAGS=$2
    echo ">>> Variant: ${VARIANT}"
    for SEED in 1 2 3 4 5; do
        LOG_FILE="${SAVE_DIR}/${VARIANT}_seed${SEED}.log"
        $PY -m new.memetic_foundation.run --mode train \
            --env ${ENV} --mpe-scenario ${SCENARIO} \
            --total-steps ${TOTAL_STEPS} --seed ${SEED} \
            --n-adversaries ${N_AGENTS} \
            --obs-radius ${OBS_RADIUS} \
            --obs-radius-curriculum \
            --save-dir "${SAVE_DIR}/${VARIANT}_seed${SEED}" \
            ${FLAGS} \
            > "${LOG_FILE}" 2>&1 &
        echo "  Launched ${VARIANT} seed${SEED} (PID $!)"
    done
}

run_variant "baseline"     "--no-memory --no-comm --comm-mode commnet"
run_variant "memory_only"  "--no-comm --comm-mode commnet"
run_variant "commnet"      "--comm-mode commnet"
run_variant "commnet_sep"  "--comm-mode commnet_sep"

echo ""
echo "All 20 partial-obs runs launched. Logs: $SAVE_DIR/"
wait
echo "Partial-obs suite complete."
