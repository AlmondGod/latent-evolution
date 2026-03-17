#!/bin/bash
# 3 new variants: comm_only, memory_persistent, commnet_persistent
# Same settings as partial obs suite: obs_radius=0.5, param_eq=True, N=3, 5 seeds, 400k steps
cd /Users/almondgod/Repositories/memeplex-capstone
export PYTHONPATH="$(pwd):$PYTHONPATH"

PY=/opt/homebrew/bin/python3.9
SAVE_DIR="new/memetic_foundation/checkpoints/new_variants_n3"
mkdir -p "$SAVE_DIR"

BASE_FLAGS="--env mpe --mpe-scenario simple_spread_v2 --total-steps 400000 --n-adversaries 3 --obs-radius 0.5 --obs-radius-curriculum"

run_variant() {
    local VARIANT=$1
    local FLAGS=$2
    echo ">>> Variant: ${VARIANT}"
    for SEED in 1 2 3 4 5; do
        LOG="${SAVE_DIR}/${VARIANT}_seed${SEED}.log"
        $PY -m new.memetic_foundation.run --mode train ${BASE_FLAGS} \
            --seed ${SEED} --save-dir "${SAVE_DIR}/${VARIANT}_seed${SEED}" \
            ${FLAGS} > "${LOG}" 2>&1 &
        echo "  Launched ${VARIANT} seed${SEED} (PID $!)"
    done
}

run_variant "comm_only"           "--no-memory --comm-mode commnet"
run_variant "mem_persistent"      "--no-comm   --comm-mode commnet --persistent-memory"
run_variant "commnet_persistent"  "--comm-mode commnet --persistent-memory"

echo ""
echo "All 15 runs launched. Logs: $SAVE_DIR/"
wait
echo "Suite complete."
