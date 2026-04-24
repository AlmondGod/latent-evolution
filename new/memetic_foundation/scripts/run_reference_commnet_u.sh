#!/bin/bash
# Run commnet_u on simple_reference_v2 — 5 seeds, all parallel.
# Compare against reference_suite results.

export PYTHONPATH="$(pwd):$PYTHONPATH"

SAVE_DIR="checkpoints/reference_commnet_u"
mkdir -p "$SAVE_DIR"

echo "Launching commnet_u × 5 seeds on simple_reference_v2..."

for SEED in 1 2 3 4 5; do
    python3.9 -m new.memetic_foundation.run --mode train \
        --env mpe --mpe-scenario simple_reference_v2 \
        --total-steps 400000 --seed "$SEED" \
        --comm-mode commnet_u \
        --cpu \
        --save-dir "${SAVE_DIR}/commnet_u_seed${SEED}" \
        > "${SAVE_DIR}/commnet_u_seed${SEED}.log" 2>&1 &
done

wait
echo "Done. Results in ${SAVE_DIR}"
