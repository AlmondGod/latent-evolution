#!/bin/bash
cd /Users/almondgod/Repositories/memeplex-capstone
export PYTHONPATH="$(pwd):$PYTHONPATH"
PY=/opt/homebrew/bin/python3.9
SAVE_DIR="new/memetic_foundation/checkpoints/meme_analysis_n10"
mkdir -p "$SAVE_DIR"

echo "Launching memory_only (hid=83, 124k)..."
for S in 1 2 3; do
    $PY -m new.memetic_foundation.run \
        --mode train --env mpe --mpe-scenario simple_spread_v2 \
        --n-adversaries 10 --obs-radius 0.5 --obs-curriculum-steps 200000 \
        --total-steps 1000000 --no-param-eq --ent-coef 0.3 \
        --probe-interval 50000 --probe-episodes 10 \
        --hidden-dim 83 --seed $S \
        --save-dir "${SAVE_DIR}/memory_only_seed${S}" \
        --no-comm --comm-mode commnet \
        > "${SAVE_DIR}/memory_only_seed${S}.log" 2>&1 &
done
wait; echo "  memory_only done"

echo "Launching commnet_persistent (hid=74, 124k)..."
for S in 1 2 3; do
    $PY -m new.memetic_foundation.run \
        --mode train --env mpe --mpe-scenario simple_spread_v2 \
        --n-adversaries 10 --obs-radius 0.5 --obs-curriculum-steps 200000 \
        --total-steps 1000000 --no-param-eq --ent-coef 0.3 \
        --probe-interval 50000 --probe-episodes 10 \
        --hidden-dim 74 --seed $S \
        --save-dir "${SAVE_DIR}/commnet_persistent_seed${S}" \
        --comm-mode commnet --persistent-memory \
        > "${SAVE_DIR}/commnet_persistent_seed${S}.log" 2>&1 &
done
wait; echo "  commnet_persistent done"

echo "Launching memory_only_persistent (hid=83, 124k)..."
for S in 1 2 3; do
    $PY -m new.memetic_foundation.run \
        --mode train --env mpe --mpe-scenario simple_spread_v2 \
        --n-adversaries 10 --obs-radius 0.5 --obs-curriculum-steps 200000 \
        --total-steps 1000000 --no-param-eq --ent-coef 0.3 \
        --probe-interval 50000 --probe-episodes 10 \
        --hidden-dim 83 --seed $S \
        --save-dir "${SAVE_DIR}/memory_only_persistent_seed${S}" \
        --no-comm --comm-mode commnet --persistent-memory \
        > "${SAVE_DIR}/memory_only_persistent_seed${S}.log" 2>&1 &
done
wait; echo "  memory_only_persistent done"

echo "ALL DONE"
