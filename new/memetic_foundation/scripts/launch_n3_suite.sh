#!/bin/bash
# Properly launches 16 runs with explicit args (no variable expansion issues)
cd /Users/almondgod/Repositories/memeplex-capstone
export PYTHONPATH="$(pwd):$PYTHONPATH"

PY="/opt/homebrew/bin/python3.9"
SAVE="new/memetic_foundation/checkpoints/commnet_n3"
mkdir -p "$SAVE"

for S in 1 2 3 4; do
    $PY -m new.memetic_foundation.run \
        --mode train --env mpe --mpe-scenario simple_spread_v2 \
        --total-steps 400000 --n-adversaries 3 --no-param-eq \
        --seed $S --no-memory --no-comm \
        --save-dir "${SAVE}/baseline_seed${S}" \
        > "${SAVE}/baseline_seed${S}.log" 2>&1 &
    echo "baseline_seed${S} PID=$!"
done

for S in 1 2 3 4; do
    $PY -m new.memetic_foundation.run \
        --mode train --env mpe --mpe-scenario simple_spread_v2 \
        --total-steps 400000 --n-adversaries 3 --no-param-eq \
        --seed $S --no-comm \
        --save-dir "${SAVE}/memory_only_seed${S}" \
        > "${SAVE}/memory_only_seed${S}.log" 2>&1 &
    echo "memory_only_seed${S} PID=$!"
done

for S in 1 2 3 4; do
    $PY -m new.memetic_foundation.run \
        --mode train --env mpe --mpe-scenario simple_spread_v2 \
        --total-steps 400000 --n-adversaries 3 --no-param-eq \
        --seed $S --comm-mode commnet \
        --save-dir "${SAVE}/commnet_seed${S}" \
        > "${SAVE}/commnet_seed${S}.log" 2>&1 &
    echo "commnet_seed${S} PID=$!"
done

for S in 1 2 3 4; do
    $PY -m new.memetic_foundation.run \
        --mode train --env mpe --mpe-scenario simple_spread_v2 \
        --total-steps 400000 --n-adversaries 3 --no-param-eq \
        --seed $S --comm-mode commnet_sep \
        --save-dir "${SAVE}/commnet_sep_seed${S}" \
        > "${SAVE}/commnet_sep_seed${S}.log" 2>&1 &
    echo "commnet_sep_seed${S} PID=$!"
done

echo "All 16 launched"
wait
echo "All 16 done"
