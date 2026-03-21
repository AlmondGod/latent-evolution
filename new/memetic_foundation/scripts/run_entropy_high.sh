#!/bin/bash
cd /Users/almondgod/Repositories/memeplex-capstone
export PYTHONPATH="$(pwd):$PYTHONPATH"

PY=/opt/homebrew/bin/python3.9
SAVE_DIR="new/memetic_foundation/checkpoints/entropy_sweep"

BASE="--env mpe --mpe-scenario simple_spread_v2 --total-steps 400000 --n-adversaries 3 --obs-radius 0.5 --obs-radius-curriculum --no-param-eq"

run_variant() {
    local NAME=$1; local HID=$2; local ENT=$3; local FLAGS=$4
    local TAG="${NAME}_ent${ENT}"
    echo ">>> $TAG"
    for SEED in $(seq 1 10); do
        LOG="${SAVE_DIR}/${TAG}_seed${SEED}.log"
        $PY -m new.memetic_foundation.run --mode train $BASE \
            --hidden-dim $HID --ent-coef $ENT --seed $SEED \
            --save-dir "${SAVE_DIR}/${TAG}_seed${SEED}" \
            $FLAGS > "$LOG" 2>&1 &
    done
    echo "  Seeds 1-10 launched"
}

for ENT in 0.3 0.5; do
    run_variant "baseline"     241 $ENT "--no-memory --no-comm --comm-mode commnet"
    run_variant "memory_only"  139 $ENT "--no-comm --comm-mode commnet"
    run_variant "commnet"      127 $ENT "--comm-mode commnet"
done

echo "All 60 runs launched."
wait
echo "Done."
