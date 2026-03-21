#!/bin/bash
# Entropy coefficient sweep: 0.1 and 0.2
# Key variants only: baseline, memory_only, commnet
# 10 seeds each, partial obs, param-matched hidden dims
cd /Users/almondgod/Repositories/memeplex-capstone
export PYTHONPATH="$(pwd):$PYTHONPATH"

PY=/opt/homebrew/bin/python3.9
SAVE_DIR="new/memetic_foundation/checkpoints/entropy_sweep"
mkdir -p "$SAVE_DIR"

BASE="--env mpe --mpe-scenario simple_spread_v2 --total-steps 400000 --n-adversaries 3 --obs-radius 0.5 --obs-radius-curriculum --no-param-eq"

run_variant() {
    local NAME=$1; local HID=$2; local ENT=$3; local FLAGS=$4
    local TAG="${NAME}_ent${ENT}"
    echo ">>> $TAG (hidden_dim=$HID, ent_coef=$ENT)"
    for SEED in $(seq 1 10); do
        LOG="${SAVE_DIR}/${TAG}_seed${SEED}.log"
        $PY -m new.memetic_foundation.run --mode train $BASE \
            --hidden-dim $HID --ent-coef $ENT --seed $SEED \
            --save-dir "${SAVE_DIR}/${TAG}_seed${SEED}" \
            $FLAGS > "$LOG" 2>&1 &
    done
    echo "  Seeds 1-10 launched"
}

for ENT in 0.1 0.2; do
    run_variant "baseline"     241 $ENT "--no-memory --no-comm --comm-mode commnet"
    run_variant "memory_only"  139 $ENT "--no-comm --comm-mode commnet"
    run_variant "commnet"      127 $ENT "--comm-mode commnet"
done

echo "All 60 runs launched. Logs: $SAVE_DIR/"
wait
echo "Suite complete."
