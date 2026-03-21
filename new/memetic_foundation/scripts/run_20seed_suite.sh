#!/bin/bash
# 6 variants × 20 seeds — partial obs, all ~185-187k params (within 1%)
cd /Users/almondgod/Repositories/memeplex-capstone
export PYTHONPATH="$(pwd):$PYTHONPATH"

PY=/opt/homebrew/bin/python3.9
SAVE_DIR="new/memetic_foundation/checkpoints/20seed_n3"
mkdir -p "$SAVE_DIR"

BASE="--env mpe --mpe-scenario simple_spread_v2 --total-steps 400000 --n-adversaries 3 --obs-radius 0.5 --obs-radius-curriculum --no-param-eq"

run_variant() {
    local NAME=$1; local HID=$2; local FLAGS=$3
    echo ">>> $NAME (hidden_dim=$HID)"
    for SEED in $(seq 1 20); do
        LOG="${SAVE_DIR}/${NAME}_seed${SEED}.log"
        $PY -m new.memetic_foundation.run --mode train $BASE \
            --hidden-dim $HID --seed $SEED \
            --save-dir "${SAVE_DIR}/${NAME}_seed${SEED}" \
            $FLAGS > "$LOG" 2>&1 &
    done
    echo "  Seeds 1-20 launched"
}

# hidden_dims tuned so all variants are within 1% of each other (~185-187k params)
run_variant "baseline"           241 "--no-memory --no-comm --comm-mode commnet"
run_variant "comm_only"          210 "--no-memory --comm-mode commnet"
run_variant "memory_only"        139 "--no-comm   --comm-mode commnet"
run_variant "mem_persistent"     139 "--no-comm   --comm-mode commnet --persistent-memory"
run_variant "commnet"            127 "--comm-mode commnet"
run_variant "commnet_persistent" 127 "--comm-mode commnet --persistent-memory"

echo "All 120 runs launched. Logs: $SAVE_DIR/"
wait
echo "Suite complete."
