#!/bin/bash
# Seeds 6-20 for 6 key variants — partial obs, param_eq=True
cd /Users/almondgod/Repositories/memeplex-capstone
export PYTHONPATH="$(pwd):$PYTHONPATH"

PY=/opt/homebrew/bin/python3.9
SAVE_DIR="new/memetic_foundation/checkpoints/extended_seeds_n3"
mkdir -p "$SAVE_DIR"

BASE="--env mpe --mpe-scenario simple_spread_v2 --total-steps 400000 --n-adversaries 3 --obs-radius 0.5 --obs-radius-curriculum"

run_variant() {
    local NAME=$1; local FLAGS=$2
    echo ">>> $NAME"
    for SEED in $(seq 6 20); do
        LOG="${SAVE_DIR}/${NAME}_seed${SEED}.log"
        $PY -m new.memetic_foundation.run --mode train $BASE \
            --seed $SEED --save-dir "${SAVE_DIR}/${NAME}_seed${SEED}" \
            $FLAGS > "$LOG" 2>&1 &
        echo "  seed${SEED} PID $!"
    done
}

run_variant "baseline"           "--no-memory --no-comm --comm-mode commnet"
run_variant "comm_only"          "--no-memory --comm-mode commnet"
run_variant "memory_only"        "--no-comm   --comm-mode commnet"
run_variant "mem_persistent"     "--no-comm   --comm-mode commnet --persistent-memory"
run_variant "commnet"            "--comm-mode commnet"
run_variant "commnet_persistent" "--comm-mode commnet --persistent-memory"

echo "All 90 runs launched. Logs: $SAVE_DIR/"
wait
echo "Done."
