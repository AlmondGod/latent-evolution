#!/bin/bash
# run_tag_400k_suite.sh
#
# 15 runs: 3 variants × 5 seeds × 400k steps on MPE simple_tag_v2
# Param-matched: ~92k params (within 2%)
# ent_coef=0.3, longer run to see if memory_only catches up

cd /Users/almondgod/Repositories/memeplex-capstone
export PYTHONPATH="$(pwd):$PYTHONPATH"

PY=/opt/homebrew/bin/python3.9
SAVE_DIR="new/memetic_foundation/checkpoints/tag_400k"
mkdir -p "$SAVE_DIR"

BASE="--mode train --env mpe --mpe-scenario simple_tag_v2 \
      --total-steps 400000 --no-param-eq --ent-coef 0.3"

echo "=================================================="
echo "Simple Tag 3-Variant Suite — 400k steps"
echo "=================================================="
echo "Variants: baseline (91k), memory_only (92k), commnet (92k)"
echo "Seeds: 5 | Steps: 400k | ent_coef: 0.3"
echo "--------------------------------------------------"

run_variant() {
    local NAME=$1
    local HID=$2
    local FLAGS=$3
    echo ""
    echo ">>> $NAME (hidden_dim=$HID)"
    for SEED in 1 2 3 4 5; do
        LOG="${SAVE_DIR}/${NAME}_seed${SEED}.log"
        $PY -m new.memetic_foundation.run $BASE \
            --hidden-dim $HID --seed $SEED \
            --save-dir "${SAVE_DIR}/${NAME}_seed${SEED}" \
            $FLAGS > "$LOG" 2>&1 &
    done
    wait
    echo "  Seeds 1-5 done: $NAME"
    echo "--------------------------------------------------"
}

# baseline=160 → 91,046 | memory_only=55 → 92,184 | commnet=48 → 92,006
run_variant "baseline"     160 "--no-memory --no-comm --comm-mode commnet"
run_variant "memory_only"  55  "--no-comm --comm-mode commnet"
run_variant "commnet"      48  "--comm-mode commnet"

echo ""
echo "=================================================="
echo "All 15 runs complete. Results in: ${SAVE_DIR}"
echo "=================================================="
