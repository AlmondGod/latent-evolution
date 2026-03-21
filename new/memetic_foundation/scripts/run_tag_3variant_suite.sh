#!/bin/bash
# run_tag_3variant_suite.sh
#
# 15 runs: 3 variants × 5 seeds × 200k steps on MPE simple_tag_v2
# Param-matched: all ~92k params (within 2%)
# ent_coef=0.3 based on sweep findings (eliminates entropy collapse)
# Variants: baseline, memory_only, commnet

cd /Users/almondgod/Repositories/memeplex-capstone
export PYTHONPATH="$(pwd):$PYTHONPATH"

PY=/opt/homebrew/bin/python3.9
SAVE_DIR="new/memetic_foundation/checkpoints/tag_3variant"
mkdir -p "$SAVE_DIR"

BASE="--mode train --env mpe --mpe-scenario simple_tag_v2 \
      --total-steps 200000 --no-param-eq --ent-coef 0.3"

echo "=================================================="
echo "Simple Tag 3-Variant Suite (GRU, param-matched)"
echo "=================================================="
echo "Variants: baseline (91k), memory_only (92k), commnet (92k)"
echo "Seeds: 5 | Steps: 200k | ent_coef: 0.3"
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

# hidden_dims tuned so all variants are within 2% of 92k params on simple_tag_v2:
# baseline=160 → 91,046 | memory_only=55 → 92,184 | commnet=48 → 92,006
run_variant "baseline"     160 "--no-memory --no-comm --comm-mode commnet"
run_variant "memory_only"  55  "--no-comm --comm-mode commnet"
run_variant "commnet"      48  "--comm-mode commnet"

echo ""
echo "=================================================="
echo "All 15 runs complete. Results in: ${SAVE_DIR}"
echo "=================================================="
