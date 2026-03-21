#!/bin/bash
# run_meme_analysis_suite.sh
#
# Meme analysis training run:
#   2 variants × 3 seeds × 1M steps on partial obs simple_spread, N=5 agents
#   Probes hidden states every 50k steps → 20 snapshots per run
#   Param-matched: ~125k params (memory_only hid=90, commnet_persistent hid=80)
#   ent_coef=0.3 for stability

cd /Users/almondgod/Repositories/memeplex-capstone
export PYTHONPATH="$(pwd):$PYTHONPATH"

PY=/opt/homebrew/bin/python3.9
SAVE_DIR="new/memetic_foundation/checkpoints/meme_analysis"
mkdir -p "$SAVE_DIR"

BASE="--mode train --env mpe --mpe-scenario simple_spread_v2 \
      --n-adversaries 5 \
      --obs-radius 0.5 --obs-curriculum-steps 200000 \
      --total-steps 1000000 --no-param-eq --ent-coef 0.3 \
      --probe-interval 50000 --probe-episodes 10"

echo "=================================================="
echo "Meme Analysis Suite — 1M steps, N=5, partial obs"
echo "=================================================="
echo "Variants: memory_only (124k), commnet_persistent (124k)"
echo "Seeds: 3 | Steps: 1M | Probe: every 50k"
echo "--------------------------------------------------"

run_variant() {
    local NAME=$1
    local HID=$2
    local FLAGS=$3
    echo ""
    echo ">>> $NAME (hidden_dim=$HID)"
    for SEED in 1 2 3; do
        LOG="${SAVE_DIR}/${NAME}_seed${SEED}.log"
        $PY -m new.memetic_foundation.run $BASE \
            --hidden-dim $HID --seed $SEED \
            --save-dir "${SAVE_DIR}/${NAME}_seed${SEED}" \
            $FLAGS > "$LOG" 2>&1 &
    done
    wait
    echo "  Seeds 1-3 done: $NAME"
    echo "--------------------------------------------------"
}

# memory_only hid=90 → 124,664 params
run_variant "memory_only" 90 "--no-comm --comm-mode commnet"

# commnet_persistent hid=80 → 124,294 params
run_variant "commnet_persistent" 80 "--comm-mode commnet --persistent-memory"

echo ""
echo "=================================================="
echo "Training complete. Run analysis with:"
echo ""
echo "  python -m new.memetic_foundation.analysis.meme_analysis \\"
echo "    --probe-dirs ${SAVE_DIR}/memory_only_seed1/probes \\"
echo "                 ${SAVE_DIR}/commnet_persistent_seed1/probes \\"
echo "    --labels memory_only commnet_persistent \\"
echo "    --out-dir ${SAVE_DIR}/analysis_seed1"
echo "=================================================="
