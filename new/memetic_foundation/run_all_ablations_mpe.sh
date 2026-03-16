#!/bin/bash
# Run all 4 Memetic Foundation variants sequentially on MPE simple_tag_v2
# 3 seeds per variant for reliable comparison
#
# Reliability measures:
#   - 3 random seeds per variant (42, 123, 456)
#   - Same simple_tag_v2 config across all runs
#   - Results JSON saved per run for statistical comparison
#   - Checkpoints saved for reproducibility

set -e
cd /Users/almondgod/Repositories/memeplex-capstone

STEPS=1000000
ROLLOUT=400
SEEDS=(42 123 456)

VARIANTS=(
    "--no-memory --no-comm"   # baseline
    "--no-comm"               # memory_only
    "--no-memory"             # comm_only
    ""                        # full
)
VARIANT_NAMES=(
    "baseline"
    "memory_only"
    "comm_only"
    "full"
)

TOTAL_RUNS=$(( ${#VARIANTS[@]} * ${#SEEDS[@]} ))
RUN_NUM=0

echo "============================================="
echo "  MEMETIC FOUNDATION — MPE ABLATION STUDY"
echo "  4 variants × 3 seeds = $TOTAL_RUNS runs"
echo "  $STEPS steps each, rollout=$ROLLOUT"
echo "============================================="
echo ""

for i in "${!VARIANTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        RUN_NUM=$((RUN_NUM + 1))
        echo "[$RUN_NUM/$TOTAL_RUNS] ${VARIANT_NAMES[$i]} (seed=$seed)"
        python3.9 -m new.memetic_foundation --mode train \
            --env mpe \
            --mpe-scenario simple_tag_v2 \
            ${VARIANTS[$i]} \
            --total-steps $STEPS \
            --rollout-steps $ROLLOUT \
            --seed $seed \
            --log-interval 50 \
            --save-interval 500
        echo ""
    done
done

echo "============================================="
echo "  ALL $TOTAL_RUNS RUNS COMPLETE"
echo "============================================="
