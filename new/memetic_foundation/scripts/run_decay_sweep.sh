#!/bin/bash
# Decay value sweep: test 5 different decay values for memory_only and full
# on simple_tag_v2, 200k steps, 4 seeds each

cd /Users/almondgod/Repositories/memeplex-capstone

STEPS=200000
SCENARIO=simple_tag_v2
SEEDS=(42 123 456 789)
DECAY_VALUES=(0.0 0.001 0.005 0.01 0.05)
VARIANTS=("memory_only" "full")

echo "=================================================="
echo "Memory Decay Sweep"
echo "=================================================="
echo "Decay values: ${DECAY_VALUES[*]}"
echo "Variants: ${VARIANTS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total runs: $((${#DECAY_VALUES[@]} * ${#VARIANTS[@]} * ${#SEEDS[@]}))"
echo "--------------------------------------------------"

OUT_DIR="checkpoints/decay_sweep"
mkdir -p "$OUT_DIR"

for decay in "${DECAY_VALUES[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        echo ""
        echo ">>> decay=$decay variant=$variant"

        # Build flags
        FLAGS=""
        if [ "$variant" = "memory_only" ]; then
            FLAGS="--no-comm"
        elif [ "$variant" = "comm_only" ]; then
            FLAGS="--no-memory"
        elif [ "$variant" = "baseline" ]; then
            FLAGS="--no-memory --no-comm"
        fi

        # Run 4 seeds in parallel
        PIDS=()
        for seed in "${SEEDS[@]}"; do
            RUN_DIR="$OUT_DIR/${variant}_decay${decay}_seed${seed}"
            mkdir -p "$RUN_DIR"
            LOG="$RUN_DIR/train.log"

            python3.9 -m new.memetic_foundation.run \
                --mode train \
                --env mpe \
                --mpe-scenario "$SCENARIO" \
                --total-steps "$STEPS" \
                --seed "$seed" \
                --mem-decay "$decay" \
                --save-dir "$RUN_DIR" \
                $FLAGS \
                > "$LOG" 2>&1 &
            PIDS+=($!)
        done

        # Wait for all seeds
        for pid in "${PIDS[@]}"; do
            wait "$pid"
        done
        echo "  Done: decay=$decay variant=$variant"
    done
done

echo ""
echo "=================================================="
echo "Decay Sweep Complete!"
echo "=================================================="
