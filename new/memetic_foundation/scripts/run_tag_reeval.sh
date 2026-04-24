#!/bin/bash
# run_tag_reeval.sh
#
# Re-evaluate all 100 tag_nscale checkpoints with 100 episodes each.
# Runs all in parallel (OS schedules across cores).
# Results written to checkpoints/tag_nscale/{run}/reeval.log

export PYTHONPATH="$(pwd):$PYTHONPATH"

BASE="checkpoints/tag_nscale"
EPISODES=100

echo "=================================================="
echo "Tag N-Scale Re-Evaluation (${EPISODES} episodes each)"
echo "=================================================="

reeval() {
    local VARIANT=$1
    local N=$2
    local SEED=$3
    local LABEL="${VARIANT}_n${N}_seed${SEED}"
    local RUN_DIR="${BASE}/${LABEL}"
    local CKPT=$(find "$RUN_DIR" -name "*_step_400000.pt" 2>/dev/null | head -1)

    if [ -z "$CKPT" ]; then
        echo "  MISSING checkpoint: $LABEL"
        return
    fi

    local LOG="${RUN_DIR}/reeval.log"

    case "$VARIANT" in
        memory_only)
            python3.9 -m new.memetic_foundation.run --mode eval \
                --env mpe --mpe-scenario simple_tag_v2 \
                --n-adversaries "$N" \
                --no-comm \
                --cpu \
                --eval-episodes "$EPISODES" \
                --load-path "$CKPT" > "$LOG" 2>&1 &
            ;;
        memory_only_persistent)
            python3.9 -m new.memetic_foundation.run --mode eval \
                --env mpe --mpe-scenario simple_tag_v2 \
                --n-adversaries "$N" \
                --no-comm --persistent-memory \
                --cpu \
                --eval-episodes "$EPISODES" \
                --load-path "$CKPT" > "$LOG" 2>&1 &
            ;;
        commnet_persistent)
            python3.9 -m new.memetic_foundation.run --mode eval \
                --env mpe --mpe-scenario simple_tag_v2 \
                --n-adversaries "$N" \
                --comm-mode commnet --persistent-memory \
                --cpu \
                --eval-episodes "$EPISODES" \
                --load-path "$CKPT" > "$LOG" 2>&1 &
            ;;
        commnet)
            python3.9 -m new.memetic_foundation.run --mode eval \
                --env mpe --mpe-scenario simple_tag_v2 \
                --n-adversaries "$N" \
                --comm-mode commnet \
                --cpu \
                --eval-episodes "$EPISODES" \
                --load-path "$CKPT" > "$LOG" 2>&1 &
            ;;
    esac
}

# Launch all 100 in parallel
for N in 3 6 9 12 15; do
    for VARIANT in memory_only memory_only_persistent commnet_persistent commnet; do
        for SEED in 1 2 3 4 5; do
            reeval "$VARIANT" "$N" "$SEED"
        done
    done
done

echo "Launched 100 eval jobs — waiting for all to complete..."
wait
echo ""
echo "All re-evals done. Results in ${BASE}/*/*/reeval.log"
