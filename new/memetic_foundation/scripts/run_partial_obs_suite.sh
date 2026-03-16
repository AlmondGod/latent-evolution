#!/bin/bash
# run_partial_obs_suite.sh
# Tests whether partial observability makes memory+comm genuinely useful.
# Hypothesis: with full obs, agents homogenize (same view → same memes).
#             With partial obs, agents see different subsets → different memes
#             → communication + memory provides real information gain.
#
# Obs radius 0.5 in MPE world units (~arena is ~2.0 wide).
# Agents can see ~25% of arena — genuinely partial.
#
# 3 variants x 8 seeds x 200k steps = 24 runs (4 parallel per batch).

export PYTHONPATH="$(pwd):$PYTHONPATH"

ENV="mpe"
SCENARIO="simple_tag_v2"
TOTAL_STEPS=200000
OBS_RADIUS=0.5
SAVE_DIR="checkpoints/mpe_tag_partial_obs"
mkdir -p "$SAVE_DIR"

echo "=================================================="
echo "Memetic Foundation Partial Observability Suite"
echo "=================================================="
echo "Obs radius: ${OBS_RADIUS} (partial, ~25% arena visibility)"
echo "Variants: baseline, memory_only, full_gated"
echo "Seeds: 8   |   Steps: ${TOTAL_STEPS}"
echo "--------------------------------------------------"

get_flags() {
    case "$1" in
        baseline)    echo "--no-memory --no-comm" ;;
        memory_only) echo "--no-comm" ;;
        full_gated)  echo "" ;;
    esac
}

run_variant() {
    local LABEL=$1
    local FLAGS=$2
    echo ""
    echo ">>> Variant: ${LABEL}"
    for SEED in 1 2 3 4; do
        python3.9 -m new.memetic_foundation.run --mode train \
            --env "$ENV" --mpe-scenario "$SCENARIO" \
            --total-steps "$TOTAL_STEPS" --seed "$SEED" \
            --obs-radius "$OBS_RADIUS" \
            --save-dir "$SAVE_DIR/${LABEL}_seed${SEED}" \
            $FLAGS > "$SAVE_DIR/${LABEL}_seed${SEED}.log" 2>&1 &
    done
    wait
    echo "  Batch 1 (seeds 1-4) done."
    for SEED in 5 6 7 8; do
        python3.9 -m new.memetic_foundation.run --mode train \
            --env "$ENV" --mpe-scenario "$SCENARIO" \
            --total-steps "$TOTAL_STEPS" --seed "$SEED" \
            --obs-radius "$OBS_RADIUS" \
            --save-dir "$SAVE_DIR/${LABEL}_seed${SEED}" \
            $FLAGS > "$SAVE_DIR/${LABEL}_seed${SEED}.log" 2>&1 &
    done
    wait
    echo "  Batch 2 (seeds 5-8) done."
    echo "  Done: ${LABEL}"
    echo "--------------------------------------------------"
}

run_variant "baseline"    "--no-memory --no-comm"
run_variant "memory_only" "--no-comm"
run_variant "full_gated"  ""

echo ""
echo "=================================================="
echo "All 24 runs complete. Results in: ${SAVE_DIR}"
echo "=================================================="
