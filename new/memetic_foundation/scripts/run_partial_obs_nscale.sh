#!/bin/bash
# run_partial_obs_nscale.sh
# The key memetic scaling experiment:
#   Does memory+comm advantage GROW with N under partial observability?
#
# Full obs: all agents see same prey → memory homogenizes → no advantage
# Partial obs (obs_radius=0.5): agents see different slices → different memory
#   states → information asymmetry → comm provides real gain
#
# Hypothesis: memory_advantage = (memory_only - baseline) / baseline
#   grows with N under partial obs, but stays flat under full obs.
#
# Design: 3 variants × N={3,5,8} × 6 seeds × 200k steps = 54 runs
# Runs 4 parallel within each batch.

export PYTHONPATH="$(pwd):$PYTHONPATH"

ENV="mpe"
SCENARIO="simple_tag_v2"
TOTAL_STEPS=200000
OBS_RADIUS=0.5
SAVE_DIR="checkpoints/mpe_partial_obs_nscale"
mkdir -p "$SAVE_DIR"

echo "=================================================="
echo "Partial-Obs N-Agent Scaling: Core Memetic Test"
echo "=================================================="
echo "Obs radius: ${OBS_RADIUS} (~25% arena visibility)"
echo "N: 3, 5, 8 adversaries | Variants: baseline, memory_only, full_gated"
echo "Seeds: 6 per cell | Steps: ${TOTAL_STEPS}"
echo "--------------------------------------------------"

get_flags() {
    case "$1" in
        baseline)    echo "--no-memory --no-comm" ;;
        memory_only) echo "--no-comm" ;;
        full_gated)  echo "" ;;
    esac
}

run_batch() {
    local LABEL=$1
    local FLAGS=$2
    local N_ADV=$3
    local SEED_START=$4
    local SEED_END=$5

    for SEED in $(seq $SEED_START $SEED_END); do
        LOG="$SAVE_DIR/${LABEL}_seed${SEED}.log"
        python3.9 -m new.memetic_foundation.run --mode train \
            --env "$ENV" --mpe-scenario "$SCENARIO" \
            --total-steps "$TOTAL_STEPS" \
            --seed "$SEED" \
            --n-adversaries "$N_ADV" \
            --obs-radius "$OBS_RADIUS" \
            --save-dir "$SAVE_DIR/${LABEL}_seed${SEED}" \
            $FLAGS > "$LOG" 2>&1 &
    done
    wait
}

for N_ADV in 3 5 8; do
    for VARIANT in baseline memory_only full_gated; do
        FLAGS=$(get_flags "$VARIANT")
        LABEL="${VARIANT}_n${N_ADV}"
        echo ""
        echo ">>> ${LABEL} (N=${N_ADV}, partial obs)"

        # 6 seeds in 2 batches of 3 (to not swamp the machine)
        run_batch "$LABEL" "$FLAGS" "$N_ADV" 1 3
        run_batch "$LABEL" "$FLAGS" "$N_ADV" 4 6

        echo "  Done: ${LABEL}"
    done
done

echo ""
echo "=================================================="
echo "All 54 runs complete. Results in: ${SAVE_DIR}"
echo "  Key analysis: python3.9 -m new.memetic_foundation.scripts.analyze_partial_obs_nscale"
echo "=================================================="
