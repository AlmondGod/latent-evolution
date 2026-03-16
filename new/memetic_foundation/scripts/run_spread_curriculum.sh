#!/bin/bash
# run_spread_curriculum.sh
# Curriculum obs_radius training: eliminates bimodal distribution in spread partial obs.
#
# Key insight: agents fail catastrophically when they initialize far from ALL landmarks.
# With obs_radius=0.5, they see nothing → random walk → fly off to extreme positions.
#
# Fix: start with full observability (obs_radius=100, effectively full obs),
# gradually anneal to obs_radius=0.5 over first 100k steps.
# By step 100k, agents have learned basic coverage → memory starts accumulating useful info.
#
# Hypothesis: curriculum training should:
# 1. Eliminate the ~16% catastrophic failure rate in baseline
# 2. Allow faster specialization in memory variants (good initial policy → GRU has signal)
# 3. Show the N-scaling advantage more clearly
#
# Design: 3 variants × N={3,5,8} × 4 seeds × 400k steps (with curriculum) = 36 runs
# Smaller seed count (4 vs 6) since curriculum should reduce variance.

export PYTHONPATH="$(pwd):$PYTHONPATH"

ENV="mpe"
SCENARIO="simple_spread_v2"
TOTAL_STEPS=400000
OBS_RADIUS=0.5
CURRICULUM_STEPS=100000   # anneal obs_radius over first 25% of training
SAVE_DIR="checkpoints/mpe_spread_curriculum"
mkdir -p "$SAVE_DIR"

echo "=================================================="
echo "Scalable Memetics: Spread + Curriculum Obs Radius"
echo "=================================================="
echo "Task: simple_spread (N agents cover N landmarks)"
echo "Curriculum: full_obs → obs_radius=0.5 over first ${CURRICULUM_STEPS} steps"
echo "N: 3, 5, 8 | Variants: baseline, memory_only, full_gated"
echo "Seeds: 4 per cell | Steps: ${TOTAL_STEPS}"
echo "Linear LR schedule: 5e-4 → 5e-5 over training"
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
    local N_AGENTS=$3
    local SEED_START=$4
    local SEED_END=$5

    for SEED in $(seq $SEED_START $SEED_END); do
        LOG="$SAVE_DIR/${LABEL}_seed${SEED}.log"
        python3.9 -m new.memetic_foundation.run --mode train \
            --env "$ENV" \
            --mpe-scenario "$SCENARIO" \
            --total-steps "$TOTAL_STEPS" \
            --seed "$SEED" \
            --n-adversaries "$N_AGENTS" \
            --obs-radius "$OBS_RADIUS" \
            --obs-radius-curriculum \
            --obs-curriculum-steps "$CURRICULUM_STEPS" \
            --save-dir "$SAVE_DIR/${LABEL}_seed${SEED}" \
            $FLAGS > "$LOG" 2>&1 &
    done
    wait
}

for N_AGENTS in 3 5 8; do
    for VARIANT in baseline memory_only full_gated; do
        FLAGS=$(get_flags "$VARIANT")
        LABEL="${VARIANT}_n${N_AGENTS}"
        echo ""
        echo ">>> ${LABEL} (N=${N_AGENTS}, curriculum obs, spread)"

        run_batch "$LABEL" "$FLAGS" "$N_AGENTS" 1 2
        run_batch "$LABEL" "$FLAGS" "$N_AGENTS" 3 4

        echo "  Done: ${LABEL}"
    done
done

echo ""
echo "=================================================="
echo "All 36 runs complete. Results in: ${SAVE_DIR}"
echo "  Analysis: python3.9 -m new.memetic_foundation.scripts.analyze_spread_nscale"
echo "  Compare curriculum vs no-curriculum: expect fewer catastrophic failures"
echo "  Key question: does memory advantage still grow with N?"
echo "=================================================="
