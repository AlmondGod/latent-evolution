#!/bin/bash
# run_spread_partial_obs.sh
# The CORRECT task for scalable memetics:
#   Simple spread with partial observability.
#
# Why this is the right task:
#   - N agents must cover N distinct landmarks (structural role asymmetry)
#   - With obs_radius=0.5, each agent sees only nearby landmarks/agents
#   - Agents naturally must specialize to different landmarks
#   - Memory helps agents "remember" their landmark assignment
#   - Communication allows sharing of complementary coverage info
#   - The task SCALES: with more agents, coordination becomes more complex
#
# This directly tests the scalable memetics hypothesis:
#   memory+comm advantage should grow with N when task requires
#   differentiated, communicating subpolicies.
#
# Design: 3 variants × N={3,5,8} × 6 seeds × 400k steps = 54 runs
# Running 3 parallel per batch (memory-intensive task).

export PYTHONPATH="$(pwd):$PYTHONPATH"

ENV="mpe"
SCENARIO="simple_spread_v2"
TOTAL_STEPS=400000   # 2x longer for stable convergence (LR schedule now helps)
OBS_RADIUS=0.5
SAVE_DIR="checkpoints/mpe_spread_partial_obs"
mkdir -p "$SAVE_DIR"

echo "=================================================="
echo "Scalable Memetics: Spread + Partial Obs + N-Scaling"
echo "=================================================="
echo "Task: simple_spread (N agents cover N landmarks)"
echo "Obs radius: ${OBS_RADIUS} → structural role asymmetry"
echo "N: 3, 5, 8 | Variants: baseline, memory_only, full_gated"
echo "Seeds: 6 per cell | Steps: ${TOTAL_STEPS}"
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
        echo ">>> ${LABEL} (N=${N_AGENTS}, partial obs, spread)"

        run_batch "$LABEL" "$FLAGS" "$N_AGENTS" 1 3
        run_batch "$LABEL" "$FLAGS" "$N_AGENTS" 4 6

        echo "  Done: ${LABEL}"
    done
done

echo ""
echo "=================================================="
echo "All 54 runs complete. Results in: ${SAVE_DIR}"
echo "  Analysis: python3.9 -m new.memetic_foundation.scripts.analyze_partial_obs_nscale"
echo "  The key question: does memory advantage grow with N on spread+partial obs?"
echo "=================================================="
