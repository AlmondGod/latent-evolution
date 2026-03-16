#!/bin/bash
# run_tag_1m_convergence.sh — 3 seeds x 4 variants x 1M steps on simple_tag
export PYTHONPATH="$(pwd):$PYTHONPATH"

ENV="mpe"
SCENARIO="simple_tag_v2"
TOTAL_STEPS=1000000
SAVE_DIR="checkpoints/mpe_tag_1m"
mkdir -p "$SAVE_DIR"

echo "=================================================="
echo "Memetic Foundation Simple Tag 1M Convergence Study"
echo "=================================================="
echo "Seeds: 3 per variant   |   Steps: ${TOTAL_STEPS}"
echo "--------------------------------------------------"

get_flags() {
    case "$1" in
        baseline)    echo "--no-memory --no-comm" ;;
        memory_only) echo "--no-comm" ;;
        comm_only)   echo "--no-memory" ;;
        full)        echo "" ;;
    esac
}

for VARIANT in baseline memory_only comm_only full; do
    FLAGS=$(get_flags "$VARIANT")
    for SEED in 1 2 3; do
        LOG="$SAVE_DIR/${VARIANT}_seed${SEED}.log"
        python3.9 -m new.memetic_foundation.run --mode train \
            --env "$ENV" \
            --mpe-scenario "$SCENARIO" \
            --total-steps "$TOTAL_STEPS" \
            --seed "$SEED" \
            --save-dir "$SAVE_DIR/${VARIANT}_seed${SEED}" \
            $FLAGS \
            > "$LOG" 2>&1 &
        echo "  Launched ${VARIANT} seed${SEED} (PID $!)"
    done
done

echo ""
echo "All 12 runs launched. Waiting..."
wait
echo "=================================================="
echo "All done. Results in: $SAVE_DIR"
echo "=================================================="
