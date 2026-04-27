#!/bin/bash
# SMACv2 Phase 1 backbone training matrix.
# Terran, fixed n_enemies=4, varying n_units in {2,4,8}, 3 seeds each.
# 500k env steps per cell, baseline (no memory, no comm) MAPPO variant.
# Skips n_units=4 seed=42 if already complete.

set -u
cd "$(dirname "$0")/../../.."

ROOT_DIR="results/smacv2_phase1_500k"
LOG_DIR="$ROOT_DIR/_logs"
mkdir -p "$LOG_DIR"

PIDS=()
LAUNCHED=()

for N in 2 4 8; do
  for SEED in 42 43 44; do
    CELL="n${N}_seed${SEED}"
    SAVE_DIR="$ROOT_DIR/$CELL"
    LOG_FILE="$LOG_DIR/${CELL}.log"

    # Skip if checkpoint at 500k already exists
    if find "$SAVE_DIR" -name "memfound_baseline_step_500000.pt" 2>/dev/null | grep -q .; then
      echo "[skip] $CELL already complete"
      continue
    fi

    mkdir -p "$SAVE_DIR"
    nohup python3 -m new.memetic_foundation.run \
      --mode train --env smacv2 \
      --race terran --n-units "$N" --n-enemies 4 \
      --total-steps 500000 --rollout-steps 400 \
      --no-memory --no-comm \
      --seed "$SEED" \
      --log-interval 25 --save-interval 125 \
      --save-dir "$SAVE_DIR" \
      > "$LOG_FILE" 2>&1 &
    PID=$!
    PIDS+=("$PID")
    LAUNCHED+=("$CELL pid=$PID")
    echo "[launch] $CELL pid=$PID log=$LOG_FILE"
  done
done

echo
echo "Launched ${#PIDS[@]} runs."
printf 'pid: %s\n' "${PIDS[@]}"
echo
echo "Tail logs with:  tail -f $LOG_DIR/*.log"
echo "Check progress:  ls $ROOT_DIR/n*_seed*/memfound_baseline_*/memfound_baseline_step_*.pt"
