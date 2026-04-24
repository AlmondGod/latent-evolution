#!/bin/bash
set -euo pipefail

ROOT="/Users/almondgod/Repositories/memeplex-capstone"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.9}"
BASE_DIR="${BASE_DIR:-new/memetic_foundation/checkpoints/vmas_phase1_discovery}"
TOTAL_STEPS="${TOTAL_STEPS:-20000}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
EVAL_EPISODES="${EVAL_EPISODES:-8}"
SEEDS="${SEEDS:-1}"
NS="${NS:-2 4 8}"
VARIANTS="${VARIANTS:-attention_hu_actor}"
VMAS_TARGETS="${VMAS_TARGETS:-4}"

for N in $NS; do
  for VARIANT in $VARIANTS; do
    for SEED in $SEEDS; do
      SAVE_DIR="${BASE_DIR}/discovery_n${N}/${VARIANT}/seed${SEED}"
      echo "=== discovery_n${N} | ${VARIANT} | seed ${SEED} ==="
      $PYTHON_BIN /Users/almondgod/Repositories/memeplex-capstone/new/memetic_foundation/scripts/direct_norm_perf.py \
        --variant "$VARIANT" \
        --env vmas \
        --vmas-scenario discovery \
        --vmas-targets "$VMAS_TARGETS" \
        --n-agents "$N" \
        --seed "$SEED" \
        --total-steps "$TOTAL_STEPS" \
        --rollout-steps "$ROLLOUT_STEPS" \
        --eval-episodes "$EVAL_EPISODES" \
        --eval-interval "$EVAL_INTERVAL" \
        --eval-snapshot-episodes "$EVAL_EPISODES" \
        --save-dir "$SAVE_DIR" \
        --device cpu
    done
  done
done
