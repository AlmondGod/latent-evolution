#!/bin/bash
# Phase-1 backbone training: attention_hu_actor on SMACv2 4v4 Terran.
# Launches 3 seeds in parallel. Logs each to logs/phase1_attn_seed<N>.log.
set -euo pipefail
ROOT="/workspace/latent-evolution"
cd "$ROOT"
export PYTHONPATH="$ROOT"
mkdir -p logs results/smacv2_phase1_attn
SEEDS=(42 43 44)
PIDS=()
for SEED in "${SEEDS[@]}"; do
  SAVE_DIR="results/smacv2_phase1_attn/seed${SEED}"
  LOG="logs/phase1_attn_seed${SEED}.log"
  echo "=== launching seed ${SEED} -> ${SAVE_DIR} (log: ${LOG}) ==="
  python3 new/memetic_foundation/scripts/direct_norm_perf.py \
    --variant attention_hu_actor \
    --env smacv2 --race terran --n-agents 4 --n-enemies 4 \
    --seed "$SEED" \
    --total-steps 500000 --rollout-steps 400 \
    --eval-interval 50000 --eval-episodes 16 \
    --eval-snapshot-episodes 16 \
    --print-every 10000 \
    --save-dir "$SAVE_DIR" \
    --device cuda \
    >"$LOG" 2>&1 &
  PIDS+=($!)
done
echo "PIDS: ${PIDS[*]}"
echo "${PIDS[*]}" > logs/phase1_attn_pids.txt
wait "${PIDS[@]}"
echo "all 3 seeds finished"
