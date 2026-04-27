#!/usr/bin/env bash
# Phase-1 attention_hu_actor backbones on SMACv2 4v4 Terran
# 3 seeds in parallel via Lever 1.
set -euo pipefail

ROOT="/workspace/latent-evolution"
cd "$ROOT"
export PYTHONPATH="$ROOT"

N_AGENTS="${N_AGENTS:-4}"
N_ENEMIES="${N_ENEMIES:-4}"
BASE_DIR="${BASE_DIR:-results/smacv2_phase1_attn_${N_AGENTS}v${N_ENEMIES}}"
TOTAL_STEPS="${TOTAL_STEPS:-500000}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-400}"
SEEDS="${SEEDS:-42 43 44}"

mkdir -p "$BASE_DIR"
PIDS=()
for SEED in $SEEDS; do
  SAVE_DIR="${BASE_DIR}/seed${SEED}"
  mkdir -p "$SAVE_DIR"
  LOG="${SAVE_DIR}/train.log"
  echo "[launch] seed=${SEED} -> ${SAVE_DIR}  (n_agents=${N_AGENTS} n_enemies=${N_ENEMIES})"
  nohup python3 new/memetic_foundation/scripts/direct_norm_perf.py \
    --variant attention_hu_actor \
    --env smacv2 --race terran --n-agents "$N_AGENTS" --n-enemies "$N_ENEMIES" \
    --seed "$SEED" \
    --total-steps "$TOTAL_STEPS" \
    --rollout-steps "$ROLLOUT_STEPS" \
    --eval-interval 0 --eval-episodes 8 \
    --save-dir "$SAVE_DIR" \
    --device cuda \
    > "$LOG" 2>&1 &
  PIDS+=("$!")
done

echo "[launched] pids=${PIDS[*]}"
echo "[hint] tail -f ${BASE_DIR}/seed*/train.log"
echo "${PIDS[*]}" > "${BASE_DIR}/pids.txt"
