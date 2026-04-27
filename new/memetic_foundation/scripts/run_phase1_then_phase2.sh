#!/usr/bin/env bash
# Run Phase-1 backbones, then 3-arm Phase-2 comparison, for a given N_AGENTS / N_ENEMIES.
# Used to chain a single team-size end-to-end. The outer driver loops over N values.

set -uo pipefail

ROOT="/workspace/latent-evolution"
cd "$ROOT"
export PYTHONPATH="$ROOT"

N_AGENTS="${N_AGENTS:-4}"
N_ENEMIES="${N_ENEMIES:-4}"
SEEDS_STR="${SEEDS:-42 43 44}"
WALLCLOCK="${WALLCLOCK:-1800}"

PHASE1_DIR="results/smacv2_phase1_attn_${N_AGENTS}v${N_ENEMIES}"
PHASE2_DIR="results/smacv2_phase2_3arm_${N_AGENTS}v${N_ENEMIES}"

echo "==================================================================="
echo "[meta] N_AGENTS=$N_AGENTS N_ENEMIES=$N_ENEMIES"
echo "[meta] PHASE1_DIR=$PHASE1_DIR"
echo "[meta] PHASE2_DIR=$PHASE2_DIR"
echo "==================================================================="

# Phase 1 (3 seeds in parallel)
echo "[meta] starting Phase 1 ..."
N_AGENTS="$N_AGENTS" N_ENEMIES="$N_ENEMIES" SEEDS="$SEEDS_STR" \
  bash new/memetic_foundation/scripts/run_smacv2_phase1_attn.sh

# Wait on the launched Phase-1 PIDs
PIDS_FILE="${PHASE1_DIR}/pids.txt"
if [[ ! -f "$PIDS_FILE" ]]; then
  echo "[meta error] no pids.txt at $PIDS_FILE" >&2
  exit 1
fi
read -ra PHASE1_PIDS < "$PIDS_FILE"
echo "[meta] waiting for Phase-1 pids: ${PHASE1_PIDS[*]}"
for pid in "${PHASE1_PIDS[@]}"; do
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
  echo "[meta] phase1 pid=$pid done"
done
echo "[meta] Phase 1 done"

# Phase 2 (sequential arms, parallel seeds within arm)
echo "[meta] starting Phase 2 ..."
mkdir -p "$PHASE2_DIR"
N_AGENTS="$N_AGENTS" N_ENEMIES="$N_ENEMIES" SEEDS="$SEEDS_STR" \
  PHASE1_DIR="$PHASE1_DIR" OUT_DIR="$PHASE2_DIR" WALLCLOCK="$WALLCLOCK" \
  bash new/memetic_foundation/scripts/run_3arm_phase2_smacv2_4v4.sh

echo "[meta] Phase 2 done"

# Aggregate
PYTHONPATH="$ROOT" python3 new/memetic_foundation/scripts/aggregate_3arm_phase2.py \
  --out-dir "$PHASE2_DIR" --seeds $SEEDS_STR \
  > "$PHASE2_DIR/summary.txt" 2>&1 || true
echo "[meta] summary saved to $PHASE2_DIR/summary.txt"
