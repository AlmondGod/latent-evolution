#!/bin/bash
set -euo pipefail

ROOT="/Users/almondgod/Repositories/memeplex-capstone"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.9}"
MATRIX_JSON="${MATRIX_JSON:-/Users/almondgod/Repositories/memeplex-capstone/results/phase2_agent_budget_matrix/phase2_agent_budget_matrix.json}"
BACKBONE_ROOT="${BACKBONE_ROOT:-/Users/almondgod/Repositories/memeplex-capstone/new/memetic_foundation/checkpoints/vmas_phase1_transport_50k}"
SAVE_ROOT="${SAVE_ROOT:-/Users/almondgod/Repositories/memeplex-capstone/results/vmas_phase2_linear_a_transport_50k}"
SEED="${SEED:-1}"
NS="${NS:-2 4 8}"

"$PYTHON_BIN" /Users/almondgod/Repositories/memeplex-capstone/new/memetic_foundation/scripts/run_phase2_agent_budget_pilot.py \
  --matrix-json "$MATRIX_JSON" \
  --regime linear_a \
  --backbone-root "$BACKBONE_ROOT" \
  --save-root "$SAVE_ROOT" \
  --ns $NS \
  --seed "$SEED" \
  --env vmas \
  --vmas-scenario transport \
  --vmas-packages 1 \
  --device cpu
