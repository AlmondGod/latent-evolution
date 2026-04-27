#!/usr/bin/env bash
# Outer driver: run Phase-1 + 3-arm Phase-2 for each N_AGENTS in NS, with N_ENEMIES fixed.
# Default: N=2 then N=8 (with n_enemies=4), matching the team-scaling protocol from the paper.

set -uo pipefail

ROOT="/workspace/latent-evolution"
cd "$ROOT"
export PYTHONPATH="$ROOT"

NS="${NS:-2 8}"
N_ENEMIES="${N_ENEMIES:-4}"
SEEDS="${SEEDS:-42 43 44}"
WALLCLOCK="${WALLCLOCK:-1800}"

for N in $NS; do
  echo
  echo "###################################################################"
  echo "###  team scaling: N=${N} vs ${N_ENEMIES} enemies"
  echo "###################################################################"
  N_AGENTS="$N" N_ENEMIES="$N_ENEMIES" SEEDS="$SEEDS" WALLCLOCK="$WALLCLOCK" \
    bash new/memetic_foundation/scripts/run_phase1_then_phase2.sh
done

echo "[driver] all team sizes done"
