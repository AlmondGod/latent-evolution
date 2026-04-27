#!/bin/bash
# Clean head-to-head: single env vs vec=N on 20k-step baseline, same seed + config.
set -u
cd "$(dirname "$0")/../../.."
BASE="results/lever2_bench"
mkdir -p "$BASE"

run_one() {
  local label=$1
  local num_envs=$2
  local save="$BASE/${label}"
  mkdir -p "$save"
  echo "[start] $label  num_envs=$num_envs"
  python3 -m new.memetic_foundation.run \
    --mode train --env smacv2 \
    --race terran --n-units 4 --n-enemies 4 \
    --total-steps 20000 --rollout-steps 400 --num-envs "$num_envs" \
    --no-memory --no-comm \
    --seed 123 --log-interval 2 --save-interval 1000 \
    --save-dir "$save" \
    > "$BASE/${label}.log" 2>&1
  local results=$(ls "$save"/memfound_baseline_*/memfound_baseline_results.json 2>/dev/null | head -1)
  python3 -c "
import json
d = json.load(open('$results'))
s, w = d['total_steps'], d['wall_clock_seconds']
print(f'[done]  $label : {s} steps in {w:.1f}s -> {s/w:.1f} env-steps/s')"
}

run_one single_env 1
run_one vec4       4
run_one vec8       8
run_one vec16      16
