#!/bin/bash
set -e
cd /Users/almondgod/Repositories/memeplex-capstone

OUTDIR="checkpoints/spread_gate_test"
mkdir -p "$OUTDIR"

PYTHON="/opt/homebrew/Cellar/python@3.9/3.9.19_1/Frameworks/Python.framework/Versions/3.9/bin/python3.9"

for seed in 1 2 3 4 5; do
  $PYTHON -m new.memetic_foundation.run --mode train \
    --env mpe --mpe-scenario simple_spread_v2 --n-adversaries 10 \
    --total-steps 400000 --eval-episodes 20 \
    --comm-mode commnet_u --seed $seed \
    --save-dir "$OUTDIR" \
    > "$OUTDIR/commnet_u_gate_n10_seed${seed}.log" 2>&1 &
  echo "commnet_u_gate seed$seed PID=$!"
done

for seed in 1 2 3 4 5; do
  $PYTHON -m new.memetic_foundation.run --mode train \
    --env mpe --mpe-scenario simple_spread_v2 --n-adversaries 10 \
    --total-steps 400000 --eval-episodes 20 \
    --comm-mode attention_u --seed $seed \
    --save-dir "$OUTDIR" \
    > "$OUTDIR/attention_u_gate_n10_seed${seed}.log" 2>&1 &
  echo "attention_u_gate seed$seed PID=$!"
done

wait
echo "All runs done."
