#!/bin/bash
set -e
cd "$(dirname "$0")/.."

PYTHON=/opt/homebrew/bin/python3.9
OUTDIR=checkpoints/spread_n5_suite
mkdir -p "$OUTDIR"

for seed in 1 2 3 4 5; do
  $PYTHON -m new.memetic_foundation.run \
    --mode train --env mpe --mpe-scenario simple_spread_v2 \
    --n-adversaries 5 --total-steps 400000 --eval-episodes 20 \
    --no-memory --no-comm --seed $seed \
    --save-dir $OUTDIR/baseline_seed${seed} \
    > $OUTDIR/baseline_seed${seed}.log 2>&1 &
done
echo "baseline launched (pids)"

for seed in 1 2 3 4 5; do
  $PYTHON -m new.memetic_foundation.run \
    --mode train --env mpe --mpe-scenario simple_spread_v2 \
    --n-adversaries 5 --total-steps 400000 --eval-episodes 20 \
    --no-comm --seed $seed \
    --save-dir $OUTDIR/memory_only_seed${seed} \
    > $OUTDIR/memory_only_seed${seed}.log 2>&1 &
done
echo "memory_only launched"

for seed in 1 2 3 4 5; do
  $PYTHON -m new.memetic_foundation.run \
    --mode train --env mpe --mpe-scenario simple_spread_v2 \
    --n-adversaries 5 --total-steps 400000 --eval-episodes 20 \
    --comm-mode commnet --seed $seed \
    --save-dir $OUTDIR/commnet_seed${seed} \
    > $OUTDIR/commnet_seed${seed}.log 2>&1 &
done
echo "commnet launched"

for seed in 1 2 3 4 5; do
  $PYTHON -m new.memetic_foundation.run \
    --mode train --env mpe --mpe-scenario simple_spread_v2 \
    --n-adversaries 5 --total-steps 400000 --eval-episodes 20 \
    --comm-mode commnet_u --seed $seed \
    --save-dir $OUTDIR/commnet_u_seed${seed} \
    > $OUTDIR/commnet_u_seed${seed}.log 2>&1 &
done
echo "commnet_u launched"

for seed in 1 2 3 4 5; do
  $PYTHON -m new.memetic_foundation.run \
    --mode train --env mpe --mpe-scenario simple_spread_v2 \
    --n-adversaries 5 --total-steps 400000 --eval-episodes 20 \
    --comm-mode attention_u --seed $seed \
    --save-dir $OUTDIR/attention_u_seed${seed} \
    > $OUTDIR/attention_u_seed${seed}.log 2>&1 &
done
echo "attention_u launched"

echo "Total processes: $(ps aux | grep memetic_foundation | grep -v grep | wc -l)"
