#!/usr/bin/env bash
# Phase-2 SMACv2 comparison: ALEC / MAPPO adapter vs whole-model ES (optional fair ES extension).
# Configurable via env vars: N_AGENTS (default 4), N_ENEMIES (default 4).
# Arms:
#   A: ALEC (frozen attention_hu_actor + adapter ES) — Phase-1 ckpt
#   B: MAPPO Phase-2 (frozen backbone + adapter RL) — Phase-1 ckpt
#   C: whole-model ES — random init, WALLCLOCK seconds only (short budget)
#   D: (optional, default on) whole-model ES continuation — init from Arm C best_theta.npy,
#      wall-clock ONLY for the Phase-1 leg: direct_perf_results.json wall_clock_seconds (per seed).
#
# Budget identity (per seed): A/B each use T_phase2 (=WALLCLOCK) after offline Phase-1 (T_phase1).
#   Fair whole-model ES total wall = T_phase2 + T_phase1 = (Arm C) + (Arm D).
# Arm C spends T_phase2 from random init; Arm D spends T_phase1 starting from C's best θ, so you do
# NOT add another T_phase2 inside D — that is what "init from from-scratch ES" is for.
#
# Set ARM_D=0 to skip Arm D. Set PHASE1_WALL_OVERRIDE=1234 to use a fixed extra budget (seconds)
# for every seed instead of reading per-seed JSON.
#
# Within each arm, seeds run in parallel. Arms run sequentially (C before D).

set -uo pipefail

ROOT="/workspace/latent-evolution"
cd "$ROOT"
export PYTHONPATH="$ROOT"

N_AGENTS="${N_AGENTS:-4}"
N_ENEMIES="${N_ENEMIES:-4}"
PHASE1_DIR="${PHASE1_DIR:-results/smacv2_phase1_attn_${N_AGENTS}v${N_ENEMIES}}"
OUT_DIR="${OUT_DIR:-results/smacv2_phase2_3arm_${N_AGENTS}v${N_ENEMIES}}"
WALLCLOCK="${WALLCLOCK:-1800}"   # 30 minutes per arm per seed
SEEDS_STR="${SEEDS:-42 43 44}"
read -ra SEEDS <<< "$SEEDS_STR"

mkdir -p "$OUT_DIR"

# Arm A: ALEC
echo "=== arm A (ALEC) — parallel across seeds ==="
A_PIDS=()
for SEED in "${SEEDS[@]}"; do
  CKPT="$PHASE1_DIR/seed${SEED}/memfound_full_attention_hu_actor_latest.pt"
  [[ -f "$CKPT" ]] || { echo "[error] missing ckpt: $CKPT" >&2; exit 1; }
  SD="$OUT_DIR/seed${SEED}/arm_a_alec"
  mkdir -p "$SD"
  python3 new/memetic_foundation/scripts/run_memetic_selection_phase2.py \
    --load-path "$CKPT" --save-dir "$SD" \
    --env smacv2 --race terran --n-agents "$N_AGENTS" --n-enemies "$N_ENEMIES" \
    --device cuda --seed "$SEED" \
    --population-size 16 --generations 100000 \
    --eval-episodes 4 --final-eval-episodes 32 \
    --es-sigma 0.05 --es-lr 0.02 \
    --wallclock-seconds "$WALLCLOCK" \
    > "$SD/run.log" 2>&1 &
  A_PIDS+=("$!")
  echo "  [arm_a seed=$SEED] pid=$!"
done
echo "${A_PIDS[*]}" > "$OUT_DIR/arm_a_pids.txt"
for pid in "${A_PIDS[@]}"; do
  wait "$pid" || echo "[warn] arm_a pid=$pid exited non-zero"
done
echo "[arm A done]"

# Arm B: MAPPO Phase-2
echo "=== arm B (MAPPO Phase-2) — parallel across seeds ==="
B_PIDS=()
for SEED in "${SEEDS[@]}"; do
  CKPT="$PHASE1_DIR/seed${SEED}/memfound_full_attention_hu_actor_latest.pt"
  SD="$OUT_DIR/seed${SEED}/arm_b_mappo"
  mkdir -p "$SD"
  python3 new/memetic_foundation/scripts/run_memetic_rl_phase2.py \
    --load-path "$CKPT" --save-dir "$SD" \
    --env smacv2 --race terran --n-agents "$N_AGENTS" --n-enemies "$N_ENEMIES" \
    --device cuda --seed "$SEED" \
    --train-transitions 100000000 \
    --rollout-steps 256 \
    --eval-interval-transitions 10000 \
    --eval-episodes 16 --final-eval-episodes 32 \
    --wallclock-seconds "$WALLCLOCK" \
    > "$SD/run.log" 2>&1 &
  B_PIDS+=("$!")
  echo "  [arm_b seed=$SEED] pid=$!"
done
echo "${B_PIDS[*]}" > "$OUT_DIR/arm_b_pids.txt"
for pid in "${B_PIDS[@]}"; do
  wait "$pid" || echo "[warn] arm_b pid=$pid exited non-zero"
done
echo "[arm B done]"

# Arm C: whole-model ES (no Phase-1 ckpt)
echo "=== arm C (whole-model ES) — parallel across seeds ==="
C_PIDS=()
for SEED in "${SEEDS[@]}"; do
  SD="$OUT_DIR/seed${SEED}/arm_c_es"
  mkdir -p "$SD"
  python3 new/memetic_foundation/scripts/run_whole_model_es_phase2.py \
    --save-dir "$SD" \
    --env smacv2 --race terran --n-agents "$N_AGENTS" --n-enemies "$N_ENEMIES" \
    --device cuda --seed "$SEED" \
    --population-size 16 --max-generations 100000 \
    --eval-episodes 4 --final-eval-episodes 32 \
    --es-sigma 0.05 --es-lr 0.02 \
    --wallclock-seconds "$WALLCLOCK" \
    > "$SD/run.log" 2>&1 &
  C_PIDS+=("$!")
  echo "  [arm_c seed=$SEED] pid=$!"
done
echo "${C_PIDS[*]}" > "$OUT_DIR/arm_c_pids.txt"
for pid in "${C_PIDS[@]}"; do
  wait "$pid" || echo "[warn] arm_c pid=$pid exited non-zero"
done
echo "[arm C done]"

# Arm D: ES continuation with pretrain-matched wall-clock (init from Arm C)
if [[ "${ARM_D:-1}" != "0" ]]; then
  echo "=== arm D (whole-model ES + Phase-1 wall budget, init from arm C) — parallel across seeds ==="
  D_PIDS=()
  for SEED in "${SEEDS[@]}"; do
    THETA_C="$OUT_DIR/seed${SEED}/arm_c_es/best_theta.npy"
    [[ -f "$THETA_C" ]] || { echo "[error] missing arm C theta: $THETA_C" >&2; exit 1; }
    if [[ -n "${PHASE1_WALL_OVERRIDE:-}" ]]; then
      D_WALL="$PHASE1_WALL_OVERRIDE"
    else
      PERF_JSON="$PHASE1_DIR/seed${SEED}/direct_perf_results.json"
      [[ -f "$PERF_JSON" ]] || { echo "[error] missing phase1 perf for wall clock: $PERF_JSON" >&2; exit 1; }
      D_WALL=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['wall_clock_seconds'])" "$PERF_JSON")
    fi
    SD="$OUT_DIR/seed${SEED}/arm_d_es_pretrain_matched"
    mkdir -p "$SD"
    python3 new/memetic_foundation/scripts/run_whole_model_es_phase2.py \
      --save-dir "$SD" \
      --env smacv2 --race terran --n-agents "$N_AGENTS" --n-enemies "$N_ENEMIES" \
      --device cuda --seed "$SEED" \
      --population-size 16 --max-generations 100000 \
      --eval-episodes 4 --final-eval-episodes 32 \
      --es-sigma 0.05 --es-lr 0.02 \
      --wallclock-seconds "$D_WALL" \
      --init-theta-path "$THETA_C" \
      --arm-tag D_es_pretrain_matched \
      > "$SD/run.log" 2>&1 &
    D_PIDS+=("$!")
    echo "  [arm_d seed=$SEED] pid=$! extra_wall=${D_WALL}s"
  done
  echo "${D_PIDS[*]}" > "$OUT_DIR/arm_d_pids.txt"
  for pid in "${D_PIDS[@]}"; do
    wait "$pid" || echo "[warn] arm_d pid=$pid exited non-zero"
  done
  echo "[arm D done]"
else
  echo "[skip arm D] ARM_D=0"
fi

echo "[all arms done] results -> $OUT_DIR"
