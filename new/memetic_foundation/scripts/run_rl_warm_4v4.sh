#!/usr/bin/env bash
# Re-run RL phase-2 4v4 with warm-init (symmetry-breaking) so the meme state cell
# is not stuck at z=0. Then re-probe and re-run the RALE alignment analysis.

set -uo pipefail

ROOT="/workspace/latent-evolution"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export SC2PATH="${SC2PATH:-/workspace/StarCraftII}"

N_AGENTS="${N_AGENTS:-4}"
N_ENEMIES="${N_ENEMIES:-4}"
WALLCLOCK="${WALLCLOCK:-1800}"
WARM_STD="${WARM_STD:-0.05}"
SEEDS_STR="${SEEDS:-42 43 44}"
read -ra SEEDS <<< "$SEEDS_STR"

PHASE1_DIR="results/smacv2_phase1_attn_${N_AGENTS}v${N_ENEMIES}"
OUT_DIR="results/smacv2_phase2_3arm_${N_AGENTS}v${N_ENEMIES}"
mkdir -p "$OUT_DIR"

echo "=== Re-train RL phase-2 with warm-init (std=$WARM_STD) — parallel across seeds ==="
B_PIDS=()
for SEED in "${SEEDS[@]}"; do
  CKPT="$PHASE1_DIR/seed${SEED}/memfound_full_attention_hu_actor_latest.pt"
  [[ -f "$CKPT" ]] || { echo "[error] missing ckpt: $CKPT" >&2; exit 1; }
  SD="$OUT_DIR/seed${SEED}/arm_b_mappo_warm"
  mkdir -p "$SD"
  python3 new/memetic_foundation/scripts/run_memetic_rl_phase2.py \
    --load-path "$CKPT" --save-dir "$SD" \
    --env smacv2 --race terran --n-agents "$N_AGENTS" --n-enemies "$N_ENEMIES" \
    --device cuda --seed "$SEED" \
    --train-transitions 100000000 \
    --rollout-steps 256 \
    --eval-interval-transitions 10000 \
    --eval-episodes 16 --final-eval-episodes 32 \
    --warm-init-std "$WARM_STD" \
    --wallclock-seconds "$WALLCLOCK" \
    > "$SD/run.log" 2>&1 &
  B_PIDS+=("$!")
  echo "  [warm seed=$SEED] pid=$!"
done
echo "${B_PIDS[*]}" > "$OUT_DIR/arm_b_mappo_warm_pids.txt"
for pid in "${B_PIDS[@]}"; do
  wait "$pid" || echo "[warn] warm pid=$pid exited non-zero"
done
echo "[warm RL training done]"

echo "=== Probe warm RL adapters (z, h logging) ==="
RALE_ROOT="results/phase2_rale_alignment/smacv2_${N_AGENTS}v${N_ENEMIES}_warm/probes"
mkdir -p "$RALE_ROOT"

# Probe ALEC (re-use existing probes) and warm-RL.
for SEED in "${SEEDS[@]}"; do
  BACKBONE="$PHASE1_DIR/seed${SEED}/memfound_full_attention_hu_actor_latest.pt"
  ALEC_ADAPTER="$OUT_DIR/seed${SEED}/arm_a_alec/best_adapter.pt"
  RL_ADAPTER="$OUT_DIR/seed${SEED}/arm_b_mappo_warm/best_adapter.pt"

  for METHOD in alec rl_warm; do
    if [[ "$METHOD" == "alec" ]]; then
      ADAPTER="$ALEC_ADAPTER"
    else
      ADAPTER="$RL_ADAPTER"
    fi
    SAVE="$RALE_ROOT/$METHOD/seed${SEED}/probe.npz"
    if [[ -f "$SAVE" ]]; then
      echo "[probe] skip $METHOD seed${SEED} (exists)"
      continue
    fi
    mkdir -p "$(dirname "$SAVE")"
    echo "[probe] $METHOD seed${SEED}"
    python3 -m new.memetic_foundation.scripts.phase2_rale_probe \
      --backbone-path "$BACKBONE" \
      --adapter-path "$ADAPTER" \
      --method "$METHOD" \
      --save-path "$SAVE" \
      --env smacv2 --race terran \
      --n-agents "$N_AGENTS" --n-enemies "$N_ENEMIES" \
      --episode-steps 200 \
      --episodes 25 \
      --seed "$SEED" \
      --device cuda
  done
done

echo "=== Analyze (z primary) ==="
python3 -m new.memetic_foundation.scripts.analyze_rale_alignment \
  --probe-root "$RALE_ROOT" \
  --methods alec rl_warm \
  --seeds ${SEEDS[*]} \
  --output-dir "results/phase2_rale_alignment/smacv2_${N_AGENTS}v${N_ENEMIES}_warm/analysis_z" \
  --feature z --k-list 8 12 16 --horizon 10 --target future

echo "=== Analyze (h control) ==="
python3 -m new.memetic_foundation.scripts.analyze_rale_alignment \
  --probe-root "$RALE_ROOT" \
  --methods alec rl_warm \
  --seeds ${SEEDS[*]} \
  --output-dir "results/phase2_rale_alignment/smacv2_${N_AGENTS}v${N_ENEMIES}_warm/analysis_h" \
  --feature h --k-list 8 12 16 --horizon 10 --target future

echo "[done] outputs in results/phase2_rale_alignment/smacv2_${N_AGENTS}v${N_ENEMIES}_warm"
