#!/usr/bin/env bash
# RALE meme-level alignment pipeline on SMACv2 4v4 Terran.
# 1) Probe per-step latents for ALEC and RL phase-2 adapters across seeds.
# 2) Analyze with shared k-means clustering, K-sweep, bootstrap, shuffle null.

set -euo pipefail

cd "$(dirname "$0")/../../.."

export SC2PATH="${SC2PATH:-/workspace/StarCraftII}"

N_AGENTS="${N_AGENTS:-4}"
N_ENEMIES="${N_ENEMIES:-4}"
RACE="${RACE:-terran}"
SEEDS="${SEEDS:-42 43 44}"
EPISODES="${EPISODES:-25}"
EPISODE_STEPS="${EPISODE_STEPS:-200}"
HORIZON="${HORIZON:-10}"
DEVICE="${DEVICE:-cuda}"
K_LIST="${K_LIST:-8 12 16}"

PHASE1_DIR="${PHASE1_DIR:-results/smacv2_phase1_attn_${N_AGENTS}v${N_ENEMIES}}"
PHASE2_DIR="${PHASE2_DIR:-results/smacv2_phase2_3arm_${N_AGENTS}v${N_ENEMIES}}"
OUT_ROOT="${OUT_ROOT:-results/phase2_rale_alignment/smacv2_${N_AGENTS}v${N_ENEMIES}}"

PROBE_ROOT="${OUT_ROOT}/probes"
mkdir -p "${PROBE_ROOT}"

echo "[rale] Phase 1) probe ALEC and RL phase-2 adapters"
for SEED in ${SEEDS}; do
  BACKBONE="${PHASE1_DIR}/seed${SEED}/memfound_full_attention_hu_actor_latest.pt"
  ALEC_ADAPTER="${PHASE2_DIR}/seed${SEED}/arm_a_alec/best_adapter.pt"
  RL_ADAPTER="${PHASE2_DIR}/seed${SEED}/arm_b_mappo/best_adapter.pt"

  for METHOD in alec rl; do
    if [[ "${METHOD}" == "alec" ]]; then
      ADAPTER="${ALEC_ADAPTER}"
    else
      ADAPTER="${RL_ADAPTER}"
    fi
    SAVE="${PROBE_ROOT}/${METHOD}/seed${SEED}/probe.npz"
    if [[ -f "${SAVE}" ]]; then
      echo "[rale] skip ${METHOD} seed${SEED} (already exists)"
      continue
    fi
    echo "[rale] probe ${METHOD} seed${SEED}"
    python3 -m new.memetic_foundation.scripts.phase2_rale_probe \
      --backbone-path "${BACKBONE}" \
      --adapter-path "${ADAPTER}" \
      --method "${METHOD}" \
      --save-path "${SAVE}" \
      --env smacv2 --race "${RACE}" \
      --n-agents "${N_AGENTS}" --n-enemies "${N_ENEMIES}" \
      --episode-steps "${EPISODE_STEPS}" \
      --episodes "${EPISODES}" \
      --seed "${SEED}" \
      --device "${DEVICE}"
  done
done

echo "[rale] Phase 2) analyze (z primary)"
python3 -m new.memetic_foundation.scripts.analyze_rale_alignment \
  --probe-root "${PROBE_ROOT}" \
  --methods alec rl \
  --seeds ${SEEDS} \
  --output-dir "${OUT_ROOT}/analysis_z" \
  --feature z \
  --k-list ${K_LIST} \
  --horizon "${HORIZON}" \
  --target future

echo "[rale] Phase 3) h-control analysis"
python3 -m new.memetic_foundation.scripts.analyze_rale_alignment \
  --probe-root "${PROBE_ROOT}" \
  --methods alec rl \
  --seeds ${SEEDS} \
  --output-dir "${OUT_ROOT}/analysis_h" \
  --feature h \
  --k-list ${K_LIST} \
  --horizon "${HORIZON}" \
  --target future

echo "[rale] done. Outputs in ${OUT_ROOT}"
