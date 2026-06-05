#!/usr/bin/env bash
set -euo pipefail

ROOT_PREFIX="${ROOT_PREFIX:-checkpoints/HFRVLA_libero_v1_generated_chunks_shard}"
OUT_REPO_PREFIX="${OUT_REPO_PREFIX:-HFRVLA_libero_v1_generated_chunks_shard}"
LOG_DIR="${LOG_DIR:-outputs/record_generated_chunks_shards}"
SRC_REPO_ID="${SRC_REPO_ID:-HuggingFaceVLA/libero}"
SRC_ROOT="${SRC_ROOT:-/home/hucenrotia/.cache/huggingface/lerobot/HuggingFaceVLA/libero}"
SMOLVLA="${SMOLVLA:-HuggingFaceVLA/smolvla_libero}"
DINO_REPO="${DINO_REPO:-checkpoints/dinov3_src}"
DINO_WEIGHTS="${DINO_WEIGHTS:-checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth}"
DINO_BATCH_SIZE="${DINO_BATCH_SIZE:-128}"
DINO_DTYPE="${DINO_DTYPE:-float16}"
DEVICE="${DEVICE:-cuda}"
HFRVLA_TMP_ROOT="${HFRVLA_TMP_ROOT:-/home/hucenrotia/tmp/hfrvla}"
SHARDS="${SHARDS:-a:0:847 b:847:1693}"

mkdir -p "${LOG_DIR}" "${HFRVLA_TMP_ROOT}/hf_datasets" "${HFRVLA_TMP_ROOT}/tmp" "${HFRVLA_TMP_ROOT}/matplotlib"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HFRVLA_CACHE_ROOT="${HFRVLA_CACHE_ROOT:-${HFRVLA_TMP_ROOT}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HFRVLA_TMP_ROOT}/hf_datasets}"
export TMPDIR="${TMPDIR:-${HFRVLA_TMP_ROOT}/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${HFRVLA_TMP_ROOT}/matplotlib}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

pids=()
names=()

for spec in ${SHARDS}; do
  IFS=: read -r name ep_from ep_to <<< "${spec}"
  out_root="${ROOT_PREFIX}_${name}"
  out_repo="${OUT_REPO_PREFIX}_${name}"
  log_path="${LOG_DIR}/record_shard_${name}.log"
  if [[ -e "${out_root}" ]]; then
    echo "[record-shards] refusing to overwrite existing root: ${out_root}" >&2
    exit 1
  fi
  echo "[record-shards] start shard ${name} episodes [${ep_from}, ${ep_to}) -> ${out_root}"
  /home/hucenrotia/Robotic_infra/lerobot/.venv/bin/python scripts/record_hfrvla_libero.py \
    --src-repo-id "${SRC_REPO_ID}" \
    --src-root "${SRC_ROOT}" \
    --out-repo-id "${out_repo}" \
    --out-root "${out_root}" \
    --source-reader parquet \
    --smolvla "${SMOLVLA}" \
    --dinov3-repo "${DINO_REPO}" \
    --dinov3-weights "${DINO_WEIGHTS}" \
    --ep-from "${ep_from}" \
    --ep-to "${ep_to}" \
    --dino-batch-size "${DINO_BATCH_SIZE}" \
    --dino-dtype "${DINO_DTYPE}" \
    --device "${DEVICE}" \
    > "${log_path}" 2>&1 &
  pids+=("$!")
  names+=("${name}")
done

status=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  name="${names[$idx]}"
  if wait "${pid}"; then
    echo "[record-shards] shard ${name} complete"
  else
    echo "[record-shards] shard ${name} failed; see ${LOG_DIR}/record_shard_${name}.log" >&2
    status=1
  fi
done

exit "${status}"
