#!/usr/bin/env bash
# Parallel precompute driver: splits LIBERO episodes across N processes
# so the GPU stays saturated.
#
# Usage:
#   bash scripts/precompute_libero_parallel.sh        # default: 4 workers
#   N_WORKERS=6 scripts/precompute_libero_parallel.sh # override
#   N_WORKERS=8 CPU_THREADS_PER_WORKER=4 bash scripts/precompute_libero_parallel.sh
#
# Each worker handles a contiguous range of episodes (--ep-from / --ep-to).
# Existing episode_*.pt files are skipped, so this can resume an interrupted
# Sprint 1 run. After all workers finish, one DINOv3 pass attaches patches.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_SRC="${REPO_ROOT}/policy/lerobot_policy_hfrvla/src"
LEROBOT_ROOT="${LEROBOT_ROOT:-${HOME}/Robotic_infra/lerobot}"
PYTHON_BIN="${PYTHON_BIN:-${LEROBOT_ROOT}/.venv/bin/python}"
N_WORKERS="${N_WORKERS:-4}"
CPU_THREADS_PER_WORKER="${CPU_THREADS_PER_WORKER:-4}"
OUT_DIR="${REPO_ROOT}/checkpoints/libero_chunks_full"
SMOLVLA="${SMOLVLA:-lerobot/smolvla_base}"
DINOV3_REPO="${REPO_ROOT}/checkpoints/dinov3_src"
DINOV3_WEIGHTS="${REPO_ROOT}/checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DINOV3_ARCH="${DINOV3_ARCH:-dinov3_vits16}"
DINOV3_BATCH_SIZE="${DINOV3_BATCH_SIZE:-256}"
TOTAL_EPISODES="${TOTAL_EPISODES:-1693}"
LOG_DIR="${REPO_ROOT}/outputs/precompute_logs"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
export PYTHONPATH="${POLICY_SRC}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CPU_THREADS_PER_WORKER}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${CPU_THREADS_PER_WORKER}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${CPU_THREADS_PER_WORKER}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${CPU_THREADS_PER_WORKER}}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-${CPU_THREADS_PER_WORKER}}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-${CPU_THREADS_PER_WORKER}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    echo "Set LEROBOT_ROOT or PYTHON_BIN to the uv-managed LeRobot environment." >&2
    exit 1
fi

PER_WORKER=$(( (TOTAL_EPISODES + N_WORKERS - 1) / N_WORKERS ))
echo "Spawning ${N_WORKERS} workers (~${PER_WORKER} episodes each) for ${TOTAL_EPISODES} total."
echo "Output dir: ${OUT_DIR}"
echo "Logs:       ${LOG_DIR}/worker_*.log"
echo "Python:     ${PYTHON_BIN}"
echo "CPU threads per worker: ${CPU_THREADS_PER_WORKER}"

PIDS=()
for ((i=0; i<N_WORKERS; i++)); do
    FROM=$(( i * PER_WORKER ))
    TO=$(( FROM + PER_WORKER ))
    if (( TO > TOTAL_EPISODES )); then
        TO=${TOTAL_EPISODES}
    fi
    if (( FROM >= TO )); then
        continue
    fi
    LOG="${LOG_DIR}/worker_${i}.log"
    echo "  worker ${i}: episodes [${FROM}, ${TO})  -> ${LOG}"
    PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${REPO_ROOT}/scripts/precompute_libero.py" \
        --repo-id HuggingFaceVLA/libero \
        --out-dir "${OUT_DIR}" \
        --smolvla "${SMOLVLA}" \
        --dinov3-repo "${DINOV3_REPO}" \
        --dinov3-weights "${DINOV3_WEIGHTS}" \
        --dinov3-arch "${DINOV3_ARCH}" \
        --ep-from "${FROM}" \
        --ep-to "${TO}" \
        --skip-existing \
        --skip-dinov3 \
        --device "${DEVICE}" \
        > "${LOG}" 2>&1 &
    PIDS+=($!)
done

echo
echo "PIDs: ${PIDS[*]}"
echo "Waiting for all workers..."
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        echo "  worker pid ${pid} FAILED"
        FAIL=$((FAIL+1))
    fi
done

if (( FAIL > 0 )); then
    echo "${FAIL} worker(s) failed; check ${LOG_DIR}"
    exit 1
fi
echo "All rollout workers finished."

# Single-process DINOv3 pass over the merged output dir.
echo "Running DINOv3 feature precomputation over all episodes..."
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${REPO_ROOT}/scripts/precompute_libero.py" \
    --out-dir "${OUT_DIR}" \
    --dinov3-repo "${DINOV3_REPO}" \
    --dinov3-weights "${DINOV3_WEIGHTS}" \
    --dinov3-arch "${DINOV3_ARCH}" \
    --dinov3-batch-size "${DINOV3_BATCH_SIZE}" \
    --dinov3-only \
    --device "${DEVICE}"
echo "Done."
