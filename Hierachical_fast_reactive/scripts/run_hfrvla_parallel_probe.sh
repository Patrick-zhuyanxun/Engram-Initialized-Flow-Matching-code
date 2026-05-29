#!/usr/bin/env bash
set -euo pipefail

# Short probe for running multiple HFRVLA 512-batch trainings concurrently.
# This uses the LR/WD sweep launcher with PARALLEL_JOBS=2 and W&B disabled.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/outputs/hfrvla_parallel_probe/$RUN_STAMP}"
GPU_LOG="$LOG_ROOT/gpu.csv"
PROBE_STEPS="${PROBE_STEPS:-1000}"
PROBE_WARMUP_STEPS="${PROBE_WARMUP_STEPS:-100}"

mkdir -p "$LOG_ROOT"

cleanup_monitor() {
  if [[ -n "${monitor_pid:-}" ]]; then
    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" >/dev/null 2>&1 || true
  fi
}

trap 'cleanup_monitor; exit 130' INT TERM

nvidia-smi \
  --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw \
  --format=csv \
  -l 2 \
  -f "$GPU_LOG" >/dev/null 2>&1 &
monitor_pid=$!

set +e
(
  export LR_VALUES="${LR_VALUES:-3e-4}"
  export WD_VALUES="${WD_VALUES:-1e-5}"
  export RUN_PREFIX="${RUN_PREFIX:-hfrvla_parallelprobe}"
  export RUN_SUFFIX="${RUN_SUFFIX:-b512_${PROBE_STEPS}s_$RUN_STAMP}"
  export OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/checkpoints}"
  export MAX_RUNS=2
  export PARALLEL_JOBS=2
  export SWEEP_LOG_ROOT="$LOG_ROOT"

  export WANDB_ENABLE=false
  export WANDB_MODE=disabled
  export WANDB_DISABLED=true

  export BATCH_SIZE=512
  export STEPS="$PROBE_STEPS"
  export WARMUP_STEPS="$PROBE_WARMUP_STEPS"
  export JOINT_STEPS="$((PROBE_STEPS - PROBE_WARMUP_STEPS))"
  export REFINE_STEPS=0
  export SAVE_FREQ=999999
  export LOG_FREQ=100

  export HFRVLA_DATASET_BACKEND="${HFRVLA_DATASET_BACKEND:-fastcache}"
  export HFRVLA_FASTCACHE_ROOT="${HFRVLA_FASTCACHE_ROOT:-checkpoints/HFRVLA_libero_v1_fastcache_v2}"
  export RESIDUAL_MERGE_MODE="${RESIDUAL_MERGE_MODE:-a2c2}"
  export A2C2_ALPHA="${A2C2_ALPHA:-1.0}"
  export A2C2_USE_LATENT_CONTEXT="${A2C2_USE_LATENT_CONTEXT:-true}"
  export SEQ_LEN="${SEQ_LEN:-2}"

  "$PROJECT_ROOT/scripts/run_hfrvla_lr_wd_sweep.sh"
) 2>&1 | tee "$LOG_ROOT/parallel_probe.log"
status=${PIPESTATUS[0]}
set -e

cleanup_monitor

if (( status != 0 )); then
  echo "[hfrvla-parallel-probe] failed status=$status log_root=$LOG_ROOT" >&2
  exit "$status"
fi

echo "[hfrvla-parallel-probe] done log_root=$LOG_ROOT gpu_log=$GPU_LOG"
