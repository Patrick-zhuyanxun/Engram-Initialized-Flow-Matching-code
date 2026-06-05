#!/usr/bin/env bash
set -euo pipefail

# Short HFRVLA batch-size throughput probe.
#
# This is intentionally separate from the LR/WD sweep so probe runs do not
# pollute the formal W&B comparison. It records GPU utilization and memory while
# running a small number of training steps for each batch size.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

BATCH_VALUES_RAW="${BATCH_VALUES:-512 1024 2048}"
BATCH_VALUES_RAW="${BATCH_VALUES_RAW//,/ }"
read -r -a BATCH_VALUES_ARRAY <<< "$BATCH_VALUES_RAW"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_PREFIX="${RUN_PREFIX:-hfrvla_batchprobe}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/checkpoints}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/outputs/hfrvla_batch_probe/$RUN_STAMP}"
PROBE_STEPS="${PROBE_STEPS:-2000}"
PROBE_WARMUP_STEPS="${PROBE_WARMUP_STEPS:-100}"
PROBE_SAVE_FREQ="${PROBE_SAVE_FREQ:-999999}"
DRY_RUN="${DRY_RUN:-false}"

mkdir -p "$LOG_ROOT"

export WANDB_ENABLE=false
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export HFRVLA_TMP_ROOT="${HFRVLA_TMP_ROOT:-$HOME/tmp/hfrvla}"
export HFRVLA_DATASET_BACKEND="${HFRVLA_DATASET_BACKEND:-fastcache}"
export HFRVLA_FASTCACHE_ROOT="${HFRVLA_FASTCACHE_ROOT:-checkpoints/HFRVLA_libero_v1_fastcache_v2}"
export RESIDUAL_MERGE_MODE="${RESIDUAL_MERGE_MODE:-fast_wrist}"
export FAST_RESIDUAL_ALPHA="${FAST_RESIDUAL_ALPHA:-${A2C2_ALPHA:-1.0}}"
export FAST_RESIDUAL_USE_LATENT_CONTEXT="${FAST_RESIDUAL_USE_LATENT_CONTEXT:-${A2C2_USE_LATENT_CONTEXT:-true}}"
export SEQ_LEN="${SEQ_LEN:-2}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export STEPS="$PROBE_STEPS"
export WARMUP_STEPS="$PROBE_WARMUP_STEPS"
export JOINT_STEPS="${JOINT_STEPS:-$((PROBE_STEPS - PROBE_WARMUP_STEPS))}"
export REFINE_STEPS=0
export SAVE_FREQ="$PROBE_SAVE_FREQ"
export LOG_FREQ="${LOG_FREQ:-100}"
export LR="${LR:-3e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
export SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-$LR}"
export SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-$STEPS}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
export DEVICE="${DEVICE:-cuda}"

cat <<EOF
[hfrvla-batch-probe] batch_values=${BATCH_VALUES_ARRAY[*]}
[hfrvla-batch-probe] steps=$STEPS warmup=$WARMUP_STEPS joint=$JOINT_STEPS save_freq=$SAVE_FREQ
[hfrvla-batch-probe] log_root=$LOG_ROOT
[hfrvla-batch-probe] wandb_enable=$WANDB_ENABLE dataset_backend=$HFRVLA_DATASET_BACKEND
EOF

for batch_size in "${BATCH_VALUES_ARRAY[@]}"; do
  run_name="${RUN_PREFIX}_b${batch_size}_${PROBE_STEPS}s_${RUN_STAMP}"
  out_dir="$OUT_ROOT/$run_name"
  train_log="$LOG_ROOT/${run_name}.log"
  gpu_log="$LOG_ROOT/${run_name}_gpu.csv"

  echo "[hfrvla-batch-probe] run_name=$run_name batch_size=$batch_size"
  echo "[hfrvla-batch-probe] out_dir=$out_dir"
  echo "[hfrvla-batch-probe] train_log=$train_log"
  echo "[hfrvla-batch-probe] gpu_log=$gpu_log"

  if [[ -e "$out_dir" && "${RESUME:-false}" != "true" ]]; then
    echo "[hfrvla-batch-probe] output directory already exists: $out_dir" >&2
    exit 1
  fi

  if is_true "$DRY_RUN"; then
    continue
  fi

  nvidia-smi \
    --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw \
    --format=csv \
    -l 2 \
    -f "$gpu_log" >/dev/null 2>&1 &
  monitor_pid=$!

  set +e
  (
    export RUN_NAME="$run_name"
    export OUT_DIR="$out_dir"
    export JOB_NAME="$run_name"
    export BATCH_SIZE="$batch_size"
    "$PROJECT_ROOT/scripts/run_hfrvla_training_foreground.sh"
  ) 2>&1 | tee "$train_log"
  status=${PIPESTATUS[0]}
  set -e

  kill "$monitor_pid" >/dev/null 2>&1 || true
  wait "$monitor_pid" >/dev/null 2>&1 || true

  if (( status != 0 )); then
    echo "[hfrvla-batch-probe] failed: $run_name status=$status" >&2
    exit "$status"
  fi
done

echo "[hfrvla-batch-probe] done"
