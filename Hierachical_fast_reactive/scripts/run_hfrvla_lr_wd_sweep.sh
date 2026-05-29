#!/usr/bin/env bash
set -euo pipefail

# Launcher for the HFRVLA learning-rate / weight-decay sweep.
#
# Defaults implement the 20-run W&B sweep:
#   LR in {1e-5, 3e-5, 1e-4, 3e-4, 5e-4}
#   WD in {0, 1e-5, 1e-4, 3e-4}
#
# Useful controls:
#   DRY_RUN=true        Print commands without launching training.
#   START_AT=<run>      Skip rows until this run name is reached.
#   MAX_RUNS=N          Launch at most N runs after START_AT filtering.
#   KEEP_GOING=true     Continue after a failed run.
#   SKIP_COMPLETED=true Skip rows with checkpoints/last/pretrained_model/model.safetensors.
#   PARALLEL_JOBS=N     Run N training jobs at a time. Logs go to SWEEP_LOG_ROOT.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

name_token() {
  local value="$1"
  value="${value//./p}"
  value="${value//e-/e_}"
  value="${value//-/_}"
  printf '%s' "$value"
}

print_assignment() {
  printf '%s=%q ' "$1" "$2"
}

LR_VALUES_RAW="${LR_VALUES:-1e-5 3e-5 1e-4 3e-4 5e-4}"
WD_VALUES_RAW="${WD_VALUES:-0 1e-5 1e-4 3e-4}"
LR_VALUES_RAW="${LR_VALUES_RAW//,/ }"
WD_VALUES_RAW="${WD_VALUES_RAW//,/ }"
read -r -a LR_VALUES_ARRAY <<< "$LR_VALUES_RAW"
read -r -a WD_VALUES_ARRAY <<< "$WD_VALUES_RAW"

RUN_PREFIX="${RUN_PREFIX:-hfrvla}"
RUN_SUFFIX="${RUN_SUFFIX:-b512_50k}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/checkpoints}"
DRY_RUN="${DRY_RUN:-false}"
START_AT="${START_AT:-}"
MAX_RUNS="${MAX_RUNS:-0}"
KEEP_GOING="${KEEP_GOING:-false}"
SKIP_COMPLETED="${SKIP_COMPLETED:-true}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
SWEEP_LOG_ROOT="${SWEEP_LOG_ROOT:-$PROJECT_ROOT/outputs/hfrvla_lr_wd_sweep_logs/$(date +%Y%m%d_%H%M%S)}"

if ! [[ "$PARALLEL_JOBS" =~ ^[0-9]+$ ]] || (( PARALLEL_JOBS < 1 )); then
  echo "[hfrvla-lr-wd-sweep] PARALLEL_JOBS must be a positive integer: $PARALLEL_JOBS" >&2
  exit 1
fi

export WANDB_ENABLE="${WANDB_ENABLE:-true}"
export WANDB_PROJECT="${WANDB_PROJECT:-hfrvla}"
if is_true "$WANDB_ENABLE"; then
  unset WANDB_MODE
  unset WANDB_DISABLED
fi

export HFRVLA_TMP_ROOT="${HFRVLA_TMP_ROOT:-$HOME/tmp/hfrvla}"
export HFRVLA_DATASET_BACKEND="${HFRVLA_DATASET_BACKEND:-fastcache}"
export HFRVLA_FASTCACHE_ROOT="${HFRVLA_FASTCACHE_ROOT:-checkpoints/HFRVLA_libero_v1_fastcache_v2}"

export RESIDUAL_MERGE_MODE="${RESIDUAL_MERGE_MODE:-a2c2}"
export A2C2_ALPHA="${A2C2_ALPHA:-1.0}"
export A2C2_USE_LATENT_CONTEXT="${A2C2_USE_LATENT_CONTEXT:-true}"
export SEQ_LEN="${SEQ_LEN:-2}"

export BATCH_SIZE="${BATCH_SIZE:-512}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export STEPS="${STEPS:-50000}"
export WARMUP_STEPS="${WARMUP_STEPS:-1000}"
export JOINT_STEPS="${JOINT_STEPS:-49000}"
export REFINE_STEPS="${REFINE_STEPS:-0}"
export SAVE_FREQ="${SAVE_FREQ:-25000}"
export LOG_FREQ="${LOG_FREQ:-100}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
export DEVICE="${DEVICE:-cuda}"
export SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-$STEPS}"

total_runs=$((${#LR_VALUES_ARRAY[@]} * ${#WD_VALUES_ARRAY[@]}))
run_index=0
launched_runs=0
start_seen=false
active_pids=()
active_names=()
active_logs=()
if [[ -z "$START_AT" ]]; then
  start_seen=true
fi

cleanup_active() {
  local pid
  for pid in "${active_pids[@]}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
}

trap 'cleanup_active; exit 130' INT TERM

wait_active_group() {
  local failed=0
  local i name pid run_log status

  for i in "${!active_pids[@]}"; do
    pid="${active_pids[$i]}"
    name="${active_names[$i]}"
    run_log="${active_logs[$i]}"
    if wait "$pid"; then
      echo "[hfrvla-lr-wd-sweep] completed: $name log=$run_log"
    else
      status=$?
      echo "[hfrvla-lr-wd-sweep] failed: $name status=$status log=$run_log" >&2
      failed=1
    fi
  done

  active_pids=()
  active_names=()
  active_logs=()

  if (( failed != 0 )); then
    if is_true "$KEEP_GOING"; then
      echo "[hfrvla-lr-wd-sweep] continuing after failed parallel group because KEEP_GOING=true" >&2
    else
      exit 1
    fi
  fi
}

cat <<EOF
[hfrvla-lr-wd-sweep] total_runs=$total_runs
[hfrvla-lr-wd-sweep] lr_values=${LR_VALUES_ARRAY[*]}
[hfrvla-lr-wd-sweep] wd_values=${WD_VALUES_ARRAY[*]}
[hfrvla-lr-wd-sweep] wandb_enable=$WANDB_ENABLE wandb_project=$WANDB_PROJECT
[hfrvla-lr-wd-sweep] steps=$STEPS batch_size=$BATCH_SIZE save_freq=$SAVE_FREQ warmup=$WARMUP_STEPS joint=$JOINT_STEPS refine=$REFINE_STEPS
[hfrvla-lr-wd-sweep] dry_run=$DRY_RUN start_at=${START_AT:-<none>} max_runs=$MAX_RUNS keep_going=$KEEP_GOING skip_completed=$SKIP_COMPLETED parallel_jobs=$PARALLEL_JOBS
EOF

if ! is_true "$DRY_RUN" && (( PARALLEL_JOBS > 1 )); then
  mkdir -p "$SWEEP_LOG_ROOT"
  echo "[hfrvla-lr-wd-sweep] sweep_log_root=$SWEEP_LOG_ROOT"
fi

for lr in "${LR_VALUES_ARRAY[@]}"; do
  for wd in "${WD_VALUES_ARRAY[@]}"; do
    run_index=$((run_index + 1))
    run_name="${RUN_PREFIX}_lr$(name_token "$lr")_wd$(name_token "$wd")_${RUN_SUFFIX}"
    out_dir="$OUT_ROOT/$run_name"
    complete_marker="$out_dir/checkpoints/last/pretrained_model/model.safetensors"

    if [[ "$start_seen" == "false" ]]; then
      if [[ "$run_name" == "$START_AT" ]]; then
        start_seen=true
      else
        echo "[hfrvla-lr-wd-sweep] skip before START_AT: $run_name"
        continue
      fi
    fi

    if (( MAX_RUNS > 0 && launched_runs >= MAX_RUNS )); then
      echo "[hfrvla-lr-wd-sweep] reached MAX_RUNS=$MAX_RUNS"
      exit 0
    fi

    if ! is_true "$DRY_RUN" && [[ -e "$out_dir" && "${RESUME:-false}" != "true" ]]; then
      if [[ -f "$complete_marker" ]] && is_true "$SKIP_COMPLETED"; then
        echo "[hfrvla-lr-wd-sweep] skip completed: $run_name"
        continue
      fi
      cat >&2 <<EOF
[hfrvla-lr-wd-sweep] output directory already exists and is not a completed checkpoint:
  $out_dir

Set START_AT to a later run, set RESUME=true for a valid resumable run, or
choose a different RUN_PREFIX/OUT_ROOT.
EOF
      exit 1
    fi

    echo "[hfrvla-lr-wd-sweep] [$run_index/$total_runs] run_name=$run_name lr=$lr weight_decay=$wd"

    if is_true "$DRY_RUN"; then
      printf '+ '
      print_assignment RUN_NAME "$run_name"
      print_assignment OUT_DIR "$out_dir"
      print_assignment WANDB_ENABLE "$WANDB_ENABLE"
      print_assignment WANDB_PROJECT "$WANDB_PROJECT"
      print_assignment HFRVLA_TMP_ROOT "$HFRVLA_TMP_ROOT"
      print_assignment HFRVLA_DATASET_BACKEND "$HFRVLA_DATASET_BACKEND"
      print_assignment HFRVLA_FASTCACHE_ROOT "$HFRVLA_FASTCACHE_ROOT"
      print_assignment RESIDUAL_MERGE_MODE "$RESIDUAL_MERGE_MODE"
      print_assignment A2C2_ALPHA "$A2C2_ALPHA"
      print_assignment A2C2_USE_LATENT_CONTEXT "$A2C2_USE_LATENT_CONTEXT"
      print_assignment SEQ_LEN "$SEQ_LEN"
      print_assignment BATCH_SIZE "$BATCH_SIZE"
      print_assignment NUM_WORKERS "$NUM_WORKERS"
      print_assignment STEPS "$STEPS"
      print_assignment WARMUP_STEPS "$WARMUP_STEPS"
      print_assignment JOINT_STEPS "$JOINT_STEPS"
      print_assignment REFINE_STEPS "$REFINE_STEPS"
      print_assignment SAVE_FREQ "$SAVE_FREQ"
      print_assignment LOG_FREQ "$LOG_FREQ"
      print_assignment LR "$lr"
      print_assignment WEIGHT_DECAY "$wd"
      print_assignment SCHEDULER_DECAY_LR "$lr"
      print_assignment SCHEDULER_DECAY_STEPS "$SCHEDULER_DECAY_STEPS"
      print_assignment GRAD_CLIP_NORM "$GRAD_CLIP_NORM"
      print_assignment DEVICE "$DEVICE"
      printf 'scripts/run_hfrvla_training_foreground.sh\n'
      launched_runs=$((launched_runs + 1))
      continue
    fi

    if (( PARALLEL_JOBS > 1 )); then
      run_log="$SWEEP_LOG_ROOT/${run_name}.log"
      echo "[hfrvla-lr-wd-sweep] launch parallel: $run_name log=$run_log"
      (
        export RUN_NAME="$run_name"
        export OUT_DIR="$out_dir"
        export LR="$lr"
        export WEIGHT_DECAY="$wd"
        export SCHEDULER_DECAY_LR="$lr"
        "$PROJECT_ROOT/scripts/run_hfrvla_training_foreground.sh"
      ) >"$run_log" 2>&1 &
      active_pids+=("$!")
      active_names+=("$run_name")
      active_logs+=("$run_log")
      launched_runs=$((launched_runs + 1))

      if (( ${#active_pids[@]} >= PARALLEL_JOBS )); then
        wait_active_group
      fi
    else
      if (
        export RUN_NAME="$run_name"
        export OUT_DIR="$out_dir"
        export LR="$lr"
        export WEIGHT_DECAY="$wd"
        export SCHEDULER_DECAY_LR="$lr"
        "$PROJECT_ROOT/scripts/run_hfrvla_training_foreground.sh"
      ); then
        launched_runs=$((launched_runs + 1))
      elif is_true "$KEEP_GOING"; then
        echo "[hfrvla-lr-wd-sweep] failed but KEEP_GOING=true: $run_name" >&2
        launched_runs=$((launched_runs + 1))
        continue
      else
        echo "[hfrvla-lr-wd-sweep] failed: $run_name" >&2
        exit 1
      fi
    fi
  done
done

if (( ${#active_pids[@]} > 0 )); then
  wait_active_group
fi

echo "[hfrvla-lr-wd-sweep] done launched_or_dry_run=$launched_runs"
