#!/usr/bin/env bash
set -euo pipefail

# Foreground HFRVLA training launcher.
#
# Run this directly when you want to see the LeRobot training progress in the
# current terminal. This script does not use nohup, systemd-run, background jobs,
# or log tailing.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_NAME="${RUN_NAME:-hfrvla_run_fg_$(date +%Y%m%d_%H%M%S)}"

export OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/checkpoints/$RUN_NAME}"
export JOB_NAME="${JOB_NAME:-$RUN_NAME}"
export DATASET_REPO_ID="${DATASET_REPO_ID:-HFRVLA_libero_v1}"
export DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/checkpoints/HFRVLA_libero_v1_merged_reindexed}"

export HFRVLA_TMP_ROOT="${HFRVLA_TMP_ROOT:-$HOME/tmp/hfrvla}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HFRVLA_TMP_ROOT/hf_datasets}"
export TMPDIR="${TMPDIR:-$HFRVLA_TMP_ROOT/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_ENABLE="${WANDB_ENABLE:-true}"

export DEVICE="${DEVICE:-cuda}"
export STEPS="${STEPS:-60000}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export SAVE_FREQ="${SAVE_FREQ:-5000}"
export LOG_FREQ="${LOG_FREQ:-50}"
export SEQ_LEN="${SEQ_LEN:-4}"

export HFRVLA_DATASET_BACKEND="${HFRVLA_DATASET_BACKEND:-lerobot}"
export HFRVLA_FASTCACHE_ROOT="${HFRVLA_FASTCACHE_ROOT:-$PROJECT_ROOT/checkpoints/HFRVLA_libero_v1_fastcache_seq${SEQ_LEN}}"

export WARMUP_STEPS="${WARMUP_STEPS:-1000}"
export JOINT_STEPS="${JOINT_STEPS:-49000}"
export REFINE_STEPS="${REFINE_STEPS:-10000}"

export LR="${LR:-3e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

mkdir -p "$HFRVLA_TMP_ROOT" "$HF_DATASETS_CACHE" "$TMPDIR"

cat <<EOF
[hfrvla-foreground] run_name=$RUN_NAME
[hfrvla-foreground] out_dir=$OUT_DIR
[hfrvla-foreground] dataset_root=$DATASET_ROOT
[hfrvla-foreground] dataset_backend=$HFRVLA_DATASET_BACKEND
[hfrvla-foreground] fastcache_root=$HFRVLA_FASTCACHE_ROOT
[hfrvla-foreground] tmp_root=$HFRVLA_TMP_ROOT
[hfrvla-foreground] steps=$STEPS batch_size=$BATCH_SIZE num_workers=$NUM_WORKERS seq_len=$SEQ_LEN device=$DEVICE
[hfrvla-foreground] wandb_enable=$WANDB_ENABLE
EOF

exec "$PROJECT_ROOT/scripts/train_hfrvla_libero_merged.sh" "$@"
