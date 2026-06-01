#!/usr/bin/env bash
set -euo pipefail

# Fine-tune lerobot/smolvla_base on the official LeRobot LIBERO dataset.
#
# This produces the LIBERO-adapted SmolVLA checkpoint that HFRVLA should use as
# its frozen slow planner for recording a_base/z_goal/z_phase.
#
# Example:
#   STEPS=100000 BATCH_SIZE=4 DEVICE=cuda \
#     scripts/train_smolvla_libero_baseline.sh
#
# Optional local dataset root:
#   DATASET_ROOT=/home/.../.cache/huggingface/lerobot/HuggingFaceVLA/libero \
#     scripts/train_smolvla_libero_baseline.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LERO_DIR="${LERO_DIR:-$HOME/Robotic_infra/lerobot}"
LERO_TRAIN="${LERO_TRAIN:-$LERO_DIR/.venv/bin/lerobot-train}"

DATASET_REPO_ID="${DATASET_REPO_ID:-HuggingFaceVLA/libero}"
DATASET_ROOT="${DATASET_ROOT:-}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/checkpoints/smolvla_libero_slow}"
POLICY_REPO_ID="${POLICY_REPO_ID:-local/smolvla_libero_slow}"
JOB_NAME="${JOB_NAME:-smolvla_libero_slow}"

DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
EVAL_FREQ="${EVAL_FREQ:-1000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_N_EPISODES="${EVAL_N_EPISODES:-1}"
ENV_TASK="${ENV_TASK:-libero_10}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
case "${WANDB_ENABLE,,}" in
  1|true|yes|on)
    WANDB_ENABLE="true"
    ;;
  *)
    WANDB_ENABLE="false"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    export WANDB_DISABLED="${WANDB_DISABLED:-true}"
    ;;
esac

export MUJOCO_GL="${MUJOCO_GL:-egl}"
HFRVLA_TMP_ROOT="${HFRVLA_TMP_ROOT:-$HOME/tmp/hfrvla}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HFRVLA_TMP_ROOT/hf_datasets}"
export TMPDIR="${TMPDIR:-$HFRVLA_TMP_ROOT/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$HFRVLA_TMP_ROOT/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HFRVLA_TMP_ROOT/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-$HFRVLA_TMP_ROOT/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HFRVLA_TMP_ROOT/triton}"
export WANDB_DIR="${WANDB_DIR:-$HFRVLA_TMP_ROOT/wandb}"

mkdir -p \
  "$HF_DATASETS_CACHE" \
  "$TMPDIR" \
  "$NUMBA_CACHE_DIR" \
  "$MPLCONFIGDIR" \
  "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" \
  "$WANDB_DIR"

dataset_root_args=()
if [[ -n "$DATASET_ROOT" ]]; then
  dataset_root_args+=(--dataset.root="$DATASET_ROOT")
fi

if [[ -e "$OUT_DIR" ]]; then
  cat >&2 <<EOF
[smolvla-libero] output directory already exists:
  $OUT_DIR

LeRobot refuses to start a new training run when --output_dir already exists.
Use one of these options:
  1. If this was only the empty directory from a failed launch:
       rmdir "$OUT_DIR"
  2. Or choose a fresh output directory:
       OUT_DIR=${OUT_DIR}_run2 scripts/train_smolvla_libero_baseline.sh
  3. Or resume a real interrupted run by passing LeRobot's resume flag after
     the wrapper command.
EOF
  exit 1
fi

echo "[smolvla-libero] training SmolVLA base -> $OUT_DIR"
echo "[smolvla-libero] dataset=$DATASET_REPO_ID root=${DATASET_ROOT:-<hub/cache>}"
echo "[smolvla-libero] wandb_enable=$WANDB_ENABLE"
echo "[smolvla-libero] tmp_root=$HFRVLA_TMP_ROOT"
echo "[smolvla-libero] hf_datasets_cache=$HF_DATASETS_CACHE"

cd "$LERO_DIR"
"$LERO_TRAIN" \
  --policy.type=smolvla \
  --policy.repo_id="$POLICY_REPO_ID" \
  --policy.push_to_hub=false \
  --policy.load_vlm_weights=true \
  --policy.device="$DEVICE" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  "${dataset_root_args[@]}" \
  --env.type=libero \
  --env.task="$ENV_TASK" \
  --output_dir="$OUT_DIR" \
  --steps="$STEPS" \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --save_freq="$SAVE_FREQ" \
  --eval_freq="$EVAL_FREQ" \
  --eval.batch_size="$EVAL_BATCH_SIZE" \
  --eval.n_episodes="$EVAL_N_EPISODES" \
  --wandb.enable="$WANDB_ENABLE" \
  --job_name="$JOB_NAME" \
  "$@"

cat <<EOF

[smolvla-libero] done.

Use this checkpoint as HFRVLA --smolvla:
  $OUT_DIR/checkpoints/last/pretrained_model

Next smoke eval:
  $PROJECT_ROOT/scripts/test_alignment.py \\
    --smolvla $OUT_DIR/checkpoints/last/pretrained_model \\
    --dataset-root $PROJECT_ROOT/checkpoints/HFRVLA_libero_v1 \\
    --suite libero_spatial \\
    --task-ids 0,1,2 \\
    --n-episodes 5
EOF
