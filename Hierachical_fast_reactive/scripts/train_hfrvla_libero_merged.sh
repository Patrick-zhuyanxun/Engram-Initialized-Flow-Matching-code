#!/usr/bin/env bash
set -euo pipefail

# Train HFRVLA fast-reactive module on the merged LIBERO shared dataset.
#
# This script intentionally pins the SmolVLA architecture knobs used by
# HuggingFaceVLA/smolvla_libero. The recorded HFRVLA dataset stores
# z_goal=(960,) and z_phase=(480,), so training with the raw smolvla_base
# defaults would create a z_phase=(720,) fast module and fail.
# Training runs in HFRVLA offline mode: the dataset already contains SmolVLA
# and DINOv3 cached features, so the train process does not construct the
# frozen slow planner or DINO backbone.
#
# Smoke:
#   STEPS=100 BATCH_SIZE=4 NUM_WORKERS=0 WARMUP_STEPS=10 JOINT_STEPS=80 \
#   REFINE_STEPS=10 OUT_DIR=outputs/train_hfrvla_smoke \
#     scripts/train_hfrvla_libero_merged.sh
#
# Full:
#   OUT_DIR=checkpoints/hfrvla_run01 scripts/train_hfrvla_libero_merged.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/Robotic_infra/lerobot/.venv/bin/python}"

DATASET_REPO_ID="${DATASET_REPO_ID:-HFRVLA_libero_v1}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/checkpoints/HFRVLA_libero_v1_merged_reindexed}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/checkpoints/hfrvla_run01}"
JOB_NAME="${JOB_NAME:-hfrvla_libero_run01}"

DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-60000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-50}"

LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-$STEPS}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-$LR}"

SEQ_LEN="${SEQ_LEN:-8}"
HFRVLA_DATASET_BACKEND="${HFRVLA_DATASET_BACKEND:-lerobot}"
HFRVLA_FASTCACHE_ROOT="${HFRVLA_FASTCACHE_ROOT:-$PROJECT_ROOT/checkpoints/HFRVLA_libero_v1_fastcache_v2}"
HFRVLA_FASTCACHE_ROLLOUT_ROOT="${HFRVLA_FASTCACHE_ROLLOUT_ROOT:-}"
case "${HFRVLA_DATASET_BACKEND,,}" in
  lerobot|fastcache)
    HFRVLA_DATASET_BACKEND="${HFRVLA_DATASET_BACKEND,,}"
    ;;
  *)
    echo "[hfrvla-train] HFRVLA_DATASET_BACKEND must be 'lerobot' or 'fastcache': $HFRVLA_DATASET_BACKEND" >&2
    exit 1
    ;;
esac
export HFRVLA_DATASET_BACKEND
export HFRVLA_FASTCACHE_ROOT
export HFRVLA_FASTCACHE_ROLLOUT_ROOT

OFFLINE_ZGOAL_DIM="${OFFLINE_ZGOAL_DIM:-960}"
OFFLINE_ZPHASE_DIM="${OFFLINE_ZPHASE_DIM:-480}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
JOINT_STEPS="${JOINT_STEPS:-49000}"
REFINE_STEPS="${REFINE_STEPS:-10000}"
LOSS_DELTA_TARGET_CLIP="${LOSS_DELTA_TARGET_CLIP:-true}"
# Stage A v2 defaults (calibrated 2026-05-22). Margins are in per-frame
# squared L2 units; LIBERO err_before has median ~2.4 so v1 defaults of 0.05
# / 0.01 were ~50x too small and caused BCE labels to stay ~100% positive
# (gate_prior=0.9) and L_preserve_zero=0. Reproduce pre-Stage-A baseline
# with: GATE_IMPROVEMENT_MARGIN=0.02 LOSS_LAMBDA_GATE_PRIOR=0.02 LOSS_LAMBDA_PRESERVE_ZERO=0.0
GATE_IMPROVEMENT_MARGIN="${GATE_IMPROVEMENT_MARGIN:-0.5}"
LOSS_LAMBDA_FINAL="${LOSS_LAMBDA_FINAL:-1.0}"
LOSS_LAMBDA_PRESERVE="${LOSS_LAMBDA_PRESERVE:-0.5}"
LOSS_LAMBDA_GATE_PRIOR="${LOSS_LAMBDA_GATE_PRIOR:-0.10}"
ERR_PRESERVE_THRESH="${ERR_PRESERVE_THRESH:-0.5}"
USE_STAGE_B="${USE_STAGE_B:-false}"
case "${USE_STAGE_B,,}" in
  1|true|yes|on)
    USE_STAGE_B="true"
    ;;
  *)
    USE_STAGE_B="false"
    ;;
esac
LOSS_LAMBDA_PRESERVE_ZERO_STAGE_B="${LOSS_LAMBDA_PRESERVE_ZERO_STAGE_B:-2.0}"
if [[ "$USE_STAGE_B" == "true" ]]; then
  LOSS_LAMBDA_GATE="${LOSS_LAMBDA_GATE:-3.0}"
  LOSS_LAMBDA_PRESERVE_ZERO="${LOSS_LAMBDA_PRESERVE_ZERO:-$LOSS_LAMBDA_PRESERVE_ZERO_STAGE_B}"
else
  LOSS_LAMBDA_GATE="${LOSS_LAMBDA_GATE:-1.0}"
  LOSS_LAMBDA_PRESERVE_ZERO="${LOSS_LAMBDA_PRESERVE_ZERO:-1.0}"
fi
LOSS_LAMBDA_CORRECT="${LOSS_LAMBDA_CORRECT:-1.0}"
LOSS_LAMBDA_RATE="${LOSS_LAMBDA_RATE:-0.5}"
LOSS_LAMBDA_SMOOTH="${LOSS_LAMBDA_SMOOTH:-0.2}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
FOCAL_POS_WEIGHT="${FOCAL_POS_WEIGHT:-4.0}"
GATE_TASK_BUDGET="${GATE_TASK_BUDGET:-0.25}"
RESIDUAL_MERGE_MODE="${RESIDUAL_MERGE_MODE:-gated}"
A2C2_ALPHA="${A2C2_ALPHA:-1.0}"
A2C2_USE_LATENT_CONTEXT="${A2C2_USE_LATENT_CONTEXT:-true}"

WANDB_ENABLE="${WANDB_ENABLE:-false}"
case "${WANDB_ENABLE,,}" in
  1|true|yes|on)
    WANDB_ENABLE="true"
    unset WANDB_MODE
    unset WANDB_DISABLED
    ;;
  *)
    WANDB_ENABLE="false"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    export WANDB_DISABLED="${WANDB_DISABLED:-true}"
    ;;
esac
WANDB_PROJECT="${WANDB_PROJECT:-hfrvla}"

DINO_REPO="${DINO_REPO:-$PROJECT_ROOT/checkpoints/dinov3_src}"
DINO_WEIGHTS="${DINO_WEIGHTS:-$PROJECT_ROOT/checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth}"
DINO_ARCH="${DINO_ARCH:-dinov3_vits16}"

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

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[hfrvla-train] dataset root does not exist: $DATASET_ROOT" >&2
  exit 1
fi

if [[ "$HFRVLA_DATASET_BACKEND" == "fastcache" && ! -f "$HFRVLA_FASTCACHE_ROOT/meta.json" ]]; then
  cat >&2 <<EOF
[hfrvla-train] fast-cache meta.json does not exist:
  $HFRVLA_FASTCACHE_ROOT/meta.json

Build it first:
  $PY $PROJECT_ROOT/scripts/build_hfrvla_fastcache.py \\
      --source-root "$DATASET_ROOT" \\
      --cache-root "$HFRVLA_FASTCACHE_ROOT"
EOF
  exit 1
fi

if [[ "$HFRVLA_DATASET_BACKEND" == "fastcache" && -n "$HFRVLA_FASTCACHE_ROLLOUT_ROOT" && ! -f "$HFRVLA_FASTCACHE_ROLLOUT_ROOT/meta.json" ]]; then
  cat >&2 <<EOF
[hfrvla-train] rollout fast-cache meta.json does not exist:
  $HFRVLA_FASTCACHE_ROLLOUT_ROOT/meta.json

Build it first:
  $PY $PROJECT_ROOT/scripts/build_hfrvla_fastcache.py \\
      --source-root "$PROJECT_ROOT/checkpoints/HFRVLA_libero_v1_zero_fast_rollouts" \\
      --cache-root "$HFRVLA_FASTCACHE_ROLLOUT_ROOT" \\
      --static-y-preserve
EOF
  exit 1
fi

if [[ ! -d "$DINO_REPO" ]]; then
  echo "[hfrvla-train] DINOv3 repo does not exist: $DINO_REPO" >&2
  exit 1
fi

if [[ ! -f "$DINO_WEIGHTS" ]]; then
  echo "[hfrvla-train] DINOv3 weights do not exist: $DINO_WEIGHTS" >&2
  exit 1
fi

if [[ -e "$OUT_DIR" && "${RESUME:-false}" != "true" ]]; then
  cat >&2 <<EOF
[hfrvla-train] output directory already exists:
  $OUT_DIR

LeRobot refuses to overwrite an existing --output_dir when resume is false.
Choose a fresh directory, for example:
  OUT_DIR=${OUT_DIR}_run2 scripts/train_hfrvla_libero_merged.sh

Or resume by passing the proper LeRobot resume flag and setting RESUME=true.
EOF
  exit 1
fi

echo "[hfrvla-train] dataset=$DATASET_REPO_ID root=$DATASET_ROOT"
echo "[hfrvla-train] dataset_backend=$HFRVLA_DATASET_BACKEND fastcache_root=$HFRVLA_FASTCACHE_ROOT rollout_fastcache_root=${HFRVLA_FASTCACHE_ROLLOUT_ROOT:-<unset>}"
echo "[hfrvla-train] output=$OUT_DIR"
echo "[hfrvla-train] steps=$STEPS batch_size=$BATCH_SIZE num_workers=$NUM_WORKERS seq_len=$SEQ_LEN device=$DEVICE"
echo "[hfrvla-train] curriculum warmup=$WARMUP_STEPS joint=$JOINT_STEPS refine=$REFINE_STEPS"
echo "[hfrvla-train] objective residual_mode=$RESIDUAL_MERGE_MODE a2c2_alpha=$A2C2_ALPHA a2c2_use_latent_context=$A2C2_USE_LATENT_CONTEXT stage_b=$USE_STAGE_B delta_target_clip=$LOSS_DELTA_TARGET_CLIP gate=$LOSS_LAMBDA_GATE gate_margin=$GATE_IMPROVEMENT_MARGIN final=$LOSS_LAMBDA_FINAL preserve=$LOSS_LAMBDA_PRESERVE gate_prior=$LOSS_LAMBDA_GATE_PRIOR preserve_zero=$LOSS_LAMBDA_PRESERVE_ZERO preserve_thresh=$ERR_PRESERVE_THRESH correct=$LOSS_LAMBDA_CORRECT rate=$LOSS_LAMBDA_RATE smooth=$LOSS_LAMBDA_SMOOTH focal_gamma=$FOCAL_GAMMA focal_pos_weight=$FOCAL_POS_WEIGHT gate_budget=$GATE_TASK_BUDGET"
echo "[hfrvla-train] offline_training_mode=true z_goal=$OFFLINE_ZGOAL_DIM z_phase=$OFFLINE_ZPHASE_DIM"
echo "[hfrvla-train] wandb_enable=$WANDB_ENABLE"
echo "[hfrvla-train] tmp_root=$HFRVLA_TMP_ROOT"
echo "[hfrvla-train] hf_datasets_cache=$HF_DATASETS_CACHE"
echo "[hfrvla-train] tmpdir=$TMPDIR"

"$PY" "$PROJECT_ROOT/scripts/train_via_lerobot.py" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=hfrvla \
  --policy.device="$DEVICE" \
  --policy.seq_len="$SEQ_LEN" \
  --policy.offline_training_mode=true \
  --policy.offline_zgoal_dim="$OFFLINE_ZGOAL_DIM" \
  --policy.offline_zphase_dim="$OFFLINE_ZPHASE_DIM" \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Instruct \
  --policy.expert_width_multiplier=0.5 \
  --policy.num_vlm_layers=0 \
  --policy.load_vlm_weights=false \
  --policy.dinov3_local_repo="$DINO_REPO" \
  --policy.dinov3_local_weights="$DINO_WEIGHTS" \
  --policy.dinov3_arch="$DINO_ARCH" \
  --policy.curriculum_warmup_steps="$WARMUP_STEPS" \
  --policy.curriculum_joint_steps="$JOINT_STEPS" \
  --policy.curriculum_refine_steps="$REFINE_STEPS" \
  --policy.loss_delta_target_clip="$LOSS_DELTA_TARGET_CLIP" \
  --policy.residual_merge_mode="$RESIDUAL_MERGE_MODE" \
  --policy.a2c2_alpha="$A2C2_ALPHA" \
  --policy.a2c2_use_latent_context="$A2C2_USE_LATENT_CONTEXT" \
  --policy.use_stage_b_objective="$USE_STAGE_B" \
  --policy.loss_lambda_gate="$LOSS_LAMBDA_GATE" \
  --policy.gate_improvement_margin="$GATE_IMPROVEMENT_MARGIN" \
  --policy.loss_lambda_final="$LOSS_LAMBDA_FINAL" \
  --policy.loss_lambda_preserve="$LOSS_LAMBDA_PRESERVE" \
  --policy.loss_lambda_gate_prior="$LOSS_LAMBDA_GATE_PRIOR" \
  --policy.loss_lambda_preserve_zero="$LOSS_LAMBDA_PRESERVE_ZERO" \
  --policy.err_preserve_thresh="$ERR_PRESERVE_THRESH" \
  --policy.loss_lambda_correct="$LOSS_LAMBDA_CORRECT" \
  --policy.loss_lambda_rate="$LOSS_LAMBDA_RATE" \
  --policy.loss_lambda_smooth="$LOSS_LAMBDA_SMOOTH" \
  --policy.focal_gamma="$FOCAL_GAMMA" \
  --policy.focal_pos_weight="$FOCAL_POS_WEIGHT" \
  --policy.gate_task_budget="$GATE_TASK_BUDGET" \
  --policy.optimizer_lr="$LR" \
  --policy.optimizer_weight_decay="$WEIGHT_DECAY" \
  --policy.optimizer_grad_clip_norm="$GRAD_CLIP_NORM" \
  --policy.scheduler_warmup_steps="$WARMUP_STEPS" \
  --policy.scheduler_decay_steps="$SCHEDULER_DECAY_STEPS" \
  --policy.scheduler_decay_lr="$SCHEDULER_DECAY_LR" \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --log_freq="$LOG_FREQ" \
  --output_dir="$OUT_DIR" \
  --wandb.enable="$WANDB_ENABLE" \
  --wandb.project="$WANDB_PROJECT" \
  --job_name="$JOB_NAME" \
  "$@"
