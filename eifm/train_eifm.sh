#!/bin/bash
# ============================================================
# EI-FM (Engram-Initialized Flow Matching) Training Script
# ============================================================
#
# Prerequisites:
#   1. X-VLA checkpoint downloaded: lerobot/xvla-libero
#   2. Engram table built: eifm/engram_table_projected.pt
#   3. Plugin installed: uv pip install -e ../lerobot_policy_eifm
#
# Usage:
#   cd /home/hucenrotia/Patrick/VLA_research/lerobot
#   bash ../eifm/train_eifm.sh
#
# ============================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────
VLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EIFM_CHECKPOINT="${VLA_ROOT}/checkpoints/eifm-libero"
OUTPUT_DIR="${VLA_ROOT}/outputs/eifm_libero_spatial_v3"

# ── Hyperparameters ──────────────────────────────────────────
STEPS=100000          # Total training steps
BATCH_SIZE=32         # Per-GPU batch size (A5000 24GB)
EVAL_FREQ=2000       # Evaluate every N steps
SAVE_FREQ=2000       # Save checkpoint every N steps
LOG_FREQ=200          # Log metrics every N steps
EVAL_EPISODES=6      # Episodes per task for evaluation
EVAL_BATCH_SIZE=2    # Eval batch size

# ── Environment ──────────────────────────────────────────────
export MUJOCO_GL=egl  # Headless rendering

# ── Run Training ─────────────────────────────────────────────
cd "${VLA_ROOT}/lerobot"

uv run lerobot-train \
  --policy.path="${EIFM_CHECKPOINT}" \
  --policy.repo_id=local/eifm-libero-spatial \
  --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --wandb.enable=true   \
  --wandb.project=Lerobot_eifm_Project   \
  --env.type=libero \
  --env.task=libero_spatial \
  --output_dir="${OUTPUT_DIR}" \
  --steps=${STEPS} \
  --batch_size=${BATCH_SIZE} \
  --resume=false \
  --eval.batch_size=${EVAL_BATCH_SIZE} \
  --eval.n_episodes=${EVAL_EPISODES} \
  --eval_freq=${EVAL_FREQ} \
  --save_freq=${SAVE_FREQ} \
  --save_checkpoint=true \
  --log_freq=${LOG_FREQ}
