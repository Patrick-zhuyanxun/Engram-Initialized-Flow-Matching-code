#!/bin/bash
# ============================================================
# <topic_name> — Training & Eval Script
# ============================================================
#
# Usage（從主題根目錄執行）：
#   cd ~/Patrick/VLA_research/<topic_name>
#   bash scripts/train.sh baseline
#   bash scripts/train.sh eval
#
# Prerequisites:
#   cd ~/Robotic_infra/lerobot
#   uv pip install -e ~/Patrick/VLA_research/<topic_name>/policy/<package_name>
# ============================================================

set -euo pipefail

TOPIC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$(dirname "${BASH_SOURCE[0]}")"
INFRA_DIR="${HOME}/Robotic_infra"
OUTPUT_DIR="${TOPIC_ROOT}/outputs/baseline"

export MUJOCO_GL=egl

cd "${INFRA_DIR}/lerobot"

PHASE="${1:-baseline}"

if [[ "${PHASE}" == "baseline" ]]; then
    echo "=== Training baseline ==="
    uv run lerobot-train \
        --policy.type=__policy_alias__ \
        --output_dir="${OUTPUT_DIR}"
fi

if [[ "${PHASE}" == "eval" ]]; then
    CHECKPOINT=$(ls -d "${OUTPUT_DIR}/checkpoints/"*"/pretrained_model" | sort -V | tail -1)
    echo "=== Evaluating: ${CHECKPOINT} ==="
    uv run lerobot-eval \
        --policy.path="${CHECKPOINT}" \
        --env.type=libero
fi
