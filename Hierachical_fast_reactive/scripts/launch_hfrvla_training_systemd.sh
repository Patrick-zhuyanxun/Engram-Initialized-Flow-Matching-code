#!/usr/bin/env bash
set -euo pipefail

# Launch HFRVLA training as a detached systemd user service.
#
# This keeps the training process out of the GNOME Terminal scope. If
# systemd-oomd kills a terminal under memory pressure, the training unit is not
# killed as part of that terminal cgroup.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_NAME="${RUN_NAME:-hfrvla_run04}"
UNIT="${UNIT:-$RUN_NAME}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/checkpoints/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/outputs/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_NAME.log}"
HFRVLA_TMP_ROOT="${HFRVLA_TMP_ROOT:-$HOME/tmp/hfrvla}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
SEQ_LEN="${SEQ_LEN:-8}"
HFRVLA_DATASET_BACKEND="${HFRVLA_DATASET_BACKEND:-lerobot}"
HFRVLA_FASTCACHE_ROOT="${HFRVLA_FASTCACHE_ROOT:-$PROJECT_ROOT/checkpoints/HFRVLA_libero_v1_fastcache_v2}"

mkdir -p "$LOG_DIR" "$HFRVLA_TMP_ROOT"

if ! command -v systemd-run >/dev/null 2>&1; then
  echo "[hfrvla-launch] systemd-run not found. Use nohup only as a fallback." >&2
  exit 1
fi

cat > "$LOG_FILE" <<EOF
[hfrvla-launch] unit=$UNIT
[hfrvla-launch] out_dir=$OUT_DIR
[hfrvla-launch] tmp_root=$HFRVLA_TMP_ROOT
[hfrvla-launch] seq_len=$SEQ_LEN
[hfrvla-launch] dataset_backend=$HFRVLA_DATASET_BACKEND
[hfrvla-launch] fastcache_root=$HFRVLA_FASTCACHE_ROOT
[hfrvla-launch] started_at=$(date --iso-8601=seconds)
EOF

systemd-run --user \
  --unit="$UNIT" \
  --collect \
  --property=WorkingDirectory="$PROJECT_ROOT" \
  --property=StandardOutput=append:"$LOG_FILE" \
  --property=StandardError=append:"$LOG_FILE" \
  /usr/bin/env \
    OUT_DIR="$OUT_DIR" \
    WANDB_ENABLE="$WANDB_ENABLE" \
    HFRVLA_TMP_ROOT="$HFRVLA_TMP_ROOT" \
    SEQ_LEN="$SEQ_LEN" \
    HFRVLA_DATASET_BACKEND="$HFRVLA_DATASET_BACKEND" \
    HFRVLA_FASTCACHE_ROOT="$HFRVLA_FASTCACHE_ROOT" \
    "$PROJECT_ROOT/scripts/train_hfrvla_libero_merged.sh" \
    "$@"

echo "$UNIT" > "$LOG_DIR/$RUN_NAME.unit"
echo "[hfrvla-launch] started systemd user unit: $UNIT"
echo "[hfrvla-launch] log: $LOG_FILE"
echo "[hfrvla-launch] follow:"
echo "  tail -f $LOG_FILE"
echo "  systemctl --user status $UNIT"
