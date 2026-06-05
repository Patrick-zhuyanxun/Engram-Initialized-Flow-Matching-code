# lerobot_policy_hfrvla

**Hierarchical Fast-Reactive VLA** — a LeRobot plugin that mounts a small,
trainable, wrist-camera residual policy on top of a frozen SmolVLA. The current
paper-facing path is Fast Wrist Residual (FWR): `fast_wrist` and
`fast_wrist_chunk` compute `a_base + alpha * clip(delta_a)` without the older
gate/contact/GRU machinery. `fast_wrist_chunk` additionally attends over the
full frozen base action chunk.

The legacy gated path is still loadable for old scripts and checkpoints, but it
is not the active paper mainline. New experiments should prefer
`RESIDUAL_MERGE_MODE=fast_wrist_chunk` with a schema-v3 fast-cache unless they
are explicitly reproducing the retired gated objective.

Design contract: `Hierachical_fast_reactive/paper/notes/implementation_spec.md`.

## Quick install

```bash
cd ~/Robotic_infra/lerobot
uv pip install -e ~/Patrick/VLA_research/Hierachical_fast_reactive/policy/lerobot_policy_hfrvla
```

## DINOv3 weights

Access must be granted by Meta:
https://github.com/facebookresearch/dinov3#downloading-weights

Default model: `facebook/dinov3-vits16-pretrain-lvd1689m` (ViT-S/16, ~21 M params).

## Train and eval

Use a SmolVLA slow planner that is already adapted to LIBERO's
`image`/`image2`/8D-state/7D-action contract. Raw `lerobot/smolvla_base` is
only a warm start for fine-tuning and is rejected by the recording/alignment
scripts by default. The canonical published checkpoint is
`HuggingFaceVLA/smolvla_libero`.

```bash
cd ~/Patrick/VLA_research/Hierachical_fast_reactive

~/Robotic_infra/lerobot/.venv/bin/python scripts/record_hfrvla_libero.py \
    --src-repo-id HuggingFaceVLA/libero \
    --out-repo-id HFRVLA_libero_v1 \
    --out-root checkpoints/HFRVLA_libero_v1 \
    --dinov3-repo checkpoints/dinov3_src \
    --dinov3-weights checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

~/Robotic_infra/lerobot/.venv/bin/python scripts/build_hfrvla_fastcache.py \
    --source-root checkpoints/HFRVLA_libero_v1_merged_reindexed \
    --cache-root checkpoints/HFRVLA_libero_v1_fastcache_v3_plan50 \
    --chunk-len 50

HFRVLA_DATASET_BACKEND=fastcache \
HFRVLA_FASTCACHE_ROOT=checkpoints/HFRVLA_libero_v1_fastcache_v3_plan50 \
SEQ_LEN=4 \
~/Robotic_infra/lerobot/.venv/bin/python scripts/train_via_lerobot.py \
    --dataset.repo_id=HFRVLA_libero_v1 \
    --dataset.root=checkpoints/HFRVLA_libero_v1_merged_reindexed \
    --policy.type=hfrvla
```

Fast-cache storage is frame-level; `SEQ_LEN` is only a training-time windowing
choice passed to `--policy.seq_len`. Schema v3 stores `a_base_chunk` and
`chunk_step_idx` for `RESIDUAL_MERGE_MODE=fast_wrist_chunk`; schema v1/v2
caches remain loadable for older modes.

## Registration

`lerobot-train` and `lerobot-eval` recognize `--policy.type=hfrvla` via the entry point in `pyproject.toml`.
