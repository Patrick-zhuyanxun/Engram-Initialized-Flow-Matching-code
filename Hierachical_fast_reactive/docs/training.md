# HFRVLA training and eval (lerobot-native)

Use `lerobot-train` for training and `lerobot-eval` for evaluation.
Recording uses a wrapper script that writes a standard LeRobotDataset v3.

Active scripts:
- `scripts/record_hfrvla_libero.py` - one-shot or sharded recording (`--ep-from/--ep-to`)
- `scripts/merge_hfrvla_shards_fast.py` - offline stitcher for parallel shards (file/metadata only, no re-encode)
- `scripts/train_via_lerobot.py` - wrapper around `lerobot-train` that adds curriculum control
- `scripts/package_hfrvla_checkpoint.py` - bundle SmolVLA + fast weights into a `lerobot-eval` checkpoint dir (`--disable-fast` for alignment tests)
- `scripts/test_alignment.py` - I/O alignment check against SmolVLA baseline before training
- `lerobot-eval` - CLI evaluation, no wrapper needed

Legacy scripts live in `scripts/legacy/`; see `scripts/legacy/README.md`.

Spec and plan:
- `docs/superpowers/specs/2026-05-15-hfrvla-lerobot-native-design.md`
- `docs/superpowers/plans/2026-05-15-hfrvla-lerobot-native.md`

## 0. Prerequisites

| Thing | Where |
|---|---|
| Python venv | `~/Robotic_infra/lerobot/.venv` |
| HFRVLA plugin | `policy/lerobot_policy_hfrvla/` installed editable into the venv |
| DINOv3 source | `checkpoints/dinov3_src/` |
| DINOv3 weights | `checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth` |
| LIBERO data | `HuggingFaceVLA/libero` |
| Wandb login | `~/Robotic_infra/lerobot/.venv/bin/wandb login` |

Working directory:

```bash
cd ~/Patrick/VLA_research/Hierachical_fast_reactive
```

For offline/local runs, set:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE=/tmp/hfrvla_hf_datasets_cache
```

## 1. Record the dataset

> **Why a custom dataset?** SmolVLA is frozen at inference; the fast module needs the SmolVLA *outputs* (the planned chunk = `a_base`) plus DINOv3 wrist patches plus the two hooked `z_goal` / `z_phase` tensors. These are all computed during recording so training is a clean offline pass over a regular LeRobotDataset v3 (no SmolVLA forward in the train loop).

> **Sharded recording (recommended for the full ~1700-episode LIBERO):** parallel processes write disjoint episode ranges to separate `--out-root` shards, then `scripts/merge_hfrvla_shards_fast.py` stitches them into one final dataset by hardlinking parquet files and renaming offsets. The merger only rewrites the `task_index` column for shards whose local task indices need shifting against the merged global table. See §1.3 for the merge command.

### 1.1 Single-process full recording

```bash
~/Robotic_infra/lerobot/.venv/bin/python scripts/record_hfrvla_libero.py \
    --src-repo-id HuggingFaceVLA/libero \
    --out-repo-id HFRVLA_libero_v1 \
    --out-root checkpoints/HFRVLA_libero_v1 \
    --smolvla lerobot/smolvla_base \
    --dinov3-repo checkpoints/dinov3_src \
    --dinov3-weights checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

If the source dataset is already cached locally and network access should not be used, pass `--src-root`:

```bash
~/Robotic_infra/lerobot/.venv/bin/python scripts/record_hfrvla_libero.py \
    --src-repo-id HuggingFaceVLA/libero \
    --src-root /home/hucenrotia/.cache/huggingface/lerobot/HuggingFaceVLA/libero \
    --out-repo-id HFRVLA_libero_v1 \
    --out-root checkpoints/HFRVLA_libero_v1 \
    --smolvla /home/hucenrotia/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205 \
    --dinov3-repo checkpoints/dinov3_src \
    --dinov3-weights checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

Smoke variant:

```bash
~/Robotic_infra/lerobot/.venv/bin/python scripts/record_hfrvla_libero.py \
    --src-repo-id HuggingFaceVLA/libero \
    --src-root /home/hucenrotia/.cache/huggingface/lerobot/HuggingFaceVLA/libero \
    --out-repo-id HFRVLA_libero_smoke \
    --out-root checkpoints/HFRVLA_libero_smoke \
    --smolvla /home/hucenrotia/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205 \
    --dinov3-repo checkpoints/dinov3_src \
    --dinov3-weights checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --max-episodes 5 \
    --device cpu
```

### 1.2 Sharded recording (parallel processes)

Split the source episodes into N disjoint ranges, write each to its own `--out-root`. On a 32-core machine 3-4 parallel processes is the sweet spot (CPU-bound: parquet write + JPEG encode dominate).

```bash
SRC_ROOT=/home/hucenrotia/.cache/huggingface/lerobot/HuggingFaceVLA/libero
SMOL=/home/hucenrotia/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205
DINO_REPO=checkpoints/dinov3_src
DINO_W=checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

for tag in a:0:400 b:400:800 c:800:1200 d:1200:1693; do
  IFS=: read shard ep_from ep_to <<< "$tag"
  nohup ~/Robotic_infra/lerobot/.venv/bin/python scripts/record_hfrvla_libero.py \
      --src-repo-id HuggingFaceVLA/libero \
      --src-root "$SRC_ROOT" \
      --out-repo-id HFRVLA_libero_v1_shard_$shard \
      --out-root checkpoints/HFRVLA_libero_v1_shard_$shard \
      --smolvla "$SMOL" \
      --dinov3-repo "$DINO_REPO" \
      --dinov3-weights "$DINO_W" \
      --ep-from "$ep_from" --ep-to "$ep_to" \
      --dino-batch-size 512 \
      > outputs/record_shard_$shard.log 2>&1 &
done
wait
```

Each shard ends up as a self-contained LeRobotDataset v3 with its own `total_episodes`, local `task_index` 0..N-1, and `data/chunk-000/file-{0..K}.parquet`. Pure inline-image dataset (`dtype: image`), so `videos/` subdir is absent.

### 1.3 Merge shards

`scripts/merge_hfrvla_shards_fast.py` is **offline**: it never calls `LeRobotDataset()` on the shards (which would try the Hub and silently fail on synthetic repo-ids). Instead it:

1. Hardlinks shard A's data parquets verbatim into `--out-root`.
2. For shards B/C/D, rewrites only the `task_index` column with a per-shard offset (one numeric column on read+write, ~600 ms/file).
3. Concatenates per-shard `meta/episodes/.../*.parquet` with `episode_index`, `data/file_index`, `dataset_from/to_index` offsets.
4. Dedupes tasks across shards into one `tasks.parquet`.
5. Aggregates per-episode stats via `compute_stats.aggregate_feature_stats` (with a flattener for image stats stored as nested object arrays).
6. Bumps `chunks_size` so every data file fits in `chunk-000`.

```bash
~/Robotic_infra/lerobot/.venv/bin/python scripts/merge_hfrvla_shards_fast.py \
    --shards checkpoints/HFRVLA_libero_v1_shard_a \
             checkpoints/HFRVLA_libero_v1_shard_b \
             checkpoints/HFRVLA_libero_v1_shard_c \
             checkpoints/HFRVLA_libero_v1_shard_d \
    --out-root checkpoints/HFRVLA_libero_v1 \
    --force-delete
```

**Why not the original `scripts/merge_hfrvla_shards.py`?** That used `LeRobotDataset(repo_id=..., root=shard)` to read each shard, which falls through to `_download()` against HuggingFace for the synthetic shard repo-id. It silent-failed after copying shard A.

Final dataset for the merge above: 1693 episodes, 273465 frames, 40 unique tasks, ~112 GB total.

### Recording flags

| Flag | Default | Description |
|---|---|---|
| `--src-repo-id` | `HuggingFaceVLA/libero` | Source LeRobot dataset. |
| `--src-root` | None | Optional local root for the source dataset. |
| `--out-repo-id` | `HFRVLA_libero_v1` | Identifier in the new dataset metadata. |
| `--out-root` | required | Local directory for the new dataset. |
| `--ep-from` | 0 | Start episode index (inclusive) for sharded recording. |
| `--ep-to` | None | End episode index (exclusive); default = all. |
| `--max-episodes` | None | Cap for smoke tests (applied after `--ep-from/--ep-to`). |
| `--dino-batch-size` | 64 | DINOv3 batched forward per episode (set 256-512 for higher GPU util). |
| `--smolvla` | `lerobot/smolvla_base` | SmolVLA base used for chunk inference. |
| `--dinov3-repo` | required | Local clone of facebookresearch/dinov3. |
| `--dinov3-weights` | required | DINOv3 `.pth` weight file. |
| `--dinov3-arch` | `dinov3_vits16` | torch.hub entry-point name. |
| `--wrist-key` | `observation.images.image2` | Eye-in-hand camera key in source dataset. |
| `--fps` | 10 | Must equal the source dataset fps. |
| `--dino-dtype` | `float32` | Set to `float16` to reduce storage. |
| `--device` | `cuda` | Inference device. |

## 2. Alignment test (run before §3)

**Goal:** confirm the HFRVLA wrapper's I/O matches LIBERO env expectations *before* spending hours training the fast module. Failure here means action shapes / normalization / observation keys are mis-wired and training will produce a broken policy.

**Mechanism:** `inference_disable_fast=True` makes `HFRVLAPolicy.select_action()` short-circuit to SmolVLA's `a_base` (the popped chunk action) before touching the fast module. Eval performance therefore equals SmolVLA-base running through the HFRVLA preprocessing pipeline.

```bash
~/Robotic_infra/lerobot/.venv/bin/python scripts/test_alignment.py \
    --suite libero_spatial \
    --task-ids 0,1,2 \
    --n-episodes 5
```

This packages `checkpoints/alignment_test/` (HFRVLA config + SmolVLA weights, fast module randomly initialized but never invoked), runs `lerobot-eval` against `libero_spatial` task_ids 0/1/2 with 5 episodes each (15 total), and prints per-task success rate.

**Pass criteria:**
- Any task with nonzero success rate → I/O is structurally sound (action shape, observation keys, normalization all correct).
- Aggregated success rate roughly in the SmolVLA-base ballpark (~30-60% on `libero_spatial` with N=15 has high variance, anything ≥ 20% is healthy).
- All 0% on every task → check `lerobot-eval` logs for shape errors, env errors, or "delta_a" path issues.

**Pass → run §3 training. Fail → debug before training.**

## 3. Train with lerobot-train via wrapper

Full run:

```bash
~/Robotic_infra/lerobot/.venv/bin/python scripts/train_via_lerobot.py \
    --dataset.repo_id=HFRVLA_libero_v1 \
    --dataset.root=checkpoints/HFRVLA_libero_v1 \
    --policy.type=hfrvla \
    --policy.curriculum_warmup_steps=1000 \
    --policy.curriculum_joint_steps=49000 \
    --policy.curriculum_refine_steps=10000 \
    --batch_size=128 \
    --num_workers=8 \
    --steps=60000 \
    --optimizer.lr=3e-4 \
    --optimizer.weight_decay=1e-4 \
    --optimizer.grad_clip_norm=1.0 \
    --save_freq=5000 \
    --log_freq=50 \
    --output_dir=checkpoints/hfrvla_run02 \
    --wandb.enable=true \
    --wandb.project=hfrvla
```

100-step smoke:

```bash
~/Robotic_infra/lerobot/.venv/bin/python scripts/train_via_lerobot.py \
    --dataset.repo_id=HFRVLA_libero_smoke \
    --dataset.root=checkpoints/HFRVLA_libero_smoke \
    --policy.type=hfrvla \
    --policy.curriculum_warmup_steps=10 \
    --policy.curriculum_joint_steps=80 \
    --policy.curriculum_refine_steps=10 \
    --batch_size=4 \
    --num_workers=0 \
    --steps=100 \
    --save_freq=100 \
    --log_freq=10 \
    --output_dir=outputs/train_smoke \
    --wandb.enable=false
```

The wrapper adds curriculum control on top of `lerobot-train`; all other flags are vanilla `lerobot-train` flags.
Use `--num_workers=0` in restricted sandboxes that block PyTorch multiprocessing.

### Curriculum and windowing knobs

| Flag | Default | Description |
|---|---|---|
| `--policy.curriculum_warmup_steps` | 1000 | Stage 0 length: delta only, heads frozen. |
| `--policy.curriculum_joint_steps` | 49000 | Stage 1 length: all losses active. |
| `--policy.curriculum_refine_steps` | 10000 | Stage 2 length: LR is reduced by 10x at entry. |
| `--policy.seq_len` | 8 | GRU window length; drives `observation_delta_indices` and `action_delta_indices`. |

### Output structure

```text
checkpoints/hfrvla_run02/
|-- checkpoints/
|   |-- 005000/
|   |   |-- pretrained_model/
|   |   `-- training_state/
|   |-- 010000/
|   `-- last/
|-- train_config.json
`-- wandb/
```

## 4. Evaluate

Single-task smoke:

```bash
NUMBA_CACHE_DIR=/tmp/hfrvla_numba_cache \
~/Robotic_infra/lerobot/.venv/bin/lerobot-eval \
    --policy.path=outputs/train_smoke/checkpoints/000100/pretrained_model \
    --env.type=libero \
    --env.task=libero_spatial \
    --env.task_ids='[0]' \
    --eval.n_episodes=1 \
    --eval.batch_size=1 \
    --output_dir=outputs/eval_smoke \
    --seed=42
```

Full suite:

```bash
for SUITE in libero_spatial libero_object libero_goal libero_10 libero_90; do
  NUMBA_CACHE_DIR=/tmp/hfrvla_numba_cache \
  ~/Robotic_infra/lerobot/.venv/bin/lerobot-eval \
      --policy.path=checkpoints/hfrvla_run02/checkpoints/last/pretrained_model \
      --env.type=libero \
      --env.task=$SUITE \
      --eval.n_episodes=20 \
      --eval.batch_size=4 \
      --output_dir=outputs/eval_run02_$SUITE \
      --seed=42
done
```

`--env.num_envs` is not a valid LiberoEnv field; use `--eval.batch_size` to control parallelism.

## 5. Health checks during training

| Stage | Boundary | Expected behavior |
|---|---|---|
| 0 warmup | `step in [0, warmup)` | Gate/contact heads frozen, lambdas are 0, `loss == delta`. |
| 1 joint | `step in [warmup, warmup+joint)` | All losses contribute; expect a small spike at the boundary, then descent. |
| 2 refine | `step >= warmup+joint` | `[curriculum] step <N>: entered Stage 2 - LR x 0.1` is printed exactly once. |

If the boundary log is missing, check `scripts/train_via_lerobot.py`; the wrapper may have stopped intercepting `update_policy`.

## 6. GPU utilization tips

The fast module is small and dataset access dominates time. If GPU utilization stays low after the first few hundred steps:

1. Increase `--batch_size` to 256 if memory allows.
2. Increase `--num_workers` to 12-16 outside restricted sandboxes.
3. Run with `--policy.use_bf16=true` if the inherited config supports it.

Past those, the next storage and bandwidth lever is re-recording with `--dino-dtype float16`.
