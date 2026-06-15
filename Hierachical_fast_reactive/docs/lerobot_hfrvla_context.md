# LeRobot and HFRVLA Context Notes

Created: 2026-05-16

Sources:

- Hugging Face LeRobot: Bring Your Own Policies
  https://huggingface.co/docs/lerobot/en/bring_your_own_policies
- Hugging Face LeRobot: Imitation Learning for Robots
  https://huggingface.co/docs/lerobot/en/il_robots
- Hugging Face LeRobot: LeRobotDataset v3.0
  https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3
- Local HFRVLA docs and code under `Hierachical_fast_reactive/`

Re-reviewed on 2026-05-18 against the live LeRobotDataset v3.0 docs and the
local LeRobot source under `~/Robotic_infra/lerobot/src/lerobot`.

Active-contract review on 2026-06-15: the current maintained HFRVLA direction
uses the `fast_wrist` / `fast_wrist_chunk` correction modes on top of frozen
`HuggingFaceVLA/smolvla_libero`. HFRVLA is the public method name; fast wrist
correction is the human-readable description. The older GRU/gate/contact design
is archive history only.

## 1. Project Identity

`Hierachical_fast_reactive` is the active project for HFRVLA:
**Hierarchical Fast-Reactive VLA**. It is a LeRobot policy plugin that wraps a
frozen SmolVLA with a small trainable fast reactive module.

Inference formula:

```text
a_final = a_base + alpha * clip(delta_a)
```

Where:

- `a_base`: popped action from the frozen SmolVLA action chunk.
- `delta_a`: residual action predicted by the fast module.
- `alpha`: deployment residual scale.
- `clip(delta_a)`: per-DoF residual cap used at training and inference.

The fast module is driven by wrist-camera DINOv3 patches, robot proprioception,
the current SmolVLA base action, chunk index, and two SmolVLA hidden summaries:
`z_goal` and `z_phase`.

## 2. Official LeRobot Policy Plugin Pattern

The official LeRobot "bring your own policies" flow expects an installable
Python package with the prefix `lerobot_policy_`.

Standard package shape:

```text
lerobot_policy_<name>/
|-- pyproject.toml
`-- src/lerobot_policy_<name>/
    |-- __init__.py
    |-- configuration_<name>.py
    |-- modeling_<name>.py
    `-- processor_<name>.py
```

Important contract points:

- Define a config class inheriting `PreTrainedConfig` or an existing policy
  config and register it with `@PreTrainedConfig.register_subclass("<name>")`.
- Define a policy class with `config_class`, `name`, `select_action()`, and
  `forward()`/loss behavior compatible with LeRobot training.
- If a custom processor factory is used, its name must match:
  `make_<policy_name>_pre_post_processors`.
- Register the policy in `pyproject.toml` under
  `[project.entry-points."lerobot.policies"]`.

HFRVLA implementation:

- Package: `policy/lerobot_policy_hfrvla`
- Entry point: `hfrvla = "lerobot_policy_hfrvla:HFRVLAPolicy"`
- Config: `HFRVLAConfig(SmolVLAConfig)` with
  `@PreTrainedConfig.register_subclass("hfrvla")`
- Policy: `HFRVLAPolicy(SmolVLAPolicy)`, `name = "hfrvla"`

After edits to the plugin, reinstall it into the LeRobot venv:

```bash
cd ~/Robotic_infra/lerobot
uv pip install -e ~/Patrick/VLA_research/Hierachical_fast_reactive/policy/lerobot_policy_hfrvla
```

## 3. Official LeRobot IL Workflow

The official imitation-learning flow is:

1. Calibrate robot and teleop device.
2. Teleoperate to verify controls and cameras.
3. Record demonstrations with `lerobot-record`.
4. Train with `lerobot-train --dataset.repo_id=... --policy.type=...`.
5. Evaluate or deploy with `lerobot-record --policy.path=...` for real robot
   evaluation, or `lerobot-eval` for supported simulated environments.

For HFRVLA, the same LeRobot-native idea is used, but recording is custom:
`scripts/record_hfrvla_libero.py` reads `HuggingFaceVLA/libero` and writes a
new local LeRobotDataset v3 with precomputed SmolVLA and DINOv3 features.

## 4. LeRobotDataset v3 Basics

Official v3 design points:

- v3 decouples storage from the user API.
- Low-dimensional signals are stored in Apache Parquet under `data/`.
- Visual streams are typically represented as MP4 shards under `videos/`.
- Metadata reconstructs episode views through offsets rather than one file per
  episode.
- The public API still returns normal Python dictionaries of PyTorch tensors.
  Temporal context is requested by passing `delta_timestamps`, whose values are
  seconds relative to the current frame.

Key metadata paths:

- `meta/info.json`: canonical schema, feature dtypes/shapes, FPS, path templates.
- `meta/stats.json`: normalization stats exposed as `dataset.meta.stats`.
- `meta/tasks.jsonl` or `meta/tasks.parquet`: task descriptions and IDs.
- `meta/episodes/`: chunked per-episode metadata, including offsets.
- `data/`: frame-level Parquet shards.
- `videos/`: per-camera MP4 shards when video-backed visual data is used.

When manually creating a v3 dataset:

```python
dataset = LeRobotDataset.create(...)
for episode in episodes:
    for frame in episode:
        dataset.add_frame(frame)
    dataset.save_episode()
dataset.finalize()
```

`finalize()` is required so parquet writers and buffered metadata are closed
cleanly before pushing or relying on the dataset.

Temporal windows are requested through `delta_timestamps` in seconds. Policy
configs expose this to the training loader through properties such as
`observation_delta_indices` and `action_delta_indices`.

## 5. HFRVLA Dataset v3 Schema

The HFRVLA training set is not plain `HuggingFaceVLA/libero`. It is a local
derived LeRobotDataset v3, normally:

```text
checkpoints/HFRVLA_libero_v1
```

The current local full dataset metadata reports:

- `codebase_version`: `v3.0`
- `total_episodes`: `1693`
- `total_frames`: `273465`
- `total_tasks`: `40`
- `fps`: `10`

Canonical features:

| Key | dtype | shape | Meaning |
|---|---:|---:|---|
| `observation.images.image` | image | `(256, 256, 3)` | LIBERO third-person camera |
| `observation.images.image2` | image | `(256, 256, 3)` | LIBERO wrist camera |
| `observation.state` | float32 | `(8,)` | LIBERO state |
| `action` | float32 | `(7,)` | expert action |
| `observation.extra.z_goal` | float32 | `(960,)` | SmolVLA text hidden summary |
| `observation.extra.z_phase` | float32 | `(480,)` | SmolVLA action-expert hidden summary from `HuggingFaceVLA/smolvla_libero` |
| `observation.extra.a_base` | float32 | `(7,)` | SmolVLA base action |
| `observation.extra.k_idx_norm` | float32 | `(1,)` | chunk index normalized to `[0, 1]` |
| `observation.extra.dino_patches` | float32 | `(196, 384)` | DINOv3 ViT-S/16 wrist patch tokens |
| `observation.extra.contact_label` | float32 | `(1,)` | legacy auxiliary label kept in the shared dataset; active HFRVLA correction losses do not use it |

Local note: the current full dataset has `data/chunk-000/*.parquet` and no
materialized `videos/` directory, even though `meta/info.json` includes a
`video_path` template. The training guide describes this as a pure inline-image
dataset.

## 5.1 HFRVLA Offline Read Path and Bottlenecks

Training uses `scripts/train_via_lerobot.py`, which monkey-patches the
LeRobot dataset factory only for `policy.type=hfrvla` with
`offline_training_mode=true`.

Current optimizations already in place:

- `resolve_delta_timestamps` is narrowed to the columns consumed by offline
  HFRVLA training. Raw `observation.images.*` keys are excluded, so LeRobot
  does not window or collate unused RGB images.
- `_load_hf_dataset` passes an explicit `columns=[...]` list to
  `datasets.Dataset.from_parquet`, so the HuggingFace dataset is materialized
  from the local parquet files without image columns.
- A post-load guard checks that the required offline columns are present:
  `state`, `action`, `a_base`, `k_idx_norm`, `z_goal`, `z_phase`,
  `dino_patches`, and `contact_label` for compatibility with the shared dataset.
- For speed-sensitive training, `scripts/build_hfrvla_fastcache.py` can derive
  a compact local cache from the canonical LeRobotDataset. The cache stores the
  offline training columns as contiguous `.npy` arrays, downcasts
  `z_goal/z_phase/dino_patches` to `float16`, records episode boundaries, and
  loads samples through `numpy.memmap`. Enable it with
  `HFRVLA_DATASET_BACKEND=fastcache` and `HFRVLA_FASTCACHE_ROOT=...`.

The expensive column is `observation.extra.dino_patches`. Its payload is:

```text
196 patches * 384 dims * 4 bytes ~= 301 KB per frame
seq_len=8 -> ~= 2.4 MB per sample before Python/Arrow/Tensor overhead
batch_size=128 -> ~= 308 MB of DINO patch payload per optimizer step
```

Run06 timing diagnosis on 2026-05-18:

| Dataset item configuration | Average single-item read time |
|---|---:|
| `seq_len=8` with `dino_patches` | `0.557s` |
| `seq_len=4` with `dino_patches` | `0.283s` |
| `seq_len=2` with `dino_patches` | `0.141s` |
| `seq_len=1` with `dino_patches` | `0.075s` |
| `seq_len=8` without `dino_patches` | `0.031s` |

This explains the train log where `updt_s ~= 0.03s` but `data_s` is
`7-15s`: the correction head is cheap, while the DataLoader is reading and
stacking large random temporal windows from a 111 GB local parquet dataset.
LeRobot's default offline DataLoader uses `shuffle=True`, so each batch tends
to touch many parquet row groups/files. With `num_workers=8` and
`prefetch_factor=2`, waiting time is still non-uniform because the main process
consumes batches much faster than workers can produce them; cache misses,
different shards, and the slowest worker dominate the next batch time.

Practical speed levers:

- Prefer the fast-cache backend for real training runs once the canonical
  LeRobotDataset has passed the contract/alignment checks.
- For current HFRVLA correction ablations, `SEQ_LEN=2` is the normal
  previous/current window. Longer windows increase DINO read payload and mostly
  matter for legacy gated GRU runs or explicit temporal experiments.
- Keep confirming the log line
  `[hfrvla-train] fast-cache dataset backend enabled` or
  `[hfrvla-train] offline dataset column pruning enabled`. If both disappear,
  raw images may be back in the LeRobot window/collate path.
- Increasing `NUM_WORKERS` only helps while CPU, RAM, and disk bandwidth have
  headroom. If workers are saturated or swap grows, larger worker counts make
  latency less predictable.
- Re-recording with `--dino-dtype float16` can halve the canonical dataset's
  DINO storage/read payload, but verify the training path casts cached DINO
  tensors to the model dtype before relying on that dataset.
- A larger future optimization is a locality-aware sampler; the fast-cache
  backend is the current compact cached feature store for `dino_patches`.

## 6. HFRVLA Recording Pipeline

Primary script:

```text
scripts/record_hfrvla_libero.py
```

Flow:

1. Load source dataset `HuggingFaceVLA/libero`, optionally through local
   `--src-root`.
2. Build `HFRVLAPolicy` from `HuggingFaceVLA/smolvla_libero` or another
   LIBERO-adapted SmolVLA slow-planner checkpoint. The raw
   `lerobot/smolvla_base` checkpoint has a 6D action/state and `camera1/2/3`
   feature contract, so it is only a warm start for fine-tuning, not a valid
   frozen LIBERO slow planner.
3. Validate and force LIBERO features:
   - `observation.images.image`: `(3, 256, 256)`
   - `observation.images.image2`: `(3, 256, 256)`
   - `observation.state`: `(8,)`
   - `action`: `(7,)`
4. For each episode, run SmolVLA chunk inference once and capture:
   - `a_base`
   - `z_goal`
   - `z_phase`
   - `k_idx_norm`
5. Batch-forward wrist images through DINOv3 to get `(196, 384)` patches.
6. Write every frame to the derived LeRobotDataset v3.
7. Call `dst.finalize()`.

Sharded recording is supported with `--ep-from` and `--ep-to`. The fast merger:

```text
scripts/merge_hfrvla_shards_fast.py
```

merges shards offline by manipulating metadata and parquet files directly. This
avoids accidentally using `LeRobotDataset(repo_id=..., root=shard)` with
synthetic local-only repo IDs, which can fall through to Hub downloads.

## 7. HFRVLA Policy Architecture

`HFRVLAConfig` extends `SmolVLAConfig` with:

- Frozen SmolVLA switch: `freeze_smolvla=True`
- DINOv3 local loading fields:
  - `dinov3_local_repo`
  - `dinov3_local_weights`
  - `dinov3_arch`
- DINOv3 shape defaults: `224` image size, `196` patches, `384` hidden dim.
- Active correction modes:
  - `fast_wrist`: stateless current-step HFRVLA correction head.
  - `fast_wrist_chunk`: chunk-aware HFRVLA correction head using schema v3
    chunk fields.
- Legacy mode:
  - `gated`: original GRU/gate/contact path kept for old checkpoints only.
- Fast wrist correction controls:
  - `fast_residual_alpha`
  - `delta_max`
  - `fast_wrist_loss_lambda_delta`
  - `fast_wrist_loss_lambda_final`
  - `fast_wrist_loss_lambda_residual`
  - `fast_wrist_loss_lambda_clip`
- `seq_len=2` is the current HFRVLA previous/current training window; the
  fast-cache itself remains frame-level and does not bake in `seq_len`.
- `inference_disable_fast` for alignment tests.

`HFRVLAPolicy`:

- Inherits `SmolVLAPolicy`.
- Freezes `self.model` when `freeze_smolvla=True`.
- Wraps `vlm_with_expert.forward` to capture final hidden summaries because
  SmolVLA calls `.forward(...)` directly and normal PyTorch hooks would not fire.
- Uses `select_action()` to pop SmolVLA chunk actions and apply the fast correction
  on every control step.
- Uses `forward()` for training on the LeRobot-native batch layout with
  `observation.extra.*` keys.
- Returns only fast-module trainable parameters from `get_optim_params()`.

Active `FastWristResidualModule`:

```text
DINOv3 wrist patches
  -> Linear(384 -> 256)
  -> wrist patch pooling / attention
  -> fuse(wrist context, proprio, a_base, k_idx_norm, z_phase, optional previous delta)
  -> delta_a
```

`FastWristChunkResidualModule` adds access to the generated SmolVLA base-action
chunk and chunk-step metadata. Both active correction modules are stateless: no
GRU, no learned gate, and no contact auxiliary head.

## 8. Training And Evaluation Commands

Working directory:

```bash
cd ~/Patrick/VLA_research/Hierachical_fast_reactive
```

Smoke record:

```bash
~/Robotic_infra/lerobot/.venv/bin/python scripts/record_hfrvla_libero.py \
    --src-repo-id HuggingFaceVLA/libero \
    --src-root /home/hucenrotia/.cache/huggingface/lerobot/HuggingFaceVLA/libero \
    --out-repo-id HFRVLA_libero_smoke \
    --out-root checkpoints/HFRVLA_libero_smoke \
    --dinov3-repo checkpoints/dinov3_src \
    --dinov3-weights checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --max-episodes 5 \
    --device cpu
```

Alignment test before long training:

```bash
~/Robotic_infra/lerobot/.venv/bin/python scripts/test_alignment.py \
    --suite libero_spatial \
    --task-ids 0,1,2 \
    --n-episodes 5
```

Training smoke:

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

Evaluation smoke:

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

Use `--eval.batch_size` for LIBERO parallelism. `--env.num_envs` is not a valid
LiberoEnv field.

## 9. Common Pitfalls

- Plugin edits require editable reinstall into `~/Robotic_infra/lerobot/.venv`.
- Do not normalize `observation.extra.*` through `input_features`; they are
  precomputed features, not raw policy inputs for the normalizer.
- Use `--num_workers=0` in restricted sandboxes that block PyTorch
  multiprocessing.
- Prefer local dataset roots and offline flags when working without network.
- Before spending time on training, run the alignment test with
  `inference_disable_fast=True` packaging. If it gets 0% on all tasks, debug
  action shape, observation keys, or normalization before training.
- The fast module is small; dataset I/O often dominates GPU utilization.
