# HFRVLA training and eval (lerobot-native)

Use `lerobot-train` for training and `lerobot-eval` for evaluation.
Recording uses a wrapper script that writes a standard LeRobotDataset v3.

Active scripts:
- `scripts/train_smolvla_libero_baseline.sh` - optional fallback to fine-tune `lerobot/smolvla_base` on LIBERO if the published checkpoint is unsuitable
- `scripts/record_hfrvla_libero.py` - one-shot or sharded recording (`--ep-from/--ep-to`)
- `scripts/merge_hfrvla_shards_fast.py` - offline stitcher for parallel shards (file/metadata only, no re-encode)
- `scripts/build_hfrvla_fastcache.py` - optional derived mmap cache for faster offline HFRVLA training
- `scripts/generate_training_presentation.py` - builds the open-slide deck and bundles it into `docs/training_presentation.html`
- `scripts/train_hfrvla_libero_merged.sh` - pinned training entrypoint for the verified merged HFRVLA dataset
- `scripts/train_via_lerobot.py` - wrapper around `lerobot-train` that adds curriculum control
- `scripts/package_hfrvla_checkpoint.py` - bundle SmolVLA + fast weights into a `lerobot-eval` checkpoint dir (`--disable-fast` for alignment tests)
- `scripts/check_hfrvla_training_contract.py` - offline train-time dataset/window/target contract check
- `scripts/test_alignment.py` - I/O alignment check against the frozen SmolVLA slow planner before training
- `scripts/run_planner_delay_eval_sweep.py` - paper-facing inference-delay stress sweep
- `lerobot-eval` - normal CLI evaluation; delay sweeps use `scripts/lerobot_eval_hfrvla.py` so debug metrics and direct-SmolVLA action delay can be recorded

Legacy scripts live in `scripts/legacy/`; see `scripts/legacy/README.md`.

Codex-local automation:
- `.agents/plugins/marketplace.json` installs the repo-local `hfrvla-training-docs-hook` plugin.
- `docs/presentations/hfrvla-training-open-slide/` is the open-slide workspace for the training deck; edit `slides/hfrvla-training/index.tsx` for visual/content changes.
- `.agents/plugins/plugins/hfrvla-training-docs-hook/hooks.json` runs after Codex file edits and rebuilds `docs/training_presentation.html` by calling the open-slide generator when this file, the generator, or the open-slide deck/viewer source changed.
- Run `python3 scripts/generate_training_presentation.py --check` before committing documentation changes if you want an explicit freshness check.

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
| LIBERO SmolVLA slow planner | `HuggingFaceVLA/smolvla_libero` or a local snapshot of it |
| Wandb login | `~/Robotic_infra/lerobot/.venv/bin/wandb login` |

Working directory:

```bash
cd ~/Patrick/VLA_research/Hierachical_fast_reactive
```

After changing files under `policy/lerobot_policy_hfrvla/`, reinstall the
editable plugin before launching training:

```bash
cd ~/Robotic_infra/lerobot
uv pip install -e ~/Patrick/VLA_research/Hierachical_fast_reactive/policy/lerobot_policy_hfrvla
cd ~/Patrick/VLA_research/Hierachical_fast_reactive
```

For offline/local runs, set:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HFRVLA_TMP_ROOT=$HOME/tmp/hfrvla
export HF_DATASETS_CACHE=$HFRVLA_TMP_ROOT/hf_datasets
export TMPDIR=$HFRVLA_TMP_ROOT/tmp
```

## 1. Choose the SmolVLA slow planner

Use the published LIBERO-adapted SmolVLA checkpoint as HFRVLA's frozen
System-2 slow planner:

```text
HuggingFaceVLA/smolvla_libero
```

Its `config.json` already matches the LIBERO contract:
`observation.images.image`, `observation.images.image2`,
`observation.state(8)`, and `action(7)`. The recording, packaging, and
alignment scripts use this model as their default `--smolvla`.

If you need an offline run, download or cache this model first and pass the
local snapshot path as `--smolvla`. Only use `scripts/train_smolvla_libero_baseline.sh`
as a fallback if you intentionally want to produce your own LIBERO SmolVLA
checkpoint from `lerobot/smolvla_base`.

Optional fallback fine-tune command:

```bash
DATASET_ROOT=/home/hucenrotia/.cache/huggingface/lerobot/HuggingFaceVLA/libero \
STEPS=100000 BATCH_SIZE=4 DEVICE=cuda \
scripts/train_smolvla_libero_baseline.sh
```

## 2. Record the HFRVLA shared dataset

> **Why a custom dataset?** SmolVLA is frozen at inference; the fast module needs the SmolVLA *outputs* (the planned chunk = `a_base`) plus DINOv3 wrist patches plus the two hooked `z_goal` / `z_phase` tensors. These are all computed during recording so training is a clean offline pass over a regular LeRobotDataset v3 (no SmolVLA forward in the train loop).

> **Slow planner contract:** `--smolvla` must point to a SmolVLA checkpoint that is already adapted to LIBERO's feature contract: `observation.images.image`, `observation.images.image2`, `observation.state` shape `(8,)`, and `action` shape `(7,)`. The raw `lerobot/smolvla_base` checkpoint is a warm-start model with a 6D action/state and `camera1/2/3` config; it is not a valid frozen LIBERO slow planner. If you change the slow planner, recollect the HFRVLA dataset so `a_base`, `z_goal`, and `z_phase` match it.

> **Sharded recording (recommended for the full ~1700-episode LIBERO):** parallel processes write disjoint episode ranges to separate `--out-root` shards, then `scripts/merge_hfrvla_shards_fast.py` stitches them into one final dataset by hardlinking parquet files and renaming offsets. The merger only rewrites the `task_index` column for shards whose local task indices need shifting against the merged global table. See §2.3 for the merge command.

### 2.1 Single-process full recording

```bash
~/Robotic_infra/lerobot/.venv/bin/python scripts/record_hfrvla_libero.py \
    --src-repo-id HuggingFaceVLA/libero \
    --out-repo-id HFRVLA_libero_v1 \
    --out-root checkpoints/HFRVLA_libero_v1 \
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
    --dinov3-repo checkpoints/dinov3_src \
    --dinov3-weights checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --max-episodes 5 \
    --device cpu
```

### 2.2 Sharded recording (parallel processes)

Split the source episodes into N disjoint ranges, write each to its own `--out-root`. On a 32-core machine 3-4 parallel processes is the sweet spot (CPU-bound: parquet write + JPEG encode dominate).

```bash
SRC_ROOT=/home/hucenrotia/.cache/huggingface/lerobot/HuggingFaceVLA/libero
SMOL=HuggingFaceVLA/smolvla_libero
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

### 2.3 Merge shards

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

Current verified merged dataset:

```text
checkpoints/HFRVLA_libero_v1_merged_reindexed
episodes=1693, frames=273465, tasks=40
z_goal=(960,), z_phase=(480,)
data episode_index=0..1692, index=0..273464
```

Use this root for the first real training run.

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
| `--smolvla` | `HuggingFaceVLA/smolvla_libero` | LIBERO-adapted SmolVLA slow planner used for chunk inference. |
| `--allow-feature-remap` | false | Debug-only escape hatch for non-LIBERO SmolVLA configs; do not use for real HFRVLA collection. |
| `--dinov3-repo` | required | Local clone of facebookresearch/dinov3. |
| `--dinov3-weights` | required | DINOv3 `.pth` weight file. |
| `--dinov3-arch` | `dinov3_vits16` | torch.hub entry-point name. |
| `--wrist-key` | `observation.images.image2` | Eye-in-hand camera key in source dataset. |
| `--fps` | 10 | Must equal the source dataset fps. |
| `--dino-dtype` | `float32` | Set to `float16` to reduce storage. |
| `--device` | `cuda` | Inference device. |

## 3. Train-time contract check (run after §2)

**Goal:** verify the recorded HFRVLA dataset can feed the actual
`lerobot-train` path before spending GPU time. This check is offline and fast:
it loads one sample through LeRobot's `delta_timestamps`, applies the same
policy preprocessor used during training, and confirms:

- raw metadata has the HFRVLA/LIBERO features (`state=8`, `action=7`, `a_base=7`, DINOv3 patches, z features).
- windowed tensors have shape `action/a_base/target_delta = (B, seq_len, 7)`.
- `target_delta = normalized(action) - a_base` is finite with zero residual output.
- LeRobot scalar collapse for `k_idx_norm/contact_label` is handled as `(B, seq_len)`.

```bash
HFRVLA_TMP_ROOT=$HOME/tmp/hfrvla \
~/Robotic_infra/lerobot/.venv/bin/python scripts/check_hfrvla_training_contract.py \
    --dataset-repo-id HFRVLA_libero_v1 \
    --dataset-root checkpoints/HFRVLA_libero_v1_merged_reindexed \
    --seq-len 8 \
    --device cpu \
    --json-out outputs/alignment_eval/training_contract_summary_merged_fixed.json
```

Expected key lines:

```text
observation.state: (1, 8, 8)
action: (1, 8, 7)
observation.extra.a_base: (1, 8, 7)
zero-fast target: target_delta=(1, 8, 7) ...
[hfrvla-contract] PASS
```

This check proves the train-time tensor contract, not policy quality. If the
dataset was recorded with the wrong slow planner, shape checks can still pass
while `a_base/z_goal/z_phase` are semantically wrong; recollect after changing
`--smolvla`.

## 4. Alignment eval (run before §5)

**Goal:** confirm the HFRVLA wrapper's I/O matches LIBERO env expectations *before* spending hours training the fast module. Failure here means action shapes / normalization / observation keys are mis-wired and training will produce a broken policy.

**Mechanism:** the script first evaluates the LIBERO-adapted SmolVLA slow
planner directly. It then packages HFRVLA with `inference_disable_fast=True`,
so `HFRVLAPolicy.select_action()` short-circuits to SmolVLA's `a_base` (the
popped chunk action) before touching the fast module. The zero-fast wrapper
should match the direct slow-planner baseline within small-N variance.

```bash
~/Robotic_infra/lerobot/.venv/bin/python scripts/test_alignment.py \
    --suite libero_spatial \
    --task-ids 0,1,2 \
    --n-episodes 5
```

This runs direct SmolVLA baseline eval, packages `checkpoints/alignment_test/`
(HFRVLA config + SmolVLA slow-planner weights, fast module randomly initialized
but never invoked), runs zero-fast HFRVLA eval against `libero_spatial`
task_ids 0/1/2 with 5 episodes each, and prints baseline vs zero-fast
success. Add `--skip-baseline` only when you already have a fresh direct
baseline number. The script refuses raw `lerobot/smolvla_base` by default
because its saved feature contract is not LIBERO-shaped.

**Pass criteria:**
- Direct SmolVLA baseline has nonzero success on at least one smoke task. If it is 0%, debug the slow-planner checkpoint/eval setup first.
- Zero-fast HFRVLA success is in the same ballpark as the direct baseline. If baseline works but zero-fast collapses, debug wrapper I/O, normalization, and action postprocessing.
- All 0% on both paths means this is not yet evidence about the fast module; the slow planner or LIBERO eval path is failing.

**Pass → run §5 training. Fail → debug before training.**

## 5. Train with lerobot-train via wrapper

### Fast-cache backend for speed-sensitive runs

The verified LeRobotDataset v3 remains the canonical dataset:

```text
checkpoints/HFRVLA_libero_v1_merged_reindexed
```

For training speed, build a derived local cache that projects only the HFRVLA
offline columns into contiguous `.npy` arrays. Large cached feature columns
`z_goal`, `z_phase`, and `dino_patches` are stored as `float16` and loaded with
`numpy.memmap`, which avoids the slow random Arrow/Python path that dominated
the run07 timing.

The fast module casts cached `float16` features to its parameter dtype at the
module boundary, so the cache can stay compact while training remains
`float32` unless AMP is enabled. Fast-cache schema v3 adds
`observation.extra.a_base_chunk=(50,7)`, `observation.extra.chunk_step_idx=(1,)`,
optional `observation.extra.chunk_age_steps=(1,)` / `chunk_age_norm=(1,)`,
and top-level `chunk_len=50` metadata for chunk-aware FWR. New recordings store
the generated SmolVLA chunk directly; older recordings without these fields are
still supported by reconstructing chunks from frame-level `a_base`. It keeps the schema
v2 static Stage B labels, `observation.extra.y_correct` and
`observation.extra.y_preserve`, plus their offline error thresholds in
`meta.json`. Schema v1/v2 caches still load for older modes, but
`RESIDUAL_MERGE_MODE=fast_wrist_chunk` requires schema v3 chunk fields. The
fast-cache is frame-level storage: it does not bake in a sequence length.
`SEQ_LEN` is a training-time sampling/windowing choice passed through
`--policy.seq_len`.

Build once per source dataset:

```bash
~/Robotic_infra/lerobot/.venv/bin/python scripts/build_hfrvla_fastcache.py \
    --source-root checkpoints/HFRVLA_libero_v1_merged_reindexed \
    --cache-root checkpoints/HFRVLA_libero_v1_fastcache_v3_plan50 \
    --chunk-len 50 \
    --correct-quantile 0.80 \
    --preserve-quantile 0.50
```

Train against the cache:

```bash
RUN_NAME=hfrvla_run_fastcache_seq4 \
WANDB_ENABLE=true \
HFRVLA_TMP_ROOT=$HOME/tmp/hfrvla \
HFRVLA_DATASET_BACKEND=fastcache \
HFRVLA_FASTCACHE_ROOT=checkpoints/HFRVLA_libero_v1_fastcache_v3_plan50 \
SEQ_LEN=4 \
BATCH_SIZE=256 \
NUM_WORKERS=8 \
scripts/run_hfrvla_training_foreground.sh
```

Expected log line:

```text
[hfrvla-train] fast-cache dataset backend enabled root=...
```

Stage C adds a second fast-cache root built from successful closed-loop
`zero_fast` rollouts. Record the rollout dataset, then build its cache with
static preserve labels:

```bash
~/Robotic_infra/lerobot/.venv/bin/python scripts/record_zero_fast_rollouts.py \
    --policy-path checkpoints/hfrvla_zero_fast_packaged \
    --out-root checkpoints/HFRVLA_libero_v1_zero_fast_rollouts \
    --task-suite libero_spatial \
    --task-ids 0,1,2,3,4,5,6,7,8,9 \
    --episodes-per-task 10

~/Robotic_infra/lerobot/.venv/bin/python scripts/build_hfrvla_fastcache.py \
    --source-root checkpoints/HFRVLA_libero_v1_zero_fast_rollouts \
    --cache-root checkpoints/HFRVLA_libero_v1_fastcache_v2_rollouts \
    --static-y-preserve
```

Train with both roots by setting `HFRVLA_FASTCACHE_ROLLOUT_ROOT`; leaving it
unset preserves the Stage A/B single-cache behavior:

```bash
HFRVLA_DATASET_BACKEND=fastcache \
HFRVLA_FASTCACHE_ROOT=checkpoints/HFRVLA_libero_v1_fastcache_v2 \
HFRVLA_FASTCACHE_ROLLOUT_ROOT=checkpoints/HFRVLA_libero_v1_fastcache_v2_rollouts \
USE_STAGE_B=true \
LOSS_LAMBDA_PRESERVE_ZERO=3.0 \
SEQ_LEN=4 \
scripts/train_hfrvla_libero_merged.sh
```

If `HFRVLA_DATASET_BACKEND` is unset, the launcher keeps the LeRobot-native
backend and should still print:

```text
[hfrvla-train] offline dataset column pruning enabled
```

One-step smoke for validating the cache/preprocessor/model path after code
changes:

```bash
RUN_NAME=hfrvla_fastcache_smoke \
WANDB_ENABLE=false \
HFRVLA_TMP_ROOT=$HOME/tmp/hfrvla \
HFRVLA_DATASET_BACKEND=fastcache \
HFRVLA_FASTCACHE_ROOT=checkpoints/HFRVLA_libero_v1_fastcache_v2 \
SEQ_LEN=4 \
BATCH_SIZE=2 \
NUM_WORKERS=0 \
STEPS=1 \
SAVE_FREQ=100 \
LOG_FREQ=1 \
DEVICE=cpu \
scripts/run_hfrvla_training_foreground.sh
```

Full run:

```bash
RUN_NAME=hfrvla_run04 \
WANDB_ENABLE=false \
HFRVLA_TMP_ROOT=$HOME/tmp/hfrvla \
scripts/launch_hfrvla_training_systemd.sh
```

The launcher uses `systemd-run --user` so the train process is not inside the
GNOME Terminal scope. This matters on this workstation because `systemd-oomd`
can kill the terminal scope under sustained memory pressure, which also kills
children launched with `nohup`.

Follow logs with:

```bash
tail -f outputs/logs/hfrvla_run04.log
systemctl --user status hfrvla_run04
```

Direct foreground run, for debugging only:

```bash
OUT_DIR=checkpoints/hfrvla_run01 \
WANDB_ENABLE=false \
HFRVLA_TMP_ROOT=$HOME/tmp/hfrvla \
scripts/train_hfrvla_libero_merged.sh
```

100-step smoke:

```bash
STEPS=100 \
BATCH_SIZE=4 \
NUM_WORKERS=0 \
WARMUP_STEPS=10 \
JOINT_STEPS=80 \
REFINE_STEPS=10 \
SAVE_FREQ=100 \
LOG_FREQ=10 \
OUT_DIR=outputs/train_hfrvla_smoke \
scripts/train_hfrvla_libero_merged.sh
```

The wrapper adds curriculum control on top of `lerobot-train`; all other flags are vanilla `lerobot-train` flags.
Use `--num_workers=0` in restricted sandboxes that block PyTorch multiprocessing.

The shell entrypoint pins the `HuggingFaceVLA/smolvla_libero` architecture
contract:

```text
offline_training_mode=true
offline_zgoal_dim=960
offline_zphase_dim=480
vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Instruct
expert_width_multiplier=0.5
num_vlm_layers=0
load_vlm_weights=false
```

`offline_training_mode=true` is intentional: the recorded shared dataset
already contains `a_base`, `z_goal`, `z_phase`, and `dino_patches`, so the
train process does not construct the frozen SmolVLA slow planner or the DINOv3
backbone. Do not drop these dimension overrides for this dataset: the recorded
shared dataset has `observation.extra.z_phase=(480,)`; raw `smolvla_base`
defaults would construct the fast module for `(720,)`.

The `scripts/train_via_lerobot.py` wrapper also prunes offline HFRVLA dataset
columns after local load: raw `observation.images.*` columns are excluded from
delta windowing and DataLoader batches, while `state/action/a_base/k_idx/z_*`,
`dino_patches`, and `contact_label` remain windowed to `seq_len`. This keeps
training on the small fast module instead of spending every step collating
unused images.

The entrypoint keeps LeRobot's policy training preset enabled so the optimizer
uses `policy.get_optim_params()` and trains only the fast module. LR values are
therefore passed as `--policy.optimizer_*` fields, not top-level
`--optimizer.*` fields.

### Curriculum and windowing knobs

| Flag | Default | Description |
|---|---|---|
| `--policy.curriculum_warmup_steps` | 1000 | Stage 0 length: delta only, heads frozen. |
| `--policy.curriculum_joint_steps` | 49000 | Stage 1 length: all losses active. |
| `--policy.curriculum_refine_steps` | 10000 | Stage 2 length: LR is reduced by 10x at entry. |
| `--policy.seq_len` | 8 | Training window length; gated mode consumes the full window, FWR-v1 requires at least 2 frames for previous/current conditioning, and FWR-v2 chunk mode uses the current frame plus the v3 full base chunk. |

### Fast Wrist Residual modes

Set `RESIDUAL_MERGE_MODE=fast_wrist` to train the FWR-v1 feed-forward
correction head. The deprecated `a2c2` value is still accepted as an alias for
old scripts and checkpoints. This mode removes the gate, contact head, GRU,
conservative preserve losses, and Stage B labels from the objective. It
supervises both the raw current-step residual and the deployed merged action:

```text
delta_target = action_t - a_base_t
delta_exec = FAST_RESIDUAL_ALPHA * clip(delta_pred)
a_hat = a_base_t + delta_exec
loss = lambda_delta * SmoothL1(delta_pred, delta_target)
     + lambda_final * SmoothL1(a_hat, action_t)
     + lambda_residual * mean(delta_exec^2)
     + lambda_clip * mean(relu(abs(delta_pred) - residual_cap))
```

At inference it applies only the configured clipped residual:

```text
a_final = a_base + FAST_RESIDUAL_ALPHA * clip(delta_pred)
```

FWR-v1 can train from the frame-level v2/v3 fast-cache and uses a two-frame
training window for previous-action conditioning:

```bash
RUN_NAME=hfrvla_fwr_wrist_seq2 \
WANDB_ENABLE=false \
HFRVLA_DATASET_BACKEND=fastcache \
HFRVLA_FASTCACHE_ROOT=checkpoints/HFRVLA_libero_v1_fastcache_v2 \
RESIDUAL_MERGE_MODE=fast_wrist \
FAST_RESIDUAL_ALPHA=1.0 \
FAST_RESIDUAL_USE_LATENT_CONTEXT=true \
SEQ_LEN=2 \
BATCH_SIZE=256 \
NUM_WORKERS=8 \
STEPS=10000 \
scripts/run_hfrvla_training_foreground.sh
```

Set `RESIDUAL_MERGE_MODE=fast_wrist_chunk` for FWR-v2. This mode requires a
schema v3 full-chunk cache and lets the residual head attend over all 50 frozen
SmolVLA base actions using action, chunk-relative, and cosine features while
still predicting only the current-step `delta_a`:

```bash
RUN_NAME=hfrvla_fwr_chunk_seq2 \
WANDB_ENABLE=false \
HFRVLA_DATASET_BACKEND=fastcache \
HFRVLA_FASTCACHE_ROOT=checkpoints/HFRVLA_libero_v1_fastcache_v3_plan50 \
RESIDUAL_MERGE_MODE=fast_wrist_chunk \
FAST_RESIDUAL_ALPHA=1.0 \
FAST_RESIDUAL_USE_LATENT_CONTEXT=true \
SEQ_LEN=2 \
BATCH_SIZE=256 \
NUM_WORKERS=8 \
STEPS=10000 \
SAVE_FREQ=25000 \
scripts/run_hfrvla_training_foreground.sh
```

For evaluation, package the checkpoint with
`scripts/package_hfrvla_checkpoint.py` and sweep `FAST_RESIDUAL_ALPHA` or
`--policy.fast_residual_alpha` over `0, 0.25, 0.5, 0.75, 1.0`.

### HFRVLA LR/weight-decay W&B sweep

Use this launcher to run the 20-run HFRVLA learning-rate and weight-decay
matrix with W&B enabled:

```bash
scripts/run_hfrvla_lr_wd_sweep.sh
```

Default fixed settings:

```text
BATCH_SIZE=512
STEPS=50000
WARMUP_STEPS=1000
JOINT_STEPS=49000
REFINE_STEPS=0
SAVE_FREQ=25000
LOG_FREQ=100
WANDB_ENABLE=true
WANDB_PROJECT=hfrvla
PARALLEL_JOBS=1
```

The sweep grid is:

```text
LR: 1e-5, 3e-5, 1e-4, 3e-4, 5e-4
WD: 0, 1e-5, 1e-4, 3e-4
```

After the 2026-05-29 matched `plan=50`, `exec/replan=8` evaluation, the
single-run default is `LR=3e-4` and `WEIGHT_DECAY=1e-5`.

Run names use the `hfrvla_*` prefix, for example
`hfrvla_lr3e_4_wd1e_5_b512_50k`. New FWR runs should set
`RESIDUAL_MERGE_MODE=fast_wrist` or `fast_wrist_chunk`; historical sweep
scripts may still carry the deprecated `a2c2` alias for older checkpoints.

Preview all commands without launching training:

```bash
DRY_RUN=true scripts/run_hfrvla_lr_wd_sweep.sh
```

Run two W&B jobs at a time when the GPU has headroom:

```bash
PARALLEL_JOBS=2 scripts/run_hfrvla_lr_wd_sweep.sh
```

Parallel runs write per-run console logs under
`outputs/hfrvla_lr_wd_sweep_logs/<timestamp>/`. W&B still receives one run per
hyperparameter row. Prefer `PARALLEL_JOBS=2` before increasing batch size: a
2026-05-28 probe on this workstation measured roughly 17.7 step/s at batch 512,
8.5 step/s at batch 1024, and 4-5 step/s at batch 2048, so larger batches did
not improve samples/sec for the current small trainable module.

Resume from a specific row after interruption:

```bash
START_AT=hfrvla_lr3e_4_wd1e_5_b512_50k scripts/run_hfrvla_lr_wd_sweep.sh
```

### Current A2C2 smoke, 10k, and alpha-sweep results

Verified on 2026-05-25 with the rebuilt frame-level cache
`checkpoints/HFRVLA_libero_v1_fastcache_v2`:

| Check | Result |
|---|---|
| Train run | `hfrvla_a2c2_smoke_seq2_500_gpu`, 500 steps |
| Throughput | 500 steps in 19 s; stable logs around 25-32 step/s |
| VRAM | about 2.9 GiB during smoke training, below the 15 GiB budget |
| Loss | `loss`/`delta` dropped from about 0.44 to 0.15 |
| Checkpoint | `checkpoints/hfrvla_a2c2_smoke_seq2_500_gpu/checkpoints/000500/pretrained_model` |
| Packaged eval artifact | `checkpoints/hfrvla_a2c2_smoke_seq2_500_gpu_packaged` |
| Partial eval | `libero_spatial`, task 0, 1 episode, 0/1 success; `eval_info.json` and video were written |

The 500-step eval is only a format/alignment smoke. It is not evidence of final
policy quality.

The first longer A2C2-Wrist run used the same frame-level cache, `seq_len=2`,
`A2C2_ALPHA=1.0`, latent context enabled, batch size 256, and 8 dataloader
workers:

| Check | Result |
|---|---|
| Train run | `hfrvla_a2c2_wrist_seq2_10k`, 10000 steps |
| Throughput | 10000 steps in 5 min 31 s; about 30.1 step/s average |
| VRAM | about 2.9 GiB during training; formal CUDA eval stayed below the 15 GiB budget |
| Loss | final logged `loss`/`delta` about 0.021 |
| Packaged eval artifact | `checkpoints/hfrvla_a2c2_wrist_seq2_10k_packaged` |
| Partial CUDA eval | `libero_spatial`, task 0, 1 episode, 1/1 success |
| Full CUDA eval | `outputs/eval_a2c2_wrist_seq2_10k_spatial_cuda`, 26/50 success = 52.0%, 451.1 s total, 9.02 s/episode |
| Per-task successes | task ids 0-9: `3, 4, 4, 4, 1, 0, 2, 4, 3, 1` out of 5 each |

This 10k A2C2-Wrist result improves over the previously recorded frozen/zero
fast baseline (`zero_fast`: 23/50 = 46.0%) and the earlier Stage A v2 residual
result (24/50 = 48.0%). The gain is modest but useful as a first wrist-only
correction baseline without GRU, gate, contact loss, or preserve-zero losses.

The follow-up 20k and 30k runs used the same architecture, same frame-level
fast-cache, `seq_len=2`, latent context enabled, batch size 256, 8 workers, and
the same packaged CUDA eval protocol over all 10 `libero_spatial` tasks with 5
episodes each. Training and eval both stayed under the requested 22000 MiB GPU
limit: training polling was about 2.9 GiB, and eval polling peaked at about
8.7 GiB.

| Check | 20k | 30k |
|---|---:|---:|
| Train run | `hfrvla_a2c2_wrist_seq2_20k` | `hfrvla_a2c2_wrist_seq2_30k` |
| Train time | about 10 min 44 s | about 16 min 18 s |
| Final logged loss | about 0.020-0.021 | about 0.019 |
| Packaged artifact | `checkpoints/hfrvla_a2c2_wrist_seq2_20k_packaged` | `checkpoints/hfrvla_a2c2_wrist_seq2_30k_packaged` |

Full alpha sweep results, evaluated on 2026-05-26:

| Checkpoint | Alpha | Success | Eval output | Per-task successes, task ids 0-9 |
|---|---:|---:|---|---|
| 20k | 0.00 | 23/50 = 46.0% | `outputs/eval_a2c2_wrist_seq2_20k_spatial_alpha_0_cuda` | `3, 2, 0, 3, 2, 0, 3, 3, 3, 4` |
| 20k | 0.25 | 23/50 = 46.0% | `outputs/eval_a2c2_wrist_seq2_20k_spatial_alpha_025_cuda` | `3, 3, 2, 3, 1, 1, 4, 3, 1, 2` |
| 20k | 0.50 | 22/50 = 44.0% | `outputs/eval_a2c2_wrist_seq2_20k_spatial_alpha_05_cuda` | `2, 3, 1, 3, 3, 0, 2, 4, 1, 3` |
| 20k | 0.75 | 19/50 = 38.0% | `outputs/eval_a2c2_wrist_seq2_20k_spatial_alpha_075_cuda` | `1, 5, 1, 3, 2, 0, 2, 3, 2, 0` |
| 20k | 1.00 | 15/50 = 30.0% | `outputs/eval_a2c2_wrist_seq2_20k_spatial_alpha_10_cuda` | `1, 4, 1, 1, 0, 0, 2, 3, 2, 1` |
| 30k | 0.00 | 23/50 = 46.0% | `outputs/eval_a2c2_wrist_seq2_30k_spatial_alpha_0_cuda` | `3, 2, 0, 3, 2, 0, 3, 3, 3, 4` |
| 30k | 0.25 | 23/50 = 46.0% | `outputs/eval_a2c2_wrist_seq2_30k_spatial_alpha_025_cuda` | `1, 4, 1, 3, 3, 0, 4, 4, 1, 2` |
| 30k | 0.50 | 26/50 = 52.0% | `outputs/eval_a2c2_wrist_seq2_30k_spatial_alpha_05_cuda` | `3, 3, 1, 3, 3, 1, 3, 4, 2, 3` |
| 30k | 0.75 | 25/50 = 50.0% | `outputs/eval_a2c2_wrist_seq2_30k_spatial_alpha_075_cuda` | `1, 5, 3, 3, 2, 1, 5, 4, 1, 0` |
| 30k | 1.00 | 21/50 = 42.0% | `outputs/eval_a2c2_wrist_seq2_30k_spatial_alpha_10_cuda` | `2, 4, 2, 1, 2, 0, 2, 3, 2, 3` |

Interpretation:

- The current best in this sweep is 30k with `alpha=0.5`: 26/50 = 52.0%.
- 20k did not improve over base-only; full-strength residual (`alpha=1.0`)
  was harmful.
- 30k learns a usable correction only at intermediate alpha. This suggests the
  raw residual head is directionally useful but too large or too poorly
  calibrated to deploy at full strength.
- The simplest current thesis baseline remains defensible: frozen SmolVLA slow
  planner plus wrist-only reactive correction, without gate, GRU, contact loss,
  or preserve losses.

Follow-up spatial+object eval on 2026-05-26:

| Policy | Spatial | Object | Combined | Eval outputs |
|---|---:|---:|---:|---|
| HFRVLA 30k, `alpha=0.5` | 26/50 = 52.0% | 24/50 = 48.0% | 50/100 = 50.0% | `outputs/eval_a2c2_wrist_seq2_30k_spatial_alpha_05_cuda`, `outputs/eval_a2c2_wrist_seq2_30k_object_alpha_05_cuda` |
| Original `HuggingFaceVLA/smolvla_libero` | 38/50 = 76.0% | 49/50 = 98.0% | 87/100 = 87.0% | `outputs/eval_smolvla_libero_spatial_5ep_cuda`, `outputs/eval_smolvla_libero_object_5ep_cuda` |

Per-task successes:

| Policy / suite | Task ids 0-9 |
|---|---|
| HFRVLA 30k `alpha=0.5`, spatial | `3, 3, 1, 3, 3, 1, 3, 4, 2, 3` |
| HFRVLA 30k `alpha=0.5`, object | `1, 1, 3, 2, 3, 4, 0, 3, 5, 2` |
| Original SmolVLA, spatial | `3, 4, 4, 3, 5, 2, 5, 5, 4, 3` |
| Original SmolVLA, object | `5, 5, 5, 5, 5, 4, 5, 5, 5, 5` |

The important diagnostic is that `alpha=0` in the HFRVLA package is not the
same policy as default `HuggingFaceVLA/smolvla_libero`. The packaged HFRVLA
config uses `n_action_steps=50`, so `select_action()` consumes a 50-step
SmolVLA action chunk before replanning. The original `smolvla_libero` config
uses `n_action_steps=1`, so it replans every environment step. This explains why
the HFRVLA base-only rows are only 23/50 = 46.0% on spatial while original
SmolVLA reaches 38/50 = 76.0% on the same 5-episode-per-task spatial protocol.
That gap is not caused by the fast residual; it is the action-chunk execution
regime the correction model is meant to improve.

Follow-up matched-chunk eval on 2026-05-27:

Both policies were evaluated with `--policy.n_action_steps=8` on the same
seed-42 protocol, 10 tasks x 5 episodes for each suite. HFRVLA also used
`--policy.a2c2_alpha=0.5`.

| Policy | Spatial | Object | Combined | Eval outputs |
|---|---:|---:|---:|---|
| HFRVLA 30k, `alpha=0.5`, `n_action_steps=8` | 34/50 = 68.0% | 47/50 = 94.0% | 81/100 = 81.0% | `outputs/eval_hfrvla_a2c2_wrist_seq2_30k_n8_alpha_05_spatial_cuda`, `outputs/eval_hfrvla_a2c2_wrist_seq2_30k_n8_alpha_05_object_cuda` |
| Original `HuggingFaceVLA/smolvla_libero`, `n_action_steps=8` | 32/50 = 64.0% | 45/50 = 90.0% | 77/100 = 77.0% | `outputs/eval_smolvla_libero_n8_spatial_cuda`, `outputs/eval_smolvla_libero_n8_object_cuda` |

Per-task successes:

| Policy / suite | Task ids 0-9 |
|---|---|
| HFRVLA 30k `alpha=0.5`, `n_action_steps=8`, spatial | `3, 4, 4, 4, 2, 3, 4, 4, 2, 4` |
| HFRVLA 30k `alpha=0.5`, `n_action_steps=8`, object | `5, 5, 5, 4, 5, 4, 5, 4, 5, 5` |
| SmolVLA `n_action_steps=8`, spatial | `3, 4, 4, 3, 4, 0, 4, 3, 3, 4` |
| SmolVLA `n_action_steps=8`, object | `5, 5, 4, 5, 5, 3, 5, 5, 5, 3` |

Interpretation: setting both policies to `n_action_steps=8` removes most of the
previous comparison mismatch. Under this matched chunk-execution regime, HFRVLA
is ahead of SmolVLA by 4/100 episodes overall, with gains on both spatial
(+2/50) and object (+2/50). The stronger default SmolVLA result from
2026-05-26 remains useful as an upper reference for every-step replanning
(`n_action_steps=1`), but it is not the matched baseline for an 8-step action
chunk experiment.

Recommended architecture experiments after this baseline:

1. **Temporal wrist visual pooling:** keep `seq_len=2` or test `seq_len=4`, but
   explicitly feed previous/current wrist DINO patches into the FWR-v1
   module. The current implementation uses previous action context, not a true
   previous-wrist visual context.
2. **Alpha-calibrated residual objective:** keep the no-gate architecture, but
   train/evaluate around the deployment scale that works (`alpha=0.5`) instead
   of treating `alpha=1.0` as the default target.
3. **Latent-context ablation:** run the same 30k recipe with
   `FAST_RESIDUAL_USE_LATENT_CONTEXT=false` to check whether `z_goal/z_phase`
   are helping correction or adding noise.
4. **Action-context ablation:** remove `prev_a_base` / `prev_delta` from the
   fuser to measure whether the current previous-action conditioning is
   responsible for the 30k gain.
5. **Do not revive gate/GRU/contact as the next step:** the data now points to
   calibration and temporal wrist evidence as the simpler unresolved variables.

### Eval registry and chunk-size sweeps

Long-term experiment comparison now lives in a repo-tracked registry instead of
ad-hoc numbers copied from `outputs/`:

```bash
python3 scripts/build_eval_results_master.py
python3 scripts/build_eval_results_master.py --check
```

Registry files:

- `experiments/eval_registry/sources.csv`: manifest that records each source
  CSV, sweep id, policy, metadata profile, tags, and notes.
- `experiments/eval_registry/eval_results_master.csv`: regenerated compact
  long/tidy master table. Do not edit this file by hand.

The first master table combines:

- `outputs/action_steps_eval_sweep/results.csv` (`30` rows): planning chunk is
  fixed at the checkpoint value (`planning_chunk_size=50`) and only
  execution/replan interval changes.
- `outputs/chunk_size_eval_sweep/results.csv` (`36` rows): planning chunk,
  execution chunk, and replan interval are all set to the same value `K`.

Current combined results, spatial + object, seed 42, 10 tasks x 5 episodes per
suite:

| Sweep | Policy | Planning | Execution / replan | Combined |
|---|---|---:|---:|---:|
| action-step | HFRVLA 30k `alpha=0.5` | 50 | 2 | 85/100 = 85.0% |
| action-step | SmolVLA | 50 | 2 | 82/100 = 82.0% |
| action-step | HFRVLA 30k `alpha=0.5` | 50 | 8 | 81/100 = 81.0% |
| action-step | SmolVLA | 50 | 8 | 77/100 = 77.0% |
| matched chunk | HFRVLA 30k `alpha=0.5` | 8 | 8 | 83/100 = 83.0% |
| matched chunk | SmolVLA | 4 | 4 | 79/100 = 79.0% |
| matched chunk | HFRVLA 30k `alpha=0.5` | 50 | 50 | 46/100 = 46.0% |
| matched chunk | SmolVLA | 50 | 50 | 43/100 = 43.0% |

Interpretation for paper writing:

1. Always distinguish `planning_chunk_size` from
   `execution_chunk_size` / `replan_interval_steps`.
2. The action-step sweep asks how often a 50-step planned chunk should be
   interrupted by replanning.
3. The chunk-size sweep asks what happens when the policy plans and executes
   shorter or longer chunks end-to-end.
4. Long chunks (`K=50`) are a failure mode for both policies under the matched
   planning/execution protocol.
5. Any future training-parameter experiment should add rows to
   `experiments/eval_registry/sources.csv`, then regenerate the master CSV.

### Async-timestep planner-delay eval

The planner-delay eval simulates slow VLA chunk generation latency with
discrete async-timestep semantics. At request step `t`, the slow planner
observes `o_t` and starts generating `A_t`. At ready step `t+d`, the chunk
replaces the active execution queue immediately. Because `d` control steps have
elapsed, execution starts from `A_t[d]`, not `A_t[0]`. The wrist residual is not
delayed and continues using current wrist feedback at every control step.

Run a smoke check with:

```bash
/home/hucenrotia/Robotic_infra/lerobot/.venv/bin/python scripts/run_planner_delay_eval_sweep.py \
    --planner-delay-steps 0,1,4 \
    --policies hfrvla,hfrvla_disable_fast \
    --planning-chunk-size 50 \
    --n-action-steps 16 \
    --async-request-interval-steps 8 \
    --suites libero_spatial \
    --task-ids '[0]' \
    --n-episodes 1 \
    --eval-batch-size 1 \
    --device cuda \
    --csv outputs/async_timestep_planner_delay_eval_sweep/smoke_results.csv \
    --eval-root outputs/async_timestep_planner_delay_eval_sweep/smoke_evals
```

Run the main spatial sweep with:

```bash
/home/hucenrotia/Robotic_infra/lerobot/.venv/bin/python scripts/run_planner_delay_eval_sweep.py \
    --planner-delay-steps 0,1,2,3,4 \
    --policies hfrvla,hfrvla_disable_fast \
    --planning-chunk-size 50 \
    --n-action-steps 16 \
    --async-request-interval-steps 8 \
    --suites libero_spatial \
    --n-episodes 10 \
    --hfrvla-alpha 0.5 \
    --hfrvla-delta-max 0.2 \
    --eval-batch-size 3 \
    --device cuda
```

Outputs are written under `outputs/async_timestep_planner_delay_eval_sweep/`.
The CSV records `planner_delay_mode`, `planner_delay_steps`,
`async_request_interval_steps`, `async_request_count`, `async_activation_count`,
`async_chunk_start_index_mean`, `async_dropped_old_queue_steps_mean`,
`slow_replan_count`, `slow_chunk_latency_ms_mean`, `fast_latency_ms_mean`,
`fast_applied_ratio`, `delta_norm_mean`, `delta_clip_fraction_mean`, and
`k_mean`. The implementation validates
`async_request_interval_steps + planner_delay_steps <= n_action_steps`; the
main setting uses `8 + 4 <= 16`.

If `scripts/package_hfrvla_checkpoint.py` is run in a sandbox where CUDA is not
visible, the saved packaged config can contain `"device": "cpu"`. For formal GPU
eval, pass `--policy.device=cuda` to `lerobot-eval` or repackage in an
environment where CUDA is visible.

### Legacy gated residual objective knobs

The fast module is trained against the same action form used at deployment:
`a_final = a_base + gate * clip(delta_a)`. Do not train the residual head to
chase large raw `a_expert - a_base` values that inference will later clip away.
This update is the main fix after the failed 60k run: the old objective made
offline residual MSE improve while closed-loop eval got worse, because the loss
did not match the clipped/gated action actually sent to LIBERO.

| Flag | Default | Description |
|---|---|---|
| `--policy.loss_delta_target_clip` | `true` | Clips the supervised delta target to the deployable residual range. |
| `--policy.gate_improvement_margin` | `0.5` | Requires the clipped residual to clear this summed-squared-error margin before the Stage A gate target opens. |
| `--policy.loss_lambda_final` | `1.0` | MSE on the deployed merged action `a_final`. |
| `--policy.loss_lambda_preserve` | `0.5` | Penalizes corrections whose merged action is worse than the frozen base action. |
| `--policy.loss_lambda_gate_prior` | `0.10` | Stage A rate term on mean gate. |
| `--policy.loss_lambda_preserve_zero` | `1.0` | Stage A zero-target penalty on frames with `err_before < err_preserve_thresh`. |
| `--policy.err_preserve_thresh` | `0.5` | Stage A preserve threshold in summed-squared action error units. |

These knobs are retained for the legacy `RESIDUAL_MERGE_MODE=gated` path. They
are not used by FWR modes, whose loss is the deployment-aligned objective in
§5: raw residual SmoothL1 plus final-action SmoothL1 and optional residual/clip
penalties.

Stage B replaces the Stage A loss with the five-term rate-distortion objective
from `docs/hfrvla_objective_debate_20260521.md`: `correct`, `preserve_zero`,
`rate`, focal `gate`, and temporal `smooth`. It uses the static fast-cache
labels rather than training-time thresholds.

| Flag | Default | Description |
|---|---|---|
| `--policy.use_stage_b_objective` | `false` | Enables the Stage B loss branch; requires fast-cache schema v2/v3 labels. |
| `--policy.loss_lambda_correct` | `1.0` | SmoothL1 residual correction loss on `y_correct=1` frames only. |
| `--policy.loss_lambda_rate` | `0.5` | Mean gate plus batch budget hinge. |
| `--policy.loss_lambda_smooth` | `0.2` | Smoothness penalty on `gate * clip(delta_a)` across the GRU window. |
| `--policy.focal_gamma` | `2.0` | Focal BCE gamma for the static gate label. |
| `--policy.focal_pos_weight` | `4.0` | Positive-class weight for correction events. |
| `--policy.gate_task_budget` | `0.25` | Batch-level mean-gate budget hinge threshold. |

The training presentation now splits this into three diagrams so the data flow
is easier to audit:

- `Training I/O`: separates fast-module inputs from expert-only supervision
  targets.
- `Fast module outputs`: shows the current FWR `delta_a` output and the
  `a_final = a_base + alpha * clip(delta_a)` merge.
- `Learning objective`: distinguishes the current deployment-aligned FWR loss
  from the legacy gated losses.

### Output structure

```text
checkpoints/hfrvla_run01/
|-- checkpoints/
|   |-- 005000/
|   |   |-- pretrained_model/
|   |   `-- training_state/
|   |-- 010000/
|   `-- last/
|-- train_config.json
`-- wandb/
```

## 6. Evaluate

Training uses cached `a_base/z_goal/z_phase` and sets
`--policy.load_vlm_weights=false`, so the raw LeRobot training checkpoint is
not the final eval artifact. First package the trained fast module together
with the frozen LIBERO SmolVLA slow planner:

```bash
~/Robotic_infra/lerobot/.venv/bin/python scripts/package_hfrvla_checkpoint.py \
    --fast-ckpt checkpoints/hfrvla_run01/checkpoints/last/pretrained_model \
    --out-dir checkpoints/hfrvla_run01_packaged \
    --dinov3-repo checkpoints/dinov3_src \
    --dinov3-weights checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --dataset-repo-id HFRVLA_libero_v1 \
    --dataset-root checkpoints/HFRVLA_libero_v1_merged_reindexed
```

Single-task smoke:

```bash
HFRVLA_TMP_ROOT=$HOME/tmp/hfrvla
mkdir -p "$HFRVLA_TMP_ROOT"/{numba,tmp,matplotlib}

NUMBA_CACHE_DIR=$HFRVLA_TMP_ROOT/numba \
TMPDIR=$HFRVLA_TMP_ROOT/tmp \
MPLCONFIGDIR=$HFRVLA_TMP_ROOT/matplotlib \
~/Robotic_infra/lerobot/.venv/bin/lerobot-eval \
    --policy.path=checkpoints/hfrvla_run01_packaged \
    --policy.device=cuda \
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
HFRVLA_TMP_ROOT=$HOME/tmp/hfrvla
mkdir -p "$HFRVLA_TMP_ROOT"/{numba,tmp,matplotlib}

for SUITE in libero_spatial libero_object libero_goal libero_10 libero_90; do
  NUMBA_CACHE_DIR=$HFRVLA_TMP_ROOT/numba \
  TMPDIR=$HFRVLA_TMP_ROOT/tmp \
  MPLCONFIGDIR=$HFRVLA_TMP_ROOT/matplotlib \
  ~/Robotic_infra/lerobot/.venv/bin/lerobot-eval \
      --policy.path=checkpoints/hfrvla_run01_packaged \
      --policy.device=cuda \
      --env.type=libero \
      --env.task=$SUITE \
      --eval.n_episodes=20 \
      --eval.batch_size=4 \
      --output_dir=outputs/eval_run01_$SUITE \
      --seed=42
done
```

`--env.num_envs` is not a valid LiberoEnv field; use `--eval.batch_size` to control parallelism.

## 7. Health checks during training

| Stage | Boundary | Expected behavior |
|---|---|---|
| 0 warmup | `step in [0, warmup)` | Gate/contact heads frozen, lambdas are 0, `loss == delta`. |
| 1 joint | `step in [warmup, warmup+joint)` | All losses contribute; expect a small spike at the boundary, then descent. |
| 2 refine | `step >= warmup+joint` | `[curriculum] step <N>: entered Stage 2 - LR x 0.1` is printed exactly once. |

If the boundary log is missing, check `scripts/train_via_lerobot.py`; the wrapper may have stopped intercepting `update_policy`.

### Interpreting run06 timing and loss

For `HFRVLA_libero_v1_merged_reindexed`, one epoch at `BATCH_SIZE=128` is
about `ceil(273465 / 128) = 2137` optimizer steps. A `STEPS=10000` run is
therefore about 4.7 passes over the dataset, which is longer than most quick
correction-head checks need.

If the log shows `updt_s ~= 0.03s` but `data_s ~= 7-15s`, the model is not the
bottleneck. The offline fast module has about 1M trainable parameters; the
slow path is LeRobot/HuggingFace Dataset reading `observation.extra.dino_patches`.
With `seq_len=8`, each sample reads 8 frames of `(196, 384)` float32 DINO
patches, or roughly 2.4 MB before Arrow/Python/Tensor overhead.

The apparent loss jump at `step ~= WARMUP_STEPS` is expected accounting unless
the gradient norm also explodes. Stage 0 logs only `L_delta` because
`loss_lambda_gate=0`, `loss_lambda_contact=0`, and the conservative objective
lambdas are zeroed. At Stage 1 entry, the total loss becomes:

```text
loss = L_delta + L_gate + L_final + 0.5 * L_preserve
       + 0.10 * L_gate_prior + L_preserve_zero + 0.1 * L_contact
```

Freshly unfrozen BCE heads usually contribute about `0.69 + 0.07`, so a
total-loss jump at `step 1000` matches the configured curriculum. After the
conservative objective update, also watch `final`, `preserve`, and
`gate_prior`: a high `preserve` value means the fast module is still learning
corrections that would harm the frozen base policy. Track component losses
before treating the total-loss discontinuity as model divergence.

Short correction-head preset:

```bash
RUN_NAME=hfrvla_short_seq4 \
WANDB_ENABLE=false \
HFRVLA_TMP_ROOT=$HOME/tmp/hfrvla \
HFRVLA_DATASET_BACKEND=fastcache \
HFRVLA_FASTCACHE_ROOT=checkpoints/HFRVLA_libero_v1_fastcache_v2 \
BATCH_SIZE=128 \
NUM_WORKERS=8 \
SEQ_LEN=4 \
STEPS=3000 \
WARMUP_STEPS=300 \
JOINT_STEPS=2400 \
REFINE_STEPS=300 \
LOG_FREQ=20 \
SAVE_FREQ=1000 \
scripts/run_hfrvla_training_foreground.sh
```

## 8. GPU utilization tips

The fast module is small and dataset access dominates time. If GPU utilization stays low after the first few hundred steps:

1. Prefer `HFRVLA_DATASET_BACKEND=fastcache` for real runs. It keeps the canonical LeRobotDataset intact but trains from compact memory-mapped arrays.
2. Use `BATCH_SIZE=128` or `256` first, then benchmark `512` if host RAM and disk bandwidth still have headroom. Larger batches mostly increase feature bandwidth; they may not visibly increase VRAM because the trainable model is only ~1M parameters.
3. Increase `NUM_WORKERS` to 12-16 only if RAM/swap pressure is low. If workers sit near 100% CPU and swap grows, reduce batch size before adding workers.
4. Confirm the train log prints either `[hfrvla-train] fast-cache dataset backend enabled` or `[hfrvla-train] offline dataset column pruning enabled`. Without either, raw images may be back in the LeRobot window/collate path.
5. If training is I/O-bound, reduce `SEQ_LEN` before increasing model-side knobs. Local single-item reads measured about `0.557s` at `SEQ_LEN=8`, `0.283s` at `SEQ_LEN=4`, `0.141s` at `SEQ_LEN=2`, and `0.075s` at `SEQ_LEN=1` on the LeRobot parquet backend.

Batch size can improve throughput, but only if the larger batch increases
samples/sec:

```text
samples/sec ~= BATCH_SIZE / (data_s + updt_s)
```

Compare short runs with identical settings except `BATCH_SIZE`. If `data_s`
roughly doubles when the batch doubles, the input pipeline is still the
bottleneck and bigger batches mainly change the training regime. If `updt_s`
dominates and GPU utilization is low, larger batches can amortize overhead and
improve throughput.

Do not compare wall-clock time at fixed `STEPS` as if it were the same
training budget. With fixed `STEPS=60000`, `BATCH_SIZE=512` sees twice as many
samples as `BATCH_SIZE=256`. For an equal sample/epoch budget, scale steps and
curriculum boundaries inversely with batch size. On the current 273465-frame
dataset:

```text
1 epoch @ batch 256 ~= 1069 optimizer steps
1 epoch @ batch 512 ~= 535 optimizer steps
```

So when doubling batch size from 256 to 512 and keeping the same sample
exposure, halve `STEPS`, `WARMUP_STEPS`, `JOINT_STEPS`, `REFINE_STEPS`, and
`SAVE_FREQ` in step units.

Past those, the remaining storage lever is re-recording with
`--dino-dtype float16`; the fast-cache builder already stores the derived
training arrays for `z_*` and `dino_patches` as `float16`.
