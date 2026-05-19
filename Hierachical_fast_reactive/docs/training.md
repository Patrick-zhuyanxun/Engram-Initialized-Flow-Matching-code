# HFRVLA training and eval (lerobot-native)

Use `lerobot-train` for training and `lerobot-eval` for evaluation.
Recording uses a wrapper script that writes a standard LeRobotDataset v3.

Active scripts:
- `scripts/train_smolvla_libero_baseline.sh` - optional fallback to fine-tune `lerobot/smolvla_base` on LIBERO if the published checkpoint is unsuitable
- `scripts/record_hfrvla_libero.py` - one-shot or sharded recording (`--ep-from/--ep-to`)
- `scripts/merge_hfrvla_shards_fast.py` - offline stitcher for parallel shards (file/metadata only, no re-encode)
- `scripts/build_hfrvla_fastcache.py` - optional derived mmap cache for faster offline HFRVLA training
- `scripts/train_hfrvla_libero_merged.sh` - pinned training entrypoint for the verified merged HFRVLA dataset
- `scripts/train_via_lerobot.py` - wrapper around `lerobot-train` that adds curriculum control
- `scripts/package_hfrvla_checkpoint.py` - bundle SmolVLA + fast weights into a `lerobot-eval` checkpoint dir (`--disable-fast` for alignment tests)
- `scripts/check_hfrvla_training_contract.py` - offline train-time dataset/window/target contract check
- `scripts/test_alignment.py` - I/O alignment check against the frozen SmolVLA slow planner before training
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
| LIBERO SmolVLA slow planner | `HuggingFaceVLA/smolvla_libero` or a local snapshot of it |
| Wandb login | `~/Robotic_infra/lerobot/.venv/bin/wandb login` |

Working directory:

```bash
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

Build once per source dataset and sequence length:

```bash
SEQ_LEN=4
~/Robotic_infra/lerobot/.venv/bin/python scripts/build_hfrvla_fastcache.py \
    --source-root checkpoints/HFRVLA_libero_v1_merged_reindexed \
    --cache-root checkpoints/HFRVLA_libero_v1_fastcache_seq${SEQ_LEN} \
    --seq-len "$SEQ_LEN"
```

Train against the cache:

```bash
RUN_NAME=hfrvla_run_fastcache_seq4 \
WANDB_ENABLE=true \
HFRVLA_TMP_ROOT=$HOME/tmp/hfrvla \
HFRVLA_DATASET_BACKEND=fastcache \
HFRVLA_FASTCACHE_ROOT=checkpoints/HFRVLA_libero_v1_fastcache_seq4 \
SEQ_LEN=4 \
BATCH_SIZE=256 \
NUM_WORKERS=8 \
scripts/run_hfrvla_training_foreground.sh
```

Expected log line:

```text
[hfrvla-train] fast-cache dataset backend enabled root=...
```

If `HFRVLA_DATASET_BACKEND` is unset, the launcher keeps the LeRobot-native
backend and should still print:

```text
[hfrvla-train] offline dataset column pruning enabled
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
| `--policy.seq_len` | 8 | GRU window length; drives `observation_delta_indices` and `action_delta_indices`. |

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
`loss_lambda_gate=0` and `loss_lambda_contact=0`. At Stage 1 entry, the total
loss becomes:

```text
loss = L_delta + L_gate + 0.1 * L_contact
```

Freshly unfrozen BCE heads usually contribute about `0.69 + 0.07`, so a
`0.30 -> 1.0` total-loss jump at `step 1000` matches the configured curriculum.
Track component losses before treating the total-loss discontinuity as model
divergence.

Short correction-head preset:

```bash
RUN_NAME=hfrvla_short_seq4 \
WANDB_ENABLE=false \
HFRVLA_TMP_ROOT=$HOME/tmp/hfrvla \
HFRVLA_DATASET_BACKEND=fastcache \
HFRVLA_FASTCACHE_ROOT=checkpoints/HFRVLA_libero_v1_fastcache_seq4 \
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
2. Use `BATCH_SIZE=128` or `256` first. Larger batches mostly increase host-side feature bandwidth; they may not visibly increase VRAM because the trainable model is only ~1M parameters.
3. Increase `NUM_WORKERS` to 12-16 only if RAM/swap pressure is low. If workers sit near 100% CPU and swap grows, reduce batch size before adding workers.
4. Confirm the train log prints either `[hfrvla-train] fast-cache dataset backend enabled` or `[hfrvla-train] offline dataset column pruning enabled`. Without either, raw images may be back in the LeRobot window/collate path.
5. If training is I/O-bound, reduce `SEQ_LEN` before increasing model-side knobs. Local single-item reads measured about `0.557s` at `SEQ_LEN=8`, `0.283s` at `SEQ_LEN=4`, `0.141s` at `SEQ_LEN=2`, and `0.075s` at `SEQ_LEN=1` on the LeRobot parquet backend.

Past those, the remaining storage lever is re-recording with
`--dino-dtype float16`; the fast-cache builder already stores the derived
training arrays for `z_*` and `dino_patches` as `float16`.
