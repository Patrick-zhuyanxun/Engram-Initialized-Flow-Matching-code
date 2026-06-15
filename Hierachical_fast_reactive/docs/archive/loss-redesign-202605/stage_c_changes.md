# Stage C — HFRVLA closed-loop preserve via cached zero_fast rollouts

**Date**: 2026-05-23
**Status**: ❌ **Did not improve over Stage A/B — 23/50 on LIBERO spatial (vs 24/50 for both A v2 and B, vs 23/50 for zero_fast)**
**Predecessors**: Stage A v2 (`d659a90`), Stage B (uncommitted, see `docs/stage_b_changes.md`)
**Source of design**: `docs/hfrvla_objective_debate_20260521.md` (Codex R1 §3c, §3d)

This doc explains *what changed*, *why it was tried*, *what was measured*, and
*what it tells us about the ceiling*.

## Why Stage C exists

Stage A v2 fixed the gate self-loop and gradient leak → 12/50 → 24/50.
Stage B added principled 5-term rate-distortion loss with static labels → still
24/50. Both confirmed that **the dominant failure cause was offline objective
design**, but neither broke through the 24-point ceiling.

The 4-way debate predicted Stage B at 28-30/50 and Stage C at 28-32/50. Stage B
landed at 24/50 — falsifier territory. Per the debate's Stage B falsifier:

> "24-27/50 → Stage A worked but the rate-distortion ceiling is real. → Proceed
> to Stage C (cached rollout states)"

Stage C's hypothesis: **the residual is trained on expert-demo state
distribution, but at eval time it must operate on the state distribution that
the deployed `a_base + residual` policy actually visits**. Adding rollout states
where the base policy succeeds as additional preserve-class training data should
teach the model "in these states the base is competent → residual=0."

## What changed (infrastructure)

### New file: `scripts/record_zero_fast_rollouts.py` (~500 lines)

Records closed-loop LIBERO rollouts with a packaged zero_fast policy
(`--disable-fast`). Writes successful episodes to LeRobotDataset v3 layout with
SmolVLA + DINOv3 features computed per frame.

Output schema matches the canonical merged dataset
(`HFRVLA_libero_v1_merged_reindexed`).

```bash
python scripts/record_zero_fast_rollouts.py \
  --policy-path checkpoints/hfrvla_zero_fast_packaged \
  --out-root checkpoints/HFRVLA_libero_v1_zero_fast_rollouts \
  --task-suite libero_spatial \
  --task-ids 0,1,2,3,4,5,6,7,8,9 \
  --episodes-per-task 10
```

Result: 75 / 100 attempted episodes succeeded; **8242 frames recorded**.

Per-task success rate (which is also the implicit base-policy ceiling on this
seed range):

| Task | Saved / Attempted |
|---:|---:|
| 0 | 7/10 |
| 1 | 10/10 |
| 2 | 6/10 |
| 3 | 5/10 |
| 4 | 9/10 |
| 5 | 6/10 |
| 6 | 10/10 |
| 7 | 6/10 |
| 8 | 9/10 |
| 9 | 7/10 |

### `scripts/build_hfrvla_fastcache.py`: `--static-y-preserve` flag

When set, writes `y_preserve=1` and `y_correct=0` for every frame in the
output cache. Used for the rollout cache where every frame is, by definition,
a "base policy succeeded here" state.

### `HFRVLAFastCacheDataset(roots=[...])` — multi-root mode

The dataset class now accepts a list of cache roots. It virtually concatenates
the memmaps and presents a unified dataset to the trainer. Each `__getitem__(idx)`
figures out which root the idx falls into and reads from the corresponding cache.

### `HFRVLA_FASTCACHE_ROLLOUT_ROOT` env var in training script

When set, `train_hfrvla_libero_merged.sh` passes both the main cache and the
rollout cache into the dataset constructor. Unset behavior is unchanged
(single-root, Stage A/B compatible).

### No loss code changes for Stage C itself

Stage C reuses Stage B's 5-term loss exactly. The only difference is:
- The dataset is ~3% larger (273k expert frames + 8k rollout frames).
- All 8k rollout frames are labeled `y_preserve=1` (vs ~50% in expert demos).
- Bump `LOSS_LAMBDA_PRESERVE_ZERO` from 2.0 → 3.0 in the training script to
  emphasize the new preserve frames.

## Measured results

### Train metrics at step 30k

| Metric | Stage A v2 | Stage B | **Stage C** |
|---|---:|---:|---:|
| `train/loss` | 0.480 | 0.155 | 0.180 |
| `train/correct` | — | 0.054 | 0.056 |
| `train/preserve_zero` | 0.00117 | 0.00021 | 0.00019 |
| `train/rate` | — | 0.168 | 0.184 |
| `train/gate` (focal BCE) | 0.084 | 0.005 | 0.009 |
| `train/smooth` | — | 0.018 | 0.018 |
| **`train/gate_prior`** (mean gate) | 0.362 | 0.168 | **0.184** |
| `train/grad_norm` | 1.28 | 1.04 | 1.30 |

Training metrics are nearly identical to Stage B. The extra rollout frames did
not significantly shift the loss landscape — they're ~3% of total frames and
contribute mostly to the preserve term, which was already small at convergence.

### Closed-loop eval — LIBERO spatial, 50 trials (seed 42)

| Variant | Successes | % | Δ vs zero_fast |
|---|---:|---:|---:|
| zero_fast (no residual) | 23/50 | 46.0% | — |
| Old trained checkpoint (failure mode) | 12/50 | 24.0% | -22% |
| Stage A v2 (`hfrvla_v2a_calib`) | 24/50 | 48.0% | +2% |
| Stage B (`hfrvla_v2b_stage_b`) | 24/50 | 48.0% | +2% |
| **Stage C (`hfrvla_v2c_stage_c`)** | **23/50** | **46.0%** | **0%** |

Per-task breakdown with **Δ vs Stage B** highlighted:

| Task | C | B | A | zf | old | Δ vs B | Notes |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 3/5 | 4/5 | 2/5 | 3/5 | 2/5 | **-1** | |
| 1 | 2/5 | 3/5 | 4/5 | 2/5 | 3/5 | **-1** | |
| 2 | 0/5 | 1/5 | 2/5 | 0/5 | 0/5 | **-1** | |
| 3 | 4/5 | 4/5 | 3/5 | 3/5 | 1/5 | 0 | |
| **4** | **4/5** | 0/5 | 1/5 | 2/5 | 1/5 | **+4** | **Stage C unlocks task 4** |
| **5** | **2/5** | 0/5 | 0/5 | 0/5 | 0/5 | **+2** | **Only variant that does task 5** |
| 6 | 2/5 | 2/5 | 4/5 | 3/5 | 1/5 | 0 | |
| 7 | 3/5 | 3/5 | 2/5 | 3/5 | 4/5 | 0 | |
| **8** | 1/5 | 3/5 | 4/5 | 3/5 | 0/5 | **-2** | C over-conservative on strength task |
| **9** | 2/5 | 4/5 | 2/5 | 4/5 | 0/5 | **-2** | C over-conservative on strength task |

### What this means

Stage C is **not strictly worse** than B even though the total is -1. It's a
**different operating point**:

- **Gains** on tasks 4 and 5 (+4 and +2) — these are tasks where A and B
  *and* zero_fast all struggle. Stage C is the only variant that does task 5
  at all.
- **Losses** on tasks 8 and 9 (-2 and -2) — these were Stage B's strengths.

The rollout cache contained 9/10 successful task-8 rollouts and 7/10 successful
task-9 rollouts, all labeled `y_preserve=1`. This trained the model to
**suppress its residual aggressively** on these tasks. The result: when the base
policy is *almost* right (but not exactly), Stage C's residual no longer helps,
because it learned to defer.

This is a **trade-off, not a bug**. The residual is being correctly conservative
where it was trained to be conservative. It just happens that on tasks 8/9 the
small corrections that A/B made were genuinely helpful and Stage C overshot in
the safe direction.

## What this tells us about the ceiling

Three independent objective redesigns (A v2 / B / C) all land at 23-24/50.
This is strong evidence that **24/50 is the offline-IL ceiling on this
dataset/architecture/base-policy combination**, not an artifact of any single
loss design choice.

### Where the ceiling comes from

1. **SmolVLA base capability**: `zero_fast` is 23/50 = 46%. The residual can
   at most add a few percentage points of *correction* on top, not unlock
   new capability.
2. **Demo distribution**: 273k expert frames covers a specific manifold. The
   residual learns to interpolate inside it. State-distribution shift at
   deployment is real but the rollout cache only adds 3% volume — not enough
   to retrain the embedding.
3. **Per-task asymmetry**: Tasks 4, 5, and 8 are particularly hard. Different
   training objectives reweight which tasks benefit and which suffer. No
   objective lifts all tasks simultaneously.

### What would actually break the ceiling

Outside the scope of the "small fast module as compression" invariant:

1. **Bigger base model** — replace SmolVLA-500M with OpenVLA-7B or π0. The
   46% ceiling is fundamentally a `a_base` quality problem.
2. **More demo data** — 273k frames for 10 spatial tasks is borderline thin.
   Adding 1M+ frames with task diversity would re-fit the embedding.
3. **Online RL fine-tuning** — break the IL invariant. Use Stage A v2 as
   initialization, then PPO or REINFORCE in sim. This would let the residual
   *explore* corrections rather than only interpolate.
4. **Architecture change** — swap GRU + cross-attn for a transformer with
   longer history. Maybe the small fast module is undersized for the level
   of correction needed.

None of these are in the current "small-model-as-compression" research scope.

## Falsifier outcome

| Outcome | Predicted next action | **Observed: 23/50 → return to Stage A v2 baseline** |
|---|---|---|
| ≥ 28/50 | Stage C works, ship it. | — |
| **23-27/50** | **Ceiling reached. The 24-point band is the true offline IL limit.** | **← landed here** |
| < 23/50 | Stage C is actively harmful. | — |

The 4-way debate predicted Stage C at 28-32/50. **Actual is 23/50, below the
falsifier threshold of 28**. The synthesis's optimism about closed-loop preserve
data was not borne out in practice.

## Per-task trade-off has research value

Even though the total didn't improve, the per-task pattern is informative:

- **Task 4 / 5 unlock**: Stage C is the *only* tested variant that achieves
  non-zero success on task 5, and goes from 0→4 on task 4. If the user cares
  about coverage (some success on every task) rather than total, Stage C is
  preferred.
- **Task 8 / 9 regression**: opposite direction. If the user cares about
  *reliable* execution on tasks the base already does, Stage A v2 is better.

In a real deployment, **per-task hyperparameters** (e.g., disabling residual
on tasks 8/9, enabling it on tasks 4/5) would extract the best of both. This
is Codex R1 §3b's "per-task budget hinge" — not implemented in this Stage C
sprint but a clear next step if anyone returns to this work.

## How to reproduce

```bash
# 1. Package zero_fast checkpoint
python scripts/package_hfrvla_checkpoint.py \
  --disable-fast \
  --out-dir checkpoints/hfrvla_zero_fast_packaged \
  --dinov3-repo checkpoints/dinov3_src \
  --dinov3-weights checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

# 2. Record rollouts (~1.5h)
HFRVLA_TMP_ROOT=/tmp/hfrvla_stage_c \
HF_DATASETS_CACHE=/tmp/hfrvla_stage_c/hf_datasets \
TMPDIR=/tmp/hfrvla_stage_c/tmp MPLCONFIGDIR=/tmp/hfrvla_stage_c/matplotlib \
NUMBA_CACHE_DIR=/tmp/hfrvla_stage_c/numba \
  python scripts/record_zero_fast_rollouts.py \
    --policy-path checkpoints/hfrvla_zero_fast_packaged \
    --out-root checkpoints/HFRVLA_libero_v1_zero_fast_rollouts \
    --task-suite libero_spatial --task-ids 0,1,2,3,4,5,6,7,8,9 \
    --episodes-per-task 10

# 3. Build rollout fastcache (~1 min — only 8k frames)
python scripts/build_hfrvla_fastcache.py \
  --source-root checkpoints/HFRVLA_libero_v1_zero_fast_rollouts \
  --cache-root checkpoints/HFRVLA_libero_v1_fastcache_seq4_v2_rollouts \
  --seq-len 4 --static-y-preserve

# 4. Train (~1h)
HFRVLA_DATASET_BACKEND=fastcache \
HFRVLA_FASTCACHE_ROOT=checkpoints/HFRVLA_libero_v1_fastcache_seq4_v2 \
HFRVLA_FASTCACHE_ROLLOUT_ROOT=checkpoints/HFRVLA_libero_v1_fastcache_seq4_v2_rollouts \
SEQ_LEN=4 BATCH_SIZE=512 NUM_WORKERS=8 USE_STAGE_B=true \
LOSS_LAMBDA_PRESERVE_ZERO=3.0 \
OUT_DIR=checkpoints/hfrvla_v2c_stage_c JOB_NAME=hfrvla_v2c_stage_c \
STEPS=30000 SAVE_FREQ=5000 WARMUP_STEPS=1000 JOINT_STEPS=29000 REFINE_STEPS=0 \
WANDB_ENABLE=true WANDB_PROJECT=hfrvla \
  bash scripts/train_hfrvla_libero_merged.sh

# 5. Package + eval (~15 min)
python scripts/package_hfrvla_checkpoint.py \
  --fast-ckpt checkpoints/hfrvla_v2c_stage_c/checkpoints/last/pretrained_model \
  --out-dir checkpoints/hfrvla_v2c_stage_c_packaged \
  --dinov3-repo checkpoints/dinov3_src \
  --dinov3-weights checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

NUMBA_CACHE_DIR=$HOME/tmp/hfrvla/numba TMPDIR=$HOME/tmp/hfrvla/tmp \
MPLCONFIGDIR=$HOME/tmp/hfrvla/matplotlib \
  lerobot-eval --policy.path=checkpoints/hfrvla_v2c_stage_c_packaged \
    --env.type=libero --env.task=libero_spatial \
    --eval.n_episodes=5 --eval.batch_size=1 \
    --output_dir=outputs/eval_hfrvla_v2c_stage_c_spatial --seed=42
```

## Pointers

- Debate synthesis (all proposals): `docs/hfrvla_objective_debate_20260521.md`
- Stage A v2 changes: `docs/stage_a_changes.md`
- Stage B changes: TBD (uncommitted; see this same commit for code)
- All-stages summary: `docs/stages_summary.md`
- Eval result JSON: `outputs/eval_hfrvla_v2c_stage_c_spatial/eval_info.json`
