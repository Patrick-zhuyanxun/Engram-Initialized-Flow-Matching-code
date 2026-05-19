# HFRVLA fast-cache training backend (design spec)

**Status:** draft for written review  
**Owner:** Patrick (Yan-Xun Chu)  
**Date:** 2026-05-19  
**Working dir:** `~/Patrick/VLA_research/Hierachical_fast_reactive`

---

## 1. Goal

Speed up HFRVLA offline training by replacing the hot training read path with a
compact derived feature cache while keeping the verified LeRobotDataset v3 as
the canonical source of truth.

The current LeRobot-native path is correct but too slow for this model. The
trainable HFRVLA fast module has only about 1.35M parameters, so CUDA memory and
compute are not the limiting factors. The bottleneck is reading and materializing
large nested `observation.extra.dino_patches` arrays from Parquet for randomly
shuffled temporal windows.

After this change:

- `checkpoints/HFRVLA_libero_v1_merged_reindexed` remains the canonical dataset.
- A derived fast-cache is built from the canonical dataset and used for training.
- The HFRVLA policy still receives the same batch keys it receives today.
- Training can opt into the fast path with a clear flag, without deleting the
  existing LeRobot path.

---

## 2. Diagnosis

The verified dataset has 273,465 frames and occupies about 112 GB. Each
`dino_patches` value is `196 x 384 x float32`, roughly 301 KB per frame. With
`SEQ_LEN=4` and `BATCH_SIZE=256`, each training step needs about 294 MiB of DINO
payload before Python, Arrow, tensor stacking, and pinned-memory overhead.

This is unlike ordinary VLA fine-tuning data, where the dataset usually stores
compressed RGB/video, state/action, language, and metadata. HFRVLA stores dense
per-frame DINO patch features because the frozen wrist encoder is precomputed.
That makes the training loop IO-bound even though the model is small.

Main external references checked:

- Open X-Embodiment stores datasets in RLDS episode format for common downstream
  consumption.
- OpenVLA uses RLDS/OXE mixtures for pretraining and fine-tuning, and recommends
  RLDS for custom datasets because that path is the tested training path.
- DROID provides RLDS data plus a training-ready loader with parallel loading,
  normalization, and augmentation.
- Diffusion Policy uses a Zarr/NumPy-backed replay buffer with chunking and
  sequence sampling rather than repeatedly materializing nested tabular values.
- LeRobot v3 is a good canonical format, and its docs note alternative dataset
  implementations such as Lance for faster loading.

---

## 3. Decisions

| # | Question | Decision |
|---|---|---|
| 1 | Keep LeRobot v3? | Yes. It remains the canonical, reproducible dataset format. |
| 2 | Fast storage format? | Use a local derived array cache. Prefer NumPy memmap/`.npy` files first; keep the layout simple and auditable. |
| 3 | DINO dtype in cache? | Store `dino_patches`, `z_goal`, and `z_phase` as `float16`; cast as needed in the training batch. |
| 4 | Policy API? | Keep the same batch keys as the current LeRobot training path. |
| 5 | Windowing semantics? | Match current LeRobot delta-timestamp behavior: past window ending at current frame, clamped at episode boundaries. |
| 6 | Rollout strategy? | Add fast-cache as opt-in. Keep the old path available for comparison and fallback. |

---

## 4. Non-goals

- Do not replace the canonical LeRobotDataset v3 recording/merge pipeline.
- Do not change SmolVLA, DINOv3, or the HFRVLA model architecture.
- Do not resurrect the legacy `.pt` cache as the primary pipeline.
- Do not introduce distributed training in this change.
- Do not change inference or checkpoint packaging behavior except for docs if
  the training command changes.

---

## 5. Architecture

### 5.1 Canonical dataset

The existing dataset stays as the source:

```text
checkpoints/HFRVLA_libero_v1_merged_reindexed
```

It provides the authoritative metadata, episode boundaries, normalization stats,
and original features.

### 5.2 Derived fast-cache

Add a build script:

```text
scripts/build_hfrvla_fastcache.py
```

Default output:

```text
checkpoints/HFRVLA_libero_v1_fastcache_seq4/
```

Cache layout:

```text
meta.json
episode_starts.npy
episode_ends.npy
state.npy
action.npy
a_base.npy
k_idx_norm.npy
z_goal.npy
z_phase.npy
dino_patches.npy
contact_label.npy
```

Array shapes:

| Array | dtype | shape |
|---|---:|---|
| `state` | `float32` | `[N, 8]` |
| `action` | `float32` | `[N, 7]` |
| `a_base` | `float32` | `[N, 7]` |
| `k_idx_norm` | `float32` | `[N, 1]` |
| `z_goal` | `float16` | `[N, 960]` |
| `z_phase` | `float16` | `[N, 480]` |
| `dino_patches` | `float16` | `[N, 196, 384]` |
| `contact_label` | `float32` | `[N, 1]` |

`meta.json` records:

- source dataset root
- source `meta/info.json` hash
- source total frames and episodes
- fast-cache schema version
- seq_len used for validation defaults
- feature shapes and dtypes
- build timestamp

The cache is rebuildable and can be deleted at any time.

### 5.3 Fast-cache dataset

Add a focused PyTorch dataset module under the policy package:

```text
policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/fast_cache_dataset.py
```

Responsibilities:

- Open arrays with memory mapping.
- Build temporal windows by direct indexing.
- Clamp indices at episode boundaries to match LeRobot padding behavior.
- Return exactly the same keys used by `HFRVLAPolicy.forward`.
- Expose a small `meta` object with the stats and fields needed by
  `lerobot-train`.

Returned batch item keys:

```text
observation.state
action
observation.extra.a_base
observation.extra.k_idx_norm
observation.extra.z_goal
observation.extra.z_phase
observation.extra.dino_patches
observation.extra.contact_label
```

### 5.4 Training integration

Modify `scripts/train_via_lerobot.py` so that HFRVLA offline training can use
the fast-cache backend.

New CLI/environment controls:

```text
HFRVLA_DATASET_BACKEND=lerobot|fastcache
HFRVLA_FASTCACHE_ROOT=/path/to/cache
```

The default remains `lerobot` until the fast-cache smoke and equivalence tests
pass. Once verified, `scripts/run_hfrvla_training_foreground.sh` and
`scripts/train_hfrvla_libero_merged.sh` can default to `fastcache` when
`HFRVLA_FASTCACHE_ROOT` exists.

Implementation strategy:

1. Keep the existing monkey patches for curriculum and offline column pruning.
2. For `HFRVLA_DATASET_BACKEND=fastcache`, patch `lerobot_train.make_dataset` to
   return the fast-cache dataset instead of constructing a LeRobotDataset.
3. Preserve `dataset.meta.stats` by loading stats from the canonical
   `meta/stats.json`, because LeRobot processors still need action/state stats.
4. Keep the same `update_policy` curriculum hook.

This avoids changing LeRobot source files and keeps all HFRVLA-specific behavior
inside this repo.

---

## 6. Data Flow

Build time:

```text
LeRobotDataset v3 parquet columns
  -> selected required HFRVLA columns
  -> dtype conversion for large features
  -> contiguous memmapped arrays
  -> metadata/fingerprint validation
```

Train time:

```text
Random index from DataLoader
  -> fast-cache direct temporal slice
  -> PyTorch collate
  -> LeRobot preprocessor normalization/device placement
  -> HFRVLAPolicy.forward
```

The policy remains unaware of whether the batch came from Parquet or fast-cache.

---

## 7. Error Handling

Fast-cache loading should fail early with clear errors when:

- `meta.json` is missing or has an unsupported schema version.
- expected array files are missing.
- array shapes do not match `meta.json`.
- source frame count does not match cache frame count.
- canonical stats are missing.
- requested `seq_len` is not compatible with the cache validation metadata.

The training script should print the active backend and cache root at startup.

---

## 8. Testing And Benchmarks

Add focused tests:

| Test | Purpose |
|---|---|
| Fast-cache metadata load | Validate `meta.json`, dtypes, shapes, and source stats wiring. |
| Windowing equivalence | Compare fast-cache sample shapes and boundary clamping against the current LeRobot path for selected indices. |
| Dataloader smoke | Confirm `DataLoader(batch_size>1)` collates the exact policy keys. |
| Training smoke | Run a tiny fast-cache training job for a few steps on CPU or CUDA. |
| Build smoke | Build a cache for a limited frame/episode subset. |

Add a benchmark command that reports:

- seconds per item
- seconds per batch
- old LeRobot backend vs fast-cache backend
- effective DINO payload throughput

Success target for the first implementation: reduce dataloading time by at
least 3x versus the current Parquet path for `SEQ_LEN=4`, with no policy API
change.

---

## 9. Documentation Updates

Update:

- `docs/training.md`
- `docs/lerobot_hfrvla_context.md`
- `policy/lerobot_policy_hfrvla/README.md` if command examples change

Document:

- how to build the cache
- how to select backend
- how to verify equivalence
- when to rebuild the cache
- that LeRobot v3 remains canonical

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| Fast-cache windows diverge from LeRobot windows | Add direct sample equivalence tests at episode starts, middles, and ends. |
| Float16 cached features change training numerics | Keep small comparison smoke with float32 source and monitor initial losses; allow `--large-feature-dtype=float32` escape hatch. |
| LeRobot processors expect a richer `meta` object | Implement the minimum needed metadata wrapper and test through `train_via_lerobot.py`. |
| Disk usage grows | Cache is derived and deletable; float16 large features should cut storage substantially versus current float32 DINO Parquet payload. |
| Hidden dependency on raw image keys | Existing offline column pruning already removes images; tests should assert no image key is required in offline training. |

---

## 11. Open Implementation Notes

- Start with NumPy `.npy` memmap arrays because they are easy to inspect and do
  not add a new dependency. If chunk-level compression is needed later, add a
  Zarr backend behind the same dataset interface.
- Reuse `LeRobotDatasetMetadata` for canonical stats where possible, avoiding a
  full LeRobotDataset load when only metadata is needed.
- The cache builder may read selected Parquet columns directly with
  `datasets.Dataset.from_parquet(columns=...)` or PyArrow. Prefer the simplest
  approach that avoids loading image columns.
- Keep cache building separate from training. Training should never silently
  create a 50+ GB cache.

