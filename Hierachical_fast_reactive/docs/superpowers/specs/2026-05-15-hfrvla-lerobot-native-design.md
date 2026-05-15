# HFRVLA → LeRobot-native migration (design spec)

**Status:** draft for review
**Owner:** Patrick (Yan-Xun Chu)
**Date:** 2026-05-15
**Working dir:** `~/Patrick/VLA_research/Hierachical_fast_reactive`

---

## 1. Goal

Replace the current custom `train_hfrvla.py` + `.pt`-cached precompute pipeline with a fully `lerobot-train` / `lerobot-eval` native pipeline, **without sacrificing the precompute speedup**.

After this migration:
- All training is invoked via `lerobot-train --policy.type=hfrvla --dataset.repo_id=<new-dataset>`.
- All evaluation is invoked via `lerobot-eval --policy.path=<train-output> --env.type=libero`.
- Custom training script is deleted (or kept as a frozen baseline in `legacy/`).
- The precompute step is rewritten to produce a **standard LeRobotDataset v3** that bakes in `z_goal / z_phase / a_base / k_idx_norm / dino_patches / contact_label` as extra `observation.*` features.

---

## 2. Decisions (locked)

| # | Question | Answer |
|---|---|---|
| ① | 資料集放哪？ | **Local-only** under `$HF_LEROBOT_HOME/HFRVLA_libero_v1/` (or `--dataset.root` override). Do not push to Hub initially. |
| ② | Curriculum 怎麼處理？ | **Keep all 3 stages.** Embed in `HFRVLAPolicy` via a `set_training_step(step)` hook called from a lightweight lerobot-train callback patch. |
| ③ | Sequence windowing | **Use `delta_timestamps`.** All extras are named `observation.extra.*` so they inherit `observation_delta_indices = list(range(-7, 1))`. |

---

## 3. Non-goals

- Multi-GPU / DDP training (out of scope; lerobot's accelerator will be future work).
- Hub upload (deferred until results are good).
- Replacing the DINOv3 backbone or SmolVLA base.
- Refactoring `HFRVLAPolicy.select_action` (eval path) — already works with packaging script.

---

## 4. Architecture

### 4.1 New LeRobotDataset: `HFRVLA_libero_v1`

Built once via a new recording script, stored locally, never re-built unless precompute logic changes.

**`meta/info.json` features** (full list — order = stable parquet column order):

| Key | dtype | shape | Source |
|---|---|---|---|
| `observation.images.image` | image (MP4) | (256, 256, 3) | LIBERO agentview |
| `observation.images.image2` | image (MP4) | (256, 256, 3) | LIBERO wrist |
| `observation.state` | float32 | (8,) | LIBERO state |
| `action` | float32 | (7,) | LIBERO expert action (= `a_expert`) |
| `observation.extra.z_goal` | float32 | `(text_hidden,)` | SmolVLA text encoder pool; dim read at recording-start from `policy.model.vlm_with_expert.config.text_config.hidden_size` and written into `info.json` |
| `observation.extra.z_phase` | float32 | `(expert_hidden,)` | SmolVLA expert hidden pool; dim read from `policy.model.vlm_with_expert.expert_hidden_size` |
| `observation.extra.a_base` | float32 | (7,) | SmolVLA-predicted base action at step t |
| `observation.extra.k_idx_norm` | float32 | (1,) | Chunk position in `[0,1]` |
| `observation.extra.dino_patches` | float32 | (196, 384) | DINOv3 ViT-S/16 patches on wrist |
| `observation.extra.contact_label` | float32 | (1,) | LIBERO heuristic; 0 by default |
| (standard auto-generated) | — | — | `timestamp`, `frame_index`, `episode_index`, `index`, `task_index`, `task` |

**Storage budget**: ~107 GB (dominated by `dino_patches` at 196×384×4 B × ~261k steps ≈ 80 GB, plus video).
Use `vcodec="libsvtav1"` for video (default; cheap on storage). `streaming_encoding=True` to overlap encoding with recording.

### 4.2 Naming rationale: `observation.extra.*`

The `OBS_PREFIX = "observation."` prefix is what `resolve_delta_timestamps()` looks at to apply `observation_delta_indices`. So:

- `observation.extra.z_goal` → gets windowed to `[B, T=8, 960]` automatically.
- The `NormalizerProcessorStep` only normalizes keys listed in `HFRVLAConfig.input_features`. We will declare ONLY the standard image/state keys there; extras are not normalized (they're already in their natural numerical range from SmolVLA / DINOv3).

If during implementation we discover the prefix logic doesn't extend to dotted paths beyond two segments, fall back to `observation.z_goal` (no `extra.` infix).

### 4.3 Code changes (high level)

```
policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/
├── configuration_hfrvla.py
│   ├── + observation_delta_indices property → list(range(-7, 1))
│   ├── + curriculum_warmup_steps / curriculum_joint_steps / curriculum_refine_steps fields
│   ├── (no change to input_features — keep image/state only)
│   └── (no change to action features)
├── modeling_hfrvla.py
│   ├── forward(batch): consume batch["observation.extra.*"] keys instead of bare names
│   ├── + set_training_step(step) method → mutates lambdas + freezes/unfreezes heads
│   └── get_optim_params() unchanged (still returns fast.* only)
├── data.py                    → move to legacy/data.py
├── processor_hfrvla.py        (no change)
├── fast_reactive.py           (no change)
└── dinov3_backbone.py         (no change)

scripts/
├── record_hfrvla_libero.py    (NEW — replaces precompute_libero.py)
├── train_via_lerobot.py       (NEW — see §4.4)
├── package_hfrvla_checkpoint.py (now optional; lerobot-train output is already
│                                 eval-loadable. Keep around as a one-shot tool
│                                 for repackaging legacy fast_final.pt files.)
└── legacy/
    ├── train_hfrvla.py        (moved, not deleted; reference baseline)
    ├── precompute_libero.py   (moved)
    ├── precompute_libero_plan.py (moved)
    └── precompute_libero_parallel.sh (moved)

docs/
└── training.md                (rewrite to use lerobot-train invocation)
```

### 4.4 Curriculum hook design

Lerobot-train's loop:
```python
for step in range(0, cfg.steps):
    batch = next(dl_iter)
    batch = preprocessor(batch)
    update_policy(..., policy, batch, optimizer, ...)
```

`update_policy()` calls `policy.forward_loss(batch)` which (for our policy) returns the same loss dict our custom script does. We add a single line:

```python
# In update_policy (or via a wrapper subclass): before forward.
if hasattr(policy, "set_training_step"):
    policy.set_training_step(step)
```

**Locked choice: Option β (wrapper script).**

`scripts/train_via_lerobot.py`:
- Imports `lerobot.scripts.lerobot_train as lt`.
- Maintains a module-level step counter.
- Monkey-patches `lt.update_policy` so that on each call it:
  1. Unwraps the policy from any accelerator wrapping (e.g. DDP). Use `getattr(policy, "module", policy)` or `accelerator.unwrap_model(policy)` if the accelerator handle is reachable.
  2. Calls `raw_policy.set_training_step(step_counter)` before delegating to the original `update_policy`.
  3. Increments `step_counter`.
- Invokes the lerobot-train CLI entry function as the final action.

Why β over α:
- No upstream lerobot patch → no rebase debt every time `~/Robotic_infra/lerobot` updates.
- All migration code lives in our repo.
- Easy to review.

Alpha-style upstream patch is deferred until we want multi-GPU (DDP) — at that point we'll consolidate.

Inside `HFRVLAPolicy`:

```python
def set_training_step(self, step: int) -> None:
    """Curriculum controller. Called by the training loop each step."""
    warmup = self.config.curriculum_warmup_steps      # default 1000
    joint  = self.config.curriculum_joint_steps       # default 49000
    refine_start = warmup + joint

    if step < warmup:
        stage = 0
        # Stage 0: only L_delta. Freeze gate/contact, zero out lambdas.
        self.config.loss_lambda_gate    = 0.0
        self.config.loss_lambda_contact = 0.0
        self._set_head_grads(gate=False, contact=False)
    elif step < refine_start:
        stage = 1
        if self._prev_stage != 1:
            # Restore lambdas + unfreeze.
            self.config.loss_lambda_gate    = self._cached_lambda_gate
            self.config.loss_lambda_contact = self._cached_lambda_contact
            self._set_head_grads(gate=True, contact=True)
    else:
        stage = 2  # LR drop handled by config-driven scheduler; no parameter change here.

    self._prev_stage = stage
```

LR drop for Stage 2: configure `lerobot-train --policy.scheduler_name=cosine_with_restarts` or use a `MultiStepLR` shim. Codex decides at impl time; if no clean lerobot-train flag works, hardcode the LR schedule in `set_training_step` by mutating `param_group['lr']` directly.

### 4.5 Recording script (`record_hfrvla_libero.py`)

```python
# Pseudocode
ds_libero = LeRobotDataset("HuggingFaceVLA/libero")
ds_new = LeRobotDataset.create(
    repo_id="HFRVLA_libero_v1",
    fps=10,
    features={
        # Standard LIBERO features (verbatim from ds_libero.meta.features)
        "observation.images.image":  {"dtype": "image", "shape": (256,256,3), ...},
        "observation.images.image2": {"dtype": "image", "shape": (256,256,3), ...},
        "observation.state":         {"dtype": "float32", "shape": (8,), ...},
        "action":                    {"dtype": "float32", "shape": (7,), ...},
        # New extras
        "observation.extra.z_goal":        {"dtype": "float32", "shape": (Z_GOAL_DIM,), ...},
        "observation.extra.z_phase":       {"dtype": "float32", "shape": (Z_PHASE_DIM,), ...},
        "observation.extra.a_base":        {"dtype": "float32", "shape": (7,), ...},
        "observation.extra.k_idx_norm":    {"dtype": "float32", "shape": (1,), ...},
        "observation.extra.dino_patches":  {"dtype": "float32", "shape": (196, 384), ...},
        "observation.extra.contact_label": {"dtype": "float32", "shape": (1,), ...},
    },
    root="checkpoints/HFRVLA_libero_v1",
    use_videos=True,
    streaming_encoding=True,
)

policy = HFRVLAPolicy.from_pretrained("lerobot/smolvla_base", config=...)
dinov3 = DINOv3Backbone(...)

for ep_idx in range(ds_libero.num_episodes):
    policy.reset()
    for t in range(ep_from, ep_to):
        sample = ds_libero[t]
        # SmolVLA chunk inference (same as current precompute_libero.py)
        z_goal, z_phase, a_base, k_norm = ...
        dino_patches = dinov3(normalize_for_dinov3(sample["observation.images.image2"]))
        frame = {
            "observation.images.image":   sample["observation.images.image"],
            "observation.images.image2":  sample["observation.images.image2"],
            "observation.state":          sample["observation.state"],
            "action":                     sample["action"],
            "observation.extra.z_goal":        z_goal,
            "observation.extra.z_phase":       z_phase,
            "observation.extra.a_base":        a_base,
            "observation.extra.k_idx_norm":    torch.tensor([k_norm], dtype=torch.float32),
            "observation.extra.dino_patches":  dino_patches,
            "observation.extra.contact_label": torch.tensor([0.0], dtype=torch.float32),
            "task": sample["task"],
        }
        ds_new.add_frame(frame)
    ds_new.save_episode()

ds_new.finalize()
```

Sharding / resume:
- `LeRobotDataset.resume(repo_id=..., root=...)` is confirmed to exist (see `lerobot/datasets/lerobot_dataset.py:712`).
- The recording script should support `--resume` (calls `resume()` instead of `create()`) and `--ep-from / --ep-to` for shard-based parallel runs.
- For first-pass implementation: single-process run is acceptable (skip sharding) to avoid concurrency bugs. Adding sharding can be a follow-up.

### 4.6 Training invocation (target)

```bash
~/Robotic_infra/lerobot/.venv/bin/python scripts/train_via_lerobot.py \
    --dataset.repo_id=HFRVLA_libero_v1 \
    --dataset.root=~/Patrick/VLA_research/Hierachical_fast_reactive/checkpoints/HFRVLA_libero_v1 \
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

(`train_via_lerobot.py` is the thin monkey-patch wrapper from §4.4; verified
lerobot-train flags: `--batch_size`, `--num_workers`, `--steps`, `--wandb.enable`,
`--wandb.project`, `--wandb.entity`, `--wandb.disable_artifact`.)

DINOv3 paths are NOT passed at training time — at this point the precomputed
`observation.extra.dino_patches` are already in the dataset, so the DINOv3
backbone is not used during training. They remain needed only for
`record_hfrvla_libero.py` and for `select_action` at eval time (where the
saved checkpoint already contains the frozen DINO weights inside
`model.safetensors`).

Each "step" = optimizer step, exactly as before. Total = 60k.

Output structure (lerobot-train standard):
```
checkpoints/hfrvla_run02/
├── checkpoints/
│   ├── 005000/pretrained_model/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── policy_preprocessor.json
│   │   └── policy_postprocessor.json
│   ├── 010000/...
│   └── last/  (symlink)
├── train_config.json
└── wandb/
```

`pretrained_model/` is directly loadable by `lerobot-eval --policy.path=...` (no packaging script needed).

### 4.7 Eval invocation (target)

```bash
~/Robotic_infra/lerobot/.venv/bin/lerobot-eval \
    --policy.path=checkpoints/hfrvla_run02/checkpoints/last/pretrained_model \
    --env.type=libero \
    --env.task=libero_spatial \
    --eval.n_episodes=20 \
    --eval.batch_size=4 \
    --output_dir=outputs/eval_run02_libero_spatial \
    --seed=42
```

---

## 5. Validation plan

Three test gates that Codex must hit before reporting done:

| Gate | What to verify | How |
|---|---|---|
| **G1: Smoke recording** | `record_hfrvla_libero.py --max-episodes 5` produces a loadable LeRobotDataset where `ds[0]` returns all extras with correct shapes. | Run, then `python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; d=LeRobotDataset(root=...); print(d[0])"`. |
| **G2: Smoke training** | `lerobot-train ... --steps 100 --batch_size 8` runs to completion, hits all 3 curriculum stages (with warmup=10, joint=80, refine=10), losses sane (delta drops, gate moves off 0.69, contact drops). | Read wandb log + final ckpt. |
| **G3: Smoke eval** | `lerobot-eval --policy.path=<step_100_ckpt> --env.type=libero --env.task=libero_spatial --env.task_ids='[0]' --eval.n_episodes=1` completes end-to-end. | Run; expect `pc_success=0.0` (only 100 steps trained); the point is no crash. |

If G1 or G2 fail, fix and re-run. G3 is the deliverable confirming the pipeline is end-to-end native.

---

## 6. Open risks

| Risk | Severity | Mitigation |
|---|---|---|
| `observation_delta_indices` doesn't apply to arbitrary `observation.extra.*` keys (only walks 2-level depth) | Medium | Fall back to `observation.z_goal` etc. (no `extra.` infix); spec already calls this out in §4.2. |
| `LeRobotDataset.create` rejects non-image tensor features at parquet write time | Low | Pre-flight test in G1 (5 episodes); if it breaks, document and pivot to `numpy` dtype encoding. |
| `lerobot-train` doesn't expose `--policy.curriculum_*` via draccus CLI because they're not standard `PreTrainedConfig` fields | Medium | Add the curriculum fields as dataclass fields on `HFRVLAConfig` (already a `PreTrainedConfig` subclass) — draccus picks them up automatically. |
| Embedding curriculum in `HFRVLAPolicy.forward` requires knowing `step`, which lerobot-train doesn't pass to forward | High | This is the §4.4 hook problem. Codex must implement Option α (5-line upstream patch) OR β (wrapper script). Cannot defer. |
| Parquet write speed too slow when each row has a 300 KB `dino_patches` tensor | Medium | Profile in G1; if slow, switch from `float32` to `float16` for `dino_patches` (halves storage, minor precision loss). Document the choice. |
| LR drop at Stage 2 (step 50000) doesn't fit cleanly into lerobot's scheduler config | Low | Inside `set_training_step`, mutate optimizer's `param_groups[*]['lr']` directly at the boundary; ignore the scheduler. |
| Recording takes 12+ hr and crashes mid-way | Medium | Implement `--ep-from`/`--ep-to`/`--resume` (use `LeRobotDataset.resume()` if available; otherwise process all episodes in a single run and accept the risk). |

---

## 7. Out-of-spec behaviour to preserve

- All `HFRVLAConfig` knobs (delta_max, gate/contact lambdas, head dims, safety velocity limits, etc.) remain as-is.
- `select_action` (inference path) is untouched.
- `FastReactiveModule` architecture is untouched.
- `processor_hfrvla.normalize_for_dinov3` is untouched (still used inside `record_hfrvla_libero.py`).
- DINOv3 frozen loading from local repo + .pth weights is untouched.

---

## 8. Acceptance criteria

This migration is done when:

1. ✅ `record_hfrvla_libero.py` exists and produces a valid LeRobotDataset (G1).
2. ✅ `HFRVLAPolicy` consumes LeRobot-style batches with curriculum control via `set_training_step` (G2).
3. ✅ `lerobot-train` runs end-to-end on 100 steps with all 3 stages exercised (G2).
4. ✅ `lerobot-eval` runs end-to-end on the resulting checkpoint (G3).
5. ✅ `docs/training.md` rewritten to reflect the new flow (record → lerobot-train → lerobot-eval).
6. ✅ Legacy files (`precompute_libero.py`, `train_hfrvla.py`, `data.py`, `package_hfrvla_checkpoint.py` if no longer needed) moved to `legacy/` OR deleted with a note in commit message.
7. ✅ A 60k full training run is **not** part of this migration — Codex only needs to verify the pipeline; Patrick runs the full training afterwards.
