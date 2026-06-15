# Methodology Blueprint — HFRVLA

> Ground-truth paper-facing design document. Keep this synchronized with
> `research_questions.md`, `contribution_statement.md`, and `paper/src/`.
> Last updated: 2026-06-05.

---

## 1. Research Question

**Core question.**
Can a small fast wrist correction module improve the execution of frozen
SmolVLA action chunks without fine-tuning the slow VLA planner?

**Current hypothesis.**
Frozen VLA chunks fail partly because later actions in the chunk are executed
under stale observations. A lightweight fast path can use the latest wrist
evidence plus slow-planner context to produce a bounded per-step residual:

```text
a_final = a_base + alpha * clip(delta_a)
```

This is a fast wrist correction study under the HFRVLA name, not the older
gated HFRVLA proposal. The per-step visual correction signal is wrist-camera
based, while `a_base`, `z_goal`, `z_phase`, state, and chunk index provide
slow-planner and robot context.

**Out of current paper scope.**
- Learned gate / confidence supervision.
- Contact auxiliary head.
- GRU-centered recurrent fast head as the main contribution.
- Real-hardware validation.
- Joint fine-tuning of SmolVLA.

---

## 2. Method Overview

The current system has two layers:

1. **Frozen slow planner.** `HuggingFaceVLA/smolvla_libero` predicts action
   chunks and provides cached context tensors.
2. **Trainable fast wrist correction module.** A small wrist-centric module
   predicts `delta_a` for the action that will be executed.

Runtime contract:

```text
slow planner:
  produce base chunk A = [a_base_0, ..., a_base_H-1]

fast control tick:
  obs_now = latest wrist/state
  a_base_i = next base action from the chunk
  delta_i = HFRVLA_correction(obs_now, a_base_i, k_i, prev_delta, slow_context)
  send a_base_i + alpha * clip(delta_i)
```

The critical engineering check is that fast correction must run at the control
tick that sends the action. If the async server only calls
`predict_action_chunk()` and bypasses `select_action()`, correction may disappear
from deployment.

---

## 3. Data Contract

HFRVLA training is not plain raw LIBERO training. The project records a custom
LeRobotDataset v3 with frozen slow-planner and wrist features baked in:

| Feature | Shape | Meaning |
|---|---:|---|
| `observation.images.image` | `(256, 256, 3)` | third-person LIBERO image |
| `observation.images.image2` | `(256, 256, 3)` | wrist image |
| `observation.state` | `(8,)` | LIBERO robot state |
| `action` | `(7,)` | expert target action |
| `observation.extra.a_base` | `(7,)` | frozen SmolVLA base action |
| `observation.extra.k_idx_norm` | `(1,)` | chunk position |
| `observation.extra.z_goal` | `(960,)` | slow-planner text/task pool |
| `observation.extra.z_phase` | `(480,)` | slow-planner action-expert pool |
| `observation.extra.dino_patches` | `(196, 384)` | frozen DINOv3 wrist patches |

The verified dataset root is:

```text
checkpoints/HFRVLA_libero_v1_merged_reindexed
```

The frame-level fast-cache is an acceleration layer only. `seq_len` is a
training-time loader/windowing choice, not a cache schema property.

---

## 4. Current Architecture

**Inputs to the fast path**
- Wrist DINOv3 patches.
- Robot state.
- Current base action `a_base`.
- Chunk index / time feature.
- Slow-planner context (`z_goal`, `z_phase`).
- Previous correction when using the previous/current training window.

**Output**
- Full 7D residual `delta_a`.

**Merge**

```text
a_final = a_base + alpha * clip(delta_a, -delta_max, delta_max)
```

The first paper draft should not describe the policy as wrist-only. A precise
phrase is: **the high-frequency new visual evidence comes from the wrist camera;
global task context remains inherited from the frozen slow planner.**

---

## 5. Training Objective

The implemented baseline uses simultaneous residual supervision:

```text
target_delta = action[t] - a_base[t]
```

Current code also supports deployment-aligned final-action terms:

```text
a_hat = a_base + alpha_train * clip(delta_pred)
L_final = SmoothL1(a_hat, action[t])
```

The next high-priority method improvement is delayed chunk-age supervision:

```text
base = base_chunk_generated_at[t-k][k]
target_delta = action[t] - base
```

This aligns training with deployment, where a long-horizon action may come from
an older observation rather than from the current step.

---

## 6. Evaluation Protocol

Always distinguish:

| Axis | Meaning |
|---|---|
| `planning_chunk_size` | how many actions the slow VLA predicts |
| `execution_chunk_size` | how many actions are executed before replanning |
| `replan_interval_steps` | control interval between slow-planner refreshes |

Default every-step SmolVLA replanning is useful as an upper reference but is not
the matched baseline for long action-chunk execution. Matched comparisons must
use the same planning/execution overrides.

Current stable evidence lives in:

```text
experiments/eval_registry/eval_results_master.csv
```

The active `n_action_steps=50` alpha/clip sweep is diagnostic until complete.
Early rows should not be elevated to final paper claims.

---

## 7. Paper Positioning

Strong current claim:

> HFRVLA is a LeRobot-native study of wrist-centric residual correction for
> frozen SmolVLA action chunks. It shows that fast correction can help under
> matched short-execution regimes, while long matched chunks expose calibration
> and training/deployment-mismatch failures.

Avoid overclaiming:
- No real-hardware success claim yet.
- No universal VLA portability claim yet.
- No claim that wrist evidence alone solves global spatial ambiguity.
- No claim that long-horizon chunk execution is solved.
