# Implementation Spec — HFRVLA LeRobot Plugin

> Contract for the implementing agent (Codex).
> Verifier: Claude. Author: Patrick Chu.
> Last updated: 2026-06-05.

This document is the **single source of truth** for implementation. Any
deviation must be flagged and justified.

**Current paper mainline.** Use Fast Wrist Residual (FWR), especially
`residual_merge_mode="fast_wrist_chunk"` for the schema-v3 generated-chunk
cache. The deployed merge is:

```text
a_final = a_base + alpha * clip(delta_a)
```

The older gated / contact / GRU path remains documented below only as a legacy
compatibility path for old checkpoints and scripts. Do not use it as the
default paper method unless the project explicitly reactivates that line.

---

## 0 ·  Goal

Implement a LeRobot plugin called `lerobot_policy_hfrvla` that:
1. Wraps a frozen pretrained SmolVLA policy (System 2).
2. Adds a small trainable Fast Reactive Module (System 1) using DINOv3 ViT-S/16 wrist features.
3. Computes a corrected action at every control step, preserving SmolVLA's base action and applying safety limits only to the fast residual. The current Fast Wrist Residual modes use `a_final = a_base + alpha · clip(δa)` with no gate, contact head, or GRU. The gated merge `a_final = a_base + g · clip(δa)` is legacy-only.
4. Trains end-to-end (System 1 only) on LIBERO demos via offline decomposition of expert vs. SmolVLA actions.
5. Is registered through entry points so `lerobot-train --policy.type=hfrvla` works.

The plugin lives at:
```
Hierachical_fast_reactive/policy/lerobot_policy_hfrvla/
```

This mirrors the pattern of `smolvla_engram` in this same workspace; reuse that as the reference for boilerplate.

---

## 1 ·  Directory layout

```
Hierachical_fast_reactive/policy/lerobot_policy_hfrvla/
├── pyproject.toml
└── src/lerobot_policy_hfrvla/
    ├── __init__.py
    ├── configuration_hfrvla.py
    ├── modeling_hfrvla.py
    ├── fast_reactive.py
    ├── fast_wrist_residual.py
    ├── dinov3_backbone.py
    ├── fast_cache_dataset.py
    ├── processor_hfrvla.py
    └── data.py
```

Training entry points live at:
```
Hierachical_fast_reactive/scripts/train_via_lerobot.py
Hierachical_fast_reactive/scripts/train_hfrvla_libero_merged.sh
```

---

## 2 ·  pyproject.toml

```toml
[project]
name = "lerobot_policy_hfrvla"
version = "0.1.0"
description = "LeRobot plugin: Hierarchical Fast-Reactive VLA with DINOv3 wrist-camera residual on frozen SmolVLA."
requires-python = ">=3.12"
dependencies = [
    "lerobot>=0.5.0",
    "torch>=2.7.1",
    "torchvision>=0.22",
    "transformers>=4.56.0",
    "einops>=0.8",
]

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[project.entry-points."lerobot.policies"]
hfrvla = "lerobot_policy_hfrvla:HFRVLAPolicy"
```

---

## 3 ·  configuration_hfrvla.py

```python
from dataclasses import dataclass, field
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

@dataclass
class HFRVLAConfig(SmolVLAConfig):
    """
    Inherits every SmolVLA field; adds the Fast Reactive Module knobs.
    """
    # Slow VLA control — frozen at training time
    freeze_smolvla: bool = True

    # DINOv3 backbone
    dinov3_model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    dinov3_image_size: int = 224         # → 14×14 = 196 patches
    dinov3_feature_dim: int = 384         # ViT-S/16 hidden
    dinov3_frozen: bool = True

    # Cross-attn pool (S-B)
    pool_query_dim: int = 256
    pool_n_heads: int = 4

    # GRU
    gru_hidden: int = 256
    gru_layers: int = 1

    # Heads
    head_hidden: int = 64
    delta_max: float = 0.2                # per-DoF normalized clamp
    contact_head_enabled: bool = True     # training-only; dropped at inference

    # Loss weights
    loss_lambda_gate: float = 1.0
    loss_lambda_contact: float = 0.1
    loss_delta_target_clip: bool = True
    gate_improvement_margin: float = 0.5
    loss_lambda_final: float = 1.0
    loss_lambda_preserve: float = 0.5
    loss_lambda_gate_prior: float = 0.10
    loss_lambda_preserve_zero: float = 1.0
    err_preserve_thresh: float = 0.5

    # Stage B rate-distortion objective; disabled by default for Stage A compatibility.
    use_stage_b_objective: bool = False
    loss_lambda_correct: float = 1.0
    loss_lambda_rate: float = 0.5
    loss_lambda_smooth: float = 0.2
    focal_gamma: float = 2.0
    focal_pos_weight: float = 4.0
    gate_task_budget: float = 0.25

    # Fast Wrist Residual modes. "gated" preserves the original HFRVLA path.
    # "fast_wrist" builds the stateless previous/current correction head.
    # "fast_wrist_chunk" attends over the full frozen base action chunk.
    # Deprecated "a2c2" aliases load as "fast_wrist".
    residual_merge_mode: str = "gated"
    fast_residual_alpha: float = 1.0
    fast_residual_use_latent_context: bool = True

    # Training-time window length. Gated mode consumes the full sequence.
    # FWR-v1 requires seq_len >= 2 and uses the previous/current pair.
    # FWR-v2 chunk mode uses the current frame plus a v3 full-chunk cache.
    seq_len: int = 8

    # Hooks on SmolVLA
    hook_zgoal_layer: str = "model.vlm_with_expert.vlm.text_model.layers.-1"
    hook_zphase_layer: str = "model.vlm_with_expert.lm_expert.layers.-1"

    # Inference
    safety_joint_velocity_limit: float = 2.0   # residual-only per-DoF limit

    name: str = "hfrvla"
```

The two hook layer paths are guesses; the implementer must inspect the SmolVLA model and pick the correct module path to register a `forward_hook`. Document the chosen layers in code comments.

---

## 4 ·  dinov3_backbone.py

```python
import torch
import torch.nn as nn
from transformers import AutoModel
from einops import rearrange

class DINOv3Backbone(nn.Module):
    """
    Frozen DINOv3 ViT-S/16. Exposes patch features as (B, N_patches, D).
    Patch grid for 224×224 input with patch=16 is 14×14 = 196 patches.

    Args:
        model_id: e.g. "facebook/dinov3-vits16-pretrain-lvd1689m"
        frozen: if True (default), set requires_grad=False on all params and
                run forward inside torch.no_grad() via .eval() + grad-off.
    """
    def __init__(self, model_id: str, frozen: bool = True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id)
        if frozen:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        self.frozen = frozen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, 224, 224) — normalized RGB
        returns: (B, N_patches, D) patch tokens, **excluding** CLS.
        """
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            out = self.model(pixel_values=x)
        # out.last_hidden_state has shape (B, 1+N_patches, D); first is CLS, rest are patches.
        return out.last_hidden_state[:, 1:, :]
```

Implementer must verify the actual transformers API returns CLS as token 0 (typical for ViT) — adjust slice index if DINOv3 differs.

---

## 5 ·  fast_reactive.py

This is the core trainable module.

### 5.1 Interface

```python
class FastReactiveModule(nn.Module):
    def forward(
        self,
        wrist_rgb: torch.Tensor,        # (B, 3, 224, 224) — already preprocessed for DINOv3
        proprio: torch.Tensor,           # (B, 14)
        a_base_k: torch.Tensor,          # (B, action_dim)
        k_idx_norm: torch.Tensor,        # (B, 1) ∈ [0, 1]
        z_goal: torch.Tensor,            # (B, 256) detached SmolVLA hook
        z_phase: torch.Tensor,           # (B, 256) detached SmolVLA hook
        hidden_state: torch.Tensor | None = None,   # GRU h, (1, B, 256)
        *,
        dino_patches: torch.Tensor | None = None,   # (B, 196, 384) — precomputed cache; if provided, skip backbone forward
    ) -> FastReactiveOutput:
        ...
```

### 5.2 Internal architecture

```
1. patches:       wrist_rgb → DINOv3 → (B, 196, 384)      (or use dino_patches cache)
2. patches_proj:  Linear(384 → 256) → (B, 196, 256)
3. pool query:    q = MLP_q(concat(z_goal, z_phase))  → (B, 1, 256)
4. cross-attn:    MultiheadAttention(q, K=patches_proj, V=patches_proj, n_heads=4)
                   → pooled (B, 256)
5. state vec:     state = Linear( concat(pooled, proprio, a_base_k, k_idx_norm, z_phase) → 256 )
6. GRU:           h_new = GRUCell or 1-layer GRU(state, hidden_state)
                   → (B, 256)
7. heads:         delta_a    = MLP_delta(h_new)          → (B, action_dim)
                  gate_logit = MLP_gate(h_new)            → (B,)
                  contact_logit = MLP_contact(h_new)      → (B,)  (only built if config.contact_head_enabled)
```

`MLP_q`, `MLP_delta`, `MLP_gate`, `MLP_contact` are all (in → 64 → out) two-layer MLPs with GELU. `MLP_gate` and `MLP_contact` output single logits (no sigmoid applied — caller applies sigmoid for the final value but logits are passed to BCE loss).

### 5.3 Output dataclass

```python
@dataclass
class FastReactiveOutput:
    delta_a: torch.Tensor       # (B, action_dim)
    gate_logit: torch.Tensor    # (B,)  — pre-sigmoid
    gate: torch.Tensor          # (B,)  — sigmoid(gate_logit)
    contact_logit: torch.Tensor | None  # (B,)
    hidden_state: torch.Tensor   # (1, B, 256)
```

### 5.4 Parameter count budget

Total trainable parameters of `FastReactiveModule` (excluding the frozen DINOv3) must be **< 10 M**. Implementer must include a `count_parameters()` method that prints and returns the count for verification.

---

## 5A ·  fast_wrist_residual.py

This is the simplified Fast Wrist Residual path for the paper comparison. It keeps
the frozen SmolVLA slow planner and wrist DINO features, but removes the gated
HFRVLA temporal machinery.

### 5A.1 Interface

```python
class FastWristResidualModule(nn.Module):
    def forward(
        self,
        wrist_rgb: torch.Tensor | None,
        proprio: torch.Tensor,       # (B, state_dim), current frame
        a_base_k: torch.Tensor,      # (B, action_dim), current base action
        k_idx_norm: torch.Tensor,    # (B, 1), current chunk index
        z_goal: torch.Tensor,        # (B, zgoal_dim), optional latent context
        z_phase: torch.Tensor,       # (B, zphase_dim), optional latent context
        prev_a_base: torch.Tensor,   # (B, action_dim), previous base action
        prev_delta: torch.Tensor,    # (B, action_dim), previous correction
        *,
        dino_patches: torch.Tensor | None = None,
    ) -> FastWristResidualOutput:
        ...
```

### 5A.2 Architecture and objective

The module is stateless: no GRU, no gate, no contact head. It cross-attention
pools current wrist DINO patches, concatenates current proprio/base action/chunk
position, optional SmolVLA latent context, and one-step previous-action
conditioning (`prev_a_base`, `prev_delta`), then predicts a raw 7D residual.

Training uses teacher-forced previous correction:

```python
prev_delta = action_{t-1} - a_base_{t-1}
target_delta = action_t - a_base_t
delta_exec = fast_residual_alpha * clip(delta_pred)
a_hat = a_base_t + delta_exec
loss = (
    lambda_delta * smooth_l1(delta_pred, target_delta)
    + lambda_final * smooth_l1(a_hat, action_t)
    + lambda_residual * mean(delta_exec ** 2)
    + lambda_clip * mean(relu(abs(delta_pred) - residual_cap))
)
```

The raw target remains available as an auxiliary term, but the main objective
now includes the same clipped and alpha-scaled action that inference executes:

```python
a_final = a_base + fast_residual_alpha * clip(delta_pred)
prev_delta_next = fast_residual_alpha * clip(delta_pred)
```

The first intended run uses `seq_len=2`, which means current plus one previous
frame. Episode starts use the existing LeRobot window clamp behavior.

`FastWristChunkResidualModule` is the FWR-v2 mode. It receives
`a_base_chunk: (B, 50, 7)` and `chunk_step_idx: (B, 1)`, builds action tokens
from `a_base_chunk[j]`, `j_norm`, `(j-k)/(K-1)`, `abs(j-k)/(K-1)`, and
`cos(a_base_chunk[j], a_base_chunk[k])`, then cross-attends over the 50 tokens
to predict the current-step residual. The merge and deployment-aligned FWR loss
remain the same as FWR-v1.

---

## 6 ·  modeling_hfrvla.py

`HFRVLAPolicy(SmolVLAPolicy)` — inherits from SmolVLA but **composes** the fast reactive module rather than restructuring.

### 6.1 Composition contract

```python
class HFRVLAPolicy(SmolVLAPolicy):
    config_class = HFRVLAConfig
    name = "hfrvla"

    def __init__(self, config: HFRVLAConfig, **kwargs):
        super().__init__(config, **kwargs)
        # Freeze SmolVLA weights
        if config.freeze_smolvla:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

        # Register forward hooks to capture z_goal, z_phase
        self._zgoal_cache: torch.Tensor | None = None
        self._zphase_cache: torch.Tensor | None = None
        self._register_hooks(config.hook_zgoal_layer, config.hook_zphase_layer)

        # Build fast reactive module
        self.fast = FastReactiveModule(config)
        self._fast_hidden_state: torch.Tensor | None = None

    def _register_hooks(self, zgoal_path: str, zphase_path: str) -> None:
        """Walk the SmolVLA tree to find the named layers and attach hooks
        that write into self._zgoal_cache / self._zphase_cache."""
        ...

    def reset(self) -> None:
        super().reset()
        self._fast_hidden_state = None
        self._zgoal_cache = None
        self._zphase_cache = None
```

### 6.2 Override `select_action`

```python
def select_action(self, batch, **kwargs):
    """At every call:
       1. If SmolVLA's action queue is empty → call _get_action_chunk()
          (this triggers SmolVLA forward, which populates _zgoal_cache and
          _zphase_cache via hooks).
       2. Pop a_base from the queue.
       3. Run self.fast(wrist_rgb, proprio, a_base, k_idx, z_goal, z_phase, h_state).
       4. Compute a_final via the merge formula, clamping only the fast residual.
       5. Return a_final.
    """
```

### 6.3 Override `forward` (training)

Used by the training loop. Receives a batch with all needed tensors:

```python
def forward(self, batch, **kwargs) -> dict[str, torch.Tensor]:
    """Training forward pass. NOT used during inference.
       Expects batch to contain:
         - "wrist_rgb"           (B, 3, 224, 224) OR
         - "dino_patches"        (B, 196, 384)   (precomputed)
         - "proprio"             (B, 14)
         - "a_base"              (B, action_dim)
         - "k_idx_norm"          (B, 1)
         - "z_goal"              (B, 256)
         - "z_phase"             (B, 256)
         - "a_expert"            (B, action_dim)
         - "contact_label"       (B,)  (optional)
       Returns: dict with loss components and total loss.
    """
    out = self.fast(...)
    losses = self._compute_losses(out, batch)
    return {"loss": losses["total"], **losses}
```

### 6.4 Loss computation

Two objectives are supported.

**Stage A v2 (default)** keeps the deployment-aligned residual objective:

```python
target_delta = clip(a_expert - a_base)
l_delta = mse(delta_a, target_delta)
a_final = a_base + stopgrad(gate) * clip(delta_a)
l_gate = bce(gate_logit, improvement(delta_a) > gate_improvement_margin)
l_final = mse(a_final, a_expert)
l_preserve = relu(err_final - err_before).mean()
l_gate_prior = gate.mean()
l_preserve_zero = mean(||delta_a||^2 on err_before < err_preserve_thresh)
```

The `stopgrad(gate)` in the merged-action path is required: `L_final` and
`L_preserve` must not train the gate to open. Gate gradients come from `L_gate`
and the rate term only.

**Stage B (optional)** is the five-term rate-distortion correction code. It
requires fast-cache schema v2 labels:

```python
y_correct  = batch["observation.extra.y_correct"]
y_preserve = batch["observation.extra.y_preserve"]
u = clip(delta_a)
r_t = clip(a_expert - a_base)
g = sigmoid(gate_logit)

L_correct = masked_mean(sum_smooth_l1(delta_a, r_t), y_correct)
L_preserve_zero = masked_mean(sum(delta_a ** 2), y_preserve)
L_rate = mean(g) + relu(mean(g) - gate_task_budget) ** 2
L_gate = focal_bce(gate_logit, y_correct, gamma=2, pos_weight=4)
L_smooth = mean(||stopgrad(g) * u_t - stopgrad(g) * u_{t-1}||^2)
```

Stage B total loss:

```python
loss = 1.0 * L_correct + 2.0 * L_preserve_zero
       + 0.5 * L_rate + 3.0 * L_gate + 0.2 * L_smooth
       + 0.1 * L_contact
```

**Stage C (optional data extension)** keeps the Stage B objective and changes
only the data distribution. A packaged `zero_fast` checkpoint is rolled out in
LIBERO closed loop; only successful episodes are retained. The derived rollout
fastcache is built with `--static-y-preserve`, so every rollout frame has
`y_preserve=1` and `y_correct=0`. Training can concatenate the expert-demo
cache and rollout cache by setting `HFRVLA_FASTCACHE_ROLLOUT_ROOT`; leaving it
unset preserves the Stage A/B single-cache path.

**Fast Wrist Residual (`residual_merge_mode="fast_wrist"` or
`"fast_wrist_chunk"`)** bypasses Stage A/B/C losses and uses a deployment-
aligned current-step objective:

```python
target_delta = a_expert[:, -1] - a_base[:, -1]
delta_exec = fast_residual_alpha * clip(delta_pred)
a_final = a_base[:, -1] + delta_exec
l_delta = smooth_l1(delta_pred, target_delta)
l_final = smooth_l1(a_final, a_expert[:, -1])
l_residual = mean(delta_exec ** 2)
l_clip = mean(relu(abs(delta_pred) - residual_cap))
loss = (
    fast_wrist_loss_lambda_delta * l_delta
    + fast_wrist_loss_lambda_final * l_final
    + fast_wrist_loss_lambda_residual * l_residual
    + fast_wrist_loss_lambda_clip * l_clip
)
```

It intentionally has no gate BCE, contact auxiliary loss, preserve-zero loss,
rate term, or Stage B static-label term.

### 6.5 Action merging

```python
def _merge(self, a_base, delta_a, gate, prev_a, dt):
    max_residual = min(
        self.config.delta_max,
        self.config.safety_joint_velocity_limit * dt,
    )
    delta_clip = delta_a.clamp(-max_residual, +max_residual)
    if self.config.residual_merge_mode in {"fast_wrist", "fast_wrist_chunk"}:
        return a_base + self.config.fast_residual_alpha * delta_clip
    return a_base + gate.unsqueeze(-1) * delta_clip
```

### 6.6 `get_optim_params`

```python
def get_optim_params(self) -> list:
    """Return only the FastReactiveModule's trainable params.
       SmolVLA params are frozen.
    """
    return [p for p in self.fast.parameters() if p.requires_grad]
```

---

## 7 ·  Data and fast-cache pipeline

The current primary path is not the legacy `.pt` cache pipeline. Record a
local LeRobotDataset v3 that bakes in frozen SmolVLA and DINOv3 intermediate
features, then optionally build a compact frame-level fast-cache for training
speed.

Expected recorded features include:

- `observation.images.image`
- `observation.images.image2`
- `observation.state`
- `action`
- `observation.extra.z_goal`
- `observation.extra.z_phase`
- `observation.extra.a_base`
- `observation.extra.a_base_chunk` (generated slow-planner chunk; schema v3)
- `observation.extra.chunk_step_idx` (schema v3)
- `observation.extra.chunk_age_steps` (optional schema v3)
- `observation.extra.chunk_age_norm` (optional schema v3)
- `observation.extra.k_idx_norm`
- `observation.extra.dino_patches`
- `observation.extra.contact_label`

The fast-cache builder stores frame-aligned arrays only. Schema v3 stores the
full frozen base chunk, current chunk index, and optional chunk age per frame,
plus top-level `chunk_len=50`. If the canonical dataset was recorded before
generated chunks were added, the builder falls back to reconstructing chunk
arrays from frame-level `a_base` for backward compatibility. It must not accept
or write `seq_len`; the sequence/window length is a training-time sampler choice provided to
`HFRVLAFastCacheDataset(..., seq_len=...)` and `--policy.seq_len`.

Legacy `.pt` decomposition notes:

Slow-planner precondition: use `HuggingFaceVLA/smolvla_libero` or another
frozen SmolVLA checkpoint that already matches LIBERO's feature contract
(`observation.images.image`, `observation.images.image2`,
`observation.state` `(8,)`, `action` `(7,)`). The raw
`lerobot/smolvla_base` checkpoint is a warm-start model with a 6D action/state
and `camera1/2/3` config; do not use it directly for HFRVLA recording or
alignment. If the slow planner changes, regenerate the HFRVLA dataset because
`a_base`, `z_goal`, and `z_phase` are policy-dependent.

### 7.1 `precompute_chunks(lerobot_dataset, smolvla_policy, output_dir) -> None`

For each demo in `lerobot_dataset`:
1. Reset `smolvla_policy.reset()`.
2. Iterate over the demo's steps. At each step:
   - Feed the observation into `smolvla_policy.select_action()` (which internally calls `_get_action_chunk` when queue is empty).
   - At a fresh chunk boundary, capture `z_goal`, `z_phase` from the hook caches; cache them for the duration of the chunk.
   - Record per-step: `wrist_rgb`, `proprio`, `a_base` (popped from queue), `k_idx`, `z_goal_chunk`, `z_phase_chunk`, `a_expert` (from demo), `contact_label` (from LIBERO MuJoCo or robot torque heuristic).
3. Save as a single `.pt` file per demo, or one combined HDF5 file.

### 7.2 `precompute_dinov3(precomputed_dir, dinov3_model_id) -> None`

Walks the precomputed chunk records and adds a `dino_patches: (T, 196, 384)` tensor per demo. This avoids running DINOv3 forward at every training step.

### 7.3 `HFRVLAFastCacheDataset(torch.utils.data.Dataset)`

Returns dicts shaped exactly as expected by `HFRVLAPolicy.forward(...)` (see 6.3).

Supports **sequence sampling** at read time: each `__getitem__` returns
`seq_len` consecutive steps ending at the sampled frame. Gated mode consumes the
full window. FWR-v1 requires `seq_len >= 2` and uses only the previous and
current frames. FWR-v2 chunk mode reads the current window element plus
`a_base_chunk` and `chunk_step_idx` from schema v3.

---

## 8 ·  processor_hfrvla.py

Mirror `processor_smolvla.py`. Add a step that runs DINOv3 preprocessing on the wrist image (normalize, resize to 224×224). If `dino_patches` is already in the batch (training), skip image preprocessing.

---

## 9 ·  Training entry points

The primary path is LeRobot-native. Use `scripts/train_hfrvla_libero_merged.sh`
or the foreground wrapper; both call `scripts/train_via_lerobot.py`, which
monkey-patches `lerobot-train` to call `policy.set_training_step(step)` for the
curriculum and to swap in `HFRVLAFastCacheDataset` when
`HFRVLA_DATASET_BACKEND=fastcache`.

For the Fast Wrist Residual v1 path, set:

```bash
RESIDUAL_MERGE_MODE=fast_wrist
SEQ_LEN=2
FAST_RESIDUAL_ALPHA=1.0
```

For chunk-aware FWR-v2, build a schema v3 fast-cache and set
`RESIDUAL_MERGE_MODE=fast_wrist_chunk`.

`get_optim_params()` must still return only trainable fast-module parameters.

### Training Stages

| Stage | Steps | Active losses | LR |
|-------|-------|---------------|-----|
| 0. Warmup | 0 – 1k | Stage A: `L_delta` only; Stage B: static residual/preserve terms, gate + contact heads frozen | 1e-4 ramped |
| 1. Joint  | 1k – 50k | Stage A v2 objective or Stage B five-term objective, plus optional `L_contact` | 3e-4 cosine |
| 2. Refine | 50k – 60k | same | 3e-5 (1/10) |

---

## 10 ·  Verification checklist (Claude will run these)

After Codex returns code:

- [ ] `pyproject.toml` parses; entry point `hfrvla = lerobot_policy_hfrvla:HFRVLAPolicy` present.
- [ ] `from lerobot_policy_hfrvla import HFRVLAPolicy` runs without error (after `uv pip install -e ...`).
- [ ] `HFRVLAPolicy.from_pretrained("HuggingFaceVLA/smolvla_libero", config=HFRVLAConfig())` instantiates.
- [ ] `policy.fast.count_parameters() < 10_000_000`.
- [ ] Forward-hook on SmolVLA captures non-None `_zgoal_cache` and `_zphase_cache` after a forward pass.
- [ ] `_compute_losses` returns finite `delta`, `gate`, `final`, `preserve`, `gate_prior`, optional `contact`, and `loss` scalars on a synthetic batch.
- [ ] FWR modes return finite `delta` and `loss` scalars without gate/contact metrics.
- [ ] `select_action()` on a synthetic single-step batch returns an action of correct shape with no NaN.
- [ ] DINOv3 backbone forward on `(1, 3, 224, 224)` returns `(1, 196, 384)`.
- [ ] Gate target uses `stop_grad(clip(delta_a))` and opens only when the clipped residual clears `gate_improvement_margin`.

---

## 11 ·  Notes for the implementing agent

1. **Do not** add LoRA, replan signal, expected-state head, or world model. These are out of scope.
2. **Use composition**: `HFRVLAPolicy` extends `SmolVLAPolicy` but does not modify `self.model` (SmolVLA's internals). `self.fast` is a sibling.
3. **Hook caches must be cleared** in `reset()` and after every chunk consumption to avoid stale features.
4. **DINOv3 weights**: requires Meta access request. For local development assume the user has cached `facebook/dinov3-vits16-pretrain-lvd1689m` in `~/.cache/huggingface/hub`. Document the request URL in the README.
5. **Hook layer paths**: the suggested paths in section 3 are guesses. Inspect SmolVLA's actual structure via `dict(policy.model.named_modules())` and pick:
   - z_goal: a layer near the end of the VLM text encoder where vision-language fusion is mature.
   - z_phase: a layer near the start of the action expert where the chunk hidden state is best summarized.
   Document the choice with a code comment naming the layer.
6. **Type hints, docstrings, and a README.md** in the plugin directory are required.
7. **One commit per file** would be ideal but is the verifier's job, not the implementer's.

---

## 12 ·  Out of scope

- Real-hardware data collection scripts (Patrick will record demos on SO-100 separately).
- LIBERO environment setup (assumed already installed under `~/Robotic_infra`).
- Inference latency benchmarking (verifier will profile after acceptance).
- Paper writing (handled by `academic-paper` skill later).
