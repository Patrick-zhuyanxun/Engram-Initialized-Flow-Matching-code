# HFRVLA → LeRobot-native Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate HFRVLA training from custom `train_hfrvla.py` + `.pt` precompute to fully native `lerobot-train` flow, by recording a new LeRobotDataset v3 that bakes in precomputed SmolVLA / DINOv3 features as `observation.extra.*` columns.

**Architecture:** A new `record_hfrvla_libero.py` writes a standard LeRobotDataset with extras under `observation.extra.*` (so `delta_timestamps` windowing works). `HFRVLAPolicy.forward()` is refactored to read those keys. Curriculum (3 stages, freeze/unfreeze, lambda toggles, LR drop) lives inside the policy via `set_training_step(step)`. A thin wrapper script `train_via_lerobot.py` monkey-patches lerobot-train's `update_policy` to invoke that hook per step.

**Tech Stack:** lerobot 0.5.1 (editable install at `~/Robotic_infra/lerobot`), LeRobotDataset v3 API, PyTorch 2.7+, draccus config, wandb 0.24+, A5000 GPU.

**Spec reference:** `docs/superpowers/specs/2026-05-15-hfrvla-lerobot-native-design.md` (read this first if any task is ambiguous).

**Working directory for all commands:** `~/Patrick/VLA_research/Hierachical_fast_reactive` unless explicitly stated otherwise.

**Python interpreter for all commands:** `~/Robotic_infra/lerobot/.venv/bin/python`.

---

## File Structure (target after migration)

```
Hierachical_fast_reactive/
├── policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/
│   ├── configuration_hfrvla.py   (MODIFY: + curriculum fields + observation_delta_indices)
│   ├── modeling_hfrvla.py        (MODIFY: forward() reads LeRobot keys; + set_training_step)
│   ├── fast_reactive.py          (no change)
│   ├── dinov3_backbone.py        (no change)
│   ├── processor_hfrvla.py       (no change)
│   └── __init__.py               (no change)
├── scripts/
│   ├── record_hfrvla_libero.py   (NEW)
│   ├── train_via_lerobot.py      (NEW)
│   ├── package_hfrvla_checkpoint.py (UNCHANGED; kept for legacy ckpt repackaging)
│   └── legacy/
│       ├── README.md             (NEW: short note explaining why these are here)
│       ├── train_hfrvla.py       (MOVED)
│       ├── precompute_libero.py  (MOVED)
│       ├── precompute_libero_plan.py (MOVED)
│       ├── precompute_libero_parallel.sh (MOVED)
│       └── data.py               (MOVED from plugin src; renamed to avoid conflict)
├── tests/
│   ├── test_hfrvla_config.py     (NEW)
│   └── test_hfrvla_forward_lerobot_batch.py (NEW)
├── docs/
│   └── training.md               (REWRITE)
└── checkpoints/
    └── HFRVLA_libero_v1/         (NEW — recording output; created by Task 5)
```

---

## Pre-flight (do once before starting)

- [ ] **Step 0.1: Verify lerobot venv & plugin are reachable**

```bash
cd ~/Patrick/VLA_research/Hierachical_fast_reactive
~/Robotic_infra/lerobot/.venv/bin/python -c "
import lerobot_policy_hfrvla
from lerobot.configs.policies import PreTrainedConfig
print('plugin OK')
assert 'hfrvla' in PreTrainedConfig.get_known_choices(), 'hfrvla not registered'
print('config registered OK')
"
```

Expected output:
```
plugin OK
config registered OK
```

If the second line fails, run `cd ~/Robotic_infra/lerobot && uv pip install -e ~/Patrick/VLA_research/Hierachical_fast_reactive/policy/lerobot_policy_hfrvla` and retry.

- [ ] **Step 0.2: Create branch**

```bash
git checkout -b lerobot-native-migration
git status
```

Expected: clean working tree on new branch.

---

## Task 1: Add curriculum fields + observation_delta_indices to HFRVLAConfig

**Files:**
- Modify: `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/configuration_hfrvla.py`
- Test: `tests/test_hfrvla_config.py` (new)

**Rationale:** The policy needs three new config knobs (warmup/joint/refine step counts) so curriculum boundaries can be configured via `--policy.curriculum_*` CLI flags. `observation_delta_indices` tells lerobot-train how to apply `delta_timestamps` for sequence windowing.

- [ ] **Step 1.1: Write the failing test**

Create file `tests/test_hfrvla_config.py`:

```python
"""Tests for HFRVLAConfig curriculum + windowing additions."""

from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig


def _bare_config() -> HFRVLAConfig:
    # Bypass SmolVLA hub download: build a config from defaults only.
    cfg = HFRVLAConfig()
    return cfg


def test_curriculum_defaults_present():
    cfg = _bare_config()
    assert cfg.curriculum_warmup_steps == 1000
    assert cfg.curriculum_joint_steps == 49000
    assert cfg.curriculum_refine_steps == 10000


def test_curriculum_total_helper():
    cfg = _bare_config()
    assert cfg.curriculum_total_steps() == 60000


def test_observation_delta_indices_default_seq_len_8():
    cfg = _bare_config()
    # 8 consecutive steps ending at the current frame.
    assert cfg.observation_delta_indices == list(range(-7, 1))
    assert len(cfg.observation_delta_indices) == 8
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
~/Robotic_infra/lerobot/.venv/bin/python -m pytest tests/test_hfrvla_config.py -v
```

Expected: 3 failures with `AttributeError: 'HFRVLAConfig' object has no attribute 'curriculum_warmup_steps'` (or similar) on the first test, and similar errors on the others.

- [ ] **Step 1.3: Add the fields and properties**

In `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/configuration_hfrvla.py`, find the `@dataclass class HFRVLAConfig(SmolVLAConfig):` body. Add (place after `# ── Loss weights ──` block, before `# ── Hook target layer names ──`):

```python
    # ── Curriculum (sprint-2 three-stage) ──
    # Stage 0 (Warmup): only L_delta, gate + contact heads frozen.
    # Stage 1 (Joint):  L_delta + λ_gate·L_gate + λ_contact·L_contact.
    # Stage 2 (Refine): same losses; LR ÷ 10 (handled by set_training_step).
    curriculum_warmup_steps: int = 1000
    curriculum_joint_steps: int = 49000
    curriculum_refine_steps: int = 10000

    # ── Sequence windowing for GRU ──
    seq_len: int = 8
```

Then add the helper method + property near the bottom of the class (after `from_smolvla` classmethod):

```python
    def curriculum_total_steps(self) -> int:
        return (
            self.curriculum_warmup_steps
            + self.curriculum_joint_steps
            + self.curriculum_refine_steps
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        """Indices passed to lerobot's delta_timestamps mechanism.

        Returns ``[-(seq_len-1), ..., -1, 0]`` so all observation.* keys are
        windowed to ``seq_len`` consecutive steps ending at the current frame.
        """
        return list(range(-(self.seq_len - 1), 1))
```

- [ ] **Step 1.4: Run test to verify it passes**

```bash
~/Robotic_infra/lerobot/.venv/bin/python -m pytest tests/test_hfrvla_config.py -v
```

Expected: 3 passed.

- [ ] **Step 1.5: Commit**

```bash
git add policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/configuration_hfrvla.py tests/test_hfrvla_config.py
git commit -m "feat(hfrvla-config): add curriculum step fields and observation_delta_indices"
```

---

## Task 2: Refactor HFRVLAPolicy.forward + add set_training_step

**Files:**
- Modify: `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/modeling_hfrvla.py`
- Test: `tests/test_hfrvla_forward_lerobot_batch.py` (new)

**Rationale:** `forward()` currently expects bare keys (`a_base`, `z_goal`, etc.) from the legacy custom dataset. After migration, the LeRobot dataloader gives us `observation.extra.*` keys. We rewire `forward()` to accept these. The curriculum hook `set_training_step(step)` mutates loss lambdas and head gradients based on the configured boundaries.

- [ ] **Step 2.1: Write the failing test**

Create file `tests/test_hfrvla_forward_lerobot_batch.py`:

```python
"""Tests for HFRVLAPolicy.forward consuming LeRobot batch keys + curriculum."""

import pytest
import torch

from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig
from lerobot_policy_hfrvla.fast_reactive import FastReactiveModule


# ────────────────────────────────────────────────────────────────────────
# We don't load the full HFRVLAPolicy (it pulls SmolVLA from HF Hub). We
# test the bits that don't need SmolVLA:
#   - FastReactiveModule still works with cached dino_patches (no SmolVLA).
#   - Curriculum logic is exercised directly on a stub policy.
# ────────────────────────────────────────────────────────────────────────


def _make_fast_module(seq_len: int = 8):
    cfg = HFRVLAConfig(seq_len=seq_len)
    fast = FastReactiveModule(
        config=cfg,
        action_dim=7,
        proprio_dim=8,
        zgoal_dim=960,
        zphase_dim=960,
    )
    return fast, cfg


def test_fast_module_accepts_time_windowed_dino_patches():
    fast, cfg = _make_fast_module(seq_len=8)
    B, T = 2, 8
    dino_patches = torch.randn(B, T, 196, 384)
    proprio      = torch.randn(B, T, 8)
    a_base       = torch.randn(B, T, 7)
    k_idx_norm   = torch.rand(B, T, 1)
    z_goal       = torch.randn(B, T, 960)
    z_phase      = torch.randn(B, T, 960)

    out = fast(
        wrist_rgb=None,
        proprio=proprio,
        a_base_k=a_base,
        k_idx_norm=k_idx_norm,
        z_goal=z_goal,
        z_phase=z_phase,
        dino_patches=dino_patches,
    )
    # delta_a should mirror the time + batch dim.
    assert out.delta_a.shape == (B, T, 7), f"got {out.delta_a.shape}"


class _StubPolicy:
    """Stand-in for HFRVLAPolicy.set_training_step, isolated from SmolVLA."""

    def __init__(self):
        self.config = HFRVLAConfig(
            curriculum_warmup_steps=10,
            curriculum_joint_steps=80,
            curriculum_refine_steps=10,
        )
        # Cache base lambdas (real policy does this in __init__).
        self._cached_lambda_gate = self.config.loss_lambda_gate
        self._cached_lambda_contact = self.config.loss_lambda_contact
        self._prev_stage = -1
        self._refine_lr_applied = False

        self.fast = FastReactiveModule(
            config=self.config,
            action_dim=7,
            proprio_dim=8,
            zgoal_dim=960,
            zphase_dim=960,
        )

    # We expect Task 2 to add this method on HFRVLAPolicy. Mirror its
    # semantics here so the test specifies the contract.
    def set_training_step(self, step: int) -> None:
        from lerobot_policy_hfrvla.modeling_hfrvla import (
            _apply_curriculum_stage,
        )
        _apply_curriculum_stage(self, step)


def test_curriculum_stage0_freezes_heads_and_zeroes_lambdas():
    pol = _StubPolicy()
    pol.set_training_step(0)
    assert pol.config.loss_lambda_gate == 0.0
    assert pol.config.loss_lambda_contact == 0.0
    assert all(not p.requires_grad for p in pol.fast.gate_head.parameters())
    if pol.fast.contact_head is not None:
        assert all(not p.requires_grad for p in pol.fast.contact_head.parameters())


def test_curriculum_stage1_unfreezes_heads_and_restores_lambdas():
    pol = _StubPolicy()
    pol.set_training_step(0)   # stage 0
    pol.set_training_step(10)  # stage 1 entry (warmup_steps == 10)
    assert pol.config.loss_lambda_gate    == pol._cached_lambda_gate
    assert pol.config.loss_lambda_contact == pol._cached_lambda_contact
    assert all(p.requires_grad for p in pol.fast.gate_head.parameters())


def test_curriculum_stage2_marks_refine_lr_pending():
    pol = _StubPolicy()
    pol.set_training_step(0)
    pol.set_training_step(50)   # joint
    pol.set_training_step(90)   # refine_start = 10 + 80 = 90
    assert pol._prev_stage == 2
    # set_training_step must set _refine_lr_pending = True at the boundary
    # so the wrapper script knows to drop LR. (The wrapper consumes & clears
    # the flag.)
    assert pol._refine_lr_pending is True
```

- [ ] **Step 2.2: Run test to verify it fails**

```bash
~/Robotic_infra/lerobot/.venv/bin/python -m pytest tests/test_hfrvla_forward_lerobot_batch.py -v
```

Expected: the first test (`test_fast_module_accepts_time_windowed_dino_patches`) should pass already (FastReactiveModule handles `(B, T, ...)`). The next three fail with `ImportError: cannot import name '_apply_curriculum_stage'` or `AttributeError: ... _refine_lr_pending`.

- [ ] **Step 2.3: Add `_apply_curriculum_stage` helper + `set_training_step` to HFRVLAPolicy**

In `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/modeling_hfrvla.py`, near the top after imports, add:

```python
def _set_head_grads(policy_or_stub, *, gate: bool, contact: bool) -> None:
    """Toggle requires_grad on the fast module's auxiliary heads."""
    for p in policy_or_stub.fast.gate_head.parameters():
        p.requires_grad = gate
    if policy_or_stub.fast.contact_head is not None:
        for p in policy_or_stub.fast.contact_head.parameters():
            p.requires_grad = contact


def _apply_curriculum_stage(policy_or_stub, step: int) -> None:
    """Mutate lambdas + head gradients based on the configured curriculum.

    Sets ``_refine_lr_pending = True`` when crossing into stage 2 so a caller
    (the wrapper training script) can drop the optimizer LR exactly once at
    that boundary.
    """
    cfg = policy_or_stub.config
    warmup = cfg.curriculum_warmup_steps
    joint  = cfg.curriculum_joint_steps
    refine_start = warmup + joint

    if step < warmup:
        stage = 0
    elif step < refine_start:
        stage = 1
    else:
        stage = 2

    prev = getattr(policy_or_stub, "_prev_stage", -1)

    if stage == 0 and prev != 0:
        cfg.loss_lambda_gate    = 0.0
        cfg.loss_lambda_contact = 0.0
        _set_head_grads(policy_or_stub, gate=False, contact=False)
    elif stage == 1 and prev != 1:
        cfg.loss_lambda_gate    = policy_or_stub._cached_lambda_gate
        cfg.loss_lambda_contact = policy_or_stub._cached_lambda_contact
        _set_head_grads(policy_or_stub, gate=True, contact=True)
    elif stage == 2 and prev != 2:
        policy_or_stub._refine_lr_pending = True

    policy_or_stub._prev_stage = stage
```

Then, inside `class HFRVLAPolicy.__init__`, after `self.fast = FastReactiveModule(...)`, add:

```python
        # Curriculum state. The wrapper training script calls
        # set_training_step(step) before each optimizer step.
        self._cached_lambda_gate    = float(config.loss_lambda_gate)
        self._cached_lambda_contact = float(config.loss_lambda_contact)
        self._prev_stage: int = -1
        self._refine_lr_pending: bool = False
```

And add the public method on the class:

```python
    def set_training_step(self, step: int) -> None:
        """Curriculum controller. Called by train_via_lerobot.py per step."""
        _apply_curriculum_stage(self, step)

    def consume_refine_lr_signal(self) -> bool:
        """Return True exactly once, when stage 2 is first entered. The
        training wrapper uses this to drop the optimizer LR at the boundary."""
        flag = self._refine_lr_pending
        self._refine_lr_pending = False
        return flag
```

- [ ] **Step 2.4: Refactor forward() to consume LeRobot batch keys**

In `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/modeling_hfrvla.py`, replace the entire `forward()` method body. Locate the existing `def forward(self, batch, **kwargs)` block — keep the signature, replace the body with:

```python
    def forward(  # type: ignore[override]
        self, batch: dict[str, Tensor], **kwargs
    ) -> dict[str, Tensor]:
        """Training forward — consumes the LeRobot-native batch layout.

        Expected keys (all already windowed to ``[B, T=seq_len, ...]`` by
        lerobot's delta_timestamps mechanism — see HFRVLAConfig.
        observation_delta_indices):

            observation.state                        → proprio
            action                                   → a_expert (target)
            observation.extra.a_base                 → a_base
            observation.extra.k_idx_norm             → k_idx_norm
            observation.extra.z_goal                 → z_goal
            observation.extra.z_phase                → z_phase
            observation.extra.dino_patches           → dino_patches
            observation.extra.contact_label (optional) → contact_label
        """
        proprio       = batch["observation.state"]
        a_expert      = batch["action"]
        a_base        = batch["observation.extra.a_base"]
        k_idx_norm    = batch["observation.extra.k_idx_norm"]
        z_goal        = batch["observation.extra.z_goal"]
        z_phase       = batch["observation.extra.z_phase"]
        dino_patches  = batch["observation.extra.dino_patches"]
        contact_label = batch.get("observation.extra.contact_label")

        fr_out: FastReactiveOutput = self.fast(
            wrist_rgb=None,
            proprio=proprio,
            a_base_k=a_base,
            k_idx_norm=k_idx_norm,
            z_goal=z_goal,
            z_phase=z_phase,
            dino_patches=dino_patches,
        )
        losses = self._compute_losses(fr_out, a_base, a_expert, contact_label)
        return losses
```

The `_compute_losses` method does NOT need to change — it already works on shapes `(B, T, ...)` (FastReactiveOutput preserves the time dim, and `F.mse_loss` / `F.binary_cross_entropy_with_logits` reduce all dims).

- [ ] **Step 2.5: Run tests to verify pass**

```bash
~/Robotic_infra/lerobot/.venv/bin/python -m pytest tests/test_hfrvla_forward_lerobot_batch.py -v
```

Expected: 4 passed.

- [ ] **Step 2.6: Commit**

```bash
git add policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/modeling_hfrvla.py tests/test_hfrvla_forward_lerobot_batch.py
git commit -m "feat(hfrvla-policy): consume LeRobot batch keys + add curriculum hook"
```

---

## Task 3: Write the recording script `record_hfrvla_libero.py`

**Files:**
- Create: `scripts/record_hfrvla_libero.py`

**Rationale:** This replaces `precompute_libero.py`. Same logic (run SmolVLA + DINOv3 per step), different sink (LeRobotDataset writer instead of `.pt` files). The new dataset becomes the canonical input for `lerobot-train`.

- [ ] **Step 3.1: Write the script**

Create `scripts/record_hfrvla_libero.py`:

```python
#!/usr/bin/env python
"""Record a LeRobotDataset v3 with HFRVLA precomputed features.

Reads HuggingFaceVLA/libero, rolls a frozen SmolVLA + DINOv3 once over each
episode, and writes a new LeRobotDataset where every frame carries:

    observation.images.image                  (LIBERO third-person)
    observation.images.image2                 (LIBERO wrist)
    observation.state, action, task           (standard)
    observation.extra.z_goal                  (SmolVLA text pool)
    observation.extra.z_phase                 (SmolVLA expert pool)
    observation.extra.a_base                  (SmolVLA-predicted base action)
    observation.extra.k_idx_norm              (chunk position in [0, 1])
    observation.extra.dino_patches            (DINOv3 patches)
    observation.extra.contact_label           (heuristic; 0 by default)

After recording, ``lerobot-train --dataset.repo_id=<out_repo_id>
--dataset.root=<out_root>`` works natively.

Example:
    python scripts/record_hfrvla_libero.py \\
        --src-repo-id HuggingFaceVLA/libero \\
        --out-repo-id HFRVLA_libero_v1 \\
        --out-root checkpoints/HFRVLA_libero_v1 \\
        --smolvla lerobot/smolvla_base \\
        --dinov3-repo checkpoints/dinov3_src \\
        --dinov3-weights checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \\
        --max-episodes 5            # smoke; omit for full run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

POLICY_SRC = Path(__file__).resolve().parents[1] / "policy" / "lerobot_policy_hfrvla" / "src"
if POLICY_SRC.exists():
    sys.path.insert(0, str(POLICY_SRC))

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_STATE

from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig
from lerobot_policy_hfrvla.modeling_hfrvla import HFRVLAPolicy
from lerobot_policy_hfrvla.dinov3_backbone import DINOv3Backbone
from lerobot_policy_hfrvla.processor_hfrvla import normalize_for_dinov3


# ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src-repo-id", default="HuggingFaceVLA/libero")
    p.add_argument("--out-repo-id", default="HFRVLA_libero_v1",
                   help="Identifier baked into the new dataset's metadata.")
    p.add_argument("--out-root", type=Path, required=True,
                   help="Local directory for the new dataset.")
    p.add_argument("--smolvla", default="lerobot/smolvla_base")

    p.add_argument("--dinov3-repo", type=str, required=True)
    p.add_argument("--dinov3-weights", type=str, required=True)
    p.add_argument("--dinov3-arch", default="dinov3_vits16")

    p.add_argument("--wrist-key", default="observation.images.image2")
    p.add_argument("--state-key", default=OBS_STATE)
    p.add_argument("--action-key", default=ACTION)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--fps", type=int, default=10,
                   help="Must match the source dataset's fps.")
    p.add_argument("--dino-dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────
def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _wrist_for_dino(wrist_raw: torch.Tensor) -> torch.Tensor:
    """Normalize a wrist image tensor for DINOv3. Returns (1, 3, H, W)."""
    x = wrist_raw.float()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.shape[-1] == 3 and x.shape[1] != 3:
        x = x.permute(0, 3, 1, 2)
    if x.max() > 1.5:
        x = x / 255.0
    return normalize_for_dinov3(x)


# ────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[record] loading source dataset: {args.src_repo_id}", flush=True)
    src = LeRobotDataset(args.src_repo_id)
    n_total = int(src.num_episodes)
    n_to_record = min(args.max_episodes, n_total) if args.max_episodes else n_total
    print(f"[record] source has {n_total} episodes; recording {n_to_record}", flush=True)

    # ── Build HFRVLA policy (gives us SmolVLA + DINOv3 in one place) ──
    print(f"[record] building HFRVLAPolicy from {args.smolvla}", flush=True)
    config = HFRVLAConfig.from_smolvla(
        args.smolvla,
        dinov3_local_repo=args.dinov3_repo,
        dinov3_local_weights=args.dinov3_weights,
        dinov3_arch=args.dinov3_arch,
    )
    config.input_features = {
        "observation.images.image":  PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        OBS_STATE:                   PolicyFeature(type=FeatureType.STATE,  shape=(8,)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    policy = HFRVLAPolicy.from_pretrained(args.smolvla, config=config).to(device).eval()

    # Capture per-policy dims for declaring the LeRobot features.
    text_hidden   = int(policy.model.vlm_with_expert.config.text_config.hidden_size)
    expert_hidden = int(policy.model.vlm_with_expert.expert_hidden_size)
    dino_dim      = int(config.dinov3_feature_dim)
    dino_npatch   = int(config.dinov3_num_patches)
    n_action_steps = int(config.n_action_steps)

    # Standalone DINOv3 backbone for the wrist-image patch precompute.
    dino = DINOv3Backbone(
        model_id=config.dinov3_model_id,
        local_repo=config.dinov3_local_repo,
        local_weights=config.dinov3_local_weights,
        arch=config.dinov3_arch,
        frozen=True,
    ).to(device).eval()

    # SmolVLA pre-processor (tokenizes 'task').
    pre_processor, _ = make_smolvla_pre_post_processors(
        config, dataset_stats=getattr(src.meta, "stats", None)
    )

    # ── Declare features for the new LeRobotDataset ──
    dino_dtype = args.dino_dtype  # "float32" or "float16"
    features = {
        "observation.images.image":  {"dtype": "image",  "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "observation.images.image2": {"dtype": "image",  "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "observation.state":         {"dtype": "float32","shape": (8,),          "names": ["state"]},
        "action":                    {"dtype": "float32","shape": (7,),          "names": ["actions"]},
        "observation.extra.z_goal":        {"dtype": "float32", "shape": (text_hidden,),   "names": None},
        "observation.extra.z_phase":       {"dtype": "float32", "shape": (expert_hidden,), "names": None},
        "observation.extra.a_base":        {"dtype": "float32", "shape": (7,),             "names": None},
        "observation.extra.k_idx_norm":    {"dtype": "float32", "shape": (1,),             "names": None},
        "observation.extra.dino_patches":  {"dtype": dino_dtype,"shape": (dino_npatch, dino_dim), "names": None},
        "observation.extra.contact_label": {"dtype": "float32", "shape": (1,),             "names": None},
    }
    dst = LeRobotDataset.create(
        repo_id=args.out_repo_id,
        fps=args.fps,
        features=features,
        root=str(args.out_root),
        use_videos=True,
        # Avoid sharded video encode complexity for the first impl pass.
        streaming_encoding=False,
        batch_encoding_size=1,
    )
    print(f"[record] dst created at {dst.root}", flush=True)

    # ── Episode loop ──
    for ep_idx in range(n_to_record):
        ep_meta = src.meta.episodes[ep_idx]
        ep_from = int(ep_meta["dataset_from_index"])
        ep_to   = int(ep_meta["dataset_to_index"])
        ep_task = (
            ep_meta["tasks"][0]
            if isinstance(ep_meta.get("tasks"), list) and ep_meta["tasks"]
            else "do the task"
        )

        policy.reset()
        cached_zgoal: torch.Tensor | None = None
        cached_zphase: torch.Tensor | None = None
        chunk_consumed = 0

        for t in range(ep_from, ep_to):
            sample = src[t]
            sample_with_task = dict(sample)
            sample_with_task.setdefault("task", ep_task)
            try:
                batch = pre_processor(sample_with_task)
            except Exception:
                batch = {
                    k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v)
                    for k, v in sample_with_task.items()
                }
            if not isinstance(batch, dict):
                batch = dict(batch)
            batch = _to_device(batch, device)

            # SmolVLA: refresh chunk if queue empty.
            batch = policy._prepare_batch(batch)
            policy._queues = populate_queues(policy._queues, batch, exclude_keys=[ACTION])
            if len(policy._queues[ACTION]) == 0:
                policy._clear_hook_cache()
                with torch.no_grad():
                    actions = policy._get_action_chunk(batch)
                policy._queues[ACTION].extend(actions.transpose(0, 1)[:n_action_steps])
                cached_zgoal = policy._zgoal_cache
                cached_zphase = policy._zphase_cache
                chunk_consumed = 0

            a_base = policy._queues[ACTION].popleft()
            chunk_consumed += 1
            k_idx = chunk_consumed - 1
            k_norm = k_idx / max(1, n_action_steps - 1)

            # DINOv3 over wrist (single frame).
            wrist_for_dino = _wrist_for_dino(sample[args.wrist_key]).to(device)
            with torch.no_grad():
                dino_patches = dino(wrist_for_dino).squeeze(0).cpu()
            if dino_dtype == "float16":
                dino_patches = dino_patches.half()

            # Source images: LeRobotDataset.create expects HWC uint8 for image
            # features; convert from the cached CHW tensor.
            def _img_for_write(t_chw: torch.Tensor) -> torch.Tensor:
                x = t_chw
                if x.dim() == 4:
                    x = x.squeeze(0)
                if x.shape[0] == 3:
                    x = x.permute(1, 2, 0)
                x = x.float()
                if x.max() <= 1.5:
                    x = x * 255.0
                return x.clamp(0, 255).to(torch.uint8).cpu()

            frame = {
                "observation.images.image":   _img_for_write(sample["observation.images.image"]),
                "observation.images.image2":  _img_for_write(sample[args.wrist_key]),
                "observation.state":          sample[args.state_key].cpu().float(),
                "action":                     sample[args.action_key].cpu().float(),
                "observation.extra.z_goal":   (cached_zgoal.squeeze(0).cpu().float()
                                                if cached_zgoal is not None
                                                else torch.zeros(text_hidden, dtype=torch.float32)),
                "observation.extra.z_phase":  (cached_zphase.squeeze(0).cpu().float()
                                                if cached_zphase is not None
                                                else torch.zeros(expert_hidden, dtype=torch.float32)),
                "observation.extra.a_base":   a_base.squeeze(0).cpu().float(),
                "observation.extra.k_idx_norm":   torch.tensor([k_norm], dtype=torch.float32),
                "observation.extra.dino_patches": dino_patches,
                "observation.extra.contact_label": torch.tensor([0.0], dtype=torch.float32),
                "task": ep_task,
            }
            dst.add_frame(frame)

        dst.save_episode()
        print(f"[record]   ep {ep_idx:>4d}  T={ep_to - ep_from:>4d}  saved", flush=True)

    dst.finalize()
    print(f"[record] done. dataset at {dst.root}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.2: Syntax check**

```bash
~/Robotic_infra/lerobot/.venv/bin/python -c "
import ast
ast.parse(open('scripts/record_hfrvla_libero.py').read())
print('syntax OK')
"
```

Expected: `syntax OK`.

- [ ] **Step 3.3: Commit**

```bash
git add scripts/record_hfrvla_libero.py
git commit -m "feat(scripts): add record_hfrvla_libero.py for LeRobotDataset v3 output"
```

---

## Task 4: Write the training wrapper `train_via_lerobot.py`

**Files:**
- Create: `scripts/train_via_lerobot.py`

**Rationale:** Lerobot-train doesn't know about HFRVLA's curriculum. This thin wrapper monkey-patches `lerobot.scripts.lerobot_train.update_policy` so that `policy.set_training_step(step)` is called before every optimizer step, and the optimizer's LR is dropped once when stage 2 starts.

- [ ] **Step 4.1: Write the wrapper**

Create `scripts/train_via_lerobot.py`:

```python
#!/usr/bin/env python
"""Wrapper that injects HFRVLA curriculum into lerobot-train.

Strategy
--------
Lerobot-train's main loop calls ``update_policy(...)`` once per optimizer
step. We monkey-patch that function so it:
  1. Unwraps the policy (in case it's been DDP-wrapped),
  2. Invokes ``policy.set_training_step(step)`` for curriculum control,
  3. If stage 2 was just entered, scales every param_group's LR by 0.1,
  4. Delegates to the original ``update_policy``.

Usage
-----
    python scripts/train_via_lerobot.py \\
        --dataset.repo_id=HFRVLA_libero_v1 \\
        --dataset.root=checkpoints/HFRVLA_libero_v1 \\
        --policy.type=hfrvla \\
        --policy.curriculum_warmup_steps=1000 \\
        --policy.curriculum_joint_steps=49000 \\
        --policy.curriculum_refine_steps=10000 \\
        --batch_size=128 --num_workers=8 --steps=60000 \\
        --output_dir=checkpoints/hfrvla_run02 \\
        --wandb.enable=true --wandb.project=hfrvla
"""

from __future__ import annotations

import sys

import lerobot.scripts.lerobot_train as lt

# Module-level step counter — lerobot-train's loop is single-threaded.
_STEP = {"value": 0}
_ORIGINAL_UPDATE_POLICY = lt.update_policy


def _unwrap(policy):
    """Strip accelerator wrapping (DDP/DataParallel) if present."""
    inner = getattr(policy, "module", policy)
    return inner


def _patched_update_policy(train_tracker, policy, batch, optimizer, *args, **kwargs):
    step = _STEP["value"]
    raw = _unwrap(policy)
    if hasattr(raw, "set_training_step"):
        raw.set_training_step(step)
        if getattr(raw, "consume_refine_lr_signal", None) and raw.consume_refine_lr_signal():
            for g in optimizer.param_groups:
                g["lr"] = g["lr"] * 0.1
            print(f"[curriculum] step {step}: entered Stage 2 — LR × 0.1", flush=True)
    result = _ORIGINAL_UPDATE_POLICY(train_tracker, policy, batch, optimizer, *args, **kwargs)
    _STEP["value"] += 1
    return result


def main() -> None:
    lt.update_policy = _patched_update_policy
    # lerobot-train's CLI entry is lt.main.
    lt.main()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Syntax check + import check**

```bash
~/Robotic_infra/lerobot/.venv/bin/python -c "
import ast
ast.parse(open('scripts/train_via_lerobot.py').read())
print('syntax OK')
import lerobot.scripts.lerobot_train as lt
assert hasattr(lt, 'update_policy'), 'expected lt.update_policy'
assert hasattr(lt, 'main'),          'expected lt.main'
print('lerobot-train symbols OK')
"
```

Expected:
```
syntax OK
lerobot-train symbols OK
```

If `lt.main` is missing, look for the actual entry function in `lerobot/scripts/lerobot_train.py` (likely `train_main` or similar) and update the wrapper accordingly.

- [ ] **Step 4.3: Commit**

```bash
git add scripts/train_via_lerobot.py
git commit -m "feat(scripts): add train_via_lerobot.py curriculum wrapper"
```

---

## Task 5: Run G1 — smoke recording (5 episodes)

**Goal:** Verify `record_hfrvla_libero.py` produces a valid LeRobotDataset where every extra has the right shape and dtype.

- [ ] **Step 5.1: Run smoke recording**

```bash
cd ~/Patrick/VLA_research/Hierachical_fast_reactive
~/Robotic_infra/lerobot/.venv/bin/python scripts/record_hfrvla_libero.py \
    --src-repo-id HuggingFaceVLA/libero \
    --out-repo-id HFRVLA_libero_smoke \
    --out-root checkpoints/HFRVLA_libero_smoke \
    --smolvla lerobot/smolvla_base \
    --dinov3-repo checkpoints/dinov3_src \
    --dinov3-weights checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --max-episodes 5
```

Expected log tail:
```
[record]   ep    4  T=...  saved
[record] done. dataset at .../HFRVLA_libero_smoke
```

If it OOMs on GPU, retry with `--device cpu` (recording will be much slower; for 5 episodes it's still acceptable, ~10 min).

- [ ] **Step 5.2: Inspect the produced dataset**

```bash
~/Robotic_infra/lerobot/.venv/bin/python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('HFRVLA_libero_smoke', root='checkpoints/HFRVLA_libero_smoke')
print('num_episodes:', ds.num_episodes)
print('num_frames:',   ds.num_frames)
print('fps:',          ds.fps)
sample = ds[0]
print('keys:', sorted(sample.keys()))
for k in ['observation.extra.z_goal',
          'observation.extra.z_phase',
          'observation.extra.a_base',
          'observation.extra.k_idx_norm',
          'observation.extra.dino_patches',
          'observation.extra.contact_label']:
    v = sample[k]
    print(f'  {k}: shape={tuple(v.shape)} dtype={v.dtype}')
"
```

Expected (text_hidden / expert_hidden values depend on SmolVLA base; usually 960):
```
num_episodes: 5
num_frames: <100..300 depending on episodes>
fps: 10
keys: [..., 'observation.extra.a_base', 'observation.extra.contact_label', 'observation.extra.dino_patches', 'observation.extra.k_idx_norm', 'observation.extra.z_goal', 'observation.extra.z_phase', ...]
  observation.extra.z_goal: shape=(960,) dtype=torch.float32
  observation.extra.z_phase: shape=(960,) dtype=torch.float32
  observation.extra.a_base: shape=(7,) dtype=torch.float32
  observation.extra.k_idx_norm: shape=(1,) dtype=torch.float32
  observation.extra.dino_patches: shape=(196, 384) dtype=torch.float32
  observation.extra.contact_label: shape=(1,) dtype=torch.float32
```

If any extra is missing or has wrong shape, **stop and fix Task 3**.

- [ ] **Step 5.3: Verify delta_timestamps windowing on the smoke dataset**

```bash
~/Robotic_infra/lerobot/.venv/bin/python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
delta = {f'observation.extra.{k}': [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0]
         for k in ['z_goal', 'z_phase', 'a_base', 'k_idx_norm', 'dino_patches', 'contact_label']}
delta['observation.state'] = [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0]
delta['action']            = [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0]
ds = LeRobotDataset('HFRVLA_libero_smoke', root='checkpoints/HFRVLA_libero_smoke',
                    delta_timestamps=delta)
sample = ds[10]
print('windowed shapes:')
for k in delta:
    v = sample[k]
    print(f'  {k}: shape={tuple(v.shape)}')
"
```

Expected: every key has a leading time dim of size 8, e.g. `observation.extra.z_goal: shape=(8, 960)`.

If the windowing fails (e.g. \"key not found in features\" or shapes don't expand), this confirms the `observation.extra.*` naming doesn't get picked up by lerobot's delta mechanism. In that case fall back to naming the extras `observation.z_goal`, `observation.z_phase`, etc. (no `extra.` infix) — update Task 3's feature dict, Task 2's `forward()` key names, and Task 5 accordingly. Re-run from Step 5.1.

- [ ] **Step 5.4: G1 gate decision**

If steps 5.2 and 5.3 pass, mark Task 5 done. Otherwise iterate on Task 3 until both pass.

- [ ] **Step 5.5: Commit (smoke artefacts ignored)**

Add the smoke dataset to .gitignore:

```bash
echo 'checkpoints/HFRVLA_libero_smoke/' >> .gitignore
echo 'checkpoints/HFRVLA_libero_v1/'    >> .gitignore
git add .gitignore
git commit -m "chore(.gitignore): exclude HFRVLA recorded datasets"
```

---

## Task 6: Run G2 — smoke training (100 steps)

**Goal:** Run `train_via_lerobot.py` for 100 steps on the smoke dataset; verify all three stages are entered, losses are sane, and a checkpoint is written.

- [ ] **Step 6.1: Run smoke training**

```bash
~/Robotic_infra/lerobot/.venv/bin/python scripts/train_via_lerobot.py \
    --dataset.repo_id=HFRVLA_libero_smoke \
    --dataset.root=checkpoints/HFRVLA_libero_smoke \
    --policy.type=hfrvla \
    --policy.curriculum_warmup_steps=10 \
    --policy.curriculum_joint_steps=80 \
    --policy.curriculum_refine_steps=10 \
    --batch_size=4 \
    --num_workers=2 \
    --steps=100 \
    --save_freq=100 \
    --log_freq=10 \
    --output_dir=outputs/train_smoke \
    --wandb.enable=false 2>&1 | tee outputs/train_smoke.log
```

Watch for these specific log lines:
- Stage 1 entry: anywhere around step 10 (no special print yet — see Step 6.2)
- Stage 2 entry: `[curriculum] step 90: entered Stage 2 — LR × 0.1`
- A checkpoint dir at `outputs/train_smoke/checkpoints/000100/pretrained_model/`

- [ ] **Step 6.2: Verify checkpoint shape**

```bash
ls outputs/train_smoke/checkpoints/000100/pretrained_model/
```

Expected contents:
```
config.json
model.safetensors
policy_preprocessor.json
policy_postprocessor.json
train_config.json
```

If `model.safetensors` is missing, training crashed before saving — read `outputs/train_smoke.log` and fix.

- [ ] **Step 6.3: Verify final loss is plausible**

Inspect the last log lines for the typical fields `loss / delta / gate / contact`. The training tracker uses lerobot's `log_dict`; the keys we emit from `_compute_losses` (`delta`, `gate`, `contact`) should appear. If they don't, check that lerobot-train forwards `losses["loss"]` (it should — `update_policy` calls `policy.forward(batch)` and treats the returned dict as the loss dict). Acceptable outcome at step 100: `delta < 1.0`, `contact < 0.1`, no NaNs.

- [ ] **Step 6.4: G2 gate decision**

If checkpoint exists and losses are non-NaN, mark Task 6 done.

- [ ] **Step 6.5: Commit log**

```bash
mkdir -p docs/runs
cp outputs/train_smoke.log docs/runs/smoke_train_100step.log
git add docs/runs/smoke_train_100step.log
git commit -m "chore(runs): capture G2 smoke training log"
```

---

## Task 7: Run G3 — smoke eval (1 LIBERO episode)

**Goal:** Confirm the lerobot-train checkpoint is directly loadable by `lerobot-eval`.

- [ ] **Step 7.1: Run smoke eval**

```bash
~/Robotic_infra/lerobot/.venv/bin/lerobot-eval \
    --policy.path=outputs/train_smoke/checkpoints/000100/pretrained_model \
    --env.type=libero \
    --env.task=libero_spatial \
    --env.task_ids='[0]' \
    --eval.n_episodes=1 \
    --eval.batch_size=1 \
    --output_dir=outputs/eval_smoke \
    --seed=42 2>&1 | tee outputs/eval_smoke.log
```

Expected outcome:
- Process exits 0.
- `outputs/eval_smoke/eval_info.json` exists.
- `pc_success = 0.0` is acceptable (only 100 training steps); the deliverable is "no crash".

- [ ] **Step 7.2: G3 gate decision**

If the script exits 0 and writes `eval_info.json`, mark Task 7 done.

- [ ] **Step 7.3: Commit log**

```bash
cp outputs/eval_smoke.log docs/runs/smoke_eval_g3.log
git add docs/runs/smoke_eval_g3.log
git commit -m "chore(runs): capture G3 smoke eval log"
```

---

## Task 8: Move legacy files

**Files:**
- Move: `scripts/train_hfrvla.py` → `scripts/legacy/train_hfrvla.py`
- Move: `scripts/precompute_libero.py` → `scripts/legacy/precompute_libero.py`
- Move: `scripts/precompute_libero_plan.py` → `scripts/legacy/precompute_libero_plan.py`
- Move: `scripts/precompute_libero_parallel.sh` → `scripts/legacy/precompute_libero_parallel.sh`
- Move: `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/data.py` → `scripts/legacy/data.py`
- Create: `scripts/legacy/README.md`

**Rationale:** Tasks 1–7 supersede these. Keep them for archaeology / fallback, not as live code. The plugin's `data.py` is no longer imported by anything — moving it removes confusion.

- [ ] **Step 8.1: Verify nothing imports the to-be-moved files**

```bash
grep -RIn "from lerobot_policy_hfrvla.data\|import.*data\b" policy/ scripts/ tests/ 2>/dev/null | grep -v legacy
grep -RIn "precompute_libero_plan\|train_hfrvla" policy/ scripts/ tests/ docs/ 2>/dev/null | grep -v legacy
```

Expected: empty output. If anything matches outside `legacy/`, **fix the import first** (likely by removing the import — these symbols are no longer needed).

- [ ] **Step 8.2: Move files**

```bash
mkdir -p scripts/legacy
git mv scripts/train_hfrvla.py             scripts/legacy/train_hfrvla.py
git mv scripts/precompute_libero.py        scripts/legacy/precompute_libero.py
git mv scripts/precompute_libero_plan.py   scripts/legacy/precompute_libero_plan.py
git mv scripts/precompute_libero_parallel.sh scripts/legacy/precompute_libero_parallel.sh
git mv policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/data.py scripts/legacy/data.py
```

- [ ] **Step 8.3: Add legacy README**

Create `scripts/legacy/README.md`:

```markdown
# Legacy scripts (pre-lerobot-train migration)

These files were the working pipeline before the 2026-05-15 migration to
`lerobot-train`. They are kept for reference and as a fallback, but are
**no longer wired into the active workflow**.

| File | Original purpose | Replaced by |
|---|---|---|
| `train_hfrvla.py` | Custom curriculum trainer reading `.pt` files. | `scripts/train_via_lerobot.py` + `lerobot-train`. |
| `precompute_libero.py` | Roll SmolVLA + DINOv3 once, save `.pt` per episode. | `scripts/record_hfrvla_libero.py` (writes a LeRobotDataset v3 instead). |
| `precompute_libero_plan.py` | Sharding helpers (`--ep-from`/`--ep-to`/`--skip-existing`). | (not yet re-implemented; single-process recording for now.) |
| `precompute_libero_parallel.sh` | Multi-shard driver. | (same.) |
| `data.py` | `HFRVLADataset` (loaded `.pt` files). | LeRobotDataset's standard dataloader. |

Spec: `docs/superpowers/specs/2026-05-15-hfrvla-lerobot-native-design.md`
Plan: `docs/superpowers/plans/2026-05-15-hfrvla-lerobot-native.md`
```

- [ ] **Step 8.4: Verify imports & tests still pass**

```bash
~/Robotic_infra/lerobot/.venv/bin/python -c "
import lerobot_policy_hfrvla
from lerobot_policy_hfrvla.modeling_hfrvla import HFRVLAPolicy
from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig
print('plugin imports OK')
"
~/Robotic_infra/lerobot/.venv/bin/python -m pytest tests/ -v
```

Expected: plugin imports OK; all tests pass.

- [ ] **Step 8.5: Commit**

```bash
git add scripts/legacy/ policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/
git commit -m "refactor: move legacy precompute / training scripts to scripts/legacy/"
```

---

## Task 9: Rewrite `docs/training.md`

**Files:**
- Rewrite: `docs/training.md`

**Rationale:** The existing `docs/training.md` documents the legacy custom-script flow. Replace with the lerobot-train flow.

- [ ] **Step 9.1: Rewrite the doc**

Replace the entire contents of `docs/training.md` with:

````markdown
# HFRVLA training & eval (lerobot-native)

> Use `lerobot-train` for training and `lerobot-eval` for evaluation.
> Recording uses a small wrapper script that writes a standard LeRobotDataset v3.

Active scripts:
- `scripts/record_hfrvla_libero.py` — one-shot recording
- `scripts/train_via_lerobot.py`    — wrapper around lerobot-train (adds curriculum)
- `lerobot-eval` (CLI, no wrapper needed)

Legacy: `scripts/legacy/` (`README.md` inside explains).

Spec / plan:
- `docs/superpowers/specs/2026-05-15-hfrvla-lerobot-native-design.md`
- `docs/superpowers/plans/2026-05-15-hfrvla-lerobot-native.md`

---

## 0. Prerequisites

| Thing | Where |
|---|---|
| Python venv | `~/Robotic_infra/lerobot/.venv` |
| HFRVLA plugin | `policy/lerobot_policy_hfrvla/` (installed editable into the venv) |
| DINOv3 source | `checkpoints/dinov3_src/` |
| DINOv3 weights | `checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth` |
| LIBERO data | `HuggingFaceVLA/libero` (first run downloads ~50 GB) |
| Wandb login | `~/Robotic_infra/lerobot/.venv/bin/wandb login` (once) |

Working dir: `~/Patrick/VLA_research/Hierachical_fast_reactive`.

---

## 1. Record the dataset (one-time, ~10–12 hr)

```bash
cd ~/Patrick/VLA_research/Hierachical_fast_reactive

~/Robotic_infra/lerobot/.venv/bin/python scripts/record_hfrvla_libero.py \
    --src-repo-id HuggingFaceVLA/libero \
    --out-repo-id HFRVLA_libero_v1 \
    --out-root checkpoints/HFRVLA_libero_v1 \
    --smolvla lerobot/smolvla_base \
    --dinov3-repo checkpoints/dinov3_src \
    --dinov3-weights checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

Smoke variant (5 episodes, ~5 min):
```bash
... --max-episodes 5 --out-repo-id HFRVLA_libero_smoke --out-root checkpoints/HFRVLA_libero_smoke
```

### `record_hfrvla_libero.py` flags

| Flag | Default | Description |
|---|---|---|
| `--src-repo-id` | `HuggingFaceVLA/libero` | Source LeRobot dataset. |
| `--out-repo-id` | `HFRVLA_libero_v1` | Identifier in the new dataset's metadata. |
| `--out-root` | required | Local directory for the new dataset. |
| `--smolvla` | `lerobot/smolvla_base` | SmolVLA base used for chunk inference. |
| `--dinov3-repo` | required | Local clone of facebookresearch/dinov3. |
| `--dinov3-weights` | required | DINOv3 `.pth` weight file. |
| `--dinov3-arch` | `dinov3_vits16` | torch.hub entry-point name. |
| `--wrist-key` | `observation.images.image2` | Eye-in-hand camera key in source dataset. |
| `--max-episodes` | None | Cap for smoke tests. |
| `--fps` | 10 | Must equal source dataset's fps. |
| `--dino-dtype` | `float32` | Set to `float16` to halve storage. |
| `--device` | `cuda` | |

---

## 2. Train (lerobot-train via wrapper)

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

The wrapper adds curriculum control on top of `lerobot-train`; all other flags are vanilla lerobot-train (see `lerobot-train --help`).

### Curriculum / windowing knobs (live on `HFRVLAConfig`)

| Flag | Default | Description |
|---|---|---|
| `--policy.curriculum_warmup_steps` | 1000 | Stage 0 length (delta only, heads frozen). |
| `--policy.curriculum_joint_steps`  | 49000 | Stage 1 length (all losses active). |
| `--policy.curriculum_refine_steps` | 10000 | Stage 2 length (LR × 0.1 at start). |
| `--policy.seq_len` | 8 | GRU window length; drives `observation_delta_indices`. |

### Output structure

```
checkpoints/hfrvla_run02/
├── checkpoints/
│   ├── 005000/
│   │   ├── pretrained_model/   ← directly loadable by lerobot-eval
│   │   └── training_state/
│   ├── 010000/...
│   └── last/                    (symlink to most recent)
├── train_config.json
└── wandb/
```

---

## 3. Evaluate

Single-task smoke (~15 s/episode):
```bash
~/Robotic_infra/lerobot/.venv/bin/lerobot-eval \
    --policy.path=checkpoints/hfrvla_run02/checkpoints/last/pretrained_model \
    --env.type=libero \
    --env.task=libero_spatial \
    --env.task_ids='[0]' \
    --eval.n_episodes=1 \
    --eval.batch_size=1 \
    --output_dir=outputs/eval_smoke \
    --seed=42
```

Full suite (10 tasks × 20 episodes):
```bash
for SUITE in libero_spatial libero_object libero_goal libero_10 libero_90; do
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

> `--env.num_envs` is NOT a valid LiberoEnv field; use `--eval.batch_size` to control parallelism.

---

## 4. Health checks during training

| Stage | Boundary | Expected behaviour |
|---|---|---|
| 0 (Warmup) | `step ∈ [0, warmup)` | gate / contact heads frozen, λ = 0, `loss == delta`. |
| 1 (Joint) | `step ∈ [warmup, warmup+joint)` | All losses contribute; expect a small spike at boundary, then descent. |
| 2 (Refine) | `step ≥ warmup+joint` | `[curriculum] step <N>: entered Stage 2 — LR × 0.1` printed exactly once. |

If any boundary log is missing, check `scripts/train_via_lerobot.py` — the monkey-patch may have stopped intercepting `update_policy`.

---

## 5. GPU utilization tips

The fast module is < 2 M parameters and the dataset access is what dominates time. If GPU util stays < 30 % after the first few hundred steps, try:

1. Bump `--batch_size` up to 256 (your A5000 fits this easily — model is tiny).
2. Bump `--num_workers` to 12–16.
3. Run with `--policy.use_bf16=true` (if HFRVLAConfig inherits the flag).

Past those, the next lever is rewriting the dataset to store `dino_patches` in float16 (re-record with `--dino-dtype float16`).
````

- [ ] **Step 9.2: Commit**

```bash
git add docs/training.md
git commit -m "docs(training): rewrite for lerobot-native flow"
```

---

## Task 10: Final sanity sweep

- [ ] **Step 10.1: Run all tests**

```bash
~/Robotic_infra/lerobot/.venv/bin/python -m pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 10.2: Re-run G3 smoke from scratch (regression check)**

```bash
~/Robotic_infra/lerobot/.venv/bin/lerobot-eval \
    --policy.path=outputs/train_smoke/checkpoints/000100/pretrained_model \
    --env.type=libero \
    --env.task=libero_spatial \
    --env.task_ids='[0]' \
    --eval.n_episodes=1 \
    --eval.batch_size=1 \
    --output_dir=outputs/eval_smoke_regression \
    --seed=42
```

Expected: exits 0.

- [ ] **Step 10.3: Summary commit + report**

```bash
git log --oneline lerobot-native-migration ^main | head -20
```

Then write a short summary in the chat:
```
Migration done. All 3 gates passed:
  G1 — 5-episode recording → loadable LeRobotDataset ✓
  G2 — 100-step training   → checkpoint at outputs/train_smoke/checkpoints/000100/ ✓
  G3 — 1-episode eval      → eval_info.json written ✓
Branch: lerobot-native-migration (N commits ahead of main).
Next (manual): full 60k training on the real HFRVLA_libero_v1 dataset.
```

---

## Self-Review (filled in by plan author)

**Spec coverage:**
- §2.① local-only: Task 5 + Task 9 explicitly use local `--out-root` / `--dataset.root`; no Hub push step. ✓
- §2.② three-stage curriculum: Tasks 1, 2, 4 implement config + hook + wrapper. ✓
- §2.③ delta_timestamps via `observation.extra.*`: Task 1 adds `observation_delta_indices`; Task 5 step 5.3 verifies it works (and prescribes the fallback if it doesn't). ✓
- §4.1 feature dict: Task 3 declares all 6 extras + standard LIBERO features. ✓
- §4.4 wrapper (Option β): Task 4 builds it. ✓
- §4.5 recording: Task 3. ✓
- §4.6 training invocation: Task 9 documents; Task 6 exercises a 100-step version. ✓
- §4.7 eval invocation: Tasks 7, 9. ✓
- §5 G1/G2/G3: Tasks 5/6/7 directly. ✓
- §6 risk "extra.* prefix not picked up by delta": Task 5 step 5.3 includes the verification + fallback. ✓
- §6 risk "wrapper unwrap_model": Task 4's `_unwrap` handles it. ✓
- §7 preserved (select_action, FastReactive, DINO loading): none of those are touched. ✓
- §8 acceptance: all 7 criteria mapped to tasks. ✓

**Placeholder scan:** No TBD/TODO/handwaves remain. The two open uncertainties (`observation.extra.*` windowing; `lt.main` symbol name) have explicit fallback instructions.

**Type consistency:** All keys are spelled identically across tasks (`observation.extra.{z_goal, z_phase, a_base, k_idx_norm, dino_patches, contact_label}`). Method names match (`set_training_step`, `consume_refine_lr_signal`, `curriculum_warmup_steps`, etc.).
