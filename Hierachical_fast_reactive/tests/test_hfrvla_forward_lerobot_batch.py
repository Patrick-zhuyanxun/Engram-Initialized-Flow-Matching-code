"""Tests for HFRVLAPolicy.forward consuming LeRobot batch keys + curriculum."""

import pytest
import torch
import torch.nn as nn

from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig
from lerobot_policy_hfrvla.fast_reactive import FastReactiveModule


class _DummyDINOv3Backbone(nn.Module):
    """Avoid loading external DINO weights in FastReactiveModule unit tests."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch = pixel_values.shape[0]
        return torch.zeros(batch, 196, 384, device=pixel_values.device)


@pytest.fixture(autouse=True)
def _patch_dino_backbone(monkeypatch):
    monkeypatch.setattr(
        "lerobot_policy_hfrvla.fast_reactive.DINOv3Backbone",
        _DummyDINOv3Backbone,
    )


# We don't load the full HFRVLAPolicy (it pulls SmolVLA from HF Hub). We
# test the bits that don't need SmolVLA:
#   - FastReactiveModule still works with cached dino_patches (no SmolVLA).
#   - Curriculum logic is exercised directly on a stub policy.


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
    proprio = torch.randn(B, T, 8)
    a_base = torch.randn(B, T, 7)
    k_idx_norm = torch.rand(B, T, 1)
    z_goal = torch.randn(B, T, 960)
    z_phase = torch.randn(B, T, 960)

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
    pol.set_training_step(0)  # stage 0
    pol.set_training_step(10)  # stage 1 entry (warmup_steps == 10)
    assert pol.config.loss_lambda_gate == pol._cached_lambda_gate
    assert pol.config.loss_lambda_contact == pol._cached_lambda_contact
    assert all(p.requires_grad for p in pol.fast.gate_head.parameters())


def test_curriculum_stage2_marks_refine_lr_pending():
    pol = _StubPolicy()
    pol.set_training_step(0)
    pol.set_training_step(50)  # joint
    pol.set_training_step(90)  # refine_start = 10 + 80 = 90
    assert pol._prev_stage == 2
    # set_training_step must set _refine_lr_pending = True at the boundary
    # so the wrapper script knows to drop LR. (The wrapper consumes & clears
    # the flag.)
    assert pol._refine_lr_pending is True
