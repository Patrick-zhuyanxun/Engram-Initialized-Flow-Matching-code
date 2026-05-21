"""Tests for HFRVLAPolicy.forward consuming LeRobot batch keys + curriculum."""

import pytest
import torch
import torch.nn as nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig
from lerobot_policy_hfrvla.fast_reactive import FastReactiveModule, FastReactiveOutput


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


def test_fast_module_accepts_float16_cached_features_with_float32_weights():
    fast, cfg = _make_fast_module(seq_len=4)
    B, T = 2, 4

    out = fast(
        wrist_rgb=None,
        proprio=torch.randn(B, T, 8),
        a_base_k=torch.randn(B, T, 7),
        k_idx_norm=torch.rand(B, T, 1),
        z_goal=torch.randn(B, T, 960).half(),
        z_phase=torch.randn(B, T, 960).half(),
        dino_patches=torch.randn(B, T, 196, 384).half(),
    )

    assert out.delta_a.shape == (B, T, 7)
    assert out.delta_a.dtype == torch.float32


def test_fast_module_encodes_sequence_in_one_batched_call(monkeypatch):
    fast, cfg = _make_fast_module(seq_len=8)
    B, T = 2, 8
    calls = 0
    original_encode_step = fast._encode_step

    def _counting_encode_step(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_encode_step(*args, **kwargs)

    monkeypatch.setattr(fast, "_encode_step", _counting_encode_step)

    fast(
        wrist_rgb=None,
        proprio=torch.randn(B, T, 8),
        a_base_k=torch.randn(B, T, 7),
        k_idx_norm=torch.rand(B, T, 1),
        z_goal=torch.randn(B, T, 960),
        z_phase=torch.randn(B, T, 960),
        dino_patches=torch.randn(B, T, 196, 384),
    )

    assert calls == 1


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
        self._cached_lambda_final = self.config.loss_lambda_final
        self._cached_lambda_preserve = self.config.loss_lambda_preserve
        self._cached_lambda_gate_prior = self.config.loss_lambda_gate_prior
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
    assert pol.config.loss_lambda_final == 0.0
    assert pol.config.loss_lambda_preserve == 0.0
    assert pol.config.loss_lambda_gate_prior == 0.0
    assert all(not p.requires_grad for p in pol.fast.gate_head.parameters())
    if pol.fast.contact_head is not None:
        assert all(not p.requires_grad for p in pol.fast.contact_head.parameters())


def test_curriculum_stage1_unfreezes_heads_and_restores_lambdas():
    pol = _StubPolicy()
    pol.set_training_step(0)  # stage 0
    pol.set_training_step(10)  # stage 1 entry (warmup_steps == 10)
    assert pol.config.loss_lambda_gate == pol._cached_lambda_gate
    assert pol.config.loss_lambda_contact == pol._cached_lambda_contact
    assert pol.config.loss_lambda_final == pol._cached_lambda_final
    assert pol.config.loss_lambda_preserve == pol._cached_lambda_preserve
    assert pol.config.loss_lambda_gate_prior == pol._cached_lambda_gate_prior
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


def test_lerobot_scalar_feature_gets_trailing_dim_for_fast_module():
    from lerobot_policy_hfrvla.modeling_hfrvla import _ensure_trailing_feature_dim

    a_base = torch.zeros(2, 8, 7)
    k_idx_norm = torch.zeros(2, 8)

    expanded = _ensure_trailing_feature_dim(k_idx_norm, reference=a_base)

    assert expanded.shape == (2, 8, 1)


def _policy_shell_for_merge(config: HFRVLAConfig):
    from lerobot_policy_hfrvla.modeling_hfrvla import HFRVLAPolicy

    policy = HFRVLAPolicy.__new__(HFRVLAPolicy)
    policy.config = config
    return policy


def test_compute_losses_clips_delta_target_to_deployment_residual_cap():
    policy = _policy_shell_for_merge(
        HFRVLAConfig(delta_max=0.2, safety_joint_velocity_limit=2.0, control_dt=0.1)
    )
    a_base = torch.zeros(1, 7)
    a_expert = torch.ones(1, 7)
    out = FastReactiveOutput(
        delta_a=torch.full((1, 7), 0.2),
        gate_logit=torch.zeros(1),
        gate=torch.full((1,), 0.5),
        contact_logit=None,
        hidden_state=torch.zeros(1, 1, 256),
    )

    losses = policy._compute_losses(out, a_base, a_expert, contact_label=None)

    assert losses["delta"] == pytest.approx(0.0)


def test_compute_losses_penalizes_final_action_that_hurts_base():
    policy = _policy_shell_for_merge(
        HFRVLAConfig(
            delta_max=0.2,
            safety_joint_velocity_limit=2.0,
            control_dt=0.1,
            loss_lambda_gate=0.0,
            loss_lambda_contact=0.0,
            loss_lambda_final=0.0,
            loss_lambda_preserve=1.0,
            loss_lambda_gate_prior=0.0,
            loss_lambda_preserve_zero=0.0,
        )
    )
    a_base = torch.zeros(1, 7)
    a_expert = torch.zeros(1, 7)
    out = FastReactiveOutput(
        delta_a=torch.full((1, 7), 0.2),
        gate_logit=torch.full((1,), 20.0),
        gate=torch.ones(1),
        contact_logit=None,
        hidden_state=torch.zeros(1, 1, 256),
    )

    losses = policy._compute_losses(out, a_base, a_expert, contact_label=None)

    assert losses["preserve"] > 0
    assert torch.allclose(losses["loss"], losses["delta"] + losses["preserve"])


def test_compute_losses_does_not_preserve_penalize_when_final_matches_base():
    policy = _policy_shell_for_merge(
        HFRVLAConfig(delta_max=0.2, safety_joint_velocity_limit=2.0, control_dt=0.1)
    )
    a_base = torch.zeros(1, 7)
    a_expert = torch.ones(1, 7)
    out = FastReactiveOutput(
        delta_a=torch.full((1, 7), 0.2),
        gate_logit=torch.full((1,), -20.0),
        gate=torch.zeros(1),
        contact_logit=None,
        hidden_state=torch.zeros(1, 1, 256),
    )

    losses = policy._compute_losses(out, a_base, a_expert, contact_label=None)

    assert losses["preserve"] == pytest.approx(0.0)


def test_stage_a_preserve_zero_fires_when_base_matches_expert():
    """Stage A (debate 20260521): L_preserve_zero must penalize residual
    magnitude on preserve-class states (base ≈ expert), driving ‖δ‖² → 0
    there. This is the term that was missing in the original loss."""
    policy = _policy_shell_for_merge(
        HFRVLAConfig(
            delta_max=0.2,
            safety_joint_velocity_limit=2.0,
            control_dt=0.1,
            # Isolate preserve_zero from every other term.
            loss_lambda_gate=0.0,
            loss_lambda_contact=0.0,
            loss_lambda_final=0.0,
            loss_lambda_preserve=0.0,
            loss_lambda_gate_prior=0.0,
            loss_lambda_preserve_zero=1.0,
            err_preserve_thresh=0.01,
        )
    )
    a_base = torch.zeros(1, 7)
    a_expert = torch.zeros(1, 7)  # err_before = 0 < 0.01 → is_preserve = 1
    out = FastReactiveOutput(
        delta_a=torch.full((1, 7), 0.2),
        gate_logit=torch.zeros(1),
        gate=torch.full((1,), 0.5),
        contact_logit=None,
        hidden_state=torch.zeros(1, 1, 256),
    )

    losses = policy._compute_losses(out, a_base, a_expert, contact_label=None)

    # 7 dims × 0.2² = 0.28, full weight, with L_delta = MSE(0.2, 0) = 0.04
    assert losses["preserve_zero"] == pytest.approx(0.28, abs=1e-5)
    expected_total = losses["delta"] + losses["preserve_zero"]
    assert torch.allclose(losses["loss"], expected_total)


def test_stage_a_preserve_zero_is_silent_when_base_far_from_expert():
    """Preserve-zero must NOT fire on correction states (err_before above the
    threshold)."""
    policy = _policy_shell_for_merge(
        HFRVLAConfig(
            delta_max=0.2,
            safety_joint_velocity_limit=2.0,
            control_dt=0.1,
            loss_lambda_preserve_zero=1.0,
            err_preserve_thresh=0.01,
        )
    )
    a_base = torch.zeros(1, 7)
    a_expert = torch.ones(1, 7)  # err_before = 7 ≫ 0.01 → is_preserve = 0
    out = FastReactiveOutput(
        delta_a=torch.full((1, 7), 0.2),
        gate_logit=torch.zeros(1),
        gate=torch.full((1,), 0.5),
        contact_logit=None,
        hidden_state=torch.zeros(1, 1, 256),
    )

    losses = policy._compute_losses(out, a_base, a_expert, contact_label=None)

    assert losses["preserve_zero"] == pytest.approx(0.0, abs=1e-6)


def test_stage_a_gate_detached_from_l_final_gradient():
    """Stage A (debate 20260521): gradient through ``out.gate`` must NOT flow
    from L_final / L_preserve. Gate is trained only by L_gate (BCE) and
    L_gate_prior (rate). Confirms the line-523 ``.detach()`` is in place."""
    policy = _policy_shell_for_merge(
        HFRVLAConfig(
            delta_max=0.2,
            safety_joint_velocity_limit=2.0,
            control_dt=0.1,
            # Isolate L_final + L_preserve as the only active terms.
            loss_lambda_gate=0.0,
            loss_lambda_contact=0.0,
            loss_lambda_final=1.0,
            loss_lambda_preserve=1.0,
            loss_lambda_gate_prior=0.0,
            loss_lambda_preserve_zero=0.0,
        )
    )
    a_base = torch.zeros(1, 7)
    a_expert = torch.ones(1, 7)
    delta_a = torch.full((1, 7), 0.2, requires_grad=True)
    gate_logit = torch.zeros(1, requires_grad=True)
    gate = torch.sigmoid(gate_logit)
    out = FastReactiveOutput(
        delta_a=delta_a,
        gate_logit=gate_logit,
        gate=gate,
        contact_logit=None,
        hidden_state=torch.zeros(1, 1, 256),
    )

    losses = policy._compute_losses(out, a_base, a_expert, contact_label=None)
    losses["loss"].backward()

    # delta_a still receives gradient from L_final/L_preserve via clipped path.
    assert delta_a.grad is not None
    assert delta_a.grad.abs().sum().item() > 0.0
    # gate_logit must NOT receive any gradient because L_gate and
    # L_gate_prior are zero-weighted and the merged-action path detaches gate.
    assert gate_logit.grad is None or gate_logit.grad.abs().sum().item() == 0.0


def test_merge_keeps_base_action_unchanged_when_residual_is_zero():
    policy = _policy_shell_for_merge(
        HFRVLAConfig(delta_max=0.0, safety_joint_velocity_limit=2.0, control_dt=0.1)
    )
    a_base = torch.tensor([[1.5, -1.5, 0.8, -0.8, 0.4, -0.4, 0.1]])
    delta = torch.full_like(a_base, 5.0)
    gate = torch.ones(a_base.shape[0])
    prev_a = torch.zeros_like(a_base)

    out = policy._merge(a_base=a_base, delta_a=delta, gate=gate, prev_a=prev_a)

    assert torch.equal(out, a_base)


def test_merge_applies_velocity_limit_to_residual_not_full_base_action():
    policy = _policy_shell_for_merge(
        HFRVLAConfig(delta_max=10.0, safety_joint_velocity_limit=2.0, control_dt=0.1)
    )
    a_base = torch.tensor([[1.5, -1.5, 0.8, -0.8, 0.4, -0.4, 0.1]])
    delta = torch.full_like(a_base, 5.0)
    gate = torch.ones(a_base.shape[0])
    prev_a = torch.zeros_like(a_base)

    out = policy._merge(a_base=a_base, delta_a=delta, gate=gate, prev_a=prev_a)

    assert torch.allclose(out, a_base + torch.full_like(a_base, 0.2))


def test_extract_wrist_image_matches_recorded_dinov3_preprocessing():
    from lerobot_policy_hfrvla.modeling_hfrvla import HFRVLAPolicy

    policy = HFRVLAPolicy.__new__(HFRVLAPolicy)
    policy.config = HFRVLAConfig(dinov3_image_size=224)
    batch = {
        "observation.images.image2": torch.full((2, 3, 256, 256), 0.5),
    }

    wrist = policy._extract_wrist_image(batch)

    assert wrist.shape == (2, 3, 224, 224)
    expected = torch.tensor([
        (0.5 - 0.485) / 0.229,
        (0.5 - 0.456) / 0.224,
        (0.5 - 0.406) / 0.225,
    ])
    assert torch.allclose(wrist.mean(dim=(-2, -1)), expected.expand(2, 3), atol=1e-6)


def test_offline_training_mode_skips_smolvla_and_trains_from_cached_features(monkeypatch):
    from lerobot_policy_hfrvla.modeling_hfrvla import HFRVLAPolicy

    def _fail_smolvla_init(*args, **kwargs):
        raise AssertionError("offline training must not construct SmolVLA")

    monkeypatch.setattr(
        "lerobot_policy_hfrvla.modeling_hfrvla.SmolVLAPolicy.__init__",
        _fail_smolvla_init,
    )

    cfg = HFRVLAConfig(
        offline_training_mode=True,
        offline_zgoal_dim=960,
        offline_zphase_dim=480,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            "observation.extra.z_goal": PolicyFeature(type=FeatureType.STATE, shape=(960,)),
            "observation.extra.z_phase": PolicyFeature(type=FeatureType.STATE, shape=(480,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
    )

    policy = HFRVLAPolicy(cfg)
    assert getattr(policy, "model", None) is None
    assert policy.fast.action_dim == 7
    assert policy.fast.proprio_dim == 8
    assert policy.fast.zgoal_dim == 960
    assert policy.fast.zphase_dim == 480

    batch = {
        "observation.state": torch.zeros(2, 8, 8),
        "action": torch.zeros(2, 8, 7),
        "observation.extra.a_base": torch.zeros(2, 8, 7),
        "observation.extra.k_idx_norm": torch.zeros(2, 8),
        "observation.extra.z_goal": torch.zeros(2, 8, 960),
        "observation.extra.z_phase": torch.zeros(2, 8, 480),
        "observation.extra.dino_patches": torch.zeros(2, 8, 196, 384),
        "observation.extra.contact_label": torch.zeros(2, 8),
    }

    loss, metrics = policy(batch)

    assert torch.isfinite(loss)
    assert set(metrics) >= {"delta", "gate", "loss"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required to reproduce GRU safetensors storage views")
def test_offline_training_policy_saves_after_cuda_gru_forward(tmp_path):
    from lerobot_policy_hfrvla.modeling_hfrvla import HFRVLAPolicy

    cfg = HFRVLAConfig(
        device="cuda",
        offline_training_mode=True,
        offline_zgoal_dim=960,
        offline_zphase_dim=480,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            "observation.extra.z_goal": PolicyFeature(type=FeatureType.STATE, shape=(960,)),
            "observation.extra.z_phase": PolicyFeature(type=FeatureType.STATE, shape=(480,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
    )
    policy = HFRVLAPolicy(cfg).to("cuda")
    batch = {
        "observation.state": torch.zeros(2, 8, 8, device="cuda"),
        "action": torch.zeros(2, 8, 7, device="cuda"),
        "observation.extra.a_base": torch.zeros(2, 8, 7, device="cuda"),
        "observation.extra.k_idx_norm": torch.zeros(2, 8, device="cuda"),
        "observation.extra.z_goal": torch.zeros(2, 8, 960, device="cuda"),
        "observation.extra.z_phase": torch.zeros(2, 8, 480, device="cuda"),
        "observation.extra.dino_patches": torch.zeros(2, 8, 196, 384, device="cuda"),
        "observation.extra.contact_label": torch.zeros(2, 8, device="cuda"),
    }
    loss, _ = policy(batch)
    loss.backward()

    policy.save_pretrained(tmp_path)

    assert (tmp_path / "model.safetensors").exists()
