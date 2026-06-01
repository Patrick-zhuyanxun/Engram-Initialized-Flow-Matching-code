"""Unit tests for the pre-training HFRVLA data/target contract check."""

import pytest
import torch

from scripts.check_hfrvla_training_contract import (
    REQUIRED_FEATURE_SHAPES,
    TrainingContractError,
    compute_zero_residual_contract,
    validate_feature_shapes,
    validate_windowed_batch_shapes,
)


def _feature(shape, dtype="float32"):
    return {"shape": list(shape), "dtype": dtype}


def test_validate_feature_shapes_rejects_non_libero_action_width():
    features = {
        key: _feature(shape)
        for key, shape in REQUIRED_FEATURE_SHAPES.items()
    }
    features["action"] = _feature((6,))

    with pytest.raises(TrainingContractError, match="action"):
        validate_feature_shapes(features)


def test_zero_residual_contract_matches_action_minus_base_target():
    batch = {
        "action": torch.ones(2, 8, 7),
        "observation.extra.a_base": torch.zeros(2, 8, 7),
        "observation.extra.k_idx_norm": torch.linspace(0, 1, 16).view(2, 8),
        "observation.extra.contact_label": torch.zeros(2, 8, 1),
    }

    report = compute_zero_residual_contract(batch)

    assert report["action_shape"] == (2, 8, 7)
    assert report["a_base_shape"] == (2, 8, 7)
    assert report["target_delta_shape"] == (2, 8, 7)
    assert report["k_idx_norm_shape"] == (2, 8, 1)
    assert report["zero_delta_mse"] == pytest.approx(1.0)


def test_zero_residual_contract_rejects_action_base_shape_mismatch():
    batch = {
        "action": torch.zeros(2, 8, 7),
        "observation.extra.a_base": torch.zeros(2, 8, 6),
        "observation.extra.k_idx_norm": torch.zeros(2, 8, 1),
    }

    with pytest.raises(TrainingContractError, match="a_base"):
        compute_zero_residual_contract(batch)


def test_windowed_shape_validator_accepts_lerobot_scalar_collapse():
    batch = {
        "observation.state": torch.zeros(1, 8, 8),
        "action": torch.zeros(1, 8, 7),
        "observation.extra.a_base": torch.zeros(1, 8, 7),
        "observation.extra.k_idx_norm": torch.zeros(1, 8),
        "observation.extra.z_goal": torch.zeros(1, 8, 960),
        "observation.extra.z_phase": torch.zeros(1, 8, 480),
        "observation.extra.dino_patches": torch.zeros(1, 8, 196, 384),
        "observation.extra.contact_label": torch.zeros(1, 8),
    }

    shapes = validate_windowed_batch_shapes(batch, seq_len=8)

    assert shapes["observation.extra.k_idx_norm"] == (1, 8)
    assert shapes["observation.extra.contact_label"] == (1, 8)
