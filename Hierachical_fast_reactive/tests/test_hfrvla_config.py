"""Tests for HFRVLAConfig curriculum + windowing additions."""

from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig
from lerobot.configs.types import FeatureType, PolicyFeature


def _bare_config() -> HFRVLAConfig:
    # Bypass SmolVLA hub download: build a config from defaults only.
    cfg = HFRVLAConfig()
    return cfg


def test_curriculum_defaults_present():
    cfg = _bare_config()
    assert cfg.curriculum_warmup_steps == 1000
    assert cfg.curriculum_joint_steps == 49000
    assert cfg.curriculum_refine_steps == 10000
    assert cfg.push_to_hub is False


def test_conservative_objective_defaults_present():
    cfg = _bare_config()
    assert cfg.loss_delta_target_clip is True
    # Stage A v2 (calibrated 2026-05-22): margin in per-frame squared-L2
    # units; LIBERO empirical err_before has median ~2.4, so v1 default 0.05
    # was ~50x too small.
    assert cfg.gate_improvement_margin == 0.5
    assert cfg.loss_lambda_final == 1.0
    assert cfg.loss_lambda_preserve == 0.5
    # Stage A: real rate term (was 0.02, a token regularizer).
    assert cfg.loss_lambda_gate_prior == 0.10
    # Stage A v2: thresh in same units. v1 0.01 caught 0% of frames.
    assert cfg.loss_lambda_preserve_zero == 1.0
    assert cfg.err_preserve_thresh == 0.5


def test_curriculum_total_helper():
    cfg = _bare_config()
    assert cfg.curriculum_total_steps() == 60000


def test_observation_delta_indices_default_seq_len_8():
    cfg = _bare_config()
    # 8 consecutive steps ending at the current frame.
    assert cfg.observation_delta_indices == list(range(-7, 1))
    assert cfg.action_delta_indices == list(range(-7, 1))
    assert len(cfg.observation_delta_indices) == 8


def test_dinov3_defaults_point_to_local_assets():
    cfg = _bare_config()
    assert cfg.dinov3_local_repo == "checkpoints/dinov3_src"
    assert cfg.dinov3_local_weights == (
        "checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    )


def test_validate_features_excludes_precomputed_extras_from_normalizer():
    cfg = _bare_config()
    cfg.input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        "observation.extra.z_goal": PolicyFeature(type=FeatureType.STATE, shape=(960,)),
    }

    cfg.validate_features()

    assert "observation.images.image" in cfg.input_features
    assert "observation.state" in cfg.input_features
    assert "observation.extra.z_goal" not in cfg.input_features


def test_offline_training_validate_features_drops_images_from_normalizer():
    cfg = HFRVLAConfig(offline_training_mode=True)
    cfg.input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        "observation.extra.z_goal": PolicyFeature(type=FeatureType.STATE, shape=(960,)),
    }

    cfg.validate_features()

    assert cfg.input_features == {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,))
    }
