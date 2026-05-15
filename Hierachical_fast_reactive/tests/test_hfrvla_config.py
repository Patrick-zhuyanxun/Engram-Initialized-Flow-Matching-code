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
