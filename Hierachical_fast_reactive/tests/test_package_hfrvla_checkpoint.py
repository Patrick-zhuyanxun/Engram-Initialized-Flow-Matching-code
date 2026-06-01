import torch

from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig
from scripts.package_hfrvla_checkpoint import _apply_fast_config_overrides, _extract_fast_state


def test_extract_fast_state_from_lerobot_full_policy_state():
    raw_state = {
        "model.vlm_with_expert.weight": torch.zeros(1),
        "fast.delta_head.weight": torch.ones(1),
        "fast.gate_head.bias": torch.full((1,), 2.0),
    }

    fast_state = _extract_fast_state(raw_state)

    assert set(fast_state) == {"delta_head.weight", "gate_head.bias"}
    assert torch.equal(fast_state["delta_head.weight"], torch.ones(1))


def test_extract_fast_state_from_legacy_fast_state_dict():
    raw_state = {
        "fast_state_dict": {
            "delta_head.weight": torch.ones(1),
        }
    }

    fast_state = _extract_fast_state(raw_state)

    assert set(fast_state) == {"delta_head.weight"}


def test_apply_fast_config_overrides_keeps_inference_relevant_training_knobs():
    cfg = HFRVLAConfig(n_action_steps=1, chunk_size=1, seq_len=8)
    fast_cfg = {
        "n_action_steps": 50,
        "chunk_size": 50,
        "seq_len": 4,
        "delta_max": 0.123,
        "offline_training_mode": True,
        "load_vlm_weights": False,
    }

    _apply_fast_config_overrides(cfg, fast_cfg)

    assert cfg.n_action_steps == 50
    assert cfg.chunk_size == 50
    assert cfg.seq_len == 4
    assert cfg.delta_max == 0.123
    assert cfg.offline_training_mode is False
    assert cfg.load_vlm_weights is True


def test_apply_fast_config_overrides_accepts_deprecated_a2c2_aliases():
    cfg = HFRVLAConfig()
    fast_cfg = {
        "residual_merge_mode": "a2c2",
        "a2c2_alpha": 0.75,
        "a2c2_use_latent_context": False,
    }

    _apply_fast_config_overrides(cfg, fast_cfg)

    assert cfg.residual_merge_mode == "fast_wrist"
    assert cfg.fast_residual_alpha == 0.75
    assert cfg.fast_residual_use_latent_context is False
    assert cfg.a2c2_alpha == 0.75
    assert cfg.a2c2_use_latent_context is False
