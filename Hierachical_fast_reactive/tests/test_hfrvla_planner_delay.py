from collections import deque

import pytest
import torch

from lerobot.utils.constants import ACTION
from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig
from lerobot_policy_hfrvla.modeling_hfrvla import HFRVLAPolicy


def _delay_policy(
    *,
    n_action_steps: int = 3,
    delay_steps: int = 2,
    planner_delay_mode: str = "async_timestep",
    async_request_interval_steps: int = 8,
) -> HFRVLAPolicy:
    policy = HFRVLAPolicy.__new__(HFRVLAPolicy)
    policy.config = HFRVLAConfig(
        n_action_steps=n_action_steps,
        chunk_size=max(50, n_action_steps),
        planner_delay_steps=delay_steps,
        planner_delay_fallback="hold_last",
        planner_delay_mode=planner_delay_mode,
        async_request_interval_steps=async_request_interval_steps,
    )
    policy._zgoal_cache = None
    policy._zphase_cache = None
    policy._init_hfrvla_state()
    return policy


def _chunk(start: float, *, length: int = 3) -> torch.Tensor:
    values = [
        torch.full((1, 7), start + idx, dtype=torch.float32)
        for idx in range(length)
    ]
    return torch.stack(values, dim=1)


def test_planner_delay_config_rejects_negative_values():
    with pytest.raises(ValueError, match="planner_delay_steps"):
        HFRVLAConfig(planner_delay_steps=-1)


def test_planner_delay_mode_defaults_to_async_timestep():
    assert HFRVLAConfig().planner_delay_mode == "async_timestep"


def test_async_timestep_activation_starts_from_delay_aligned_chunk_index(monkeypatch):
    policy = _delay_policy(
        n_action_steps=6,
        delay_steps=2,
        planner_delay_mode="async_timestep",
        async_request_interval_steps=3,
    )
    old_zgoal = torch.full((1, 4), 1.0)
    old_zphase = torch.full((1, 4), 2.0)
    new_zgoal = torch.full((1, 4), 3.0)
    new_zphase = torch.full((1, 4), 4.0)
    policy._zgoal_cache = old_zgoal
    policy._zphase_cache = old_zphase
    policy._current_a_base_chunk = _chunk(10.0, length=6)
    policy._queues[ACTION] = deque(
        policy._current_a_base_chunk.transpose(0, 1),
        maxlen=policy.config.n_action_steps,
    )
    policy._async_control_step = 3

    def fake_get_action_chunk(batch):
        policy._zgoal_cache = new_zgoal
        policy._zphase_cache = new_zphase
        return _chunk(20.0, length=8)

    monkeypatch.setattr(policy, "_get_action_chunk", fake_get_action_chunk)

    assert policy._maybe_start_async_timestep_request({"observation.state": torch.zeros(1, 8)})
    assert len(policy._queues[ACTION]) == 6
    assert policy._pending_chunk_start_index == 2
    assert policy._pending_request_step == 3
    assert policy._pending_observation_step == 3
    assert policy._pending_ready_step == 5
    assert torch.equal(policy._zgoal_cache, old_zgoal)
    assert torch.equal(policy._zphase_cache, old_zphase)

    policy._async_control_step = 5
    assert policy._activate_ready_delayed_chunk_if_any()

    a_base, used_fallback = policy._pop_or_fallback_base_action()
    assert not used_fallback
    assert torch.equal(a_base, torch.full((1, 7), 22.0))
    assert policy._current_chunk_step_index() == 2
    assert torch.equal(policy._zgoal_cache, new_zgoal)
    assert torch.equal(policy._zphase_cache, new_zphase)

    stats = policy.get_inference_debug_stats()
    assert stats["async_request_count"] == 1
    assert stats["async_activation_count"] == 1
    assert stats["async_chunk_start_index_last"] == 2
    assert stats["async_dropped_old_queue_steps_last"] == 6


def test_async_timestep_select_action_requests_by_interval_and_replaces_at_ready(monkeypatch):
    policy = _delay_policy(
        n_action_steps=6,
        delay_steps=2,
        planner_delay_mode="async_timestep",
        async_request_interval_steps=3,
    )
    policy.config.inference_disable_fast = True
    chunks = iter([_chunk(10.0, length=8), _chunk(20.0, length=8)])

    def fake_get_action_chunk(batch):
        del batch
        policy._zgoal_cache = torch.ones(1, 4)
        policy._zphase_cache = torch.ones(1, 4)
        return next(chunks)

    monkeypatch.setattr(policy, "eval", lambda: policy)
    monkeypatch.setattr(policy, "_prepare_batch", lambda batch: batch)
    monkeypatch.setattr(policy, "_get_action_chunk", fake_get_action_chunk)

    batch = {"observation.state": torch.zeros(1, 8)}
    outputs = [policy.select_action(batch) for _ in range(6)]

    assert [float(out[0, 0]) for out in outputs] == [10.0, 11.0, 12.0, 13.0, 14.0, 22.0]
    stats = policy.get_inference_debug_stats()
    assert stats["async_request_count"] == 1
    assert stats["async_activation_count"] == 1
    assert stats["async_request_step_last"] == 3
    assert stats["async_ready_step_last"] == 5
    assert stats["async_activate_step_last"] == 5
    assert stats["async_chunk_start_index_last"] == 2
    assert stats["async_dropped_old_queue_steps_last"] == 1


def test_async_timestep_request_interval_and_delay_must_fit_execution_horizon():
    HFRVLAConfig(
        n_action_steps=16,
        chunk_size=50,
        planner_delay_mode="async_timestep",
        async_request_interval_steps=8,
        planner_delay_steps=4,
    )

    with pytest.raises(ValueError, match="async_request_interval_steps \\+ planner_delay_steps"):
        HFRVLAConfig(
            n_action_steps=16,
            chunk_size=50,
            planner_delay_mode="async_timestep",
            async_request_interval_steps=13,
            planner_delay_steps=4,
        )


def test_planner_delay_config_rejects_non_async_modes():
    with pytest.raises(ValueError, match="planner_delay_mode"):
        HFRVLAConfig(planner_delay_mode="rtc")
