"""HFRVLAPolicy — frozen SmolVLA + Fast Reactive Module via composition.

See implementation_spec.md §6.

Design notes:
    * SmolVLA (self.model) is left untouched and frozen.
    * Two forward hooks are registered on the LAST layer of
        - VLM text encoder            → z_goal cache (semantic / task)
        - LM expert transformer       → z_phase cache (action-expert state)
      Both are mean-pooled over the sequence dimension when the hook fires.
    * select_action overrides the SmolVLA queue-based loop to apply the
      fast residual at every step.
    * forward() is used by the training pipeline and assumes batch tensors
      shaped per implementation_spec §5–§6 (single step or sequence).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_STATE

from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig
from lerobot_policy_hfrvla.fast_reactive import FastReactiveModule, FastReactiveOutput
from lerobot_policy_hfrvla.fast_wrist_residual import (
    FastWristChunkResidualModule,
    FastWristResidualModule,
    FastWristResidualOutput,
)
from lerobot_policy_hfrvla.processor_hfrvla import normalize_for_dinov3


def _residual_merge_mode(config: HFRVLAConfig) -> str:
    mode = str(getattr(config, "residual_merge_mode", "gated")).strip().lower()
    return "fast_wrist" if mode == "a2c2" else mode


def _fast_residual_alpha(config: HFRVLAConfig) -> float:
    value = getattr(config, "fast_residual_alpha", None)
    if value is None:
        value = getattr(config, "a2c2_alpha", 1.0)
    return float(value)


def _is_fast_wrist_mode(config: HFRVLAConfig) -> bool:
    return _residual_merge_mode(config) in {"fast_wrist", "fast_wrist_chunk"}


def _set_head_grads(policy_or_stub, *, gate: bool, contact: bool) -> None:
    """Toggle requires_grad on the fast module's auxiliary heads."""
    gate_head = getattr(policy_or_stub.fast, "gate_head", None)
    if gate_head is not None:
        for p in gate_head.parameters():
            p.requires_grad = gate
    contact_head = getattr(policy_or_stub.fast, "contact_head", None)
    if contact_head is not None:
        for p in contact_head.parameters():
            p.requires_grad = contact


def _apply_curriculum_stage(policy_or_stub, step: int) -> None:
    """Mutate lambdas + head gradients based on the configured curriculum.

    Sets ``_refine_lr_pending = True`` when crossing into stage 2 so a caller
    (the wrapper training script) can drop the optimizer LR exactly once at
    that boundary.
    """
    cfg = policy_or_stub.config
    warmup = cfg.curriculum_warmup_steps
    joint = cfg.curriculum_joint_steps
    refine_start = warmup + joint

    if step < warmup:
        stage = 0
    elif step < refine_start:
        stage = 1
    else:
        stage = 2

    prev = getattr(policy_or_stub, "_prev_stage", -1)

    if stage == 0 and prev != 0:
        cfg.loss_lambda_gate = 0.0
        cfg.loss_lambda_contact = 0.0
        cfg.loss_lambda_final = 0.0
        cfg.loss_lambda_preserve = 0.0
        cfg.loss_lambda_gate_prior = 0.0
        _set_head_grads(policy_or_stub, gate=False, contact=False)
    elif stage == 1 and prev != 1:
        cfg.loss_lambda_gate = policy_or_stub._cached_lambda_gate
        cfg.loss_lambda_contact = policy_or_stub._cached_lambda_contact
        cfg.loss_lambda_final = policy_or_stub._cached_lambda_final
        cfg.loss_lambda_preserve = policy_or_stub._cached_lambda_preserve
        cfg.loss_lambda_gate_prior = policy_or_stub._cached_lambda_gate_prior
        _set_head_grads(policy_or_stub, gate=True, contact=True)
    elif stage == 2 and prev != 2:
        cfg.loss_lambda_gate = policy_or_stub._cached_lambda_gate
        cfg.loss_lambda_contact = policy_or_stub._cached_lambda_contact
        cfg.loss_lambda_final = policy_or_stub._cached_lambda_final
        cfg.loss_lambda_preserve = policy_or_stub._cached_lambda_preserve
        cfg.loss_lambda_gate_prior = policy_or_stub._cached_lambda_gate_prior
        _set_head_grads(policy_or_stub, gate=True, contact=True)
        policy_or_stub._refine_lr_pending = True

    policy_or_stub._prev_stage = stage


def _ensure_trailing_feature_dim(value: Tensor, reference: Tensor) -> Tensor:
    """Restore singleton feature dims collapsed by LeRobot scalar features."""
    if value.dim() == reference.dim() - 1:
        return value.unsqueeze(-1)
    return value


def _feature_width(feature, fallback: int) -> int:
    shape = tuple(getattr(feature, "shape", ()) or ())
    if not shape:
        return int(fallback)
    return int(shape[-1])


class HFRVLAPolicy(SmolVLAPolicy):
    """Hierarchical Fast-Reactive VLA policy.

    Inherits everything from :class:`SmolVLAPolicy`. The slow VLA is frozen.
    A trainable :class:`FastReactiveModule` is composed alongside (not nested
    inside) and called at every control step.
    """

    config_class = HFRVLAConfig
    name = "hfrvla"

    def __init__(self, config: HFRVLAConfig, **kwargs):
        self._offline_training_mode = bool(getattr(config, "offline_training_mode", False))
        raw_input_features = dict(config.input_features or {})

        if self._offline_training_mode:
            PreTrainedPolicy.__init__(self, config)
            config.validate_features()
            self.config: HFRVLAConfig = config
            self.rtc_processor = None
            self.model = None

            self._zgoal_cache: Optional[Tensor] = None
            self._zphase_cache: Optional[Tensor] = None
            self._zgoal_dim: Optional[int] = None
            self._zphase_dim: Optional[int] = None
            self._hook_handles: list = []

            text_hidden = _feature_width(
                raw_input_features.get("observation.extra.z_goal"),
                self.config.offline_zgoal_dim,
            )
            expert_hidden = _feature_width(
                raw_input_features.get("observation.extra.z_phase"),
                self.config.offline_zphase_dim,
            )
            self._init_fast_module(zgoal_dim=text_hidden, zphase_dim=expert_hidden)
            self._init_hfrvla_state()
            return

        super().__init__(config, **kwargs)
        self.config: HFRVLAConfig = config

        # ── 1. Freeze SmolVLA ──
        if config.freeze_smolvla:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

        # ── 2. Hook caches ──
        self._zgoal_cache: Optional[Tensor] = None
        self._zphase_cache: Optional[Tensor] = None
        # Resolved dims once hooks fire for the first time.
        self._zgoal_dim: Optional[int] = None
        self._zphase_dim: Optional[int] = None
        self._hook_handles: list = []

        self._register_hooks(
            zgoal_path=config.hook_zgoal_layer,
            zphase_path=config.hook_zphase_layer,
        )

        # The hidden dim of z_goal / z_phase depends on SmolVLA; we discover
        # them lazily from a probe forward. The module itself defaults to the
        # SmolVLA text_config sizes so it can be instantiated up-front.
        text_hidden = int(self.model.vlm_with_expert.config.text_config.hidden_size)
        expert_hidden = int(self.model.vlm_with_expert.expert_hidden_size)
        self._init_fast_module(zgoal_dim=text_hidden, zphase_dim=expert_hidden)

        self._init_hfrvla_state()

    def _init_fast_module(self, *, zgoal_dim: int, zphase_dim: int) -> None:
        """Build the trainable fast module from policy feature dimensions."""
        # action_feature / robot_state_feature are derived from input/output_features;
        # when instantiated without a dataset they are None. Fall back to the
        # SmolVLA padding caps so the module shape is well-defined.
        action_feat = getattr(self.config, "action_feature", None)
        action_dim = (
            int(action_feat.shape[0]) if action_feat is not None
            else int(getattr(self.config, "max_action_dim", 7))
        )
        state_feat = getattr(self.config, "robot_state_feature", None)
        proprio_dim = (
            int(state_feat.shape[0]) if state_feat is not None
            else int(getattr(self.config, "max_state_dim", 32))
        )

        mode = _residual_merge_mode(self.config)
        if mode == "fast_wrist":
            self.fast = FastWristResidualModule(
                config=self.config,
                action_dim=action_dim,
                proprio_dim=proprio_dim,
                zgoal_dim=zgoal_dim,
                zphase_dim=zphase_dim,
            )
        elif mode == "fast_wrist_chunk":
            self.fast = FastWristChunkResidualModule(
                config=self.config,
                action_dim=action_dim,
                proprio_dim=proprio_dim,
                zgoal_dim=zgoal_dim,
                zphase_dim=zphase_dim,
            )
        elif mode == "gated":
            self.fast = FastReactiveModule(
                config=self.config,
                action_dim=action_dim,
                proprio_dim=proprio_dim,
                zgoal_dim=zgoal_dim,
                zphase_dim=zphase_dim,
            )
        else:
            raise ValueError(
                f"Unsupported residual_merge_mode={self.config.residual_merge_mode!r}; "
                "expected 'gated', 'fast_wrist', or 'fast_wrist_chunk'."
            )

    def _init_hfrvla_state(self) -> None:
        """Initialize curriculum, queues, and per-episode fast state."""
        # ── 4. Curriculum state ──
        # The wrapper training script calls set_training_step(step) before
        # each optimizer step.
        self._cached_lambda_gate = float(self.config.loss_lambda_gate)
        self._cached_lambda_contact = float(self.config.loss_lambda_contact)
        self._cached_lambda_final = float(self.config.loss_lambda_final)
        self._cached_lambda_preserve = float(self.config.loss_lambda_preserve)
        self._cached_lambda_gate_prior = float(self.config.loss_lambda_gate_prior)
        self._prev_stage: int = -1
        self._refine_lr_pending: bool = False

        # ── 5. Inference state ──
        self._fast_hidden_state: Optional[Tensor] = None
        self._prev_action: Optional[Tensor] = None
        self._prev_a_base: Optional[Tensor] = None
        self._prev_delta_exec: Optional[Tensor] = None
        self._current_a_base_chunk: Optional[Tensor] = None
        self._last_base_action: Optional[Tensor] = None
        self._pending_a_base_chunk: Optional[Tensor] = None
        self._pending_zgoal_cache: Optional[Tensor] = None
        self._pending_zphase_cache: Optional[Tensor] = None
        self._pending_chunk_ready_in: int = 0
        self._pending_request_step: Optional[int] = None
        self._pending_observation_step: Optional[int] = None
        self._pending_ready_step: Optional[int] = None
        self._pending_chunk_start_index: int = 0
        self._async_control_step: int = 0
        self._async_next_request_step: int = int(
            getattr(self.config, "async_request_interval_steps", 8)
        )
        self._active_chunk_start_index: int = 0
        self._chunk_size: int = self.config.n_action_steps
        self._chunk_consumed: int = 0
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        self.reset_inference_debug_stats()

    def reset_inference_debug_stats(self) -> None:
        """Reset lightweight counters for verifying runtime correction usage."""
        self._inference_debug_stats = {
            "select_action_calls": 0,
            "new_chunks": 0,
            "fast_applied": 0,
            "fast_skipped_disabled": 0,
            "fast_skipped_hook_cache_miss": 0,
            "delta_norm_sum": 0.0,
            "delta_clip_fraction_sum": 0.0,
            "k_sum": 0.0,
            "slow_chunk_latency_ms_sum": 0.0,
            "slow_chunk_latency_calls": 0,
            "fast_latency_ms_sum": 0.0,
            "fast_latency_calls": 0,
            "planner_delay_mode": str(getattr(self.config, "planner_delay_mode", "async_timestep")),
            "planner_delay_steps": int(getattr(self.config, "planner_delay_steps", 0)),
            "async_request_interval_steps": int(
                getattr(self.config, "async_request_interval_steps", 8)
            ),
            "planner_delay_requests": 0,
            "delayed_chunks_enqueued": 0,
            "async_request_count": 0,
            "async_activation_count": 0,
            "async_skipped_request_count": 0,
            "async_request_step_last": -1,
            "async_observation_step_last": -1,
            "async_ready_step_last": -1,
            "async_activate_step_last": -1,
            "async_chunk_start_index_last": 0,
            "async_chunk_start_index_sum": 0.0,
            "async_dropped_old_queue_steps_last": 0,
            "async_dropped_old_queue_steps_sum": 0.0,
            "async_trace_event_count": 0,
            "async_event_trace": [],
            "fallback_steps": 0,
            "hold_last_fallback_steps": 0,
            "zero_fallback_steps": 0,
        }

    def _ensure_inference_debug_stats(self) -> dict[str, float | int]:
        if not hasattr(self, "_inference_debug_stats"):
            self.reset_inference_debug_stats()
        return self._inference_debug_stats

    def _record_inference_new_chunk(self) -> None:
        stats = self._ensure_inference_debug_stats()
        stats["new_chunks"] += 1

    def _record_inference_fast_skipped(self, reason: str) -> None:
        stats = self._ensure_inference_debug_stats()
        stats["select_action_calls"] += 1
        key = f"fast_skipped_{reason}"
        if key not in stats:
            stats[key] = 0
        stats[key] += 1

    def _record_inference_fast_applied(self, *, k: int, delta_a: Tensor) -> None:
        stats = self._ensure_inference_debug_stats()
        stats["select_action_calls"] += 1
        stats["fast_applied"] += 1
        cap = self._fast_residual_cap()
        stats["delta_norm_sum"] += float(delta_a.detach().norm(dim=-1).mean().item())
        stats["delta_clip_fraction_sum"] += float(
            (delta_a.detach().abs() > cap).to(dtype=torch.float32).mean().item()
        )
        stats["k_sum"] += float(k)

    def _record_slow_chunk_latency(self, elapsed_s: float) -> None:
        stats = self._ensure_inference_debug_stats()
        stats["slow_chunk_latency_ms_sum"] += float(elapsed_s * 1000.0)
        stats["slow_chunk_latency_calls"] += 1

    def _record_fast_latency(self, elapsed_s: float) -> None:
        stats = self._ensure_inference_debug_stats()
        stats["fast_latency_ms_sum"] += float(elapsed_s * 1000.0)
        stats["fast_latency_calls"] += 1

    def get_inference_debug_stats(self) -> dict[str, float | int]:
        """Return counters plus aggregate ratios for eval/deployment auditing."""
        stats = dict(self._ensure_inference_debug_stats())
        calls = max(1, int(stats["select_action_calls"]))
        applied = max(1, int(stats["fast_applied"]))
        stats["fast_applied_ratio"] = float(stats["fast_applied"]) / calls
        stats["delta_norm_mean"] = float(stats["delta_norm_sum"]) / applied
        stats["delta_clip_fraction_mean"] = float(stats["delta_clip_fraction_sum"]) / applied
        stats["k_mean"] = float(stats["k_sum"]) / applied
        slow_calls = max(1, int(stats["slow_chunk_latency_calls"]))
        fast_calls = max(1, int(stats["fast_latency_calls"]))
        stats["slow_chunk_latency_ms_mean"] = float(stats["slow_chunk_latency_ms_sum"]) / slow_calls
        stats["fast_latency_ms_mean"] = float(stats["fast_latency_ms_sum"]) / fast_calls
        stats["slow_replan_count"] = int(stats["new_chunks"])
        async_activations = max(1, int(stats.get("async_activation_count", 0)))
        stats["async_chunk_start_index_mean"] = (
            float(stats.get("async_chunk_start_index_sum", 0.0)) / async_activations
        )
        stats["async_dropped_old_queue_steps_mean"] = (
            float(stats.get("async_dropped_old_queue_steps_sum", 0.0)) / async_activations
        )
        return stats

    # ────────────────────────────────────────────────────────────────────
    # Hook plumbing
    # ────────────────────────────────────────────────────────────────────
    def _register_hooks(self, zgoal_path: str, zphase_path: str) -> None:
        """Wrap ``vlm_with_expert.forward`` so we can capture z_goal / z_phase
        directly from its return value.

        Why a monkey-patch instead of ``register_forward_hook``? PyTorch hooks
        only fire when a module is invoked via ``module()`` (i.e. ``__call__``).
        SmolVLA's flow-matching denoiser calls ``self.vlm_with_expert.forward(...)``
        explicitly (see ``modeling_smolvla.py``), bypassing ``__call__`` and
        therefore bypassing any registered hook. Wrapping the bound ``forward``
        method is the only reliable way to intercept those calls.

        Additionally SmolVLA's ``SmolVLMWithExpertModel.forward`` returns
        ``(outputs_embeds_norm, past_key_values)`` where
        ``outputs_embeds_norm[0]`` is the final VLM text hidden state
        (-> ``z_goal``) and ``outputs_embeds_norm[1]`` is the final action-
        expert hidden state (-> ``z_phase``).

        The ``zgoal_path`` / ``zphase_path`` config fields are retained for
        backward compatibility but are not used here.
        """
        del zgoal_path, zphase_path  # legacy, unused

        vwe = self.model.vlm_with_expert
        original_forward = vwe.forward
        policy_self = self

        def wrapped_forward(*args, **kwargs):
            outputs = original_forward(*args, **kwargs)
            outputs_embeds_norm = outputs[0] if isinstance(outputs, tuple) else outputs
            if isinstance(outputs_embeds_norm, (list, tuple)):
                if len(outputs_embeds_norm) >= 1 and outputs_embeds_norm[0] is not None:
                    t = outputs_embeds_norm[0]
                    if t.dim() == 3:
                        policy_self._zgoal_cache = t.mean(dim=1).detach().float()
                if len(outputs_embeds_norm) >= 2 and outputs_embeds_norm[1] is not None:
                    t = outputs_embeds_norm[1]
                    if t.dim() == 3:
                        policy_self._zphase_cache = t.mean(dim=1).detach().float()
            return outputs

        vwe.forward = wrapped_forward
        # Stash the original so we can restore on detach.
        self._hook_handles.append(("vwe_forward", vwe, original_forward))

    @staticmethod
    def _resolve_last_layer(root: nn.Module, dotted: str) -> nn.Module:
        """Resolve a dotted attribute path that ends with a ``nn.ModuleList``
        (or similar) and return its LAST element.
        """
        node = root
        for part in dotted.split("."):
            if hasattr(node, part):
                node = getattr(node, part)
            else:
                raise AttributeError(
                    f"Could not resolve `{part}` in `{dotted}` while walking "
                    f"{type(root).__name__}. Inspect named_modules() and "
                    f"update HFRVLAConfig.hook_zgoal_layer / hook_zphase_layer."
                )
        # If we ended on a ModuleList, take the last layer.
        if isinstance(node, (nn.ModuleList, list)):
            return node[-1]
        return node

    def _clear_hook_cache(self) -> None:
        self._zgoal_cache = None
        self._zphase_cache = None

    def _detach_hooks(self) -> None:
        for entry in self._hook_handles:
            # Either a PyTorch hook handle or our wrapped-forward tuple.
            if isinstance(entry, tuple) and len(entry) == 3 and entry[0] == "vwe_forward":
                _, module, original = entry
                module.forward = original
            else:
                entry.remove()
        self._hook_handles = []

    # ────────────────────────────────────────────────────────────────────
    # Reset / lifecycle
    # ────────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        if getattr(self, "_offline_training_mode", False):
            self._queues = {
                ACTION: deque(maxlen=self.config.n_action_steps),
            }
        else:
            super().reset()
        self._fast_hidden_state = None
        self._prev_action = None
        self._prev_a_base = None
        self._prev_delta_exec = None
        self._current_a_base_chunk = None
        self._last_base_action = None
        self._pending_a_base_chunk = None
        self._pending_zgoal_cache = None
        self._pending_zphase_cache = None
        self._pending_chunk_ready_in = 0
        self._pending_request_step = None
        self._pending_observation_step = None
        self._pending_ready_step = None
        self._pending_chunk_start_index = 0
        self._async_control_step = 0
        self._async_next_request_step = int(
            getattr(self.config, "async_request_interval_steps", 8)
        )
        self._active_chunk_start_index = 0
        self._chunk_consumed = 0
        self._clear_hook_cache()

    # ────────────────────────────────────────────────────────────────────
    # Planner-delay queue simulation
    # ────────────────────────────────────────────────────────────────────
    def _planner_delay_steps(self) -> int:
        return max(0, int(getattr(self.config, "planner_delay_steps", 0)))

    def _planner_delay_fallback(self) -> str:
        fallback = str(getattr(self.config, "planner_delay_fallback", "hold_last")).strip().lower()
        if fallback not in {"hold_last", "zero"}:
            raise ValueError("planner_delay_fallback must be 'hold_last' or 'zero'")
        return fallback

    def _async_request_interval_steps(self) -> int:
        return max(1, int(getattr(self.config, "async_request_interval_steps", 8)))

    def _trace_planner_delay_event(self, event: dict[str, int | str]) -> None:
        stats = self._ensure_inference_debug_stats()
        stats["async_trace_event_count"] += 1
        max_events = int(getattr(self.config, "planner_delay_trace_max_events", 128))
        if max_events <= 0:
            return
        trace = stats.setdefault("async_event_trace", [])
        if not isinstance(trace, list):
            trace = []
            stats["async_event_trace"] = trace
        trace.append(event)
        del trace[:-max_events]

    def _request_immediate_action_chunk(self, batch: dict[str, Tensor]) -> None:
        self._clear_hook_cache()
        start = time.perf_counter()
        actions = self._get_action_chunk(batch)
        self._record_slow_chunk_latency(time.perf_counter() - start)
        self._current_a_base_chunk = actions.detach()
        self._queues[ACTION].extend(
            actions.transpose(0, 1)[: self.config.n_action_steps]
        )
        self._active_chunk_start_index = 0
        self._chunk_consumed = 0
        self._async_next_request_step = (
            self._async_control_step + self._async_request_interval_steps()
        )
        self._trace_planner_delay_event(
            {
                "event": "cold_start",
                "step": int(self._async_control_step),
                "observation_step": int(self._async_control_step),
                "chunk_start_index": 0,
                "enqueued_steps": int(len(self._queues[ACTION])),
            }
        )
        self._record_inference_new_chunk()

    def _maybe_start_async_timestep_request(self, batch: dict[str, Tensor]) -> bool:
        if self._pending_a_base_chunk is not None:
            return False
        if self._async_control_step < self._async_next_request_step:
            return False
        if len(self._queues[ACTION]) == 0 and self._last_base_action is None:
            return False

        delay_steps = self._planner_delay_steps()
        request_step = int(self._async_control_step)
        observation_step = request_step
        ready_step = request_step + delay_steps
        chunk_start_index = delay_steps

        active_zgoal = self._zgoal_cache
        active_zphase = self._zphase_cache
        self._clear_hook_cache()
        start = time.perf_counter()
        actions = self._get_action_chunk(batch)
        self._record_slow_chunk_latency(time.perf_counter() - start)
        self._pending_a_base_chunk = actions.detach()
        self._pending_zgoal_cache = self._zgoal_cache
        self._pending_zphase_cache = self._zphase_cache
        self._pending_chunk_ready_in = delay_steps
        self._pending_request_step = request_step
        self._pending_observation_step = observation_step
        self._pending_ready_step = ready_step
        self._pending_chunk_start_index = chunk_start_index
        self._zgoal_cache = active_zgoal
        self._zphase_cache = active_zphase
        self._async_next_request_step = request_step + self._async_request_interval_steps()

        stats = self._ensure_inference_debug_stats()
        stats["planner_delay_requests"] += 1
        stats["async_request_count"] += 1
        stats["async_request_step_last"] = request_step
        stats["async_observation_step_last"] = observation_step
        stats["async_ready_step_last"] = ready_step
        self._trace_planner_delay_event(
            {
                "event": "request",
                "request_step": request_step,
                "observation_step": observation_step,
                "ready_step": ready_step,
                "delay_steps": delay_steps,
                "request_interval_steps": self._async_request_interval_steps(),
            }
        )
        self._record_inference_new_chunk()
        return True

    def _activate_ready_delayed_chunk_if_any(self) -> bool:
        if self._pending_a_base_chunk is None:
            return False
        ready_step = (
            int(self._pending_ready_step)
            if self._pending_ready_step is not None
            else int(self._async_control_step)
        )
        if self._async_control_step < ready_step:
            return False

        chunk_start_index = int(self._pending_chunk_start_index)
        chunk_len = int(self._pending_a_base_chunk.shape[1])
        chunk_start_index = max(0, min(chunk_start_index, max(0, chunk_len - 1)))
        dropped_old_queue_steps = len(self._queues[ACTION])
        self._queues[ACTION].clear()
        self._current_a_base_chunk = self._pending_a_base_chunk
        self._queues[ACTION].extend(
            self._pending_a_base_chunk.transpose(0, 1)[
                chunk_start_index : chunk_start_index + self.config.n_action_steps
            ]
        )
        self._zgoal_cache = self._pending_zgoal_cache
        self._zphase_cache = self._pending_zphase_cache
        self._pending_a_base_chunk = None
        self._pending_zgoal_cache = None
        self._pending_zphase_cache = None
        self._pending_chunk_ready_in = 0
        self._active_chunk_start_index = chunk_start_index
        self._chunk_consumed = 0
        stats = self._ensure_inference_debug_stats()
        stats["delayed_chunks_enqueued"] += 1
        request_step = (
            int(self._pending_request_step)
            if self._pending_request_step is not None
            else int(self._async_control_step)
        )
        observation_step = (
            int(self._pending_observation_step)
            if self._pending_observation_step is not None
            else request_step
        )
        ready_step = (
            int(self._pending_ready_step)
            if self._pending_ready_step is not None
            else int(self._async_control_step)
        )
        stats["async_activation_count"] += 1
        stats["async_activate_step_last"] = int(self._async_control_step)
        stats["async_chunk_start_index_last"] = chunk_start_index
        stats["async_chunk_start_index_sum"] += float(chunk_start_index)
        stats["async_dropped_old_queue_steps_last"] = dropped_old_queue_steps
        stats["async_dropped_old_queue_steps_sum"] += float(dropped_old_queue_steps)
        self._trace_planner_delay_event(
            {
                "event": "activate",
                "request_step": request_step,
                "observation_step": observation_step,
                "ready_step": ready_step,
                "activate_step": int(self._async_control_step),
                "chunk_start_index": chunk_start_index,
                "dropped_old_queue_steps": int(dropped_old_queue_steps),
                "enqueued_steps": int(len(self._queues[ACTION])),
            }
        )
        self._pending_request_step = None
        self._pending_observation_step = None
        self._pending_ready_step = None
        self._pending_chunk_start_index = 0
        return True

    def _current_chunk_step_index(self) -> int:
        return max(0, int(self._active_chunk_start_index) + max(0, self._chunk_consumed - 1))

    def _advance_async_control_step_after_action(self) -> None:
        self._async_control_step += 1

    def _pop_or_fallback_base_action(self) -> tuple[Tensor, bool]:
        if len(self._queues[ACTION]) > 0:
            a_base = self._queues[ACTION].popleft()
            self._chunk_consumed += 1
            self._last_base_action = a_base.detach()
            return a_base, False

        stats = self._ensure_inference_debug_stats()
        stats["fallback_steps"] += 1
        fallback = self._planner_delay_fallback()
        if fallback == "hold_last" and self._last_base_action is not None:
            stats["hold_last_fallback_steps"] += 1
            a_base = self._last_base_action.clone()
        else:
            stats["zero_fallback_steps"] += 1
            if self._last_base_action is not None:
                a_base = torch.zeros_like(self._last_base_action)
            elif self._pending_a_base_chunk is not None:
                a_base = torch.zeros_like(self._pending_a_base_chunk[:, 0])
            else:
                raise RuntimeError("planner delay fallback requested before any base action exists")
        self._chunk_consumed += 1
        self._last_base_action = a_base.detach()
        return a_base, True

    # ────────────────────────────────────────────────────────────────────
    # Inference
    # ────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Pop the next a_base from SmolVLA's queue and apply the fast residual.

        SmolVLA's ``_get_action_chunk`` is invoked only when the queue is empty;
        the hooks fire during that call and populate ``_zgoal_cache`` and
        ``_zphase_cache``.
        """
        if getattr(self, "_offline_training_mode", False):
            raise RuntimeError(
                "HFRVLAPolicy was initialized with offline_training_mode=True, "
                "which is only for training from cached HFRVLA features. Package "
                "the trained fast checkpoint with scripts/package_hfrvla_checkpoint.py "
                "before running lerobot-eval."
            )
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        self._activate_ready_delayed_chunk_if_any()
        if (
            len(self._queues[ACTION]) == 0
            and self._last_base_action is None
            and self._pending_a_base_chunk is None
        ):
            self._request_immediate_action_chunk(batch)
        else:
            self._maybe_start_async_timestep_request(batch)
            self._activate_ready_delayed_chunk_if_any()
            if len(self._queues[ACTION]) == 0 and self._pending_a_base_chunk is None:
                self._request_immediate_action_chunk(batch)

        a_base, _used_fallback = self._pop_or_fallback_base_action()

        # Alignment-test short-circuit: behave exactly like SmolVLA base.
        # Verifies that the HFRVLA wrapper + dataset feature config + action
        # post-processing produce outputs LIBERO env accepts.
        if getattr(self.config, "inference_disable_fast", False):
            self._record_inference_fast_skipped("disabled")
            self._advance_async_control_step_after_action()
            return a_base

        k = self._current_chunk_step_index()
        k_norm_denom = max(
            1,
            (
                int(self._current_a_base_chunk.shape[1]) - 1
                if _residual_merge_mode(self.config) == "fast_wrist_chunk"
                and self._current_a_base_chunk is not None
                and self._current_a_base_chunk.dim() == 3
                else self.config.n_action_steps - 1
            ),
        )
        k_for_feature = min(k, k_norm_denom)
        k_norm = torch.full(
            (a_base.shape[0], 1),
            k_for_feature / k_norm_denom,
            device=a_base.device,
            dtype=a_base.dtype,
        )

        wrist_rgb = self._extract_wrist_image(batch)
        proprio = batch[OBS_STATE]

        if self._zgoal_cache is None or self._zphase_cache is None:
            # Hooks never fired (e.g. first call has empty queue but SmolVLA
            # forward path differed); fall back to a_base alone.
            self._record_inference_fast_skipped("hook_cache_miss")
            self._advance_async_control_step_after_action()
            return a_base

        mode = _residual_merge_mode(self.config)
        if mode == "fast_wrist_chunk":
            if self._current_a_base_chunk is None:
                a_base_chunk = a_base.unsqueeze(1)
            else:
                a_base_chunk = self._current_a_base_chunk.to(device=a_base.device, dtype=a_base.dtype)
            chunk_step_idx = torch.full(
                (a_base.shape[0], 1),
                k,
                device=a_base.device,
                dtype=torch.long,
            )
            fast_start = time.perf_counter()
            fwr_out: FastWristResidualOutput = self.fast(
                wrist_rgb=wrist_rgb,
                proprio=proprio,
                a_base_k=a_base,
                k_idx_norm=k_norm,
                z_goal=self._zgoal_cache,
                z_phase=self._zphase_cache,
                a_base_chunk=a_base_chunk,
                chunk_step_idx=chunk_step_idx,
            )
            self._record_fast_latency(time.perf_counter() - fast_start)
            a_final = self._merge(
                a_base=a_base,
                delta_a=fwr_out.delta_a,
                gate=torch.ones(a_base.shape[0], device=a_base.device, dtype=a_base.dtype),
                prev_a=self._prev_action,
            )
            delta_exec = (
                _fast_residual_alpha(self.config)
                * self._clip_fast_residual(fwr_out.delta_a)
            )
            self._record_inference_fast_applied(k=k, delta_a=fwr_out.delta_a)
            self._prev_a_base = a_base.detach()
            self._prev_delta_exec = delta_exec.detach()
            self._prev_action = a_final.detach()
            self._advance_async_control_step_after_action()
            return a_final

        if mode == "fast_wrist":
            prev_a_base = self._prev_a_base if self._prev_a_base is not None else a_base
            prev_delta = (
                self._prev_delta_exec
                if self._prev_delta_exec is not None
                else torch.zeros_like(a_base)
            )
            fast_start = time.perf_counter()
            fwr_out: FastWristResidualOutput = self.fast(
                wrist_rgb=wrist_rgb,
                proprio=proprio,
                a_base_k=a_base,
                k_idx_norm=k_norm,
                z_goal=self._zgoal_cache,
                z_phase=self._zphase_cache,
                prev_a_base=prev_a_base,
                prev_delta=prev_delta,
            )
            self._record_fast_latency(time.perf_counter() - fast_start)
            a_final = self._merge(
                a_base=a_base,
                delta_a=fwr_out.delta_a,
                gate=torch.ones(a_base.shape[0], device=a_base.device, dtype=a_base.dtype),
                prev_a=self._prev_action,
            )
            delta_exec = (
                _fast_residual_alpha(self.config)
                * self._clip_fast_residual(fwr_out.delta_a)
            )
            self._record_inference_fast_applied(k=k, delta_a=fwr_out.delta_a)
            self._prev_a_base = a_base.detach()
            self._prev_delta_exec = delta_exec.detach()
            self._prev_action = a_final.detach()
            self._advance_async_control_step_after_action()
            return a_final

        fast_start = time.perf_counter()
        fr_out: FastReactiveOutput = self.fast(
            wrist_rgb=wrist_rgb,
            proprio=proprio,
            a_base_k=a_base,
            k_idx_norm=k_norm,
            z_goal=self._zgoal_cache,
            z_phase=self._zphase_cache,
            hidden_state=self._fast_hidden_state,
        )
        self._record_fast_latency(time.perf_counter() - fast_start)
        self._fast_hidden_state = fr_out.hidden_state

        a_final = self._merge(
            a_base=a_base,
            delta_a=fr_out.delta_a,
            gate=fr_out.gate,
            prev_a=self._prev_action,
        )
        self._record_inference_fast_applied(k=k, delta_a=fr_out.delta_a)
        self._prev_action = a_final.detach()
        self._advance_async_control_step_after_action()
        return a_final

    def _extract_wrist_image(self, batch: dict[str, Tensor]) -> Optional[Tensor]:
        """Find the wrist image in the batch. LeRobot dataset names vary by
        embodiment; we try the most common keys.

        ``observation.images.image2`` is LIBERO's eye-in-hand camera (mapped
        from ``robot0_eye_in_hand_image`` in lerobot/envs/configs.py); it
        matches the ``--wrist-key`` used by precompute_libero.py.
        """
        for key in (
            "observation.images.wrist",
            "observation.images.wrist_image",
            "observation.images.image_wrist",
            "observation.image.wrist",
            "observation.images.image2",
        ):
            if key in batch:
                return self._prepare_wrist_image_for_fast(batch[key])
        return None

    def _prepare_wrist_image_for_fast(self, wrist_rgb: Tensor) -> Tensor:
        """Mirror the recorder's DINOv3 wrist preprocessing for online eval."""
        x = wrist_rgb.float()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError(f"wrist image must be 3-D or 4-D, got {x.dim()}-D")
        if x.shape[-1] == 3 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.shape[1] != 3:
            raise ValueError(f"wrist image must have 3 channels, got shape {tuple(x.shape)}")
        if x.numel() and x.max() > 1.5:
            x = x / 255.0
        return normalize_for_dinov3(x, size=int(self.config.dinov3_image_size))

    # ────────────────────────────────────────────────────────────────────
    # Action merging + safety
    # ────────────────────────────────────────────────────────────────────
    def _merge(
        self,
        a_base: Tensor,
        delta_a: Tensor,
        gate: Tensor,
        prev_a: Optional[Tensor],
    ) -> Tensor:
        del prev_a  # Base SmolVLA chunks are already valid actions.
        delta_clip = self._clip_fast_residual(delta_a)
        if _is_fast_wrist_mode(self.config):
            return a_base + _fast_residual_alpha(self.config) * delta_clip
        return a_base + gate.unsqueeze(-1) * delta_clip

    def _clip_fast_residual(self, delta_a: Tensor) -> Tensor:
        return delta_a.clamp(-self._fast_residual_cap(), self._fast_residual_cap())

    def _fast_residual_cap(self) -> float:
        max_residual = float(self.config.delta_max)
        velocity_residual = (
            float(self.config.safety_joint_velocity_limit)
            * float(self.config.control_dt)
        )
        if velocity_residual > 0:
            max_residual = min(max_residual, velocity_residual)
        return max_residual

    # ────────────────────────────────────────────────────────────────────
    # Training
    # ────────────────────────────────────────────────────────────────────
    def forward(  # type: ignore[override]
        self, batch: dict[str, Tensor], **kwargs
    ) -> tuple[Tensor, dict[str, float]]:
        """Training forward - consumes the LeRobot-native batch layout.

        Expected keys (all already windowed to ``[B, T=seq_len, ...]`` by
        lerobot's delta_timestamps mechanism - see HFRVLAConfig.
        observation_delta_indices):

            observation.state
            action
            observation.extra.a_base
            observation.extra.k_idx_norm
            observation.extra.z_goal
            observation.extra.z_phase
            observation.extra.dino_patches
            observation.extra.contact_label (optional)
        """
        proprio = batch["observation.state"]
        a_expert = batch["action"]
        a_base = batch["observation.extra.a_base"]
        k_idx_norm = batch["observation.extra.k_idx_norm"]
        k_idx_norm = _ensure_trailing_feature_dim(k_idx_norm, reference=a_base)
        z_goal = batch["observation.extra.z_goal"]
        z_phase = batch["observation.extra.z_phase"]
        dino_patches = batch["observation.extra.dino_patches"]
        contact_label = batch.get("observation.extra.contact_label")

        mode = _residual_merge_mode(self.config)
        if mode == "fast_wrist_chunk":
            a_base_chunk = batch["observation.extra.a_base_chunk"]
            chunk_step_idx = batch["observation.extra.chunk_step_idx"]
            fwr_out: FastWristResidualOutput = self.fast(
                wrist_rgb=None,
                proprio=proprio[:, -1],
                a_base_k=a_base[:, -1],
                k_idx_norm=k_idx_norm[:, -1],
                z_goal=z_goal[:, -1],
                z_phase=z_phase[:, -1],
                a_base_chunk=a_base_chunk[:, -1],
                chunk_step_idx=chunk_step_idx[:, -1],
                dino_patches=dino_patches[:, -1],
            )
            losses = self._compute_losses(
                fwr_out,
                a_base,
                a_expert,
                contact_label=None,
                batch=batch,
            )
            output_dict = {key: value.detach().item() for key, value in losses.items()}
            return losses["loss"], output_dict

        if mode == "fast_wrist":
            if a_base.dim() != 3 or a_base.size(1) < 2:
                raise ValueError(
                    "Fast Wrist Residual requires policy.seq_len >= 2 so the "
                    "batch contains [previous, current] frames."
                )
            prev_a_base = a_base[:, -2]
            prev_delta = a_expert[:, -2] - a_base[:, -2]
            fwr_out: FastWristResidualOutput = self.fast(
                wrist_rgb=None,
                proprio=proprio[:, -1],
                a_base_k=a_base[:, -1],
                k_idx_norm=k_idx_norm[:, -1],
                z_goal=z_goal[:, -1],
                z_phase=z_phase[:, -1],
                prev_a_base=prev_a_base,
                prev_delta=prev_delta,
                dino_patches=dino_patches[:, -1],
            )
            losses = self._compute_losses(
                fwr_out,
                a_base,
                a_expert,
                contact_label=None,
                batch=batch,
            )
            output_dict = {key: value.detach().item() for key, value in losses.items()}
            return losses["loss"], output_dict

        fr_out: FastReactiveOutput = self.fast(
            wrist_rgb=None,
            proprio=proprio,
            a_base_k=a_base,
            k_idx_norm=k_idx_norm,
            z_goal=z_goal,
            z_phase=z_phase,
            dino_patches=dino_patches,
        )
        losses = self._compute_losses(fr_out, a_base, a_expert, contact_label, batch=batch)
        output_dict = {key: value.detach().item() for key, value in losses.items()}
        return losses["loss"], output_dict

    def set_training_step(self, step: int) -> None:
        """Curriculum controller. Called by train_via_lerobot.py per step."""
        _apply_curriculum_stage(self, step)

    def consume_refine_lr_signal(self) -> bool:
        """Return True exactly once when stage 2 is first entered."""
        flag = self._refine_lr_pending
        self._refine_lr_pending = False
        return flag

    def _compute_losses(
        self,
        out: FastReactiveOutput | FastWristResidualOutput,
        a_base: Tensor,
        a_expert: Tensor,
        contact_label: Optional[Tensor],
        batch: Optional[dict[str, Tensor]] = None,
    ) -> dict[str, Tensor]:
        if _is_fast_wrist_mode(self.config):
            if not isinstance(out, FastWristResidualOutput):
                raise TypeError("Fast Wrist Residual objective requires FastWristResidualOutput.")
            return self._compute_fast_wrist_losses(out=out, a_base=a_base, a_expert=a_expert)

        if getattr(self.config, "use_stage_b_objective", False):
            return self._compute_stage_b_losses(
                out=out,
                a_base=a_base,
                a_expert=a_expert,
                contact_label=contact_label,
                batch=batch,
            )

        # L_delta: train the residual that deployment can actually execute.
        target_delta = a_expert - a_base
        if getattr(self.config, "loss_delta_target_clip", True):
            target_delta = self._clip_fast_residual(target_delta)
        l_delta = F.mse_loss(out.delta_a, target_delta)

        delta_clip = self._clip_fast_residual(out.delta_a)
        # Stage A (debate 20260521): detach gate inside the merged-action path
        # so L_final / L_preserve no longer reward opening the gate. Gate is
        # supervised solely by L_gate (BCE on improvement label) + L_gate_prior
        # (rate). Breaks the gradient route that drove gate_prior to ~0.95.
        a_final = a_base + out.gate.detach().unsqueeze(-1) * delta_clip
        err_before = ((a_expert - a_base) ** 2).sum(dim=-1)
        err_final = ((a_expert - a_final) ** 2).sum(dim=-1)

        # L_gate: only open the gate when the clipped residual clears a margin.
        with torch.no_grad():
            delta_detached = self._clip_fast_residual(out.delta_a.detach())
            err_after = ((a_expert - (a_base + delta_detached)) ** 2).sum(dim=-1)
            improvement = err_before - err_after
            margin = float(getattr(self.config, "gate_improvement_margin", 0.0))
            g_target = (improvement > margin).to(dtype=out.gate_logit.dtype)
        l_gate = F.binary_cross_entropy_with_logits(out.gate_logit, g_target)

        # L_final trains the deployed merged action. L_preserve explicitly
        # penalizes corrections that make the frozen base action worse.
        l_final = F.mse_loss(a_final, a_expert)
        l_preserve = F.relu(err_final - err_before).mean()
        l_gate_prior = out.gate.mean()

        # Stage A (debate 20260521): explicit zero-target on preserve-class
        # states (base already close to expert). Drives ||delta_a||^2 → 0 on
        # those frames so the residual cannot learn "small everywhere".
        preserve_thresh = float(getattr(self.config, "err_preserve_thresh", 0.01))
        is_preserve = (err_before < preserve_thresh).to(dtype=out.delta_a.dtype)
        l_preserve_zero = (
            out.delta_a.pow(2).sum(dim=-1) * is_preserve
        ).mean()

        losses: dict[str, Tensor] = {
            "delta": l_delta,
            "gate": l_gate,
            "final": l_final,
            "preserve": l_preserve,
            "gate_prior": l_gate_prior,
            "preserve_zero": l_preserve_zero,
        }

        # L_contact — optional aux head
        if out.contact_logit is not None and contact_label is not None:
            target = contact_label.float()
            if target.dim() == out.contact_logit.dim() + 1 and target.shape[-1] == 1:
                target = target.squeeze(-1)
            # Broadcast if the contact head produced a sequence dim.
            if target.shape != out.contact_logit.shape:
                target = target.expand_as(out.contact_logit)
            l_contact = F.binary_cross_entropy_with_logits(out.contact_logit, target)
            losses["contact"] = l_contact

        total = (
            l_delta
            + self.config.loss_lambda_gate * l_gate
            + self.config.loss_lambda_final * l_final
            + self.config.loss_lambda_preserve * l_preserve
            + self.config.loss_lambda_gate_prior * l_gate_prior
            + float(getattr(self.config, "loss_lambda_preserve_zero", 0.0))
            * l_preserve_zero
        )
        if "contact" in losses:
            total = total + self.config.loss_lambda_contact * losses["contact"]
        losses["loss"] = total
        return losses

    def _compute_fast_wrist_losses(
        self,
        *,
        out: FastWristResidualOutput,
        a_base: Tensor,
        a_expert: Tensor,
    ) -> dict[str, Tensor]:
        if a_base.dim() == 3:
            a_base_curr = a_base[:, -1]
            a_expert_curr = a_expert[:, -1]
        elif a_base.dim() == 2:
            a_base_curr = a_base
            a_expert_curr = a_expert
        else:
            raise ValueError(f"a_base must be 2-D or 3-D, got {a_base.dim()}-D")

        target_delta = a_expert_curr - a_base_curr
        l_delta = F.smooth_l1_loss(out.delta_a, target_delta)
        delta_clip = self._clip_fast_residual(out.delta_a)
        delta_exec = _fast_residual_alpha(self.config) * delta_clip
        a_final = a_base_curr + delta_exec
        l_final = F.smooth_l1_loss(a_final, a_expert_curr)
        l_residual = delta_exec.pow(2).mean()
        l_clip = F.relu(out.delta_a.abs() - self._fast_residual_cap()).mean()
        total = (
            float(getattr(self.config, "fast_wrist_loss_lambda_delta", 1.0)) * l_delta
            + float(getattr(self.config, "fast_wrist_loss_lambda_final", 1.0)) * l_final
            + float(getattr(self.config, "fast_wrist_loss_lambda_residual", 0.0)) * l_residual
            + float(getattr(self.config, "fast_wrist_loss_lambda_clip", 0.0)) * l_clip
        )
        losses: dict[str, Tensor] = {
            "delta": l_delta,
            "final": l_final,
            "residual_reg": l_residual,
            "clip_penalty": l_clip,
            "loss": total,
            "delta_norm": out.delta_a.norm(dim=-1).mean().detach(),
            "target_delta_norm": target_delta.norm(dim=-1).mean().detach(),
            "delta_clip_fraction": (
                (out.delta_a.abs() > self._fast_residual_cap())
                .to(dtype=out.delta_a.dtype)
                .mean()
                .detach()
            ),
        }
        return losses

    def _stage_b_label(
        self,
        batch: Optional[dict[str, Tensor]],
        key: str,
        reference: Tensor,
    ) -> Tensor:
        if batch is None or key not in batch:
            raise ValueError(
                "Stage B objective requires fast-cache schema v2 static labels "
                "('observation.extra.y_correct' and 'observation.extra.y_preserve'). "
                "Rebuild with scripts/build_hfrvla_fastcache.py using a _v2 cache path."
            )
        label = batch[key].to(device=reference.device, dtype=reference.dtype)
        while label.dim() > reference.dim() - 1 and label.shape[-1] == 1:
            label = label.squeeze(-1)
        expected_shape = reference.shape[:-1]
        if label.shape != expected_shape:
            raise ValueError(
                f"Stage B label {key} has shape {tuple(label.shape)}, "
                f"expected {tuple(expected_shape)}"
            )
        return label

    def _compute_stage_b_losses(
        self,
        *,
        out: FastReactiveOutput,
        a_base: Tensor,
        a_expert: Tensor,
        contact_label: Optional[Tensor],
        batch: Optional[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        u = self._clip_fast_residual(out.delta_a)
        r_t = self._clip_fast_residual(a_expert - a_base)

        y_correct = self._stage_b_label(
            batch,
            "observation.extra.y_correct",
            out.delta_a,
        )
        y_preserve = self._stage_b_label(
            batch,
            "observation.extra.y_preserve",
            out.delta_a,
        )

        # L_correct: distortion only on static correction frames.
        correct_per_frame = F.smooth_l1_loss(
            out.delta_a,
            r_t,
            reduction="none",
        ).sum(dim=-1)
        n_correct = y_correct.sum().clamp(min=1.0)
        l_correct = (correct_per_frame * y_correct).sum() / n_correct

        # L_preserve_zero: explicit zero target on static preserve frames.
        preserve_per_frame = out.delta_a.pow(2).sum(dim=-1)
        n_preserve = y_preserve.sum().clamp(min=1.0)
        l_preserve_zero = (preserve_per_frame * y_preserve).sum() / n_preserve

        # L_rate: Bernoulli gate rate plus a batch-level budget hinge.
        g = torch.sigmoid(out.gate_logit)
        l_budget = F.relu(g.mean() - float(self.config.gate_task_budget)) ** 2
        l_rate = g.mean() + l_budget

        # L_gate: focal BCE against the static correction label.
        p = torch.sigmoid(out.gate_logit)
        p_t = p * y_correct + (1 - p) * (1 - y_correct)
        alpha = float(self.config.focal_pos_weight) * y_correct + (1 - y_correct)
        gamma = float(self.config.focal_gamma)
        focal_weight = (1 - p_t).clamp(min=1e-6) ** gamma
        log_p_t = torch.log(p_t.clamp(min=1e-6))
        l_gate = -(alpha * focal_weight * log_p_t).mean()

        # L_smooth: temporal smoothness on the executed residual. Gate is
        # detached so it remains controlled by L_gate + L_rate only.
        gu = g.detach().unsqueeze(-1) * u
        if gu.dim() >= 3 and gu.size(1) > 1:
            diff = gu[:, 1:] - gu[:, :-1]
            l_smooth = diff.pow(2).sum(dim=-1).mean()
        else:
            l_smooth = torch.zeros((), device=gu.device, dtype=gu.dtype)

        total = (
            self.config.loss_lambda_correct * l_correct
            + float(getattr(self.config, "loss_lambda_preserve_zero", 2.0))
            * l_preserve_zero
            + self.config.loss_lambda_rate * l_rate
            + self.config.loss_lambda_gate * l_gate
            + self.config.loss_lambda_smooth * l_smooth
        )

        losses: dict[str, Tensor] = {
            "correct": l_correct,
            "preserve_zero": l_preserve_zero,
            "rate": l_rate,
            "gate": l_gate,
            "smooth": l_smooth,
            "loss": total,
            # Legacy metric keys kept so existing dashboards/tests do not break.
            "delta": l_correct.detach(),
            "final": torch.zeros((), device=total.device, dtype=total.dtype),
            "preserve": torch.zeros((), device=total.device, dtype=total.dtype),
            "gate_prior": g.mean().detach(),
        }

        if out.contact_logit is not None and contact_label is not None:
            target = contact_label.float()
            if target.dim() == out.contact_logit.dim() + 1 and target.shape[-1] == 1:
                target = target.squeeze(-1)
            if target.shape != out.contact_logit.shape:
                target = target.expand_as(out.contact_logit)
            l_contact = F.binary_cross_entropy_with_logits(out.contact_logit, target)
            losses["contact"] = l_contact
            total = total + self.config.loss_lambda_contact * l_contact
            losses["loss"] = total

        return losses

    # ────────────────────────────────────────────────────────────────────
    def get_optim_params(self) -> list[nn.Parameter]:
        """Return only the trainable parameters of the fast reactive module."""
        return [p for p in self.fast.parameters() if p.requires_grad]

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save offline training checkpoints without safetensors GRU alias errors.

        CUDA RNN modules can expose weights as views into a flattened cuDNN
        storage after the first forward pass. ``safetensors.torch.save_model``
        refuses those views because no tensor covers the whole storage. Offline
        training checkpoints only need the fast module, so clone each tensor to
        a compact contiguous CPU tensor before writing.
        """
        if not getattr(self, "_offline_training_mode", False):
            return super()._save_pretrained(save_directory)

        from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
        from safetensors.torch import save_file

        save_directory = Path(save_directory)
        self.config._save_pretrained(save_directory)
        model_to_save = self.module if hasattr(self, "module") else self
        state = {
            key: value.detach().cpu().contiguous().clone()
            for key, value in model_to_save.state_dict().items()
            if torch.is_tensor(value)
        }
        save_file(
            state,
            str(save_directory / SAFETENSORS_SINGLE_FILE),
            metadata={"format": "pt"},
        )
