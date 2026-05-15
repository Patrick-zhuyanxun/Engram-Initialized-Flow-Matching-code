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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_STATE

from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig
from lerobot_policy_hfrvla.fast_reactive import FastReactiveModule, FastReactiveOutput


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
        _set_head_grads(policy_or_stub, gate=False, contact=False)
    elif stage == 1 and prev != 1:
        cfg.loss_lambda_gate = policy_or_stub._cached_lambda_gate
        cfg.loss_lambda_contact = policy_or_stub._cached_lambda_contact
        _set_head_grads(policy_or_stub, gate=True, contact=True)
    elif stage == 2 and prev != 2:
        cfg.loss_lambda_gate = policy_or_stub._cached_lambda_gate
        cfg.loss_lambda_contact = policy_or_stub._cached_lambda_contact
        _set_head_grads(policy_or_stub, gate=True, contact=True)
        policy_or_stub._refine_lr_pending = True

    policy_or_stub._prev_stage = stage


class HFRVLAPolicy(SmolVLAPolicy):
    """Hierarchical Fast-Reactive VLA policy.

    Inherits everything from :class:`SmolVLAPolicy`. The slow VLA is frozen.
    A trainable :class:`FastReactiveModule` is composed alongside (not nested
    inside) and called at every control step.
    """

    config_class = HFRVLAConfig
    name = "hfrvla"

    def __init__(self, config: HFRVLAConfig, **kwargs):
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

        # ── 3. Fast reactive module ──
        # The hidden dim of z_goal / z_phase depends on SmolVLA; we discover
        # them lazily from a probe forward. The module itself defaults to the
        # SmolVLA text_config sizes so it can be instantiated up-front.
        text_hidden = int(self.model.vlm_with_expert.config.text_config.hidden_size)
        expert_hidden = int(self.model.vlm_with_expert.expert_hidden_size)
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

        self.fast = FastReactiveModule(
            config=config,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            zgoal_dim=text_hidden,
            zphase_dim=expert_hidden,
        )

        # ── 4. Curriculum state ──
        # The wrapper training script calls set_training_step(step) before
        # each optimizer step.
        self._cached_lambda_gate = float(config.loss_lambda_gate)
        self._cached_lambda_contact = float(config.loss_lambda_contact)
        self._prev_stage: int = -1
        self._refine_lr_pending: bool = False

        # ── 5. Inference state ──
        self._fast_hidden_state: Optional[Tensor] = None
        self._prev_action: Optional[Tensor] = None
        self._chunk_size: int = self.config.n_action_steps
        self._chunk_consumed: int = 0

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
        super().reset()
        self._fast_hidden_state = None
        self._prev_action = None
        self._chunk_consumed = 0
        self._clear_hook_cache()

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
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        is_new_chunk = self._check_get_actions_condition()
        if is_new_chunk:
            self._clear_hook_cache()
            actions = self._get_action_chunk(batch)
            self._queues[ACTION].extend(
                actions.transpose(0, 1)[: self.config.n_action_steps]
            )
            self._chunk_consumed = 0

        a_base = self._queues[ACTION].popleft()
        self._chunk_consumed += 1
        k = self._chunk_consumed - 1
        k_norm = torch.full(
            (a_base.shape[0], 1),
            k / max(1, self.config.n_action_steps - 1),
            device=a_base.device,
            dtype=a_base.dtype,
        )

        wrist_rgb = self._extract_wrist_image(batch)
        proprio = batch[OBS_STATE]

        if self._zgoal_cache is None or self._zphase_cache is None:
            # Hooks never fired (e.g. first call has empty queue but SmolVLA
            # forward path differed); fall back to a_base alone.
            return a_base

        fr_out: FastReactiveOutput = self.fast(
            wrist_rgb=wrist_rgb,
            proprio=proprio,
            a_base_k=a_base,
            k_idx_norm=k_norm,
            z_goal=self._zgoal_cache,
            z_phase=self._zphase_cache,
            hidden_state=self._fast_hidden_state,
        )
        self._fast_hidden_state = fr_out.hidden_state

        a_final = self._merge(
            a_base=a_base,
            delta_a=fr_out.delta_a,
            gate=fr_out.gate,
            prev_a=self._prev_action,
        )
        self._prev_action = a_final.detach()
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
                return batch[key]
        return None

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
        delta_clip = delta_a.clamp(-self.config.delta_max, self.config.delta_max)
        a_cand = a_base + gate.unsqueeze(-1) * delta_clip
        return self._enforce_velocity_limits(a_cand, prev_a)

    def _enforce_velocity_limits(
        self, a_cand: Tensor, prev_a: Optional[Tensor]
    ) -> Tensor:
        if prev_a is None:
            return a_cand
        max_step = self.config.safety_joint_velocity_limit * self.config.control_dt
        delta = (a_cand - prev_a).clamp(-max_step, max_step)
        return prev_a + delta

    # ────────────────────────────────────────────────────────────────────
    # Training
    # ────────────────────────────────────────────────────────────────────
    def forward(  # type: ignore[override]
        self, batch: dict[str, Tensor], **kwargs
    ) -> dict[str, Tensor]:
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
        z_goal = batch["observation.extra.z_goal"]
        z_phase = batch["observation.extra.z_phase"]
        dino_patches = batch["observation.extra.dino_patches"]
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
        out: FastReactiveOutput,
        a_base: Tensor,
        a_expert: Tensor,
        contact_label: Optional[Tensor],
    ) -> dict[str, Tensor]:
        # L_delta — MSE on residual
        target_delta = a_expert - a_base
        l_delta = F.mse_loss(out.delta_a, target_delta)

        # L_gate — G-C: stop_grad on delta_a, target = prediction improvement
        with torch.no_grad():
            delta_detached = out.delta_a.detach()
            err_before = ((a_expert - a_base) ** 2).sum(dim=-1)
            err_after = ((a_expert - (a_base + delta_detached)) ** 2).sum(dim=-1)
            g_target = torch.sigmoid(err_before - err_after)
        l_gate = F.binary_cross_entropy_with_logits(out.gate_logit, g_target)

        losses: dict[str, Tensor] = {"delta": l_delta, "gate": l_gate}

        # L_contact — optional aux head
        if out.contact_logit is not None and contact_label is not None:
            target = contact_label.float()
            # Broadcast if the contact head produced a sequence dim.
            if target.shape != out.contact_logit.shape:
                target = target.expand_as(out.contact_logit)
            l_contact = F.binary_cross_entropy_with_logits(out.contact_logit, target)
            losses["contact"] = l_contact

        total = l_delta + self.config.loss_lambda_gate * l_gate
        if "contact" in losses:
            total = total + self.config.loss_lambda_contact * losses["contact"]
        losses["loss"] = total
        return losses

    # ────────────────────────────────────────────────────────────────────
    def get_optim_params(self) -> list[nn.Parameter]:
        """Return only the trainable parameters of the fast reactive module."""
        return [p for p in self.fast.parameters() if p.requires_grad]
