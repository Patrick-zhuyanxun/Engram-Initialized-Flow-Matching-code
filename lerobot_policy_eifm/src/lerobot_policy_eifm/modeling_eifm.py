#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team and 2toINF (https://github.com/2toINF)
# Copyright 2025 EI-FM authors — Engram-Initialized Flow Matching extensions
# Licensed under the Apache License, Version 2.0

"""
EI-FM (Engram-Initialized Flow Matching) policy for LeRobot.

Key differences from vanilla X-VLA:
  1. forward()  — mixed noise initialization: engram (prob p) or Gaussian (1-p)
  2. generate_actions() — engram replaces x1 = randn(...)
  3. w_engram   — learned linear: R^hidden_size → R^(chunk_size * dim_action)
  4. engram_table — {str: Tensor(hidden_size,)} loaded from .pt file
"""

from __future__ import annotations

import builtins
import logging
import os
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_STATE

from .action_hub import build_action_space
from .configuration_florence2 import Florence2Config
from .configuration_eifm import EIFMConfig
from .modeling_florence2 import Florence2ForConditionalGeneration
from .soft_transformer import SoftPromptedTransformer


class EIFMModel(nn.Module):
    """
    EI-FM backbone: Florence-2 + SoftPromptedTransformer + Engram Init.
    """

    def __init__(
        self,
        config: EIFMConfig,
        florence_config: Florence2Config,
        proprio_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.chunk_size: int = config.chunk_size
        self.use_proprio: bool = config.use_proprio

        # Build action space
        if config.action_mode.lower() == "auto":
            real_dim = (
                config.action_feature.shape[-1]
                if config.action_feature is not None
                else config.max_action_dim
            )
            self.action_space = build_action_space(
                config.action_mode.lower(),
                real_dim=real_dim,
                max_dim=config.max_action_dim,
            )
        else:
            self.action_space = build_action_space(config.action_mode.lower())

        self.dim_action = self.action_space.dim_action
        self.dim_proprio = proprio_dim

        # VLM backbone
        self.vlm = Florence2ForConditionalGeneration(florence_config)
        if hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "decoder"):
                del lm.model.decoder
            if hasattr(lm, "lm_head"):
                del lm.lm_head

        projection_dim = getattr(self.vlm.config, "projection_dim", None)
        if projection_dim is None:
            raise ValueError("Florence2 config must provide `projection_dim` for multimodal fusion.")

        # Policy transformer
        self.transformer = SoftPromptedTransformer(
            hidden_size=config.hidden_size,
            multi_modal_input_size=projection_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            dim_action=self.dim_action,
            dim_propio=self.dim_proprio,
            len_soft_prompts=config.len_soft_prompts,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
        )

        # ── EI-FM: Engram components ─────────────────────────
        self.engram_table: dict[str, Tensor] | None = None
        self._instruction_key_map: dict[str, str | None] = {}
        self._engram_global_mean: Tensor | None = None
        self.w_engram: nn.Linear | None = None

        if config.engram_path is not None:
            self._load_engram(config.engram_path)

        # Track current instructions for per-sample engram lookup.
        self._current_instructions: list[str | None] | None = None

        # Apply freezing and dtype
        self._apply_freezing()
        self._apply_dtype()

    def _load_engram(self, engram_path: str) -> None:
        """Load engram table and create w_engram projection layer."""
        logging.info(f"Loading engram table from {engram_path}")
        data = torch.load(engram_path, map_location="cpu", weights_only=False)

        self.engram_table = data["engram_table"]  # {str: Tensor(hidden_size,)}
        self._instruction_key_map = data.get("instruction_key_map", {})
        self._engram_global_mean = data.get("global_mean", None)

        engram_dim = data.get("dim", self.config.hidden_size)
        out_dim = self.chunk_size * self.dim_action

        # Learned projection: engram (hidden_size,) → action space (chunk_size * dim_action,)
        self.w_engram = nn.Linear(engram_dim, out_dim)
        # Initialize w_engram so initial output has reasonable scale
        nn.init.xavier_normal_(self.w_engram.weight)
        nn.init.zeros_(self.w_engram.bias)

        logging.info(
            f"  Engram table: {len(self.engram_table)} keys, dim={engram_dim}, "
            f"w_engram: {engram_dim} → {out_dim}"
        )

    def _get_target_dtype(self) -> torch.dtype:
        if self.config.dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32

    def _apply_dtype(self) -> None:
        target_dtype = self._get_target_dtype()
        self.to(dtype=target_dtype)

    def _apply_freezing(self) -> None:
        """
        Freeze VLM encoders; optionally keep action_encoder/decoder trainable for EI-FM.
        """
        # Freeze vision encoder (including projection layers)
        if self.config.freeze_vision_encoder:
            if hasattr(self.vlm, "vision_tower"):
                for param in self.vlm.vision_tower.parameters():
                    param.requires_grad = False
            # Also freeze image projection & pos embed (part of VLM, not policy)
            for attr_name in ("image_projection", "image_proj_norm", "image_pos_embed"):
                if hasattr(self.vlm, attr_name):
                    module = getattr(self.vlm, attr_name)
                    if hasattr(module, "parameters"):
                        for param in module.parameters():
                            param.requires_grad = False
                    elif isinstance(module, torch.nn.Parameter):
                        module.requires_grad = False

        # Freeze language encoder
        if self.config.freeze_language_encoder and hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "encoder"):
                for param in lm.model.encoder.parameters():
                    param.requires_grad = False
            if hasattr(lm, "model") and hasattr(lm.model, "shared"):
                for param in lm.model.shared.parameters():
                    param.requires_grad = False

        # Freeze or unfreeze policy transformer
        if not self.config.train_policy_transformer:
            for name, param in self.transformer.named_parameters():
                # Keep soft_prompt params if train_soft_prompts is True
                if "soft_prompt" not in name:
                    param.requires_grad = False

        # Freeze or unfreeze soft prompts
        if not self.config.train_soft_prompts:
            for name, param in self.transformer.named_parameters():
                if "soft_prompt" in name:
                    param.requires_grad = False

        # EI-FM: optionally keep action_encoder / action_decoder trainable
        if self.config.train_action_projections:
            for name, param in self.transformer.named_parameters():
                if "action_encoder" in name or "action_decoder" in name:
                    param.requires_grad = True

    def _get_engram_noise(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        """
        Retrieve per-sample engrams for the current batch, project to action space.

        Returns: (B, chunk_size, dim_action) tensor.
        Falls back to Gaussian noise on cache miss.
        """
        fallback = torch.randn(
            batch_size, self.chunk_size, self.dim_action,
            device=device, dtype=dtype,
        )

        if self.engram_table is None or self.w_engram is None:
            return fallback

        instructions = self._current_instructions
        if not instructions:
            return fallback

        if len(instructions) == 1 and batch_size > 1:
            instructions = instructions * batch_size
        elif len(instructions) != batch_size:
            logging.warning(
                "Instruction batch size mismatch for engram lookup: "
                f"got {len(instructions)} instructions for batch_size={batch_size}. "
                "Falling back to Gaussian noise."
            )
            return fallback

        valid_indices: list[int] = []
        engram_vectors: list[Tensor] = []
        for idx, instruction in enumerate(instructions):
            if instruction is None:
                continue
            key = self._instruction_key_map.get(instruction)
            if key is None or key not in self.engram_table:
                continue
            valid_indices.append(idx)
            engram_vectors.append(self.engram_table[key].to(device=device, dtype=dtype))

        if not engram_vectors:
            return fallback

        projected = self.w_engram(torch.stack(engram_vectors, dim=0))
        projected = projected.view(len(valid_indices), self.chunk_size, self.dim_action)
        fallback[valid_indices] = projected
        return fallback

    # ── VLM forward ───────────────────────────────────────────

    def forward_vlm(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_views = pixel_values.shape[:2]
        flat_mask = image_mask.view(-1).to(dtype=torch.bool)
        flat_images = pixel_values.flatten(0, 1)
        num_valid = int(flat_mask.sum().item())
        if num_valid == 0:
            raise ValueError("At least one image view must be valid per batch.")

        valid_images = flat_images[flat_mask]
        valid_feats = self.vlm._encode_image(valid_images)
        tokens_per_view, hidden_dim = valid_feats.shape[1:]

        image_features = valid_feats.new_zeros((batch_size * num_views, tokens_per_view, hidden_dim))
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(batch_size, num_views, tokens_per_view, hidden_dim)
        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)
        merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0],
            inputs_embeds,
        )

        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )[0]

        aux_visual_inputs = image_features[:, 1:].reshape(batch_size, -1, hidden_dim)
        return {"vlm_features": enc_out, "aux_visual_inputs": aux_visual_inputs}

    # ── Training forward ──────────────────────────────────────

    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target_dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)
        action = action.to(dtype=target_dtype)

        enc = self.forward_vlm(input_ids, image_input, image_mask)

        batch_size = input_ids.shape[0]
        t = (
            torch.rand(1, device=input_ids.device, dtype=target_dtype)
            + torch.arange(batch_size, device=input_ids.device, dtype=target_dtype) / batch_size
        ) % (1 - 1e-5)

        # ── EI-FM: Mixed noise initialization ─────────────────
        noise = torch.randn_like(action)

        if self.engram_table is not None and self.w_engram is not None:
            engram_noise = self._get_engram_noise(batch_size, action.device, action.dtype)
            # Per-sample Bernoulli mask: use engram with probability p_engram
            mask = (
                torch.rand(batch_size, 1, 1, device=action.device)
                < self.config.engram_p_engram
            )
            noise = torch.where(mask, engram_noise, noise)

        action_noisy = noise * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)

        pred_action = self.transformer(
            domain_id=domain_id,
            action_with_noise=action_noisy_m,
            t=t,
            proprio=proprio_m,
            **enc,
        )
        return self.action_space.compute_loss(pred_action, action)

    # ── Inference ─────────────────────────────────────────────

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        self.eval()

        target_dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)

        enc = self.forward_vlm(input_ids, image_input, image_mask)

        batch_size = input_ids.shape[0]
        action_dim = self.dim_action

        # ── EI-FM: Replace x1 = randn(...) with engram ───────
        if self.engram_table is not None and self.w_engram is not None:
            x1 = self._get_engram_noise(batch_size, proprio.device, target_dtype)
        else:
            x1 = torch.randn(
                batch_size, self.chunk_size, action_dim,
                device=proprio.device, dtype=target_dtype,
            )

        action = torch.zeros_like(x1)

        steps = max(1, int(steps))
        for i in range(steps, 0, -1):
            t = torch.full((batch_size,), i / steps, device=proprio.device, dtype=target_dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            action = self.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                **enc,
            )
        return self.action_space.postprocess(action).float()


class EIFMPolicy(PreTrainedPolicy):
    """LeRobot-compliant wrapper for the EI-FM model."""

    config_class = EIFMConfig
    name = "eifm"

    def __init__(self, config: EIFMConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        florence_config = config.get_florence_config()
        proprio_dim = config.max_state_dim if config.use_proprio else 0
        self.model = EIFMModel(config=config, florence_config=florence_config, proprio_dim=proprio_dim)
        self.reset()

    def reset(self) -> None:
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def get_optim_params(self) -> dict:
        return dict(filter(lambda kv: kv[1].requires_grad, self.named_parameters()))

    def _set_current_instruction(self, batch: dict[str, Tensor]) -> None:
        """Extract raw instruction text from batch and store per-sample instructions on the model."""
        instructions: list[str | None] | None = None
        for key in ("language_instruction", "task_description", "instruction", "task"):
            if key in batch:
                val = batch[key]
                if isinstance(val, (list, tuple)):
                    instructions = [item if isinstance(item, str) else None for item in val]
                elif isinstance(val, str):
                    instructions = [val]
                break
        self.model._current_instructions = instructions

    def _prepare_state(self, batch: dict[str, Tensor], batch_size: int, device: torch.device) -> Tensor:
        if not self.config.use_proprio or OBS_STATE not in batch:
            return torch.zeros(batch_size, 0, device=device)
        state = batch[OBS_STATE]
        if state.ndim > 2:
            state = state[:, -1, :]
        return pad_vector(state, self.model.dim_proprio)

    def _prepare_images(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        present_img_keys = [key for key in self.config.image_features if key in batch]
        if len(present_img_keys) == 0:
            raise ValueError(
                "All image features are missing from the batch. "
                f"Batch keys: {list(batch.keys())}, expected at least one of {list(self.config.image_features)}."
            )

        images = []
        masks = []
        for key in present_img_keys:
            img = batch[key][:, -1] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding)
            images.append(img)
            masks.append(torch.ones(img.size(0), dtype=torch.bool, device=img.device))

        stacked_imgs = torch.stack(images, dim=1)
        stacked_masks = torch.stack(masks, dim=1)

        total_views = self.config.num_image_views or stacked_imgs.size(1)
        total_views = max(total_views, stacked_imgs.size(1))
        num_pad = total_views - stacked_imgs.size(1)
        if num_pad > 0:
            pad_shape = (stacked_imgs.size(0), num_pad, *stacked_imgs.shape[2:])
            pad_imgs = stacked_imgs.new_zeros(pad_shape)
            pad_masks = stacked_masks.new_zeros((stacked_masks.size(0), num_pad))
            stacked_imgs = torch.cat([stacked_imgs, pad_imgs], dim=1)
            stacked_masks = torch.cat([stacked_masks, pad_masks], dim=1)

        return stacked_imgs, stacked_masks

    def _get_domain_id(self, batch: dict[str, Tensor], batch_size: int, device: torch.device) -> Tensor:
        candidate = None
        if self.config.domain_feature_key and self.config.domain_feature_key in batch:
            candidate = batch[self.config.domain_feature_key]
        elif "domain_id" in batch:
            candidate = batch["domain_id"]

        if candidate is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        if not isinstance(candidate, torch.Tensor):
            candidate = torch.as_tensor(candidate, device=device)
        else:
            candidate = candidate.to(device=device)

        if candidate.ndim == 0:
            candidate = candidate.expand(batch_size)
        if candidate.ndim > 1:
            candidate = candidate.view(candidate.shape[0], -1)[:, 0]
        if candidate.shape[0] != batch_size:
            candidate = candidate.expand(batch_size)
        return candidate.to(dtype=torch.long)

    def _prepare_action_targets(self, batch: dict[str, Tensor]) -> Tensor:
        if ACTION not in batch:
            raise ValueError("Batch is missing action targets required for training.")
        actions = batch[ACTION]
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        actions = pad_tensor_along_dim(actions, self.config.chunk_size, dim=1)
        if actions.shape[-1] != self.model.dim_action:
            actions = pad_vector(actions, self.model.dim_action)
        return actions

    def _build_model_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        input_ids = batch[OBS_LANGUAGE_TOKENS]
        batch_size = input_ids.shape[0]
        images, image_mask = self._prepare_images(batch)
        domain_id = self._get_domain_id(batch, batch_size, images.device)
        proprio = self._prepare_state(batch, batch_size, images.device)
        return {
            "input_ids": input_ids,
            "image_input": images,
            "image_mask": image_mask,
            "domain_id": domain_id,
            "proprio": proprio,
        }

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        self._set_current_instruction(batch)
        inputs = self._build_model_inputs(batch)
        targets = self._prepare_action_targets(batch)
        losses = self.model(action=targets, **inputs)
        total_loss = sum(losses.values())

        log_dict = {k: v.detach().item() for k, v in losses.items()}
        log_dict["loss"] = total_loss.detach().item()
        return total_loss, log_dict

    def _get_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self._set_current_instruction(batch)
        inputs = self._build_model_inputs(batch)
        actions = self.model.generate_actions(**inputs, steps=self.config.num_denoising_steps)
        return actions

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:  # noqa: ARG002
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        return self._get_action_chunk(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:  # noqa: ARG002
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft().float()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ):
        """
        Load EI-FM model weights from an X-VLA checkpoint.

        This adds the 'model.' prefix and handles EI-FM-specific keys
        (w_engram) that may not exist in the base checkpoint.
        """
        import safetensors.torch

        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)

        # Locate model.safetensors
        if os.path.isdir(model_id):
            logging.info("Loading weights from local directory")
            model_file = os.path.join(model_id, "model.safetensors")
        else:
            try:
                from huggingface_hub import hf_hub_download
                from huggingface_hub.utils import HfHubHTTPError

                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename="model.safetensors",
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(f"model.safetensors not found on the Hub at {model_id}") from e

        logging.info(f"Loading checkpoint from {model_file}")
        state_dict = safetensors.torch.load_file(model_file)

        # Fix shared embedding weight
        encoder_key = "model.vlm.language_model.model.encoder.embed_tokens.weight"
        shared_key = "model.vlm.language_model.model.shared.weight"
        if encoder_key in state_dict:
            state_dict[shared_key] = state_dict[encoder_key]

        # Load with strict=False to allow missing w_engram keys
        missing, unexpected = instance.load_state_dict(state_dict, strict=False)

        # Only w_engram keys should be missing (if engram_path is set)
        engram_missing = [k for k in missing if "w_engram" in k]
        other_missing = [k for k in missing if "w_engram" not in k]
        if other_missing:
            logging.warning(f"Missing non-engram keys: {other_missing}")
        if engram_missing:
            logging.info(f"w_engram keys initialized randomly (expected): {engram_missing}")

        logging.info("Loaded EI-FM checkpoint")

        instance.model._apply_dtype()
        instance.to(config.device)
        instance.eval()
        return instance


# ── Utility functions (same as X-VLA) ────────────────────────

def resize_with_pad(img: torch.Tensor, height: int, width: int, pad_value: float = 0.0) -> torch.Tensor:
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but got {img.shape}")

    current_height, current_width = img.shape[2:]
    if current_height == height and current_width == width:
        return img

    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, height - resized_height)
    pad_width = max(0, width - resized_width)
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    if vector.shape[-1] == new_dim:
        return vector
    if new_dim == 0:
        shape = list(vector.shape)
        shape[-1] = 0
        return vector.new_zeros(*shape)
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = vector.new_zeros(*shape)
    length = min(current_dim, new_dim)
    new_vector[..., :length] = vector[..., :length]
    return new_vector


def pad_tensor_along_dim(tensor: Tensor, target_len: int, dim: int = 1) -> Tensor:
    current_len = tensor.size(dim)
    if current_len == target_len:
        return tensor
    if current_len > target_len:
        slices = [slice(None)] * tensor.dim()
        slices[dim] = slice(0, target_len)
        return tensor[tuple(slices)]
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_len - current_len
    pad_tensor = tensor.new_zeros(pad_shape)
    return torch.cat([tensor, pad_tensor], dim=dim)
