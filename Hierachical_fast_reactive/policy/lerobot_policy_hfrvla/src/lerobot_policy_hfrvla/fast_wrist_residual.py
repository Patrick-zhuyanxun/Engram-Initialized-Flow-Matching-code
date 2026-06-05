"""Fast Wrist Residual head for frozen SmolVLA chunks.

This module is intentionally stateless: no GRU, no gate, no contact head. It
predicts a raw 7D residual from the latest wrist features plus base-policy
context and one-step previous-action conditioning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig
from lerobot_policy_hfrvla.dinov3_backbone import DINOv3Backbone


@dataclass
class FastWristResidualOutput:
    """Per-step output of :class:`FastWristResidualModule`."""

    delta_a: torch.Tensor


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


def _use_latent_context(config: HFRVLAConfig) -> bool:
    value = getattr(config, "fast_residual_use_latent_context", None)
    if value is None:
        value = getattr(config, "a2c2_use_latent_context", True)
    return bool(value)


class FastWristResidualModule(nn.Module):
    """Feed-forward wrist-conditioned residual correction head.

    Inputs mirror the existing HFRVLA cache where possible, with explicit
    previous-step features supplied by the policy/loss branch:
    ``prev_a_base`` and ``prev_delta``.
    """

    def __init__(
        self,
        config: HFRVLAConfig,
        action_dim: int,
        proprio_dim: int,
        zgoal_dim: int,
        zphase_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.zgoal_dim = zgoal_dim
        self.zphase_dim = zphase_dim

        d_pool = config.pool_query_dim
        d_dino = config.dinov3_feature_dim

        if getattr(config, "offline_training_mode", False):
            self.backbone = None
        else:
            self.backbone = DINOv3Backbone(
                model_id=config.dinov3_model_id,
                local_repo=getattr(config, "dinov3_local_repo", None),
                local_weights=getattr(config, "dinov3_local_weights", None),
                arch=getattr(config, "dinov3_arch", "dinov3_vits16"),
                frozen=config.dinov3_frozen,
            )

        self.patch_proj = nn.Linear(d_dino, d_pool)
        self.query_mlp = _mlp(zgoal_dim + zphase_dim, config.head_hidden * 2, d_pool)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_pool,
            num_heads=config.pool_n_heads,
            batch_first=True,
        )
        self.zphase_proj = nn.Linear(zphase_dim, d_pool)

        fuse_in = d_pool + proprio_dim + action_dim + 1 + d_pool + action_dim + action_dim
        self.fuser = nn.Sequential(
            nn.Linear(fuse_in, d_pool),
            nn.GELU(),
            nn.Linear(d_pool, d_pool),
            nn.GELU(),
            nn.Linear(d_pool, action_dim),
        )

    def _encode_wrist(
        self,
        wrist_rgb: Optional[torch.Tensor],
        z_goal: torch.Tensor,
        z_phase: torch.Tensor,
        dino_patches: Optional[torch.Tensor],
    ) -> torch.Tensor:
        module_dtype = self.patch_proj.weight.dtype
        z_goal = z_goal.to(dtype=module_dtype)
        z_phase = z_phase.to(dtype=module_dtype)

        if dino_patches is None:
            assert wrist_rgb is not None, "Either wrist_rgb or dino_patches required."
            if self.backbone is None:
                raise RuntimeError(
                    "FastWristResidualModule was initialized in offline_training_mode; "
                    "pass precomputed observation.extra.dino_patches or disable "
                    "offline_training_mode for inference."
                )
            patches = self.backbone(wrist_rgb)
        else:
            patches = dino_patches

        patches = patches.to(dtype=module_dtype)
        patches_proj = self.patch_proj(patches)

        if _use_latent_context(self.config):
            query_in = torch.cat([z_goal, z_phase], dim=-1)
        else:
            query_in = torch.zeros(
                z_goal.shape[0],
                self.zgoal_dim + self.zphase_dim,
                device=z_goal.device,
                dtype=module_dtype,
            )
        query = self.query_mlp(query_in).unsqueeze(1)
        pooled, _ = self.cross_attn(query, patches_proj, patches_proj)
        return pooled.squeeze(1)

    def forward(
        self,
        wrist_rgb: Optional[torch.Tensor],
        proprio: torch.Tensor,
        a_base_k: torch.Tensor,
        k_idx_norm: torch.Tensor,
        z_goal: torch.Tensor,
        z_phase: torch.Tensor,
        prev_a_base: torch.Tensor,
        prev_delta: torch.Tensor,
        *,
        dino_patches: Optional[torch.Tensor] = None,
    ) -> FastWristResidualOutput:
        """Run one feed-forward correction step.

        All tensors are single-step tensors with leading batch dimension.
        """
        if proprio.dim() != 2:
            raise ValueError(
                "FastWristResidualModule expects single-step tensors; "
                f"got proprio dim={proprio.dim()}"
            )

        module_dtype = self.patch_proj.weight.dtype
        pooled = self._encode_wrist(wrist_rgb, z_goal, z_phase, dino_patches)
        proprio = proprio.to(dtype=module_dtype)
        a_base_k = a_base_k.to(dtype=module_dtype)
        k_idx_norm = k_idx_norm.to(dtype=module_dtype)
        z_phase = z_phase.to(dtype=module_dtype)
        prev_a_base = prev_a_base.to(dtype=module_dtype)
        prev_delta = prev_delta.to(dtype=module_dtype)

        if _use_latent_context(self.config):
            zphase_p = self.zphase_proj(z_phase)
        else:
            zphase_p = torch.zeros(
                z_phase.shape[0],
                self.config.pool_query_dim,
                device=z_phase.device,
                dtype=module_dtype,
            )
        fused = torch.cat(
            [pooled, proprio, a_base_k, k_idx_norm, zphase_p, prev_a_base, prev_delta],
            dim=-1,
        )
        return FastWristResidualOutput(delta_a=self.fuser(fused))

    def count_parameters(self, exclude_backbone: bool = True) -> int:
        """Return the number of trainable parameters, excluding frozen DINO by default."""
        n = 0
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if exclude_backbone and name.startswith("backbone."):
                continue
            n += p.numel()
        return n


class FastWristChunkResidualModule(FastWristResidualModule):
    """Chunk-aware FWR-v2 head.

    The wrist encoder is shared with FWR-v1, but this head attends over all
    action tokens in the frozen SmolVLA planning chunk before predicting the
    current-step residual.
    """

    def __init__(
        self,
        config: HFRVLAConfig,
        action_dim: int,
        proprio_dim: int,
        zgoal_dim: int,
        zphase_dim: int,
    ) -> None:
        super().__init__(
            config=config,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            zgoal_dim=zgoal_dim,
            zphase_dim=zphase_dim,
        )
        d_pool = config.pool_query_dim
        token_in = action_dim + 4
        self.chunk_token_proj = nn.Sequential(
            nn.Linear(token_in, d_pool),
            nn.GELU(),
            nn.Linear(d_pool, d_pool),
        )
        self.chunk_query_proj = _mlp(
            d_pool + proprio_dim + action_dim + 1 + d_pool,
            d_pool,
            d_pool,
        )
        self.chunk_attn = nn.MultiheadAttention(
            embed_dim=d_pool,
            num_heads=config.pool_n_heads,
            batch_first=True,
        )
        fuse_in = d_pool + proprio_dim + action_dim + 1 + d_pool + d_pool
        self.fuser = nn.Sequential(
            nn.Linear(fuse_in, d_pool),
            nn.GELU(),
            nn.Linear(d_pool, d_pool),
            nn.GELU(),
            nn.Linear(d_pool, action_dim),
        )

    def _chunk_tokens(
        self,
        a_base_chunk: torch.Tensor,
        a_base_k: torch.Tensor,
        chunk_step_idx: torch.Tensor,
    ) -> torch.Tensor:
        if a_base_chunk.dim() != 3:
            raise ValueError(
                "FastWristChunkResidualModule expects a_base_chunk shape "
                f"(B, K, action_dim), got {tuple(a_base_chunk.shape)}"
            )
        if a_base_chunk.shape[-1] != self.action_dim:
            raise ValueError(
                f"a_base_chunk action dim {a_base_chunk.shape[-1]} != {self.action_dim}"
            )

        module_dtype = self.patch_proj.weight.dtype
        chunk = a_base_chunk.to(dtype=module_dtype)
        current = a_base_k.to(dtype=module_dtype)
        batch, chunk_len, _ = chunk.shape
        denom = max(1, chunk_len - 1)

        step = chunk_step_idx.reshape(batch, -1)[:, 0].to(device=chunk.device)
        step_float = step.to(dtype=module_dtype).clamp(0, chunk_len - 1)
        j = torch.arange(chunk_len, device=chunk.device, dtype=module_dtype).view(1, chunk_len, 1)
        j_norm = j / denom
        rel = (j - step_float.view(batch, 1, 1)) / denom
        abs_rel = rel.abs()
        cos = F.cosine_similarity(
            chunk,
            current.unsqueeze(1).expand_as(chunk),
            dim=-1,
            eps=1e-6,
        ).unsqueeze(-1)
        token_features = torch.cat(
            [
                chunk,
                j_norm.expand(batch, -1, -1),
                rel,
                abs_rel,
                cos.to(dtype=module_dtype),
            ],
            dim=-1,
        )
        return self.chunk_token_proj(token_features)

    def forward(
        self,
        wrist_rgb: Optional[torch.Tensor],
        proprio: torch.Tensor,
        a_base_k: torch.Tensor,
        k_idx_norm: torch.Tensor,
        z_goal: torch.Tensor,
        z_phase: torch.Tensor,
        a_base_chunk: torch.Tensor,
        chunk_step_idx: torch.Tensor,
        *,
        dino_patches: Optional[torch.Tensor] = None,
    ) -> FastWristResidualOutput:
        """Run current-step residual prediction conditioned on the full chunk."""
        if proprio.dim() != 2:
            raise ValueError(
                "FastWristChunkResidualModule expects single-step tensors; "
                f"got proprio dim={proprio.dim()}"
            )

        module_dtype = self.patch_proj.weight.dtype
        pooled = self._encode_wrist(wrist_rgb, z_goal, z_phase, dino_patches)
        proprio = proprio.to(dtype=module_dtype)
        a_base_k = a_base_k.to(dtype=module_dtype)
        k_idx_norm = k_idx_norm.to(dtype=module_dtype)
        z_phase = z_phase.to(dtype=module_dtype)

        if _use_latent_context(self.config):
            zphase_p = self.zphase_proj(z_phase)
        else:
            zphase_p = torch.zeros(
                z_phase.shape[0],
                self.config.pool_query_dim,
                device=z_phase.device,
                dtype=module_dtype,
            )

        tokens = self._chunk_tokens(a_base_chunk, a_base_k, chunk_step_idx)
        query_in = torch.cat([pooled, proprio, a_base_k, k_idx_norm, zphase_p], dim=-1)
        query = self.chunk_query_proj(query_in).unsqueeze(1)
        chunk_ctx, _ = self.chunk_attn(query, tokens, tokens)
        chunk_ctx = chunk_ctx.squeeze(1)
        fused = torch.cat(
            [pooled, proprio, a_base_k, k_idx_norm, zphase_p, chunk_ctx],
            dim=-1,
        )
        return FastWristResidualOutput(delta_a=self.fuser(fused))
