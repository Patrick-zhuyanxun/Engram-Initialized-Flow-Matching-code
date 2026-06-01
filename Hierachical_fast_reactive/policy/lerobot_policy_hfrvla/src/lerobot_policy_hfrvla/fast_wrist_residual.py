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
    return bool(
        getattr(
            config,
            "fast_residual_use_latent_context",
            getattr(config, "a2c2_use_latent_context", True),
        )
    )


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
