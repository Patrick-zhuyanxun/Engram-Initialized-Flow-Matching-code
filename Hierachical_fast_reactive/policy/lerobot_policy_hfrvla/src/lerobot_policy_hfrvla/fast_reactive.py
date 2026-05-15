"""Fast Reactive Module — the trainable core of HFRVLA.

DINOv3 patch features → task-conditioned spatial pool (Strategy S-B) → GRU
→ three MLP heads (delta_a, gate, contact_aux).

See implementation_spec.md §5.
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
class FastReactiveOutput:
    """Per-step output of :class:`FastReactiveModule`.

    Attributes:
        delta_a:        ``(B, action_dim)`` residual to add to ``a_base``.
        gate_logit:     ``(B,)`` pre-sigmoid gate; fed to BCEWithLogits.
        gate:           ``(B,)`` ∈ [0, 1] = ``sigmoid(gate_logit)``.
        contact_logit:  ``(B,)`` or ``None`` (training-only auxiliary head).
        hidden_state:   GRU hidden state ``(num_layers, B, gru_hidden)``,
                        to be fed into the next call.
    """

    delta_a: torch.Tensor
    gate_logit: torch.Tensor
    gate: torch.Tensor
    contact_logit: Optional[torch.Tensor]
    hidden_state: torch.Tensor


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class FastReactiveModule(nn.Module):
    """The wrist-camera reactive controller.

    Trainable parameter budget: < 10 M (excluding the frozen DINOv3 backbone).
    The DINOv3 backbone is owned by this module but is excluded from
    :meth:`count_parameters` when ``trainable_only=True``.
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

        D_pool = config.pool_query_dim                       # 256
        D_dino = config.dinov3_feature_dim                   # 384

        # ── 1. DINOv3 backbone (frozen) ──
        self.backbone = DINOv3Backbone(
            model_id=config.dinov3_model_id,
            local_repo=getattr(config, "dinov3_local_repo", None),
            local_weights=getattr(config, "dinov3_local_weights", None),
            arch=getattr(config, "dinov3_arch", "dinov3_vits16"),
            frozen=config.dinov3_frozen,
        )

        # ── 2. Patch projection 384 → 256 ──
        self.patch_proj = nn.Linear(D_dino, D_pool)

        # ── 3. Query builder: [z_goal, z_phase] → (B, 1, 256) ──
        self.query_mlp = _mlp(zgoal_dim + zphase_dim, config.head_hidden * 2, D_pool)

        # ── 4. Cross-attention pool (Q from task, K/V from patches) ──
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=D_pool,
            num_heads=config.pool_n_heads,
            batch_first=True,
        )

        # ── 5. State fuser: pooled + proprio + a_base + k_idx + z_phase_proj → 256 ──
        # Project z_phase down to 256 once so the fuser input dim is fixed.
        self.zphase_proj = nn.Linear(zphase_dim, D_pool)
        fuse_in = D_pool + proprio_dim + action_dim + 1 + D_pool
        self.fuser = _mlp(fuse_in, D_pool, D_pool)

        # ── 6. GRU (carries short-term temporal context) ──
        self.gru = nn.GRU(
            input_size=D_pool,
            hidden_size=config.gru_hidden,
            num_layers=config.gru_layers,
            batch_first=True,
        )

        # ── 7. Three small MLP heads ──
        self.delta_head = _mlp(config.gru_hidden, config.head_hidden, action_dim)
        self.gate_head = _mlp(config.gru_hidden, config.head_hidden, 1)
        if config.contact_head_enabled:
            self.contact_head = _mlp(config.gru_hidden, config.head_hidden, 1)
        else:
            self.contact_head = None

    # ────────────────────────────────────────────────────────────────────
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            self.config.gru_layers, batch_size, self.config.gru_hidden, device=device
        )

    def forward(
        self,
        wrist_rgb: Optional[torch.Tensor],
        proprio: torch.Tensor,
        a_base_k: torch.Tensor,
        k_idx_norm: torch.Tensor,
        z_goal: torch.Tensor,
        z_phase: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        *,
        dino_patches: Optional[torch.Tensor] = None,
    ) -> FastReactiveOutput:
        """Run one (or a temporal sequence of) fast-reactive steps.

        Either ``wrist_rgb`` (image input) **or** ``dino_patches`` (cached
        backbone output) must be provided. The training pipeline pre-computes
        ``dino_patches`` for speed.

        Shapes:
            wrist_rgb:    ``(B, 3, H, W)`` — single step; or ``(B, T, 3, H, W)``
                          for a sequence.
            dino_patches: ``(B, N_patches, D_dino)`` or ``(B, T, N_patches, D_dino)``.
            proprio:      ``(B, proprio_dim)`` or ``(B, T, proprio_dim)``.
            a_base_k:     ``(B, action_dim)`` or ``(B, T, action_dim)``.
            k_idx_norm:   ``(B, 1)``       or ``(B, T, 1)``.
            z_goal:       ``(B, zgoal_dim)`` or ``(B, T, zgoal_dim)``.
            z_phase:      ``(B, zphase_dim)`` or ``(B, T, zphase_dim)``.
        """
        # Detect single-step vs sequence by proprio's dim count.
        if proprio.dim() == 2:
            return self._forward_single(
                wrist_rgb, proprio, a_base_k, k_idx_norm, z_goal, z_phase,
                hidden_state, dino_patches=dino_patches,
            )
        elif proprio.dim() == 3:
            return self._forward_sequence(
                wrist_rgb, proprio, a_base_k, k_idx_norm, z_goal, z_phase,
                hidden_state, dino_patches=dino_patches,
            )
        else:
            raise ValueError(
                f"proprio must be 2-D (single step) or 3-D (sequence), got "
                f"{proprio.dim()}-D"
            )

    # ────────────────────────────────────────────────────────────────────
    def _encode_step(
        self,
        wrist_rgb: Optional[torch.Tensor],   # (B, 3, H, W) or None
        proprio: torch.Tensor,                # (B, D)
        a_base_k: torch.Tensor,               # (B, A)
        k_idx_norm: torch.Tensor,             # (B, 1)
        z_goal: torch.Tensor,                 # (B, Zg)
        z_phase: torch.Tensor,                # (B, Zp)
        dino_patches: Optional[torch.Tensor], # (B, N, D_dino) or None
    ) -> torch.Tensor:
        """Encode a single time-step into a fused 256-d state vector."""
        # 1. Patch features
        if dino_patches is None:
            assert wrist_rgb is not None, "Either wrist_rgb or dino_patches required."
            patches = self.backbone(wrist_rgb)
        else:
            patches = dino_patches
        patches_proj = self.patch_proj(patches)                # (B, N, D_pool)

        # 2. Task-conditioned query
        query = self.query_mlp(torch.cat([z_goal, z_phase], dim=-1))  # (B, D_pool)
        query = query.unsqueeze(1)                              # (B, 1, D_pool)

        # 3. Cross-attention pool: Q=task, K=V=patches
        pooled, _ = self.cross_attn(query, patches_proj, patches_proj)
        pooled = pooled.squeeze(1)                              # (B, D_pool)

        # 4. State fusion
        zphase_p = self.zphase_proj(z_phase)
        fused_in = torch.cat([pooled, proprio, a_base_k, k_idx_norm, zphase_p], dim=-1)
        state = self.fuser(fused_in)                            # (B, D_pool)
        return state

    def _heads(self, h: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
    ]:
        delta_a = self.delta_head(h)
        gate_logit = self.gate_head(h).squeeze(-1)
        gate = torch.sigmoid(gate_logit)
        contact_logit = (
            self.contact_head(h).squeeze(-1) if self.contact_head is not None else None
        )
        return delta_a, gate_logit, gate, contact_logit

    def _forward_single(
        self,
        wrist_rgb, proprio, a_base_k, k_idx_norm, z_goal, z_phase,
        hidden_state, *, dino_patches,
    ) -> FastReactiveOutput:
        B = proprio.shape[0]
        device = proprio.device
        if hidden_state is None:
            hidden_state = self.init_hidden(B, device)

        state = self._encode_step(
            wrist_rgb, proprio, a_base_k, k_idx_norm, z_goal, z_phase, dino_patches
        )
        # GRU expects (B, T, D). T=1 here.
        out, h_new = self.gru(state.unsqueeze(1), hidden_state)
        h = out.squeeze(1)
        delta_a, gate_logit, gate, contact_logit = self._heads(h)
        return FastReactiveOutput(
            delta_a=delta_a,
            gate_logit=gate_logit,
            gate=gate,
            contact_logit=contact_logit,
            hidden_state=h_new,
        )

    def _forward_sequence(
        self,
        wrist_rgb, proprio, a_base_k, k_idx_norm, z_goal, z_phase,
        hidden_state, *, dino_patches,
    ) -> FastReactiveOutput:
        B, T = proprio.shape[0], proprio.shape[1]
        device = proprio.device
        if hidden_state is None:
            hidden_state = self.init_hidden(B, device)

        # Encode each step.
        states = []
        for t in range(T):
            w_t = wrist_rgb[:, t] if wrist_rgb is not None else None
            d_t = dino_patches[:, t] if dino_patches is not None else None
            s_t = self._encode_step(
                w_t,
                proprio[:, t],
                a_base_k[:, t],
                k_idx_norm[:, t],
                z_goal[:, t],
                z_phase[:, t],
                d_t,
            )
            states.append(s_t)
        states = torch.stack(states, dim=1)                     # (B, T, D_pool)

        out, h_new = self.gru(states, hidden_state)             # (B, T, hidden)
        # Flatten time × batch for the heads.
        h_flat = out.reshape(B * T, -1)
        delta_a, gate_logit, gate, contact_logit = self._heads(h_flat)
        delta_a = delta_a.reshape(B, T, -1)
        gate_logit = gate_logit.reshape(B, T)
        gate = gate.reshape(B, T)
        if contact_logit is not None:
            contact_logit = contact_logit.reshape(B, T)
        return FastReactiveOutput(
            delta_a=delta_a,
            gate_logit=gate_logit,
            gate=gate,
            contact_logit=contact_logit,
            hidden_state=h_new,
        )

    # ────────────────────────────────────────────────────────────────────
    def count_parameters(self, exclude_backbone: bool = True) -> int:
        """Return the number of trainable parameters.

        By default the DINOv3 backbone is excluded (it is frozen and not
        counted against the < 10 M budget).
        """
        n = 0
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if exclude_backbone and name.startswith("backbone."):
                continue
            n += p.numel()
        return n
