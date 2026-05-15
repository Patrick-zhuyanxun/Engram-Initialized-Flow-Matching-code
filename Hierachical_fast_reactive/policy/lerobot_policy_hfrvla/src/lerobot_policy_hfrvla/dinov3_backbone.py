"""DINOv3 frozen backbone for wrist-camera dense features.

Supports two loading paths:

* **HuggingFace** ``AutoModel.from_pretrained(model_id)`` — requires gated access.
* **Local repo + ``.pth``** via ``torch.hub.load(local_repo, arch, source='local',
  weights=local_pth)`` — recommended (bypasses the gated repo).

Both paths return patch tokens shaped ``(B, N_patches, D)`` at 224×224 input.
For ViT-S/16 this is ``(B, 196, 384)``.

See implementation_spec.md §4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class DINOv3Backbone(nn.Module):
    """Wrap a DINOv3 ViT and expose patch features only (CLS / storage tokens dropped).

    Args:
        model_id: HuggingFace identifier, used only when ``local_repo`` is None.
        local_repo: Path to a local clone of facebookresearch/dinov3. When set,
            takes precedence and loads via ``torch.hub.load(source='local')``.
        local_weights: Path to a ``.pth`` state-dict file. Used together with
            ``local_repo``.
        arch: torch.hub entry-point name (``dinov3_vits16`` etc.).
        frozen: when True (default) parameters are frozen and forward runs
            under ``torch.no_grad()``.
    """

    def __init__(
        self,
        model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        local_repo: str | None = None,
        local_weights: str | None = None,
        arch: str = "dinov3_vits16",
        frozen: bool = True,
    ) -> None:
        super().__init__()
        self.frozen = frozen
        self.arch = arch
        self._source = "hf"

        if local_repo is not None:
            repo_path = Path(local_repo).expanduser().resolve()
            if not (repo_path / "hubconf.py").exists():
                raise FileNotFoundError(
                    f"Local DINOv3 repo missing hubconf.py at {repo_path}. "
                    f"Clone via: git clone https://github.com/facebookresearch/dinov3.git"
                )
            weights_arg: str | bool = True
            if local_weights is not None:
                w = Path(local_weights).expanduser().resolve()
                if not w.is_file():
                    raise FileNotFoundError(f"DINOv3 weights not found: {w}")
                weights_arg = str(w)

            self.model = torch.hub.load(
                str(repo_path),
                arch,
                source="local",
                weights=weights_arg,
                trust_repo=True,
            )
            self._source = "local"
            # Cache hidden dim from the loaded model.
            self._feature_dim = int(self.model.embed_dim)
        else:
            # HuggingFace path (requires gated access)
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(model_id)
            self._feature_dim = int(self.model.config.hidden_size)
            self._source = "hf"

        if frozen:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    # ────────────────────────────────────────────────────────────────────
    def train(self, mode: bool = True) -> "DINOv3Backbone":
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return patch tokens ``(B, N_patches, D)`` excluding CLS / storage tokens."""
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            if self._source == "local":
                # Meta's DINOv3 ViT returns a dict of normalized tokens.
                feats = self.model.forward_features(pixel_values)
                patches = feats["x_norm_patchtokens"]
            else:
                outputs = self.model(pixel_values=pixel_values)
                # HuggingFace ViTs typically return (B, 1+N, D) with CLS at 0.
                patches = outputs.last_hidden_state[:, 1:, :]
        return patches

    # ────────────────────────────────────────────────────────────────────
    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def count_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if (p.requires_grad or not trainable_only)
        )
