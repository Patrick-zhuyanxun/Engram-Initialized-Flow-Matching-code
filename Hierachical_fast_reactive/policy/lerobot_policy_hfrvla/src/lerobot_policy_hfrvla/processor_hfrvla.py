"""Processor for HFRVLA — extends SmolVLA's preprocessing with DINOv3 norm.

If a batch already carries pre-computed ``dino_patches`` (as produced by
:mod:`lerobot_policy_hfrvla.data`), the wrist image normalization step is
skipped to save compute. See implementation_spec.md §8.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


# DINOv3 normalization stats (LVD-1689M pretraining). These match the values
# the HuggingFace pre-processor would apply.
DINOV3_MEAN = (0.485, 0.456, 0.406)
DINOV3_STD = (0.229, 0.224, 0.225)


def normalize_for_dinov3(
    image: torch.Tensor,
    size: int = 224,
) -> torch.Tensor:
    """Resize to ``(size, size)`` and normalize to DINOv3 stats.

    Args:
        image: ``(B, 3, H, W)`` or ``(3, H, W)``, values in ``[0, 1]``.
        size:  spatial dim after resize (default 224).
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.shape[-1] != size or image.shape[-2] != size:
        image = F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False)
    mean = image.new_tensor(DINOV3_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(DINOV3_STD).view(1, 3, 1, 1)
    return (image - mean) / std


def prepare_fast_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    """Best-effort prep of a batch for the Fast Reactive Module.

    Inserts ``wrist_rgb`` (normalized for DINOv3) when an image is present
    and ``dino_patches`` is not already cached.
    """
    if "dino_patches" in batch:
        return batch
    # Find wrist image under any of the common dataset keys.
    for key in (
        "observation.images.wrist",
        "observation.images.wrist_image",
        "observation.images.image_wrist",
        "wrist_image",
        "wrist_rgb_raw",
    ):
        if key in batch:
            batch["wrist_rgb"] = normalize_for_dinov3(batch[key])
            break
    return batch
