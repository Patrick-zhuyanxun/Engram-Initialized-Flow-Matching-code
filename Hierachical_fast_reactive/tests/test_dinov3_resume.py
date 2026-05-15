from __future__ import annotations

import sys
from pathlib import Path

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


from legacy.data import EpisodeRecord, has_complete_dino_patches  # noqa: E402


def _record(*, length: int, dino_length: int | None) -> EpisodeRecord:
    dino = None
    if dino_length is not None:
        dino = torch.zeros(dino_length, 196, 384)
    return EpisodeRecord(
        wrist_rgb=torch.zeros(length, 3, 224, 224),
        proprio=torch.zeros(length, 8),
        a_base=torch.zeros(length, 7),
        k_idx_norm=torch.zeros(length, 1),
        z_goal=torch.zeros(length, 1),
        z_phase=torch.zeros(length, 1),
        a_expert=torch.zeros(length, 7),
        contact_label=torch.zeros(length),
        dino_patches=dino,
    )


def test_has_complete_dino_patches_requires_matching_time_dim() -> None:
    assert has_complete_dino_patches(_record(length=5, dino_length=5)) is True
    assert has_complete_dino_patches(_record(length=5, dino_length=4)) is False
    assert has_complete_dino_patches(_record(length=5, dino_length=None)) is False
