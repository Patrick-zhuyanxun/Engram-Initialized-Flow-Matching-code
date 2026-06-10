import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.visualize_dino_cross_attention import (
    compute_cross_attention_maps,
    dino_patch_norm_heatmap,
    iter_strided_rows,
    load_fast_module_state,
    render_comparison_frame,
    render_three_panel_frame,
    write_video,
)
from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig
from lerobot_policy_hfrvla.fast_wrist_residual import FastWristChunkResidualModule


def test_load_fast_module_state_strips_fast_prefix_and_skips_backbone(tmp_path):
    checkpoint = tmp_path / "policy"
    checkpoint.mkdir()
    save_file(
        {
            "fast.patch_proj.weight": torch.ones(2, 3),
            "fast.patch_proj.bias": torch.zeros(2),
            "fast.backbone.model.blocks.0.attn.qkv.weight": torch.full((1,), 9.0),
            "model.vlm_with_expert.weight": torch.full((1,), 7.0),
        },
        checkpoint / "model.safetensors",
    )

    state = load_fast_module_state(checkpoint)

    assert set(state) == {"patch_proj.weight", "patch_proj.bias"}
    assert torch.equal(state["patch_proj.weight"], torch.ones(2, 3))


def test_dino_patch_norm_heatmap_returns_normalized_14x14_map():
    patches = torch.zeros(196, 384)
    patches[0, 0] = 1.0
    patches[-1, 0] = 3.0

    heatmap = dino_patch_norm_heatmap(patches)

    assert heatmap.shape == (14, 14)
    assert heatmap[0, 0] == pytest.approx(1 / 3)
    assert heatmap[-1, -1] == pytest.approx(1.0)
    assert float(heatmap.min()) == pytest.approx(0.0)


def test_compute_cross_attention_maps_returns_per_head_and_mean_maps():
    torch.manual_seed(0)
    cfg = HFRVLAConfig(
        residual_merge_mode="fast_wrist_chunk",
        offline_training_mode=True,
        pool_n_heads=4,
    )
    module = FastWristChunkResidualModule(
        config=cfg,
        action_dim=7,
        proprio_dim=8,
        zgoal_dim=960,
        zphase_dim=480,
    )

    maps = compute_cross_attention_maps(
        module,
        dino_patches=torch.randn(196, 384),
        z_goal=torch.randn(960),
        z_phase=torch.randn(480),
    )

    assert maps.per_head.shape == (4, 14, 14)
    assert maps.mean.shape == (14, 14)
    assert maps.entropy >= 0.0
    assert np.isfinite(maps.per_head).all()
    assert np.isfinite(maps.mean).all()


def test_render_three_panel_frame_and_write_video(tmp_path):
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[:, :, 1] = 120
    before = np.eye(14, dtype=np.float32)
    after = np.flipud(before)

    frame = render_three_panel_frame(
        wrist_rgb=rgb,
        before_heatmap=before,
        after_heatmap=after,
        title="task 8 | episode 14 | frame 3",
    )

    assert frame.dtype == np.uint8
    assert frame.ndim == 3
    assert frame.shape[2] == 3

    out_path = tmp_path / "episode.mp4"
    write_video([frame, frame], out_path, fps=10)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_render_comparison_frame_adds_vla_and_wrist_inputs():
    vla_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    vla_rgb[:, :, 0] = 180
    wrist_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    wrist_rgb[:, :, 1] = 120
    before = np.eye(14, dtype=np.float32)
    after = np.flipud(before)

    frame = render_comparison_frame(
        vla_rgb=vla_rgb,
        wrist_rgb=wrist_rgb,
        before_heatmap=before,
        after_heatmap=after,
        title="task 8 | episode 14 | frame 3",
        panel_size=64,
    )

    assert frame.dtype == np.uint8
    assert frame.shape == (176, 336, 3)


def test_iter_strided_rows_uses_positional_rows_without_iterrows(monkeypatch):
    import pandas as pd

    df = pd.DataFrame({"value": [0, 1, 2, 3, 4]})

    def _fail_iterrows(self):
        raise AssertionError("iterrows should not be used for nested parquet frames")

    monkeypatch.setattr(pd.DataFrame, "iterrows", _fail_iterrows)

    rows = list(iter_strided_rows(df, stride=2))

    assert [int(row["value"]) for row in rows] == [0, 2, 4]
