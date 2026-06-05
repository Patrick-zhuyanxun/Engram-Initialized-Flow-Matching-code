from io import BytesIO

import numpy as np
import pandas as pd
import torch
from PIL import Image

from scripts.record_hfrvla_libero import (
    _LocalParquetEpisodeSource,
    _compute_episode_stats_basic_fallback,
    _resolve_episode_indices,
)


def test_recording_stats_fallback_handles_constant_generated_chunks():
    episode_data = {
        "observation.extra.a_base_chunk": np.ones((3, 50, 7), dtype=np.float32),
        "observation.extra.chunk_step_idx": np.zeros((3, 1), dtype=np.int64),
        "observation.extra.chunk_age_norm": np.zeros((3, 1), dtype=np.float32),
    }
    features = {
        "observation.extra.a_base_chunk": {
            "dtype": "float32",
            "shape": (50, 7),
        },
        "observation.extra.chunk_step_idx": {
            "dtype": "int64",
            "shape": (1,),
        },
        "observation.extra.chunk_age_norm": {
            "dtype": "float32",
            "shape": (1,),
        },
    }

    stats = _compute_episode_stats_basic_fallback(episode_data, features)

    assert stats["observation.extra.a_base_chunk"]["mean"].shape == (50, 7)
    assert np.allclose(stats["observation.extra.a_base_chunk"]["mean"], 1.0)
    assert np.allclose(stats["observation.extra.chunk_age_norm"]["max"], 0.0)


def test_resolve_episode_indices_applies_range_before_max_episodes():
    assert _resolve_episode_indices(
        total_episodes=100,
        ep_from=20,
        ep_to=50,
        max_episodes=7,
    ) == list(range(20, 27))


def test_resolve_episode_indices_rejects_empty_range():
    try:
        _resolve_episode_indices(total_episodes=10, ep_from=10, ep_to=None, max_episodes=None)
    except ValueError as exc:
        assert "empty range" in str(exc)
    else:
        raise AssertionError("expected empty range to fail")


def _png_bytes(value: int) -> bytes:
    image = Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8), mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_local_parquet_episode_source_streams_lerobot_like_samples(tmp_path):
    root = tmp_path / "source"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)

    (root / "meta/info.json").write_text(
        '{"total_episodes": 1, "total_frames": 3, "features": {}}'
    )
    (root / "meta/stats.json").write_text(
        '{"observation.state": {"mean": [0, 0], "std": [1, 1]}, '
        '"action": {"q01": [0, 0], "q99": [1, 1]}}'
    )
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["pick up the mug"], name="task")).to_parquet(
        root / "meta/tasks.parquet"
    )
    pd.DataFrame(
        {
            "episode_index": [0],
            "data/chunk_index": [0],
            "data/file_index": [0],
            "dataset_from_index": [1],
            "dataset_to_index": [3],
            "tasks": [["pick up the mug"]],
        }
    ).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet", index=False)
    pd.DataFrame(
        {
            "observation.images.image": [{"bytes": _png_bytes(1), "path": None}],
            "observation.images.image2": [{"bytes": _png_bytes(2), "path": None}],
            "observation.state": [[9.0, 9.0]],
            "action": [[9.0, 9.0]],
            "timestamp": [0.0],
            "frame_index": [0],
            "episode_index": [99],
            "index": [0],
            "task_index": [0],
        }
    ).to_parquet(root / "data/chunk-000/file-000.parquet", index=False)
    pd.DataFrame(
        {
            "observation.images.image": [{"bytes": _png_bytes(64), "path": None}, {"bytes": _png_bytes(96), "path": None}],
            "observation.images.image2": [{"bytes": _png_bytes(128), "path": None}, {"bytes": _png_bytes(160), "path": None}],
            "observation.state": [[0.1, 0.2], [0.3, 0.4]],
            "action": [[0.5, 0.6], [0.7, 0.8]],
            "timestamp": [0.0, 0.1],
            "frame_index": [0, 1],
            "episode_index": [0, 0],
            "index": [1, 2],
            "task_index": [0, 0],
        }
    ).to_parquet(root / "data/chunk-000/file-001.parquet", index=False)

    source = _LocalParquetEpisodeSource(root)

    assert source.num_episodes == 1
    assert source.episode_bounds(0) == (1, 3, "pick up the mug")
    assert source.meta.stats["action"]["q99"].shape == (2,)

    sample = source[2]

    assert sample["task"] == "pick up the mug"
    assert tuple(sample["observation.images.image"].shape) == (3, 4, 4)
    assert sample["observation.images.image"].dtype == torch.float32
    assert np.isclose(float(sample["observation.images.image"].max()), 96 / 255)
    assert tuple(sample["observation.state"].shape) == (2,)
    assert np.allclose(sample["action"].numpy(), [0.7, 0.8])
