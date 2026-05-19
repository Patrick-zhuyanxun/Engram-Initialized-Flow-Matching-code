import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot_policy_hfrvla.fast_cache_dataset import (
    HFRVLAFastCacheDataset,
    load_fast_cache_metadata,
)


def _write_cache(root: Path) -> None:
    root.mkdir()
    source_root = root / "source"
    source_root.mkdir()
    (source_root / "meta").mkdir()
    (source_root / "meta" / "stats.json").write_text(
        json.dumps(
            {
                "observation.state": {"mean": [0.0] * 8, "std": [1.0] * 8},
                "action": {"mean": [0.0] * 7, "std": [1.0] * 7},
            }
        )
    )
    meta = {
        "schema_version": 1,
        "source_dataset_root": str(source_root),
        "source_info_sha256": "fixture",
        "total_frames": 6,
        "total_episodes": 2,
        "fps": 10,
        "seq_len": 3,
        "features": {
            "observation.state": {
                "array": "state.npy",
                "dtype": "float32",
                "shape": [6, 8],
            },
            "action": {"array": "action.npy", "dtype": "float32", "shape": [6, 7]},
            "observation.extra.a_base": {
                "array": "a_base.npy",
                "dtype": "float32",
                "shape": [6, 7],
            },
            "observation.extra.k_idx_norm": {
                "array": "k_idx_norm.npy",
                "dtype": "float32",
                "shape": [6, 1],
            },
            "observation.extra.z_goal": {
                "array": "z_goal.npy",
                "dtype": "float16",
                "shape": [6, 960],
            },
            "observation.extra.z_phase": {
                "array": "z_phase.npy",
                "dtype": "float16",
                "shape": [6, 480],
            },
            "observation.extra.dino_patches": {
                "array": "dino_patches.npy",
                "dtype": "float16",
                "shape": [6, 196, 384],
            },
            "observation.extra.contact_label": {
                "array": "contact_label.npy",
                "dtype": "float32",
                "shape": [6, 1],
            },
        },
    }
    (root / "meta.json").write_text(json.dumps(meta))
    np.save(root / "episode_starts.npy", np.array([0, 3], dtype=np.int64))
    np.save(root / "episode_ends.npy", np.array([3, 6], dtype=np.int64))
    np.save(root / "state.npy", np.arange(6 * 8, dtype=np.float32).reshape(6, 8))
    np.save(root / "action.npy", np.arange(6 * 7, dtype=np.float32).reshape(6, 7))
    np.save(root / "a_base.npy", np.ones((6, 7), dtype=np.float32))
    np.save(root / "k_idx_norm.npy", np.arange(6, dtype=np.float32).reshape(6, 1))
    np.save(root / "z_goal.npy", np.ones((6, 960), dtype=np.float16))
    np.save(root / "z_phase.npy", np.ones((6, 480), dtype=np.float16))
    np.save(root / "dino_patches.npy", np.ones((6, 196, 384), dtype=np.float16))
    np.save(root / "contact_label.npy", np.zeros((6, 1), dtype=np.float32))


def test_load_fast_cache_metadata_validates_required_arrays(tmp_path):
    cache_root = tmp_path / "cache"
    _write_cache(cache_root)

    metadata = load_fast_cache_metadata(cache_root)

    assert metadata["total_frames"] == 6
    assert metadata["features"]["observation.extra.dino_patches"]["dtype"] == "float16"


def test_fast_cache_dataset_clamps_window_at_episode_start(tmp_path):
    cache_root = tmp_path / "cache"
    _write_cache(cache_root)
    dataset = HFRVLAFastCacheDataset(cache_root, seq_len=3)

    sample = dataset[3]

    assert sample["observation.state"].shape == (3, 8)
    assert sample["action"].shape == (3, 7)
    assert sample["observation.extra.dino_patches"].shape == (3, 196, 384)
    assert sample["observation.extra.k_idx_norm"].shape == (3, 1)
    assert torch.equal(sample["observation.state"][0], sample["observation.state"][1])
    assert torch.equal(sample["observation.state"][1], sample["observation.state"][2])


def test_fast_cache_dataset_collates_policy_keys(tmp_path):
    cache_root = tmp_path / "cache"
    _write_cache(cache_root)
    dataset = HFRVLAFastCacheDataset(cache_root, seq_len=3)

    batch = next(iter(DataLoader(dataset, batch_size=2)))

    assert batch["observation.state"].shape == (2, 3, 8)
    assert batch["observation.extra.z_goal"].dtype == torch.float16
    assert batch["observation.extra.contact_label"].shape == (2, 3, 1)
    assert dataset.meta.camera_keys == []
    assert dataset.meta.stats["action"]["mean"].shape == (7,)
