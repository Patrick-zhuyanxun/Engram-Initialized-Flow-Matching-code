import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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
    (source_root / "data/chunk-000").mkdir(parents=True)
    (source_root / "meta" / "stats.json").write_text(
        json.dumps(
            {
                "observation.state": {"mean": [0.0] * 8, "std": [1.0] * 8},
                "action": {"mean": [0.0] * 7, "std": [1.0] * 7},
            }
        )
    )
    pd.DataFrame(
        {"task_index": [0, 1]},
        index=pd.Index(["pick up the block", "open the drawer"], name="task"),
    ).to_parquet(source_root / "meta" / "tasks.parquet")
    pd.DataFrame(
        {
            "index": list(range(6)),
            "task_index": [0, 0, 0, 1, 1, 1],
        }
    ).to_parquet(source_root / "data/chunk-000/file-000.parquet", index=False)
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


def _add_v2_labels(
    root: Path,
    *,
    y_correct: list[int] | None = None,
    y_preserve: list[int] | None = None,
) -> None:
    y_correct = y_correct or [0, 0, 1, 0, 1, 0]
    y_preserve = y_preserve or [1, 1, 0, 1, 0, 0]
    meta = json.loads((root / "meta.json").read_text())
    meta["schema_version"] = 2
    meta["features"]["observation.extra.y_correct"] = {
        "array": "y_correct.npy",
        "dtype": "uint8",
        "shape": [6],
    }
    meta["features"]["observation.extra.y_preserve"] = {
        "array": "y_preserve.npy",
        "dtype": "uint8",
        "shape": [6],
    }
    meta["static_label_thresholds"] = {
        "err_offline_correct": 5.0,
        "err_offline_preserve": 2.0,
    }
    (root / "meta.json").write_text(json.dumps(meta))
    np.save(root / "y_correct.npy", np.array(y_correct, dtype=np.uint8))
    np.save(root / "y_preserve.npy", np.array(y_preserve, dtype=np.uint8))


def _add_v3_chunk_fields(root: Path, *, chunk_len: int = 4) -> None:
    meta = json.loads((root / "meta.json").read_text())
    meta["schema_version"] = 3
    meta["chunk_len"] = chunk_len
    meta["features"]["observation.extra.a_base_chunk"] = {
        "array": "a_base_chunk.npy",
        "dtype": "float32",
        "shape": [6, chunk_len, 7],
    }
    meta["features"]["observation.extra.chunk_step_idx"] = {
        "array": "chunk_step_idx.npy",
        "dtype": "int64",
        "shape": [6, 1],
    }
    meta["features"]["observation.extra.chunk_age_steps"] = {
        "array": "chunk_age_steps.npy",
        "dtype": "float32",
        "shape": [6, 1],
    }
    meta["features"]["observation.extra.chunk_age_norm"] = {
        "array": "chunk_age_norm.npy",
        "dtype": "float32",
        "shape": [6, 1],
    }
    (root / "meta.json").write_text(json.dumps(meta))

    chunks = np.zeros((6, chunk_len, 7), dtype=np.float32)
    step_idx = np.zeros((6, 1), dtype=np.int64)
    for frame in range(6):
        chunk_start = 0 if frame < 3 else 3
        local_step = frame - chunk_start
        for j in range(chunk_len):
            chunks[frame, j] = chunk_start + j + 1
        chunks[frame, local_step] = np.ones((7,), dtype=np.float32)
        step_idx[frame, 0] = local_step
    np.save(root / "a_base_chunk.npy", chunks)
    np.save(root / "chunk_step_idx.npy", step_idx)
    np.save(root / "chunk_age_steps.npy", step_idx.astype(np.float32))
    np.save(root / "chunk_age_norm.npy", step_idx.astype(np.float32) / max(1, chunk_len - 1))


def test_load_fast_cache_metadata_validates_required_arrays(tmp_path):
    cache_root = tmp_path / "cache"
    _write_cache(cache_root)

    metadata = load_fast_cache_metadata(cache_root)

    assert metadata["total_frames"] == 6
    assert metadata["features"]["observation.extra.dino_patches"]["dtype"] == "float16"


def test_fast_cache_dataset_requires_explicit_seq_len(tmp_path):
    cache_root = tmp_path / "cache"
    _write_cache(cache_root)

    with pytest.raises(ValueError, match="seq_len must be provided"):
        HFRVLAFastCacheDataset(cache_root)


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


def test_fast_cache_dataset_returns_v2_static_labels(tmp_path):
    cache_root = tmp_path / "cache"
    _write_cache(cache_root)
    _add_v2_labels(cache_root)
    dataset = HFRVLAFastCacheDataset(cache_root, seq_len=3)

    sample = dataset[2]
    batch = next(iter(DataLoader(dataset, batch_size=2)))

    assert sample["observation.extra.y_correct"].shape == (3,)
    assert sample["observation.extra.y_preserve"].dtype == torch.uint8
    assert sample["observation.extra.y_correct"].tolist() == [0, 0, 1]
    assert batch["observation.extra.y_correct"].shape == (2, 3)


def test_fast_cache_dataset_returns_v3_chunk_fields_and_preserves_current_base(tmp_path):
    cache_root = tmp_path / "cache"
    _write_cache(cache_root)
    _add_v3_chunk_fields(cache_root, chunk_len=4)
    dataset = HFRVLAFastCacheDataset(cache_root, seq_len=3)

    sample = dataset[4]

    assert dataset.info["schema_version"] == 3
    assert dataset.info["chunk_len"] == 4
    assert sample["observation.extra.a_base_chunk"].shape == (3, 4, 7)
    assert sample["observation.extra.chunk_step_idx"].shape == (3, 1)
    assert sample["observation.extra.chunk_age_steps"].shape == (3, 1)
    assert sample["observation.extra.chunk_age_norm"].shape == (3, 1)
    assert sample["observation.extra.chunk_step_idx"].dtype == torch.int64
    current_step = int(sample["observation.extra.chunk_step_idx"][-1, 0].item())
    current_chunk_action = sample["observation.extra.a_base_chunk"][-1, current_step]
    assert torch.equal(sample["observation.extra.a_base"][-1], current_chunk_action)
    assert sample["observation.extra.chunk_age_steps"][-1, 0].item() == float(current_step)


def test_fast_cache_dataset_accepts_v1_without_static_labels(tmp_path):
    cache_root = tmp_path / "cache"
    _write_cache(cache_root)
    dataset = HFRVLAFastCacheDataset(cache_root, seq_len=3)

    sample = dataset[0]

    assert "observation.extra.y_correct" not in sample
    assert "observation.extra.y_preserve" not in sample


def test_fast_cache_dataset_returns_task_complementary_data(tmp_path):
    cache_root = tmp_path / "cache"
    _write_cache(cache_root)
    dataset = HFRVLAFastCacheDataset(cache_root, seq_len=3)

    sample = dataset[3]
    batch = next(iter(DataLoader(dataset, batch_size=2)))

    assert sample["task"] == "open the drawer"
    assert sample["index"].item() == 3
    assert sample["episode_index"].item() == 1
    assert sample["task_index"].item() == 1
    assert batch["task"] == ["pick up the block", "pick up the block"]


def test_fast_cache_dataset_concatenates_roots_and_static_label_fractions(tmp_path):
    cache_a = tmp_path / "cache_a"
    cache_b = tmp_path / "cache_b"
    _write_cache(cache_a)
    _write_cache(cache_b)
    _add_v2_labels(cache_a)
    _add_v2_labels(cache_b, y_correct=[0, 0, 0, 0, 0, 0], y_preserve=[1, 1, 1, 1, 1, 1])
    np.save(cache_b / "state.npy", np.arange(1000, 1000 + 6 * 8, dtype=np.float32).reshape(6, 8))

    dataset = HFRVLAFastCacheDataset(roots=[cache_a, cache_b], seq_len=3)

    assert len(dataset) == 12
    assert dataset.num_frames == 12
    assert dataset.num_episodes == 4

    first_preserve = [
        int(dataset[i]["observation.extra.y_preserve"][-1].item())
        for i in range(6)
    ]
    second_preserve = [
        int(dataset[i]["observation.extra.y_preserve"][-1].item())
        for i in range(6, 12)
    ]
    assert sum(first_preserve) / len(first_preserve) == 0.5
    assert sum(second_preserve) / len(second_preserve) == 1.0

    first_rollout_sample = dataset[6]
    assert first_rollout_sample["index"].item() == 6
    assert first_rollout_sample["episode_index"].item() == 2
    assert torch.equal(
        first_rollout_sample["observation.state"][0],
        first_rollout_sample["observation.state"][1],
    )
    assert first_rollout_sample["observation.state"][0, 0].item() == 1000.0
