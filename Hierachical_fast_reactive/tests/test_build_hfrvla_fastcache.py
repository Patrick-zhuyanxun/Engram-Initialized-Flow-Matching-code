import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_hfrvla_fastcache import build_fast_cache


def _write_source(root: Path) -> None:
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "meta").mkdir(exist_ok=True)
    info = {
        "fps": 10,
        "total_frames": 4,
        "total_episodes": 2,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [8], "names": None},
            "action": {"dtype": "float32", "shape": [7], "names": ["actions"]},
            "observation.extra.a_base": {
                "dtype": "float32",
                "shape": [7],
                "names": None,
            },
            "observation.extra.k_idx_norm": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            },
            "observation.extra.z_goal": {
                "dtype": "float32",
                "shape": [960],
                "names": None,
            },
            "observation.extra.z_phase": {
                "dtype": "float32",
                "shape": [480],
                "names": None,
            },
            "observation.extra.dino_patches": {
                "dtype": "float32",
                "shape": [196, 384],
                "names": None,
            },
            "observation.extra.contact_label": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            },
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/stats.json").write_text(
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
    ).to_parquet(root / "meta/tasks.parquet")
    pd.DataFrame(
        {
            "episode_index": [0, 1],
            "dataset_from_index": [0, 2],
            "dataset_to_index": [2, 4],
        }
    ).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet", index=False)
    rows = []
    for i in range(4):
        rows.append(
            {
                "observation.state": np.full((8,), i, dtype=np.float32).tolist(),
                "action": np.full((7,), i, dtype=np.float32).tolist(),
                "observation.extra.a_base": np.full((7,), 1, dtype=np.float32).tolist(),
                "observation.extra.k_idx_norm": np.array([i / 3], dtype=np.float32).tolist(),
                "observation.extra.z_goal": np.full((960,), i, dtype=np.float32).tolist(),
                "observation.extra.z_phase": np.full((480,), i, dtype=np.float32).tolist(),
                "observation.extra.dino_patches": np.full(
                    (196, 384), i, dtype=np.float32
                ).tolist(),
                "observation.extra.contact_label": np.array([0], dtype=np.float32).tolist(),
                "episode_index": 0 if i < 2 else 1,
                "task_index": 0 if i < 2 else 1,
                "index": i,
            }
        )
    pd.DataFrame(rows).to_parquet(root / "data/chunk-000/file-000.parquet", index=False)


def test_build_fast_cache_writes_float16_large_arrays(tmp_path):
    source = tmp_path / "source"
    cache = tmp_path / "cache"
    _write_source(source)

    build_fast_cache(source, cache, seq_len=3)

    meta = json.loads((cache / "meta.json").read_text())
    assert meta["schema_version"] == 2
    assert meta["total_frames"] == 4
    assert np.load(cache / "episode_starts.npy").tolist() == [0, 2]
    assert np.load(cache / "episode_ends.npy").tolist() == [2, 4]
    assert np.load(cache / "task_index.npy").tolist() == [0, 0, 1, 1]
    assert meta["index_arrays"]["task_index"]["array"] == "task_index.npy"
    assert np.load(cache / "z_goal.npy", mmap_mode="r").dtype == np.float16
    assert np.load(cache / "dino_patches.npy", mmap_mode="r").shape == (4, 196, 384)
    assert np.load(cache / "y_correct.npy").dtype == np.uint8
    assert np.load(cache / "y_preserve.npy").dtype == np.uint8
    assert np.load(cache / "y_correct.npy").shape == (4,)
    assert meta["features"]["observation.extra.y_correct"]["array"] == "y_correct.npy"
    assert meta["static_label_quantiles"] == {"correct": 0.8, "preserve": 0.5}
    assert meta["static_label_fractions"]["y_correct"] == 0.25
    assert meta["static_label_fractions"]["y_preserve"] == 0.25


def test_build_fast_cache_static_y_preserve_marks_all_frames_preserve(tmp_path):
    source = tmp_path / "source"
    cache = tmp_path / "cache"
    _write_source(source)

    build_fast_cache(source, cache, seq_len=3, static_y_preserve=True)

    meta = json.loads((cache / "meta.json").read_text())
    assert meta["static_label_mode"] == "static_y_preserve"
    assert meta["static_label_thresholds"] == {
        "err_offline_correct": None,
        "err_offline_preserve": None,
    }
    assert np.load(cache / "y_correct.npy").tolist() == [0, 0, 0, 0]
    assert np.load(cache / "y_preserve.npy").tolist() == [1, 1, 1, 1]
    assert meta["static_label_fractions"]["y_correct"] == 0.0
    assert meta["static_label_fractions"]["y_preserve"] == 1.0
