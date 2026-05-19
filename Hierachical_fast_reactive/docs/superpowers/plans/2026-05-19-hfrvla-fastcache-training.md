# HFRVLA Fast-Cache Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in fast-cache dataset backend that trains HFRVLA from compact contiguous arrays instead of randomly materializing large nested Parquet DINO patch columns.

**Architecture:** Keep the existing LeRobotDataset v3 as the canonical source of truth. Add a rebuildable NumPy `.npy` cache, a focused PyTorch dataset that returns the same policy batch keys as the LeRobot path, and a small `train_via_lerobot.py` integration that swaps the dataset backend when requested.

**Tech Stack:** Python 3.12, PyTorch, NumPy memmap, PyArrow/Hugging Face Datasets, LeRobot v3 metadata, pytest.

---

## File Structure

- Create `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/fast_cache_dataset.py`
  - Owns cache metadata validation, memory-mapped array loading, LeRobot-compatible metadata wrapper, and temporal window sampling.
- Create `scripts/build_hfrvla_fastcache.py`
  - Builds a cache from selected canonical LeRobot v3 Parquet columns.
- Create `tests/test_hfrvla_fast_cache_dataset.py`
  - Unit tests for metadata validation, window clamping, collate behavior, and dtype choices.
- Create `tests/test_build_hfrvla_fastcache.py`
  - Unit tests for cache writing from a small Parquet fixture.
- Modify `scripts/train_via_lerobot.py`
  - Adds backend selection and returns `HFRVLAFastCacheDataset` from `make_dataset` when requested.
- Modify `scripts/train_hfrvla_libero_merged.sh`
  - Adds `DATASET_BACKEND` and `FASTCACHE_ROOT` environment wiring.
- Modify `scripts/run_hfrvla_training_foreground.sh`
  - Prints the selected dataset backend and default cache root.
- Modify `docs/training.md`, `docs/lerobot_hfrvla_context.md`, and `policy/lerobot_policy_hfrvla/README.md`
  - Documents cache build and train commands.

---

### Task 1: Fast-Cache Dataset Core

**Files:**
- Create: `tests/test_hfrvla_fast_cache_dataset.py`
- Create: `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/fast_cache_dataset.py`

- [ ] **Step 1: Write failing tests for metadata loading and temporal windows**

Add `tests/test_hfrvla_fast_cache_dataset.py`:

```python
import json
from pathlib import Path

import numpy as np
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
            "observation.state": {"array": "state.npy", "dtype": "float32", "shape": [6, 8]},
            "action": {"array": "action.npy", "dtype": "float32", "shape": [6, 7]},
            "observation.extra.a_base": {"array": "a_base.npy", "dtype": "float32", "shape": [6, 7]},
            "observation.extra.k_idx_norm": {"array": "k_idx_norm.npy", "dtype": "float32", "shape": [6, 1]},
            "observation.extra.z_goal": {"array": "z_goal.npy", "dtype": "float16", "shape": [6, 960]},
            "observation.extra.z_phase": {"array": "z_phase.npy", "dtype": "float16", "shape": [6, 480]},
            "observation.extra.dino_patches": {"array": "dino_patches.npy", "dtype": "float16", "shape": [6, 196, 384]},
            "observation.extra.contact_label": {"array": "contact_label.npy", "dtype": "float32", "shape": [6, 1]},
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
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
uv run pytest tests/test_hfrvla_fast_cache_dataset.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'lerobot_policy_hfrvla.fast_cache_dataset'`.

- [ ] **Step 3: Implement the dataset core**

Create `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/fast_cache_dataset.py` with:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets.io_utils import load_stats
from torch.utils.data import Dataset

ACTION = "action"
OBS_STATE = "observation.state"

CACHE_SCHEMA_VERSION = 1
POLICY_KEY_TO_ARRAY = {
    OBS_STATE: "state",
    ACTION: "action",
    "observation.extra.a_base": "a_base",
    "observation.extra.k_idx_norm": "k_idx_norm",
    "observation.extra.z_goal": "z_goal",
    "observation.extra.z_phase": "z_phase",
    "observation.extra.dino_patches": "dino_patches",
    "observation.extra.contact_label": "contact_label",
}


@dataclass
class HFRVLAFastCacheMetadata:
    root: Path
    info: dict[str, Any]
    features: dict[str, dict[str, Any]]
    stats: dict[str, dict[str, np.ndarray]]
    fps: int
    total_frames: int
    total_episodes: int
    camera_keys: list[str]
    video_keys: list[str]

    @property
    def num_frames(self) -> int:
        return self.total_frames

    @property
    def num_episodes(self) -> int:
        return self.total_episodes


def load_fast_cache_metadata(cache_root: str | Path) -> dict[str, Any]:
    root = Path(cache_root)
    meta_path = root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Fast-cache metadata not found: {meta_path}")
    metadata = json.loads(meta_path.read_text())
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported fast-cache schema_version={metadata.get('schema_version')}; "
            f"expected {CACHE_SCHEMA_VERSION}"
        )
    missing = [
        spec["array"]
        for spec in metadata["features"].values()
        if not (root / spec["array"]).exists()
    ]
    if missing:
        raise FileNotFoundError("Fast-cache is missing array files: " + ", ".join(sorted(missing)))
    for name in ("episode_starts.npy", "episode_ends.npy"):
        if not (root / name).exists():
            raise FileNotFoundError(f"Fast-cache is missing {name}")
    return metadata


def _feature_specs_for_policy(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    features = {}
    for key, spec in metadata["features"].items():
        shape = tuple(spec["shape"][1:])
        features[key] = {"dtype": spec["dtype"], "shape": shape, "names": None}
    features[ACTION]["names"] = ["actions"]
    return features


class HFRVLAFastCacheDataset(Dataset):
    def __init__(self, cache_root: str | Path, *, seq_len: int | None = None):
        self.root = Path(cache_root)
        self.info = load_fast_cache_metadata(self.root)
        self.seq_len = int(seq_len or self.info.get("seq_len") or 1)
        self.episode_starts = np.load(self.root / "episode_starts.npy", mmap_mode="r")
        self.episode_ends = np.load(self.root / "episode_ends.npy", mmap_mode="r")
        self.arrays = {
            key: np.load(self.root / spec["array"], mmap_mode="r")
            for key, spec in self.info["features"].items()
        }
        self._validate_array_shapes()
        source_root = Path(self.info["source_dataset_root"])
        stats = load_stats(source_root)
        if stats is None:
            raise FileNotFoundError(f"Canonical stats not found under {source_root / 'meta'}")
        self.meta = HFRVLAFastCacheMetadata(
            root=self.root,
            info=self.info,
            features=_feature_specs_for_policy(self.info),
            stats=stats,
            fps=int(self.info["fps"]),
            total_frames=int(self.info["total_frames"]),
            total_episodes=int(self.info["total_episodes"]),
            camera_keys=[],
            video_keys=[],
        )
        self.num_frames = self.meta.total_frames
        self.num_episodes = self.meta.total_episodes
        self.episodes = None

    def _validate_array_shapes(self) -> None:
        total_frames = int(self.info["total_frames"])
        for key, spec in self.info["features"].items():
            actual = tuple(self.arrays[key].shape)
            expected = tuple(spec["shape"])
            if actual != expected:
                raise ValueError(f"Fast-cache array {key} has shape {actual}, expected {expected}")
            if actual[0] != total_frames:
                raise ValueError(f"Fast-cache array {key} frame count {actual[0]} != {total_frames}")

    def __len__(self) -> int:
        return int(self.info["total_frames"])

    def _episode_bounds_for_index(self, idx: int) -> tuple[int, int]:
        episode_idx = int(np.searchsorted(self.episode_ends, idx, side="right"))
        return int(self.episode_starts[episode_idx]), int(self.episode_ends[episode_idx])

    def _window_indices(self, idx: int) -> np.ndarray:
        start, end = self._episode_bounds_for_index(idx)
        raw = np.arange(idx - self.seq_len + 1, idx + 1, dtype=np.int64)
        return np.clip(raw, start, end - 1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        idx = int(idx)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        window = self._window_indices(idx)
        return {
            key: torch.from_numpy(np.asarray(self.arrays[key][window]))
            for key in POLICY_KEY_TO_ARRAY
        }
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
uv run pytest tests/test_hfrvla_fast_cache_dataset.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit dataset core**

```bash
git add tests/test_hfrvla_fast_cache_dataset.py policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/fast_cache_dataset.py
git commit -m "feat: add hfrvla fastcache dataset"
```

---

### Task 2: Fast-Cache Builder Script

**Files:**
- Create: `tests/test_build_hfrvla_fastcache.py`
- Create: `scripts/build_hfrvla_fastcache.py`

- [ ] **Step 1: Write failing builder test**

Add `tests/test_build_hfrvla_fastcache.py`:

```python
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
            "observation.extra.a_base": {"dtype": "float32", "shape": [7], "names": None},
            "observation.extra.k_idx_norm": {"dtype": "float32", "shape": [1], "names": None},
            "observation.extra.z_goal": {"dtype": "float32", "shape": [960], "names": None},
            "observation.extra.z_phase": {"dtype": "float32", "shape": [480], "names": None},
            "observation.extra.dino_patches": {"dtype": "float32", "shape": [196, 384], "names": None},
            "observation.extra.contact_label": {"dtype": "float32", "shape": [1], "names": None},
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
                "observation.state": np.full((8,), i, dtype=np.float32),
                "action": np.full((7,), i, dtype=np.float32),
                "observation.extra.a_base": np.full((7,), 1, dtype=np.float32),
                "observation.extra.k_idx_norm": np.array([i / 3], dtype=np.float32),
                "observation.extra.z_goal": np.full((960,), i, dtype=np.float32),
                "observation.extra.z_phase": np.full((480,), i, dtype=np.float32),
                "observation.extra.dino_patches": np.full((196, 384), i, dtype=np.float32),
                "observation.extra.contact_label": np.array([0], dtype=np.float32),
                "episode_index": 0 if i < 2 else 1,
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
    assert meta["total_frames"] == 4
    assert np.load(cache / "episode_starts.npy").tolist() == [0, 2]
    assert np.load(cache / "episode_ends.npy").tolist() == [2, 4]
    assert np.load(cache / "z_goal.npy", mmap_mode="r").dtype == np.float16
    assert np.load(cache / "dino_patches.npy", mmap_mode="r").shape == (4, 196, 384)
```

- [ ] **Step 2: Run builder test to verify RED**

Run:

```bash
uv run pytest tests/test_build_hfrvla_fastcache.py -q
```

Expected: FAIL with `ModuleNotFoundError` or missing `build_fast_cache`.

- [ ] **Step 3: Implement builder script**

Create `scripts/build_hfrvla_fastcache.py` with:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot_policy_hfrvla.fast_cache_dataset import CACHE_SCHEMA_VERSION

POLICY_TO_ARRAY = {
    "observation.state": ("state.npy", np.float32),
    "action": ("action.npy", np.float32),
    "observation.extra.a_base": ("a_base.npy", np.float32),
    "observation.extra.k_idx_norm": ("k_idx_norm.npy", np.float32),
    "observation.extra.z_goal": ("z_goal.npy", np.float16),
    "observation.extra.z_phase": ("z_phase.npy", np.float16),
    "observation.extra.dino_patches": ("dino_patches.npy", np.float16),
    "observation.extra.contact_label": ("contact_label.npy", np.float32),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _load_info(source_root: Path) -> dict:
    info_path = source_root / "meta/info.json"
    if not info_path.exists():
        raise FileNotFoundError(info_path)
    return json.loads(info_path.read_text())


def _load_episode_bounds(source_root: Path) -> tuple[np.ndarray, np.ndarray]:
    files = sorted((source_root / "meta/episodes").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata under {source_root / 'meta/episodes'}")
    episodes = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    episodes = episodes.sort_values("episode_index")
    return (
        episodes["dataset_from_index"].to_numpy(dtype=np.int64),
        episodes["dataset_to_index"].to_numpy(dtype=np.int64),
    )


def _empty_arrays(cache_root: Path, info: dict) -> dict[str, np.memmap]:
    total_frames = int(info["total_frames"])
    arrays = {}
    for key, (filename, dtype) in POLICY_TO_ARRAY.items():
        feature = info["features"][key]
        shape = (total_frames, *tuple(feature["shape"]))
        arrays[key] = np.lib.format.open_memmap(cache_root / filename, mode="w+", dtype=dtype, shape=shape)
    return arrays


def build_fast_cache(source_root: str | Path, cache_root: str | Path, *, seq_len: int) -> None:
    source_root = Path(source_root)
    cache_root = Path(cache_root)
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True)
    info = _load_info(source_root)
    starts, ends = _load_episode_bounds(source_root)
    np.save(cache_root / "episode_starts.npy", starts)
    np.save(cache_root / "episode_ends.npy", ends)
    arrays = _empty_arrays(cache_root, info)
    data_files = sorted((source_root / "data").glob("*/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data parquet files under {source_root / 'data'}")
    offset = 0
    columns = list(POLICY_TO_ARRAY)
    for path in data_files:
        df = pd.read_parquet(path, columns=columns)
        count = len(df)
        for key in columns:
            arrays[key][offset : offset + count] = np.stack(df[key].to_numpy()).astype(arrays[key].dtype)
        offset += count
    expected = int(info["total_frames"])
    if offset != expected:
        raise ValueError(f"Copied {offset} frames, expected {expected}")
    for arr in arrays.values():
        arr.flush()
    features = {}
    for key, (filename, dtype) in POLICY_TO_ARRAY.items():
        features[key] = {
            "array": filename,
            "dtype": np.dtype(dtype).name,
            "shape": [expected, *list(info["features"][key]["shape"])],
        }
    meta = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_dataset_root": str(source_root.resolve()),
        "source_info_sha256": _sha256(source_root / "meta/info.json"),
        "total_frames": expected,
        "total_episodes": int(info["total_episodes"]),
        "fps": int(info["fps"]),
        "seq_len": int(seq_len),
        "features": features,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (cache_root / "meta.json").write_text(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_fast_cache(args.source_root, args.cache_root, seq_len=args.seq_len)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run builder tests**

Run:

```bash
uv run pytest tests/test_build_hfrvla_fastcache.py tests/test_hfrvla_fast_cache_dataset.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit builder**

```bash
git add scripts/build_hfrvla_fastcache.py tests/test_build_hfrvla_fastcache.py
git commit -m "feat: add hfrvla fastcache builder"
```

---

### Task 3: LeRobot Training Wrapper Integration

**Files:**
- Modify: `tests/test_train_via_lerobot_fast_dataset.py`
- Modify: `scripts/train_via_lerobot.py`

- [ ] **Step 1: Add failing train-wrapper tests**

Append to `tests/test_train_via_lerobot_fast_dataset.py`:

```python
def test_dataset_backend_defaults_to_lerobot(monkeypatch):
    monkeypatch.delenv("HFRVLA_DATASET_BACKEND", raising=False)

    from scripts.train_via_lerobot import _dataset_backend

    assert _dataset_backend() == "lerobot"


def test_dataset_backend_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("HFRVLA_DATASET_BACKEND", "bad")

    from scripts.train_via_lerobot import _dataset_backend

    with pytest.raises(ValueError, match="HFRVLA_DATASET_BACKEND"):
        _dataset_backend()


def test_fastcache_make_dataset_requires_root(monkeypatch):
    monkeypatch.setenv("HFRVLA_DATASET_BACKEND", "fastcache")
    monkeypatch.delenv("HFRVLA_FASTCACHE_ROOT", raising=False)
    cfg = SimpleNamespace(policy=SimpleNamespace(type="hfrvla", offline_training_mode=True, seq_len=4))

    from scripts.train_via_lerobot import _patched_make_dataset

    with pytest.raises(ValueError, match="HFRVLA_FASTCACHE_ROOT"):
        _patched_make_dataset(cfg)
```

- [ ] **Step 2: Run wrapper tests to verify RED**

Run:

```bash
uv run pytest tests/test_train_via_lerobot_fast_dataset.py -q
```

Expected: FAIL because `_dataset_backend` does not exist.

- [ ] **Step 3: Implement backend selection and fast-cache dataset return**

Modify `scripts/train_via_lerobot.py`:

```python
import os
from pathlib import Path
```

Add:

```python
def _dataset_backend() -> str:
    backend = os.environ.get("HFRVLA_DATASET_BACKEND", "lerobot").strip().lower()
    if backend not in {"lerobot", "fastcache"}:
        raise ValueError(
            "HFRVLA_DATASET_BACKEND must be either 'lerobot' or 'fastcache', "
            f"got {backend!r}"
        )
    return backend


def _make_fastcache_dataset(cfg):
    root = os.environ.get("HFRVLA_FASTCACHE_ROOT")
    if not root:
        raise ValueError("HFRVLA_FASTCACHE_ROOT must be set when HFRVLA_DATASET_BACKEND=fastcache")
    from lerobot_policy_hfrvla.fast_cache_dataset import HFRVLAFastCacheDataset

    seq_len = int(getattr(getattr(cfg, "policy", cfg), "seq_len", 1))
    dataset = HFRVLAFastCacheDataset(Path(root), seq_len=seq_len)
    print(f"[hfrvla-train] fast-cache dataset backend enabled root={Path(root)}", flush=True)
    return dataset
```

At the top of `_patched_make_dataset`, insert:

```python
    if _is_offline_hfrvla(cfg) and _dataset_backend() == "fastcache":
        return _make_fastcache_dataset(cfg)
```

- [ ] **Step 4: Run wrapper tests**

Run:

```bash
uv run pytest tests/test_train_via_lerobot_fast_dataset.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit wrapper integration**

```bash
git add scripts/train_via_lerobot.py tests/test_train_via_lerobot_fast_dataset.py
git commit -m "feat: wire hfrvla fastcache training backend"
```

---

### Task 4: Launcher Wiring And Documentation

**Files:**
- Modify: `scripts/train_hfrvla_libero_merged.sh`
- Modify: `scripts/run_hfrvla_training_foreground.sh`
- Modify: `docs/training.md`
- Modify: `docs/lerobot_hfrvla_context.md`
- Modify: `policy/lerobot_policy_hfrvla/README.md`

- [ ] **Step 1: Add launcher environment wiring**

In both launcher scripts, add:

```bash
export HFRVLA_DATASET_BACKEND="${HFRVLA_DATASET_BACKEND:-lerobot}"
export HFRVLA_FASTCACHE_ROOT="${HFRVLA_FASTCACHE_ROOT:-$PROJECT_ROOT/checkpoints/HFRVLA_libero_v1_fastcache_seq${SEQ_LEN:-4}}"
```

In startup logs, add:

```bash
echo "[hfrvla-train] dataset_backend=$HFRVLA_DATASET_BACKEND"
echo "[hfrvla-train] fastcache_root=$HFRVLA_FASTCACHE_ROOT"
```

- [ ] **Step 2: Document cache build and train commands**

Add this command to `docs/training.md`:

```bash
~/Robotic_infra/lerobot/.venv/bin/python scripts/build_hfrvla_fastcache.py \
    --source-root checkpoints/HFRVLA_libero_v1_merged_reindexed \
    --cache-root checkpoints/HFRVLA_libero_v1_fastcache_seq4 \
    --seq-len 4
```

Add this training invocation:

```bash
RUN_NAME=hfrvla_run_fastcache_seq4_30k \
HFRVLA_DATASET_BACKEND=fastcache \
HFRVLA_FASTCACHE_ROOT=$PWD/checkpoints/HFRVLA_libero_v1_fastcache_seq4 \
SEQ_LEN=4 \
BATCH_SIZE=256 \
NUM_WORKERS=8 \
STEPS=30000 \
scripts/run_hfrvla_training_foreground.sh
```

- [ ] **Step 3: Commit docs and launcher wiring**

```bash
git add scripts/train_hfrvla_libero_merged.sh scripts/run_hfrvla_training_foreground.sh docs/training.md docs/lerobot_hfrvla_context.md policy/lerobot_policy_hfrvla/README.md
git commit -m "docs: document hfrvla fastcache training"
```

---

### Task 5: Verification

**Files:**
- No new files required.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
uv run pytest \
  tests/test_hfrvla_fast_cache_dataset.py \
  tests/test_build_hfrvla_fastcache.py \
  tests/test_train_via_lerobot_fast_dataset.py \
  tests/test_hfrvla_training_contract.py \
  tests/test_hfrvla_forward_lerobot_batch.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run script syntax checks**

Run:

```bash
python3 -m py_compile \
  scripts/build_hfrvla_fastcache.py \
  scripts/train_via_lerobot.py \
  policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/fast_cache_dataset.py
```

Expected: exits 0.

- [ ] **Step 3: Report cache build command without running full 112 GB build**

State the exact command to build the full cache:

```bash
~/Robotic_infra/lerobot/.venv/bin/python scripts/build_hfrvla_fastcache.py \
    --source-root checkpoints/HFRVLA_libero_v1_merged_reindexed \
    --cache-root checkpoints/HFRVLA_libero_v1_fastcache_seq4 \
    --seq-len 4
```

- [ ] **Step 4: Final status**

Summarize changed files, test results, and whether the full cache build was run.

