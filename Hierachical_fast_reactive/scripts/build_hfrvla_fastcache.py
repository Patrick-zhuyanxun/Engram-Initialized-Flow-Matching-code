#!/usr/bin/env python
"""Build a compact HFRVLA fast-cache from the canonical LeRobotDataset v3."""

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


def _numeric_file_index(path: Path) -> int:
    return int(path.stem.split("-")[-1])


def _load_episode_bounds(source_root: Path) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(
        (source_root / "meta/episodes").glob("chunk-*/*.parquet"),
        key=_numeric_file_index,
    )
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
        arrays[key] = np.lib.format.open_memmap(
            cache_root / filename,
            mode="w+",
            dtype=dtype,
            shape=shape,
        )
    return arrays


def _stack_column(values: pd.Series, feature_shape: tuple[int, ...], dtype) -> np.ndarray:
    arrays = []
    for value in values.to_numpy():
        arr = np.asarray(value)
        if arr.dtype == object:
            arr = np.asarray(arr.tolist(), dtype=dtype)
        else:
            arr = arr.astype(dtype, copy=False)
        arrays.append(arr)
    stacked = np.stack(arrays)
    if feature_shape == (1,) and stacked.ndim == 1:
        stacked = stacked.reshape(-1, 1)
    return stacked


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
    data_files = sorted((source_root / "data").glob("*/*.parquet"), key=_numeric_file_index)
    if not data_files:
        raise FileNotFoundError(f"No data parquet files under {source_root / 'data'}")

    offset = 0
    columns = list(POLICY_TO_ARRAY)
    for path in data_files:
        df = pd.read_parquet(path, columns=columns)
        count = len(df)
        for key in columns:
            feature_shape = tuple(info["features"][key]["shape"])
            values = _stack_column(df[key], feature_shape, arrays[key].dtype)
            arrays[key][offset : offset + count] = values
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
