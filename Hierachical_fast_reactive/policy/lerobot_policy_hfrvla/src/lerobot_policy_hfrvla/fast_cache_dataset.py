"""Fast-cache dataset for HFRVLA offline training.

The canonical data remains a LeRobotDataset v3. This module reads a derived
array cache built from that dataset and returns the same batch keys consumed by
``HFRVLAPolicy.forward``.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from lerobot.datasets.io_utils import load_stats
from torch.utils.data import Dataset

ACTION = "action"
OBS_STATE = "observation.state"

CACHE_SCHEMA_VERSION = 2
SUPPORTED_CACHE_SCHEMA_VERSIONS = {1, CACHE_SCHEMA_VERSION}
POLICY_KEY_TO_ARRAY = {
    OBS_STATE: "state",
    ACTION: "action",
    "observation.extra.a_base": "a_base",
    "observation.extra.k_idx_norm": "k_idx_norm",
    "observation.extra.z_goal": "z_goal",
    "observation.extra.z_phase": "z_phase",
    "observation.extra.dino_patches": "dino_patches",
    "observation.extra.contact_label": "contact_label",
    "observation.extra.y_correct": "y_correct",
    "observation.extra.y_preserve": "y_preserve",
}
INDEX_ARRAY_FILES = {
    "index": "index.npy",
    "episode_index": "episode_index.npy",
    "task_index": "task_index.npy",
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
    schema_version = int(metadata.get("schema_version", -1))
    if schema_version not in SUPPORTED_CACHE_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported fast-cache schema_version={metadata.get('schema_version')}; "
            f"expected one of {sorted(SUPPORTED_CACHE_SCHEMA_VERSIONS)}"
        )

    missing = [
        spec["array"]
        for spec in metadata["features"].values()
        if not (root / spec["array"]).exists()
    ]
    missing.extend(
        spec["array"]
        for spec in metadata.get("index_arrays", {}).values()
        if not (root / spec["array"]).exists()
    )
    if missing:
        raise FileNotFoundError(
            "Fast-cache is missing array files: " + ", ".join(sorted(missing))
        )

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


def _numeric_file_index(path: Path) -> int:
    return int(path.stem.split("-")[-1])


def _load_tasks(source_root: Path) -> list[str]:
    tasks_path = source_root / "meta/tasks.parquet"
    if not tasks_path.exists():
        return [""]

    tasks = pd.read_parquet(tasks_path)
    if "task_index" in tasks.columns:
        tasks = tasks.sort_values("task_index")
    return [str(task) for task in tasks.index.to_list()]


def _load_task_indices_from_source(source_root: Path, total_frames: int) -> np.ndarray:
    paths = sorted((source_root / "data").glob("*/*.parquet"), key=_numeric_file_index)
    if not paths:
        return np.zeros(total_frames, dtype=np.int64)

    task_indices = np.zeros(total_frames, dtype=np.int64)
    for path in paths:
        df = pd.read_parquet(path, columns=["index", "task_index"])
        indices = df["index"].to_numpy(dtype=np.int64)
        task_indices[indices] = df["task_index"].to_numpy(dtype=np.int64)
    return task_indices


def _is_root_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, Path))


class HFRVLAFastCacheDataset(Dataset):
    def __init__(
        self,
        cache_root: str | Path | Sequence[str | Path] | None = None,
        *,
        roots: Sequence[str | Path] | None = None,
        seq_len: int | None = None,
    ):
        if roots is not None or _is_root_sequence(cache_root):
            root_list = list(roots if roots is not None else cache_root)
            self._init_multi_root(root_list, seq_len=seq_len)
            return

        if cache_root is None:
            raise ValueError("cache_root is required unless roots=[...] is provided")

        self._children: list[HFRVLAFastCacheDataset] | None = None
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
        self.index_arrays = self._load_index_arrays(source_root)
        self.tasks = _load_tasks(source_root)
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

    def _init_multi_root(
        self,
        roots: Sequence[str | Path],
        *,
        seq_len: int | None,
    ) -> None:
        if not roots:
            raise ValueError("roots must contain at least one fast-cache root")

        children = [HFRVLAFastCacheDataset(root, seq_len=seq_len) for root in roots]
        first = children[0]
        first_feature_shapes = {
            key: tuple(spec["shape"][1:])
            for key, spec in first.info["features"].items()
        }
        for child in children[1:]:
            feature_shapes = {
                key: tuple(spec["shape"][1:])
                for key, spec in child.info["features"].items()
            }
            if feature_shapes != first_feature_shapes:
                raise ValueError(
                    "Cannot concatenate fast-cache roots with different feature schemas: "
                    f"{first.root} vs {child.root}"
                )
            if int(child.info["fps"]) != int(first.info["fps"]):
                raise ValueError(
                    "Cannot concatenate fast-cache roots with different fps values: "
                    f"{first.root} fps={first.info['fps']} vs "
                    f"{child.root} fps={child.info['fps']}"
                )

        self._children = children
        lengths = [len(child) for child in children]
        episode_counts = [child.num_episodes for child in children]
        self._cum_lengths = np.cumsum(lengths, dtype=np.int64)
        self._cum_episodes = np.cumsum(episode_counts, dtype=np.int64)
        self.root = first.root
        self.roots = [child.root for child in children]
        self.info = dict(first.info)
        self.info["source_dataset_root"] = [
            child.info["source_dataset_root"] for child in children
        ]
        self.info["total_frames"] = int(sum(lengths))
        self.info["total_episodes"] = int(sum(episode_counts))
        self.seq_len = first.seq_len
        self.tasks = first.tasks
        self.meta = HFRVLAFastCacheMetadata(
            root=self.root,
            info=self.info,
            features=first.meta.features,
            stats=first.meta.stats,
            fps=first.meta.fps,
            total_frames=int(sum(lengths)),
            total_episodes=int(sum(episode_counts)),
            camera_keys=[],
            video_keys=[],
        )
        self.num_frames = self.meta.total_frames
        self.num_episodes = self.meta.total_episodes
        self.episodes = None

    def _load_index_arrays(self, source_root: Path) -> dict[str, np.ndarray]:
        total_frames = int(self.info["total_frames"])
        arrays = {}
        index_specs = self.info.get("index_arrays", {})
        for key, filename in INDEX_ARRAY_FILES.items():
            spec = index_specs.get(key)
            path = self.root / (spec["array"] if spec is not None else filename)
            if path.exists():
                arrays[key] = np.load(path, mmap_mode="r")

        if "index" not in arrays:
            arrays["index"] = np.arange(total_frames, dtype=np.int64)
        if "episode_index" not in arrays:
            frame_indices = np.arange(total_frames, dtype=np.int64)
            arrays["episode_index"] = np.searchsorted(
                self.episode_ends,
                frame_indices,
                side="right",
            ).astype(np.int64)
        if "task_index" not in arrays:
            arrays["task_index"] = _load_task_indices_from_source(source_root, total_frames)
        return arrays

    def _validate_array_shapes(self) -> None:
        total_frames = int(self.info["total_frames"])
        for key, spec in self.info["features"].items():
            actual = tuple(self.arrays[key].shape)
            expected = tuple(spec["shape"])
            if actual != expected:
                raise ValueError(
                    f"Fast-cache array {key} has shape {actual}, expected {expected}"
                )
            if actual[0] != total_frames:
                raise ValueError(
                    f"Fast-cache array {key} frame count {actual[0]} != {total_frames}"
                )

    def __len__(self) -> int:
        if self._children is not None:
            return int(self._cum_lengths[-1])
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
        if self._children is not None:
            child_idx = bisect_right(self._cum_lengths.tolist(), idx)
            frame_offset = 0 if child_idx == 0 else int(self._cum_lengths[child_idx - 1])
            episode_offset = 0 if child_idx == 0 else int(self._cum_episodes[child_idx - 1])
            sample = self._children[child_idx][idx - frame_offset]
            sample["index"] = sample["index"] + frame_offset
            sample["episode_index"] = sample["episode_index"] + episode_offset
            return sample

        window = self._window_indices(idx)
        sample = {
            key: torch.from_numpy(np.asarray(self.arrays[key][window]))
            for key in POLICY_KEY_TO_ARRAY
            if key in self.arrays
        }
        task_index = int(self.index_arrays["task_index"][idx])
        sample["index"] = torch.tensor(int(self.index_arrays["index"][idx]), dtype=torch.int64)
        sample["episode_index"] = torch.tensor(
            int(self.index_arrays["episode_index"][idx]),
            dtype=torch.int64,
        )
        sample["task_index"] = torch.tensor(task_index, dtype=torch.int64)
        sample["task"] = self.tasks[task_index] if 0 <= task_index < len(self.tasks) else ""
        return sample
