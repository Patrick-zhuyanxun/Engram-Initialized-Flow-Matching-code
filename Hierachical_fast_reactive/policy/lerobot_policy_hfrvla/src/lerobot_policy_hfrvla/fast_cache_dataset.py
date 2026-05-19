"""Fast-cache dataset for HFRVLA offline training.

The canonical data remains a LeRobotDataset v3. This module reads a derived
array cache built from that dataset and returns the same batch keys consumed by
``HFRVLAPolicy.forward``.
"""

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
                raise ValueError(
                    f"Fast-cache array {key} has shape {actual}, expected {expected}"
                )
            if actual[0] != total_frames:
                raise ValueError(
                    f"Fast-cache array {key} frame count {actual[0]} != {total_frames}"
                )

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
