"""Small helpers for resumable, sharded LIBERO precompute jobs."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Protocol


class EpisodeRecordLike(Protocol):
    def save(self, path: Path) -> None:
        """Persist the episode record to ``path``."""


def resolve_episode_indices(
    *,
    n_total: int,
    max_episodes: int | None,
    ep_from: int,
    ep_to: int | None,
) -> list[int]:
    """Return the episode ids selected by the global cap and shard bounds."""
    if n_total < 0:
        raise ValueError("n_total must be non-negative")
    if max_episodes is not None and max_episodes < 0:
        raise ValueError("max-episodes must be non-negative")
    if ep_from < 0:
        raise ValueError("ep-from must be non-negative")
    if ep_to is not None and ep_to < 0:
        raise ValueError("ep-to must be non-negative")

    selected_total = min(max_episodes, n_total) if max_episodes is not None else n_total
    stop = selected_total if ep_to is None else min(ep_to, selected_total)

    if ep_from > selected_total:
        raise ValueError(
            f"ep-from={ep_from} is outside the selected episodes [0, {selected_total})"
        )
    if stop < ep_from:
        raise ValueError(f"ep-to={ep_to} must be greater than or equal to ep-from={ep_from}")

    return list(range(ep_from, stop))


def episode_output_path(out_dir: Path, ep_idx: int) -> Path:
    return out_dir / f"episode_{ep_idx:05d}.pt"


def is_existing_episode_usable(path: Path) -> bool:
    """Return whether an existing torch-save archive is readable enough to keep."""
    return path.exists() and zipfile.is_zipfile(path)


def should_process_episode(out_dir: Path, ep_idx: int, *, skip_existing: bool) -> bool:
    path = episode_output_path(out_dir, ep_idx)
    return not (skip_existing and is_existing_episode_usable(path))


def atomic_save_episode(record: EpisodeRecordLike, out_dir: Path, ep_idx: int) -> Path:
    """Save an episode through a temporary file, then atomically publish it."""
    final_path = episode_output_path(out_dir, ep_idx)
    tmp_path = final_path.with_name(f"{final_path.name}.tmp.{os.getpid()}")
    try:
        record.save(tmp_path)
        tmp_path.replace(final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return final_path
