from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


from legacy.precompute_libero_plan import (  # noqa: E402
    atomic_save_episode,
    episode_output_path,
    is_existing_episode_usable,
    resolve_episode_indices,
    should_process_episode,
)


def test_resolve_episode_indices_honors_explicit_shard() -> None:
    assert resolve_episode_indices(
        n_total=10,
        max_episodes=None,
        ep_from=2,
        ep_to=5,
    ) == [2, 3, 4]


def test_resolve_episode_indices_caps_shard_by_max_episodes() -> None:
    assert resolve_episode_indices(
        n_total=10,
        max_episodes=6,
        ep_from=4,
        ep_to=None,
    ) == [4, 5]


def test_resolve_episode_indices_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="ep-from"):
        resolve_episode_indices(n_total=10, max_episodes=None, ep_from=-1, ep_to=None)

    with pytest.raises(ValueError, match="ep-to"):
        resolve_episode_indices(n_total=10, max_episodes=None, ep_from=5, ep_to=4)

    with pytest.raises(ValueError, match="selected episodes"):
        resolve_episode_indices(n_total=10, max_episodes=3, ep_from=4, ep_to=None)


def test_should_process_episode_skips_existing_when_requested(tmp_path: Path) -> None:
    path = episode_output_path(tmp_path, 7)
    atomic_save_episode(_TextRecord("already done"), tmp_path, 7)

    assert should_process_episode(tmp_path, 7, skip_existing=True) is False
    assert should_process_episode(tmp_path, 7, skip_existing=False) is True
    assert should_process_episode(tmp_path, 8, skip_existing=True) is True


def test_should_process_episode_recomputes_corrupt_existing_file(tmp_path: Path) -> None:
    path = episode_output_path(tmp_path, 7)
    path.write_bytes(b"truncated torch archive")

    assert is_existing_episode_usable(path) is False
    assert should_process_episode(tmp_path, 7, skip_existing=True) is True


def test_atomic_save_episode_replaces_final_file_and_cleans_temp(tmp_path: Path) -> None:
    final_path = episode_output_path(tmp_path, 3)
    final_path.write_text("old record")

    saved_path = atomic_save_episode(_TextRecord("new record"), tmp_path, 3)

    assert saved_path == final_path
    import torch

    assert torch.load(final_path, weights_only=True)["text"] == "new record"
    assert list(tmp_path.glob("episode_00003.pt.tmp.*")) == []


class _TextRecord:
    def __init__(self, text: str) -> None:
        self.text = text

    def save(self, path: Path) -> None:
        import torch

        torch.save({"text": self.text}, path)
