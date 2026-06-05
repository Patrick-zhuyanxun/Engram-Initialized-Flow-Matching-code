#!/usr/bin/env python
"""Wrapper that injects HFRVLA curriculum into lerobot-train.

Lerobot-train's main loop calls update_policy(...) once per optimizer step.
This wrapper monkey-patches that function so it:

1. Unwraps the policy if accelerator/DDP wrapped it.
2. Invokes policy.set_training_step(step) for curriculum control.
3. Drops optimizer LR once when stage 2 starts.
4. Delegates to the original update_policy.
"""

from __future__ import annotations

import os
from pathlib import Path

import lerobot.scripts.lerobot_train as lt
import lerobot.datasets.factory as dataset_factory
import lerobot.datasets.dataset_reader as dataset_reader_module
import pyarrow.dataset as pa_ds
import torch
from lerobot.utils.constants import ACTION, OBS_PREFIX, REWARD
from lerobot.utils.utils import SuppressProgressBars

_STEP = {"value": 0}
_ORIGINAL_UPDATE_POLICY = lt.update_policy
_ORIGINAL_MAKE_DATASET = lt.make_dataset
_ORIGINAL_DATALOADER = torch.utils.data.DataLoader

HFRVLA_OFFLINE_REQUIRED_COLUMNS = {
    "index",
    "episode_index",
    "task_index",
    "timestamp",
    "frame_index",
    ACTION,
    "observation.state",
    "observation.extra.a_base",
    "observation.extra.k_idx_norm",
    "observation.extra.z_goal",
    "observation.extra.z_phase",
    "observation.extra.dino_patches",
    "observation.extra.contact_label",
}
HFRVLA_OFFLINE_OPTIONAL_COLUMNS = {
    "observation.extra.a_base_chunk",
    "observation.extra.chunk_step_idx",
    "observation.extra.chunk_age_steps",
    "observation.extra.chunk_age_norm",
}
HFRVLA_OFFLINE_ALLOWED_COLUMNS = (
    HFRVLA_OFFLINE_REQUIRED_COLUMNS | HFRVLA_OFFLINE_OPTIONAL_COLUMNS
)

_HFRVLA_OFFLINE_REQUIRED_TRAIN_COLUMNS = HFRVLA_OFFLINE_REQUIRED_COLUMNS - {
    "frame_index",
    "timestamp",
}


def _unwrap(policy, accelerator=None):
    """Strip accelerator wrapping when possible."""
    if accelerator is not None and hasattr(accelerator, "unwrap_model"):
        return accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    return getattr(policy, "module", policy)


def _iter_scheduler_chain(lr_scheduler):
    """Yield a scheduler and common wrappers around an inner scheduler."""
    current = lr_scheduler
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "scheduler", None)


def _apply_lr_scale(optimizer, lr_scheduler, *, factor: float) -> bool:
    """Scale optimizer LR and scheduler bases, including Accelerate wrappers."""
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * factor
        if "initial_lr" in group:
            group["initial_lr"] = group["initial_lr"] * factor

    changed_scheduler = False
    for scheduler in _iter_scheduler_chain(lr_scheduler):
        if hasattr(scheduler, "base_lrs"):
            scheduler.base_lrs = [lr * factor for lr in scheduler.base_lrs]
            changed_scheduler = True
        if hasattr(scheduler, "_last_lr"):
            scheduler._last_lr = [lr * factor for lr in scheduler._last_lr]
            changed_scheduler = True

    return changed_scheduler


def _patched_update_policy(train_tracker, policy, batch, optimizer, *args, **kwargs):
    step = _STEP["value"]
    accelerator = kwargs.get("accelerator")
    if accelerator is None and len(args) >= 2:
        accelerator = args[1]
    lr_scheduler = kwargs.get("lr_scheduler")
    if lr_scheduler is None and len(args) >= 3:
        lr_scheduler = args[2]

    raw = _unwrap(policy, accelerator)
    if hasattr(raw, "set_training_step"):
        raw.set_training_step(step)
        consume_signal = getattr(raw, "consume_refine_lr_signal", None)
        if consume_signal is not None and consume_signal():
            _apply_lr_scale(optimizer, lr_scheduler, factor=0.1)
            print(f"[curriculum] step {step}: entered Stage 2 — LR × 0.1", flush=True)

    result = _ORIGINAL_UPDATE_POLICY(train_tracker, policy, batch, optimizer, *args, **kwargs)
    _STEP["value"] += 1
    return result


def _is_offline_hfrvla(cfg) -> bool:
    policy = getattr(cfg, "policy", None)
    return getattr(policy, "type", None) == "hfrvla" and bool(
        getattr(policy, "offline_training_mode", False)
    )


def _dataset_backend() -> str:
    backend = os.environ.get("HFRVLA_DATASET_BACKEND", "lerobot").strip().lower()
    if backend not in {"lerobot", "fastcache"}:
        raise ValueError(
            "HFRVLA_DATASET_BACKEND must be either 'lerobot' or 'fastcache'; "
            f"got {backend!r}"
        )
    return backend


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


class _PatchedDataLoader(_ORIGINAL_DATALOADER):
    def __init__(self, *args, **kwargs):
        if _dataset_backend() == "fastcache":
            num_workers = int(kwargs.get("num_workers", 0) or 0)
            if num_workers > 0:
                if _env_bool("HFRVLA_DATALOADER_PERSISTENT_WORKERS", default=True):
                    kwargs.setdefault("persistent_workers", True)

                prefetch = os.environ.get("HFRVLA_DATALOADER_PREFETCH_FACTOR")
                if prefetch:
                    kwargs["prefetch_factor"] = int(prefetch)

        super().__init__(*args, **kwargs)


def _make_fastcache_dataset(cfg):
    root = os.environ.get("HFRVLA_FASTCACHE_ROOT")
    if not root:
        raise ValueError(
            "HFRVLA_FASTCACHE_ROOT must point to a cache built by "
            "scripts/build_hfrvla_fastcache.py when HFRVLA_DATASET_BACKEND=fastcache"
        )

    from lerobot_policy_hfrvla.fast_cache_dataset import HFRVLAFastCacheDataset

    policy = getattr(cfg, "policy", cfg)
    seq_len = int(getattr(policy, "seq_len", 1))
    residual_mode = str(getattr(policy, "residual_merge_mode", "gated")).strip().lower()
    cache_root = Path(root).expanduser()
    rollout_root_raw = os.environ.get("HFRVLA_FASTCACHE_ROLLOUT_ROOT", "").strip()
    if rollout_root_raw:
        rollout_root = Path(rollout_root_raw).expanduser()
        dataset = HFRVLAFastCacheDataset(roots=[cache_root, rollout_root], seq_len=seq_len)
        _validate_fastcache_for_policy(dataset, residual_mode=residual_mode)
        print(
            "[hfrvla-train] fast-cache dataset backend enabled "
            f"roots=[{cache_root}, {rollout_root}]",
            flush=True,
        )
        return dataset

    dataset = HFRVLAFastCacheDataset(cache_root, seq_len=seq_len)
    _validate_fastcache_for_policy(dataset, residual_mode=residual_mode)
    print(
        f"[hfrvla-train] fast-cache dataset backend enabled root={cache_root}",
        flush=True,
    )
    return dataset


def _validate_fastcache_for_policy(dataset, *, residual_mode: str) -> None:
    if residual_mode == "a2c2":
        residual_mode = "fast_wrist"
    if residual_mode != "fast_wrist_chunk":
        return
    required = {
        "observation.extra.a_base_chunk",
        "observation.extra.chunk_step_idx",
    }
    missing = required - set(dataset.info.get("features", {}))
    if missing:
        raise ValueError(
            "RESIDUAL_MERGE_MODE=fast_wrist_chunk requires fast-cache schema v3 "
            "with full base chunk fields. Missing: " + ", ".join(sorted(missing))
        )


def _offline_hfrvla_delta_timestamps(cfg, ds_meta) -> dict[str, list] | None:
    """Window only the columns consumed by offline fast-module training."""
    policy = getattr(cfg, "policy", cfg)
    delta_timestamps = {}
    for key in ds_meta.features:
        if key not in HFRVLA_OFFLINE_ALLOWED_COLUMNS:
            continue
        reward_delta_indices = getattr(policy, "reward_delta_indices", None)
        action_delta_indices = getattr(policy, "action_delta_indices", None)
        observation_delta_indices = getattr(policy, "observation_delta_indices", None)
        if key == REWARD and reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in reward_delta_indices]
        if key == ACTION and action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in action_delta_indices]
        if key.startswith(OBS_PREFIX) and observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in observation_delta_indices]

    return delta_timestamps or None


def _offline_hfrvla_features(features: dict) -> dict:
    return {
        key: feature
        for key, feature in features.items()
        if key in HFRVLA_OFFLINE_ALLOWED_COLUMNS
    }


def _load_offline_hfrvla_hf_dataset(reader):
    selected_features = _offline_hfrvla_features(reader._meta.features)
    features = dataset_reader_module.get_hf_features_from_features(selected_features)
    paths = sorted((reader.root / "data").glob("*/*.parquet"))
    if len(paths) == 0:
        raise FileNotFoundError(
            f"Provided directory does not contain any parquet file: {reader.root / 'data'}"
        )

    filters = (
        pa_ds.field("episode_index").isin(reader.episodes)
        if reader.episodes is not None
        else None
    )
    columns = list(selected_features.keys())
    with SuppressProgressBars():
        hf_dataset = dataset_reader_module.datasets.Dataset.from_parquet(
            [str(path) for path in paths],
            features=features,
            columns=columns,
            filters=filters,
        )
    hf_dataset.set_transform(dataset_reader_module.hf_transform_to_torch)
    return hf_dataset


def _select_offline_hfrvla_columns(dataset):
    """Drop large raw images after LeRobot has loaded the local HF dataset."""
    reader = getattr(dataset, "reader", None)
    hf_dataset = getattr(reader, "hf_dataset", None)
    if hf_dataset is None or not hasattr(hf_dataset, "select_columns"):
        return dataset

    column_names = list(getattr(hf_dataset, "column_names", []))
    available = set(column_names)
    missing = _HFRVLA_OFFLINE_REQUIRED_TRAIN_COLUMNS - available
    if missing:
        raise KeyError(
            "Offline HFRVLA dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    keep = [column for column in column_names if column in HFRVLA_OFFLINE_ALLOWED_COLUMNS]
    reader.hf_dataset = hf_dataset.select_columns(keep)
    return dataset


def _patched_make_dataset(cfg):
    if not _is_offline_hfrvla(cfg):
        return _ORIGINAL_MAKE_DATASET(cfg)

    if _dataset_backend() == "fastcache":
        return _make_fastcache_dataset(cfg)

    original_resolve_delta_timestamps = dataset_factory.resolve_delta_timestamps
    original_load_hf_dataset = dataset_reader_module.DatasetReader._load_hf_dataset
    try:
        dataset_factory.resolve_delta_timestamps = _offline_hfrvla_delta_timestamps
        dataset_reader_module.DatasetReader._load_hf_dataset = _load_offline_hfrvla_hf_dataset
        dataset = _ORIGINAL_MAKE_DATASET(cfg)
    finally:
        dataset_factory.resolve_delta_timestamps = original_resolve_delta_timestamps
        dataset_reader_module.DatasetReader._load_hf_dataset = original_load_hf_dataset

    dataset = _select_offline_hfrvla_columns(dataset)
    print("[hfrvla-train] offline dataset column pruning enabled", flush=True)
    return dataset


def main() -> None:
    lt.update_policy = _patched_update_policy
    lt.make_dataset = _patched_make_dataset
    torch.utils.data.DataLoader = _PatchedDataLoader
    lt.main()


if __name__ == "__main__":
    main()
