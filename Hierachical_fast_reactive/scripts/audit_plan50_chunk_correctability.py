#!/usr/bin/env python
"""Audit whether Plan-50 SmolVLA chunks are correctable by a bounded residual.

This is an offline diagnostic. It does not train, evaluate in LIBERO, or depend
on the derived fast-cache. For each selected episode, it queries the frozen
slow planner once every ``chunk_len`` frames, compares ``a_base_chunk[k]`` with
the expert action at that frame, and reports how much of the error can fit
inside candidate per-DoF residual clip budgets.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_SRC = REPO_ROOT / "policy" / "lerobot_policy_hfrvla" / "src"

ACTION_GROUPS: dict[str, slice] = {
    "all": slice(0, 7),
    "dims_0_2": slice(0, 3),
    "dims_3_5": slice(3, 6),
    "dim_6": slice(6, 7),
}

SUITE_TASK_INDICES: dict[str, set[int] | None] = {
    "all": None,
    "libero_10": set(range(0, 10)),
    "libero_spatial": set(range(10, 20)),
    "libero_object": set(range(20, 30)),
    "libero_goal": set(range(30, 40)),
}


def budget_label(value: float) -> str:
    return f"b{value:g}".replace("-", "m").replace(".", "p")


def parse_csv_floats(text: str | Iterable[float]) -> list[float]:
    if isinstance(text, str):
        values = [item.strip() for item in text.split(",") if item.strip()]
        return [float(item) for item in values]
    return [float(item) for item in text]


def parse_csv_ints(text: str | Iterable[int] | None) -> list[int] | None:
    if text is None:
        return None
    if isinstance(text, str):
        if not text.strip():
            return None
        values = [item.strip() for item in text.split(",") if item.strip()]
        return [int(item) for item in values]
    return [int(item) for item in text]


def task_indices_for_suite(suite: str) -> set[int] | None:
    key = suite.strip().lower()
    if key not in SUITE_TASK_INDICES:
        supported = ", ".join(sorted(SUITE_TASK_INDICES))
        raise ValueError(f"unknown suite {suite!r}; supported: {supported}")
    allowed = SUITE_TASK_INDICES[key]
    return None if allowed is None else set(allowed)


def cosine_similarity(base: np.ndarray, expert: np.ndarray, eps: float = 1e-8) -> float:
    base = np.asarray(base, dtype=np.float64).reshape(-1)
    expert = np.asarray(expert, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(base) * np.linalg.norm(expert))
    if denom < eps:
        return 0.0
    return float(np.dot(base, expert) / denom)


def compute_frame_metric_rows(
    *,
    base_action: np.ndarray,
    expert_action: np.ndarray,
    budgets: Iterable[float],
    episode_index: int,
    frame_index: int,
    task_index: int,
    task: str,
    k: int,
) -> list[dict[str, Any]]:
    """Return one metric row per action group for a single frame."""
    base = np.asarray(base_action, dtype=np.float64).reshape(-1)
    expert = np.asarray(expert_action, dtype=np.float64).reshape(-1)
    if base.shape != expert.shape:
        raise ValueError(f"base/expert action shapes differ: {base.shape} vs {expert.shape}")
    if base.shape[-1] < 7:
        raise ValueError(f"expected at least 7 action dims, got {base.shape[-1]}")

    rows: list[dict[str, Any]] = []
    for group, group_slice in ACTION_GROUPS.items():
        base_g = base[group_slice]
        expert_g = expert[group_slice]
        delta = expert_g - base_g
        row: dict[str, Any] = {
            "episode_index": int(episode_index),
            "frame_index": int(frame_index),
            "task_index": int(task_index),
            "task": task,
            "k": int(k),
            "group": group,
            "mse": float(np.mean(delta * delta)),
            "target_delta_norm": float(np.linalg.norm(delta)),
            "cosine_base_expert": cosine_similarity(base_g, expert_g),
            "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
        }
        for budget in budgets:
            label = budget_label(float(budget))
            abs_delta = np.abs(delta)
            row[f"clip_dim_fraction_{label}"] = float(np.mean(abs_delta > float(budget)))
            row[f"correctable_action_{label}"] = float(np.all(abs_delta <= float(budget)))
        rows.append(row)
    return rows


def aggregate_metric_rows(
    rows: list[dict[str, Any]],
    *,
    group_by: tuple[str, ...],
    budgets: Iterable[float],
    late_start: int,
) -> list[dict[str, Any]]:
    metric_fields = [
        "mse",
        "target_delta_norm",
        "cosine_base_expert",
        "max_abs_delta",
    ]
    for budget in budgets:
        label = budget_label(float(budget))
        metric_fields.extend(
            [
                f"clip_dim_fraction_{label}",
                f"correctable_action_{label}",
            ]
        )

    buckets: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_by)
        if key not in buckets:
            buckets[key] = {
                "n_frames": 0,
                "early_mse_sum": 0.0,
                "early_n": 0,
                "late_mse_sum": 0.0,
                "late_n": 0,
                **{f"{field}_sum": 0.0 for field in metric_fields},
            }
            for field, value in zip(group_by, key, strict=True):
                buckets[key][field] = value
        bucket = buckets[key]
        bucket["n_frames"] += 1
        for field in metric_fields:
            bucket[f"{field}_sum"] += float(row[field])
        if int(row["k"]) >= late_start:
            bucket["late_mse_sum"] += float(row["mse"])
            bucket["late_n"] += 1
        else:
            bucket["early_mse_sum"] += float(row["mse"])
            bucket["early_n"] += 1

    out: list[dict[str, Any]] = []
    for key in sorted(buckets):
        bucket = buckets[key]
        n = max(1, int(bucket["n_frames"]))
        item = {field: bucket[field] for field in group_by}
        item["n_frames"] = int(bucket["n_frames"])
        for field in metric_fields:
            item[field] = float(bucket[f"{field}_sum"] / n)
        early_n = int(bucket["early_n"])
        late_n = int(bucket["late_n"])
        item["early_mse"] = (
            float(bucket["early_mse_sum"] / early_n) if early_n else None
        )
        item["late_mse"] = float(bucket["late_mse_sum"] / late_n) if late_n else None
        item["late_minus_early_mse"] = (
            item["late_mse"] - item["early_mse"]
            if item["late_mse"] is not None and item["early_mse"] is not None
            else None
        )
        out.append(item)
    return out


def classify_soft_gate(
    per_group_rows: list[dict[str, Any]],
    *,
    budgets: Iterable[float],
) -> dict[str, Any]:
    """Return a soft green/yellow/red diagnostic from the all-dim group."""
    budgets = list(parse_csv_floats(budgets))
    all_row = next((row for row in per_group_rows if row.get("group") == "all"), None)
    if not all_row:
        return {"status": "red", "reason": "no all-dim rows"}

    def _nearest_budget(target: float) -> float:
        return min(budgets, key=lambda value: abs(value - target))

    b02 = _nearest_budget(0.2)
    b04 = _nearest_budget(0.4)
    c02 = float(all_row.get(f"correctable_action_{budget_label(b02)}", 0.0))
    c04 = float(all_row.get(f"correctable_action_{budget_label(b04)}", 0.0))
    late_gap = all_row.get("late_minus_early_mse")
    if c02 >= 0.50 or c04 >= 0.75:
        status = "green"
        reason = "many Plan-50 errors fit inside the tested residual budget"
    elif c04 >= 0.50 or (c04 - c02 >= 0.20 and c04 >= 0.35):
        status = "yellow"
        reason = "correctability improves with a wider budget; inspect late-chunk drift"
    else:
        status = "red"
        reason = "most Plan-50 errors exceed the tested residual budgets"

    return {
        "status": status,
        "reason": reason,
        "budget_for_0p2_column": b02,
        "budget_for_0p4_column": b04,
        "correctable_at_0p2": c02,
        "correctable_at_0p4": c04,
        "late_minus_early_mse": late_gap,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_safe(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _setup_local_env() -> None:
    cache_root = Path(
        os.environ.get("HFRVLA_CACHE_ROOT", Path.home() / "tmp" / "hfrvla")
    ).expanduser()
    hf_cache = cache_root / "hf_datasets"
    tmpdir = cache_root / "tmp"
    hf_cache.mkdir(parents=True, exist_ok=True)
    tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache))
    os.environ.setdefault("TMPDIR", str(tmpdir))
    if POLICY_SRC.exists():
        sys.path.insert(0, str(POLICY_SRC))


def _to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def _episode_bounds(src: Any, ep_idx: int) -> tuple[int, int, str]:
    ep_meta = src.meta.episodes[ep_idx]
    ep_from = int(ep_meta["dataset_from_index"])
    ep_to = int(ep_meta["dataset_to_index"])
    ep_task = (
        ep_meta["tasks"][0]
        if isinstance(ep_meta.get("tasks"), list) and ep_meta["tasks"]
        else "do the task"
    )
    return ep_from, ep_to, ep_task


def _scalar_int(value: Any, default: int = -1) -> int:
    import torch

    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return int(value.detach().cpu().reshape(-1)[0].item())
    return int(value)


def _tensor_to_numpy(value: Any) -> np.ndarray:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)


def _actions_to_chunk(actions: Any, chunk_len: int) -> np.ndarray:
    import torch

    if not isinstance(actions, torch.Tensor):
        raise TypeError(f"expected torch.Tensor actions, got {type(actions)!r}")
    x = actions.detach().cpu().float()
    if x.dim() == 3:
        if x.shape[0] == 1:
            chunk = x[0]
        elif x.shape[1] == 1:
            chunk = x[:, 0]
        else:
            chunk = x.transpose(0, 1)[:, 0]
    elif x.dim() == 2:
        chunk = x
    else:
        raise ValueError(f"unsupported action chunk shape {tuple(x.shape)}")
    chunk = chunk[:chunk_len]
    if chunk.dim() != 2 or chunk.shape[-1] < 7:
        raise ValueError(f"expected action chunk shape (T, >=7), got {tuple(chunk.shape)}")
    return chunk.numpy()


def _prepare_smolvla_batch(
    *,
    sample: dict[str, Any],
    task: str,
    pre_processor: Any,
    policy: Any,
    device: Any,
) -> dict[str, Any]:
    import torch

    sample_with_task = dict(sample)
    sample_with_task.setdefault("task", task)
    try:
        batch = pre_processor(sample_with_task)
    except Exception:
        batch = {
            key: (value.unsqueeze(0) if isinstance(value, torch.Tensor) else value)
            for key, value in sample_with_task.items()
        }
    if not isinstance(batch, dict):
        batch = dict(batch)
    batch = _to_device(batch, device)
    return policy._prepare_batch(batch)


def _predict_action_chunk(
    *,
    src: Any,
    policy: Any,
    pre_processor: Any,
    device: Any,
    frame_index: int,
    task: str,
    chunk_len: int,
) -> np.ndarray:
    import torch
    from lerobot.policies.utils import populate_queues
    from lerobot.utils.constants import ACTION

    sample = src[frame_index]
    batch = _prepare_smolvla_batch(
        sample=sample,
        task=task,
        pre_processor=pre_processor,
        policy=policy,
        device=device,
    )
    policy._queues = populate_queues(policy._queues, batch, exclude_keys=[ACTION])
    policy._clear_hook_cache()
    with torch.no_grad():
        actions = policy._get_action_chunk(batch)
    return _actions_to_chunk(actions, chunk_len)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=Path("checkpoints/HFRVLA_libero_v1_merged_reindexed"),
        help="Local canonical HFRVLA LeRobotDataset root.",
    )
    parser.add_argument("--source-repo-id", default="HFRVLA_libero_v1")
    parser.add_argument("--smolvla", default=None)
    parser.add_argument("--suite", default="libero_spatial", choices=sorted(SUITE_TASK_INDICES))
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Comma-separated canonical dataset task_index values. Overrides --suite.",
    )
    parser.add_argument("--chunk-len", type=int, default=50)
    parser.add_argument("--delta-budgets", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dinov3-repo", default="checkpoints/dinov3_src")
    parser.add_argument(
        "--dinov3-weights",
        default="checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    )
    parser.add_argument("--dinov3-arch", default="dinov3_vits16")
    parser.add_argument("--allow-feature-remap", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plan50_chunk_correctability"),
    )
    return parser.parse_args(argv)


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    _setup_local_env()

    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_STATE

    from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig
    from lerobot_policy_hfrvla.modeling_hfrvla import HFRVLAPolicy

    try:
        from scripts.hfrvla_alignment_utils import (
            DEFAULT_LIBERO_SMOLVLA,
            load_policy_config_json,
            warn_or_validate_libero_slow_planner,
        )
    except ModuleNotFoundError:
        from hfrvla_alignment_utils import (
            DEFAULT_LIBERO_SMOLVLA,
            load_policy_config_json,
            warn_or_validate_libero_slow_planner,
        )

    budgets = parse_csv_floats(args.delta_budgets)
    if not budgets:
        raise SystemExit("[audit] --delta-budgets must contain at least one value")
    if args.chunk_len <= 0:
        raise SystemExit("[audit] --chunk-len must be positive")

    explicit_task_ids = parse_csv_ints(args.task_ids)
    allowed_task_indices = (
        set(explicit_task_ids)
        if explicit_task_ids is not None
        else task_indices_for_suite(args.suite)
    )
    smolvla = args.smolvla or DEFAULT_LIBERO_SMOLVLA

    raw_smolvla_config = load_policy_config_json(smolvla)
    try:
        warn_or_validate_libero_slow_planner(
            raw_smolvla_config,
            source=smolvla,
            allow_feature_remap=args.allow_feature_remap,
        )
    except ValueError as exc:
        raise SystemExit(f"[audit] {exc}") from exc

    device = torch.device(args.device)
    print(f"[audit] loading source dataset: {args.source_dataset}", flush=True)
    src = LeRobotDataset(
        args.source_repo_id,
        root=args.source_dataset,
    )

    print(f"[audit] building frozen slow planner from {smolvla}", flush=True)
    config = HFRVLAConfig.from_smolvla(
        smolvla,
        dinov3_local_repo=args.dinov3_repo,
        dinov3_local_weights=args.dinov3_weights,
        dinov3_arch=args.dinov3_arch,
        device=args.device,
    )
    config.input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    policy = HFRVLAPolicy.from_pretrained(smolvla, config=config).to(device).eval()
    pre_processor, _ = make_smolvla_pre_post_processors(
        config,
        dataset_stats=getattr(src.meta, "stats", None),
    )

    rows: list[dict[str, Any]] = []
    processed_episodes = 0
    skipped_by_task = 0
    task_counts: dict[int, int] = defaultdict(int)
    n_total = int(src.num_episodes)

    for ep_idx in range(n_total):
        ep_from, ep_to, ep_task = _episode_bounds(src, ep_idx)
        first_sample = src[ep_from]
        task_index = _scalar_int(first_sample.get("task_index"), default=-1)
        if allowed_task_indices is not None and task_index not in allowed_task_indices:
            skipped_by_task += 1
            continue
        if args.max_episodes is not None and processed_episodes >= args.max_episodes:
            break

        processed_episodes += 1
        task_counts[task_index] += 1
        policy.reset()

        for chunk_start in range(ep_from, ep_to, args.chunk_len):
            chunk = _predict_action_chunk(
                src=src,
                policy=policy,
                pre_processor=pre_processor,
                device=device,
                frame_index=chunk_start,
                task=ep_task,
                chunk_len=args.chunk_len,
            )
            usable = min(args.chunk_len, ep_to - chunk_start, int(chunk.shape[0]))
            for k in range(usable):
                frame_index = chunk_start + k
                sample = src[frame_index]
                frame_task_index = _scalar_int(sample.get("task_index"), default=task_index)
                expert = _tensor_to_numpy(sample[args.action_key])
                rows.extend(
                    compute_frame_metric_rows(
                        base_action=chunk[k],
                        expert_action=expert,
                        budgets=budgets,
                        episode_index=ep_idx,
                        frame_index=frame_index,
                        task_index=frame_task_index,
                        task=ep_task,
                        k=k,
                    )
                )

        if args.log_every > 0 and processed_episodes % args.log_every == 0:
            print(
                f"[audit] processed {processed_episodes} selected episodes "
                f"({len(rows) // len(ACTION_GROUPS)} frames)",
                flush=True,
            )

    late_start = args.chunk_len // 2
    per_k = aggregate_metric_rows(
        rows,
        group_by=("group", "k"),
        budgets=budgets,
        late_start=late_start,
    )
    per_task = aggregate_metric_rows(
        rows,
        group_by=("group", "task_index", "task"),
        budgets=budgets,
        late_start=late_start,
    )
    per_group = aggregate_metric_rows(
        rows,
        group_by=("group",),
        budgets=budgets,
        late_start=late_start,
    )
    gate = classify_soft_gate(per_group, budgets=budgets)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    budget_fields: list[str] = []
    for budget in budgets:
        label = budget_label(budget)
        budget_fields.extend([f"clip_dim_fraction_{label}", f"correctable_action_{label}"])
    aggregate_fields = [
        "group",
        "n_frames",
        "mse",
        "target_delta_norm",
        "cosine_base_expert",
        "max_abs_delta",
        *budget_fields,
        "early_mse",
        "late_mse",
        "late_minus_early_mse",
    ]
    write_csv(args.output_dir / "per_k.csv", per_k, ["group", "k", *aggregate_fields[1:]])
    write_csv(
        args.output_dir / "per_task.csv",
        per_task,
        ["group", "task_index", "task", *aggregate_fields[1:]],
    )
    write_csv(args.output_dir / "per_group.csv", per_group, aggregate_fields)

    n_frames = len(rows) // len(ACTION_GROUPS)
    summary = {
        "source_dataset": str(args.source_dataset),
        "source_repo_id": args.source_repo_id,
        "smolvla": smolvla,
        "suite": args.suite,
        "task_ids": sorted(allowed_task_indices) if allowed_task_indices is not None else None,
        "chunk_len": args.chunk_len,
        "delta_budgets": budgets,
        "processed_episodes": processed_episodes,
        "skipped_episodes_by_task": skipped_by_task,
        "task_episode_counts": {str(k): v for k, v in sorted(task_counts.items())},
        "n_frames": n_frames,
        "n_group_rows": len(rows),
        "soft_gate": gate,
        "outputs": {
            "per_k": str(args.output_dir / "per_k.csv"),
            "per_task": str(args.output_dir / "per_task.csv"),
            "per_group": str(args.output_dir / "per_group.csv"),
            "summary": str(args.output_dir / "summary.json"),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[audit] done: {n_frames} frames, {processed_episodes} episodes, "
        f"soft_gate={gate['status']} -> {args.output_dir}",
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_audit(args)


if __name__ == "__main__":
    main()
