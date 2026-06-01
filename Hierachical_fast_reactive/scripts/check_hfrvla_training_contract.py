#!/usr/bin/env python
"""Pre-training contract check for HFRVLA LeRobotDataset samples.

This is the train-time counterpart to ``scripts/test_alignment.py``.

It verifies that an already recorded HFRVLA dataset can be loaded through the
same LeRobot windowing + preprocessing path used by ``lerobot-train`` and that
the fast-module training target is structurally valid:

    target_delta = normalized(action) - normalized(a_base)

The check intentionally uses a zero residual output. It is not a performance
test; it confirms that the tensors a future training run will optimize over
have the expected keys, shapes, finite values, and matching 7D action space.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
HFRVLA_TMP_ROOT = Path(
    os.environ.get("HFRVLA_TMP_ROOT", Path.home() / "tmp" / "hfrvla")
).expanduser()
HFRVLA_HF_DATASETS_CACHE = Path(
    os.environ.get("HF_DATASETS_CACHE", HFRVLA_TMP_ROOT / "hf_datasets")
).expanduser()
HFRVLA_TMPDIR = Path(os.environ.get("TMPDIR", HFRVLA_TMP_ROOT / "tmp")).expanduser()
os.environ.setdefault("HF_DATASETS_CACHE", str(HFRVLA_HF_DATASETS_CACHE))
os.environ.setdefault("TMPDIR", str(HFRVLA_TMPDIR))
os.environ.setdefault("TMP", str(HFRVLA_TMPDIR))
os.environ.setdefault("TEMP", str(HFRVLA_TMPDIR))

POLICY_SRC = REPO_ROOT / "policy/lerobot_policy_hfrvla/src"
if str(POLICY_SRC) not in sys.path:
    sys.path.insert(0, str(POLICY_SRC))

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig

try:
    from scripts.hfrvla_alignment_utils import (
        load_policy_config_json,
        warn_or_validate_libero_slow_planner,
    )
except ModuleNotFoundError:
    from hfrvla_alignment_utils import (
        load_policy_config_json,
        warn_or_validate_libero_slow_planner,
    )


EXTRA_A_BASE = "observation.extra.a_base"
EXTRA_K_IDX_NORM = "observation.extra.k_idx_norm"
EXTRA_Z_GOAL = "observation.extra.z_goal"
EXTRA_Z_PHASE = "observation.extra.z_phase"
EXTRA_DINO_PATCHES = "observation.extra.dino_patches"
EXTRA_CONTACT_LABEL = "observation.extra.contact_label"


def _ensure_cache_dirs() -> None:
    HFRVLA_HF_DATASETS_CACHE.mkdir(parents=True, exist_ok=True)
    HFRVLA_TMPDIR.mkdir(parents=True, exist_ok=True)

REQUIRED_FEATURE_SHAPES: dict[str, tuple[int, ...]] = {
    "observation.images.image": (256, 256, 3),
    "observation.images.image2": (256, 256, 3),
    OBS_STATE: (8,),
    ACTION: (7,),
    EXTRA_Z_GOAL: (960,),
    EXTRA_Z_PHASE: (480,),
    EXTRA_A_BASE: (7,),
    EXTRA_K_IDX_NORM: (1,),
    EXTRA_DINO_PATCHES: (196, 384),
    EXTRA_CONTACT_LABEL: (1,),
}


class TrainingContractError(ValueError):
    """Raised when the recorded dataset cannot feed HFRVLA training safely."""


def _feature_shape(feature: Any) -> tuple[int, ...] | None:
    if isinstance(feature, dict):
        shape = feature.get("shape")
    else:
        shape = getattr(feature, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def validate_feature_shapes(
    features: dict[str, Any],
    expected: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, tuple[int, ...]]:
    """Validate the raw dataset metadata feature contract."""
    expected = expected or REQUIRED_FEATURE_SHAPES
    seen: dict[str, tuple[int, ...]] = {}
    errors: list[str] = []

    for key, wanted in expected.items():
        if key not in features:
            errors.append(f"{key}: missing")
            continue
        got = _feature_shape(features[key])
        if got is None:
            errors.append(f"{key}: missing shape metadata")
            continue
        seen[key] = got
        if got != wanted:
            errors.append(f"{key}: expected shape {wanted}, got {got}")

    if errors:
        raise TrainingContractError(
            "HFRVLA dataset feature contract failed:\n  - " + "\n  - ".join(errors)
        )
    return seen


def _require_tensor(batch: dict[str, Any], key: str) -> torch.Tensor:
    if key not in batch:
        raise TrainingContractError(f"processed batch is missing `{key}`")
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise TrainingContractError(f"`{key}` must be a torch.Tensor, got {type(value).__name__}")
    if not torch.isfinite(value.float()).all():
        raise TrainingContractError(f"`{key}` contains non-finite values")
    return value


def _check_windowed_shape(
    batch: dict[str, Any],
    key: str,
    trailing_shape: tuple[int, ...],
    seq_len: int,
    *,
    allow_scalar_collapse: bool = False,
) -> tuple[int, ...]:
    tensor = _require_tensor(batch, key)
    shape = tuple(int(dim) for dim in tensor.shape)
    if allow_scalar_collapse and trailing_shape == (1,) and shape and shape[-1] == seq_len:
        return shape
    if len(shape) < len(trailing_shape) + 1:
        raise TrainingContractError(
            f"`{key}` should include a time dimension before {trailing_shape}, got {shape}"
        )
    if shape[-len(trailing_shape):] != trailing_shape:
        raise TrainingContractError(
            f"`{key}` expected trailing shape {trailing_shape}, got {shape}"
        )
    time_dim = shape[-(len(trailing_shape) + 1)]
    if time_dim != seq_len:
        raise TrainingContractError(
            f"`{key}` expected seq_len {seq_len}, got time dimension {time_dim} in {shape}"
        )
    return shape


def validate_windowed_batch_shapes(
    batch: dict[str, Any],
    *,
    seq_len: int,
    z_goal_dim: int = 960,
    z_phase_dim: int = 480,
    dino_num_patches: int = 196,
    dino_feature_dim: int = 384,
) -> dict[str, tuple[int, ...]]:
    """Validate the processed sample shape after LeRobot delta windows."""
    expected = {
        OBS_STATE: (8,),
        ACTION: (7,),
        EXTRA_A_BASE: (7,),
        EXTRA_K_IDX_NORM: (1,),
        EXTRA_Z_GOAL: (z_goal_dim,),
        EXTRA_Z_PHASE: (z_phase_dim,),
        EXTRA_DINO_PATCHES: (dino_num_patches, dino_feature_dim),
        EXTRA_CONTACT_LABEL: (1,),
    }
    scalar_collapse_keys = {EXTRA_K_IDX_NORM, EXTRA_CONTACT_LABEL}
    return {
        key: _check_windowed_shape(
            batch,
            key,
            trailing,
            seq_len,
            allow_scalar_collapse=key in scalar_collapse_keys,
        )
        for key, trailing in expected.items()
    }


def _ensure_k_idx_feature_dim(k_idx_norm: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if k_idx_norm.dim() == reference.dim() - 1:
        return k_idx_norm.unsqueeze(-1)
    return k_idx_norm


def compute_zero_residual_contract(batch: dict[str, Any]) -> dict[str, Any]:
    """Compute the HFRVLA target contract with delta_a fixed to zero."""
    action = _require_tensor(batch, ACTION).float()
    a_base = _require_tensor(batch, EXTRA_A_BASE).float()
    if action.shape != a_base.shape:
        raise TrainingContractError(
            f"`{EXTRA_A_BASE}` shape must match normalized `action`; "
            f"got a_base={tuple(a_base.shape)} action={tuple(action.shape)}"
        )
    if action.shape[-1] != 7:
        raise TrainingContractError(f"`action` must be 7D, got shape {tuple(action.shape)}")

    k_idx_norm = _ensure_k_idx_feature_dim(
        _require_tensor(batch, EXTRA_K_IDX_NORM).float(),
        reference=a_base,
    )
    expected_k_shape = a_base.shape[:-1] + (1,)
    if k_idx_norm.shape != expected_k_shape:
        raise TrainingContractError(
            f"`{EXTRA_K_IDX_NORM}` expected shape {tuple(expected_k_shape)}, "
            f"got {tuple(k_idx_norm.shape)}"
        )
    if k_idx_norm.numel() and (
        k_idx_norm.min().item() < -1e-5 or k_idx_norm.max().item() > 1.0 + 1e-5
    ):
        raise TrainingContractError(
            f"`{EXTRA_K_IDX_NORM}` must be in [0, 1], "
            f"got min={k_idx_norm.min().item():.4f} max={k_idx_norm.max().item():.4f}"
        )

    contact_label = batch.get(EXTRA_CONTACT_LABEL)
    contact_shape = None
    if isinstance(contact_label, torch.Tensor):
        contact = contact_label.float()
        if contact.dim() == action.dim() and contact.shape[-1] == 1:
            contact = contact.squeeze(-1)
        if contact.shape != action.shape[:-1]:
            raise TrainingContractError(
                f"`{EXTRA_CONTACT_LABEL}` expected {tuple(action.shape[:-1])} "
                f"or {tuple(action.shape[:-1] + (1,))}, got {tuple(contact_label.shape)}"
            )
        if contact.numel() and (
            contact.min().item() < -1e-5 or contact.max().item() > 1.0 + 1e-5
        ):
            raise TrainingContractError(
                f"`{EXTRA_CONTACT_LABEL}` must be in [0, 1], "
                f"got min={contact.min().item():.4f} max={contact.max().item():.4f}"
            )
        contact_shape = tuple(int(dim) for dim in contact_label.shape)

    target_delta = action - a_base
    zero_delta = torch.zeros_like(a_base)
    zero_delta_mse = torch.mean((zero_delta - target_delta) ** 2)
    if not torch.isfinite(zero_delta_mse):
        raise TrainingContractError("zero-residual MSE is not finite")

    return {
        "action_shape": tuple(int(dim) for dim in action.shape),
        "a_base_shape": tuple(int(dim) for dim in a_base.shape),
        "target_delta_shape": tuple(int(dim) for dim in target_delta.shape),
        "k_idx_norm_shape": tuple(int(dim) for dim in k_idx_norm.shape),
        "contact_label_shape": contact_shape,
        "zero_delta_mse": float(zero_delta_mse.item()),
        "action_abs_max": float(action.abs().max().item()),
        "a_base_abs_max": float(a_base.abs().max().item()),
        "target_delta_abs_mean": float(target_delta.abs().mean().item()),
        "target_delta_abs_max": float(target_delta.abs().max().item()),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(val) for val in value]
    return value


def build_contract_config(args: argparse.Namespace) -> HFRVLAConfig:
    cfg = HFRVLAConfig(
        seq_len=args.seq_len,
        device=args.device,
        dinov3_num_patches=args.dino_num_patches,
        dinov3_feature_dim=args.dino_feature_dim,
    )
    cfg.input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    cfg.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-repo-id", default="HFRVLA_libero_v1")
    parser.add_argument(
        "--dataset-root",
        default=str(REPO_ROOT / "checkpoints/HFRVLA_libero_v1"),
        help="Local HFRVLA LeRobotDataset root.",
    )
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--z-goal-dim", type=int, default=960)
    parser.add_argument("--z-phase-dim", type=int, default=480)
    parser.add_argument("--dino-num-patches", type=int, default=196)
    parser.add_argument("--dino-feature-dim", type=int, default=384)
    parser.add_argument(
        "--smolvla",
        default=None,
        help="Optional LIBERO-adapted slow-planner checkpoint to feature-check alongside the dataset.",
    )
    parser.add_argument(
        "--allow-feature-remap",
        action="store_true",
        help="Debug-only escape hatch when checking a non-LIBERO SmolVLA config.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    _ensure_cache_dirs()
    args = parse_args()

    if args.smolvla is not None:
        raw_smolvla_config = load_policy_config_json(args.smolvla)
        warn_or_validate_libero_slow_planner(
            raw_smolvla_config,
            source=args.smolvla,
            allow_feature_remap=args.allow_feature_remap,
        )

    expected_shapes = dict(REQUIRED_FEATURE_SHAPES)
    expected_shapes[EXTRA_Z_GOAL] = (args.z_goal_dim,)
    expected_shapes[EXTRA_Z_PHASE] = (args.z_phase_dim,)
    expected_shapes[EXTRA_DINO_PATCHES] = (args.dino_num_patches, args.dino_feature_dim)

    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    metadata_shapes = validate_feature_shapes(meta.features, expected=expected_shapes)

    cfg = build_contract_config(args)
    delta_timestamps = resolve_delta_timestamps(cfg, meta)
    if not delta_timestamps:
        raise TrainingContractError("HFRVLAConfig did not produce LeRobot delta_timestamps")

    episodes = [args.episode_index] if args.episode_index is not None else None
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        download_videos=False,
    )
    if len(dataset) == 0:
        raise TrainingContractError("dataset selection is empty")
    sample_index = args.sample_index
    if sample_index is None:
        sample_index = min(max(args.seq_len - 1, 0), len(dataset) - 1)
    if sample_index < 0 or sample_index >= len(dataset):
        raise TrainingContractError(
            f"--sample-index {sample_index} outside selected dataset length {len(dataset)}"
        )

    # Mirror ``lerobot-train``: DataLoader collates samples into a batch before
    # the policy preprocessor runs. Passing a single item directly would miss
    # the batch dimension and produce shapes that differ from training.
    sample = torch.utils.data.default_collate([dataset[sample_index]])
    preprocessor, _ = make_pre_post_processors(policy_cfg=cfg, dataset_stats=meta.stats)
    batch = preprocessor(sample)
    if not isinstance(batch, dict):
        batch = dict(batch)

    batch_shapes = validate_windowed_batch_shapes(
        batch,
        seq_len=args.seq_len,
        z_goal_dim=args.z_goal_dim,
        z_phase_dim=args.z_phase_dim,
        dino_num_patches=args.dino_num_patches,
        dino_feature_dim=args.dino_feature_dim,
    )
    zero_report = compute_zero_residual_contract(batch)

    report = {
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root),
        "episode_index": args.episode_index,
        "sample_index": sample_index,
        "seq_len": args.seq_len,
        "metadata_shapes": metadata_shapes,
        "delta_timestamps_keys": sorted(delta_timestamps),
        "processed_batch_shapes": batch_shapes,
        "zero_residual_contract": zero_report,
    }

    print("\n[hfrvla-contract] metadata feature shapes: OK")
    print(
        f"[hfrvla-contract] LeRobot delta_timestamps: {len(delta_timestamps)} keys, "
        f"seq_len={args.seq_len}"
    )
    print(
        f"[hfrvla-contract] checked episode={args.episode_index} "
        f"sample={sample_index} after training preprocessor"
    )
    for key in (
        OBS_STATE,
        ACTION,
        EXTRA_A_BASE,
        EXTRA_K_IDX_NORM,
        EXTRA_Z_GOAL,
        EXTRA_Z_PHASE,
        EXTRA_DINO_PATCHES,
        EXTRA_CONTACT_LABEL,
    ):
        print(f"  {key}: {batch_shapes[key]}")
    print(
        "[hfrvla-contract] zero-fast target: "
        f"target_delta={zero_report['target_delta_shape']} "
        f"mse={zero_report['zero_delta_mse']:.6g} "
        f"mean_abs={zero_report['target_delta_abs_mean']:.6g} "
        f"max_abs={zero_report['target_delta_abs_max']:.6g}"
    )
    print(
        "[hfrvla-contract] normalized ranges: "
        f"|action|max={zero_report['action_abs_max']:.6g} "
        f"|a_base|max={zero_report['a_base_abs_max']:.6g}"
    )
    print("[hfrvla-contract] PASS")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(_jsonable(report), f, indent=2)
        print(f"[hfrvla-contract] wrote {args.json_out}")


if __name__ == "__main__":
    main()
