#!/usr/bin/env python
"""Package a trained HFRVLA fast checkpoint into a lerobot-eval-loadable dir.

Combines the frozen SmolVLA backbone weights with the trained Fast Reactive
Module weights, then writes a single safetensors file plus the HFRVLA
``config.json`` so that ``lerobot-eval --policy.path=<dir>`` can load it.

Example:
    python scripts/package_hfrvla_checkpoint.py \\
        --fast-ckpt checkpoints/hfrvla_run01/checkpoints/last/pretrained_model \\
        --out-dir checkpoints/hfrvla_run01_packaged \\
        --dinov3-repo checkpoints/dinov3_src \\
        --dinov3-weights checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

By default, the frozen slow planner is ``HuggingFaceVLA/smolvla_libero``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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

POLICY_SRC = REPO_ROOT / "policy" / "lerobot_policy_hfrvla" / "src"
if POLICY_SRC.exists():
    sys.path.insert(0, str(POLICY_SRC))

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fast-ckpt", type=Path, default=None,
                   help="Path to a LeRobot pretrained_model directory, model.safetensors, "
                        "fast_final.pt, or any legacy step_*.pt checkpoint. "
                        "Required unless --disable-fast is set.")
    p.add_argument("--disable-fast", action="store_true",
                   help="Package an alignment-test checkpoint that short-circuits "
                        "select_action() to return SmolVLA's a_base directly "
                        "(no fast residual). Fast module weights stay randomly "
                        "initialized but are never invoked at inference. Use "
                        "this to confirm I/O compatibility against the frozen "
                        "SmolVLA slow planner before training the fast module.")
    p.add_argument(
        "--smolvla-pretrained",
        type=str,
        default=DEFAULT_LIBERO_SMOLVLA,
        help="LIBERO-adapted SmolVLA slow-planner checkpoint used to generate "
             "the HFRVLA dataset. Do not use raw `lerobot/smolvla_base` unless "
             "--allow-feature-remap is set for debugging. "
             f"Default: {DEFAULT_LIBERO_SMOLVLA}.",
    )
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Destination directory; will contain config.json + model.safetensors.")
    p.add_argument("--dinov3-repo", type=str, required=True)
    p.add_argument("--dinov3-weights", type=str, required=True)
    p.add_argument("--dinov3-arch", type=str, default="dinov3_vits16")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device for the (one-shot) load+save. CPU is fine and avoids OOM.")
    p.add_argument("--dataset-repo-id", type=str, default="HuggingFaceVLA/libero",
                   help="LeRobot dataset to source normalization stats from. "
                        "Should match what precompute used.")
    p.add_argument("--dataset-root", type=str, default=None,
                   help="Local dataset root for stats loading (avoids HF Hub).")
    p.add_argument("--no-stats", action="store_true",
                   help="Skip loading dataset stats (faster; but eval will use raw input).")
    p.add_argument("--allow-feature-remap", action="store_true",
                   help="Allow packaging a SmolVLA checkpoint whose own config is "
                        "not already LIBERO-shaped. This is only for low-level "
                        "debugging; HFRVLA alignment/recording should use a "
                        "LIBERO-adapted SmolVLA slow planner.")
    args = p.parse_args()
    if not args.disable_fast and args.fast_ckpt is None:
        p.error("--fast-ckpt is required unless --disable-fast is set")
    return args


def _ensure_cache_dirs() -> None:
    HFRVLA_HF_DATASETS_CACHE.mkdir(parents=True, exist_ok=True)
    HFRVLA_TMPDIR.mkdir(parents=True, exist_ok=True)


def _resolve_fast_ckpt_path(path: Path) -> Path:
    if path.is_dir():
        safetensors = path / "model.safetensors"
        if safetensors.exists():
            return safetensors
        pytorch_bin = path / "pytorch_model.bin"
        if pytorch_bin.exists():
            return pytorch_bin
        raise FileNotFoundError(
            f"No model.safetensors or pytorch_model.bin found under {path}"
        )
    return path


def _extract_fast_state(raw_state: dict[str, torch.Tensor] | dict) -> dict[str, torch.Tensor]:
    if "fast_state_dict" in raw_state:
        raw_state = raw_state["fast_state_dict"]
    elif "state_dict" in raw_state:
        raw_state = raw_state["state_dict"]

    if not isinstance(raw_state, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(raw_state).__name__}")

    tensor_state = {
        str(key): value
        for key, value in raw_state.items()
        if torch.is_tensor(value)
    }

    for prefix in ("fast.", "module.fast."):
        selected = {
            key[len(prefix):]: value
            for key, value in tensor_state.items()
            if key.startswith(prefix)
        }
        if selected:
            return selected

    return tensor_state


def load_fast_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    path = _resolve_fast_ckpt_path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        raw_state = load_file(path, device=str(device))
    else:
        raw_state = torch.load(path, map_location=device, weights_only=True)
    return _extract_fast_state(raw_state)


def _libero_feature_overrides(config: HFRVLAConfig) -> None:
    """Match what training/precompute used so eval shapes line up."""
    config.input_features = {
        "observation.images.image":  PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        OBS_STATE:                   PolicyFeature(type=FeatureType.STATE,  shape=(8,)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }


_FAST_CONFIG_OVERRIDE_KEYS = {
    "chunk_size",
    "contact_head_enabled",
    "control_dt",
    "delta_max",
    "dinov3_feature_dim",
    "dinov3_image_size",
    "dinov3_num_patches",
    "gru_hidden",
    "gru_layers",
    "head_hidden",
    "inference_disable_fast",
    "max_action_dim",
    "max_state_dim",
    "n_action_steps",
    "pool_n_heads",
    "pool_query_dim",
    "safety_joint_velocity_limit",
    "seq_len",
    "zgoal_proj_dim",
    "zphase_proj_dim",
}


def _load_fast_config(fast_ckpt: Path | None) -> dict:
    if fast_ckpt is None:
        return {}
    ckpt_dir = fast_ckpt if fast_ckpt.is_dir() else fast_ckpt.parent
    cfg_path = ckpt_dir / "config.json"
    if not cfg_path.exists():
        return {}
    import json

    with open(cfg_path) as f:
        return json.load(f)


def _apply_fast_config_overrides(config: HFRVLAConfig, fast_config: dict) -> None:
    """Preserve inference-relevant knobs from the trained fast checkpoint."""
    for key in _FAST_CONFIG_OVERRIDE_KEYS:
        if key in fast_config and hasattr(config, key):
            setattr(config, key, fast_config[key])

    # The packaged checkpoint is for online eval/deployment, not cached-feature
    # offline training.
    config.offline_training_mode = False
    config.load_vlm_weights = True


def main() -> None:
    _ensure_cache_dirs()
    args = parse_args()
    device = torch.device(args.device)

    raw_smolvla_config = load_policy_config_json(args.smolvla_pretrained)
    try:
        warn_or_validate_libero_slow_planner(
            raw_smolvla_config,
            source=args.smolvla_pretrained,
            allow_feature_remap=args.allow_feature_remap,
        )
    except ValueError as exc:
        raise SystemExit(f"[package] {exc}") from exc

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[package] building HFRVLAConfig from {args.smolvla_pretrained}")
    config = HFRVLAConfig.from_smolvla(
        args.smolvla_pretrained,
        dinov3_local_repo=args.dinov3_repo,
        dinov3_local_weights=args.dinov3_weights,
        dinov3_arch=args.dinov3_arch,
    )
    _apply_fast_config_overrides(config, _load_fast_config(args.fast_ckpt))
    _libero_feature_overrides(config)
    if args.disable_fast:
        config.inference_disable_fast = True
        print(f"[package] --disable-fast set: select_action will short-circuit "
              f"to SmolVLA's a_base (fast module weights are unused).")

    print(f"[package] loading SmolVLA weights into HFRVLAPolicy ...")
    policy = HFRVLAPolicy.from_pretrained(args.smolvla_pretrained, config=config)
    policy = policy.to(device).eval()

    if args.fast_ckpt is not None:
        print(f"[package] loading fast checkpoint: {args.fast_ckpt}")
        fast_state = load_fast_state(args.fast_ckpt, device)
        missing, unexpected = policy.fast.load_state_dict(fast_state, strict=False)
        if missing:
            print(f"[package]   WARNING missing fast keys: {sorted(missing)[:8]}"
                  f"{' ...' if len(missing) > 8 else ''}")
        if unexpected:
            print(f"[package]   WARNING unexpected fast keys: {sorted(unexpected)[:8]}"
                  f"{' ...' if len(unexpected) > 8 else ''}")
    else:
        print(f"[package] skipping fast checkpoint load (--disable-fast set)")

    print(f"[package] writing checkpoint dir -> {args.out_dir}")
    policy.save_pretrained(args.out_dir)

    # ── Build + save SmolVLA pre/post processors so lerobot-eval can load. ──
    # Use LeRobotDatasetMetadata (not LeRobotDataset) so we read only
    # meta/stats.json — full LeRobotDataset would also scan data/ + videos and,
    # on any mismatch, fall through to a Hub lookup that 404s for local-only
    # repo ids like "HFRVLA_libero_v1".
    dataset_stats = None
    if not args.no_stats:
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        print(f"[package] loading dataset stats from {args.dataset_repo_id} ...")
        meta_kwargs = {"repo_id": args.dataset_repo_id}
        if args.dataset_root:
            meta_kwargs["root"] = args.dataset_root
        meta = LeRobotDatasetMetadata(**meta_kwargs)
        dataset_stats = getattr(meta, "stats", None)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=None,
        dataset_stats=dataset_stats,
    )
    preprocessor.save_pretrained(args.out_dir)
    postprocessor.save_pretrained(args.out_dir)

    safetensors = args.out_dir / "model.safetensors"
    cfg_json = args.out_dir / "config.json"
    print(f"[package] done.")
    print(f"           {cfg_json} ({cfg_json.stat().st_size / 1024:.1f} KB)")
    print(f"           {safetensors} ({safetensors.stat().st_size / 1024 / 1024:.1f} MB)")
    for f in sorted(args.out_dir.glob("*.json")):
        print(f"           {f}")
    print(f"\nNext: lerobot-eval --policy.path={args.out_dir} --env.type=libero ...")


if __name__ == "__main__":
    main()
