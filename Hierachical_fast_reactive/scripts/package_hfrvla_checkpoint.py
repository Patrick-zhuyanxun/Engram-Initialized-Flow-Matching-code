#!/usr/bin/env python
"""Package a trained HFRVLA fast checkpoint into a lerobot-eval-loadable dir.

Combines the frozen SmolVLA backbone weights with the trained Fast Reactive
Module weights, then writes a single safetensors file plus the HFRVLA
``config.json`` so that ``lerobot-eval --policy.path=<dir>`` can load it.

Example:
    python scripts/package_hfrvla_checkpoint.py \\
        --fast-ckpt checkpoints/hfrvla_run01/fast_final.pt \\
        --smolvla-pretrained lerobot/smolvla_base \\
        --out-dir checkpoints/hfrvla_run01_packaged \\
        --dinov3-repo checkpoints/dinov3_src \\
        --dinov3-weights checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

POLICY_SRC = Path(__file__).resolve().parents[1] / "policy" / "lerobot_policy_hfrvla" / "src"
if POLICY_SRC.exists():
    sys.path.insert(0, str(POLICY_SRC))

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig
from lerobot_policy_hfrvla.modeling_hfrvla import HFRVLAPolicy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fast-ckpt", type=Path, default=None,
                   help="Path to fast_final.pt or any legacy step_*.pt checkpoint. "
                        "Required unless --disable-fast is set.")
    p.add_argument("--disable-fast", action="store_true",
                   help="Package an alignment-test checkpoint that short-circuits "
                        "select_action() to return SmolVLA's a_base directly "
                        "(no fast residual). Fast module weights stay randomly "
                        "initialized but are never invoked at inference. Use "
                        "this to confirm I/O compatibility against the SmolVLA "
                        "baseline before training the fast module.")
    p.add_argument("--smolvla-pretrained", type=str, default="lerobot/smolvla_base")
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
    args = p.parse_args()
    if not args.disable_fast and args.fast_ckpt is None:
        p.error("--fast-ckpt is required unless --disable-fast is set")
    return args


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


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[package] building HFRVLAConfig from {args.smolvla_pretrained}")
    config = HFRVLAConfig.from_smolvla(
        args.smolvla_pretrained,
        dinov3_local_repo=args.dinov3_repo,
        dinov3_local_weights=args.dinov3_weights,
        dinov3_arch=args.dinov3_arch,
    )
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
        ckpt = torch.load(args.fast_ckpt, map_location=device, weights_only=True)
        fast_state = ckpt.get("fast_state_dict", ckpt)
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
    dataset_stats = None
    if not args.no_stats:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        print(f"[package] loading dataset stats from {args.dataset_repo_id} ...")
        ds_kwargs = {"repo_id": args.dataset_repo_id}
        if args.dataset_root:
            ds_kwargs["root"] = args.dataset_root
        ds = LeRobotDataset(**ds_kwargs)
        dataset_stats = getattr(ds.meta, "stats", None)
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
