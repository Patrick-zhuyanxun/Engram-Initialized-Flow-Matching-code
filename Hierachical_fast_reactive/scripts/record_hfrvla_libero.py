#!/usr/bin/env python
"""Record a LeRobotDataset v3 with HFRVLA precomputed features.

Reads HuggingFaceVLA/libero, rolls a frozen SmolVLA + DINOv3 once over each
episode, and writes a new LeRobotDataset where every frame carries:

    observation.images.image
    observation.images.image2
    observation.state, action, task
    observation.extra.z_goal
    observation.extra.z_phase
    observation.extra.a_base
    observation.extra.k_idx_norm
    observation.extra.dino_patches
    observation.extra.contact_label

After recording, lerobot-train can consume the dataset with:

    --dataset.repo_id=<out_repo_id> --dataset.root=<out_root>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hfrvla_hf_datasets_cache")

POLICY_SRC = Path(__file__).resolve().parents[1] / "policy" / "lerobot_policy_hfrvla" / "src"
if POLICY_SRC.exists():
    sys.path.insert(0, str(POLICY_SRC))

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_STATE

from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig
from lerobot_policy_hfrvla.dinov3_backbone import DINOv3Backbone
from lerobot_policy_hfrvla.modeling_hfrvla import HFRVLAPolicy
from lerobot_policy_hfrvla.processor_hfrvla import normalize_for_dinov3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument(
        "--src-root",
        type=Path,
        default=None,
        help="Optional local root for the source LeRobotDataset.",
    )
    parser.add_argument(
        "--out-repo-id",
        default="HFRVLA_libero_v1",
        help="Identifier baked into the new dataset's metadata.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Local directory for the new dataset. Must not already exist.",
    )
    parser.add_argument("--smolvla", default="lerobot/smolvla_base")

    parser.add_argument("--dinov3-repo", type=str, required=True)
    parser.add_argument("--dinov3-weights", type=str, required=True)
    parser.add_argument("--dinov3-arch", default="dinov3_vits16")

    parser.add_argument("--wrist-key", default="observation.images.image2")
    parser.add_argument("--state-key", default=OBS_STATE)
    parser.add_argument("--action-key", default=ACTION)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--fps", type=int, default=10, help="Must match the source dataset fps.")
    parser.add_argument("--dino-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def _wrist_for_dino(wrist_raw: torch.Tensor) -> torch.Tensor:
    """Normalize a wrist image tensor for DINOv3. Returns (1, 3, H, W)."""
    x = wrist_raw.float()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.shape[-1] == 3 and x.shape[1] != 3:
        x = x.permute(0, 3, 1, 2)
    if x.max() > 1.5:
        x = x / 255.0
    return normalize_for_dinov3(x)


def _img_for_write(image: torch.Tensor) -> torch.Tensor:
    """Convert a source image tensor to HWC uint8 for LeRobotDataset.add_frame."""
    x = image
    if x.dim() == 4:
        x = x.squeeze(0)
    if x.shape[0] == 3:
        x = x.permute(1, 2, 0)
    x = x.float()
    if x.max() <= 1.5:
        x = x * 255.0
    return x.clamp(0, 255).to(torch.uint8).cpu()


def _episode_bounds(src: LeRobotDataset, ep_idx: int) -> tuple[int, int, str]:
    ep_meta = src.meta.episodes[ep_idx]
    ep_from = int(ep_meta["dataset_from_index"])
    ep_to = int(ep_meta["dataset_to_index"])
    ep_task = (
        ep_meta["tasks"][0]
        if isinstance(ep_meta.get("tasks"), list) and ep_meta["tasks"]
        else "do the task"
    )
    return ep_from, ep_to, ep_task


def main() -> None:
    args = parse_args()
    args.out_root.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[record] loading source dataset: {args.src_repo_id}", flush=True)
    src_episodes = list(range(args.max_episodes)) if args.max_episodes is not None else None
    src = LeRobotDataset(
        args.src_repo_id,
        root=args.src_root,
        episodes=src_episodes,
    )
    n_total = int(src.num_episodes)
    n_to_record = min(args.max_episodes, n_total) if args.max_episodes else n_total
    print(f"[record] source has {n_total} episodes; recording {n_to_record}", flush=True)

    print(f"[record] building HFRVLAPolicy from {args.smolvla}", flush=True)
    config = HFRVLAConfig.from_smolvla(
        args.smolvla,
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
    policy = HFRVLAPolicy.from_pretrained(args.smolvla, config=config).to(device).eval()

    text_hidden = int(policy.model.vlm_with_expert.config.text_config.hidden_size)
    expert_hidden = int(policy.model.vlm_with_expert.expert_hidden_size)
    dino_dim = int(config.dinov3_feature_dim)
    dino_npatch = int(config.dinov3_num_patches)
    n_action_steps = int(config.n_action_steps)

    dino = DINOv3Backbone(
        model_id=config.dinov3_model_id,
        local_repo=config.dinov3_local_repo,
        local_weights=config.dinov3_local_weights,
        arch=config.dinov3_arch,
        frozen=True,
    ).to(device).eval()

    pre_processor, _ = make_smolvla_pre_post_processors(
        config,
        dataset_stats=getattr(src.meta, "stats", None),
    )

    dino_dtype = args.dino_dtype
    features = {
        "observation.images.image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.image2": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        "observation.extra.z_goal": {
            "dtype": "float32",
            "shape": (text_hidden,),
            "names": None,
        },
        "observation.extra.z_phase": {
            "dtype": "float32",
            "shape": (expert_hidden,),
            "names": None,
        },
        "observation.extra.a_base": {"dtype": "float32", "shape": (7,), "names": None},
        "observation.extra.k_idx_norm": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "observation.extra.dino_patches": {
            "dtype": dino_dtype,
            "shape": (dino_npatch, dino_dim),
            "names": None,
        },
        "observation.extra.contact_label": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
    }
    dst = LeRobotDataset.create(
        repo_id=args.out_repo_id,
        fps=args.fps,
        features=features,
        root=str(args.out_root),
        use_videos=True,
        streaming_encoding=False,
        batch_encoding_size=1,
    )
    print(f"[record] dst created at {dst.root}", flush=True)

    for ep_idx in range(n_to_record):
        ep_from, ep_to, ep_task = _episode_bounds(src, ep_idx)

        policy.reset()
        cached_zgoal: torch.Tensor | None = None
        cached_zphase: torch.Tensor | None = None
        chunk_consumed = 0

        for t in range(ep_from, ep_to):
            sample = src[t]
            sample_with_task = dict(sample)
            sample_with_task.setdefault("task", ep_task)
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

            batch = policy._prepare_batch(batch)
            policy._queues = populate_queues(policy._queues, batch, exclude_keys=[ACTION])
            if len(policy._queues[ACTION]) == 0:
                policy._clear_hook_cache()
                with torch.no_grad():
                    actions = policy._get_action_chunk(batch)
                policy._queues[ACTION].extend(actions.transpose(0, 1)[:n_action_steps])
                cached_zgoal = policy._zgoal_cache
                cached_zphase = policy._zphase_cache
                chunk_consumed = 0

            a_base = policy._queues[ACTION].popleft()
            chunk_consumed += 1
            k_idx = chunk_consumed - 1
            k_norm = k_idx / max(1, n_action_steps - 1)

            wrist_for_dino = _wrist_for_dino(sample[args.wrist_key]).to(device)
            with torch.no_grad():
                dino_patches = dino(wrist_for_dino).squeeze(0).cpu()
            if dino_dtype == "float16":
                dino_patches = dino_patches.half()

            frame = {
                "observation.images.image": _img_for_write(sample["observation.images.image"]),
                "observation.images.image2": _img_for_write(sample[args.wrist_key]),
                "observation.state": sample[args.state_key].cpu().float(),
                "action": sample[args.action_key].cpu().float(),
                "observation.extra.z_goal": (
                    cached_zgoal.squeeze(0).cpu().float()
                    if cached_zgoal is not None
                    else torch.zeros(text_hidden, dtype=torch.float32)
                ),
                "observation.extra.z_phase": (
                    cached_zphase.squeeze(0).cpu().float()
                    if cached_zphase is not None
                    else torch.zeros(expert_hidden, dtype=torch.float32)
                ),
                "observation.extra.a_base": a_base.squeeze(0).cpu().float(),
                "observation.extra.k_idx_norm": torch.tensor([k_norm], dtype=torch.float32),
                "observation.extra.dino_patches": dino_patches,
                "observation.extra.contact_label": torch.tensor([0.0], dtype=torch.float32),
                "task": ep_task,
            }
            dst.add_frame(frame)

        dst.save_episode()
        print(f"[record]   ep {ep_idx:>4d}  T={ep_to - ep_from:>4d}  saved", flush=True)

    dst.finalize()
    print(f"[record] done. dataset at {dst.root}", flush=True)


if __name__ == "__main__":
    main()
