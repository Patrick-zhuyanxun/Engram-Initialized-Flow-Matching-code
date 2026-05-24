#!/usr/bin/env python
"""Record successful zero-fast LIBERO rollouts as an HFRVLA LeRobotDataset v3.

This is the Stage C DAgger-lite data path: run the packaged HFRVLA checkpoint
with ``inference_disable_fast=True`` in closed loop, keep only successful
episodes, and write the observations plus cached SmolVLA / DINO features needed
by ``scripts/build_hfrvla_fastcache.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
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

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed

from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig  # noqa: F401
from lerobot_policy_hfrvla.dinov3_backbone import DINOv3Backbone
from lerobot_policy_hfrvla.processor_hfrvla import normalize_for_dinov3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=REPO_ROOT / "checkpoints/hfrvla_zero_fast_packaged",
        help="Packaged HFRVLA checkpoint with inference_disable_fast=True.",
    )
    parser.add_argument("--out-repo-id", default="HFRVLA_libero_v1_zero_fast_rollouts")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "checkpoints/HFRVLA_libero_v1_zero_fast_rollouts",
    )
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument(
        "--task-ids",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated LIBERO task ids.",
    )
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dino-batch-size", type=int, default=64)
    parser.add_argument("--dino-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _parse_task_ids(raw: str) -> list[int]:
    task_ids = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not task_ids:
        raise ValueError("--task-ids must contain at least one task id")
    return task_ids


def _img_for_write(image: torch.Tensor) -> torch.Tensor:
    x = image.detach()
    if x.dim() == 4:
        x = x.squeeze(0)
    if x.shape[0] == 3:
        x = x.permute(1, 2, 0)
    x = x.float()
    if x.max() <= 1.5:
        x = x * 255.0
    return x.clamp(0, 255).to(torch.uint8).cpu()


def _wrist_for_dino(wrist: torch.Tensor, *, size: int) -> torch.Tensor:
    x = wrist.detach().float()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.shape[-1] == 3 and x.shape[1] != 3:
        x = x.permute(0, 3, 1, 2).contiguous()
    if x.max() > 1.5:
        x = x / 255.0
    return normalize_for_dinov3(x, size=size).cpu()


def _first_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().flatten()[0].item())
    arr = np.asarray(value)
    return bool(arr.reshape(-1)[0].item())


def _success_from_info(info: dict[str, Any]) -> bool:
    final_info = info.get("final_info")
    if isinstance(final_info, dict) and "is_success" in final_info:
        return _first_bool(final_info["is_success"])
    if "is_success" in info:
        return _first_bool(info["is_success"])
    return False


def _first_task(task_value: Any) -> str:
    if isinstance(task_value, str):
        return task_value.rstrip("\n")
    if isinstance(task_value, (list, tuple)) and task_value:
        return str(task_value[0]).rstrip("\n")
    return "do the task"


def _build_dataset(args: argparse.Namespace, policy_config) -> LeRobotDataset:
    action_dim = int(policy_config.output_features[ACTION].shape[0])
    text_hidden = int(getattr(policy_config, "offline_zgoal_dim", 960))
    expert_hidden = int(getattr(policy_config, "offline_zphase_dim", 480))
    dino_dim = int(policy_config.dinov3_feature_dim)
    dino_npatch = int(policy_config.dinov3_num_patches)
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
        OBS_STATE: {"dtype": "float32", "shape": (8,), "names": ["state"]},
        ACTION: {"dtype": "float32", "shape": (action_dim,), "names": ["actions"]},
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
        "observation.extra.a_base": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": None,
        },
        "observation.extra.k_idx_norm": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "observation.extra.dino_patches": {
            "dtype": args.dino_dtype,
            "shape": (dino_npatch, dino_dim),
            "names": None,
        },
        "observation.extra.contact_label": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "observation.extra.rollout_success": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
    }
    return LeRobotDataset.create(
        repo_id=args.out_repo_id,
        fps=args.fps,
        features=features,
        root=str(args.out_root),
        use_videos=True,
        streaming_encoding=False,
        batch_encoding_size=1,
    )


def _load_policy_and_processors(args: argparse.Namespace, env_cfg: LiberoEnvConfig):
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    policy_cfg.device = args.device
    policy_cfg.use_amp = False
    if hasattr(policy_cfg, "inference_disable_fast"):
        policy_cfg.inference_disable_fast = True
    if hasattr(policy_cfg, "offline_training_mode"):
        policy_cfg.offline_training_mode = False

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(args.policy_path),
        preprocessor_overrides={
            "device_processor": {"device": str(policy_cfg.device)},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=policy_cfg,
    )
    return policy, policy_cfg, preprocessor, postprocessor, env_preprocessor, env_postprocessor


def _compute_dino_patches(
    dino: DINOv3Backbone,
    wrist_tensors: list[torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    dtype: str,
) -> torch.Tensor:
    chunks = []
    wrist_stack = torch.cat(wrist_tensors, dim=0)
    with torch.no_grad():
        for start in range(0, wrist_stack.shape[0], batch_size):
            batch = wrist_stack[start : start + batch_size].to(device, non_blocking=True)
            chunks.append(dino(batch).cpu())
    patches = torch.cat(chunks, dim=0)
    if dtype == "float16":
        patches = patches.half()
    return patches


def _record_successful_episode(
    *,
    dst: LeRobotDataset,
    records: list[dict[str, Any]],
    dino: DINOv3Backbone,
    device: torch.device,
    dino_batch_size: int,
    dino_dtype: str,
) -> int:
    if not records:
        return 0
    dino_patches = _compute_dino_patches(
        dino,
        [rec["wrist_for_dino"] for rec in records],
        device=device,
        batch_size=dino_batch_size,
        dtype=dino_dtype,
    )
    for idx, rec in enumerate(records):
        dst.add_frame(
            {
                "observation.images.image": rec["image"],
                "observation.images.image2": rec["image2"],
                OBS_STATE: rec["state"],
                ACTION: rec["action"],
                "observation.extra.z_goal": rec["z_goal"],
                "observation.extra.z_phase": rec["z_phase"],
                "observation.extra.a_base": rec["a_base"],
                "observation.extra.k_idx_norm": torch.tensor(
                    [rec["k_norm"]],
                    dtype=torch.float32,
                ),
                "observation.extra.dino_patches": dino_patches[idx],
                "observation.extra.contact_label": torch.tensor([0.0], dtype=torch.float32),
                "observation.extra.rollout_success": torch.tensor([1.0], dtype=torch.float32),
                "task": rec["task"],
            }
        )
    dst.save_episode()
    return len(records)


def _run_one_episode(
    *,
    env,
    policy,
    preprocessor,
    postprocessor,
    env_preprocessor,
    env_postprocessor,
    seed: int,
    dino_image_size: int,
) -> tuple[bool, list[dict[str, Any]]]:
    policy.reset()
    observation, _ = env.reset(seed=[seed])
    max_steps = int(env.call("_max_episode_steps")[0])
    n_action_steps = int(policy.config.n_action_steps)
    zgoal_dim = int(policy.model.vlm_with_expert.config.text_config.hidden_size)
    zphase_dim = int(policy.model.vlm_with_expert.expert_hidden_size)
    records: list[dict[str, Any]] = []
    success = False

    for step in range(max_steps):
        lerobot_obs = preprocess_observation(observation)
        lerobot_obs = add_envs_task(env, lerobot_obs)
        env_obs = env_preprocessor(lerobot_obs)
        task = _first_task(env_obs.get("task", "do the task"))
        image = _img_for_write(env_obs["observation.images.image"][0])
        image2 = _img_for_write(env_obs["observation.images.image2"][0])
        state = env_obs[OBS_STATE][0].detach().cpu().float()
        wrist_for_dino = _wrist_for_dino(
            env_obs["observation.images.image2"],
            size=dino_image_size,
        )

        policy_obs = preprocessor(env_obs)
        with torch.inference_mode():
            a_base = policy.select_action(policy_obs)

        z_goal = getattr(policy, "_zgoal_cache", None)
        z_phase = getattr(policy, "_zphase_cache", None)
        if z_goal is None:
            z_goal_cpu = torch.zeros(zgoal_dim, dtype=torch.float32)
        else:
            z_goal_cpu = z_goal[0].detach().cpu().float()
        if z_phase is None:
            z_phase_cpu = torch.zeros(zphase_dim, dtype=torch.float32)
        else:
            z_phase_cpu = z_phase[0].detach().cpu().float()

        chunk_consumed = int(getattr(policy, "_chunk_consumed", 1))
        k_norm = (chunk_consumed - 1) / max(1, n_action_steps - 1)

        action = postprocessor(a_base)
        action_transition = env_postprocessor({ACTION: action})
        action_env = action_transition[ACTION]
        action_numpy = action_env.detach().cpu().numpy()

        records.append(
            {
                "image": image,
                "image2": image2,
                "state": state,
                "action": action_env[0].detach().cpu().float(),
                "a_base": a_base[0].detach().cpu().float(),
                "z_goal": z_goal_cpu,
                "z_phase": z_phase_cpu,
                "k_norm": float(k_norm),
                "wrist_for_dino": wrist_for_dino,
                "task": task,
            }
        )

        observation, _reward, terminated, truncated, info = env.step(action_numpy)
        success = success or _success_from_info(info)
        done = bool(np.asarray(terminated | truncated).reshape(-1)[0])
        if done:
            break
        if step + 1 == max_steps:
            break

    return success, records


def main() -> None:
    args = parse_args()
    register_third_party_plugins()
    set_seed(args.seed)
    HFRVLA_HF_DATASETS_CACHE.mkdir(parents=True, exist_ok=True)
    HFRVLA_TMPDIR.mkdir(parents=True, exist_ok=True)

    if args.out_root.exists():
        if not args.overwrite:
            raise SystemExit(f"[rollout] out-root already exists: {args.out_root}")
        shutil.rmtree(args.out_root)
    args.out_root.parent.mkdir(parents=True, exist_ok=True)

    task_ids = _parse_task_ids(args.task_ids)
    env_cfg = LiberoEnvConfig(
        task=args.task_suite,
        task_ids=task_ids,
        fps=args.fps,
        episode_length=args.episode_length,
    )
    policy, policy_cfg, preprocessor, postprocessor, env_preprocessor, env_postprocessor = (
        _load_policy_and_processors(args, env_cfg)
    )
    if not bool(getattr(policy.config, "inference_disable_fast", False)):
        raise RuntimeError("Loaded policy is not in zero_fast mode")

    device = torch.device(args.device)
    dino = DINOv3Backbone(
        model_id=policy.config.dinov3_model_id,
        local_repo=policy.config.dinov3_local_repo,
        local_weights=policy.config.dinov3_local_weights,
        arch=policy.config.dinov3_arch,
        frozen=True,
    ).to(device).eval()

    dst = _build_dataset(args, policy_cfg)
    summary: dict[str, Any] = {
        "mode": "zero_fast_successful_rollouts",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "policy_path": str(args.policy_path),
        "out_root": str(args.out_root),
        "task_suite": args.task_suite,
        "episodes_per_task": args.episodes_per_task,
        "seed": args.seed,
        "task_stats": defaultdict(
            lambda: {
                "attempted": 0,
                "successes": 0,
                "saved_frames": 0,
                "success_episode_lengths": [],
                "success_seeds": [],
            }
        ),
    }

    total_frames = 0
    total_successes = 0
    for task_id in task_ids:
        task_env_cfg = LiberoEnvConfig(
            task=args.task_suite,
            task_ids=[task_id],
            fps=args.fps,
            episode_length=args.episode_length,
        )
        envs = make_env(task_env_cfg, n_envs=1, use_async_envs=False)
        env = envs[args.task_suite][task_id]
        try:
            for ep in range(args.episodes_per_task):
                seed = args.seed + task_id * 10_000 + ep
                stats = summary["task_stats"][str(task_id)]
                stats["attempted"] += 1
                success, records = _run_one_episode(
                    env=env,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    seed=seed,
                    dino_image_size=int(policy.config.dinov3_image_size),
                )
                if success:
                    saved = _record_successful_episode(
                        dst=dst,
                        records=records,
                        dino=dino,
                        device=device,
                        dino_batch_size=args.dino_batch_size,
                        dino_dtype=args.dino_dtype,
                    )
                    stats["successes"] += 1
                    stats["saved_frames"] += saved
                    stats["success_episode_lengths"].append(saved)
                    stats["success_seeds"].append(seed)
                    total_successes += 1
                    total_frames += saved
                print(
                    f"[rollout] task={task_id} ep={ep} seed={seed} "
                    f"success={success} frames={len(records)} saved={success}",
                    flush=True,
                )
        finally:
            close_envs(envs)

    dst.finalize()
    plain_stats = dict(summary["task_stats"])
    summary["task_stats"] = plain_stats
    summary["total_successes"] = total_successes
    summary["total_attempted"] = len(task_ids) * args.episodes_per_task
    summary["total_saved_frames"] = total_frames
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary_path = args.out_root / "zero_fast_rollout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(
        f"[rollout] done: successes={total_successes}/{summary['total_attempted']} "
        f"frames={total_frames} summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
