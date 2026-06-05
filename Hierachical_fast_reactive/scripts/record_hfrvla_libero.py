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
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
HFRVLA_CACHE_ROOT = Path(
    os.environ.get("HFRVLA_CACHE_ROOT", Path.home() / "tmp" / "hfrvla")
).expanduser()
HFRVLA_HF_DATASETS_CACHE = HFRVLA_CACHE_ROOT / "hf_datasets"
HFRVLA_TMPDIR = HFRVLA_CACHE_ROOT / "tmp"
HFRVLA_HF_DATASETS_CACHE.mkdir(parents=True, exist_ok=True)
HFRVLA_TMPDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_DATASETS_CACHE", str(HFRVLA_HF_DATASETS_CACHE))
os.environ.setdefault("TMPDIR", str(HFRVLA_TMPDIR))

POLICY_SRC = REPO_ROOT / "policy" / "lerobot_policy_hfrvla" / "src"
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
    parser.add_argument(
        "--smolvla",
        default=DEFAULT_LIBERO_SMOLVLA,
        help="LIBERO-adapted SmolVLA slow-planner checkpoint. Do not use raw "
             "`lerobot/smolvla_base` unless --allow-feature-remap is set for debugging. "
             f"Default: {DEFAULT_LIBERO_SMOLVLA}.",
    )

    parser.add_argument("--dinov3-repo", type=str, required=True)
    parser.add_argument("--dinov3-weights", type=str, required=True)
    parser.add_argument("--dinov3-arch", default="dinov3_vits16")

    parser.add_argument("--wrist-key", default="observation.images.image2")
    parser.add_argument("--state-key", default=OBS_STATE)
    parser.add_argument("--action-key", default=ACTION)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--ep-from",
        type=int,
        default=0,
        help="Start episode index (inclusive). Used for multi-process sharding.",
    )
    parser.add_argument(
        "--ep-to",
        type=int,
        default=None,
        help="End episode index (exclusive). Default: process to the dataset end. "
             "Used together with --ep-from to record a sub-range per shard.",
    )
    parser.add_argument("--fps", type=int, default=10, help="Must match the source dataset fps.")
    parser.add_argument("--dino-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument(
        "--dino-batch-size",
        type=int,
        default=64,
        help="Number of wrist frames to forward through DINOv3 per call. "
             "Larger = better GPU utilization, more VRAM. 64 fits comfortably on a 24 GB GPU at 224x224.",
    )
    parser.add_argument(
        "--allow-feature-remap",
        action="store_true",
        help="Allow recording with a SmolVLA checkpoint whose own config is not "
             "already LIBERO-shaped. This records weak/misaligned a_base features "
             "unless you know exactly why you need it.",
    )
    parser.add_argument(
        "--source-reader",
        choices=["auto", "lerobot", "parquet"],
        default="auto",
        help="Source-side reader. 'parquet' streams local LeRobot parquet files "
             "directly and avoids expensive HF Datasets materialization for large shards.",
    )
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


def _source_total_episodes(src_root: Path | None) -> int | None:
    if src_root is None:
        return None
    info_path = src_root / "meta/info.json"
    if not info_path.exists():
        return None
    return int(json.loads(info_path.read_text())["total_episodes"])


def _resolve_episode_indices(
    *,
    total_episodes: int,
    ep_from: int,
    ep_to: int | None,
    max_episodes: int | None,
) -> list[int]:
    start = max(0, int(ep_from))
    end = int(total_episodes) if ep_to is None else min(int(ep_to), int(total_episodes))
    if max_episodes is not None:
        end = min(end, start + int(max_episodes))
    if start >= end:
        raise ValueError(
            f"empty range: ep_from={start} ep_to={end} "
            f"(source has {total_episodes} episodes)"
        )
    return list(range(start, end))


def _stats_json_to_tensors(stats: dict) -> dict[str, dict[str, torch.Tensor]]:
    tensor_stats = {}
    for key, feature_stats in stats.items():
        tensor_stats[key] = {
            stat_name: torch.as_tensor(value, dtype=torch.float32)
            for stat_name, value in feature_stats.items()
        }
    return tensor_stats


def _decode_lerobot_image(value: dict | bytes | bytearray | memoryview, root: Path) -> torch.Tensor:
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")
    else:
        image_bytes = value
        image_path = None

    if image_bytes is None:
        if not image_path:
            raise ValueError("LeRobot image entry has neither bytes nor path")
        image_file = Path(image_path)
        if not image_file.is_absolute():
            image_file = root / image_file
        image_bytes = image_file.read_bytes()

    image = Image.open(BytesIO(bytes(image_bytes))).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class _LocalParquetEpisodeSource:
    """Small source-side reader for local LeRobotDataset v3 parquet roots."""

    _DATA_COLUMNS = [
        "observation.images.image",
        "observation.images.image2",
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]

    def __init__(self, root: str | Path):
        self.root = Path(root)
        info_path = self.root / "meta/info.json"
        episodes_root = self.root / "meta/episodes"
        tasks_path = self.root / "meta/tasks.parquet"
        if not info_path.exists() or not episodes_root.exists() or not tasks_path.exists():
            raise FileNotFoundError(
                f"{self.root} is missing required LeRobotDataset metadata"
            )

        self.info = json.loads(info_path.read_text())
        self.num_episodes = int(self.info["total_episodes"])
        self.total_frames = int(self.info["total_frames"])
        stats_path = self.root / "meta/stats.json"
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        self.meta = SimpleNamespace(stats=_stats_json_to_tensors(stats))

        episode_files = sorted(episodes_root.glob("chunk-*/*.parquet"))
        if not episode_files:
            raise FileNotFoundError(f"No episode metadata parquet files under {episodes_root}")
        episodes = pd.concat(
            [pd.read_parquet(path) for path in episode_files],
            ignore_index=True,
        ).sort_values("episode_index")
        self._episodes = episodes.set_index("episode_index", drop=False)

        tasks_df = pd.read_parquet(tasks_path)
        if "task_index" in tasks_df.columns:
            if "task" in tasks_df.columns:
                task_strings = tasks_df["task"].tolist()
            else:
                task_strings = list(tasks_df.index)
            self._task_by_index = {
                int(task_idx): str(task)
                for task_idx, task in zip(tasks_df["task_index"].tolist(), task_strings, strict=True)
            }
        else:
            self._task_by_index = {idx: str(task) for idx, task in enumerate(tasks_df.index)}

        self._frame_to_chunk = np.full(self.total_frames, -1, dtype=np.int64)
        self._frame_to_file = np.full(self.total_frames, -1, dtype=np.int64)
        data_files = sorted((self.root / "data").glob("chunk-*/*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"No data parquet files under {self.root / 'data'}")
        for path in data_files:
            chunk_index = int(path.parent.name.split("-")[-1])
            file_index = int(path.stem.split("-")[-1])
            index_df = pd.read_parquet(path, columns=["index"])
            if len(index_df) == 0:
                continue
            indices = index_df["index"].to_numpy(dtype=np.int64)
            self._frame_to_chunk[indices] = chunk_index
            self._frame_to_file[indices] = file_index

        self._cached_chunk_index: int | None = None
        self._cached_file_index: int | None = None
        self._cached_df: pd.DataFrame | None = None
        self._cached_positions: dict[int, int] = {}

    def episode_bounds(self, ep_idx: int) -> tuple[int, int, str]:
        row = self._episodes.loc[int(ep_idx)]
        task = row.get("tasks", None)
        if isinstance(task, np.ndarray):
            task = task.tolist()
        if isinstance(task, (list, tuple)) and task:
            task_text = str(task[0])
        else:
            task_text = self._task_by_index.get(int(row.get("task_index", -1)), "do the task")
        return int(row["dataset_from_index"]), int(row["dataset_to_index"]), task_text

    def _data_frame(self, chunk_index: int, file_index: int) -> pd.DataFrame:
        if (
            self._cached_df is not None
            and self._cached_chunk_index == chunk_index
            and self._cached_file_index == file_index
        ):
            return self._cached_df

        path = self.root / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path, columns=self._DATA_COLUMNS)
        self._cached_chunk_index = int(chunk_index)
        self._cached_file_index = int(file_index)
        self._cached_df = df
        self._cached_positions = {
            int(frame_index): idx
            for idx, frame_index in enumerate(df["index"].to_numpy())
        }
        return df

    def __getitem__(self, frame_index: int) -> dict:
        frame_index = int(frame_index)
        if frame_index < 0 or frame_index >= self.total_frames:
            raise IndexError(frame_index)
        chunk_index = int(self._frame_to_chunk[frame_index])
        file_index = int(self._frame_to_file[frame_index])
        if chunk_index < 0 or file_index < 0:
            raise IndexError(f"frame {frame_index} is not covered by episode metadata")

        df = self._data_frame(chunk_index, file_index)
        row = df.iloc[self._cached_positions[frame_index]]
        task_index = int(row["task_index"])
        return {
            "observation.images.image": _decode_lerobot_image(
                row["observation.images.image"],
                self.root,
            ),
            "observation.images.image2": _decode_lerobot_image(
                row["observation.images.image2"],
                self.root,
            ),
            "observation.state": torch.as_tensor(
                np.asarray(row["observation.state"], dtype=np.float32).copy(),
                dtype=torch.float32,
            ),
            "action": torch.as_tensor(
                np.asarray(row["action"], dtype=np.float32).copy(),
                dtype=torch.float32,
            ),
            "timestamp": torch.tensor(float(row["timestamp"]), dtype=torch.float32),
            "frame_index": torch.tensor(int(row["frame_index"]), dtype=torch.int64),
            "episode_index": torch.tensor(int(row["episode_index"]), dtype=torch.int64),
            "index": torch.tensor(int(row["index"]), dtype=torch.int64),
            "task_index": torch.tensor(task_index, dtype=torch.int64),
            "task": self._task_by_index.get(task_index, "do the task"),
        }


def _episode_bounds(src, ep_idx: int) -> tuple[int, int, str]:
    if hasattr(src, "episode_bounds"):
        return src.episode_bounds(ep_idx)
    ep_meta = src.meta.episodes[ep_idx]
    ep_from_abs = int(ep_meta["dataset_from_index"])
    ep_to_abs = int(ep_meta["dataset_to_index"])
    index_map = getattr(getattr(src, "reader", None), "_absolute_to_relative_idx", None)
    if index_map is None:
        ep_from = ep_from_abs
        ep_to = ep_to_abs
    else:
        ep_from = int(index_map[ep_from_abs])
        ep_to = int(index_map[ep_to_abs - 1]) + 1
    ep_task = (
        ep_meta["tasks"][0]
        if isinstance(ep_meta.get("tasks"), list) and ep_meta["tasks"]
        else "do the task"
    )
    return ep_from, ep_to, ep_task


def _basic_feature_stats(
    array: np.ndarray,
    *,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    quantile_list: list[float] | None,
) -> dict[str, np.ndarray]:
    from lerobot.datasets import compute_stats as lr_compute_stats

    original_shape = array.shape
    reshaped, sample_count = lr_compute_stats._prepare_array_for_stats(array, axis)
    stats = lr_compute_stats._compute_basic_stats(
        reshaped,
        sample_count,
        quantile_list or lr_compute_stats.DEFAULT_QUANTILES,
    )
    return lr_compute_stats._reshape_stats_by_axis(
        stats,
        axis,
        keepdims,
        original_shape,
    )


def _compute_episode_stats_basic_fallback(
    episode_data: dict[str, list[str] | np.ndarray],
    features: dict,
    quantile_list: list[float] | None = None,
) -> dict:
    """Compute episode stats without histogram bins.

    LeRobot's streaming quantile stats can fail on constant high-dimensional
    generated chunk fields because numpy requires strictly increasing histogram
    bin edges. Recording only needs valid per-feature stats, so this fallback
    uses exact numpy reductions/quantiles instead of histograms.
    """
    from lerobot.datasets import compute_stats as lr_compute_stats

    ep_stats = {}
    for key, data in episode_data.items():
        feature = features[key]
        if feature["dtype"] == "string":
            continue

        if feature["dtype"] in ["image", "video"]:
            ep_ft_array = lr_compute_stats.sample_images(data)
            stats = _basic_feature_stats(
                ep_ft_array,
                axis=(0, 2, 3),
                keepdims=True,
                quantile_list=quantile_list,
            )
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / 255.0, axis=0)
                for k, v in stats.items()
            }
            continue

        ep_ft_array = np.asarray(data)
        ep_stats[key] = _basic_feature_stats(
            ep_ft_array,
            axis=0,
            keepdims=ep_ft_array.ndim == 1,
            quantile_list=quantile_list,
        )

    return ep_stats


def _install_recording_stats_fallback() -> None:
    import lerobot.datasets.dataset_writer as dataset_writer

    original = dataset_writer.compute_episode_stats
    if getattr(original, "_hfrvla_stats_fallback", False):
        return

    def _wrapped_compute_episode_stats(episode_data, features, quantile_list=None):
        try:
            return original(episode_data, features, quantile_list=quantile_list)
        except ValueError as exc:
            if "`bins` must increase monotonically" not in str(exc):
                raise
            print(
                "[record] LeRobot stats histogram fallback activated for "
                "constant generated chunk fields.",
                flush=True,
            )
            return _compute_episode_stats_basic_fallback(
                episode_data,
                features,
                quantile_list=quantile_list,
            )

    _wrapped_compute_episode_stats._hfrvla_stats_fallback = True
    dataset_writer.compute_episode_stats = _wrapped_compute_episode_stats


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    _install_recording_stats_fallback()

    raw_smolvla_config = load_policy_config_json(args.smolvla)
    try:
        warn_or_validate_libero_slow_planner(
            raw_smolvla_config,
            source=args.smolvla,
            allow_feature_remap=args.allow_feature_remap,
        )
    except ValueError as exc:
        raise SystemExit(f"[record] {exc}") from exc

    args.out_root.parent.mkdir(parents=True, exist_ok=True)

    source_total = _source_total_episodes(args.src_root)
    src_episodes = None
    episode_indices = None
    if source_total is not None:
        try:
            episode_indices = _resolve_episode_indices(
                total_episodes=source_total,
                ep_from=args.ep_from,
                ep_to=args.ep_to,
                max_episodes=args.max_episodes,
            )
        except ValueError as exc:
            raise SystemExit(f"[record] {exc}") from exc
        src_episodes = episode_indices

    source_reader = args.source_reader
    if source_reader == "auto":
        source_reader = "parquet" if args.src_root is not None and (args.src_root / "meta/info.json").exists() else "lerobot"

    print(
        f"[record] loading source dataset: {args.src_repo_id} "
        f"(reader={source_reader})",
        flush=True,
    )
    if source_reader == "parquet":
        if args.src_root is None:
            raise SystemExit("[record] --source-reader parquet requires --src-root")
        src = _LocalParquetEpisodeSource(args.src_root)
    else:
        src = LeRobotDataset(
            args.src_repo_id,
            root=args.src_root,
            episodes=src_episodes,
        )
    n_total = source_total if source_total is not None else int(src.num_episodes)
    if episode_indices is None:
        try:
            episode_indices = _resolve_episode_indices(
                total_episodes=n_total,
                ep_from=args.ep_from,
                ep_to=args.ep_to,
                max_episodes=args.max_episodes,
            )
        except ValueError as exc:
            raise SystemExit(f"[record] {exc}") from exc
    print(
        f"[record] source has {n_total} episodes; recording shard "
        f"[{episode_indices[0]}, {episode_indices[-1] + 1}) = {len(episode_indices)} episodes",
        flush=True,
    )

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
    chunk_len = int(config.chunk_size)
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
        "observation.extra.a_base_chunk": {
            "dtype": "float32",
            "shape": (chunk_len, 7),
            "names": None,
        },
        "observation.extra.chunk_step_idx": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
        "observation.extra.chunk_age_steps": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "observation.extra.chunk_age_norm": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
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

    for ep_idx in episode_indices:
        ep_from, ep_to, ep_task = _episode_bounds(src, ep_idx)

        policy.reset()
        cached_zgoal: torch.Tensor | None = None
        cached_zphase: torch.Tensor | None = None
        current_a_base_chunk = torch.zeros(chunk_len, 7, dtype=torch.float32)
        current_chunk_start_t = ep_from
        chunk_consumed = 0

        # ── Pass 1: SmolVLA chunk loop + collect everything except dino_patches.
        # Per-frame records hold ALL per-row tensors that go to disk EXCEPT
        # dino_patches (computed in Pass 2). Wrist tensors for DINO live in
        # wrist_for_dino_buf as already-normalized (1,3,H,W) tensors on CPU.
        frame_records: list[dict] = []
        wrist_for_dino_buf: list[torch.Tensor] = []

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
                actions_cpu = actions.detach().squeeze(0).cpu().float()
                current_a_base_chunk = torch.empty(chunk_len, actions_cpu.shape[-1], dtype=torch.float32)
                usable = min(chunk_len, int(actions_cpu.shape[0]))
                current_a_base_chunk[:usable] = actions_cpu[:usable]
                if usable < chunk_len:
                    current_a_base_chunk[usable:] = current_a_base_chunk[usable - 1]
                policy._queues[ACTION].extend(actions.transpose(0, 1)[:n_action_steps])
                cached_zgoal = policy._zgoal_cache
                cached_zphase = policy._zphase_cache
                current_chunk_start_t = t
                chunk_consumed = 0

            a_base = policy._queues[ACTION].popleft()
            chunk_consumed += 1
            k_idx = chunk_consumed - 1
            k_norm = k_idx / max(1, chunk_len - 1)
            chunk_age_steps = float(t - current_chunk_start_t)
            chunk_age_norm = chunk_age_steps / max(1, chunk_len - 1)

            # Stash normalized wrist for batched DINO pass below.
            # _wrist_for_dino returns (1,3,H,W); strip the batch dim for stacking.
            wrist_for_dino_buf.append(_wrist_for_dino(sample[args.wrist_key]).squeeze(0).cpu())

            frame_records.append({
                "image_uint8":  _img_for_write(sample["observation.images.image"]),
                "image2_uint8": _img_for_write(sample[args.wrist_key]),
                "state":  sample[args.state_key].cpu().float(),
                "action": sample[args.action_key].cpu().float(),
                "z_goal": (
                    cached_zgoal.squeeze(0).cpu().float()
                    if cached_zgoal is not None
                    else torch.zeros(text_hidden, dtype=torch.float32)
                ),
                "z_phase": (
                    cached_zphase.squeeze(0).cpu().float()
                    if cached_zphase is not None
                    else torch.zeros(expert_hidden, dtype=torch.float32)
                ),
                "a_base": a_base.squeeze(0).cpu().float(),
                "a_base_chunk": current_a_base_chunk.clone(),
                "chunk_step_idx": int(k_idx),
                "chunk_age_steps": chunk_age_steps,
                "chunk_age_norm": chunk_age_norm,
                "k_norm": float(k_norm),
            })

        # ── Pass 2: batched DINOv3 over the whole episode.
        # Stacks all wrist frames once, then forwards in --dino-batch-size chunks.
        # Replaces N single-image DINO calls with ceil(N / batch_size) — typically
        # ~50x fewer kernel launches per episode for batch_size=64 on a 200-frame
        # episode, with proportionally higher GPU utilization.
        T = len(wrist_for_dino_buf)
        all_dino_patches_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            wrist_stack = torch.stack(wrist_for_dino_buf, dim=0)   # (T, 3, H, W) CPU
            for i in range(0, T, args.dino_batch_size):
                chunk = wrist_stack[i : i + args.dino_batch_size].to(device, non_blocking=True)
                patches = dino(chunk).cpu()                        # (b, n_patches, dim)
                all_dino_patches_chunks.append(patches)
        all_dino_patches = torch.cat(all_dino_patches_chunks, dim=0)  # (T, n_patches, dim)
        if dino_dtype == "float16":
            all_dino_patches = all_dino_patches.half()

        # Free the stacked wrist buffer ASAP — it's the largest transient.
        del wrist_stack, wrist_for_dino_buf, all_dino_patches_chunks

        # ── Pass 3: write rows.
        for local_t, rec in enumerate(frame_records):
            frame = {
                "observation.images.image":  rec["image_uint8"],
                "observation.images.image2": rec["image2_uint8"],
                "observation.state":         rec["state"],
                "action":                    rec["action"],
                "observation.extra.z_goal":     rec["z_goal"],
                "observation.extra.z_phase":    rec["z_phase"],
                "observation.extra.a_base":     rec["a_base"],
                "observation.extra.a_base_chunk": rec["a_base_chunk"],
                "observation.extra.chunk_step_idx": torch.tensor([rec["chunk_step_idx"]], dtype=torch.int64),
                "observation.extra.chunk_age_steps": torch.tensor([rec["chunk_age_steps"]], dtype=torch.float32),
                "observation.extra.chunk_age_norm": torch.tensor([rec["chunk_age_norm"]], dtype=torch.float32),
                "observation.extra.k_idx_norm": torch.tensor([rec["k_norm"]], dtype=torch.float32),
                "observation.extra.dino_patches":  all_dino_patches[local_t],
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
