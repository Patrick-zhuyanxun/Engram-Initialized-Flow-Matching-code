#!/usr/bin/env python3
"""Render DINOv3 / FWR cross-attention diagnostics for HFRVLA.

The default target is the current paper-facing diagnostic:

    VLA third-person RGB | Wrist RGB | DINO patch norm | FWR cross-attn

This script intentionally uses the recorded LeRobotDataset v3 with generated
chunk features, not the fast-cache arrays, because the overlay needs wrist RGB.
It reuses cached DINO patch tokens and loads only the trainable fast head from a
packaged HFRVLA checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hfrvla_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
POLICY_SRC = ROOT / "policy/lerobot_policy_hfrvla/src"
if str(POLICY_SRC) not in sys.path:
    sys.path.insert(0, str(POLICY_SRC))

from lerobot_policy_hfrvla.configuration_hfrvla import HFRVLAConfig
from lerobot_policy_hfrvla.fast_wrist_residual import (
    FastWristChunkResidualModule,
    _use_latent_context,
)


DEFAULT_DATASET_ROOT = ROOT / "checkpoints/HFRVLA_libero_v1_generated_chunks_merged"
DEFAULT_CHECKPOINT = ROOT / "checkpoints/hfrvla_fwr_chunk_generated_seq2_b1024p3_50k_packaged"
DEFAULT_OUT_DIR = ROOT / "outputs/dino_cross_attention_task8_fwr_v2_generated_50k"


VLA_IMAGE_KEY = "observation.images.image"
WRIST_IMAGE_KEY = "observation.images.image2"
IMAGE_KEY = WRIST_IMAGE_KEY
DINO_KEY = "observation.extra.dino_patches"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
ABASE_KEY = "observation.extra.a_base"
ABASE_CHUNK_KEY = "observation.extra.a_base_chunk"
CHUNK_STEP_KEY = "observation.extra.chunk_step_idx"
KIDX_KEY = "observation.extra.k_idx_norm"
ZGOAL_KEY = "observation.extra.z_goal"
ZPHASE_KEY = "observation.extra.z_phase"


REQUIRED_COLUMNS = [
    VLA_IMAGE_KEY,
    WRIST_IMAGE_KEY,
    STATE_KEY,
    ACTION_KEY,
    ABASE_KEY,
    ABASE_CHUNK_KEY,
    CHUNK_STEP_KEY,
    KIDX_KEY,
    ZGOAL_KEY,
    ZPHASE_KEY,
    DINO_KEY,
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]


@dataclass(frozen=True)
class AttentionMaps:
    per_head: np.ndarray
    mean: np.ndarray
    entropy: float


@dataclass(frozen=True)
class EpisodeSpec:
    episode_index: int
    parquet_path: Path
    num_task_frames: int


def _numeric_file_index(path: Path) -> int:
    return int(path.stem.split("-")[-1])


def _parquet_paths(dataset_root: Path) -> list[Path]:
    paths = sorted((dataset_root / "data").glob("*/*.parquet"), key=_numeric_file_index)
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")
    return paths


def _normalize_map(values: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _to_nested_array(value, *, dtype=np.float32) -> np.ndarray:
    """Convert pandas object/list encodings into a dense ndarray."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = np.asarray(value.tolist(), dtype=dtype)
    return arr.astype(dtype, copy=False).copy()


def _to_vector(value, *, dtype=np.float32) -> np.ndarray:
    arr = _to_nested_array(value, dtype=dtype)
    return arr.reshape(-1)


def _to_scalar(value) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    arr = np.asarray(value).reshape(-1)
    return float(arr[0])


def iter_strided_rows(df: pd.DataFrame, *, stride: int):
    """Yield row dictionaries without materializing pandas row blocks.

    Pandas ``iterrows()`` and DataFrame-level ``iloc`` can try to interleave
    nested Arrow extension dtypes from LeRobot parquet files and fail before
    yielding a row. Column-level access keeps each nested cell intact.
    """
    if stride <= 0:
        raise ValueError("stride must be > 0")
    columns = list(df.columns)
    for pos in range(0, len(df), stride):
        yield {column: df[column].iloc[pos] for column in columns}


def _decode_image(cell, *, dataset_root: Path | None = None) -> np.ndarray:
    if isinstance(cell, dict):
        if cell.get("bytes") is not None:
            image = Image.open(io.BytesIO(cell["bytes"]))
        elif cell.get("path"):
            if dataset_root is None:
                raise ValueError("Image cell stores a path but dataset_root was not provided.")
            image = Image.open(dataset_root / cell["path"])
        else:
            raise ValueError("Image cell has neither bytes nor path.")
    elif isinstance(cell, (str, os.PathLike)):
        image = Image.open(cell)
    else:
        image = Image.fromarray(np.asarray(cell))
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    heat = _normalize_map(heatmap)
    image = Image.fromarray(np.uint8(np.clip(heat, 0, 1) * 255), mode="L")
    return np.asarray(image.resize(size, resample=Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def _apply_colormap(heatmap: np.ndarray, *, cmap: str) -> np.ndarray:
    import matplotlib

    mapped = matplotlib.colormaps[cmap](_normalize_map(heatmap))[..., :3]
    return np.asarray(mapped * 255, dtype=np.uint8)


def _overlay_heatmap(
    rgb: np.ndarray,
    heatmap: np.ndarray,
    *,
    cmap: str,
    alpha: float = 0.48,
) -> np.ndarray:
    heat = _resize_heatmap(heatmap, (rgb.shape[1], rgb.shape[0]))
    color = _apply_colormap(heat, cmap=cmap).astype(np.float32)
    base = rgb.astype(np.float32)
    mixed = (1.0 - alpha) * base + alpha * color
    return np.clip(mixed, 0, 255).astype(np.uint8)


def dino_patch_norm_heatmap(dino_patches: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return a normalized 14x14 heatmap from DINO patch-token L2 norms."""
    patches = torch.as_tensor(dino_patches, dtype=torch.float32)
    if patches.ndim != 2 or patches.shape[0] != 196:
        raise ValueError(f"Expected dino_patches shape (196, D), got {tuple(patches.shape)}")
    norms = torch.linalg.vector_norm(patches, dim=-1).reshape(14, 14)
    return _normalize_map(norms.detach().cpu().numpy())


def compute_cross_attention_maps(
    module: FastWristChunkResidualModule,
    *,
    dino_patches: torch.Tensor | np.ndarray,
    z_goal: torch.Tensor | np.ndarray,
    z_phase: torch.Tensor | np.ndarray,
) -> AttentionMaps:
    """Compute per-head and mean cross-attention maps for one frame.

    The map is from the task-conditioned query to the 14x14 wrist DINO patch
    tokens. Returned attention values are raw probabilities; rendering
    normalizes them per frame.
    """
    module.eval()
    device = next(module.parameters()).device
    module_dtype = module.patch_proj.weight.dtype

    patches = torch.as_tensor(dino_patches, device=device, dtype=module_dtype).reshape(1, 196, -1)
    z_goal_t = torch.as_tensor(z_goal, device=device, dtype=module_dtype).reshape(1, -1)
    z_phase_t = torch.as_tensor(z_phase, device=device, dtype=module_dtype).reshape(1, -1)

    with torch.no_grad():
        patches_proj = module.patch_proj(patches)
        if _use_latent_context(module.config):
            query_in = torch.cat([z_goal_t, z_phase_t], dim=-1)
        else:
            query_in = torch.zeros(
                z_goal_t.shape[0],
                module.zgoal_dim + module.zphase_dim,
                device=device,
                dtype=module_dtype,
            )
        query = module.query_mlp(query_in).unsqueeze(1)
        _, weights = module.cross_attn(
            query,
            patches_proj,
            patches_proj,
            need_weights=True,
            average_attn_weights=False,
        )

    # PyTorch returns (B, heads, target_len, source_len) for batch_first MHA.
    if weights.ndim != 4:
        raise ValueError(f"Expected attention weights with 4 dims, got {tuple(weights.shape)}")
    per_head = weights[0, :, 0, :].detach().float().cpu().numpy().reshape(-1, 14, 14)
    mean = per_head.mean(axis=0)
    flat = mean.reshape(-1).astype(np.float64)
    total = float(flat.sum())
    if total <= 0:
        entropy = 0.0
    else:
        prob = flat / total
        entropy = float(-(prob * np.log(prob + 1e-12)).sum())
    return AttentionMaps(per_head=per_head, mean=mean, entropy=entropy)


def _font(size: int = 16):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _panel(label: str, image: np.ndarray, *, size: int, label_height: int, font) -> Image.Image:
    thumb = Image.fromarray(image).resize((size, size), resample=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size + label_height), "white")
    canvas.paste(thumb, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, size, size, size + label_height), fill=(255, 255, 255))
    draw.text((8, size + 7), label, fill=(24, 33, 42), font=font)
    return canvas


def _render_labeled_panels(
    panels: list[tuple[str, np.ndarray]],
    *,
    title: str,
    panel_size: int,
) -> np.ndarray:
    margin = 16
    label_h = 32
    title_h = 48 if title else 0
    font = _font(15)
    title_font = _font(16)
    rendered = [
        _panel(label, np.asarray(image, dtype=np.uint8), size=panel_size, label_height=label_h, font=font)
        for label, image in panels
    ]
    width = panel_size * len(rendered) + margin * (len(rendered) + 1)
    height = title_h + panel_size + label_h + margin * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((margin, 10), title[:160], fill=(24, 33, 42), font=title_font)
    y = title_h + margin
    x = margin
    for panel in rendered:
        canvas.paste(panel, (x, y))
        x += panel_size + margin
    return np.asarray(canvas, dtype=np.uint8)


def render_three_panel_frame(
    *,
    wrist_rgb: np.ndarray,
    before_heatmap: np.ndarray,
    after_heatmap: np.ndarray,
    title: str = "",
    panel_size: int = 256,
) -> np.ndarray:
    """Render one 3-panel diagnostic frame as uint8 RGB."""
    rgb = np.asarray(wrist_rgb, dtype=np.uint8)
    before = _overlay_heatmap(rgb, before_heatmap, cmap="viridis")
    after = _overlay_heatmap(rgb, after_heatmap, cmap="cividis")
    return _render_labeled_panels(
        [
            ("Wrist RGB", rgb),
            ("DINO patch norm", before),
            ("Cross-attn mean", after),
        ],
        title=title,
        panel_size=panel_size,
    )


def render_comparison_frame(
    *,
    vla_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    before_heatmap: np.ndarray,
    after_heatmap: np.ndarray,
    title: str = "",
    panel_size: int = 256,
) -> np.ndarray:
    """Render one 4-panel frame with raw VLA inputs and FWR attention."""
    vla = np.asarray(vla_rgb, dtype=np.uint8)
    wrist = np.asarray(wrist_rgb, dtype=np.uint8)
    before = _overlay_heatmap(wrist, before_heatmap, cmap="viridis")
    after = _overlay_heatmap(wrist, after_heatmap, cmap="cividis")
    return _render_labeled_panels(
        [
            ("VLA third-person RGB", vla),
            ("Wrist RGB", wrist),
            ("DINO patch norm", before),
            ("FWR cross-attn", after),
        ],
        title=title,
        panel_size=panel_size,
    )


def write_video(frames: Iterable[np.ndarray], out_path: Path, *, fps: int) -> None:
    frame_list = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    if not frame_list:
        raise ValueError(f"No frames to write for {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, np.stack(frame_list, axis=0), fps=fps, codec="libx264", quality=8)


def load_fast_module_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    """Load non-backbone fast head weights from a packaged policy checkpoint."""
    path = checkpoint
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"model.safetensors not found: {path}")
    raw = load_file(str(path), device="cpu")
    state = {}
    for key, value in raw.items():
        if not key.startswith("fast."):
            continue
        stripped = key.removeprefix("fast.")
        if stripped.startswith("backbone."):
            continue
        state[stripped] = value
    if not state:
        raise ValueError(f"No non-backbone fast.* weights found in {path}")
    return state


def _load_checkpoint_config(checkpoint: Path) -> dict:
    config_path = checkpoint / "config.json" if checkpoint.is_dir() else checkpoint.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found next to checkpoint: {config_path}")
    return json.loads(config_path.read_text())


def build_fast_module(checkpoint: Path, *, device: torch.device) -> FastWristChunkResidualModule:
    raw = _load_checkpoint_config(checkpoint)
    mode = str(raw.get("residual_merge_mode", "fast_wrist_chunk")).lower()
    if mode == "a2c2":
        mode = "fast_wrist"
    if mode != "fast_wrist_chunk":
        raise ValueError(
            f"This script is for FWR-v2 fast_wrist_chunk checkpoints; got residual_merge_mode={mode!r}."
        )

    cfg = HFRVLAConfig(
        residual_merge_mode="fast_wrist_chunk",
        offline_training_mode=True,
        n_action_steps=int(raw.get("n_action_steps", 50)),
        chunk_size=int(raw.get("chunk_size", 50)),
        seq_len=int(raw.get("seq_len", 2)),
        dinov3_feature_dim=int(raw.get("dinov3_feature_dim", 384)),
        pool_query_dim=int(raw.get("pool_query_dim", 256)),
        pool_n_heads=int(raw.get("pool_n_heads", 4)),
        head_hidden=int(raw.get("head_hidden", 64)),
        device=str(device),
        delta_max=float(raw.get("delta_max", 0.2)),
        fast_residual_alpha=raw.get("fast_residual_alpha", raw.get("a2c2_alpha", 1.0)),
        fast_residual_use_latent_context=raw.get(
            "fast_residual_use_latent_context",
            raw.get("a2c2_use_latent_context", True),
        ),
    )
    module = FastWristChunkResidualModule(
        config=cfg,
        action_dim=7,
        proprio_dim=8,
        zgoal_dim=int(raw.get("offline_zgoal_dim", 960)),
        zphase_dim=int(raw.get("offline_zphase_dim", 480)),
    )
    state = load_fast_module_state(checkpoint)
    missing, unexpected = module.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected fast module keys: {sorted(unexpected)[:12]}")
    relevant_missing = [key for key in missing if not key.startswith("backbone.")]
    if relevant_missing:
        raise ValueError(f"Missing fast module keys: {sorted(relevant_missing)[:12]}")
    return module.to(device).eval()


def discover_task_episodes(
    dataset_root: Path,
    *,
    task_index: int,
    max_episodes: int,
) -> list[EpisodeSpec]:
    episodes: list[EpisodeSpec] = []
    seen: set[int] = set()
    for path in _parquet_paths(dataset_root):
        df = pd.read_parquet(path, columns=["episode_index", "task_index"])
        selected = df[df["task_index"] == task_index]
        if selected.empty:
            continue
        for episode_index, group in selected.groupby("episode_index", sort=True):
            episode = int(episode_index)
            if episode in seen:
                continue
            seen.add(episode)
            episodes.append(
                EpisodeSpec(
                    episode_index=episode,
                    parquet_path=path,
                    num_task_frames=int(len(group)),
                )
            )
            if len(episodes) >= max_episodes:
                return episodes
    return episodes


def load_task_text(dataset_root: Path, *, task_index: int) -> str:
    tasks_path = dataset_root / "meta/tasks.parquet"
    if not tasks_path.exists():
        return f"task_index={task_index}"
    tasks = pd.read_parquet(tasks_path)
    if "task_index" not in tasks.columns:
        return f"task_index={task_index}"
    match = tasks[tasks["task_index"] == task_index]
    if match.empty:
        return f"task_index={task_index}"
    return str(match.index[0])


def _compute_delta_norm(
    module: FastWristChunkResidualModule,
    *,
    state: np.ndarray,
    a_base: np.ndarray,
    k_idx_norm: float,
    z_goal: np.ndarray,
    z_phase: np.ndarray,
    a_base_chunk: np.ndarray,
    chunk_step_idx: int,
    dino_patches: np.ndarray,
) -> float:
    device = next(module.parameters()).device
    with torch.no_grad():
        out = module(
            wrist_rgb=None,
            proprio=torch.as_tensor(state, device=device, dtype=torch.float32).reshape(1, -1),
            a_base_k=torch.as_tensor(a_base, device=device, dtype=torch.float32).reshape(1, -1),
            k_idx_norm=torch.tensor([[float(k_idx_norm)]], device=device, dtype=torch.float32),
            z_goal=torch.as_tensor(z_goal, device=device, dtype=torch.float32).reshape(1, -1),
            z_phase=torch.as_tensor(z_phase, device=device, dtype=torch.float32).reshape(1, -1),
            a_base_chunk=torch.as_tensor(a_base_chunk, device=device, dtype=torch.float32).reshape(1, -1, 7),
            chunk_step_idx=torch.tensor([[int(chunk_step_idx)]], device=device, dtype=torch.int64),
            dino_patches=torch.as_tensor(dino_patches, device=device, dtype=torch.float32).reshape(1, 196, -1),
        )
    return float(torch.linalg.vector_norm(out.delta_a.detach().float()).cpu().item())


def _save_debug_heads(
    path: Path,
    *,
    attention: AttentionMaps,
    before_heatmap: np.ndarray,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        per_head=attention.per_head,
        mean=attention.mean,
        before=before_heatmap,
        metadata=json.dumps(metadata, ensure_ascii=False),
    )


def render_episode(
    spec: EpisodeSpec,
    *,
    module: FastWristChunkResidualModule,
    dataset_root: Path,
    task_index: int,
    task_text: str,
    out_dir: Path,
    fps: int,
    stride: int,
    debug_head_stride: int,
) -> list[dict]:
    df = pd.read_parquet(spec.parquet_path, columns=REQUIRED_COLUMNS)
    df = df[(df["episode_index"] == spec.episode_index) & (df["task_index"] == task_index)].copy()
    df = df.sort_values("frame_index")
    if df.empty:
        return []

    frames = []
    manifest_rows: list[dict] = []
    debug_dir = out_dir / "debug_heads"
    video_name = f"task{task_index:02d}_episode{spec.episode_index:04d}.mp4"
    video_path = out_dir / "videos" / video_name

    rendered_idx = 0
    for row in iter_strided_rows(df, stride=stride):
        dino = _to_nested_array(row[DINO_KEY], dtype=np.float32).reshape(196, -1)
        z_goal = _to_vector(row[ZGOAL_KEY], dtype=np.float32)
        z_phase = _to_vector(row[ZPHASE_KEY], dtype=np.float32)
        vla_rgb = _decode_image(row[VLA_IMAGE_KEY], dataset_root=dataset_root)
        wrist_rgb = _decode_image(row[WRIST_IMAGE_KEY], dataset_root=dataset_root)
        before = dino_patch_norm_heatmap(dino)
        attention = compute_cross_attention_maps(module, dino_patches=dino, z_goal=z_goal, z_phase=z_phase)

        frame_index = int(row["frame_index"])
        global_index = int(row["index"])
        timestamp = float(row["timestamp"])
        title = f"task {task_index} | episode {spec.episode_index} | frame {frame_index} | t={timestamp:.2f}s"
        frames.append(
            render_comparison_frame(
                vla_rgb=vla_rgb,
                wrist_rgb=wrist_rgb,
                before_heatmap=before,
                after_heatmap=attention.mean,
                title=title,
            )
        )

        state = _to_vector(row[STATE_KEY], dtype=np.float32)
        action = _to_vector(row[ACTION_KEY], dtype=np.float32)
        a_base = _to_vector(row[ABASE_KEY], dtype=np.float32)
        a_base_chunk = _to_nested_array(row[ABASE_CHUNK_KEY], dtype=np.float32).reshape(-1, 7)
        chunk_step_idx = int(_to_scalar(row[CHUNK_STEP_KEY]))
        k_idx_norm = float(_to_scalar(row[KIDX_KEY]))
        expert_delta_norm = float(np.linalg.norm(action - a_base))
        pred_delta_norm = _compute_delta_norm(
            module,
            state=state,
            a_base=a_base,
            k_idx_norm=k_idx_norm,
            z_goal=z_goal,
            z_phase=z_phase,
            a_base_chunk=a_base_chunk,
            chunk_step_idx=chunk_step_idx,
            dino_patches=dino,
        )
        metadata = {
            "task_index": task_index,
            "task": task_text,
            "episode_index": spec.episode_index,
            "frame_index": frame_index,
            "global_index": global_index,
            "timestamp": timestamp,
            "chunk_step_idx": chunk_step_idx,
            "k_idx_norm": k_idx_norm,
            "cross_attn_entropy": attention.entropy,
            "expert_delta_norm": expert_delta_norm,
            "pred_delta_norm": pred_delta_norm,
            "vla_image_key": VLA_IMAGE_KEY,
            "wrist_image_key": WRIST_IMAGE_KEY,
        }
        if debug_head_stride > 0 and rendered_idx % debug_head_stride == 0:
            _save_debug_heads(
                debug_dir / f"task{task_index:02d}_episode{spec.episode_index:04d}_frame{frame_index:04d}.npz",
                attention=attention,
                before_heatmap=before,
                metadata=metadata,
            )
        manifest_rows.append(
            {
                **metadata,
                "video_path": str(video_path.relative_to(out_dir)),
                "video_frame_index": rendered_idx,
                "source_parquet": str(spec.parquet_path.relative_to(dataset_root)),
            }
        )
        rendered_idx += 1

    write_video(frames, video_path, fps=fps)
    return manifest_rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_episode_indices(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render task-conditioned DINO/FWR cross-attention videos for HFRVLA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--task-index", type=int, default=8)
    parser.add_argument("--max-episodes", type=int, default=5)
    parser.add_argument(
        "--episode-indices",
        default=None,
        help="Comma-separated explicit episode_index values. Overrides --max-episodes discovery.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--debug-head-stride", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _write_run_metadata(path: Path, *, args: argparse.Namespace, task_text: str, episodes: list[EpisodeSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_root": str(args.dataset_root),
        "checkpoint": str(args.checkpoint),
        "task_index": args.task_index,
        "task": task_text,
        "fps": args.fps,
        "stride": args.stride,
        "debug_head_stride": args.debug_head_stride,
        "episodes": [
            {
                "episode_index": spec.episode_index,
                "parquet_path": str(spec.parquet_path),
                "num_task_frames": spec.num_task_frames,
            }
            for spec in episodes
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.stride <= 0:
        raise SystemExit("--stride must be > 0")
    if args.fps <= 0:
        raise SystemExit("--fps must be > 0")

    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[dino-viz] dataset_root={args.dataset_root}")
    print(f"[dino-viz] checkpoint={args.checkpoint}")
    print(f"[dino-viz] out_dir={args.out_dir}")

    task_text = load_task_text(args.dataset_root, task_index=args.task_index)
    explicit = parse_episode_indices(args.episode_indices)
    if explicit is None:
        episodes = discover_task_episodes(
            args.dataset_root,
            task_index=args.task_index,
            max_episodes=args.max_episodes,
        )
    else:
        by_episode = {
            spec.episode_index: spec
            for spec in discover_task_episodes(
                args.dataset_root,
                task_index=args.task_index,
                max_episodes=10_000,
            )
        }
        missing = [episode for episode in explicit if episode not in by_episode]
        if missing:
            raise SystemExit(f"Requested task-{args.task_index} episodes not found: {missing}")
        episodes = [by_episode[episode] for episode in explicit]
    if not episodes:
        raise SystemExit(f"No episodes found for task_index={args.task_index}")

    print(f"[dino-viz] task={args.task_index}: {task_text}")
    print("[dino-viz] episodes=" + ", ".join(str(spec.episode_index) for spec in episodes))
    device = torch.device(args.device)
    module = build_fast_module(args.checkpoint, device=device)

    all_rows: list[dict] = []
    for spec in episodes:
        print(f"[dino-viz] rendering episode={spec.episode_index} frames={spec.num_task_frames}")
        rows = render_episode(
            spec,
            module=module,
            dataset_root=args.dataset_root,
            task_index=args.task_index,
            task_text=task_text,
            out_dir=args.out_dir,
            fps=args.fps,
            stride=args.stride,
            debug_head_stride=args.debug_head_stride,
        )
        all_rows.extend(rows)

    write_manifest(args.out_dir / "manifest.csv", all_rows)
    _write_run_metadata(args.out_dir / "run_metadata.json", args=args, task_text=task_text, episodes=episodes)
    print(f"[dino-viz] wrote {len(all_rows)} manifest rows")
    print(f"[dino-viz] videos: {args.out_dir / 'videos'}")
    print(f"[dino-viz] manifest: {args.out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
