#!/usr/bin/env python
"""Merge HFRVLA recording shards into a single LeRobotDataset.

After running ``record_hfrvla_libero.py`` with disjoint ``--ep-from / --ep-to``
ranges into N separate output dirs (shards), this script consolidates them
into one final dataset that ``lerobot-train`` can consume.

Mechanism: use the FIRST shard as the destination (via ``LeRobotDataset.resume``)
and read every frame from each subsequent shard, re-emitting it through
``add_frame`` + ``save_episode``. Videos are re-encoded into the destination's
codec — slow but correct, and avoids hand-patching v3 metadata offsets.

Example:
    python scripts/merge_hfrvla_shards.py \\
        --shards checkpoints/HFRVLA_libero_v1_shard_a \\
                 checkpoints/HFRVLA_libero_v1_shard_b \\
                 checkpoints/HFRVLA_libero_v1_shard_c \\
        --out-root checkpoints/HFRVLA_libero_v1

NOTE: --out-root must NOT exist beforehand; this script COPIES shard A to
out-root (preserving its metadata + parquet/videos) and then appends B, C.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch

POLICY_SRC = Path(__file__).resolve().parents[1] / "policy" / "lerobot_policy_hfrvla" / "src"
if POLICY_SRC.exists():
    sys.path.insert(0, str(POLICY_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shards",
        nargs="+",
        type=Path,
        required=True,
        help="Shard dirs in episode order (first shard becomes the destination base).",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output directory for the merged dataset. Must not exist.",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        default="HFRVLA_libero_v1",
        help="Identifier for the merged dataset's metadata.",
    )
    p.add_argument(
        "--log-every-frames",
        type=int,
        default=2000,
        help="Print progress every N frames during the merge pass.",
    )
    return p.parse_args()


def _copy_shard_a(shard_a: Path, out_root: Path) -> None:
    """Copy the first shard verbatim — it becomes the destination base."""
    if out_root.exists():
        raise SystemExit(f"[merge] --out-root already exists: {out_root}")
    print(f"[merge] copying shard A → out_root (preserving videos/parquet) ...", flush=True)
    shutil.copytree(shard_a, out_root)
    print(f"[merge] base copy done.", flush=True)


def _append_shard(dst: LeRobotDataset, shard: Path, *, log_every: int) -> int:
    """Read every frame from a shard and add_frame() it into dst.

    Returns the number of episodes appended.
    """
    src = LeRobotDataset(repo_id=src_repo_id_from(shard), root=str(shard))
    n_eps = int(src.num_episodes)
    print(f"[merge] appending {shard.name}: {n_eps} episodes, {src.num_frames} frames ...", flush=True)

    frames_written = 0
    t0 = time.time()
    for ep_idx in range(n_eps):
        ep_meta = src.meta.episodes[ep_idx]
        ep_from = int(ep_meta["dataset_from_index"])
        ep_to = int(ep_meta["dataset_to_index"])
        ep_task = (
            ep_meta["tasks"][0]
            if isinstance(ep_meta.get("tasks"), list) and ep_meta["tasks"]
            else "do the task"
        )
        for t in range(ep_from, ep_to):
            sample = src[t]
            frame = _sample_to_frame(sample, ep_task)
            dst.add_frame(frame)
            frames_written += 1
            if frames_written % log_every == 0:
                rate = frames_written / max(1e-6, time.time() - t0)
                print(
                    f"[merge]   {shard.name}: {frames_written} frames "
                    f"({rate:.1f} f/s)", flush=True
                )
        dst.save_episode()
    return n_eps


def src_repo_id_from(shard: Path) -> str:
    """Read the shard's repo_id from its info.json (best-effort).

    Falls back to the directory name if the field is missing.
    """
    import json
    info_path = shard / "meta" / "info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            rid = info.get("repo_id") or info.get("dataset_repo_id")
            if rid:
                return str(rid)
        except Exception:
            pass
    return shard.name


def _sample_to_frame(sample: dict, ep_task: str) -> dict:
    """Translate a LeRobotDataset __getitem__ row into add_frame() input.

    The source sample has decoded image tensors in (C, H, W) float[0,1].
    LeRobotDataset.add_frame expects:
      - image features: (H, W, C) uint8
      - other features: tensors with their declared shape/dtype
      - 'task': string
    """
    def _img(t: torch.Tensor) -> torch.Tensor:
        x = t
        if x.dim() == 4:
            x = x.squeeze(0)
        if x.shape[0] == 3:
            x = x.permute(1, 2, 0)
        x = x.float()
        if x.max() <= 1.5:
            x = x * 255.0
        return x.clamp(0, 255).to(torch.uint8).cpu()

    frame = {
        "observation.images.image":  _img(sample["observation.images.image"]),
        "observation.images.image2": _img(sample["observation.images.image2"]),
        "observation.state":         sample["observation.state"].cpu().float(),
        "action":                    sample["action"].cpu().float(),
        "observation.extra.z_goal":        sample["observation.extra.z_goal"].cpu().float(),
        "observation.extra.z_phase":       sample["observation.extra.z_phase"].cpu().float(),
        "observation.extra.a_base":        sample["observation.extra.a_base"].cpu().float(),
        "observation.extra.k_idx_norm":    sample["observation.extra.k_idx_norm"].cpu().float(),
        "observation.extra.dino_patches":  sample["observation.extra.dino_patches"].cpu(),
        "observation.extra.contact_label": sample["observation.extra.contact_label"].cpu().float(),
        "task": ep_task,
    }
    return frame


def main() -> None:
    args = parse_args()
    if len(args.shards) < 2:
        raise SystemExit("[merge] need at least 2 shards to merge")
    for s in args.shards:
        if not s.exists():
            raise SystemExit(f"[merge] shard not found: {s}")
        if not (s / "meta" / "info.json").exists():
            raise SystemExit(f"[merge] shard {s} is missing meta/info.json — was it finalized?")

    # Step 1: copy shard A to out_root verbatim.
    _copy_shard_a(args.shards[0], args.out_root)

    # Step 2: resume the now-populated out_root as a writable dataset.
    print(f"[merge] resuming dst at {args.out_root} ...", flush=True)
    dst = LeRobotDataset.resume(
        repo_id=args.repo_id,
        root=str(args.out_root),
    )
    print(
        f"[merge]   resumed with {int(dst.num_episodes)} episodes, "
        f"{int(dst.num_frames)} frames", flush=True
    )

    # Step 3: append shards B, C, ...
    appended = 0
    for shard in args.shards[1:]:
        appended += _append_shard(dst, shard, log_every=args.log_every_frames)

    # Step 4: finalize.
    print(f"[merge] finalizing ...", flush=True)
    dst.finalize()
    print(
        f"[merge] done. merged dataset at {args.out_root} "
        f"(appended {appended} episodes from {len(args.shards) - 1} shards)",
        flush=True,
    )


if __name__ == "__main__":
    main()
