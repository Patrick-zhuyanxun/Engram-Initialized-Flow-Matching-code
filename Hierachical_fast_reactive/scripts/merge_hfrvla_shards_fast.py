#!/usr/bin/env python
"""Fast merger for HFRVLA recording shards (no re-decode, no re-encode).

The previous merger (``merge_hfrvla_shards.py``) re-instantiated each shard
through ``LeRobotDataset(repo_id=..., root=...)``, which silently failed
because the synthetic repo_id isn't on the Hub. Result: only shard A was
copied; B/C/D never got appended.

This script bypasses ``LeRobotDataset`` entirely. It manipulates files +
metadata directly:

  1. Hardlink each shard's ``data/chunk-000/file-{k}.parquet`` into the
     merged dir with an offset rename. For shards whose local task_index
     mapping differs from the global one, rewrite the ``task_index``
     column in-flight.
  2. Concatenate per-shard ``meta/episodes/chunk-000/file-000.parquet``
     after offsetting ``episode_index``, ``data/file_index``,
     ``dataset_{from,to}_index``.
  3. Dedupe task strings across shards, write a single ``meta/tasks.parquet``.
  4. Aggregate per-episode stats into a dataset-level ``meta/stats.json``
     via ``compute_stats.aggregate_feature_stats``.
  5. Bump ``chunks_size`` in ``meta/info.json`` so every file fits in
     ``chunk-000`` (no rollover bookkeeping needed).

Runs in seconds-to-minutes (hardlinks are O(1); task_index rewrites are the
only slow step, and only kick in if a shard's tasks overlap with earlier
shards in non-identity order).

Example:
    python scripts/merge_hfrvla_shards_fast.py \\
        --shards checkpoints/HFRVLA_libero_v1_shard_a \\
                 checkpoints/HFRVLA_libero_v1_shard_b \\
                 checkpoints/HFRVLA_libero_v1_shard_c \\
                 checkpoints/HFRVLA_libero_v1_shard_d \\
        --out-root checkpoints/HFRVLA_libero_v1 \\
        --force-delete
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

LEROBOT_SRC = Path.home() / "Robotic_infra/lerobot/src"
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--shards", nargs="+", type=Path, required=True,
                   help="Shard root dirs (each must contain meta/info.json + "
                        "meta/tasks.parquet).")
    p.add_argument("--out-root", type=Path, required=True,
                   help="Destination dir for merged dataset.")
    p.add_argument("--repo-id", type=str, default="HFRVLA_libero_v1",
                   help="Logical name; written into info.json if missing.")
    p.add_argument("--force-delete", action="store_true",
                   help="If --out-root exists, remove it first.")
    p.add_argument("--copy", action="store_true",
                   help="Copy files instead of hardlinking (use across "
                        "filesystems; slower + uses 2× disk).")
    return p.parse_args()


def _build_task_remaps(shards):
    """Return (global_tasks_list, [per_shard_remap_dict]).

    Each shard's tasks.parquet has index=task_string, column=task_index.
    Merged global tasks list dedupes by string and assigns new sequential
    indices in the order shards are passed.
    """
    global_tasks: list[str] = []
    task_to_global: dict[str, int] = {}
    remaps: list[dict[int, int]] = []
    for s in shards:
        df = pd.read_parquet(s / "meta/tasks.parquet")
        local_to_task = dict(zip(df["task_index"].tolist(), df.index.tolist()))
        remap = {}
        for li in sorted(local_to_task.keys()):
            t = local_to_task[li]
            if t not in task_to_global:
                task_to_global[t] = len(global_tasks)
                global_tasks.append(t)
            remap[li] = task_to_global[t]
        remaps.append(remap)
    return global_tasks, remaps


def _compute_offsets(shards):
    """Return cumulative offsets (length=len(shards)+1) + per-shard info dicts."""
    ep_off = [0]
    fr_off = [0]
    fi_off = [0]
    infos = []
    for s in shards:
        info = json.load(open(s / "meta/info.json"))
        n_files = len(list((s / "data/chunk-000").glob("file-*.parquet")))
        ep_off.append(ep_off[-1] + info["total_episodes"])
        fr_off.append(fr_off[-1] + info["total_frames"])
        fi_off.append(fi_off[-1] + n_files)
        infos.append(info)
    return ep_off, fr_off, fi_off, infos


def _link_data_files(shards, fi_offsets, remaps, out_data_dir, *, use_copy):
    """Hardlink/copy data parquet files with offset renames.

    For shards with non-identity task_index remap, rewrite the column.
    """
    out_data_dir.mkdir(parents=True, exist_ok=True)
    total_written = 0
    t0 = time.time()
    for i, shard in enumerate(shards):
        remap = remaps[i]
        identity = all(k == v for k, v in remap.items())
        offset = fi_offsets[i]
        src_dir = shard / "data/chunk-000"
        files = sorted(src_dir.glob("file-*.parquet"))
        print(f"  shard {shard.name}: {len(files)} files, file_offset={offset}, "
              f"identity_remap={identity}", flush=True)
        for j, fp in enumerate(files):
            local_idx = int(fp.stem.split("-")[1])
            new_idx = local_idx + offset
            dst = out_data_dir / f"file-{new_idx:03d}.parquet"
            if dst.exists():
                dst.unlink()
            if identity:
                if use_copy:
                    shutil.copy2(fp, dst)
                else:
                    os.link(fp, dst)
            else:
                table = pq.read_table(fp)
                if "task_index" in table.column_names:
                    col_idx = table.schema.get_field_index("task_index")
                    field = table.schema.field(col_idx)
                    old = table.column(col_idx).to_pylist()
                    new = [remap[v] for v in old]
                    new_array = pa.array(new, type=field.type)
                    table = table.set_column(col_idx, field, new_array)
                pq.write_table(table, dst, compression="snappy")
            total_written += 1
            if total_written % 100 == 0:
                rate = total_written / max(1e-6, time.time() - t0)
                print(f"    {total_written} files done ({rate:.0f} f/s)",
                      flush=True)
    return total_written


def _merge_episodes(shards, ep_off, fr_off, fi_off, out_eps_dir):
    """Concatenate per-shard episodes parquet with offsets applied."""
    dfs = []
    for i, shard in enumerate(shards):
        df = pd.read_parquet(shard / "meta/episodes/chunk-000/file-000.parquet")
        df = df.copy()
        df["episode_index"] = df["episode_index"] + ep_off[i]
        df["data/file_index"] = df["data/file_index"] + fi_off[i]
        df["dataset_from_index"] = df["dataset_from_index"] + fr_off[i]
        df["dataset_to_index"] = df["dataset_to_index"] + fr_off[i]
        # data/chunk_index stays 0 — we bump chunks_size so all fits in chunk-000.
        # meta/episodes/chunk_index + meta/episodes/file_index stay 0 — single output file.
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    out_eps_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_eps_dir / "file-000.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"  wrote {out_path}: {len(merged)} episodes, {len(merged.columns)} cols",
          flush=True)
    return merged


def _write_tasks_parquet(global_tasks, out_path):
    df = pd.DataFrame(
        {"task_index": list(range(len(global_tasks)))},
        index=pd.Index(global_tasks, name="task"),
    )
    df.to_parquet(out_path)
    print(f"  wrote {out_path}: {len(df)} tasks", flush=True)


def _write_info_json(template_info, total_eps, total_frames, total_tasks,
                     total_files, repo_id, out_path):
    info = dict(template_info)
    info["total_episodes"] = int(total_eps)
    info["total_frames"] = int(total_frames)
    info["total_tasks"] = int(total_tasks)
    needed = total_files + 100
    info["chunks_size"] = int(max(info.get("chunks_size", 1000), needed))
    info["splits"] = {"train": f"0:{int(total_eps)}"}
    if "repo_id" not in info:
        info["repo_id"] = repo_id
    with open(out_path, "w") as f:
        json.dump(info, f, indent=4)
    print(f"  wrote {out_path}: total_eps={total_eps}, total_frames={total_frames}, "
          f"total_tasks={total_tasks}, chunks_size={info['chunks_size']}",
          flush=True)


def _coerce_stat(value, target_shape):
    """Flatten parquet-stored stat arrays into clean float arrays.

    Image stats are stored as deeply nested object arrays
    (e.g., array of array of array of scalar for shape (3,1,1)) because
    pyarrow's list-of-list encoding doesn't auto-flatten back to ndarray.
    Walk the structure, collect float leaves, reshape to target.
    """
    arr = np.asarray(value)
    if arr.dtype != np.object_:
        # already a clean numeric array — ensure shape
        if arr.shape != target_shape and arr.size == int(np.prod(target_shape)):
            arr = arr.reshape(target_shape)
        return arr.astype(np.float64, copy=False)
    flat: list[float] = []

    def walk(v):
        if isinstance(v, np.ndarray):
            if v.dtype == np.object_:
                for elem in v.flat:
                    walk(elem)
            else:
                flat.extend(v.flatten().tolist())
        elif isinstance(v, (list, tuple)):
            for elem in v:
                walk(elem)
        else:
            flat.append(float(v))

    walk(value)
    out = np.array(flat, dtype=np.float64)
    expected = int(np.prod(target_shape))
    if out.size != expected:
        raise ValueError(
            f"flattened stat has {out.size} elements but target_shape "
            f"{target_shape} expects {expected}"
        )
    return out.reshape(target_shape)


def _aggregate_stats_json(merged_eps_df, template_stats, out_path):
    """Aggregate per-episode stats from the merged episodes parquet."""
    from lerobot.datasets.compute_stats import aggregate_feature_stats

    out = {}
    for feat in template_stats.keys():
        needed = ("count", "mean", "std", "min", "max")
        if any(f"stats/{feat}/{k}" not in merged_eps_df.columns for k in needed):
            print(f"    SKIP feature {feat} (missing required stats columns)",
                  flush=True)
            continue
        # Target shapes inferred from the template (per-shard) stats.json.
        target_shapes = {
            k: np.asarray(template_stats[feat][k]).shape
            for k in needed if k in template_stats[feat]
        }
        q_keys = [k for k in ("q01", "q10", "q50", "q90", "q99")
                  if f"stats/{feat}/{k}" in merged_eps_df.columns
                  and k in template_stats[feat]]
        for q in q_keys:
            target_shapes[q] = np.asarray(template_stats[feat][q]).shape

        per_ep = []
        for _, row in merged_eps_df.iterrows():
            s = {}
            # count is always a clean int array of shape (1,); keep it int.
            s["count"] = np.asarray(row[f"stats/{feat}/count"])
            for k in ("mean", "std", "min", "max"):
                s[k] = _coerce_stat(row[f"stats/{feat}/{k}"], target_shapes[k])
            for q in q_keys:
                s[q] = _coerce_stat(row[f"stats/{feat}/{q}"], target_shapes[q])
            per_ep.append(s)
        agg = aggregate_feature_stats(per_ep)
        out[feat] = {k: np.asarray(v).tolist() for k, v in agg.items()}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {out_path}: {len(out)} features aggregated", flush=True)


def main() -> None:
    args = parse_args()
    if len(args.shards) < 2:
        raise SystemExit("[merge] need at least 2 shards")
    for s in args.shards:
        for f in ("meta/info.json", "meta/tasks.parquet",
                  "meta/episodes/chunk-000/file-000.parquet",
                  "meta/stats.json"):
            if not (s / f).exists():
                raise SystemExit(f"[merge] shard {s} missing: {f}")
        if not (s / "data/chunk-000").exists():
            raise SystemExit(f"[merge] shard {s} missing data/chunk-000/")

    if args.out_root.exists():
        if args.force_delete:
            print(f"[merge] --force-delete: removing {args.out_root}", flush=True)
            shutil.rmtree(args.out_root)
        else:
            raise SystemExit(
                f"[merge] --out-root exists: {args.out_root}\n"
                f"        pass --force-delete to overwrite, or pick a fresh dir"
            )

    args.out_root.mkdir(parents=True, exist_ok=True)

    print(f"[merge] step 1: build global task remap")
    global_tasks, remaps = _build_task_remaps(args.shards)
    print(f"  unique global tasks: {len(global_tasks)}", flush=True)
    for i, r in enumerate(remaps):
        identity = all(k == v for k, v in r.items())
        print(f"    shard {args.shards[i].name}: {len(r)} local tasks, "
              f"identity_remap={identity}", flush=True)

    print(f"[merge] step 2: compute offsets")
    ep_off, fr_off, fi_off, infos = _compute_offsets(args.shards)
    print(f"  total episodes: {ep_off[-1]}", flush=True)
    print(f"  total frames:   {fr_off[-1]}", flush=True)
    print(f"  total files:    {fi_off[-1]}", flush=True)

    print(f"[merge] step 3: link/copy data files")
    out_data_dir = args.out_root / "data/chunk-000"
    total_files = _link_data_files(
        args.shards, fi_off, remaps, out_data_dir, use_copy=args.copy,
    )
    print(f"  total data files written: {total_files}", flush=True)

    print(f"[merge] step 4: build merged episodes parquet")
    out_eps_dir = args.out_root / "meta/episodes/chunk-000"
    merged_eps = _merge_episodes(args.shards, ep_off, fr_off, fi_off, out_eps_dir)

    print(f"[merge] step 5: write tasks.parquet")
    _write_tasks_parquet(global_tasks, args.out_root / "meta/tasks.parquet")

    print(f"[merge] step 6: write info.json")
    _write_info_json(
        infos[0], ep_off[-1], fr_off[-1], len(global_tasks),
        total_files, args.repo_id, args.out_root / "meta/info.json",
    )

    print(f"[merge] step 7: aggregate stats.json")
    template_stats = json.load(open(args.shards[0] / "meta/stats.json"))
    _aggregate_stats_json(merged_eps, template_stats,
                          args.out_root / "meta/stats.json")

    print(f"\n[merge] DONE → {args.out_root}")
    print(f"  total_episodes: {ep_off[-1]}")
    print(f"  total_frames:   {fr_off[-1]}")
    print(f"  total_tasks:    {len(global_tasks)}")


if __name__ == "__main__":
    main()
