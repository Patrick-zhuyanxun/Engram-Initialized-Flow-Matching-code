#!/usr/bin/env python
"""Alignment test: verify HFRVLA wrapper's I/O matches LIBERO env expectations.

What it does:
  1. Build an HFRVLA package with ``inference_disable_fast=True`` so
     ``select_action()`` short-circuits to SmolVLA's ``a_base``.
  2. Run ``lerobot-eval`` against ``libero_spatial`` task_ids 0, 1, 2 with
     5 episodes each.
  3. Report per-task success rate.

Why:
  Before training the fast module, we need to confirm:
    - HFRVLAConfig's feature overrides match what LIBERO env emits.
    - SmolVLA's action chunk pops out via the queue → returns through
      ``select_action()`` → reaches the env with correct shape / range.
    - Normalization stats (loaded from HFRVLA_libero_v1 by default) are
      compatible with LIBERO env's raw observations.

  Expected outcome: per-task success rate similar to published SmolVLA-base
  on libero_spatial (≈60% with significant variance on 5-episode samples).
  Anything > 0% on any task confirms the I/O is structurally sound.

Usage:
    python scripts/test_alignment.py \\
        --out-dir checkpoints/alignment_test \\
        --dataset-root checkpoints/HFRVLA_libero_v1

Optional:
    --skip-package    Skip rebuilding the alignment checkpoint (use existing)
    --task-ids 0,1,2  Comma-separated LIBERO task ids (default 0,1,2)
    --n-episodes 5    Episodes per task (default 5)
    --suite libero_spatial   LIBERO suite (default libero_spatial)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = Path.home() / "Robotic_infra/lerobot/.venv/bin/python"
VENV_EVAL = Path.home() / "Robotic_infra/lerobot/.venv/bin/lerobot-eval"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "checkpoints/alignment_test",
                   help="Directory for the alignment-test packaged policy.")
    p.add_argument("--smolvla", type=str,
                   default="/home/hucenrotia/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205",
                   help="SmolVLA base local snapshot path (or HF repo id).")
    p.add_argument("--dataset-repo-id", type=str, default="HFRVLA_libero_v1",
                   help="Source of normalization stats.")
    p.add_argument("--dataset-root", type=str,
                   default=str(REPO_ROOT / "checkpoints/HFRVLA_libero_v1"),
                   help="Local dataset root for stats loading.")
    p.add_argument("--dinov3-repo", type=str,
                   default=str(REPO_ROOT / "checkpoints/dinov3_src"))
    p.add_argument("--dinov3-weights", type=str,
                   default=str(REPO_ROOT / "checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"))
    p.add_argument("--suite", type=str, default="libero_spatial",
                   help="LIBERO suite (libero_spatial / libero_object / "
                        "libero_goal / libero_10 / libero_90).")
    p.add_argument("--task-ids", type=str, default="0,1,2",
                   help="Comma-separated task ids within the suite.")
    p.add_argument("--n-episodes", type=int, default=5,
                   help="Episodes per task.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-output-root", type=Path,
                   default=REPO_ROOT / "outputs/alignment_eval",
                   help="Root for per-task eval output dirs.")
    p.add_argument("--skip-package", action="store_true",
                   help="Reuse an existing packaged checkpoint in --out-dir.")
    return p.parse_args()


def run_package(args) -> None:
    """Package the HFRVLA alignment-test checkpoint."""
    cmd = [
        str(VENV_PY),
        str(REPO_ROOT / "scripts/package_hfrvla_checkpoint.py"),
        "--disable-fast",
        "--smolvla-pretrained", args.smolvla,
        "--out-dir", str(args.out_dir),
        "--dinov3-repo", args.dinov3_repo,
        "--dinov3-weights", args.dinov3_weights,
        "--dataset-repo-id", args.dataset_repo_id,
        "--dataset-root", args.dataset_root,
    ]
    print(f"\n[align] $ {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def run_eval_one_task(args, task_id: int, out_subdir: Path) -> dict:
    """Run lerobot-eval for one task_id; return parsed eval_info.json."""
    env_extras = {
        **os.environ,
        "NUMBA_CACHE_DIR": "/tmp/hfrvla_numba_cache",
    }
    cmd = [
        str(VENV_EVAL),
        f"--policy.path={args.out_dir}",
        "--env.type=libero",
        f"--env.task={args.suite}",
        f"--env.task_ids=[{task_id}]",
        f"--eval.n_episodes={args.n_episodes}",
        "--eval.batch_size=1",
        f"--output_dir={out_subdir}",
        f"--seed={args.seed}",
    ]
    print(f"\n[align] $ {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env_extras)
    # eval_info.json holds aggregated metrics.
    info_path = out_subdir / "eval_info.json"
    if not info_path.exists():
        # Some lerobot versions write to a different name.
        candidates = list(out_subdir.glob("*.json"))
        raise SystemExit(
            f"[align] could not find eval_info.json under {out_subdir}; "
            f"found: {candidates}"
        )
    with open(info_path) as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    args.eval_output_root.mkdir(parents=True, exist_ok=True)
    task_ids = [int(t.strip()) for t in args.task_ids.split(",") if t.strip()]

    if not args.skip_package:
        run_package(args)
    else:
        if not (args.out_dir / "config.json").exists():
            raise SystemExit(
                f"[align] --skip-package set but {args.out_dir}/config.json "
                f"does not exist; remove --skip-package or run packaging first."
            )

    print(f"\n[align] running eval on {args.suite} task_ids {task_ids} "
          f"({args.n_episodes} episodes each)\n", flush=True)

    results = []
    for tid in task_ids:
        out_sub = args.eval_output_root / f"{args.suite}_task{tid}"
        info = run_eval_one_task(args, tid, out_sub)
        aggregated = info.get("aggregated", info)
        results.append({
            "suite": args.suite,
            "task_id": tid,
            "n_episodes": args.n_episodes,
            "aggregated": aggregated,
        })

    # ── Report ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"ALIGNMENT TEST RESULTS  ({args.suite})")
    print("=" * 68)
    total_success = 0
    total_eps = 0
    for r in results:
        ag = r["aggregated"]
        succ = ag.get("pc_success") or ag.get("success_rate") or ag.get("avg_max_reward") or "?"
        sum_rew = ag.get("avg_sum_reward")
        print(f"  task {r['task_id']:>2}: n_ep={r['n_episodes']}  "
              f"success={succ}  avg_sum_reward={sum_rew}")
        if isinstance(succ, (int, float)):
            total_success += succ * r["n_episodes"]
            total_eps += r["n_episodes"]
    if total_eps:
        overall = total_success / total_eps
        print("-" * 68)
        print(f"  overall: {overall * 100:.1f}% success over {total_eps} eps")
    print("=" * 68)
    print("\nReference: published SmolVLA-base on libero_spatial ≈ 60% (full suite).")
    print("Small N (5 ep/task) gives high variance; treat 0% as a fail signal,")
    print("any nonzero per-task success as I/O alignment passing.")

    out_json = args.eval_output_root / "alignment_summary.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[align] summary written to {out_json}")


if __name__ == "__main__":
    main()
