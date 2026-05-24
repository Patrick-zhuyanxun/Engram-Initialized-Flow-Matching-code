#!/usr/bin/env python
"""Alignment test: verify HFRVLA wrapper's I/O matches LIBERO env expectations.

What it does:
  1. Run the LIBERO-adapted SmolVLA slow planner directly with ``lerobot-eval``.
  2. Build an HFRVLA package with ``inference_disable_fast=True`` so
     ``select_action()`` short-circuits to SmolVLA's ``a_base``.
  3. Run ``lerobot-eval`` against ``libero_spatial`` task_ids 0, 1, 2 with
     5 episodes each.
  4. Report per-task success rate for direct SmolVLA and zero-fast HFRVLA.

Why:
  Before training the fast module, we need to confirm:
    - HFRVLAConfig's feature overrides match what LIBERO env emits.
    - SmolVLA's action chunk pops out via the queue → returns through
      ``select_action()`` → reaches the env with correct shape / range.
    - Normalization stats (loaded from HFRVLA_libero_v1 by default) are
      compatible with LIBERO env's raw observations.

  Expected outcome: per-task success rate similar to the LIBERO-adapted
  SmolVLA slow planner used for ``--smolvla``. Anything > 0% on any task is
  a useful smoke signal that the I/O is structurally sound.

Usage:
    python scripts/test_alignment.py \\
        --out-dir checkpoints/alignment_test \\
        --dataset-root checkpoints/HFRVLA_libero_v1_merged_reindexed

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

try:
    from scripts.hfrvla_alignment_utils import (
        DEFAULT_LIBERO_SMOLVLA,
        extract_eval_metrics,
        format_success_metric,
        load_policy_config_json,
        success_metric_to_fraction,
        warn_or_validate_libero_slow_planner,
    )
except ModuleNotFoundError:
    from hfrvla_alignment_utils import (
        DEFAULT_LIBERO_SMOLVLA,
        extract_eval_metrics,
        format_success_metric,
        load_policy_config_json,
        success_metric_to_fraction,
        warn_or_validate_libero_slow_planner,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = Path.home() / "Robotic_infra/lerobot/.venv/bin/python"
VENV_EVAL = Path.home() / "Robotic_infra/lerobot/.venv/bin/lerobot-eval"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "checkpoints/alignment_test",
                   help="Directory for the alignment-test packaged policy.")
    p.add_argument("--smolvla", type=str, default=DEFAULT_LIBERO_SMOLVLA,
                   help="LIBERO-adapted SmolVLA slow-planner checkpoint. Raw "
                        "`lerobot/smolvla_base` is only a warm-start model and "
                        "will fail the default feature-contract guard. "
                        f"Default: {DEFAULT_LIBERO_SMOLVLA}.")
    p.add_argument("--dataset-repo-id", type=str, default="HFRVLA_libero_v1",
                   help="Source of normalization stats.")
    p.add_argument("--dataset-root", type=str,
                   default=str(REPO_ROOT / "checkpoints/HFRVLA_libero_v1_merged_reindexed"),
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
    p.add_argument("--skip-baseline", action="store_true",
                   help="Skip direct SmolVLA baseline eval and only run the "
                        "zero-fast HFRVLA wrapper.")
    p.add_argument("--allow-feature-remap", action="store_true",
                   help="Forward to package_hfrvla_checkpoint.py for low-level "
                        "debugging with a non-LIBERO SmolVLA config. This should "
                        "not be used for a real alignment pass.")
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
    if args.allow_feature_remap:
        cmd.append("--allow-feature-remap")
    print(f"\n[align] $ {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def run_eval_one_task(args, task_id: int, out_subdir: Path, policy_path: str | Path) -> dict:
    """Run lerobot-eval for one task_id; return parsed eval_info.json."""
    tmp_root = Path(os.environ.get("HFRVLA_TMP_ROOT", Path.home() / "tmp" / "hfrvla")).expanduser()
    hf_datasets_cache = Path(os.environ.get("HF_DATASETS_CACHE", tmp_root / "hf_datasets")).expanduser()
    tmpdir = Path(os.environ.get("TMPDIR", tmp_root / "tmp")).expanduser()
    numba_cache = Path(os.environ.get("NUMBA_CACHE_DIR", tmp_root / "numba")).expanduser()
    mpl_cache = Path(os.environ.get("MPLCONFIGDIR", tmp_root / "matplotlib")).expanduser()
    for path in (hf_datasets_cache, tmpdir, numba_cache, mpl_cache):
        path.mkdir(parents=True, exist_ok=True)
    env_extras = {
        **os.environ,
        "HF_DATASETS_CACHE": str(hf_datasets_cache),
        "TMPDIR": str(tmpdir),
        "TMP": str(tmpdir),
        "TEMP": str(tmpdir),
        "NUMBA_CACHE_DIR": str(numba_cache),
        "MPLCONFIGDIR": str(mpl_cache),
    }
    cmd = [
        str(VENV_EVAL),
        f"--policy.path={policy_path}",
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

    raw_smolvla_config = load_policy_config_json(args.smolvla)
    try:
        warn_or_validate_libero_slow_planner(
            raw_smolvla_config,
            source=args.smolvla,
            allow_feature_remap=args.allow_feature_remap,
        )
    except ValueError as exc:
        raise SystemExit(f"[align] {exc}") from exc

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

    baseline_results = []
    if not args.skip_baseline:
        print("\n[align] direct SmolVLA baseline eval\n", flush=True)
        for tid in task_ids:
            out_sub = args.eval_output_root / f"baseline_{args.suite}_task{tid}"
            info = run_eval_one_task(args, tid, out_sub, args.smolvla)
            metrics = extract_eval_metrics(info)
            baseline_results.append({
                "suite": args.suite,
                "task_id": tid,
                "n_episodes": args.n_episodes,
                "metrics": metrics,
                "eval_info": info,
            })

    wrapper_results = []
    print("\n[align] HFRVLA zero-fast wrapper eval\n", flush=True)
    for tid in task_ids:
        out_sub = args.eval_output_root / f"hfrvla_zero_fast_{args.suite}_task{tid}"
        info = run_eval_one_task(args, tid, out_sub, args.out_dir)
        metrics = extract_eval_metrics(info)
        wrapper_results.append({
            "suite": args.suite,
            "task_id": tid,
            "n_episodes": args.n_episodes,
            "metrics": metrics,
            "eval_info": info,
        })

    # ── Report ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"ALIGNMENT TEST RESULTS  ({args.suite})")
    print("=" * 68)
    total_success = 0
    total_eps = 0
    baseline_by_task = {r["task_id"]: r for r in baseline_results}
    for r in wrapper_results:
        metrics = r["metrics"]
        succ = metrics.get("pc_success")
        if succ is None:
            succ = metrics.get("success_rate")
        if succ is None:
            succ = metrics.get("avg_max_reward")
        baseline_succ = None
        if r["task_id"] in baseline_by_task:
            baseline_metrics = baseline_by_task[r["task_id"]]["metrics"]
            baseline_succ = baseline_metrics.get("pc_success")
            if baseline_succ is None:
                baseline_succ = baseline_metrics.get("success_rate")
            if baseline_succ is None:
                baseline_succ = baseline_metrics.get("avg_max_reward")
        sum_rew = metrics.get("avg_sum_reward")
        succ_text = format_success_metric(succ)
        baseline_text = "skipped" if args.skip_baseline else format_success_metric(baseline_succ)
        print(f"  task {r['task_id']:>2}: n_ep={r['n_episodes']}  "
              f"baseline={baseline_text}  zero_fast={succ_text}  "
              f"avg_sum_reward={sum_rew}")
        succ_fraction = success_metric_to_fraction(succ)
        if succ_fraction is not None:
            total_success += succ_fraction * r["n_episodes"]
            total_eps += r["n_episodes"]
    if total_eps:
        overall = total_success / total_eps
        print("-" * 68)
        print(f"  overall: {overall * 100:.1f}% success over {total_eps} eps")
    print("=" * 68)
    print("\nReference: compare against the same LIBERO-adapted SmolVLA slow planner.")
    print("Small N (5 ep/task) gives high variance. If the direct baseline is 0%,")
    print("debug the slow planner/eval setup first. If baseline works but zero-fast")
    print("HFRVLA collapses, debug wrapper I/O, normalization, and action postprocessing.")

    out_json = args.eval_output_root / "alignment_summary.json"
    with open(out_json, "w") as f:
        json.dump({
            "baseline": baseline_results,
            "hfrvla_zero_fast": wrapper_results,
        }, f, indent=2)
    print(f"\n[align] summary written to {out_json}")


if __name__ == "__main__":
    main()
