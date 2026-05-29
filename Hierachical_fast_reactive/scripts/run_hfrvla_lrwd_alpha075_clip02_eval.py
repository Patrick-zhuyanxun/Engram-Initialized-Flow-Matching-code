#!/usr/bin/env python3
"""Evaluate high-LR HFRVLA LR/WD sweep checkpoints at alpha=0.75, clip=0.2.

The eval protocol is fixed to the current matched n=8 comparison:

    planning_chunk_size=50, execution_chunk_size=8, replan_interval_steps=8

Each raw training checkpoint is packaged before eval if needed. Results are
written as a compact CSV plus a markdown summary and can be resumed from
existing eval_info.json files unless --force is used.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_TMP_ROOT = Path.home() / "tmp/hfrvla"
DEFAULT_EVAL_ROOT = REPO_ROOT / "outputs/hfrvla_lrwd_alpha075_clip02_eval/evals"
DEFAULT_CSV = REPO_ROOT / "outputs/hfrvla_lrwd_alpha075_clip02_eval/results.csv"
DEFAULT_SUMMARY = REPO_ROOT / "outputs/hfrvla_lrwd_alpha075_clip02_eval/summary.md"
DEFAULT_PACKAGE_PYTHON = Path.home() / "Robotic_infra/lerobot/.venv/bin/python"
DEFAULT_EVAL_BIN = Path.home() / "Robotic_infra/lerobot/.venv/bin/lerobot-eval"

DEFAULT_REFERENCE_ROWS = (
    ("smolvla_plan50_exec8_baseline", "libero_spatial", 32, 50, 64.0),
    ("smolvla_plan50_exec8_baseline", "libero_object", 45, 50, 90.0),
    ("smolvla_plan50_exec8_baseline", "combined", 77, 100, 77.0),
)


@dataclass(frozen=True)
class Candidate:
    run_name: str
    lr: str
    weight_decay: str
    train_steps: str = "50000"
    seq_len: str = "2"
    batch_size: str = "512"


DEFAULT_CANDIDATES = (
    Candidate("hfrvla_lr3e_4_wd0_b512_50k", "3e-4", "0"),
    Candidate("hfrvla_lr3e_4_wd1e_5_b512_50k", "3e-4", "1e-5"),
    Candidate("hfrvla_lr3e_4_wd1e_4_b512_50k", "3e-4", "1e-4"),
    Candidate("hfrvla_lr3e_4_wd3e_4_b512_50k", "3e-4", "3e-4"),
    Candidate("hfrvla_lr5e_4_wd0_b512_50k", "5e-4", "0"),
    Candidate("hfrvla_lr5e_4_wd1e_5_b512_50k", "5e-4", "1e-5"),
    Candidate("hfrvla_lr5e_4_wd1e_4_b512_50k", "5e-4", "1e-4"),
    Candidate("hfrvla_lr5e_4_wd3e_4_b512_50k", "5e-4", "3e-4"),
)


from scripts.run_action_steps_eval_sweep import (  # noqa: E402
    EvalSpec,
    build_eval_command,
    build_eval_env,
    combined_rows,
    parse_csv_strings,
    read_action_step_metadata,
    row_from_eval_info,
)


CSV_FIELDS = [
    "policy",
    "run_name",
    "checkpoint_id",
    "train_steps",
    "seq_len",
    "batch_size",
    "lr",
    "weight_decay",
    "suite",
    "alpha",
    "delta_max",
    "safety_joint_velocity_limit",
    "control_dt",
    "residual_clip_mode",
    "eval_delta_max",
    "eval_safety_joint_velocity_limit",
    "eval_control_dt",
    "n_action_steps",
    "planning_chunk_size",
    "execution_chunk_size",
    "replan_interval_steps",
    "policy_config_n_action_steps",
    "seed",
    "n_episodes_per_task",
    "n_episodes",
    "n_successes",
    "pc_success",
    "per_task_successes",
    "status",
    "output_dir",
    "eval_info_path",
    "updated_at",
]


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(".", "p").replace(",", "-").replace(" ", "")


def raw_fast_ckpt_dir(candidate: Candidate, args: argparse.Namespace) -> Path:
    return args.repo_root / "checkpoints" / candidate.run_name / "checkpoints" / "last" / "pretrained_model"


def packaged_policy_dir(candidate: Candidate, args: argparse.Namespace) -> Path:
    return args.repo_root / "checkpoints" / f"{candidate.run_name}_packaged"


def package_is_ready(policy_dir: Path) -> bool:
    return (policy_dir / "config.json").exists() and (policy_dir / "model.safetensors").exists()


def build_package_command(candidate: Candidate, args: argparse.Namespace) -> list[str]:
    return [
        str(args.package_python),
        "scripts/package_hfrvla_checkpoint.py",
        "--fast-ckpt",
        str(raw_fast_ckpt_dir(candidate, args)),
        "--out-dir",
        str(packaged_policy_dir(candidate, args)),
        "--dinov3-repo",
        str(args.dinov3_repo),
        "--dinov3-weights",
        str(args.dinov3_weights),
        f"--dataset-repo-id={args.dataset_repo_id}",
        "--dataset-root",
        str(args.dataset_root),
    ]


def build_eval_spec(
    candidate: Candidate,
    suite: str,
    packaged_policy: Path,
    args: argparse.Namespace,
) -> EvalSpec:
    metadata = read_action_step_metadata(packaged_policy)
    safety_limit = round(args.delta_max / args.control_dt, 10) if args.control_dt > 0 else 0.0
    return EvalSpec(
        policy="hfrvla",
        policy_path=packaged_policy,
        n_action_steps=args.n_action_steps,
        suite=suite,
        alpha=args.alpha,
        seed=args.seed,
        n_episodes_per_task=args.n_episodes,
        device=args.device,
        planning_chunk_size=metadata.planning_chunk_size,
        policy_config_n_action_steps=metadata.policy_config_n_action_steps,
        residual_clip_mode="config",
        eval_delta_max=args.delta_max,
        eval_safety_joint_velocity_limit=safety_limit,
        eval_control_dt=args.control_dt,
    )


def output_dir_for_candidate(candidate: Candidate, suite: str, args: argparse.Namespace) -> Path:
    return (
        args.eval_root
        / candidate.run_name
        / f"{safe_name(suite)}_{args.n_episodes}ep_seed{args.seed}_{safe_name(args.device)}"
    )


def eval_info_path(candidate: Candidate, suite: str, args: argparse.Namespace) -> Path:
    return output_dir_for_candidate(candidate, suite, args) / "eval_info.json"


def run_logged_command(
    cmd: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {log_path}")


def row_for_eval_info(
    candidate: Candidate,
    spec: EvalSpec,
    info_path: Path,
    *,
    status: str,
) -> dict[str, Any]:
    row = row_from_eval_info(spec, info_path, status=status)
    return {
        "policy": "hfrvla",
        "run_name": candidate.run_name,
        "checkpoint_id": candidate.run_name,
        "train_steps": candidate.train_steps,
        "seq_len": candidate.seq_len,
        "batch_size": candidate.batch_size,
        "lr": candidate.lr,
        "weight_decay": candidate.weight_decay,
        "suite": row.suite,
        "alpha": row.alpha,
        "delta_max": row.eval_delta_max,
        "safety_joint_velocity_limit": row.eval_safety_joint_velocity_limit,
        "control_dt": row.eval_control_dt,
        "residual_clip_mode": row.residual_clip_mode,
        "eval_delta_max": row.eval_delta_max,
        "eval_safety_joint_velocity_limit": row.eval_safety_joint_velocity_limit,
        "eval_control_dt": row.eval_control_dt,
        "n_action_steps": row.n_action_steps,
        "planning_chunk_size": row.planning_chunk_size,
        "execution_chunk_size": row.execution_chunk_size,
        "replan_interval_steps": row.replan_interval_steps,
        "policy_config_n_action_steps": row.policy_config_n_action_steps,
        "seed": row.seed,
        "n_episodes_per_task": row.n_episodes_per_task,
        "n_episodes": row.n_episodes,
        "n_successes": row.n_successes,
        "pc_success": row.pc_success,
        "per_task_successes": row.per_task_successes,
        "status": status,
        "output_dir": row.output_dir,
        "eval_info_path": row.eval_info_path,
        "updated_at": row.updated_at,
    }


def pending_row(
    candidate: Candidate,
    suite: str,
    args: argparse.Namespace,
    *,
    status: str,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "policy": "hfrvla",
        "run_name": candidate.run_name,
        "checkpoint_id": candidate.run_name,
        "train_steps": candidate.train_steps,
        "seq_len": candidate.seq_len,
        "batch_size": candidate.batch_size,
        "lr": candidate.lr,
        "weight_decay": candidate.weight_decay,
        "suite": suite,
        "alpha": args.alpha,
        "delta_max": args.delta_max,
        "safety_joint_velocity_limit": round(args.delta_max / args.control_dt, 10),
        "control_dt": args.control_dt,
        "residual_clip_mode": "config",
        "eval_delta_max": args.delta_max,
        "eval_safety_joint_velocity_limit": round(args.delta_max / args.control_dt, 10),
        "eval_control_dt": args.control_dt,
        "n_action_steps": args.n_action_steps,
        "planning_chunk_size": 50,
        "execution_chunk_size": args.n_action_steps,
        "replan_interval_steps": args.n_action_steps,
        "policy_config_n_action_steps": "",
        "seed": args.seed,
        "n_episodes_per_task": args.n_episodes,
        "n_episodes": "",
        "n_successes": "",
        "pc_success": "",
        "per_task_successes": detail,
        "status": status,
        "output_dir": str(output_dir_for_candidate(candidate, suite, args)),
        "eval_info_path": str(eval_info_path(candidate, suite, args)),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def result_sort_key(row: dict[str, Any]) -> tuple[int, str, int]:
    suite_order = {"libero_spatial": 0, "libero_object": 1, "combined": 2}
    return (suite_order.get(str(row["suite"]), 99), str(row["lr"]), int(str(row["weight_decay"]) == "0"))


def combined_candidate_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for candidate in DEFAULT_CANDIDATES:
        selected = [
            row
            for row in rows
            if row["run_name"] == candidate.run_name
            and row["suite"] in {"libero_spatial", "libero_object"}
            and row["status"] in {"ok", "cached"}
        ]
        if {row["suite"] for row in selected} != {"libero_spatial", "libero_object"}:
            continue
        n_episodes = sum(int(row["n_episodes"]) for row in selected)
        n_successes = sum(int(row["n_successes"]) for row in selected)
        out.append(
            {
                "policy": "hfrvla",
                "run_name": candidate.run_name,
                "checkpoint_id": candidate.run_name,
                "train_steps": candidate.train_steps,
                "seq_len": candidate.seq_len,
                "batch_size": candidate.batch_size,
                "lr": candidate.lr,
                "weight_decay": candidate.weight_decay,
                "suite": "combined",
                "alpha": args.alpha,
                "delta_max": args.delta_max,
                "safety_joint_velocity_limit": round(args.delta_max / args.control_dt, 10),
                "control_dt": args.control_dt,
                "residual_clip_mode": "config",
                "eval_delta_max": args.delta_max,
                "eval_safety_joint_velocity_limit": round(args.delta_max / args.control_dt, 10),
                "eval_control_dt": args.control_dt,
                "n_action_steps": args.n_action_steps,
                "planning_chunk_size": selected[0].get("planning_chunk_size", 50),
                "execution_chunk_size": args.n_action_steps,
                "replan_interval_steps": args.n_action_steps,
                "policy_config_n_action_steps": selected[0].get("policy_config_n_action_steps", ""),
                "seed": args.seed,
                "n_episodes_per_task": args.n_episodes,
                "n_episodes": n_episodes,
                "n_successes": n_successes,
                "pc_success": round(100.0 * n_successes / n_episodes, 4) if n_episodes else 0.0,
                "per_task_successes": "",
                "status": "derived",
                "output_dir": "",
                "eval_info_path": "",
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
    return out


def write_csv(csv_path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = rows + combined_candidate_rows(rows, args)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(all_rows, key=result_sort_key))


def write_summary(summary_path: Path, csv_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    rows = csv_rows + combined_candidate_rows(csv_rows, args)
    planning_chunk_size = next(
        (row.get("planning_chunk_size") for row in rows if row.get("planning_chunk_size")),
        50,
    )
    spatial = [
        row
        for row in rows
        if row["suite"] == "libero_spatial" and row["status"] in {"ok", "cached"}
    ]
    spatial.sort(key=lambda row: float(row["pc_success"]), reverse=True)

    lines = [
        "# HFRVLA LR/WD alpha=0.75 clip=0.2 eval summary",
        "",
        f"- alpha: `{args.alpha}`",
        f"- delta_max: `{args.delta_max}`",
        f"- safety_joint_velocity_limit: `{round(args.delta_max / args.control_dt, 10)}`",
        f"- control_dt: `{args.control_dt}`",
        f"- n_action_steps: `{args.n_action_steps}`",
        f"- planning_chunk_size: `{planning_chunk_size}`",
        f"- execution_chunk_size: `{args.n_action_steps}`",
        f"- replan_interval_steps: `{args.n_action_steps}`",
        f"- seed: `{args.seed}`",
        "",
        "## Reference",
        "",
        "| reference | suite | success |",
        "|---|---|---:|",
    ]
    for name, suite, n_successes, n_episodes, pc_success in DEFAULT_REFERENCE_ROWS:
        lines.append(f"| {name} | {suite} | {n_successes}/{n_episodes} = {pc_success:.1f}% |")

    lines.extend(["", "## Spatial Ranking", "", "| rank | run | lr | wd | spatial success |", "|---:|---|---:|---:|---:|"])
    for idx, row in enumerate(spatial, start=1):
        lines.append(
            f"| {idx} | {row['run_name']} | {row['lr']} | {row['weight_decay']} | "
            f"{row['n_successes']}/{row['n_episodes']} = {float(row['pc_success']):.1f}% |"
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--delta-max", type=float, default=0.2)
    p.add_argument("--control-dt", type=float, default=0.1)
    p.add_argument("--n-action-steps", type=int, default=8)
    p.add_argument("--suites", default="libero_spatial,libero_object")
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--eval-bin", type=Path, default=DEFAULT_EVAL_BIN)
    p.add_argument("--package-python", type=Path, default=DEFAULT_PACKAGE_PYTHON)
    p.add_argument("--tmp-root", type=Path, default=DEFAULT_TMP_ROOT)
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    p.add_argument("--dinov3-repo", type=Path, default=Path("checkpoints/dinov3_src"))
    p.add_argument(
        "--dinov3-weights",
        type=Path,
        default=Path("checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
    )
    p.add_argument("--dataset-repo-id", default="HFRVLA_libero_v1")
    p.add_argument("--dataset-root", type=Path, default=Path("checkpoints/HFRVLA_libero_v1_merged_reindexed"))
    p.add_argument("--force", action="store_true", help="Repackage and rerun even if cached outputs exist.")
    p.add_argument("--keep-going", action="store_true", help="Record failed rows and continue.")
    p.add_argument("--dry-run", action="store_true", help="Print package/eval commands without running or writing CSV.")
    p.add_argument("--collect-only", action="store_true", help="Collect existing eval_info.json files and mark missing rows.")
    return p.parse_args()


def run_eval_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    suites = parse_csv_strings(args.suites)
    env = build_eval_env(args.tmp_root)
    rows: list[dict[str, Any]] = []
    for candidate in DEFAULT_CANDIDATES:
        package_dir = packaged_policy_dir(candidate, args)
        if not package_is_ready(package_dir) or args.force:
            package_cmd = build_package_command(candidate, args)
            try:
                run_logged_command(
                    package_cmd,
                    env=env,
                    log_path=package_dir / "package.log",
                    dry_run=args.dry_run or args.collect_only,
                )
            except Exception as exc:
                if not args.keep_going:
                    raise
                for suite in suites:
                    rows.append(pending_row(candidate, suite, args, status="package_failed", detail=str(exc)))
                continue

        for suite in suites:
            info_path = eval_info_path(candidate, suite, args)
            package_path = packaged_policy_dir(candidate, args)
            spec = build_eval_spec(candidate, suite, package_path, args)
            if info_path.exists() and not args.force:
                rows.append(row_for_eval_info(candidate, spec, info_path, status="cached"))
                continue
            if args.collect_only:
                rows.append(pending_row(candidate, suite, args, status="pending"))
                continue
            output_dir = output_dir_for_candidate(candidate, suite, args)
            cmd = build_eval_command(spec, args.eval_bin, output_dir)
            try:
                run_logged_command(
                    cmd,
                    env=env,
                    log_path=output_dir / "eval.log",
                    dry_run=args.dry_run,
                )
                if args.dry_run:
                    continue
                rows.append(row_for_eval_info(candidate, spec, info_path, status="ok"))
            except Exception as exc:
                if not args.keep_going:
                    raise
                rows.append(pending_row(candidate, suite, args, status="failed", detail=str(exc)))

    if not args.dry_run:
        write_csv(args.csv, rows, args)
        write_summary(args.summary, rows, args)
    return rows


def main() -> None:
    args = parse_args()
    rows = run_eval_plan(args)
    print(f"[hfrvla-alpha075-clip02-eval] candidate_runs={len(DEFAULT_CANDIDATES)}")
    if args.dry_run:
        print("[hfrvla-alpha075-clip02-eval] dry-run only; no CSV written")
        return
    completed = sum(1 for row in rows if row["status"] in {"ok", "cached"})
    print(f"[hfrvla-alpha075-clip02-eval] completed suite rows: {completed}/{len(rows)}")
    print(f"[hfrvla-alpha075-clip02-eval] wrote {args.csv}")
    print(f"[hfrvla-alpha075-clip02-eval] wrote {args.summary}")


if __name__ == "__main__":
    main()
