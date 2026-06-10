#!/usr/bin/env python3
"""Run the HFRVLA async-timestep planner-delay eval.

The slow planner observes at control step ``t``, ``A_t`` arrives at ``t+d``,
the active queue is replaced immediately, and execution starts from ``A_t[d]``.
The sweep compares HFRVLA against the same wrapper with the fast residual
disabled.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_action_steps_eval_sweep import (  # noqa: E402
    DEFAULT_HFRVLA_POLICY,
    DEFAULT_PLANNING_CHUNK_SIZE,
    DEFAULT_TMP_ROOT,
    build_eval_env,
    format_optional_float,
    parse_csv_ints,
    parse_csv_strings,
    read_action_step_metadata,
    safe_name,
    summarize_eval_info,
)


DEFAULT_EVAL_ROOT = REPO_ROOT / "outputs/async_timestep_planner_delay_eval_sweep/evals"
DEFAULT_CSV = REPO_ROOT / "outputs/async_timestep_planner_delay_eval_sweep/results.csv"
DEFAULT_DELAYS = tuple(range(5))
DEFAULT_POLICIES = ("hfrvla", "hfrvla_disable_fast")
DEFAULT_SUITES = ("libero_spatial",)


@dataclass(frozen=True)
class DelayEvalSpec:
    policy: str
    policy_path: Path
    planner_delay_steps: int
    n_action_steps: int
    planning_chunk_size: int | None
    suite: str
    alpha: float | None
    seed: int
    n_episodes_per_task: int
    device: str
    fallback: str = "hold_last"
    planner_delay_mode: str = "async_timestep"
    async_request_interval_steps: int = 8
    eval_batch_size: int = 1
    policy_config_n_action_steps: int | None = None
    eval_delta_max: float | None = None
    eval_safety_joint_velocity_limit: float | None = None
    eval_control_dt: float | None = None


@dataclass
class DelayResultRow:
    policy: str
    planner_delay_mode: str
    planner_delay_steps: int
    planner_delay_fallback: str
    async_request_interval_steps: int
    n_action_steps: int
    planning_chunk_size: int | None
    execution_chunk_size: int
    replan_interval_steps: int
    policy_config_n_action_steps: int | None
    suite: str
    alpha: str
    seed: int
    n_episodes_per_task: int
    n_episodes: int | None
    n_successes: int | None
    pc_success: float | None
    avg_sum_reward: float | None
    avg_max_reward: float | None
    eval_s: float | None
    per_task_successes: str
    fallback_steps_total: str
    fallback_steps_mean: str
    slow_replan_count: str
    slow_chunk_latency_ms_mean: str
    fast_latency_ms_mean: str
    fast_applied_ratio: str
    delta_norm_mean: str
    delta_clip_fraction_mean: str
    k_mean: str
    async_request_count: str
    async_activation_count: str
    async_chunk_start_index_last: str
    async_chunk_start_index_mean: str
    async_dropped_old_queue_steps_last: str
    async_dropped_old_queue_steps_mean: str
    status: str
    output_dir: str
    eval_info_path: str
    updated_at: str


CSV_FIELDS = list(DelayResultRow.__dataclass_fields__)


def output_dir_for_spec(spec: DelayEvalSpec, eval_root: Path) -> Path:
    alpha_part = "" if spec.alpha is None else f"_alpha_{str(spec.alpha).replace('.', 'p')}"
    plan = spec.planning_chunk_size or DEFAULT_PLANNING_CHUNK_SIZE
    mode_part = f"_async_timestep_N{spec.async_request_interval_steps}"
    return (
        eval_root
        / (
            f"{safe_name(spec.policy)}_plan{plan}_exec{spec.n_action_steps}"
            f"_delay{spec.planner_delay_steps}{mode_part}{alpha_part}"
        )
        / f"{safe_name(spec.suite)}_{spec.n_episodes_per_task}ep_seed{spec.seed}_{spec.device}"
    )


def env_overrides_for_spec(spec: DelayEvalSpec) -> dict[str, str]:
    del spec
    return {}


def build_eval_command(
    spec: DelayEvalSpec,
    eval_bin: Path,
    output_dir: Path,
    *,
    task_ids: str | None = None,
) -> list[str]:
    cmd = [
        str(eval_bin),
        f"--policy.path={spec.policy_path}",
        f"--policy.device={spec.device}",
        f"--policy.n_action_steps={spec.n_action_steps}",
        "--env.type=libero",
        f"--env.task={spec.suite}",
        f"--eval.n_episodes={spec.n_episodes_per_task}",
        f"--eval.batch_size={spec.eval_batch_size}",
        f"--output_dir={output_dir}",
        f"--seed={spec.seed}",
    ]
    if task_ids:
        cmd.insert(-3, f"--env.task_ids={task_ids}")
    if spec.planning_chunk_size is not None:
        cmd.append(f"--policy.chunk_size={spec.planning_chunk_size}")
    cmd.append("--policy.planner_delay_mode=async_timestep")
    cmd.append(f"--policy.planner_delay_steps={spec.planner_delay_steps}")
    cmd.append(f"--policy.planner_delay_fallback={spec.fallback}")
    cmd.append(f"--policy.async_request_interval_steps={spec.async_request_interval_steps}")
    if spec.alpha is not None:
        cmd.append(f"--policy.fast_residual_alpha={spec.alpha}")
    if spec.eval_delta_max is not None:
        cmd.append(f"--policy.delta_max={spec.eval_delta_max}")
    if spec.eval_safety_joint_velocity_limit is not None:
        cmd.append(f"--policy.safety_joint_velocity_limit={spec.eval_safety_joint_velocity_limit}")
    if spec.eval_control_dt is not None:
        cmd.append(f"--policy.control_dt={spec.eval_control_dt}")
    if spec.policy == "hfrvla_disable_fast":
        cmd.append("--policy.inference_disable_fast=true")
    return cmd


def _debug_value(debug: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key in debug:
            return str(debug[key])
    return ""


def row_from_eval_info(spec: DelayEvalSpec, info_path: Path, *, status: str) -> DelayResultRow:
    info = json.loads(info_path.read_text())
    (
        n_episodes,
        n_successes,
        pc_success,
        avg_sum_reward,
        avg_max_reward,
        eval_s,
        per_task_successes,
    ) = summarize_eval_info(info)
    debug = info.get("policy_inference_debug", {})
    fallback_total = _debug_value(debug, "fallback_steps")
    fallback_mean = ""
    if fallback_total and n_episodes:
        fallback_mean = str(round(float(fallback_total) / float(n_episodes), 4))
    return DelayResultRow(
        policy=spec.policy,
        planner_delay_mode=spec.planner_delay_mode,
        planner_delay_steps=spec.planner_delay_steps,
        planner_delay_fallback=spec.fallback,
        async_request_interval_steps=spec.async_request_interval_steps,
        n_action_steps=spec.n_action_steps,
        planning_chunk_size=spec.planning_chunk_size,
        execution_chunk_size=spec.n_action_steps,
        replan_interval_steps=spec.n_action_steps,
        policy_config_n_action_steps=spec.policy_config_n_action_steps,
        suite=spec.suite,
        alpha="" if spec.alpha is None else str(spec.alpha),
        seed=spec.seed,
        n_episodes_per_task=spec.n_episodes_per_task,
        n_episodes=n_episodes,
        n_successes=n_successes,
        pc_success=round(pc_success, 4),
        avg_sum_reward=round(avg_sum_reward, 6),
        avg_max_reward=round(avg_max_reward, 6),
        eval_s=round(eval_s, 3),
        per_task_successes=",".join(str(v) for v in per_task_successes),
        fallback_steps_total=fallback_total,
        fallback_steps_mean=fallback_mean,
        slow_replan_count=_debug_value(debug, "slow_replan_count", "new_chunks"),
        slow_chunk_latency_ms_mean=_debug_value(debug, "slow_chunk_latency_ms_mean"),
        fast_latency_ms_mean=_debug_value(debug, "fast_latency_ms_mean"),
        fast_applied_ratio=_debug_value(debug, "fast_applied_ratio"),
        delta_norm_mean=_debug_value(debug, "delta_norm_mean"),
        delta_clip_fraction_mean=_debug_value(debug, "delta_clip_fraction_mean"),
        k_mean=_debug_value(debug, "k_mean"),
        async_request_count=_debug_value(debug, "async_request_count"),
        async_activation_count=_debug_value(debug, "async_activation_count"),
        async_chunk_start_index_last=_debug_value(debug, "async_chunk_start_index_last"),
        async_chunk_start_index_mean=_debug_value(debug, "async_chunk_start_index_mean"),
        async_dropped_old_queue_steps_last=_debug_value(debug, "async_dropped_old_queue_steps_last"),
        async_dropped_old_queue_steps_mean=_debug_value(debug, "async_dropped_old_queue_steps_mean"),
        status=status,
        output_dir=str(info_path.parent),
        eval_info_path=str(info_path),
        updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def pending_row(spec: DelayEvalSpec, info_path: Path, status: str) -> DelayResultRow:
    return DelayResultRow(
        policy=spec.policy,
        planner_delay_mode=spec.planner_delay_mode,
        planner_delay_steps=spec.planner_delay_steps,
        planner_delay_fallback=spec.fallback,
        async_request_interval_steps=spec.async_request_interval_steps,
        n_action_steps=spec.n_action_steps,
        planning_chunk_size=spec.planning_chunk_size,
        execution_chunk_size=spec.n_action_steps,
        replan_interval_steps=spec.n_action_steps,
        policy_config_n_action_steps=spec.policy_config_n_action_steps,
        suite=spec.suite,
        alpha="" if spec.alpha is None else str(spec.alpha),
        seed=spec.seed,
        n_episodes_per_task=spec.n_episodes_per_task,
        n_episodes=None,
        n_successes=None,
        pc_success=None,
        avg_sum_reward=None,
        avg_max_reward=None,
        eval_s=None,
        per_task_successes="",
        fallback_steps_total="",
        fallback_steps_mean="",
        slow_replan_count="",
        slow_chunk_latency_ms_mean="",
        fast_latency_ms_mean="",
        fast_applied_ratio="",
        delta_norm_mean="",
        delta_clip_fraction_mean="",
        k_mean="",
        async_request_count="",
        async_activation_count="",
        async_chunk_start_index_last="",
        async_chunk_start_index_mean="",
        async_dropped_old_queue_steps_last="",
        async_dropped_old_queue_steps_mean="",
        status=status,
        output_dir=str(info_path.parent),
        eval_info_path=str(info_path),
        updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def write_csv(csv_path: Path, rows: list[DelayResultRow]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        rows,
        key=lambda row: (
            row.planner_delay_steps,
            row.policy,
            row.suite,
            row.n_action_steps,
        ),
    )
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def build_specs(args: argparse.Namespace) -> list[DelayEvalSpec]:
    policies = parse_csv_strings(args.policies)
    delays = parse_csv_ints(args.planner_delay_steps)
    suites = parse_csv_strings(args.suites)
    if args.async_request_interval_steps <= 0:
        raise ValueError("async_request_interval_steps must be > 0")
    policy_paths = {
        "hfrvla": args.hfrvla_policy,
        "hfrvla_disable_fast": args.hfrvla_policy,
    }
    metadata = {
        key: read_action_step_metadata(path)
        for key, path in policy_paths.items()
    }
    specs: list[DelayEvalSpec] = []
    for delay in delays:
        if args.async_request_interval_steps + delay > args.n_action_steps:
            raise ValueError(
                "async_request_interval_steps + planner_delay_steps must be <= n_action_steps"
            )
        for policy in policies:
            if policy not in policy_paths:
                raise ValueError(
                    f"unknown policy {policy!r}; expected hfrvla or hfrvla_disable_fast"
                )
            policy_metadata = metadata[policy]
            eval_delta_max = args.hfrvla_delta_max
            eval_control_dt = args.hfrvla_control_dt
            eval_safety_joint_velocity_limit = (
                round(args.hfrvla_delta_max / args.hfrvla_control_dt, 10)
                if args.hfrvla_control_dt > 0
                else 0.0
            )
            for suite in suites:
                specs.append(
                    DelayEvalSpec(
                        policy=policy,
                        policy_path=policy_paths[policy],
                        planner_delay_steps=delay,
                        n_action_steps=args.n_action_steps,
                        planning_chunk_size=args.planning_chunk_size,
                        suite=suite,
                        alpha=args.hfrvla_alpha,
                        seed=args.seed,
                        n_episodes_per_task=args.n_episodes,
                        device=args.device,
                        fallback=args.fallback,
                        planner_delay_mode="async_timestep",
                        async_request_interval_steps=args.async_request_interval_steps,
                        eval_batch_size=args.eval_batch_size,
                        policy_config_n_action_steps=policy_metadata.policy_config_n_action_steps,
                        eval_delta_max=eval_delta_max,
                        eval_safety_joint_velocity_limit=eval_safety_joint_velocity_limit,
                        eval_control_dt=eval_control_dt,
                    )
                )
    return specs


def run_command(
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
        raise RuntimeError(f"eval failed with exit code {proc.returncode}: {log_path}")


def run_sweep(args: argparse.Namespace) -> list[DelayResultRow]:
    base_env = build_eval_env(args.tmp_root)
    rows: list[DelayResultRow] = []
    for spec in build_specs(args):
        output_dir = output_dir_for_spec(spec, args.eval_root)
        info_path = output_dir / "eval_info.json"
        if info_path.exists() and not args.force:
            rows.append(row_from_eval_info(spec, info_path, status="cached"))
            write_csv(args.csv, rows)
            continue

        if args.collect_only:
            rows.append(pending_row(spec, info_path, "pending"))
            write_csv(args.csv, rows)
            continue

        cmd = build_eval_command(spec, args.eval_bin, output_dir, task_ids=args.task_ids)
        env = dict(base_env)
        env.update(env_overrides_for_spec(spec))
        try:
            run_command(
                cmd,
                env=env,
                log_path=output_dir / "eval.log",
                dry_run=args.dry_run,
            )
            if args.dry_run:
                rows.append(pending_row(spec, info_path, "dry-run"))
            else:
                rows.append(row_from_eval_info(spec, info_path, status="ok"))
        except Exception as exc:
            if not args.keep_going:
                raise
            row = pending_row(spec, info_path, "failed")
            row.per_task_successes = str(exc)
            rows.append(row)
        write_csv(args.csv, rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-delay-steps", default=",".join(str(v) for v in DEFAULT_DELAYS))
    parser.add_argument("--async-request-interval-steps", type=int, default=8)
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--suites", default=",".join(DEFAULT_SUITES))
    parser.add_argument("--hfrvla-policy", type=Path, default=DEFAULT_HFRVLA_POLICY)
    parser.add_argument("--planning-chunk-size", type=int, default=DEFAULT_PLANNING_CHUNK_SIZE)
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--fallback", choices=("hold_last", "zero"), default="hold_last")
    parser.add_argument("--hfrvla-alpha", type=float, default=0.5)
    parser.add_argument("--hfrvla-delta-max", type=float, default=0.2)
    parser.add_argument("--hfrvla-control-dt", type=float, default=0.1)
    parser.add_argument("--n-episodes", type=int, default=10, help="Episodes per task.")
    parser.add_argument("--eval-batch-size", type=int, default=3, help="Parallel eval envs per task.")
    parser.add_argument("--task-ids", default=None, help="Optional LIBERO task id list, e.g. '[0]'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--eval-bin", type=Path, default=REPO_ROOT / "scripts/lerobot_eval_hfrvla.py")
    parser.add_argument("--tmp-root", type=Path, default=DEFAULT_TMP_ROOT)
    parser.add_argument("--force", action="store_true", help="Rerun even if eval_info.json exists.")
    parser.add_argument("--keep-going", action="store_true", help="Write failed rows and continue.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write dry-run rows.")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect existing eval_info.json files and mark missing rows pending.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_sweep(args)
    print(f"[planner-delay-sweep] wrote {args.csv}")
    completed = sum(1 for row in rows if row.status in {"ok", "cached"})
    print(f"[planner-delay-sweep] completed suite rows: {completed}/{len(rows)}")


if __name__ == "__main__":
    main()
