#!/usr/bin/env python3
"""Run matched SmolVLA/HFRVLA LIBERO evals across planning chunk sizes.

This sweep differs from ``run_action_steps_eval_sweep.py``:

* action-step sweep: ``chunk_size`` stays at the checkpoint value and only
  ``n_action_steps`` changes.
* chunk-size sweep: both ``chunk_size`` and ``n_action_steps`` are set to K, so
  the policy plans K future actions and executes all K before replanning.

Results are written to a separate CSV under ``outputs/chunk_size_eval_sweep``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_action_steps_eval_sweep import (  # noqa: E402
    DEFAULT_HFRVLA_POLICY,
    DEFAULT_SMOLVLA_POLICY,
    DEFAULT_TMP_ROOT,
    EvalSpec,
    ResultRow,
    build_eval_env,
    parse_csv_ints,
    parse_csv_strings,
    pending_row,
    read_action_step_metadata,
    row_from_eval_info,
    run_command,
    safe_name,
    write_csv,
)


DEFAULT_EVAL_ROOT = REPO_ROOT / "outputs/chunk_size_eval_sweep/evals"
DEFAULT_CSV = REPO_ROOT / "outputs/chunk_size_eval_sweep/results.csv"
DEFAULT_CHUNK_SIZES = (2, 4, 8, 16, 32, 50)
DEFAULT_SUITES = ("libero_spatial", "libero_object")


def output_dir_for_spec(spec: EvalSpec, eval_root: Path) -> Path:
    alpha_part = "" if spec.alpha is None else f"_alpha_{str(spec.alpha).replace('.', 'p')}"
    clip_part = (
        ""
        if spec.policy != "hfrvla" or spec.residual_clip_mode in {"", "config"}
        else f"_resclip_{safe_name(spec.residual_clip_mode)}"
    )
    chunk_size = spec.planning_chunk_size or spec.n_action_steps
    return (
        eval_root
        / f"{safe_name(spec.policy)}_chunk{chunk_size}{alpha_part}{clip_part}"
        / f"{safe_name(spec.suite)}_{spec.n_episodes_per_task}ep_seed{spec.seed}_{spec.device}"
    )


def build_eval_command(
    spec: EvalSpec,
    eval_bin: Path,
    output_dir: Path,
    *,
    task_ids: str | None = None,
) -> list[str]:
    chunk_size = spec.planning_chunk_size or spec.n_action_steps
    cmd = [
        str(eval_bin),
        f"--policy.path={spec.policy_path}",
        f"--policy.device={spec.device}",
        f"--policy.chunk_size={chunk_size}",
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
    if spec.policy == "hfrvla" and spec.alpha is not None:
        cmd.append(f"--policy.fast_residual_alpha={spec.alpha}")
        cmd.append(f"--policy.a2c2_alpha={spec.alpha}")
    if spec.policy == "hfrvla" and spec.eval_delta_max is not None:
        cmd.append(f"--policy.delta_max={spec.eval_delta_max}")
    if spec.policy == "hfrvla" and spec.eval_safety_joint_velocity_limit is not None:
        cmd.append(f"--policy.safety_joint_velocity_limit={spec.eval_safety_joint_velocity_limit}")
    if spec.policy == "hfrvla" and spec.eval_control_dt is not None:
        cmd.append(f"--policy.control_dt={spec.eval_control_dt}")
    return cmd


def build_specs(args: argparse.Namespace) -> list[EvalSpec]:
    policies = parse_csv_strings(args.policies)
    chunk_sizes = parse_csv_ints(args.chunk_sizes)
    suites = parse_csv_strings(args.suites)
    policy_paths = {
        "hfrvla": args.hfrvla_policy,
        "smolvla": args.smolvla_policy,
    }
    policy_metadata = {
        policy: read_action_step_metadata(path)
        for policy, path in policy_paths.items()
    }

    specs: list[EvalSpec] = []
    for chunk_size in chunk_sizes:
        for policy in policies:
            if policy not in policy_paths:
                raise ValueError(f"unknown policy {policy!r}; expected hfrvla or smolvla")
            metadata = policy_metadata[policy]
            residual_clip_mode = ""
            eval_delta_max = None
            eval_safety_joint_velocity_limit = None
            eval_control_dt = None
            if policy == "hfrvla":
                residual_clip_mode = args.hfrvla_residual_clip_mode
                eval_delta_max = metadata.delta_max
                eval_safety_joint_velocity_limit = metadata.safety_joint_velocity_limit
                eval_control_dt = metadata.control_dt
                if residual_clip_mode == "none":
                    eval_delta_max = args.hfrvla_no_clip_delta_max
                    eval_safety_joint_velocity_limit = 0.0
            for suite in suites:
                specs.append(
                    EvalSpec(
                        policy=policy,
                        policy_path=policy_paths[policy],
                        n_action_steps=chunk_size,
                        suite=suite,
                        alpha=args.hfrvla_alpha if policy == "hfrvla" else None,
                        seed=args.seed,
                        n_episodes_per_task=args.n_episodes,
                        device=args.device,
                        eval_batch_size=args.eval_batch_size,
                        planning_chunk_size=chunk_size,
                        policy_config_n_action_steps=metadata.policy_config_n_action_steps,
                        residual_clip_mode=residual_clip_mode,
                        eval_delta_max=eval_delta_max,
                        eval_safety_joint_velocity_limit=eval_safety_joint_velocity_limit,
                        eval_control_dt=eval_control_dt,
                    )
                )
    return specs


def run_sweep(args: argparse.Namespace) -> list[ResultRow]:
    env = build_eval_env(args.tmp_root)
    rows: list[ResultRow] = []
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

        cmd = build_eval_command(
            spec,
            args.eval_bin,
            output_dir,
            task_ids=args.task_ids,
        )
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunk-sizes", default=",".join(str(v) for v in DEFAULT_CHUNK_SIZES))
    p.add_argument("--policies", default="hfrvla,smolvla")
    p.add_argument("--suites", default=",".join(DEFAULT_SUITES))
    p.add_argument("--hfrvla-policy", type=Path, default=DEFAULT_HFRVLA_POLICY)
    p.add_argument("--smolvla-policy", type=Path, default=DEFAULT_SMOLVLA_POLICY)
    p.add_argument("--hfrvla-alpha", type=float, default=0.5)
    p.add_argument(
        "--hfrvla-residual-clip-mode",
        choices=("config", "none"),
        default="config",
        help="HFRVLA eval residual clip mode. 'none' disables the velocity clamp and uses a very large delta_max.",
    )
    p.add_argument(
        "--hfrvla-no-clip-delta-max",
        type=float,
        default=999.0,
        help="delta_max override used when --hfrvla-residual-clip-mode=none.",
    )
    p.add_argument("--n-episodes", type=int, default=5, help="Episodes per task.")
    p.add_argument("--eval-batch-size", type=int, default=1, help="Parallel eval envs per task.")
    p.add_argument("--task-ids", default=None, help="Optional LIBERO task id list, e.g. '[0]'.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument(
        "--eval-bin",
        type=Path,
        default=Path.home() / "Robotic_infra/lerobot/.venv/bin/lerobot-eval",
    )
    p.add_argument("--tmp-root", type=Path, default=DEFAULT_TMP_ROOT)
    p.add_argument("--force", action="store_true", help="Rerun even if eval_info.json exists.")
    p.add_argument("--keep-going", action="store_true", help="Write failed rows and continue.")
    p.add_argument("--dry-run", action="store_true", help="Print commands and write dry-run rows.")
    p.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect existing eval_info.json files and mark missing rows pending.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_sweep(args)
    print(f"[chunk-size-sweep] wrote {args.csv}")
    completed = sum(1 for row in rows if row.status in {"ok", "cached", "imported"})
    print(f"[chunk-size-sweep] completed suite rows: {completed}/{len(rows)}")


if __name__ == "__main__":
    main()
