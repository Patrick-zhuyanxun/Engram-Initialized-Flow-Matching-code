#!/usr/bin/env python3
"""Run matched SmolVLA/HFRVLA LIBERO evals across action chunk lengths.

The default sweep compares both policies at n_action_steps in {2, 4, 8, 16, 32}
on LIBERO spatial and object. Results are stored in a single CSV and can be
resumed: completed eval_info.json files are reused unless --force is passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TMP_ROOT = Path.home() / "tmp/hfrvla"
DEFAULT_EVAL_ROOT = REPO_ROOT / "outputs/action_steps_eval_sweep/evals"
DEFAULT_CSV = REPO_ROOT / "outputs/action_steps_eval_sweep/results.csv"
DEFAULT_HFRVLA_POLICY = REPO_ROOT / "checkpoints/hfrvla_a2c2_wrist_seq2_30k_packaged"
DEFAULT_SMOLVLA_POLICY = (
    Path.home()
    / ".cache/huggingface/hub/models--HuggingFaceVLA--smolvla_libero/"
    / "snapshots/6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
)
DEFAULT_STEPS = (2, 4, 8, 16, 32)
DEFAULT_SUITES = ("libero_spatial", "libero_object")
DEFAULT_PLANNING_CHUNK_SIZE = 50

# Previous runs used hand-written output directories. Keeping these aliases lets
# the CSV ingest the already completed n=8 comparison without rerunning it.
KNOWN_OUTPUT_ALIASES: dict[tuple[str, int, str], Path] = {
    (
        "hfrvla",
        8,
        "libero_spatial",
    ): REPO_ROOT / "outputs/eval_hfrvla_a2c2_wrist_seq2_30k_n8_alpha_05_spatial_cuda",
    (
        "hfrvla",
        8,
        "libero_object",
    ): REPO_ROOT / "outputs/eval_hfrvla_a2c2_wrist_seq2_30k_n8_alpha_05_object_cuda",
    (
        "smolvla",
        8,
        "libero_spatial",
    ): REPO_ROOT / "outputs/eval_smolvla_libero_n8_spatial_cuda",
    (
        "smolvla",
        8,
        "libero_object",
    ): REPO_ROOT / "outputs/eval_smolvla_libero_n8_object_cuda",
}


@dataclass(frozen=True)
class EvalSpec:
    policy: str
    policy_path: Path
    n_action_steps: int
    suite: str
    alpha: float | None
    seed: int
    n_episodes_per_task: int
    device: str
    eval_batch_size: int = 1
    planning_chunk_size: int | None = DEFAULT_PLANNING_CHUNK_SIZE
    policy_config_n_action_steps: int | None = None
    residual_clip_mode: str = ""
    eval_delta_max: float | None = None
    eval_safety_joint_velocity_limit: float | None = None
    eval_control_dt: float | None = None


@dataclass
class ResultRow:
    policy: str
    n_action_steps: int
    planning_chunk_size: int | None
    execution_chunk_size: int
    replan_interval_steps: int
    policy_config_n_action_steps: int | None
    residual_clip_mode: str
    eval_delta_max: str
    eval_safety_joint_velocity_limit: str
    eval_control_dt: str
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
    status: str
    output_dir: str
    eval_info_path: str
    updated_at: str


CSV_FIELDS = list(ResultRow.__dataclass_fields__)


@dataclass(frozen=True)
class ActionStepMetadata:
    planning_chunk_size: int | None
    policy_config_n_action_steps: int | None
    delta_max: float | None = None
    safety_joint_velocity_limit: float | None = None
    control_dt: float | None = None


def read_action_step_metadata(policy_path: Path) -> ActionStepMetadata:
    """Read planning/execution defaults from a packaged policy config."""
    config_path = policy_path / "config.json"
    if not config_path.exists():
        return ActionStepMetadata(
            planning_chunk_size=DEFAULT_PLANNING_CHUNK_SIZE,
            policy_config_n_action_steps=None,
        )
    config = json.loads(config_path.read_text())
    return ActionStepMetadata(
        planning_chunk_size=config.get("chunk_size", DEFAULT_PLANNING_CHUNK_SIZE),
        policy_config_n_action_steps=config.get("n_action_steps"),
        delta_max=config.get("delta_max"),
        safety_joint_velocity_limit=config.get("safety_joint_velocity_limit"),
        control_dt=config.get("control_dt"),
    )


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_csv_strings(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one value")
    return values


def safe_name(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("+", "_")
        .replace(".", "p")
        .replace("[", "")
        .replace("]", "")
        .replace(",", "-")
        .replace(" ", "")
        .replace("=", "")
    )


def format_optional_float(value: float | None) -> str:
    return "" if value is None else str(value)


def build_eval_env(tmp_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HFRVLA_TMP_ROOT", str(tmp_root))
    env.setdefault("HF_DATASETS_CACHE", str(tmp_root / "hf_datasets"))
    env.setdefault("TMPDIR", str(tmp_root / "tmp"))
    env.setdefault("TMP", env["TMPDIR"])
    env.setdefault("TEMP", env["TMPDIR"])
    env.setdefault("NUMBA_CACHE_DIR", str(tmp_root / "numba"))
    env.setdefault("NUMBA_DISABLE_JIT", "1")
    env.setdefault("MPLCONFIGDIR", str(tmp_root / "matplotlib"))
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    for key in ("HF_DATASETS_CACHE", "TMPDIR", "NUMBA_CACHE_DIR", "MPLCONFIGDIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def output_dir_for_spec(spec: EvalSpec, eval_root: Path) -> Path:
    alpha_part = "" if spec.alpha is None else f"_alpha_{str(spec.alpha).replace('.', 'p')}"
    clip_part = (
        ""
        if spec.policy != "hfrvla" or spec.residual_clip_mode in {"", "config"}
        else f"_resclip_{safe_name(spec.residual_clip_mode)}"
    )
    return (
        eval_root
        / f"{safe_name(spec.policy)}_n{spec.n_action_steps}{alpha_part}{clip_part}"
        / f"{safe_name(spec.suite)}_{spec.n_episodes_per_task}ep_seed{spec.seed}_{spec.device}"
    )


def eval_info_path_for_spec(
    spec: EvalSpec,
    eval_root: Path,
    *,
    include_aliases: bool,
) -> Path:
    canonical = output_dir_for_spec(spec, eval_root) / "eval_info.json"
    if canonical.exists() or not include_aliases:
        return canonical
    alias = KNOWN_OUTPUT_ALIASES.get((spec.policy, spec.n_action_steps, spec.suite))
    if alias is not None and (alias / "eval_info.json").exists():
        return alias / "eval_info.json"
    return canonical


def summarize_eval_info(info: dict[str, Any]) -> tuple[int, int, float, float, float, float, list[int]]:
    overall = info.get("overall", {})
    per_task_successes = [
        int(sum(task.get("metrics", {}).get("successes", [])))
        for task in info.get("per_task", [])
    ]
    n_successes = int(sum(per_task_successes))
    n_episodes = int(overall.get("n_episodes") or sum(
        len(task.get("metrics", {}).get("successes", []))
        for task in info.get("per_task", [])
    ))
    pc_success = float(overall.get("pc_success", 100.0 * n_successes / n_episodes if n_episodes else 0.0))
    avg_sum_reward = float(overall.get("avg_sum_reward", 0.0))
    avg_max_reward = float(overall.get("avg_max_reward", 0.0))
    eval_s = float(overall.get("eval_s", 0.0))
    return (
        n_episodes,
        n_successes,
        pc_success,
        avg_sum_reward,
        avg_max_reward,
        eval_s,
        per_task_successes,
    )


def row_from_eval_info(
    spec: EvalSpec,
    info_path: Path,
    *,
    status: str,
) -> ResultRow:
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
    return ResultRow(
        policy=spec.policy,
        n_action_steps=spec.n_action_steps,
        planning_chunk_size=spec.planning_chunk_size,
        execution_chunk_size=spec.n_action_steps,
        replan_interval_steps=spec.n_action_steps,
        policy_config_n_action_steps=spec.policy_config_n_action_steps,
        residual_clip_mode=spec.residual_clip_mode,
        eval_delta_max=format_optional_float(spec.eval_delta_max),
        eval_safety_joint_velocity_limit=format_optional_float(spec.eval_safety_joint_velocity_limit),
        eval_control_dt=format_optional_float(spec.eval_control_dt),
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
        status=status,
        output_dir=str(info_path.parent),
        eval_info_path=str(info_path),
        updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def pending_row(spec: EvalSpec, info_path: Path, status: str) -> ResultRow:
    return ResultRow(
        policy=spec.policy,
        n_action_steps=spec.n_action_steps,
        planning_chunk_size=spec.planning_chunk_size,
        execution_chunk_size=spec.n_action_steps,
        replan_interval_steps=spec.n_action_steps,
        policy_config_n_action_steps=spec.policy_config_n_action_steps,
        residual_clip_mode=spec.residual_clip_mode,
        eval_delta_max=format_optional_float(spec.eval_delta_max),
        eval_safety_joint_velocity_limit=format_optional_float(spec.eval_safety_joint_velocity_limit),
        eval_control_dt=format_optional_float(spec.eval_control_dt),
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
        status=status,
        output_dir=str(info_path.parent),
        eval_info_path=str(info_path),
        updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def combined_rows(rows: list[ResultRow]) -> list[ResultRow]:
    groups: dict[
        tuple[str, int, int | None, int | None, str, str, str, str, str, int, int],
        list[ResultRow],
    ] = {}
    for row in rows:
        if row.suite == "combined" or row.status not in {"ok", "cached", "imported"}:
            continue
        key = (
            row.policy,
            row.n_action_steps,
            row.planning_chunk_size,
            row.policy_config_n_action_steps,
            row.residual_clip_mode,
            row.eval_delta_max,
            row.eval_safety_joint_velocity_limit,
            row.eval_control_dt,
            row.alpha,
            row.seed,
            row.n_episodes_per_task,
        )
        groups.setdefault(key, []).append(row)

    out: list[ResultRow] = []
    for (
        policy,
        n_action_steps,
        planning_chunk_size,
        policy_config_n_action_steps,
        residual_clip_mode,
        eval_delta_max,
        eval_safety_joint_velocity_limit,
        eval_control_dt,
        alpha,
        seed,
        n_episodes_per_task,
    ), group in groups.items():
        suites = {row.suite for row in group}
        if not {"libero_spatial", "libero_object"}.issubset(suites):
            continue
        selected = [
            next(row for row in group if row.suite == "libero_spatial"),
            next(row for row in group if row.suite == "libero_object"),
        ]
        n_episodes = sum(int(row.n_episodes or 0) for row in selected)
        n_successes = sum(int(row.n_successes or 0) for row in selected)
        pc_success = 100.0 * n_successes / n_episodes if n_episodes else 0.0
        eval_s = sum(float(row.eval_s or 0.0) for row in selected)
        out.append(
            ResultRow(
                policy=policy,
                n_action_steps=n_action_steps,
                planning_chunk_size=planning_chunk_size,
                execution_chunk_size=n_action_steps,
                replan_interval_steps=n_action_steps,
                policy_config_n_action_steps=policy_config_n_action_steps,
                residual_clip_mode=residual_clip_mode,
                eval_delta_max=eval_delta_max,
                eval_safety_joint_velocity_limit=eval_safety_joint_velocity_limit,
                eval_control_dt=eval_control_dt,
                suite="combined",
                alpha=alpha,
                seed=seed,
                n_episodes_per_task=n_episodes_per_task,
                n_episodes=n_episodes,
                n_successes=n_successes,
                pc_success=round(pc_success, 4),
                avg_sum_reward=None,
                avg_max_reward=None,
                eval_s=round(eval_s, 3),
                per_task_successes="",
                status="derived",
                output_dir="",
                eval_info_path="",
                updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
        )
    return out


def sort_key(row: ResultRow) -> tuple[int, str, int]:
    suite_order = {"libero_spatial": 0, "libero_object": 1, "combined": 2}
    return (row.n_action_steps, row.policy, suite_order.get(row.suite, 99))


def write_csv(csv_path: Path, rows: list[ResultRow]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    base_rows = [row for row in rows if row.suite != "combined"]
    all_rows = sorted(base_rows + combined_rows(base_rows), key=sort_key)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in all_rows)


def build_eval_command(spec: EvalSpec, eval_bin: Path, output_dir: Path) -> list[str]:
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
    if spec.policy == "hfrvla" and spec.alpha is not None:
        cmd.append(f"--policy.fast_residual_alpha={spec.alpha}")
    if spec.policy == "hfrvla" and spec.eval_delta_max is not None:
        cmd.append(f"--policy.delta_max={spec.eval_delta_max}")
    if spec.policy == "hfrvla" and spec.eval_safety_joint_velocity_limit is not None:
        cmd.append(f"--policy.safety_joint_velocity_limit={spec.eval_safety_joint_velocity_limit}")
    if spec.policy == "hfrvla" and spec.eval_control_dt is not None:
        cmd.append(f"--policy.control_dt={spec.eval_control_dt}")
    return cmd


def run_command(cmd: list[str], *, env: dict[str, str], log_path: Path, dry_run: bool) -> None:
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


def build_specs(args: argparse.Namespace) -> list[EvalSpec]:
    policies = parse_csv_strings(args.policies)
    steps = parse_csv_ints(args.n_action_steps)
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
    for n_action_steps in steps:
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
                        n_action_steps=n_action_steps,
                        suite=suite,
                        alpha=args.hfrvla_alpha if policy == "hfrvla" else None,
                        seed=args.seed,
                        n_episodes_per_task=args.n_episodes,
                        device=args.device,
                        eval_batch_size=args.eval_batch_size,
                        planning_chunk_size=metadata.planning_chunk_size,
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
        canonical_output_dir = output_dir_for_spec(spec, args.eval_root)
        info_path = eval_info_path_for_spec(
            spec,
            args.eval_root,
            include_aliases=args.import_known_results,
        )
        if info_path.exists() and not args.force:
            status = "imported" if info_path.parent != canonical_output_dir else "cached"
            rows.append(row_from_eval_info(spec, info_path, status=status))
            write_csv(args.csv, rows)
            continue

        if args.collect_only:
            rows.append(pending_row(spec, canonical_output_dir / "eval_info.json", "pending"))
            write_csv(args.csv, rows)
            continue

        cmd = build_eval_command(spec, args.eval_bin, canonical_output_dir)
        try:
            run_command(
                cmd,
                env=env,
                log_path=canonical_output_dir / "eval.log",
                dry_run=args.dry_run,
            )
            if args.dry_run:
                rows.append(pending_row(spec, canonical_output_dir / "eval_info.json", "dry-run"))
            else:
                rows.append(
                    row_from_eval_info(
                        spec,
                        canonical_output_dir / "eval_info.json",
                        status="ok",
                    )
                )
        except Exception as exc:
            if not args.keep_going:
                raise
            row = pending_row(spec, canonical_output_dir / "eval_info.json", "failed")
            row.per_task_successes = str(exc)
            rows.append(row)
        write_csv(args.csv, rows)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-action-steps", default="2,4,8,16,32")
    p.add_argument("--policies", default="hfrvla,smolvla")
    p.add_argument("--suites", default="libero_spatial,libero_object")
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
    p.add_argument("--eval-batch-size", type=int, default=3, help="Parallel eval envs per task.")
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
    p.add_argument(
        "--import-known-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Import known hand-run n=8 output directories when canonical outputs are absent.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_sweep(args)
    print(f"[action-steps-sweep] wrote {args.csv}")
    completed = sum(1 for row in rows if row.status in {"ok", "cached", "imported"})
    print(f"[action-steps-sweep] completed suite rows: {completed}/{len(rows)}")


if __name__ == "__main__":
    main()
