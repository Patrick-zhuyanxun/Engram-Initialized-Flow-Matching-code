#!/usr/bin/env python3
"""Run HFRVLA n=8 LIBERO evals across alpha and residual clip budgets.

This sweep keeps the action-step setup fixed to the existing matched n=8
execution/replan comparison:

    planning_chunk_size=50, execution_chunk_size=8, replan_interval_steps=8

It varies only the deployment merge scalar and the fast residual clip budget.
Results are stored in one CSV and can be resumed from completed eval_info.json
files. The known alpha=0.5, delta_max=0.2 baseline is imported by default.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_action_steps_eval_sweep import (  # noqa: E402
    CSV_FIELDS,
    DEFAULT_HFRVLA_POLICY,
    DEFAULT_TMP_ROOT,
    EvalSpec,
    ResultRow,
    build_eval_command,
    build_eval_env,
    combined_rows,
    format_optional_float,
    parse_csv_strings,
    pending_row,
    read_action_step_metadata,
    row_from_eval_info,
    run_command,
    safe_name,
)


DEFAULT_EVAL_ROOT = REPO_ROOT / "outputs/action_step8_alpha_clip_sweep/evals"
DEFAULT_CSV = REPO_ROOT / "outputs/action_step8_alpha_clip_sweep/results.csv"
DEFAULT_N_ACTION_STEPS = 8
DEFAULT_ALPHAS = (0.25, 0.5, 0.75, 1.0)
DEFAULT_DELTA_MAXES = (0.2, 0.3, 0.4)
DEFAULT_CONTROL_DT = 0.1
DEFAULT_SUITES = ("libero_spatial", "libero_object")

KNOWN_BASELINE_OUTPUT_ALIASES: dict[tuple[float, float, str], Path] = {
    (
        0.5,
        0.2,
        "libero_spatial",
    ): REPO_ROOT / "outputs/eval_hfrvla_a2c2_wrist_seq2_30k_n8_alpha_05_spatial_cuda",
    (
        0.5,
        0.2,
        "libero_object",
    ): REPO_ROOT / "outputs/eval_hfrvla_a2c2_wrist_seq2_30k_n8_alpha_05_object_cuda",
}


def parse_csv_floats(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one float")
    return values


def output_dir_for_spec(spec: EvalSpec, eval_root: Path) -> Path:
    alpha = "none" if spec.alpha is None else safe_name(str(spec.alpha))
    delta = "config" if spec.eval_delta_max is None else safe_name(format_optional_float(spec.eval_delta_max))
    return (
        eval_root
        / f"hfrvla_n{spec.n_action_steps}_alpha_{alpha}_delta_{delta}"
        / f"{safe_name(spec.suite)}_{spec.n_episodes_per_task}ep_seed{spec.seed}_{spec.device}"
    )


def eval_info_path_for_spec(
    spec: EvalSpec,
    eval_root: Path,
    *,
    import_known_results: bool,
) -> Path:
    canonical = output_dir_for_spec(spec, eval_root) / "eval_info.json"
    if canonical.exists() or not import_known_results:
        return canonical
    if spec.alpha is None or spec.eval_delta_max is None:
        return canonical
    alias = KNOWN_BASELINE_OUTPUT_ALIASES.get((float(spec.alpha), float(spec.eval_delta_max), spec.suite))
    if alias is not None and (alias / "eval_info.json").exists():
        return alias / "eval_info.json"
    return canonical


def build_specs(args: argparse.Namespace) -> list[EvalSpec]:
    alphas = parse_csv_floats(args.alphas)
    delta_maxes = parse_csv_floats(args.delta_maxes)
    suites = parse_csv_strings(args.suites)
    metadata = read_action_step_metadata(args.hfrvla_policy)

    specs: list[EvalSpec] = []
    for alpha in alphas:
        for delta_max in delta_maxes:
            safety_limit = round(delta_max / args.control_dt, 10) if args.control_dt > 0 else 0.0
            for suite in suites:
                specs.append(
                    EvalSpec(
                        policy="hfrvla",
                        policy_path=args.hfrvla_policy,
                        n_action_steps=args.n_action_steps,
                        suite=suite,
                        alpha=alpha,
                        seed=args.seed,
                        n_episodes_per_task=args.n_episodes,
                        device=args.device,
                        eval_batch_size=args.eval_batch_size,
                        planning_chunk_size=metadata.planning_chunk_size,
                        policy_config_n_action_steps=metadata.policy_config_n_action_steps,
                        residual_clip_mode="config",
                        eval_delta_max=delta_max,
                        eval_safety_joint_velocity_limit=safety_limit,
                        eval_control_dt=args.control_dt,
                    )
                )
    return specs


def result_sort_key(row: ResultRow) -> tuple[float, float, int]:
    suite_order = {"libero_spatial": 0, "libero_object": 1, "combined": 2}
    alpha = float(row.alpha) if row.alpha else -1.0
    delta_max = float(row.eval_delta_max) if row.eval_delta_max else -1.0
    return (alpha, delta_max, suite_order.get(row.suite, 99))


def write_csv(csv_path: Path, rows: list[ResultRow]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    base_rows = [row for row in rows if row.suite != "combined"]
    all_rows = sorted(base_rows + combined_rows(base_rows), key=result_sort_key)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in all_rows)


def run_sweep(args: argparse.Namespace) -> list[ResultRow]:
    env = build_eval_env(args.tmp_root)
    rows: list[ResultRow] = []
    for spec in build_specs(args):
        canonical_output_dir = output_dir_for_spec(spec, args.eval_root)
        info_path = eval_info_path_for_spec(
            spec,
            args.eval_root,
            import_known_results=args.import_known_results and not args.force,
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
                rows.append(row_from_eval_info(spec, canonical_output_dir / "eval_info.json", status="ok"))
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
    p.add_argument("--n-action-steps", type=int, default=DEFAULT_N_ACTION_STEPS)
    p.add_argument("--alphas", default=",".join(str(v) for v in DEFAULT_ALPHAS))
    p.add_argument("--delta-maxes", default=",".join(str(v) for v in DEFAULT_DELTA_MAXES))
    p.add_argument("--control-dt", type=float, default=DEFAULT_CONTROL_DT)
    p.add_argument("--suites", default=",".join(DEFAULT_SUITES))
    p.add_argument("--hfrvla-policy", type=Path, default=DEFAULT_HFRVLA_POLICY)
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
        help="Import the known n=8 alpha=0.5 delta=0.2 baseline outputs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_sweep(args)
    print(f"[action-step8-alpha-clip-sweep] wrote {args.csv}")
    completed = sum(1 for row in rows if row.status in {"ok", "cached", "imported"})
    print(f"[action-step8-alpha-clip-sweep] completed suite rows: {completed}/{len(rows)}")
    print(f"[action-step8-alpha-clip-sweep] updated_at={datetime.now(timezone.utc).isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
