#!/usr/bin/env python3
"""Build a repo-tracked master CSV for LIBERO evaluation results.

The raw sweep CSVs stay under ignored ``outputs/``. This script joins those
rows with a small manifest that records training and checkpoint metadata, then
emits a tidy long-format master CSV suitable for plotting and paper tables.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "experiments/eval_registry/sources.csv"
DEFAULT_OUTPUT = REPO_ROOT / "experiments/eval_registry/eval_results_master.csv"

METADATA_FIELDS = [
    "sweep_id",
    "sweep_type",
    "source_csv",
    "source_row_index",
    "comparison_axis",
    "policy",
    "metadata_profile",
    "checkpoint_id",
    "train_steps",
    "seq_len",
    "lr",
    "weight_decay",
    "batch_size",
    "tags",
    "notes",
]

RESULT_FIELDS = [
    "n_action_steps",
    "planning_chunk_size",
    "execution_chunk_size",
    "replan_interval_steps",
    "policy_config_n_action_steps",
    "eval_alpha",
    "residual_clip_mode",
    "eval_delta_max",
    "eval_safety_joint_velocity_limit",
    "eval_control_dt",
    "suite",
    "seed",
    "n_episodes_per_task",
    "n_episodes",
    "n_successes",
    "pc_success",
    "avg_sum_reward",
    "avg_max_reward",
    "eval_s",
    "per_task_successes",
    "status",
    "updated_at",
]

MASTER_FIELDS = METADATA_FIELDS + RESULT_FIELDS

METADATA_PROFILES = {
    "hfrvla_30k": {
        "policy_family": "hfrvla",
        "checkpoint_id": "hfrvla_a2c2_wrist_seq2_30k",
        "checkpoint_path": "checkpoints/hfrvla_a2c2_wrist_seq2_30k_packaged",
        "train_run_name": "hfrvla_a2c2_wrist_seq2_30k",
        "train_steps": "30000",
        "dataset_id": "HFRVLA_libero_v1",
        "dataset_backend": "fastcache",
        "seq_len": "2",
        "residual_merge_mode": "a2c2",
        "latent_context": "true",
        "train_alpha": "1.0",
        "config_alpha": "1.0",
        "lr": "3e-4",
        "weight_decay": "1e-4",
        "grad_clip_norm": "1.0",
        "batch_size": "256",
        "num_workers": "8",
        "save_freq": "15000",
        "log_freq": "100",
    },
    "hfrvla_lrwd_b512_50k": {
        "policy_family": "hfrvla",
        "train_steps": "50000",
        "dataset_id": "HFRVLA_libero_v1",
        "dataset_backend": "fastcache",
        "seq_len": "2",
        "residual_merge_mode": "a2c2",
        "latent_context": "true",
        "train_alpha": "1.0",
        "config_alpha": "1.0",
        "batch_size": "512",
        "num_workers": "8",
        "save_freq": "25000",
        "log_freq": "100",
    },
    "hfrvla_fwr_chunk_b512_50k": {
        "policy_family": "hfrvla",
        "checkpoint_id": "hfrvla_fwr_chunk_seq2_b512_50k_packaged_cuda",
        "checkpoint_path": "checkpoints/hfrvla_fwr_chunk_seq2_b512_50k_packaged_cuda",
        "train_run_name": "hfrvla_fwr_chunk_seq2_b512_50k",
        "train_steps": "50000",
        "dataset_id": "HFRVLA_libero_v1_fastcache_v3_plan50",
        "dataset_backend": "fastcache_v3",
        "seq_len": "2",
        "residual_merge_mode": "fast_wrist_chunk",
        "latent_context": "true",
        "train_alpha": "1.0",
        "config_alpha": "1.0",
        "lr": "3e-4",
        "weight_decay": "1e-5",
        "batch_size": "512",
        "num_workers": "8",
        "save_freq": "25000",
        "log_freq": "100",
    },
    "smolvla_libero": {
        "policy_family": "smolvla",
        "checkpoint_id": "smolvla_libero",
        "model_id": "HuggingFaceVLA/smolvla_libero",
        "snapshot_id": "6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
    },
}

SWEEP_PROFILES = {
    "action_steps_eval_sweep": {
        "sweep_type": "execution_replan_sweep",
        "comparison_axis": "execution_chunk_size",
    },
    "action_steps_eval_sweep_no_resclip": {
        "sweep_type": "execution_replan_sweep",
        "comparison_axis": "execution_chunk_size",
    },
    "chunk_size_eval_sweep": {
        "sweep_type": "matched_planning_execution_sweep",
        "comparison_axis": "planning_and_execution_chunk_size",
    },
    "chunk_size_eval_sweep_no_resclip": {
        "sweep_type": "matched_planning_execution_sweep",
        "comparison_axis": "planning_and_execution_chunk_size",
    },
    "action_step8_alpha_clip_sweep": {
        "sweep_type": "alpha_clip_sweep",
        "comparison_axis": "alpha_delta_max",
    },
    "hfrvla_lrwd_alpha075_clip02_eval": {
        "sweep_type": "learning_rate_weight_decay_sweep",
        "comparison_axis": "learning_rate_weight_decay",
    },
    "fwr_chunk_plan50_exec50_eval": {
        "sweep_type": "matched_planning_execution_sweep",
        "comparison_axis": "planning_and_execution_chunk_size",
    },
    "fwr_action_steps_10x10": {
        "sweep_type": "execution_replan_sweep",
        "comparison_axis": "execution_chunk_size",
    },
    "fwr_chunk_size_10x10": {
        "sweep_type": "matched_planning_execution_sweep",
        "comparison_axis": "planning_and_execution_chunk_size",
    },
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def resolve_repo_path(value: str, *, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def repo_relative_path(value: str, *, repo_root: Path) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return value


def manifest_value(manifest_row: dict[str, str], field: str) -> str:
    if manifest_row.get(field, ""):
        return manifest_row[field]

    sweep_id = manifest_row.get("sweep_id", "")
    if sweep_id and field in {"sweep_type", "comparison_axis"}:
        return SWEEP_PROFILES.get(sweep_id, {}).get(field, "")

    profile_id = manifest_row.get("metadata_profile", "")
    if not profile_id:
        return ""
    if profile_id not in METADATA_PROFILES:
        raise ValueError(f"unknown metadata_profile: {profile_id}")
    return METADATA_PROFILES[profile_id].get(field, "")


def source_metadata_value(
    source_row: dict[str, str],
    field: str,
) -> str:
    if field == "checkpoint_id":
        return source_row.get("checkpoint_id") or source_row.get("run_name", "")
    if field == "lr":
        return source_row.get("lr") or source_row.get("train_lr", "")
    if field == "weight_decay":
        return source_row.get("weight_decay") or source_row.get("train_weight_decay", "")
    return source_row.get(field, "")


def source_result_value(source_row: dict[str, str], field: str) -> str:
    aliases = {
        "eval_delta_max": ("eval_delta_max", "delta_max"),
        "eval_safety_joint_velocity_limit": ("eval_safety_joint_velocity_limit", "safety_joint_velocity_limit"),
        "eval_control_dt": ("eval_control_dt", "control_dt"),
    }
    for key in aliases.get(field, (field,)):
        if source_row.get(key, ""):
            return source_row[key]
    return ""



def master_row_from_source(
    manifest_row: dict[str, str],
    source_row: dict[str, str],
    *,
    source_row_index: int,
    repo_root: Path,
) -> dict[str, str]:
    row: dict[str, str] = {}
    for field in METADATA_FIELDS:
        if field == "source_row_index":
            row[field] = str(source_row_index)
        elif field == "source_csv":
            row[field] = repo_relative_path(manifest_value(manifest_row, field), repo_root=repo_root)
        elif field == "policy":
            row[field] = source_row.get("policy", manifest_value(manifest_row, field))
        else:
            row[field] = source_metadata_value(source_row, field) or manifest_value(manifest_row, field)

    for field in RESULT_FIELDS:
        if field == "eval_alpha":
            row[field] = source_row.get("alpha", "")
        else:
            row[field] = source_result_value(source_row, field)

    return {field: row.get(field, "") for field in MASTER_FIELDS}


def build_master_rows(
    manifest_rows: list[dict[str, str]],
    *,
    repo_root: Path = REPO_ROOT,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    source_cache: dict[str, list[dict[str, str]]] = {}

    for manifest_row in manifest_rows:
        source_csv = manifest_value(manifest_row, "source_csv")
        policy = manifest_value(manifest_row, "policy")
        if not source_csv:
            raise ValueError("manifest row is missing source_csv")
        if not policy:
            raise ValueError(f"manifest row for {source_csv} is missing policy")

        if source_csv not in source_cache:
            source_cache[source_csv] = read_csv_rows(resolve_repo_path(source_csv, repo_root=repo_root))

        for source_row_index, source_row in enumerate(source_cache[source_csv], start=1):
            source_policy = source_row.get("policy") or policy
            if source_policy != policy:
                continue
            rows.append(
                master_row_from_source(
                    manifest_row,
                    source_row,
                    source_row_index=source_row_index,
                    repo_root=repo_root,
                )
            )
    return rows


def render_csv(rows: list[dict[str, str]]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=MASTER_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in MASTER_FIELDS})
    return out.getvalue()


def build_from_files(
    *,
    manifest_path: Path,
    output_path: Path,
    repo_root: Path = REPO_ROOT,
    check: bool,
) -> int:
    manifest_rows = read_csv_rows(manifest_path)
    rows = build_master_rows(manifest_rows, repo_root=repo_root)
    text = render_csv(rows)

    if check:
        if not output_path.exists():
            print(f"[eval-registry] missing output: {output_path}", file=sys.stderr)
            return 1
        current = output_path.read_bytes()
        expected = text.encode()
        if current != expected:
            print(f"[eval-registry] stale output: {output_path}", file=sys.stderr)
            return 1
        print(f"[eval-registry] check ok: {output_path} ({len(rows)} rows)")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(text.encode())
    print(f"[eval-registry] wrote {output_path} ({len(rows)} rows)")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--check", action="store_true", help="Fail if the master CSV is stale.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(
        build_from_files(
            manifest_path=args.manifest,
            output_path=args.output,
            check=args.check,
        )
    )


if __name__ == "__main__":
    main()
