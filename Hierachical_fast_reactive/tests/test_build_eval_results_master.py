import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_eval_results_master import (  # noqa: E402
    MASTER_FIELDS,
    build_from_files,
    build_master_rows,
    render_csv,
)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_row(policy: str, suite: str, *, output_dir: str = "") -> dict[str, str]:
    return {
        "policy": policy,
        "n_action_steps": "8",
        "planning_chunk_size": "50",
        "execution_chunk_size": "8",
        "replan_interval_steps": "8",
        "policy_config_n_action_steps": "50" if policy == "hfrvla" else "1",
        "residual_clip_mode": "config" if policy == "hfrvla" else "",
        "eval_delta_max": "0.2" if policy == "hfrvla" else "",
        "eval_safety_joint_velocity_limit": "2.0" if policy == "hfrvla" else "",
        "eval_control_dt": "0.1" if policy == "hfrvla" else "",
        "suite": suite,
        "alpha": "0.5" if policy == "hfrvla" else "",
        "seed": "42",
        "n_episodes_per_task": "5",
        "n_episodes": "50",
        "n_successes": "41",
        "pc_success": "82.0",
        "avg_sum_reward": "0.82",
        "avg_max_reward": "0.82",
        "eval_s": "12.5",
        "per_task_successes": "5,5,4,4,5,2,5,3,4,4",
        "status": "ok",
        "output_dir": output_dir,
        "eval_info_path": f"{output_dir}/eval_info.json" if output_dir else "",
        "updated_at": "2026-05-28T00:00:00+00:00",
    }


def test_build_master_rows_joins_manifest_metadata_and_normalizes_repo_paths(tmp_path):
    repo_root = tmp_path / "repo"
    source_csv = repo_root / "outputs/action_steps_eval_sweep/results.csv"
    output_dir = repo_root / "outputs/action_steps_eval_sweep/evals/hfrvla_n8"
    write_csv(source_csv, [source_row("hfrvla", "libero_spatial", output_dir=str(output_dir))])

    manifest_rows = [
        {
            "source_csv": "outputs/action_steps_eval_sweep/results.csv",
            "sweep_id": "action_steps_eval_sweep",
            "sweep_type": "execution_replan_sweep",
            "comparison_axis": "execution_chunk_size",
            "policy": "hfrvla",
            "policy_family": "hfrvla",
            "checkpoint_id": "hfrvla_a2c2_wrist_seq2_30k",
            "checkpoint_path": str(repo_root / "checkpoints/hfrvla_a2c2_wrist_seq2_30k_packaged"),
            "model_id": "",
            "snapshot_id": "",
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
            "warmup_steps": "",
            "joint_steps": "",
            "refine_steps": "",
            "save_freq": "15000",
            "log_freq": "100",
            "tags": "a2c2;wrist",
            "notes": "",
        }
    ]

    rows = build_master_rows(manifest_rows, repo_root=repo_root)

    assert len(rows) == 1
    row = rows[0]
    assert set(row) == set(MASTER_FIELDS)
    assert row["source_csv"] == "outputs/action_steps_eval_sweep/results.csv"
    assert row["source_row_index"] == "1"
    assert row["eval_alpha"] == "0.5"
    assert row["residual_clip_mode"] == "config"
    assert row["eval_delta_max"] == "0.2"
    assert row["train_steps"] == "30000"
    assert row["lr"] == "3e-4"
    assert row["weight_decay"] == "1e-4"
    assert row["batch_size"] == "256"
    assert "alpha" not in row
    assert "output_dir" not in row


def test_build_master_rows_keeps_only_manifest_policy_rows(tmp_path):
    repo_root = tmp_path / "repo"
    source_csv = repo_root / "outputs/sweep/results.csv"
    write_csv(
        source_csv,
        [
            source_row("hfrvla", "libero_spatial"),
            source_row("smolvla", "libero_spatial"),
            source_row("hfrvla", "combined"),
            source_row("smolvla", "combined"),
        ],
    )
    manifest_rows = [
        {"source_csv": "outputs/sweep/results.csv", "sweep_id": "sweep", "policy": "hfrvla"},
        {"source_csv": "outputs/sweep/results.csv", "sweep_id": "sweep", "policy": "smolvla"},
    ]

    rows = build_master_rows(manifest_rows, repo_root=repo_root)

    assert len(rows) == 4
    assert [row["source_row_index"] for row in rows] == ["1", "3", "2", "4"]
    assert [row["policy"] for row in rows] == ["hfrvla", "hfrvla", "smolvla", "smolvla"]


def test_build_master_rows_expands_metadata_profile(tmp_path):
    repo_root = tmp_path / "repo"
    source_csv = repo_root / "outputs/sweep/results.csv"
    write_csv(source_csv, [source_row("hfrvla", "combined")])

    rows = build_master_rows(
        [
            {
                "source_csv": "outputs/sweep/results.csv",
                "sweep_id": "sweep",
                "policy": "hfrvla",
                "metadata_profile": "hfrvla_30k",
            }
        ],
        repo_root=repo_root,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["metadata_profile"] == "hfrvla_30k"
    assert row["checkpoint_id"] == "hfrvla_a2c2_wrist_seq2_30k"
    assert row["train_steps"] == "30000"
    assert row["seq_len"] == "2"
    assert row["lr"] == "3e-4"
    assert row["weight_decay"] == "1e-4"


def test_build_master_rows_uses_source_checkpoint_and_lr_wd_overrides(tmp_path):
    repo_root = tmp_path / "repo"
    source_csv = repo_root / "outputs/lrwd/results.csv"
    row = source_row("hfrvla", "combined")
    row.update(
        {
            "run_name": "hfrvla_lr5e_4_wd1e_5_b512_50k",
            "lr": "5e-4",
            "weight_decay": "1e-5",
            "batch_size": "512",
            "train_steps": "50000",
        }
    )
    write_csv(source_csv, [row])

    rows = build_master_rows(
        [
            {
                "source_csv": "outputs/lrwd/results.csv",
                "sweep_id": "hfrvla_lrwd_alpha075_clip02_eval",
                "policy": "hfrvla",
                "metadata_profile": "hfrvla_lrwd_b512_50k",
            }
        ],
        repo_root=repo_root,
    )

    assert len(rows) == 1
    out = rows[0]
    assert out["sweep_type"] == "learning_rate_weight_decay_sweep"
    assert out["comparison_axis"] == "learning_rate_weight_decay"
    assert out["checkpoint_id"] == "hfrvla_lr5e_4_wd1e_5_b512_50k"
    assert out["lr"] == "5e-4"
    assert out["weight_decay"] == "1e-5"
    assert out["batch_size"] == "512"
    assert out["train_steps"] == "50000"


def test_build_master_rows_expands_sweep_profile(tmp_path):
    repo_root = tmp_path / "repo"
    source_csv = repo_root / "outputs/action_steps_eval_sweep/results.csv"
    write_csv(source_csv, [source_row("smolvla", "combined")])

    rows = build_master_rows(
        [
            {
                "source_csv": "outputs/action_steps_eval_sweep/results.csv",
                "sweep_id": "action_steps_eval_sweep",
                "policy": "smolvla",
                "metadata_profile": "smolvla_libero",
            }
        ],
        repo_root=repo_root,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["sweep_type"] == "execution_replan_sweep"
    assert row["comparison_axis"] == "execution_chunk_size"
    assert row["metadata_profile"] == "smolvla_libero"
    assert row["checkpoint_id"] == "smolvla_libero"


def test_build_master_rows_preserves_planner_delay_fields(tmp_path):
    repo_root = tmp_path / "repo"
    source_csv = repo_root / "outputs/async_timestep_planner_delay_eval_sweep/results.csv"
    row = source_row("hfrvla", "libero_spatial")
    row.update(
        {
            "planner_delay_mode": "async_timestep",
            "planner_delay_steps": "4",
            "planner_delay_fallback": "hold_last",
            "async_request_interval_steps": "8",
            "fallback_steps_total": "2",
            "fallback_steps_mean": "0.02",
            "slow_replan_count": "713",
            "slow_chunk_latency_ms_mean": "275.3",
            "fast_latency_ms_mean": "10.7",
            "fast_applied_ratio": "1.0",
            "delta_norm_mean": "0.856",
            "delta_clip_fraction_mean": "0.391",
            "k_mean": "7.386",
            "async_request_count": "12",
            "async_activation_count": "12",
            "async_chunk_start_index_last": "4",
            "async_chunk_start_index_mean": "4.0",
            "async_dropped_old_queue_steps_last": "4",
            "async_dropped_old_queue_steps_mean": "4.0",
        }
    )
    write_csv(source_csv, [row])

    rows = build_master_rows(
        [
            {
                "source_csv": "outputs/async_timestep_planner_delay_eval_sweep/results.csv",
                "sweep_id": "async_timestep_planner_delay_eval_sweep",
                "policy": "hfrvla",
                "metadata_profile": "hfrvla_30k",
            }
        ],
        repo_root=repo_root,
    )

    assert len(rows) == 1
    out = rows[0]
    assert out["sweep_type"] == "planner_delay_sweep"
    assert out["comparison_axis"] == "planner_delay_steps"
    assert out["planner_delay_mode"] == "async_timestep"
    assert out["planner_delay_steps"] == "4"
    assert out["planner_delay_fallback"] == "hold_last"
    assert out["async_request_interval_steps"] == "8"
    assert out["fallback_steps_total"] == "2"
    assert out["slow_replan_count"] == "713"
    assert out["fast_latency_ms_mean"] == "10.7"
    assert out["delta_clip_fraction_mean"] == "0.391"
    assert out["async_chunk_start_index_mean"] == "4.0"
    assert out["async_dropped_old_queue_steps_mean"] == "4.0"


def test_render_csv_is_stable_and_includes_header(tmp_path):
    repo_root = tmp_path / "repo"
    source_csv = repo_root / "outputs/sweep/results.csv"
    write_csv(source_csv, [source_row("smolvla", "combined")])

    rows = build_master_rows(
        [{"source_csv": "outputs/sweep/results.csv", "sweep_id": "sweep", "policy": "smolvla"}],
        repo_root=repo_root,
    )
    text = render_csv(rows)

    assert text.startswith(",".join(MASTER_FIELDS) + "\n")
    assert "smolvla" in text
    assert "combined" in text


def test_build_from_files_check_accepts_current_output(tmp_path):
    repo_root = tmp_path / "repo"
    source_csv = repo_root / "outputs/sweep/results.csv"
    manifest = repo_root / "experiments/eval_registry/sources.csv"
    output = repo_root / "experiments/eval_registry/eval_results_master.csv"
    write_csv(source_csv, [source_row("smolvla", "combined")])
    write_csv(
        manifest,
        [{"source_csv": "outputs/sweep/results.csv", "sweep_id": "sweep", "policy": "smolvla"}],
    )

    assert build_from_files(
        manifest_path=manifest,
        output_path=output,
        repo_root=repo_root,
        check=False,
    ) == 0

    assert build_from_files(
        manifest_path=manifest,
        output_path=output,
        repo_root=repo_root,
        check=True,
    ) == 0
