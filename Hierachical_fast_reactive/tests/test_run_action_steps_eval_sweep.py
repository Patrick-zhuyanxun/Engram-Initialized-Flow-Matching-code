import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_action_steps_eval_sweep import (
    EvalSpec,
    build_eval_command,
    combined_rows,
    output_dir_for_spec,
    read_action_step_metadata,
    row_from_eval_info,
    write_csv,
)


def test_build_eval_command_sets_matched_n_action_steps_and_hfrvla_alpha(tmp_path):
    spec = EvalSpec(
        policy="hfrvla",
        policy_path=tmp_path / "hfrvla_policy",
        n_action_steps=8,
        suite="libero_spatial",
        alpha=0.5,
        seed=42,
        n_episodes_per_task=5,
        device="cuda",
    )

    cmd = build_eval_command(spec, tmp_path / "lerobot-eval", tmp_path / "out")

    assert "--policy.n_action_steps=8" in cmd
    assert "--policy.fast_residual_alpha=0.5" in cmd
    assert not any(part.startswith("--policy.a2c2_alpha=") for part in cmd)
    assert "--env.task=libero_spatial" in cmd
    assert "--eval.n_episodes=5" in cmd


def test_build_eval_command_can_disable_hfrvla_residual_clip(tmp_path):
    spec = EvalSpec(
        policy="hfrvla",
        policy_path=tmp_path / "hfrvla_policy",
        n_action_steps=8,
        suite="libero_spatial",
        alpha=0.5,
        seed=42,
        n_episodes_per_task=5,
        device="cuda",
        residual_clip_mode="none",
        eval_delta_max=999.0,
        eval_safety_joint_velocity_limit=0.0,
        eval_control_dt=0.1,
    )

    cmd = build_eval_command(spec, tmp_path / "lerobot-eval", tmp_path / "out")

    assert "--policy.fast_residual_alpha=0.5" in cmd
    assert not any(part.startswith("--policy.a2c2_alpha=") for part in cmd)
    assert "--policy.delta_max=999.0" in cmd
    assert "--policy.safety_joint_velocity_limit=0.0" in cmd
    assert "--policy.control_dt=0.1" in cmd


def test_build_eval_command_does_not_set_alpha_for_smolvla(tmp_path):
    spec = EvalSpec(
        policy="smolvla",
        policy_path=tmp_path / "smolvla_policy",
        n_action_steps=16,
        suite="libero_object",
        alpha=None,
        seed=42,
        n_episodes_per_task=5,
        device="cuda",
    )

    cmd = build_eval_command(spec, tmp_path / "lerobot-eval", tmp_path / "out")

    assert "--policy.n_action_steps=16" in cmd
    assert not any(part.startswith("--policy.fast_residual_alpha=") for part in cmd)
    assert not any(part.startswith("--policy.a2c2_alpha=") for part in cmd)


def test_row_from_eval_info_and_combined_rows(tmp_path):
    spatial_info = {
        "overall": {"pc_success": 60.0, "n_episodes": 10, "eval_s": 12.5},
        "per_task": [
            {"metrics": {"successes": [True, False, True, True, False]}},
            {"metrics": {"successes": [True, True, False, False, True]}},
        ],
    }
    object_info = {
        "overall": {"pc_success": 80.0, "n_episodes": 10, "eval_s": 10.0},
        "per_task": [
            {"metrics": {"successes": [True, True, True, True, False]}},
            {"metrics": {"successes": [True, True, False, True, True]}},
        ],
    }
    spatial_path = tmp_path / "spatial/eval_info.json"
    object_path = tmp_path / "object/eval_info.json"
    spatial_path.parent.mkdir()
    object_path.parent.mkdir()
    spatial_path.write_text(json.dumps(spatial_info))
    object_path.write_text(json.dumps(object_info))

    base = dict(
        policy="hfrvla",
        policy_path=tmp_path / "policy",
        n_action_steps=4,
        alpha=0.5,
        seed=42,
        n_episodes_per_task=5,
        device="cuda",
    )
    spatial_row = row_from_eval_info(
        EvalSpec(suite="libero_spatial", **base),
        spatial_path,
        status="ok",
    )
    object_row = row_from_eval_info(
        EvalSpec(suite="libero_object", **base),
        object_path,
        status="ok",
    )

    assert spatial_row.n_successes == 6
    assert spatial_row.per_task_successes == "3,3"
    assert spatial_row.planning_chunk_size == 50
    assert spatial_row.execution_chunk_size == 4
    assert spatial_row.replan_interval_steps == 4
    assert spatial_row.residual_clip_mode == ""

    combined = combined_rows([spatial_row, object_row])
    assert len(combined) == 1
    assert combined[0].suite == "combined"
    assert combined[0].n_successes == 14
    assert combined[0].n_episodes == 20
    assert combined[0].pc_success == 70.0
    assert combined[0].eval_s == 22.5
    assert combined[0].planning_chunk_size == 50
    assert combined[0].execution_chunk_size == 4


def test_write_csv_includes_derived_combined_row(tmp_path):
    for suite in ("libero_spatial", "libero_object"):
        info_path = tmp_path / suite / "eval_info.json"
        info_path.parent.mkdir()
        info_path.write_text(
            json.dumps(
                {
                    "overall": {"pc_success": 100.0, "n_episodes": 1, "eval_s": 1.0},
                    "per_task": [{"metrics": {"successes": [True]}}],
                }
            )
        )

    rows = [
        row_from_eval_info(
            EvalSpec(
                policy="smolvla",
                policy_path=tmp_path / "policy",
                n_action_steps=2,
                suite=suite,
                alpha=None,
                seed=42,
                n_episodes_per_task=1,
                device="cuda",
            ),
            tmp_path / suite / "eval_info.json",
            status="ok",
        )
        for suite in ("libero_spatial", "libero_object")
    ]
    csv_path = tmp_path / "results.csv"

    write_csv(csv_path, rows)

    text = csv_path.read_text()
    assert "planning_chunk_size,execution_chunk_size,replan_interval_steps" in text
    assert "libero_spatial" in text
    assert "libero_object" in text
    assert "combined" in text


def test_read_action_step_metadata_from_policy_config(tmp_path):
    policy_path = tmp_path / "policy"
    policy_path.mkdir()
    (policy_path / "config.json").write_text(
        json.dumps({"chunk_size": 50, "n_action_steps": 1})
    )

    metadata = read_action_step_metadata(policy_path)

    assert metadata.planning_chunk_size == 50
    assert metadata.policy_config_n_action_steps == 1


def test_output_dir_for_spec_is_stable(tmp_path):
    spec = EvalSpec(
        policy="hfrvla",
        policy_path=tmp_path / "policy",
        n_action_steps=32,
        suite="libero_object",
        alpha=0.5,
        seed=42,
        n_episodes_per_task=5,
        device="cuda",
    )

    assert output_dir_for_spec(spec, tmp_path).as_posix().endswith(
        "hfrvla_n32_alpha_0p5/libero_object_5ep_seed42_cuda"
    )


def test_output_dir_for_no_clip_spec_is_distinct(tmp_path):
    spec = EvalSpec(
        policy="hfrvla",
        policy_path=tmp_path / "policy",
        n_action_steps=32,
        suite="libero_object",
        alpha=0.5,
        seed=42,
        n_episodes_per_task=5,
        device="cuda",
        residual_clip_mode="none",
    )

    assert output_dir_for_spec(spec, tmp_path).as_posix().endswith(
        "hfrvla_n32_alpha_0p5_resclip_none/libero_object_5ep_seed42_cuda"
    )
