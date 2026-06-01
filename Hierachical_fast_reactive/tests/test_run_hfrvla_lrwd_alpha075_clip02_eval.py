import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hfrvla_lrwd_alpha075_clip02_eval import (  # noqa: E402
    CSV_FIELDS,
    DEFAULT_CANDIDATES,
    DEFAULT_REFERENCE_ROWS,
    build_eval_spec,
    build_package_command,
    output_dir_for_candidate,
    package_is_ready,
)
from scripts.run_action_steps_eval_sweep import build_eval_command  # noqa: E402


def make_args(tmp_path):
    return argparse.Namespace(
        alpha=0.75,
        delta_max=0.2,
        control_dt=0.1,
        n_action_steps=8,
        seed=42,
        n_episodes=5,
        device="cuda",
        suites="libero_spatial,libero_object",
        eval_root=tmp_path / "hfrvla_lrwd_alpha075_clip02_eval/evals",
        repo_root=tmp_path,
        package_python=Path("/venv/bin/python"),
        dinov3_repo=Path("checkpoints/dinov3_src"),
        dinov3_weights=Path("checkpoints/Dino_weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
        dataset_repo_id="HFRVLA_libero_v1",
        dataset_root=Path("checkpoints/HFRVLA_libero_v1_merged_reindexed"),
    )


def test_default_candidates_are_high_lr_shortlist() -> None:
    assert len(DEFAULT_CANDIDATES) == 8
    assert [candidate.run_name for candidate in DEFAULT_CANDIDATES] == [
        "hfrvla_lr3e_4_wd0_b512_50k",
        "hfrvla_lr3e_4_wd1e_5_b512_50k",
        "hfrvla_lr3e_4_wd1e_4_b512_50k",
        "hfrvla_lr3e_4_wd3e_4_b512_50k",
        "hfrvla_lr5e_4_wd0_b512_50k",
        "hfrvla_lr5e_4_wd1e_5_b512_50k",
        "hfrvla_lr5e_4_wd1e_4_b512_50k",
        "hfrvla_lr5e_4_wd3e_4_b512_50k",
    ]
    assert {candidate.lr for candidate in DEFAULT_CANDIDATES} == {"3e-4", "5e-4"}
    assert {candidate.batch_size for candidate in DEFAULT_CANDIDATES} == {"512"}


def test_csv_fields_include_planning_execution_and_registry_metadata() -> None:
    assert "policy" in CSV_FIELDS
    assert "checkpoint_id" in CSV_FIELDS
    assert "planning_chunk_size" in CSV_FIELDS
    assert "execution_chunk_size" in CSV_FIELDS
    assert "replan_interval_steps" in CSV_FIELDS
    assert "policy_config_n_action_steps" in CSV_FIELDS
    assert "eval_delta_max" in CSV_FIELDS
    assert "eval_safety_joint_velocity_limit" in CSV_FIELDS
    assert "eval_control_dt" in CSV_FIELDS


def test_reference_rows_use_matched_smolvla_plan50_exec8_baseline() -> None:
    assert DEFAULT_REFERENCE_ROWS == (
        ("smolvla_plan50_exec8_baseline", "libero_spatial", 32, 50, 64.0),
        ("smolvla_plan50_exec8_baseline", "libero_object", 45, 50, 90.0),
        ("smolvla_plan50_exec8_baseline", "combined", 77, 100, 77.0),
    )


def test_package_command_uses_last_fast_checkpoint_and_packaged_output(tmp_path) -> None:
    args = make_args(tmp_path)
    candidate = DEFAULT_CANDIDATES[0]

    cmd = build_package_command(candidate, args)

    assert cmd[:2] == ["/venv/bin/python", "scripts/package_hfrvla_checkpoint.py"]
    assert "--fast-ckpt" in cmd
    assert str(tmp_path / "checkpoints/hfrvla_lr3e_4_wd0_b512_50k/checkpoints/last/pretrained_model") in cmd
    assert "--out-dir" in cmd
    assert str(tmp_path / "checkpoints/hfrvla_lr3e_4_wd0_b512_50k_packaged") in cmd
    assert "--dataset-repo-id=HFRVLA_libero_v1" in cmd


def test_eval_spec_sets_alpha_clip_safety_and_n8(tmp_path) -> None:
    args = make_args(tmp_path)
    candidate = DEFAULT_CANDIDATES[-1]
    packaged = tmp_path / "checkpoints" / f"{candidate.run_name}_packaged"
    packaged.mkdir(parents=True)
    (packaged / "config.json").write_text(json.dumps({"chunk_size": 50, "n_action_steps": 50}))

    spec = build_eval_spec(candidate, "libero_spatial", packaged, args)
    cmd = build_eval_command(spec, tmp_path / "lerobot-eval", tmp_path / "out")

    assert spec.alpha == 0.75
    assert spec.eval_delta_max == 0.2
    assert spec.eval_safety_joint_velocity_limit == 2.0
    assert spec.eval_control_dt == 0.1
    assert spec.n_action_steps == 8
    assert spec.planning_chunk_size == 50
    assert "--policy.fast_residual_alpha=0.75" in cmd
    assert not any(part.startswith("--policy.a2c2_alpha=") for part in cmd)
    assert "--policy.delta_max=0.2" in cmd
    assert "--policy.safety_joint_velocity_limit=2.0" in cmd
    assert "--policy.control_dt=0.1" in cmd
    assert "--policy.n_action_steps=8" in cmd


def test_output_dir_is_stable_and_package_readiness_checks_required_files(tmp_path) -> None:
    args = make_args(tmp_path)
    candidate = DEFAULT_CANDIDATES[0]
    packaged = tmp_path / "policy"
    packaged.mkdir()

    assert output_dir_for_candidate(candidate, "libero_object", args).as_posix().endswith(
        "hfrvla_lrwd_alpha075_clip02_eval/evals/hfrvla_lr3e_4_wd0_b512_50k/"
        "libero_object_5ep_seed42_cuda"
    )
    assert not package_is_ready(packaged)

    (packaged / "config.json").write_text("{}")
    (packaged / "model.safetensors").write_text("")

    assert package_is_ready(packaged)
