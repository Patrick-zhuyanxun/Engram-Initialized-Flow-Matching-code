import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_chunk_size_eval_sweep import (  # noqa: E402
    build_eval_command,
    build_specs,
    output_dir_for_spec,
)
from scripts.run_action_steps_eval_sweep import EvalSpec  # noqa: E402


def test_build_eval_command_sets_planning_and_execution_chunk_sizes(tmp_path):
    spec = EvalSpec(
        policy="hfrvla",
        policy_path=tmp_path / "hfrvla_policy",
        n_action_steps=8,
        suite="libero_spatial",
        alpha=0.5,
        seed=42,
        n_episodes_per_task=1,
        device="cuda",
        planning_chunk_size=8,
        policy_config_n_action_steps=50,
    )

    cmd = build_eval_command(
        spec,
        tmp_path / "lerobot-eval",
        tmp_path / "out",
        task_ids="[0]",
    )

    assert "--policy.chunk_size=8" in cmd
    assert "--policy.n_action_steps=8" in cmd
    assert "--policy.fast_residual_alpha=0.5" in cmd
    assert not any(part.startswith("--policy.a2c2_alpha=") for part in cmd)
    assert "--env.task_ids=[0]" in cmd
    assert "--eval.n_episodes=1" in cmd


def test_build_eval_command_can_disable_hfrvla_residual_clip(tmp_path):
    spec = EvalSpec(
        policy="hfrvla",
        policy_path=tmp_path / "hfrvla_policy",
        n_action_steps=8,
        suite="libero_spatial",
        alpha=0.5,
        seed=42,
        n_episodes_per_task=1,
        device="cuda",
        planning_chunk_size=8,
        policy_config_n_action_steps=50,
        residual_clip_mode="none",
        eval_delta_max=999.0,
        eval_safety_joint_velocity_limit=0.0,
        eval_control_dt=0.1,
    )

    cmd = build_eval_command(spec, tmp_path / "lerobot-eval", tmp_path / "out")

    assert "--policy.chunk_size=8" in cmd
    assert "--policy.n_action_steps=8" in cmd
    assert "--policy.fast_residual_alpha=0.5" in cmd
    assert not any(part.startswith("--policy.a2c2_alpha=") for part in cmd)
    assert "--policy.delta_max=999.0" in cmd
    assert "--policy.safety_joint_velocity_limit=0.0" in cmd
    assert "--policy.control_dt=0.1" in cmd


def test_build_eval_command_omits_alpha_for_smolvla(tmp_path):
    spec = EvalSpec(
        policy="smolvla",
        policy_path=tmp_path / "smolvla_policy",
        n_action_steps=16,
        suite="libero_object",
        alpha=None,
        seed=42,
        n_episodes_per_task=5,
        device="cuda",
        planning_chunk_size=16,
        policy_config_n_action_steps=1,
    )

    cmd = build_eval_command(spec, tmp_path / "lerobot-eval", tmp_path / "out")

    assert "--policy.chunk_size=16" in cmd
    assert "--policy.n_action_steps=16" in cmd
    assert not any(part.startswith("--policy.fast_residual_alpha=") for part in cmd)
    assert not any(part.startswith("--policy.a2c2_alpha=") for part in cmd)
    assert not any(part.startswith("--env.task_ids=") for part in cmd)


def test_output_dir_for_spec_uses_chunk_size(tmp_path):
    spec = EvalSpec(
        policy="hfrvla",
        policy_path=tmp_path / "policy",
        n_action_steps=32,
        suite="libero_object",
        alpha=0.5,
        seed=42,
        n_episodes_per_task=5,
        device="cuda",
        planning_chunk_size=32,
        policy_config_n_action_steps=50,
    )

    assert output_dir_for_spec(spec, tmp_path).as_posix().endswith(
        "hfrvla_chunk32_alpha_0p5/libero_object_5ep_seed42_cuda"
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
        planning_chunk_size=32,
        policy_config_n_action_steps=50,
        residual_clip_mode="none",
    )

    assert output_dir_for_spec(spec, tmp_path).as_posix().endswith(
        "hfrvla_chunk32_alpha_0p5_resclip_none/libero_object_5ep_seed42_cuda"
    )


def test_build_specs_sets_plan_exec_to_same_chunk_size(tmp_path):
    hfrvla = tmp_path / "hfrvla"
    smolvla = tmp_path / "smolvla"
    hfrvla.mkdir()
    smolvla.mkdir()
    (hfrvla / "config.json").write_text('{"chunk_size": 50, "n_action_steps": 50}')
    (smolvla / "config.json").write_text('{"chunk_size": 50, "n_action_steps": 1}')
    args = argparse.Namespace(
        policies="hfrvla,smolvla",
        chunk_sizes="4",
        suites="libero_spatial",
        hfrvla_policy=hfrvla,
        smolvla_policy=smolvla,
        hfrvla_alpha=0.5,
        hfrvla_residual_clip_mode="config",
        hfrvla_no_clip_delta_max=999.0,
        seed=42,
        n_episodes=5,
        device="cuda",
    )

    specs = build_specs(args)

    assert [(s.policy, s.planning_chunk_size, s.n_action_steps) for s in specs] == [
        ("hfrvla", 4, 4),
        ("smolvla", 4, 4),
    ]
    assert [s.policy_config_n_action_steps for s in specs] == [50, 1]
    assert [(s.residual_clip_mode, s.eval_delta_max) for s in specs] == [
        ("config", None),
        ("", None),
    ]
