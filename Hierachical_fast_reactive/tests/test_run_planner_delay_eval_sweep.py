import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_planner_delay_eval_sweep import (  # noqa: E402
    DEFAULT_DELAYS,
    DEFAULT_POLICIES,
    DelayEvalSpec,
    build_eval_command,
    env_overrides_for_spec,
    output_dir_for_spec,
)


def test_default_delay_grid_is_async_phase1_grid():
    assert DEFAULT_DELAYS == tuple(range(5))
    assert DEFAULT_POLICIES == ("hfrvla", "hfrvla_disable_fast")


def test_hfrvla_delay_command_uses_async_timestep_policy_delay(tmp_path):
    spec = DelayEvalSpec(
        policy="hfrvla",
        policy_path=tmp_path / "hfrvla_policy",
        planner_delay_steps=4,
        n_action_steps=16,
        planning_chunk_size=50,
        suite="libero_spatial",
        alpha=0.5,
        seed=42,
        n_episodes_per_task=10,
        device="cuda",
        fallback="hold_last",
        eval_delta_max=0.2,
        eval_safety_joint_velocity_limit=2.0,
        eval_control_dt=0.1,
    )

    cmd = build_eval_command(spec, tmp_path / "lerobot-eval", tmp_path / "out")

    assert "--policy.n_action_steps=16" in cmd
    assert "--policy.planner_delay_mode=async_timestep" in cmd
    assert "--policy.planner_delay_steps=4" in cmd
    assert "--policy.planner_delay_fallback=hold_last" in cmd
    assert "--policy.async_request_interval_steps=8" in cmd
    assert "--policy.fast_residual_alpha=0.5" in cmd
    assert "--policy.delta_max=0.2" in cmd
    assert env_overrides_for_spec(spec) == {}


def test_hfrvla_async_timestep_delay_command_sets_mode_and_interval(tmp_path):
    spec = DelayEvalSpec(
        policy="hfrvla",
        policy_path=tmp_path / "hfrvla_policy",
        planner_delay_steps=4,
        n_action_steps=16,
        planning_chunk_size=50,
        suite="libero_spatial",
        alpha=0.5,
        seed=42,
        n_episodes_per_task=10,
        device="cuda",
        fallback="hold_last",
        planner_delay_mode="async_timestep",
        async_request_interval_steps=8,
    )

    cmd = build_eval_command(spec, tmp_path / "lerobot-eval", tmp_path / "out")

    assert "--policy.planner_delay_mode=async_timestep" in cmd
    assert "--policy.async_request_interval_steps=8" in cmd
    assert output_dir_for_spec(spec, tmp_path).as_posix().endswith(
        "hfrvla_plan50_exec16_delay4_async_timestep_N8_alpha_0p5/"
        "libero_spatial_10ep_seed42_cuda"
    )


def test_delay_command_can_limit_libero_task_ids_for_smoke(tmp_path):
    spec = DelayEvalSpec(
        policy="hfrvla",
        policy_path=tmp_path / "hfrvla_policy",
        planner_delay_steps=1,
        n_action_steps=16,
        planning_chunk_size=50,
        suite="libero_spatial",
        alpha=0.5,
        seed=42,
        n_episodes_per_task=1,
        device="cuda",
        fallback="hold_last",
    )

    cmd = build_eval_command(spec, tmp_path / "lerobot-eval", tmp_path / "out", task_ids="[0]")

    assert "--env.task_ids=[0]" in cmd


def test_hfrvla_disable_fast_delay_command_uses_same_policy_path(tmp_path):
    spec = DelayEvalSpec(
        policy="hfrvla_disable_fast",
        policy_path=tmp_path / "hfrvla_policy",
        planner_delay_steps=2,
        n_action_steps=16,
        planning_chunk_size=50,
        suite="libero_spatial",
        alpha=0.5,
        seed=42,
        n_episodes_per_task=10,
        device="cuda",
        fallback="hold_last",
    )

    cmd = build_eval_command(spec, tmp_path / "lerobot-eval", tmp_path / "out")

    assert f"--policy.path={tmp_path / 'hfrvla_policy'}" in cmd
    assert "--policy.planner_delay_mode=async_timestep" in cmd
    assert "--policy.planner_delay_steps=2" in cmd
    assert "--policy.inference_disable_fast=true" in cmd
    assert env_overrides_for_spec(spec) == {}


def test_delay_output_dir_includes_policy_execution_and_delay(tmp_path):
    spec = DelayEvalSpec(
        policy="hfrvla_disable_fast",
        policy_path=tmp_path / "policy",
        planner_delay_steps=4,
        n_action_steps=16,
        planning_chunk_size=50,
        suite="libero_spatial",
        alpha=0.5,
        seed=42,
        n_episodes_per_task=10,
        device="cuda",
        fallback="hold_last",
    )

    assert output_dir_for_spec(spec, tmp_path).as_posix().endswith(
        "hfrvla_disable_fast_plan50_exec16_delay4_async_timestep_N8_alpha_0p5/"
        "libero_spatial_10ep_seed42_cuda"
    )
