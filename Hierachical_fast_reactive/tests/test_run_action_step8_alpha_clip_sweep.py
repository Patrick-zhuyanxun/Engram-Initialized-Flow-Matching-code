import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_action_step8_alpha_clip_sweep import (  # noqa: E402
    build_specs,
    output_dir_for_spec,
)
from scripts.run_action_steps_eval_sweep import build_eval_command  # noqa: E402


def make_args(tmp_path, *, alphas="0.5,1.0", delta_maxes="0.2,0.4"):
    policy_path = tmp_path / "hfrvla_policy"
    policy_path.mkdir()
    (policy_path / "config.json").write_text(json.dumps({"chunk_size": 50, "n_action_steps": 50}))
    return argparse.Namespace(
        alphas=alphas,
        delta_maxes=delta_maxes,
        suites="libero_spatial",
        hfrvla_policy=policy_path,
        n_action_steps=8,
        seed=42,
        n_episodes=5,
        device="cuda",
        control_dt=0.1,
    )


def test_build_specs_creates_alpha_delta_grid_and_scales_safety_limit(tmp_path):
    specs = build_specs(make_args(tmp_path))

    assert len(specs) == 4
    assert {(spec.alpha, spec.eval_delta_max) for spec in specs} == {
        (0.5, 0.2),
        (0.5, 0.4),
        (1.0, 0.2),
        (1.0, 0.4),
    }
    assert {spec.eval_safety_joint_velocity_limit for spec in specs} == {2.0, 4.0}
    assert all(spec.n_action_steps == 8 for spec in specs)
    assert all(spec.planning_chunk_size == 50 for spec in specs)
    assert all(spec.execution_chunk_size if hasattr(spec, "execution_chunk_size") else True for spec in specs)
    assert all(spec.residual_clip_mode == "config" for spec in specs)
    assert all(spec.eval_control_dt == 0.1 for spec in specs)


def test_output_dir_for_spec_includes_alpha_and_delta(tmp_path):
    spec = build_specs(make_args(tmp_path, alphas="0.75", delta_maxes="0.3"))[0]

    assert output_dir_for_spec(spec, tmp_path).as_posix().endswith(
        "hfrvla_n8_alpha_0p75_delta_0p3/libero_spatial_5ep_seed42_cuda"
    )


def test_eval_command_sets_alpha_and_effective_clip_budget(tmp_path):
    spec = build_specs(make_args(tmp_path, alphas="0.75", delta_maxes="0.4"))[0]

    cmd = build_eval_command(spec, tmp_path / "lerobot-eval", tmp_path / "out")

    assert "--policy.n_action_steps=8" in cmd
    assert "--policy.fast_residual_alpha=0.75" in cmd
    assert not any(part.startswith("--policy.a2c2_alpha=") for part in cmd)
    assert "--policy.delta_max=0.4" in cmd
    assert "--policy.safety_joint_velocity_limit=4.0" in cmd
    assert "--policy.control_dt=0.1" in cmd
