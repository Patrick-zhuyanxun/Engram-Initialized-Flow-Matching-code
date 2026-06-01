import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_hfrvla_eval import (
    EvalMetrics,
    materialize_policy_variant,
    summarize_eval_info,
)


def test_summarize_eval_info_counts_successes_from_lerobot_schema():
    info = {
        "overall": {
            "avg_sum_reward": 0.7,
            "avg_max_reward": 0.7,
            "pc_success": 70.0,
            "n_episodes": 20,
        },
        "per_task": [
            {
                "task_group": "libero_spatial",
                "task_id": 0,
                "metrics": {"successes": [True, False, True]},
            }
        ],
    }

    metrics = summarize_eval_info(info)

    assert metrics == EvalMetrics(
        pc_success=70.0,
        avg_sum_reward=0.7,
        avg_max_reward=0.7,
        n_episodes=20,
        n_successes=2,
    )


def test_materialize_policy_variant_overrides_config_without_mutating_source(tmp_path):
    source = tmp_path / "source_policy"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "delta_max": 0.2,
                "inference_disable_fast": False,
                "safety_joint_velocity_limit": 2.0,
            }
        )
    )
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "policy_preprocessor.json").write_text("{}")

    dest = tmp_path / "delta0_policy"
    materialize_policy_variant(
        source,
        dest,
        {
            "delta_max": 0.0,
            "inference_disable_fast": True,
        },
    )

    assert json.loads((source / "config.json").read_text())["delta_max"] == 0.2
    dest_config = json.loads((dest / "config.json").read_text())
    assert dest_config["delta_max"] == 0.0
    assert dest_config["inference_disable_fast"] is True
    assert (dest / "model.safetensors").read_bytes() == b"weights"
    assert (dest / "policy_preprocessor.json").read_text() == "{}"
