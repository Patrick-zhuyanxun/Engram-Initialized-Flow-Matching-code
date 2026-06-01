"""Regression tests for HFRVLA alignment/packaging guardrails."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.hfrvla_alignment_utils import (
    DEFAULT_LIBERO_SMOLVLA,
    extract_eval_metrics,
    format_success_metric,
    success_metric_to_fraction,
    validate_libero_slow_planner_config,
)


def _generic_smolvla_config() -> dict:
    return {
        "type": "smolvla",
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [6]},
            "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]},
            "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]},
            "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]},
        },
        "output_features": {
            "action": {"type": "ACTION", "shape": [6]},
        },
    }


def _libero_smolvla_config() -> dict:
    return {
        "type": "smolvla",
        "input_features": {
            "observation.images.image": {"type": "VISUAL", "shape": [3, 256, 256]},
            "observation.images.image2": {"type": "VISUAL", "shape": [3, 256, 256]},
            "observation.state": {"type": "STATE", "shape": [8]},
        },
        "output_features": {
            "action": {"type": "ACTION", "shape": [7]},
        },
    }


def test_rejects_generic_smolvla_base_for_libero_alignment():
    with pytest.raises(ValueError, match="not LIBERO-compatible"):
        validate_libero_slow_planner_config(_generic_smolvla_config(), source="local-smolvla-base")


def test_accepts_libero_adapted_smolvla_slow_planner():
    validate_libero_slow_planner_config(_libero_smolvla_config(), source="local-libero-smolvla")


def test_default_slow_planner_is_published_libero_smolvla():
    assert DEFAULT_LIBERO_SMOLVLA == "HuggingFaceVLA/smolvla_libero"


def test_extract_eval_metrics_handles_current_lerobot_schema_and_zero_success():
    info = {
        "overall": {
            "avg_sum_reward": 0.0,
            "avg_max_reward": 0.0,
            "pc_success": 0.0,
            "n_episodes": 5,
        }
    }

    metrics = extract_eval_metrics(info)

    assert metrics["pc_success"] == 0.0
    assert metrics["avg_sum_reward"] == 0.0
    assert metrics["n_episodes"] == 5


def test_success_metric_helpers_accept_fraction_or_percent_schema():
    assert success_metric_to_fraction(0.8) == pytest.approx(0.8)
    assert success_metric_to_fraction(80.0) == pytest.approx(0.8)
    assert success_metric_to_fraction(100.0) == pytest.approx(1.0)
    assert format_success_metric(80.0) == "80.0%"
    assert format_success_metric(0.8) == "80.0%"
