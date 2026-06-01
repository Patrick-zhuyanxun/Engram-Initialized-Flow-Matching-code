"""Shared helpers for HFRVLA slow-planner / eval alignment checks."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

LIBERO_REQUIRED_INPUT_SHAPES = {
    "observation.images.image": (3, 256, 256),
    "observation.images.image2": (3, 256, 256),
    "observation.state": (8,),
}
LIBERO_REQUIRED_OUTPUT_SHAPES = {
    "action": (7,),
}
DEFAULT_LIBERO_SMOLVLA = "HuggingFaceVLA/smolvla_libero"


def load_policy_config_json(repo_id_or_path: str | Path) -> dict[str, Any]:
    """Load a policy config.json from a local checkpoint path or HF repo id."""
    local = Path(repo_id_or_path).expanduser()
    if local.is_dir():
        cfg_path = local / "config.json"
    else:
        from huggingface_hub import hf_hub_download

        cfg_path = Path(hf_hub_download(str(repo_id_or_path), filename="config.json"))
    with open(cfg_path) as f:
        return json.load(f)


def _shape_of(features: dict[str, Any], key: str) -> tuple[int, ...] | None:
    raw = features.get(key)
    if not isinstance(raw, dict) or "shape" not in raw:
        return None
    return tuple(int(x) for x in raw["shape"])


def _collect_feature_mismatches(
    raw_config: dict[str, Any],
    *,
    required_inputs: dict[str, tuple[int, ...]] = LIBERO_REQUIRED_INPUT_SHAPES,
    required_outputs: dict[str, tuple[int, ...]] = LIBERO_REQUIRED_OUTPUT_SHAPES,
) -> list[str]:
    mismatches: list[str] = []
    input_features = raw_config.get("input_features") or {}
    output_features = raw_config.get("output_features") or {}

    for key, expected in required_inputs.items():
        actual = _shape_of(input_features, key)
        if actual != expected:
            got = "missing" if actual is None else actual
            mismatches.append(f"{key}: expected {expected}, got {got}")

    for key, expected in required_outputs.items():
        actual = _shape_of(output_features, key)
        if actual != expected:
            got = "missing" if actual is None else actual
            mismatches.append(f"{key}: expected {expected}, got {got}")

    return mismatches


def validate_libero_slow_planner_config(raw_config: dict[str, Any], *, source: str) -> None:
    """Reject slow-planner checkpoints that are not already LIBERO-adapted.

    HFRVLA trains a residual on top of the frozen slow planner. If the slow
    planner itself was never adapted to LIBERO's observation/action contract,
    alignment eval can still run but the resulting actions are not meaningful.
    """
    mismatches = _collect_feature_mismatches(raw_config)
    if not mismatches:
        return

    details = "\n  - ".join(mismatches)
    raise ValueError(
        f"Slow planner `{source}` is not LIBERO-compatible.\n"
        f"  - {details}\n"
        f"Use `{DEFAULT_LIBERO_SMOLVLA}` or another SmolVLA checkpoint already "
        "fine-tuned/adapted on HuggingFaceVLA/libero for HFRVLA recording, "
        "packaging, and alignment. The raw `lerobot/smolvla_base` checkpoint is "
        "a warm-start model, not a LIBERO slow planner; if you change the slow "
        "planner, recollect the HFRVLA dataset so `a_base`, `z_goal`, and "
        "`z_phase` come from the same policy."
    )


def warn_or_validate_libero_slow_planner(
    raw_config: dict[str, Any],
    *,
    source: str,
    allow_feature_remap: bool,
) -> None:
    """Validate by default; print a warning when remapping is explicitly allowed."""
    try:
        validate_libero_slow_planner_config(raw_config, source=source)
    except ValueError as exc:
        if not allow_feature_remap:
            raise
        print(f"[hfrvla] WARNING: {exc}", flush=True)


def _first_present(dct: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in dct and dct[key] is not None:
            return dct[key]
    return None


def _metrics_from_per_task(per_task: list[dict[str, Any]]) -> dict[str, Any]:
    sum_rewards: list[float] = []
    max_rewards: list[float] = []
    successes: list[bool] = []
    video_paths: list[str] = []

    for item in per_task:
        metrics = item.get("metrics", {})
        sum_rewards.extend(float(x) for x in metrics.get("sum_rewards", []))
        max_rewards.extend(float(x) for x in metrics.get("max_rewards", []))
        successes.extend(bool(x) for x in metrics.get("successes", []))
        video_paths.extend(str(x) for x in metrics.get("video_paths", []))

    n_episodes = len(successes) or len(sum_rewards) or len(max_rewards)
    return {
        "avg_sum_reward": mean(sum_rewards) if sum_rewards else None,
        "avg_max_reward": mean(max_rewards) if max_rewards else None,
        "pc_success": (sum(successes) / len(successes)) if successes else None,
        "n_episodes": n_episodes,
        "video_paths": video_paths,
    }


def extract_eval_metrics(info: dict[str, Any]) -> dict[str, Any]:
    """Normalize old/new lerobot-eval JSON schemas to one metrics dict."""
    if "aggregated" in info and isinstance(info["aggregated"], dict):
        return extract_eval_metrics(info["aggregated"])

    if "overall" in info and isinstance(info["overall"], dict):
        return dict(info["overall"])

    if "per_task" in info and isinstance(info["per_task"], list):
        return _metrics_from_per_task(info["per_task"])

    pc_success = _first_present(info, "pc_success", "success_rate", "avg_max_reward")
    return {
        "avg_sum_reward": _first_present(info, "avg_sum_reward"),
        "avg_max_reward": _first_present(info, "avg_max_reward"),
        "pc_success": pc_success,
        "n_episodes": _first_present(info, "n_episodes"),
        "video_paths": _first_present(info, "video_paths") or [],
    }


def success_metric_to_fraction(value: Any) -> float | None:
    """Convert LeRobot success metrics to [0, 1].

    LeRobot versions are inconsistent here: some report ``pc_success`` as a
    fraction (0.8), while current LIBERO eval reports a percent (80.0). Treat
    values greater than 1 as percentages so aggregate reporting does not multiply
    by 100 twice.
    """
    if not isinstance(value, (int, float)):
        return None
    fraction = float(value)
    if fraction > 1.0:
        fraction = fraction / 100.0
    return fraction


def format_success_metric(value: Any) -> str:
    """Format a LeRobot success metric as a percent string."""
    fraction = success_metric_to_fraction(value)
    if fraction is None:
        return "?"
    return f"{fraction * 100:.1f}%"
