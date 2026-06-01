import sys
from pathlib import Path

import pytest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_plan50_chunk_correctability import (
    aggregate_metric_rows,
    budget_label,
    classify_soft_gate,
    compute_frame_metric_rows,
    parse_csv_floats,
    parse_csv_ints,
    task_indices_for_suite,
)


def test_parse_helpers_and_suite_mapping():
    assert parse_csv_floats("0.1, 0.2,,0.4") == [0.1, 0.2, 0.4]
    assert parse_csv_ints("10, 12,19") == [10, 12, 19]
    assert parse_csv_ints("") is None
    assert task_indices_for_suite("libero_spatial") == set(range(10, 20))
    assert task_indices_for_suite("all") is None
    with pytest.raises(ValueError):
        task_indices_for_suite("bad_suite")


def test_compute_frame_metric_rows_tracks_groups_and_budgets():
    base = np.zeros(7, dtype=np.float32)
    expert = np.array([0.05, 0.10, 0.30, 0.01, -0.25, 0.00, 0.50], dtype=np.float32)

    rows = compute_frame_metric_rows(
        base_action=base,
        expert_action=expert,
        budgets=[0.2, 0.4],
        episode_index=3,
        frame_index=123,
        task_index=10,
        task="put the bowl on the plate",
        k=7,
    )

    all_row = next(row for row in rows if row["group"] == "all")
    assert len(rows) == 4
    assert all_row["episode_index"] == 3
    assert all_row["k"] == 7
    assert all_row[f"clip_dim_fraction_{budget_label(0.2)}"] == pytest.approx(3 / 7)
    assert all_row[f"correctable_action_{budget_label(0.2)}"] == 0.0
    assert all_row[f"clip_dim_fraction_{budget_label(0.4)}"] == pytest.approx(1 / 7)
    assert all_row[f"correctable_action_{budget_label(0.4)}"] == 0.0

    wrist_xyz = next(row for row in rows if row["group"] == "dims_0_2")
    assert wrist_xyz[f"correctable_action_{budget_label(0.4)}"] == 1.0
    assert wrist_xyz["target_delta_norm"] == pytest.approx(np.linalg.norm(expert[:3]))


def test_aggregation_and_soft_gate_report_late_drift():
    budgets = [0.2, 0.4]
    rows = []
    for k, delta in [(1, 0.1), (30, 0.3)]:
        base = np.zeros(7, dtype=np.float32)
        expert = np.full(7, delta, dtype=np.float32)
        rows.extend(
            compute_frame_metric_rows(
                base_action=base,
                expert_action=expert,
                budgets=budgets,
                episode_index=0,
                frame_index=k,
                task_index=10,
                task="spatial task",
                k=k,
            )
        )

    per_group = aggregate_metric_rows(
        rows,
        group_by=("group",),
        budgets=budgets,
        late_start=25,
    )
    all_row = next(row for row in per_group if row["group"] == "all")
    assert all_row["n_frames"] == 2
    assert all_row[f"correctable_action_{budget_label(0.2)}"] == 0.5
    assert all_row[f"correctable_action_{budget_label(0.4)}"] == 1.0
    assert all_row["late_minus_early_mse"] > 0

    gate = classify_soft_gate(per_group, budgets=budgets)
    assert gate["status"] == "green"
    assert gate["correctable_at_0p2"] == 0.5
    assert gate["correctable_at_0p4"] == 1.0
