#!/usr/bin/env python3
"""Write a Markdown summary for the FWR-v2 10x10 plan/exec sweeps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = REPO_ROOT / "experiments/eval_registry/eval_results_master.csv"
DEFAULT_SUMMARY = REPO_ROOT / "outputs/fwr_10x10_eval/summary.md"

NEW_SWEEPS = {
    "fwr_action_steps_10x10": {
        "title": "Plan=50, Exec/Replan=K",
        "reference_sweep": "action_steps_eval_sweep",
        "k_field": "execution_chunk_size",
    },
    "fwr_chunk_size_10x10": {
        "title": "Plan=Exec=Replan=K",
        "reference_sweep": "chunk_size_eval_sweep",
        "k_field": "execution_chunk_size",
    },
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def success_text(row: dict[str, str] | None) -> str:
    if row is None:
        return "-"
    status = row.get("status", "")
    n_successes = row.get("n_successes", "")
    n_episodes = row.get("n_episodes", "")
    pc_success = row.get("pc_success", "")
    if not n_successes or not n_episodes or not pc_success:
        return status or "-"
    return f"{n_successes}/{n_episodes} = {float(pc_success):.1f}%"


def row_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str, str], dict[str, str]]:
    index: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            row.get("sweep_id", ""),
            row.get("policy", ""),
            row.get("suite", ""),
            row.get("planning_chunk_size", ""),
            row.get("execution_chunk_size", ""),
        )
        index[key] = row
    return index


def k_values(rows: list[dict[str, str]], sweep_id: str) -> list[int]:
    values: set[int] = set()
    for row in rows:
        if row.get("sweep_id") != sweep_id or row.get("suite") != "combined":
            continue
        raw = row.get("execution_chunk_size", "")
        if raw:
            values.add(int(raw))
    return sorted(values)


def render_sweep(
    rows: list[dict[str, str]],
    index: dict[tuple[str, str, str, str, str], dict[str, str]],
    sweep_id: str,
) -> list[str]:
    profile = NEW_SWEEPS[sweep_id]
    ref_sweep = profile["reference_sweep"]
    lines = [
        f"## {profile['title']}",
        "",
        "| K | New spatial 10x10 | New object 10x10 | New combined | Old HFRVLA ref | Old SmolVLA ref |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for k in k_values(rows, sweep_id):
        planning = "50" if sweep_id == "fwr_action_steps_10x10" else str(k)
        execution = str(k)
        new_spatial = index.get((sweep_id, "hfrvla", "libero_spatial", planning, execution))
        new_object = index.get((sweep_id, "hfrvla", "libero_object", planning, execution))
        new_combined = index.get((sweep_id, "hfrvla", "combined", planning, execution))
        ref_hfrvla = index.get((ref_sweep, "hfrvla", "combined", planning, execution))
        ref_smolvla = index.get((ref_sweep, "smolvla", "combined", planning, execution))
        lines.append(
            f"| {k} | {success_text(new_spatial)} | {success_text(new_object)} | "
            f"{success_text(new_combined)} | {success_text(ref_hfrvla)} | {success_text(ref_smolvla)} |"
        )
    return lines


def write_summary(master: Path, summary: Path) -> None:
    rows = read_rows(master)
    index = row_index(rows)
    lines = [
        "# FWR-v2 10x10 Plan/Exec Sweep Summary",
        "",
        "- New policy: `hfrvla_fwr_chunk_seq2_b512_50k_packaged_cuda`.",
        "- Eval alpha: `0.5`; seed: `42`; suites: `libero_spatial`, `libero_object`.",
        "- New rows use 10 tasks x 10 episodes per suite; combined rows use 200 episodes.",
        "- Old HFRVLA/SmolVLA references are existing registry rows with 5 episodes/task, so they are not strict same-count baselines.",
        "",
    ]
    for sweep_id in NEW_SWEEPS:
        lines.extend(render_sweep(rows, index, sweep_id))
        lines.append("")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("\n".join(lines).rstrip() + "\n")
    print(f"[fwr-10x10-summary] wrote {summary}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    write_summary(args.master, args.summary)


if __name__ == "__main__":
    main()
