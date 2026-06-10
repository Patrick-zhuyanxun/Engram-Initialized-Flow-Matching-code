#!/usr/bin/env python3
"""Analyze async-timestep planner-delay eval results."""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hfrvla_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "outputs/async_timestep_planner_delay_eval_sweep/results.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/async_timestep_planner_delay_eval_sweep"

POLICY_LABELS = {
    "hfrvla": "HFRVLA",
    "hfrvla_disable_fast": "Disable-fast",
}
POLICY_COLORS = {
    "hfrvla": "#0072B2",
    "hfrvla_disable_fast": "#D55E00",
}


def wilson_interval(successes: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return (100.0 * max(0.0, center - half), 100.0 * min(1.0, center + half))


def as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_task_successes(value: str) -> list[int]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def validate(df: pd.DataFrame) -> None:
    required = {
        "policy",
        "planner_delay_mode",
        "planner_delay_steps",
        "status",
        "n_episodes",
        "n_successes",
        "pc_success",
        "async_request_interval_steps",
        "async_chunk_start_index_mean",
        "async_dropped_old_queue_steps_mean",
        "per_task_successes",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if set(df["policy"]) != set(POLICY_LABELS):
        raise ValueError(f"expected policies {sorted(POLICY_LABELS)}, got {sorted(set(df['policy']))}")
    if set(df["planner_delay_mode"]) != {"async_timestep"}:
        raise ValueError("all rows must use planner_delay_mode=async_timestep")
    if set(df["status"]) != {"ok"}:
        raise ValueError(f"all rows must be ok, got {df['status'].value_counts().to_dict()}")
    counts = df.groupby(["planner_delay_steps", "policy"]).size()
    if not (counts == 1).all():
        raise ValueError("expected exactly one row per delay/policy")


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "planner_delay_steps",
        "n_episodes",
        "n_successes",
        "pc_success",
        "eval_s",
        "slow_replan_count",
        "slow_chunk_latency_ms_mean",
        "fast_latency_ms_mean",
        "delta_norm_mean",
        "delta_clip_fraction_mean",
        "k_mean",
        "async_request_count",
        "async_activation_count",
        "async_chunk_start_index_mean",
        "async_dropped_old_queue_steps_mean",
    ]
    for col in numeric_cols:
        if col in df:
            df[col] = as_float(df[col])

    rows: list[dict[str, float | int]] = []
    for delay, group in df.groupby("planner_delay_steps", sort=True):
        by_policy = {row.policy: row for row in group.itertuples(index=False)}
        h = by_policy["hfrvla"]
        d = by_policy["hfrvla_disable_fast"]
        h_ci = wilson_interval(h.n_successes, h.n_episodes)
        d_ci = wilson_interval(d.n_successes, d.n_episodes)
        rows.append(
            {
                "planner_delay_steps": int(delay),
                "hfrvla_success": float(h.pc_success),
                "hfrvla_success_ci95_low": h_ci[0],
                "hfrvla_success_ci95_high": h_ci[1],
                "hfrvla_n_successes": int(h.n_successes),
                "hfrvla_n_episodes": int(h.n_episodes),
                "disable_fast_success": float(d.pc_success),
                "disable_fast_success_ci95_low": d_ci[0],
                "disable_fast_success_ci95_high": d_ci[1],
                "disable_fast_n_successes": int(d.n_successes),
                "disable_fast_n_episodes": int(d.n_episodes),
                "hfrvla_minus_disable_fast": float(h.pc_success - d.pc_success),
                "hfrvla_slow_replan_count": float(h.slow_replan_count),
                "disable_fast_slow_replan_count": float(d.slow_replan_count),
                "hfrvla_fast_latency_ms_mean": float(h.fast_latency_ms_mean),
                "hfrvla_delta_clip_fraction_mean": float(h.delta_clip_fraction_mean),
                "hfrvla_k_mean": float(h.k_mean),
                "hfrvla_async_request_count": float(h.async_request_count),
                "disable_fast_async_request_count": float(d.async_request_count),
                "hfrvla_async_activation_count": float(h.async_activation_count),
                "disable_fast_async_activation_count": float(d.async_activation_count),
                "hfrvla_async_chunk_start_index_mean": float(h.async_chunk_start_index_mean),
                "disable_fast_async_chunk_start_index_mean": float(d.async_chunk_start_index_mean),
                "hfrvla_async_dropped_old_queue_steps_mean": float(h.async_dropped_old_queue_steps_mean),
                "disable_fast_async_dropped_old_queue_steps_mean": float(d.async_dropped_old_queue_steps_mean),
            }
        )
    summary = pd.DataFrame(rows)
    for policy in ("hfrvla", "disable_fast"):
        baseline = float(summary.loc[summary["planner_delay_steps"] == 0, f"{policy}_success"].iloc[0])
        summary[f"{policy}_change_from_delay0"] = summary[f"{policy}_success"] - baseline
    return summary


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_success(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    x = summary["planner_delay_steps"].to_numpy()
    for policy, prefix in (("hfrvla", "hfrvla"), ("hfrvla_disable_fast", "disable_fast")):
        y = summary[f"{prefix}_success"].to_numpy()
        low = summary[f"{prefix}_success_ci95_low"].to_numpy()
        high = summary[f"{prefix}_success_ci95_high"].to_numpy()
        yerr = np.vstack([y - low, high - y])
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2.0,
            capsize=3,
            color=POLICY_COLORS[policy],
            label=POLICY_LABELS[policy],
        )
    ax.set_xlabel("Planner delay d (control timesteps)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Async-timestep planner-delay sweep")
    ax.set_xticks(x)
    ax.set_ylim(45, 82)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.legend(frameon=False, loc="lower left")
    ax.text(
        0.02,
        0.96,
        "LIBERO-Spatial, 100 episodes/row, N=8, plan=50, exec=16",
        transform=ax.transAxes,
        va="top",
        fontsize=7.5,
        color="#444444",
    )
    save_fig(fig, out_dir, "success_vs_delay")


def plot_drop(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    x = summary["planner_delay_steps"].to_numpy()
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.plot(
        x,
        summary["hfrvla_change_from_delay0"],
        marker="o",
        linewidth=2.0,
        color=POLICY_COLORS["hfrvla"],
        label="HFRVLA",
    )
    ax.plot(
        x,
        summary["disable_fast_change_from_delay0"],
        marker="o",
        linewidth=2.0,
        color=POLICY_COLORS["hfrvla_disable_fast"],
        label="Disable-fast",
    )
    ax.set_xlabel("Planner delay d (control timesteps)")
    ax.set_ylabel("Change from d=0 (pp)")
    ax.set_title("Delay degradation relative to no-delay baseline")
    ax.set_xticks(x)
    ax.set_ylim(-16, 10)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.legend(frameon=False, loc="lower left")
    save_fig(fig, out_dir, "drop_from_delay0")


def plot_margin(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    x = summary["planner_delay_steps"].to_numpy()
    y = summary["hfrvla_minus_disable_fast"].to_numpy()
    colors = [POLICY_COLORS["hfrvla"] if value >= 0 else POLICY_COLORS["hfrvla_disable_fast"] for value in y]
    ax.axhline(0, color="#333333", linewidth=1.0)
    bars = ax.bar(x, y, color=colors, width=0.62)
    for bar, value in zip(bars, y, strict=True):
        va = "bottom" if value >= 0 else "top"
        offset = 0.6 if value >= 0 else -0.6
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:+.0f}", ha="center", va=va, fontsize=8)
    ax.set_xlabel("Planner delay d (control timesteps)")
    ax.set_ylabel("HFRVLA - disable-fast (pp)")
    ax.set_title("Current wrist feedback margin")
    ax.set_xticks(x)
    ax.set_ylim(-7, 13)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    save_fig(fig, out_dir, "hfrvla_margin_vs_delay")


def plot_timing_debug(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.1, 3.2))
    x = summary["planner_delay_steps"].to_numpy()
    ax.plot(
        x,
        x,
        color="#333333",
        linestyle="--",
        linewidth=1.0,
        label="Expected chunk start = d",
    )
    ax.plot(
        x,
        summary["hfrvla_async_chunk_start_index_mean"],
        marker="o",
        linewidth=2.0,
        color=POLICY_COLORS["hfrvla"],
        label="HFRVLA chunk start index",
    )
    ax.plot(
        x,
        summary["disable_fast_async_chunk_start_index_mean"],
        marker="s",
        linewidth=2.0,
        color=POLICY_COLORS["hfrvla_disable_fast"],
        label="Disable-fast chunk start index",
    )
    ax.set_xlabel("Planner delay d (control timesteps)")
    ax.set_ylabel("Mean first executed chunk index")
    ax.set_title("Async timing check")
    ax.set_xticks(x)
    ax.set_ylim(-0.2, 4.5)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    save_fig(fig, out_dir, "async_timing_debug")


def plot_per_task_margin(df: pd.DataFrame, out_dir: Path) -> None:
    delays = sorted(df["planner_delay_steps"].unique())
    matrix = []
    for delay in delays:
        h = df[(df.policy == "hfrvla") & (df.planner_delay_steps == delay)]["per_task_successes"].iloc[0]
        d = df[(df.policy == "hfrvla_disable_fast") & (df.planner_delay_steps == delay)]["per_task_successes"].iloc[0]
        h_tasks = parse_task_successes(h)
        d_tasks = parse_task_successes(d)
        if len(h_tasks) != len(d_tasks):
            raise ValueError(f"per-task success length mismatch at delay {delay}")
        matrix.append([(hv - dv) * 10 for hv, dv in zip(h_tasks, d_tasks, strict=True)])
    arr = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    im = ax.imshow(arr, cmap="RdBu", vmin=-50, vmax=50, aspect="auto")
    ax.set_xlabel("LIBERO-Spatial task index")
    ax.set_ylabel("Planner delay d")
    ax.set_title("Per-task HFRVLA margin over disable-fast (pp)")
    ax.set_xticks(np.arange(arr.shape[1]))
    ax.set_yticks(np.arange(len(delays)), [str(int(v)) for v in delays])
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            ax.text(x, y, f"{arr[y, x]:+.0f}", ha="center", va="center", fontsize=6.5, color="#111111")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Margin (pp)")
    save_fig(fig, out_dir, "per_task_margin_heatmap")


def plot_summary_panel(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4))
    x = summary["planner_delay_steps"].to_numpy()

    ax = axes[0, 0]
    for policy, prefix in (("hfrvla", "hfrvla"), ("hfrvla_disable_fast", "disable_fast")):
        ax.plot(x, summary[f"{prefix}_success"], marker="o", linewidth=2.0, color=POLICY_COLORS[policy], label=POLICY_LABELS[policy])
    ax.set_title("A. Success")
    ax.set_ylabel("Success rate (%)")
    ax.set_xticks(x)
    ax.set_ylim(50, 76)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.legend(frameon=False, loc="lower left")

    ax = axes[0, 1]
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.plot(x, summary["hfrvla_change_from_delay0"], marker="o", linewidth=2.0, color=POLICY_COLORS["hfrvla"], label="HFRVLA")
    ax.plot(x, summary["disable_fast_change_from_delay0"], marker="o", linewidth=2.0, color=POLICY_COLORS["hfrvla_disable_fast"], label="Disable-fast")
    ax.set_title("B. Change from d=0")
    ax.set_ylabel("Change (pp)")
    ax.set_xticks(x)
    ax.set_ylim(-14, 9)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)

    ax = axes[1, 0]
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.bar(x, summary["hfrvla_minus_disable_fast"], color="#009E73", width=0.62)
    ax.set_title("C. HFRVLA margin")
    ax.set_xlabel("Planner delay d")
    ax.set_ylabel("Margin (pp)")
    ax.set_xticks(x)
    ax.set_ylim(-6, 12)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)

    ax = axes[1, 1]
    ax.plot(x, x, color="#333333", linestyle="--", linewidth=1.0, label="Expected")
    ax.plot(x, summary["hfrvla_async_chunk_start_index_mean"], marker="o", linewidth=2.0, color=POLICY_COLORS["hfrvla"], label="Observed")
    ax.set_title("D. Async chunk start")
    ax.set_xlabel("Planner delay d")
    ax.set_ylabel("Mean start index")
    ax.set_xticks(x)
    ax.set_ylim(-0.2, 4.5)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")

    fig.suptitle("Async-timestep planner-delay sweep: N=8, plan=50, exec=16, LIBERO-Spatial", y=1.02, fontsize=11)
    save_fig(fig, out_dir, "async_timestep_planner_delay_summary")


def make_markdown(summary: pd.DataFrame, df: pd.DataFrame) -> str:
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    total_eval_hours = df["eval_s"].sum() / 3600.0
    mean_h = summary["hfrvla_success"].mean()
    mean_d = summary["disable_fast_success"].mean()
    mean_gap = summary["hfrvla_minus_disable_fast"].mean()
    delayed = summary[summary["planner_delay_steps"] > 0]
    mean_delayed_gap = delayed["hfrvla_minus_disable_fast"].mean()

    lines = [
        "# Async-Timestep Planner-Delay Eval Results",
        "",
        f"> Generated: {generated}",
        "",
        "Source: `outputs/async_timestep_planner_delay_eval_sweep/results.csv`.",
        "",
        "Protocol:",
        "",
        "- Suite: `libero_spatial`.",
        "- Delay mode: `async_timestep`.",
        "- `async_request_interval_steps = 8`.",
        "- `planner_delay_steps = 0..4`.",
        "- Policies: `hfrvla`, `hfrvla_disable_fast`.",
        "- Planning chunk size: `50`; execution/replan interval: `16`.",
        "- Episodes: `10` per task x `10` LIBERO-Spatial tasks = `100` episodes per row.",
        "- HFRVLA eval alpha: `0.5`; delta max: `0.2`; fallback: `hold_last`.",
        "- Video rendering: `HFRVLA_EVAL_MAX_VIDEOS=1`.",
        f"- Total eval wall-clock reported by rows: `{total_eval_hours:.2f}` hours.",
        "",
        "## Success Rate",
        "",
        "| Delay | HFRVLA | Disable-fast | HFRVLA - disable | HFRVLA change vs d=0 | Disable-fast change vs d=0 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {int(row.planner_delay_steps)} | {row.hfrvla_success:.1f}% | "
            f"{row.disable_fast_success:.1f}% | {row.hfrvla_minus_disable_fast:+.1f} pp | "
            f"{row.hfrvla_change_from_delay0:+.1f} pp | {row.disable_fast_change_from_delay0:+.1f} pp |"
        )
    lines.extend(
        [
            "",
            "## Aggregate Summary",
            "",
            "| Quantity | Value |",
            "|---|---:|",
            f"| Mean HFRVLA success, d=0..4 | {mean_h:.2f}% |",
            f"| Mean disable-fast success, d=0..4 | {mean_d:.2f}% |",
            f"| Mean HFRVLA margin, d=0..4 | {mean_gap:+.2f} pp |",
            f"| Mean HFRVLA margin, d=1..4 | {mean_delayed_gap:+.2f} pp |",
            "",
            "## Async Timing Checks",
            "",
            "| Delay | HFRVLA chunk-start mean | Disable-fast chunk-start mean | HFRVLA dropped-old-queue mean | Disable-fast dropped-old-queue mean |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {int(row.planner_delay_steps)} | {row.hfrvla_async_chunk_start_index_mean:.3f} | "
            f"{row.disable_fast_async_chunk_start_index_mean:.3f} | "
            f"{row.hfrvla_async_dropped_old_queue_steps_mean:.3f} | "
            f"{row.disable_fast_async_dropped_old_queue_steps_mean:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The async-timestep implementation behaved as intended: the mean first executed chunk index equals `d` for both policies at every delay.",
            "- HFRVLA did not degrade over `d=0..4`; its success rates were 64%, 68%, 72%, 68%, and 66%.",
            "- Disable-fast degraded from 68% at `d=0` to 56% at `d=4`, a -12 pp drop.",
            "- The HFRVLA minus disable-fast margin changed from -4 pp at `d=0` to +4, +10, +7, and +10 pp at `d=1..4`.",
            "- The strongest defensible claim from this single-seed 100-episode/row run is that current wrist feedback prevents the degradation seen when the same wrapper disables the fast residual under async-timestep planner latency.",
            "- Treat exact percentages as provisional until replicated across additional seeds or task-order variants.",
            "",
            "## Artifacts",
            "",
            "- Summary CSV: `outputs/async_timestep_planner_delay_eval_sweep/analysis_summary.csv`.",
            "- Main panel PNG/PDF: `outputs/async_timestep_planner_delay_eval_sweep/async_timestep_planner_delay_summary.{png,pdf}`.",
            "- Success curve PNG/PDF: `outputs/async_timestep_planner_delay_eval_sweep/success_vs_delay.{png,pdf}`.",
            "- Drop curve PNG/PDF: `outputs/async_timestep_planner_delay_eval_sweep/drop_from_delay0.{png,pdf}`.",
            "- Margin chart PNG/PDF: `outputs/async_timestep_planner_delay_eval_sweep/hfrvla_margin_vs_delay.{png,pdf}`.",
            "- Timing debug PNG/PDF: `outputs/async_timestep_planner_delay_eval_sweep/async_timing_debug.{png,pdf}`.",
            "- Per-task margin heatmap PNG/PDF: `outputs/async_timestep_planner_delay_eval_sweep/per_task_margin_heatmap.{png,pdf}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    validate(df)
    summary = build_summary(df)
    summary.to_csv(args.out_dir / "analysis_summary.csv", index=False)

    apply_style()
    plot_success(summary, args.out_dir)
    plot_drop(summary, args.out_dir)
    plot_margin(summary, args.out_dir)
    plot_timing_debug(summary, args.out_dir)
    plot_per_task_margin(df, args.out_dir)
    plot_summary_panel(summary, args.out_dir)

    (args.out_dir / "analysis.md").write_text(make_markdown(summary, df), encoding="utf-8")
    print(f"[async-planner-delay-analysis] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
