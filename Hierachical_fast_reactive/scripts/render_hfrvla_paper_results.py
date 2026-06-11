#!/usr/bin/env python3
"""Render HFRVLA paper figures from the eval registry.

The script intentionally avoids pandas because some local environments have
incompatible pandas/numpy wheels. It reads the registry CSV directly and emits
both PDF and PNG assets for the bilingual paper sources.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hfrvla_matplotlib")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = REPO_ROOT / "experiments/eval_registry/eval_results_master.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "paper/src/figures"

OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "sky": "#56B4E9",
    "black": "#222222",
    "gray": "#777777",
}

POLICY_LABEL = {
    "hfrvla": "HFRVLA",
    "smolvla": "SmolVLA",
    "hfrvla_disable_fast": "Disable-fast",
}

POLICY_COLOR = {
    "hfrvla": OKABE["blue"],
    "smolvla": OKABE["orange"],
    "hfrvla_disable_fast": "#555555",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    return int(float(value)) if value else 0


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else math.nan


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
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


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def filter_rows(
    rows: list[dict[str, str]],
    *,
    sweep_id: str,
    suite: str = "libero_spatial",
    policy: str | None = None,
) -> list[dict[str, str]]:
    selected = [r for r in rows if r.get("sweep_id") == sweep_id and r.get("suite") == suite]
    if policy is not None:
        selected = [r for r in selected if r.get("policy") == policy]
    return selected


def sweep_series(
    rows: list[dict[str, str]],
    *,
    sweep_id: str,
    policy: str,
    k_field: str,
    suite: str = "libero_spatial",
) -> tuple[list[int], list[float], list[int], list[int]]:
    selected = filter_rows(rows, sweep_id=sweep_id, suite=suite, policy=policy)
    series = sorted(
        (
            as_int(r, k_field),
            as_float(r, "pc_success"),
            as_int(r, "n_successes"),
            as_int(r, "n_episodes"),
        )
        for r in selected
    )
    return (
        [item[0] for item in series],
        [item[1] for item in series],
        [item[2] for item in series],
        [item[3] for item in series],
    )


def row_lookup(
    rows: list[dict[str, str]],
    *,
    sweep_id: str,
    policy: str,
    suite: str,
    k_field: str,
) -> dict[int, dict[str, str]]:
    return {as_int(r, k_field): r for r in filter_rows(rows, sweep_id=sweep_id, suite=suite, policy=policy)}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return (100.0 * max(0.0, center - half), 100.0 * min(1.0, center + half))


def box(ax, xy, w, h, text, *, fc="white", ec=OKABE["black"], lw=1.5, fs=9, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)
    return patch


def arrow(ax, start, end, *, color=OKABE["black"], lw=1.5):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def bracket(ax, start, end, y, text, *, color=OKABE["black"], fs=7.5, dy=0.13):
    ax.annotate(
        "",
        xy=(end, y),
        xytext=(start, y),
        arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.2, "shrinkA": 0, "shrinkB": 0},
    )
    ax.text((start + end) / 2, y + dy, text, ha="center", va="bottom", fontsize=fs, color=color, weight="bold")


def action_cells(ax, x, y, n, *, w=0.43, h=0.34, edge=OKABE["blue"], fills=None, labels=None):
    fills = fills or ["white"] * n
    labels = labels or [""] * n
    for i in range(n):
        ax.add_patch(Rectangle((x + i * w, y), w, h, facecolor=fills[i], edgecolor=edge, linewidth=1.0))
        if labels[i]:
            ax.text(x + i * w + w / 2, y + h / 2, labels[i], ha="center", va="center", fontsize=6.2)


def plot_problem_schematic(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.9, 4.45))
    ax.set_xlim(0, 18.8)
    ax.set_ylim(0, 6.95)
    ax.axis("off")

    ax.text(0.25, 6.58, "Gantt view of async action chunks", fontsize=11.4, weight="bold", color=OKABE["black"])
    ax.text(
        0.25,
        6.22,
        "Planning for the next chunk starts during the tail of the current chunk; execution stays back-to-back but remains open loop between switches.",
        fontsize=6.9,
        color="#444444",
    )

    # Time grid and lane labels.
    x0 = 2.75
    step = 0.52
    for i in range(30):
        fc = "#f7f7f7" if i % 2 == 0 else "white"
        ax.add_patch(Rectangle((x0 + i * step, 0.55), step, 5.35, facecolor=fc, edgecolor="#ececec", linewidth=0.35))
    arrow(ax, (x0 - 0.25, 0.5), (x0 + 30.5 * step, 0.5), color=OKABE["black"], lw=1.1)
    ax.text(x0 - 0.1, 0.28, "time", fontsize=7.3, color=OKABE["black"])

    # Timing constants. Each execution segment has eight control steps.
    observe_a = x0
    infer_a = observe_a + 0.78
    ready_a = observe_a + 2.72
    exec_a_end = ready_a + 4.16
    request_b = exec_a_end - 1.95
    infer_b = request_b + 0.62
    ready_b = exec_a_end
    exec_b_end = ready_b + 4.16
    request_c = exec_b_end - 1.95
    infer_c = request_c + 0.62
    ready_c = exec_b_end
    exec_c_end = ready_c + 3.12

    chunk_rows = [
        ("chunk A", observe_a, infer_a, ready_a, exec_a_end, 4.85, "observe\nscene", "execute A"),
        ("chunk B", request_b, infer_b, ready_b, exec_b_end, 3.55, "request\nB", "execute B"),
        ("chunk C", request_c, infer_c, ready_c, exec_c_end, 2.25, "request\nC", "future C"),
    ]

    for row_name, x_req, x_inf, x_ready, x_end, y, req_label, exec_label in chunk_rows:
        ax.text(0.16, y + 0.24, row_name, ha="left", va="center", fontsize=8.0, weight="bold", color=OKABE["black"])
        ax.text(2.05, y + 0.24, "plan", ha="right", va="center", fontsize=5.8, color=OKABE["blue"], weight="bold")
        ax.text(2.05, y - 0.18, "execute", ha="right", va="center", fontsize=5.8, color=OKABE["green"], weight="bold")

        box(ax, (x_req, y + 0.04), 0.7, 0.36, req_label, fc="#fff7d6", ec=OKABE["orange"], fs=4.8, weight="bold")
        box(ax, (x_inf, y + 0.04), x_ready - x_inf - 0.58, 0.36, "SmolVLA\ninference", fc="#dbeafe", ec=OKABE["blue"], fs=5.1, weight="bold")
        box(ax, (x_ready - 0.58, y + 0.04), 0.76, 0.36, "chunk\nready", fc="#e7f6ec", ec=OKABE["green"], fs=4.8, weight="bold")

        # Execution cells are deliberately split so the number of open-loop control steps is visible.
        n_steps = 8 if row_name != "chunk C" else 6
        cell_w = (x_end - x_ready) / n_steps
        exec_y = y - 0.54
        cell_fill = "#dff1df" if row_name != "chunk C" else "#f7fff7"
        for i in range(n_steps):
            ax.add_patch(
                Rectangle((x_ready + i * cell_w, exec_y), cell_w, 0.42, facecolor=cell_fill, edgecolor=OKABE["green"], linewidth=0.9)
            )
            if i in (0, n_steps - 1):
                ax.text(x_ready + i * cell_w + cell_w / 2, exec_y + 0.08, str(i + 1), ha="center", va="center", fontsize=4.7, color="#666666")
        ax.add_patch(Rectangle((x_ready, exec_y), x_end - x_ready, 0.42, facecolor="none", edgecolor=OKABE["green"], linewidth=1.5))
        ax.text((x_ready + x_end) / 2, exec_y + 0.26, exec_label, ha="center", va="center", fontsize=6.2, weight="bold")

        if row_name != "chunk C":
            bracket(ax, x_ready, x_end, exec_y - 0.18, "open loop: e=8 control steps", color=OKABE["green"], fs=5.7, dy=-0.18)
        else:
            ax.text((x_ready + x_end) / 2, exec_y - 0.23, "prepared next segment", ha="center", va="top", fontsize=5.5, color=OKABE["green"], weight="bold")

        ax.plot([x_ready, x_ready], [exec_y - 0.1, y + 0.55], color="#333333", linewidth=0.8)

    # Tail overlap bands connect next-chunk planning to the current chunk's execution tail.
    for xs, xe, y0_band, label_y in [(request_b, ready_b, 3.98, 4.9), (request_c, ready_c, 2.68, 3.56)]:
        ax.add_patch(Rectangle((xs, y0_band), xe - xs, 0.7, facecolor="#e8f4fb", edgecolor=OKABE["blue"], linewidth=0.7, alpha=0.65))
        ax.plot([xs, xs], [y0_band - 0.18, y0_band + 0.88], color=OKABE["blue"], linewidth=0.7, linestyle="--", alpha=0.8)
        ax.plot([xe, xe], [y0_band - 0.18, y0_band + 0.88], color=OKABE["blue"], linewidth=0.7, linestyle="--", alpha=0.8)
        ax.text(
            (xs + xe) / 2,
            label_y,
            "tail-overlap planning",
            ha="center",
            va="center",
            fontsize=5.6,
            color=OKABE["blue"],
            weight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6},
        )

    # Continuity markers between executed chunks.
    ax.plot([ready_b, ready_b], [2.95, 4.08], color=OKABE["black"], linewidth=1.0)
    ax.text(ready_b + 0.08, 3.47, "no gap", ha="left", va="bottom", fontsize=5.4, color=OKABE["black"])
    ax.plot([ready_c, ready_c], [1.65, 2.78], color=OKABE["black"], linewidth=1.0)
    ax.text(ready_c + 0.08, 2.16, "no gap", ha="left", va="bottom", fontsize=5.4, color=OKABE["black"])

    # Planned horizon preview: the planner emits a longer action horizon, but only e steps are sent before the next switch.
    action_cells(ax, ready_a, 5.42, 10, w=0.28, h=0.18, edge=OKABE["blue"], fills=["#eef6ff"] * 10)
    bracket(ax, ready_a, ready_a + 10 * 0.28, 5.74, "H", color=OKABE["blue"], fs=6.4, dy=0.04)
    bracket(ax, infer_a, ready_a, 6.02, "d", color=OKABE["red"], fs=6.4, dy=0.04)
    bracket(ax, request_b, request_c, 5.94, "K", color=OKABE["purple"], fs=6.4, dy=0.04)

    # Missing-feedback callout.
    feedback_y = 0.98
    ax.text(0.12, feedback_y + 0.22, "feedback\nstate", ha="left", va="center", fontsize=8, weight="bold", color=OKABE["red"])
    box(ax, (ready_a - 0.2, feedback_y), 2.18, 0.46, "visual feedback\nat chunk start", fc="#fff3e6", ec=OKABE["red"], fs=5.9, weight="bold")
    box(ax, (ready_a + 2.65, feedback_y), 3.2, 0.46, "middle steps use\nstale feedback", fc="#fff3e6", ec=OKABE["red"], fs=6.1, weight="bold")
    box(ax, (ready_a + 6.65, feedback_y), 2.05, 0.46, "local mismatch\ncan grow", fc="#fff3e6", ec=OKABE["red"], fs=5.8, weight="bold")
    arrow(ax, (ready_a + 1.98, feedback_y + 0.23), (ready_a + 2.65, feedback_y + 0.23), color=OKABE["red"], lw=1.0)
    arrow(ax, (ready_a + 5.85, feedback_y + 0.23), (ready_a + 6.65, feedback_y + 0.23), color=OKABE["red"], lw=1.0)
    for cx in [ready_a + 0.55, ready_a + 1.45, ready_a + 2.35, ready_a + 3.25, ready_a + 4.15, ready_a + 5.05, ready_a + 5.95, ready_a + 6.85]:
        ax.plot([cx, cx], [1.58, 1.88], color="#bbbbbb", linewidth=0.85)

    # Variable legend in human terms.
    ax.add_patch(Rectangle((14.15, 4.72), 3.75, 1.08, facecolor="white", edgecolor="#d9d9d9", linewidth=0.8))
    legend = [
        ("H", "planned action horizon"),
        ("e", "open-loop steps executed"),
        ("K", "time between replans"),
        ("d", "SmolVLA compute delay"),
    ]
    lx = 14.3
    for i, (sym, desc) in enumerate(legend):
        y = 5.54 - i * 0.22
        ax.text(lx, y, f"{sym}:", fontsize=6.1, weight="bold", color=OKABE["black"])
        ax.text(lx + 0.28, y, desc, fontsize=6.1, color="#444444")

    save_figure(fig, out_dir, "fig1_problem_schematic")


def plot_solution_schematic(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.35, 4.25))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    ax.text(0.25, 6.78, "HFRVLA architecture: slow chunk, fast wrist residual", fontsize=11.4, weight="bold", color=OKABE["black"])
    ax.text(
        0.25,
        6.45,
        "SmolVLA uses language, top/wrist RGB, and state; the fast path adds current wrist residual corrections.",
        fontsize=6.9,
        color="#444444",
    )

    # Slow low-frequency branch.
    ax.text(0.35, 5.82, "low-frequency slow planner", fontsize=6.8, color=OKABE["blue"], weight="bold")
    box(ax, (0.35, 5.05), 1.55, 0.54, "language\ninstruction", fc="#eef5fb", ec=OKABE["blue"], fs=6.4, weight="bold")
    box(ax, (0.35, 4.08), 1.72, 0.76, "top RGB\nwrist RGB\n+ state", fc="#f5f5f5", ec=OKABE["gray"], fs=6.1, weight="bold")
    box(ax, (2.55, 4.35), 2.45, 1.04, "Frozen SmolVLA\nVLM + action expert", fc="#dbeafe", ec=OKABE["blue"], fs=7.1, weight="bold")
    ax.text(2.68, 4.08, "frozen slow planner", fontsize=6.0, color=OKABE["blue"], weight="bold")
    box(ax, (5.55, 4.95), 1.85, 0.52, "base action\nchunk", fc="white", ec=OKABE["blue"], fs=6.5, weight="bold")
    box(ax, (5.55, 4.18), 1.85, 0.52, "slow context\n+ chunk index", fc="#fff7d6", ec=OKABE["orange"], fs=6.1, weight="bold")

    arrow(ax, (1.9, 5.32), (2.55, 5.05), color=OKABE["blue"], lw=1.2)
    arrow(ax, (2.07, 4.46), (2.55, 4.75), color=OKABE["gray"], lw=1.0)
    arrow(ax, (5.0, 5.03), (5.55, 5.21), color=OKABE["blue"], lw=1.2)
    arrow(ax, (5.0, 4.7), (5.55, 4.43), color=OKABE["orange"], lw=1.0)

    # Select the current base action from the queued chunk.
    action_cells(ax, 7.95, 5.03, 8, w=0.34, h=0.25, edge=OKABE["blue"], fills=["#dff1df"] + ["#f2f7fd"] * 7)
    ax.text(9.35, 5.44, "queued base actions", ha="center", fontsize=6.2, color=OKABE["blue"], weight="bold")
    box(ax, (8.45, 4.22), 1.9, 0.5, "selected\nbase action", fc="white", ec=OKABE["blue"], fs=6.4, weight="bold")
    arrow(ax, (7.4, 5.2), (7.95, 5.16), color=OKABE["blue"], lw=1.0)
    arrow(ax, (8.3, 5.03), (9.0, 4.72), color=OKABE["blue"], lw=1.0)

    # High-frequency wrist residual branch.
    ax.text(0.35, 3.18, "high-frequency wrist correction", fontsize=6.8, color=OKABE["green"], weight="bold")
    box(ax, (0.35, 2.42), 1.55, 0.58, "current\nwrist view", fc="#e7f6ec", ec=OKABE["green"], fs=6.8, weight="bold")
    box(ax, (2.35, 2.35), 2.0, 0.72, "Frozen DINOv3\nwrist patches", fc="#e7f6ec", ec=OKABE["green"], fs=6.8, weight="bold")
    ax.text(2.46, 2.14, "frozen feature extractor", fontsize=5.9, color=OKABE["green"], weight="bold")
    box(ax, (5.6, 1.32), 0.95, 0.42, "state", fc="#f5f5f5", ec=OKABE["gray"], fs=6.4, weight="bold")
    box(ax, (5.0, 2.12), 2.35, 1.12, "Trainable fast\nwrist residual\nmodule", fc="#dff3e9", ec=OKABE["green"], fs=7.1, weight="bold")

    arrow(ax, (1.9, 2.71), (2.35, 2.71), color=OKABE["green"], lw=1.2)
    arrow(ax, (4.35, 2.71), (5.0, 2.78), color=OKABE["green"], lw=1.2)
    arrow(ax, (6.07, 1.74), (6.07, 2.12), color=OKABE["gray"], lw=0.9)
    arrow(ax, (7.4, 4.43), (5.85, 3.24), color=OKABE["orange"], lw=1.0)
    arrow(ax, (8.45, 4.47), (7.35, 3.02), color=OKABE["blue"], lw=1.0)

    # Bounded residual merge.
    box(ax, (8.05, 2.42), 1.28, 0.55, "residual\ncorrection", fc="white", ec=OKABE["green"], fs=5.9, weight="bold")
    box(ax, (9.78, 2.35), 1.45, 0.68, "clip +\nscale", fc="white", ec=OKABE["green"], fs=6.3, weight="bold")
    box(ax, (11.85, 2.42), 0.58, 0.58, "+", fc="white", ec=OKABE["black"], fs=10.2, weight="bold")
    box(ax, (13.05, 2.36), 1.5, 0.68, "final\naction", fc="#f7f7f7", ec=OKABE["black"], fs=6.8, weight="bold")
    arrow(ax, (7.35, 2.68), (8.05, 2.69), color=OKABE["green"], lw=1.2)
    arrow(ax, (9.33, 2.69), (9.78, 2.69), color=OKABE["green"], lw=1.2)
    arrow(ax, (11.23, 2.69), (11.85, 2.69), color=OKABE["green"], lw=1.2)
    arrow(ax, (10.05, 4.47), (11.85, 2.86), color=OKABE["blue"], lw=1.0)
    arrow(ax, (12.43, 2.71), (13.05, 2.71), color=OKABE["black"], lw=1.2)

    # Robot and feedback loop.
    box(ax, (15.15, 2.36), 1.4, 0.68, "control\nrobot", fc="#f5f5f5", ec=OKABE["black"], fs=6.8, weight="bold")
    arrow(ax, (14.55, 2.7), (15.15, 2.7), color=OKABE["black"], lw=1.2)
    ax.annotate(
        "",
        xy=(1.1, 2.42),
        xytext=(15.85, 2.36),
        arrowprops={
            "arrowstyle": "->",
            "color": OKABE["green"],
            "lw": 1.0,
            "connectionstyle": "arc3,rad=-0.28",
            "shrinkA": 4,
            "shrinkB": 4,
        },
    )
    ax.text(8.25, 0.42, "current wrist feedback updates the correction at every control step", ha="center", fontsize=6.5, color=OKABE["green"], weight="bold")

    save_figure(fig, out_dir, "fig1_solution_schematic")


def plot_method_schematic(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Panel frames make the method/problem distinction visible in the paper.
    ax.add_patch(Rectangle((0.2, 3.55), 13.6, 5.15, fill=False, ec="#dddddd", lw=0.9))
    ax.add_patch(Rectangle((0.2, 0.25), 13.6, 2.9, fill=False, ec="#dddddd", lw=0.9))
    ax.text(0.38, 8.46, "A  HFRVLA architecture", color=OKABE["black"], fontsize=9.5, weight="bold")
    ax.text(0.38, 2.9, "B  stale-chunk problem", color=OKABE["black"], fontsize=9.5, weight="bold")

    # Architecture panel.
    box(ax, (0.45, 7.0), 1.75, 0.62, "language\n+ top view", fc="#eef5fb", ec=OKABE["blue"], fs=7.5, weight="bold")
    box(ax, (0.45, 6.1), 1.75, 0.58, "robot\nstate", fc="#f5f5f5", ec=OKABE["gray"], fs=7.5, weight="bold")
    box(ax, (2.65, 6.25), 2.0, 1.35, "System 2\nfrozen SmolVLA\nslow planner", fc="#dbeafe", ec=OKABE["blue"], fs=8.3, weight="bold")
    box(ax, (5.15, 7.12), 1.45, 0.62, "action\nchunk A_t", fc="white", ec=OKABE["blue"], fs=7.5, weight="bold")
    box(ax, (5.15, 6.05), 1.45, 0.74, "a_base\nz_goal z_phase\nk_idx", fc="#fff7d6", ec=OKABE["orange"], fs=6.9, weight="bold")

    box(ax, (0.45, 4.45), 1.75, 0.62, "current\nwrist cam", fc="#e7f6ec", ec=OKABE["green"], fs=7.6, weight="bold")
    box(ax, (2.65, 4.45), 2.0, 0.62, "DINOv3 wrist\npatches 14x14", fc="#e7f6ec", ec=OKABE["green"], fs=7.4, weight="bold")
    box(ax, (6.95, 4.85), 2.05, 1.1, "System 1\ntrainable fast\nwrist residual", fc="#dff3e9", ec=OKABE["green"], fs=8.1, weight="bold")
    box(ax, (9.65, 4.95), 1.42, 0.82, "alpha *\nclip(delta_a)", fc="white", ec=OKABE["green"], fs=7.4, weight="bold")
    box(ax, (9.65, 6.55), 1.42, 0.62, "a_base", fc="white", ec=OKABE["blue"], fs=8.0, weight="bold")
    box(ax, (11.55, 5.65), 0.55, 0.55, "+", fc="white", ec=OKABE["black"], fs=11.5, weight="bold")
    box(ax, (12.48, 5.65), 0.88, 0.55, "a_final", fc="white", ec=OKABE["black"], fs=7.8, weight="bold")
    box(
        ax,
        (8.65, 3.72),
        4.15,
        0.68,
        "a_final = a_base\n+ alpha * clip(delta_a)",
        fc="#f7f7f7",
        ec=OKABE["purple"],
        fs=7.0,
        weight="bold",
    )

    arrow(ax, (2.2, 7.31), (2.65, 7.12), color=OKABE["blue"], lw=1.4)
    arrow(ax, (2.2, 6.39), (2.65, 6.6), color=OKABE["gray"], lw=1.2)
    arrow(ax, (4.65, 7.05), (5.15, 7.42), color=OKABE["blue"], lw=1.4)
    arrow(ax, (4.65, 6.55), (5.15, 6.42), color=OKABE["blue"], lw=1.4)
    arrow(ax, (2.2, 4.76), (2.65, 4.76), color=OKABE["green"], lw=1.4)
    arrow(ax, (4.65, 4.76), (6.95, 5.22), color=OKABE["green"], lw=1.6)
    arrow(ax, (6.6, 6.42), (6.95, 5.57), color=OKABE["orange"], lw=1.3)
    arrow(ax, (6.6, 6.42), (9.65, 6.85), color=OKABE["blue"], lw=1.3)
    arrow(ax, (9.0, 5.42), (9.65, 5.36), color=OKABE["green"], lw=1.6)
    arrow(ax, (11.07, 5.36), (11.55, 5.9), color=OKABE["green"], lw=1.4)
    arrow(ax, (11.07, 6.86), (11.55, 6.13), color=OKABE["blue"], lw=1.3)
    arrow(ax, (12.1, 5.93), (12.48, 5.93), color=OKABE["black"], lw=1.4)
    arrow(ax, (13.36, 5.93), (13.64, 5.93), color=OKABE["black"], lw=1.2)
    ax.text(13.67, 5.93, "robot", va="center", fontsize=7.2, color=OKABE["black"])
    ax.text(2.66, 6.08, "frozen", color=OKABE["blue"], fontsize=6.8, weight="bold")
    ax.text(2.66, 4.25, "frozen feature extractor", color=OKABE["green"], fontsize=6.8, weight="bold")
    ax.text(6.95, 4.55, "only trainable policy block", color=OKABE["green"], fontsize=6.8, weight="bold")

    # Problem panel: planner delay and stale chunk execution.
    y = 1.32
    x0 = 1.0
    step = 0.72
    for i in range(14):
        fc = "#f7f7f7" if i % 2 == 0 else "white"
        ax.add_patch(Rectangle((x0 + i * step, 0.62), step, 1.48, fc=fc, ec="#e0e0e0", lw=0.45))
    arrow(ax, (x0 - 0.18, 0.55), (x0 + 14.2 * step, 0.55), color=OKABE["black"], lw=1.2)
    ax.text(x0, 0.35, "t", ha="center", fontsize=7.8)
    ax.text(x0 + 2 * step, 0.35, "t+d", ha="center", color=OKABE["red"], fontsize=7.8, weight="bold")
    ax.text(x0 + 8 * step, 0.35, "t+d+K", ha="center", fontsize=7.8)

    box(ax, (0.48, 2.15), 1.5, 0.44, "observe o_t", fc="#fff7d6", ec=OKABE["orange"], fs=7.0, weight="bold")
    box(ax, (2.35, 2.15), 2.05, 0.44, "slow compute\nA_t from o_t", fc="#dbeafe", ec=OKABE["blue"], fs=6.8, weight="bold")
    box(ax, (4.75, 2.15), 1.28, 0.44, "A_t ready", fc="#dff3e9", ec=OKABE["green"], fs=7.0, weight="bold")
    arrow(ax, (1.98, 2.37), (2.35, 2.37), color=OKABE["blue"], lw=1.1)
    arrow(ax, (4.4, 2.37), (4.75, 2.37), color=OKABE["blue"], lw=1.1)

    ax.annotate(
        "",
        xy=(x0 + 2 * step, 1.96),
        xytext=(x0, 1.96),
        arrowprops={"arrowstyle": "<->", "color": OKABE["red"], "lw": 1.2},
    )
    ax.text(x0 + step, 2.05, "planner delay d", ha="center", fontsize=7.1, color=OKABE["red"], weight="bold")

    # Active stale base chunk segment.
    for j in range(6):
        x = x0 + (2 + j) * step
        fc = "#f7d6d6" if j == 0 else "#dff1df"
        ax.add_patch(Rectangle((x, y), step, 0.32, fc=fc, ec=OKABE["blue"], lw=1.0))
        if j in (0, 1, 5):
            ax.text(x + step / 2, y + 0.16, f"a{j + 2}", ha="center", va="center", fontsize=6.5)
        elif j == 2:
            ax.text(x + step / 2, y + 0.16, "...", ha="center", va="center", fontsize=7.2)
    ax.text(x0 + 5 * step, y + 0.52, "execute delayed base actions", ha="center", fontsize=7.0, color=OKABE["blue"])
    ax.annotate(
        "",
        xy=(x0 + 8 * step, y - 0.16),
        xytext=(x0 + 2 * step, y - 0.16),
        arrowprops={"arrowstyle": "<->", "color": OKABE["green"], "lw": 1.2},
    )
    ax.text(x0 + 5 * step, y - 0.42, "execution/replan interval K", ha="center", fontsize=7.0, color=OKABE["green"], weight="bold")

    # Per-step residual correction row.
    for j in range(7):
        cx = x0 + (2 + j) * step + step / 2
        arrow(ax, (cx, 0.88), (cx, 1.18), color=OKABE["green"], lw=0.9)
    ax.text(8.3, 1.01, "current wrist feedback -> delta_a every control step", fontsize=7.1, color=OKABE["green"], weight="bold")
    box(ax, (9.7, 2.08), 3.45, 0.52, "stale base action is corrected, not replanned", fc="#fff3e6", ec=OKABE["red"], fs=7.2, weight="bold")
    save_figure(fig, out_dir, "fig1_method_schematic")


def plot_success_sweep(
    rows: list[dict[str, str]],
    out_dir: Path,
    *,
    sweep_id: str,
    k_field: str,
    stem: str,
    title: str,
    xlabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    all_k: list[int] = []
    for policy in ("smolvla", "hfrvla"):
        ks, ys, successes, episodes = sweep_series(rows, sweep_id=sweep_id, policy=policy, k_field=k_field)
        all_k = sorted(set(all_k) | set(ks))
        lows = []
        highs = []
        for s, n, y in zip(successes, episodes, ys, strict=True):
            low, high = wilson_interval(s, n)
            lows.append(y - low)
            highs.append(high - y)
        marker = "o" if policy == "hfrvla" else "s"
        ax.errorbar(
            ks,
            ys,
            yerr=[lows, highs],
            marker=marker,
            linewidth=2.0,
            markersize=4.5,
            capsize=2.5,
            color=POLICY_COLOR[policy],
            label=POLICY_LABEL[policy],
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_k)
    ax.set_xticklabels([str(k) for k in all_k])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Success rate (%)")
    ax.set_title(title)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.set_ylim(34, 86)
    ax.legend(frameon=False, loc="lower left")
    save_figure(fig, out_dir, stem)


def plot_calibration(rows: list[dict[str, str]], out_dir: Path) -> None:
    selected = filter_rows(rows, sweep_id="action_step8_alpha_clip_sweep", suite="libero_spatial", policy="hfrvla")
    alphas = sorted({as_float(r, "eval_alpha") for r in selected})
    deltas = sorted({as_float(r, "eval_delta_max") for r in selected})
    values = [[math.nan for _ in deltas] for _ in alphas]
    for r in selected:
        ai = alphas.index(as_float(r, "eval_alpha"))
        di = deltas.index(as_float(r, "eval_delta_max"))
        values[ai][di] = as_float(r, "pc_success")

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    calibration_cmap = ListedColormap(["#3b4cc0", "#2c7bb6", "#00a6ca", "#00ccbc", "#90eb9d", "#ffff8c"])
    calibration_bounds = [45, 55, 65, 70, 75, 80, 85]
    calibration_norm = BoundaryNorm(calibration_bounds, calibration_cmap.N)
    im = ax.imshow(
        values,
        cmap=calibration_cmap,
        norm=calibration_norm,
        aspect="auto",
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_xticks(range(len(deltas)), [f"{d:.1f}" for d in deltas])
    ax.set_yticks(range(len(alphas)), [f"{a:.2g}" for a in alphas])
    ax.set_xlabel("Residual clip delta_max")
    ax.set_ylabel("Residual scale alpha")
    ax.set_title("Spatial residual calibration, plan=50, exec=8")
    for ai, alpha in enumerate(alphas):
        for di, delta in enumerate(deltas):
            value = values[ai][di]
            label = f"{value:.0f}%"
            color = "white" if value < 60 else "black"
            ax.text(di, ai, label, ha="center", va="center", fontsize=8, color=color)
            if abs(alpha - 0.75) < 1e-9 and abs(delta - 0.2) < 1e-9:
                ax.add_patch(plt.Rectangle((di - 0.48, ai - 0.48), 0.96, 0.96, fill=False, ec="white", lw=2.4))
                ax.add_patch(plt.Rectangle((di - 0.43, ai - 0.43), 0.86, 0.86, fill=False, ec=OKABE["black"], lw=1.2))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, boundaries=calibration_bounds, ticks=calibration_bounds)
    cbar.set_label("Success rate (%)")
    ax.text(
        0.5,
        -0.28,
        "Best Spatial setting in this calibration: alpha=0.75, delta_max=0.2 (41/50).",
        transform=ax.transAxes,
        ha="center",
        fontsize=7.5,
        color="#444444",
    )
    save_figure(fig, out_dir, "fig4_residual_calibration")


def plot_async_delay(rows: list[dict[str, str]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.9, 3.0))
    series_style = {
        "hfrvla": {"marker": "o", "linestyle": "-", "x_offset": -0.045, "zorder": 3},
        "hfrvla_disable_fast": {"marker": "s", "linestyle": "--", "x_offset": 0.045, "zorder": 2},
    }
    for policy in ("hfrvla", "hfrvla_disable_fast"):
        selected = sorted(
            filter_rows(rows, sweep_id="async_timestep_planner_delay_eval_sweep", suite="libero_spatial", policy=policy),
            key=lambda r: as_int(r, "planner_delay_steps"),
        )
        style = series_style[policy]
        color = POLICY_COLOR[policy]
        x = [as_int(r, "planner_delay_steps") + style["x_offset"] for r in selected]
        y = [as_float(r, "pc_success") for r in selected]
        lows = []
        highs = []
        for r, value in zip(selected, y, strict=True):
            low, high = wilson_interval(as_int(r, "n_successes"), as_int(r, "n_episodes"))
            lows.append(value - low)
            highs.append(high - value)
        ax.errorbar(
            x,
            y,
            yerr=[lows, highs],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=4.5,
            capsize=2.5,
            capthick=1.2,
            color=color,
            ecolor=color,
            markeredgecolor=color,
            markerfacecolor="white" if policy == "hfrvla_disable_fast" else color,
            label=POLICY_LABEL[policy],
            zorder=style["zorder"],
        )
    ax.set_xlabel("Planner delay d (control timesteps)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Async-timestep planner-delay stress test")
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_ylim(45, 82)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.legend(frameon=False, loc="lower left")
    save_figure(fig, out_dir, "fig5_async_planner_delay")


def plot_generated_success_summary(rows: list[dict[str, str]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    configs = [
        ("fwr_generated_plan50_exec_10x10_spatial", "execution_chunk_size", "Plan=50, exec/replan=K"),
        ("fwr_generated_matched_chunk_10x10_spatial", "execution_chunk_size", "Plan=exec=replan=K"),
    ]
    for ax, (sweep_id, k_field, title) in zip(axes, configs, strict=True):
        for policy, marker in (("smolvla", "s"), ("hfrvla", "o")):
            ks, ys, _, _ = sweep_series(rows, sweep_id=sweep_id, policy=policy, k_field=k_field)
            ax.plot(
                ks,
                ys,
                marker=marker,
                linewidth=2.0,
                markersize=4.5,
                color=POLICY_COLOR[policy],
                label=POLICY_LABEL[policy],
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 50])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "50"])
        ax.set_xlabel("K")
        ax.set_title(title)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
        ax.set_ylim(34, 86)
    axes[0].set_ylabel("Success rate (%)")
    axes[0].legend(frameon=False, loc="lower left")
    save_figure(fig, out_dir, "fig6_generated_spatial_success")


def render_all(master: Path, out_dir: Path) -> None:
    rows = read_rows(master)
    apply_style()
    plot_problem_schematic(out_dir)
    plot_solution_schematic(out_dir)
    plot_success_sweep(
        rows,
        out_dir,
        sweep_id="fwr_generated_plan50_exec_10x10_spatial",
        k_field="execution_chunk_size",
        stem="fig2_plan50_execution_sweep",
        title="Plan-50 synchronous execution sweep",
        xlabel="Execution/replan interval K",
    )
    plot_success_sweep(
        rows,
        out_dir,
        sweep_id="fwr_generated_matched_chunk_10x10_spatial",
        k_field="execution_chunk_size",
        stem="fig3_matched_chunk_sweep",
        title="Synchronous matched chunk sweep",
        xlabel="Planning/execution chunk K",
    )
    plot_calibration(rows, out_dir)
    plot_async_delay(rows, out_dir)
    plot_generated_success_summary(rows, out_dir)
    print(f"[hfrvla-paper-results] wrote figures to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_all(args.master, args.out_dir)


if __name__ == "__main__":
    main()
