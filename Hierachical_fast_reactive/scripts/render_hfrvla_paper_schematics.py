#!/usr/bin/env python3
"""Render deterministic HFRVLA paper schematics.

The timing figure uses exact H/e/d cell counts, so it is kept as code rather
than a generated bitmap.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIRS = [
    ROOT / "docs/assets/hfrvla-paper",
    ROOT / "docs/presentations/hfrvla-training-open-slide/assets/hfrvla-paper",
]


BLUE = "#2b6cb0"
BLUE_LIGHT = "#e8f1fb"
GREEN = "#2f855a"
GREEN_LIGHT = "#e7f6ec"
YELLOW = "#ffe8a3"
YELLOW_DARK = "#d69e2e"
PINK = "#f7d6d6"
GRAY = "#4a5568"
GRAY_LIGHT = "#f3f4f6"
BLACK = "#1a202c"
CYAN = "#319795"
MAGENTA = "#b83280"


def rounded(ax, xy, w, h, text="", fc="white", ec=BLACK, lw=1.5, r=0.08, fontsize=12, weight="normal"):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={r}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    if text:
        ax.text(
            xy[0] + w / 2,
            xy[1] + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=BLACK,
            weight=weight,
        )
    return box


def arrow(ax, start, end, color=BLACK, lw=1.6, mutation_scale=14, style="-|>"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


def double_arrow(ax, start, end, text, color=BLACK, y_offset=0.18, fontsize=13):
    arrow(ax, start, end, color=color, lw=1.8, mutation_scale=15, style="<->")
    ax.text(
        (start[0] + end[0]) / 2,
        start[1] + y_offset,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color=color,
        weight="bold",
    )


def save_all(fig, name: str) -> None:
    for asset_dir in ASSET_DIRS:
        asset_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(asset_dir / name, dpi=180, bbox_inches="tight", pad_inches=0.08)


def render_architecture() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor("white")

    # Input stack for the frozen slow planner.
    ax.text(0.45, 8.45, "slow planner inputs", fontsize=15, weight="bold", color=BLUE, ha="left")
    slow_inputs = [
        ("task\ninstruction", 7.55),
        ("third-person\nscene image", 6.65),
        ("wrist\nimage", 5.75),
        ("robot\nstate", 4.85),
    ]
    for label, y in slow_inputs:
        rounded(ax, (0.45, y), 1.75, 0.6, label, fc="white", ec=GRAY, fontsize=10)
        arrow(ax, (2.2, y + 0.3), (3.0, y + 0.3), color=GRAY, lw=1.4)

    # Frozen SmolVLA container.
    slow = FancyBboxPatch(
        (3.0, 4.1),
        6.15,
        4.25,
        boxstyle="round,pad=0.05,rounding_size=0.28",
        linewidth=2.2,
        edgecolor=BLUE,
        facecolor=BLUE_LIGHT,
        linestyle="--",
    )
    ax.add_patch(slow)
    ax.text(6.07, 8.02, "Frozen SmolVLA slow planner", fontsize=16, weight="bold", color=BLUE, ha="center")

    rounded(ax, (3.45, 6.75), 3.05, 0.72, "Vision-language trunk\n(frozen)", fc="white", ec=BLUE, fontsize=12, weight="bold")
    rounded(ax, (6.9, 6.75), 1.75, 0.72, "action chunk\npredictor", fc="white", ec=BLUE, fontsize=11, weight="bold")
    rounded(ax, (3.45, 5.55), 2.45, 0.58, "image + language\nfeatures", fc=YELLOW, ec=YELLOW_DARK, fontsize=10)
    rounded(ax, (6.2, 5.55), 2.45, 0.58, "slow context taps\nz_goal / z_phase", fc=YELLOW, ec=YELLOW_DARK, fontsize=10)
    rounded(ax, (5.45, 4.6), 2.45, 0.58, "base action chunk\nA_t = [a_t ... a_{t+H}]", fc="white", ec=BLUE, fontsize=10)

    arrow(ax, (6.5, 7.11), (6.9, 7.11), color=BLUE, lw=1.8)
    arrow(ax, (4.68, 6.75), (4.68, 6.15), color=BLUE, lw=1.4)
    arrow(ax, (7.43, 6.75), (7.43, 6.15), color=BLUE, lw=1.4)
    arrow(ax, (7.43, 5.55), (6.68, 5.18), color=BLUE, lw=1.5)

    # Slow outputs.
    rounded(ax, (9.65, 7.02), 1.65, 0.62, "a_base", fc="white", ec=BLUE, fontsize=13, weight="bold")
    rounded(ax, (9.65, 5.86), 2.25, 0.72, "slow context:\nz_goal / z_phase", fc="white", ec=BLUE, fontsize=11, weight="bold")
    arrow(ax, (7.9, 4.89), (9.65, 7.32), color=BLUE, lw=1.8)
    arrow(ax, (8.65, 5.84), (9.65, 6.22), color=BLUE, lw=1.8)

    # Fast residual path.
    ax.text(0.45, 3.45, "fast residual inputs", fontsize=15, weight="bold", color=GREEN, ha="left")
    fast_inputs = [
        ("wrist camera", 2.72),
        ("DINO wrist\npatches", 1.97),
        ("robot state", 1.22),
        ("k_idx_norm", 0.47),
    ]
    for i, (label, y) in enumerate(fast_inputs):
        edge = CYAN if i < 2 else GRAY
        rounded(ax, (0.45, y), 1.75, 0.52, label, fc="white", ec=edge, fontsize=10)
    arrow(ax, (2.2, 2.98), (2.75, 2.23), color=CYAN, lw=1.5)

    rounded(ax, (2.75, 1.83), 1.65, 0.66, "wrist patch\nencoder", fc="#e6fffb", ec=CYAN, fontsize=10, weight="bold")
    arrow(ax, (2.2, 2.23), (2.75, 2.16), color=CYAN, lw=1.5)
    arrow(ax, (4.4, 2.16), (5.15, 2.16), color=GREEN, lw=1.8)
    arrow(ax, (2.2, 1.48), (5.15, 1.55), color=GRAY, lw=1.2)
    arrow(ax, (2.2, 0.73), (5.15, 0.95), color=GRAY, lw=1.2)

    rounded(ax, (5.15, 0.62), 3.35, 2.35, "Trainable wrist fast\nresidual module", fc=GREEN_LIGHT, ec=GREEN, lw=2.2, fontsize=14, weight="bold")
    rounded(ax, (5.55, 2.1), 1.15, 0.42, "fusion", fc="white", ec=GREEN, fontsize=9)
    rounded(ax, (6.95, 2.1), 1.15, 0.42, "residual\nhead", fc="white", ec=GREEN, fontsize=9)
    arrow(ax, (6.7, 2.31), (6.95, 2.31), color=GREEN, lw=1.3)

    # Context connections into the fast module.
    arrow(ax, (10.48, 7.02), (7.85, 2.97), color=BLUE, lw=1.6)
    arrow(ax, (10.78, 5.86), (7.2, 2.97), color=BLUE, lw=1.6)
    rounded(ax, (3.0, 0.05), 1.4, 0.42, "previous\nresidual", fc="white", ec=GRAY, fontsize=8)
    arrow(ax, (4.4, 0.26), (5.15, 0.78), color=GRAY, lw=1.2)

    # Merge and final output.
    rounded(ax, (9.2, 1.68), 1.15, 0.62, "delta_a", fc="white", ec=GREEN, fontsize=12, weight="bold")
    rounded(ax, (10.85, 1.58), 1.85, 0.82, "alpha *\nclip(delta_a)", fc=GREEN_LIGHT, ec=GREEN, fontsize=11, weight="bold")
    plus = plt.Circle((13.42, 1.98), 0.33, fill=False, color=BLACK, linewidth=2.0)
    ax.add_patch(plus)
    ax.text(13.42, 1.98, "+", ha="center", va="center", fontsize=20, weight="bold")
    rounded(
        ax,
        (14.05, 1.5),
        2.45,
        0.92,
        "a_final = a_base +\nalpha * clip(delta_a)",
        fc="white",
        ec=BLACK,
        fontsize=10,
        weight="bold",
    )

    arrow(ax, (8.5, 1.8), (9.2, 1.99), color=GREEN, lw=1.8)
    arrow(ax, (10.35, 1.99), (10.85, 1.99), color=GREEN, lw=1.8)
    arrow(ax, (12.7, 1.99), (13.09, 1.99), color=GREEN, lw=1.8)
    arrow(ax, (13.75, 1.98), (14.05, 1.98), color=BLACK, lw=1.8)
    arrow(ax, (10.48, 7.02), (13.42, 2.31), color=BLUE, lw=1.7)

    # Small legend.
    rounded(ax, (11.95, 7.9), 0.42, 0.28, "", fc=BLUE_LIGHT, ec=BLUE, lw=1.6)
    ax.text(12.5, 8.04, "frozen slow path", fontsize=9, va="center", color=BLUE)
    rounded(ax, (11.95, 7.45), 0.42, 0.28, "", fc=GREEN_LIGHT, ec=GREEN, lw=1.6)
    ax.text(12.5, 7.59, "trainable fast path", fontsize=9, va="center", color=GREEN)

    save_all(fig, "hfrvla-architecture-schematic.png")
    plt.close(fig)


def cell(ax, x, y, w, h, fc, ec, lw=1.2, text="", fontsize=9, color=BLACK):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=color)


def draw_chunk(ax, start, y, outline, row_label, H=8, e=4, d=2, cw=0.55, ch=0.42):
    ax.text(-0.15, y + ch / 2, row_label, ha="right", va="center", fontsize=15, weight="bold", color=BLACK)
    for i in range(H):
        if i < d:
            fc = PINK
        elif i < d + e:
            fc = "#dff1df"
        else:
            fc = "white"
        label = ""
        if i in (0, 1, 2, H - 1):
            label = f"a{i}" if i < H - 1 else f"a{H - 1}"
        elif i == 3:
            label = "..."
        cell(ax, start + i * cw, y, cw, ch, fc, outline, lw=1.4, text=label, fontsize=8)
    return start + d * cw, start + (d + e) * cw


def render_timing() -> None:
    H = 8
    e = 4
    d = 2
    n_chunks = 4
    cw = 0.55
    fig, ax = plt.subplots(figsize=(17, 9.4))
    ax.set_xlim(-2.9, 13.7)
    ax.set_ylim(0, 9.05)
    ax.axis("off")
    ax.set_facecolor("white")

    # Time grid.
    max_steps = d + n_chunks * e + 2
    for i in range(max_steps + 1):
        x = i * cw
        if i % 2 == 0:
            ax.add_patch(Rectangle((x, 0.8), cw, 7.0, facecolor="#f7f7f7", edgecolor="none", zorder=0))
        ax.plot([x, x], [0.8, 7.8], color="#d8d8d8", linewidth=0.65, zorder=0)
    ax.plot([0, max_steps * cw], [0.48, 0.48], color=BLACK, linewidth=1.4)
    arrow(ax, (0, 0.48), (max_steps * cw + 0.35, 0.48), color=BLACK, lw=1.5)
    ax.text(max_steps * cw + 0.45, 0.48, "time", va="center", fontsize=11, color=BLACK)

    # Top statement, deliberately not matching the provided reference styling.
    ax.text(0.0, 8.58, "Planner-delay chunk stitching", fontsize=17, weight="bold", color=BLACK)
    rounded(ax, (8.45, 8.34), 1.1, 0.38, "H = 8", fc="white", ec=GRAY, fontsize=11, weight="bold")
    rounded(ax, (9.75, 8.34), 1.1, 0.38, "e = 4", fc="white", ec=GREEN, fontsize=11, weight="bold")
    rounded(ax, (11.05, 8.34), 1.1, 0.38, "d = 2", fc="white", ec="#c53030", fontsize=11, weight="bold")
    ax.text(0.0, 8.18, "queued execution continues while the next slow chunk is generated", fontsize=11.5, color=GRAY)
    ax.text(0.0, 7.86, "base actions come from an observation that is d control steps old", fontsize=11.5, color=GRAY)

    # Brackets for H/e/d.
    double_arrow(ax, (0, 7.18), (H * cw, 7.18), "planning horizon H", color=BLACK, fontsize=10.5)
    double_arrow(ax, (d * cw, 6.76), ((d + e) * cw, 6.76), "execution horizon e", color=GREEN, fontsize=10.5)
    double_arrow(ax, (0, 6.34), (d * cw, 6.34), "delay d", color="#c53030", fontsize=10.5)

    outlines = [CYAN, BLUE, "#9f7aea", MAGENTA]
    labels = ["A_t", "A_t+e", "A_t+2e", "A_t+3e"]
    ys = [5.82, 5.03, 4.24, 3.45]
    green_windows = []
    for idx, (label, y) in enumerate(zip(labels, ys)):
        start = idx * e * cw
        left, right = draw_chunk(ax, start, y, outlines[idx], label, H=H, e=e, d=d, cw=cw)
        green_windows.append((left, right, outlines[idx]))
        ax.text(start, y + 0.5, "observe", fontsize=8.2, color=GRAY, ha="left")
        ax.text(start + d * cw, y + 0.5, "ready", fontsize=8.2, color=GRAY, ha="center")

    # Explain cell colors without using copied icons.
    cell(ax, 7.05, 6.18, 0.35, 0.26, PINK, "#c53030")
    ax.text(7.5, 6.31, "waiting prefix", fontsize=9, va="center", color=GRAY)
    cell(ax, 7.05, 5.82, 0.35, 0.26, "#dff1df", GREEN)
    ax.text(7.5, 5.95, "executed window", fontsize=9, va="center", color=GRAY)
    cell(ax, 7.05, 5.46, 0.35, 0.26, "white", GRAY)
    ax.text(7.5, 5.59, "lookahead", fontsize=9, va="center", color=GRAY)

    # Downward aggregation arrows.
    for left, right, outline in green_windows:
        arrow(ax, ((left + right) / 2, 3.38), ((left + right) / 2, 2.65), color=outline, lw=1.4)

    # Executed base sequence.
    ax.text(-0.15, 2.5, "executed\nbase sequence", ha="right", va="center", fontsize=12.5, weight="bold")
    y_base = 2.31
    for idx, (left, right, outline) in enumerate(green_windows):
        for j in range(e):
            x = left + j * cw
            cell(ax, x, y_base, cw, 0.36, "#dff1df", outline, lw=1.5)
    ax.text(green_windows[-1][1] + 0.18, y_base + 0.18, "...", fontsize=18, va="center", color=BLACK)

    # Current wrist residual row.
    ax.text(-0.15, 1.72, "current\nwrist residual", ha="right", va="center", fontsize=12.5, weight="bold", color=GREEN)
    y_res = 1.58
    for idx in range(n_chunks * e):
        x = d * cw + idx * cw
        ax.plot([x + cw / 2, x + cw / 2], [y_res, y_res + 0.36], color=GREEN, linewidth=1.6)
        arrow(ax, (x + cw / 2, y_res + 0.18), (x + cw / 2, y_res + 0.38), color=GREEN, lw=1.0, mutation_scale=9)
    ax.text(5.95, y_res + 0.45, "delta_a from current wrist feedback every step", fontsize=9.5, color=GREEN, va="center")

    # Final executed sequence.
    ax.text(-0.15, 1.05, "final\nexecuted sequence", ha="right", va="center", fontsize=12.5, weight="bold")
    y_final = 0.84
    for idx, (left, right, outline) in enumerate(green_windows):
        for j in range(e):
            x = left + j * cw
            cell(ax, x, y_final, cw, 0.36, "#edf7ed", outline, lw=1.35)
            ax.text(x + cw / 2, y_final + 0.18, "+", ha="center", va="center", fontsize=9, color=GREEN, weight="bold")

    rounded(
        ax,
        (6.05, 0.16),
        3.55,
        0.42,
        "a_final = a_base + alpha * clip(delta_a)",
        fc="white",
        ec=BLACK,
        fontsize=9.6,
        weight="bold",
    )
    rounded(
        ax,
        (9.75, 2.58),
        2.6,
        0.72,
        "slow chunk may be stale;\nwrist residual is current",
        fc=GREEN_LIGHT,
        ec=GREEN,
        fontsize=9.5,
        weight="bold",
    )

    save_all(fig, "planner-delay-timing-schematic.png")
    plt.close(fig)


def main() -> None:
    render_architecture()
    render_timing()


if __name__ == "__main__":
    main()
