#!/usr/bin/env python3
"""Render thesis-specific HFRVLA experiment tables from the eval registry."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = REPO_ROOT / "experiments/eval_registry/eval_results_master.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "paper/thesis/generated"


def latex_escape(value: object) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_int(value: str) -> int:
    return int(float(value)) if value else 0


def success_cell(row: dict[str, str]) -> str:
    if row.get("n_successes") and row.get("n_episodes") and row.get("pc_success"):
        return f"{row['n_successes']}/{row['n_episodes']} ({float(row['pc_success']):.1f}\\%)"
    return latex_escape(row.get("status", ""))


def short_id(value: str, max_len: int = 44) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def begin_longtable(
    spec: str,
    caption: str,
    label: str,
    header: list[str],
    *,
    size: str = "small",
) -> list[str]:
    header_line = " & ".join(header) + r" \\"
    return [
        rf"\begin{{{size}}}",
        rf"\begin{{longtable}}{{{spec}}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        header_line,
        r"\midrule",
        r"\endfirsthead",
        rf"\caption[]{{{caption}（續）}}\\",
        r"\toprule",
        header_line,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        rf"\multicolumn{{{len(header)}}}{{r}}{{續下頁}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]


def end_longtable(size: str = "small") -> list[str]:
    return [r"\end{longtable}", rf"\end{{{size}}}", ""]


def text_or_not_recorded(value: str | None) -> str:
    if value is None or value == "":
        return "not recorded"
    return value


def url_cell(value: str | None) -> str:
    text = text_or_not_recorded(value)
    if text == "not recorded":
        return r"\emph{not recorded}"
    return r"\url{" + text.replace("\\", "/") + "}"


def url_list_cell(values: list[str]) -> str:
    filtered = [v for v in values if v]
    if not filtered:
        return r"\emph{not recorded}"
    return r"; ".join(url_cell(v) for v in filtered)


def alpha_delta_cell(row: dict[str, str] | None = None, values: list[str] | None = None) -> str:
    if values is not None:
        filtered = [v for v in values if v]
        if not filtered:
            return r"\emph{not recorded}"
        return latex_escape("; ".join(filtered))
    if row is None:
        return r"\emph{not recorded}"
    alpha = text_or_not_recorded(row.get("eval_alpha"))
    delta = text_or_not_recorded(row.get("eval_delta_max"))
    if alpha == "not recorded" and delta == "not recorded":
        return r"\emph{not recorded}"
    return latex_escape(f"{alpha}/{delta}")


def alpha_delta_text(row: dict[str, str]) -> str:
    if not row.get("eval_alpha") and not row.get("eval_delta_max"):
        return ""
    return f"{text_or_not_recorded(row.get('eval_alpha'))}/{text_or_not_recorded(row.get('eval_delta_max'))}"


def render_inventory(rows: list[dict[str, str]]) -> list[str]:
    by_sweep: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_sweep[row["sweep_id"]].append(row)
    lines = begin_longtable(
        r"@{}p{0.25\linewidth}p{0.18\linewidth}p{0.14\linewidth}p{0.16\linewidth}p{0.10\linewidth}r@{}",
        "評估註冊表實驗群組總覽。Rows 代表 master registry 中該 sweep 的列數；status 保留 pending/cached/derived 等 provenance。",
        "tab:registry-inventory",
        ["Sweep", "Type", "Policies", "Suites", "Status", "Rows"],
        size="scriptsize",
    )
    for sweep_id in sorted(by_sweep):
        group = by_sweep[sweep_id]
        sweep_type = group[0].get("sweep_type", "")
        policies = ",".join(sorted({r.get("policy", "") for r in group if r.get("policy")}))
        suites = ",".join(sorted({r.get("suite", "") for r in group if r.get("suite")}))
        statuses = ",".join(sorted({r.get("status", "") for r in group if r.get("status")}))
        lines.append(
            " & ".join(
                [
                    url_cell(sweep_id),
                    latex_escape(sweep_type),
                    latex_escape(policies),
                    latex_escape(suites),
                    latex_escape(statuses),
                    str(len(group)),
                ]
            )
            + r" \\"
        )
    lines.extend(end_longtable("scriptsize"))
    return lines


def render_provenance(rows: list[dict[str, str]]) -> list[str]:
    by_family: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row.get("sweep_id", ""), row.get("policy", ""), row.get("suite", ""))
        by_family[key].append(row)

    lines = begin_longtable(
        r"@{}p{0.20\linewidth}p{0.10\linewidth}p{0.10\linewidth}p{0.16\linewidth}p{0.07\linewidth}p{0.12\linewidth}p{0.17\linewidth}p{0.08\linewidth}@{}",
        "結果 provenance 表。此表以 sweep/policy/suite 為單位列出 checkpoint 與評估設定；所有缺漏欄位以 not recorded 標示，避免把 metadata blanks 視為 verified hyperparameters。",
        "tab:result-provenance",
        ["Sweep", "Policy", "Suite", "Checkpoint", "Steps", r"$\alpha/\delta$", "Source", "Status"],
        size="scriptsize",
    )
    for key in sorted(by_family):
        group = by_family[key]
        first = group[0]
        checkpoints = sorted({r.get("checkpoint_id", "") for r in group if r.get("checkpoint_id")})
        steps = sorted({r.get("train_steps", "") for r in group if r.get("train_steps")})
        alpha_delta = sorted({alpha_delta_text(r) for r in group if alpha_delta_text(r)})
        sources = sorted({r.get("source_csv", "") for r in group if r.get("source_csv")})
        statuses = sorted({r.get("status", "") for r in group if r.get("status")})
        lines.append(
            " & ".join(
                [
                    url_cell(first.get("sweep_id", "")),
                    url_cell(first.get("policy", "")),
                    url_cell(first.get("suite", "")),
                    url_list_cell(checkpoints),
                    latex_escape(",".join(steps) if steps else "not recorded"),
                    alpha_delta_cell(values=alpha_delta),
                    url_list_cell(sources),
                    latex_escape(",".join(statuses) if statuses else "not recorded"),
                ]
            )
            + r" \\"
        )
    lines.extend(end_longtable("scriptsize"))
    return lines


def render_checkpoint_protocol_matrix(rows: list[dict[str, str]]) -> list[str]:
    tiers = {
        "fwr_generated_plan50_exec_10x10_spatial": "main generated FWR-v2",
        "fwr_generated_matched_chunk_10x10_spatial": "main generated FWR-v2",
        "action_step8_alpha_clip_sweep": "30k calibration diagnostic",
        "n50_alpha_clip_spatial_50eps": "long-chunk calibration diagnostic",
        "async_timestep_planner_delay_eval_sweep": "async diagnostic",
        "hfrvla_lrwd_alpha075_clip02_eval": "LR/WD diagnostic",
        "action_steps_eval_sweep": "historical 30k diagnostic",
        "chunk_size_eval_sweep": "historical 30k diagnostic",
        "fwr_action_steps_10x10": "historical FWR-v2 diagnostic",
        "fwr_chunk_size_10x10": "historical FWR-v2 diagnostic",
        "fwr_chunk_plan50_exec50_eval": "historical FWR-v2 smoke",
        "action_steps_eval_sweep_no_resclip": "pending control",
        "chunk_size_eval_sweep_no_resclip": "pending control",
    }
    by_sweep: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_sweep[row["sweep_id"]].append(row)
    lines = begin_longtable(
        r"@{}p{0.24\linewidth}p{0.17\linewidth}p{0.18\linewidth}p{0.18\linewidth}p{0.09\linewidth}p{0.08\linewidth}@{}",
        "Checkpoint and protocol matrix. Tier 欄位定義本文如何使用該 sweep：main、diagnostic、historical 或 pending。",
        "tab:checkpoint-protocol-matrix",
        ["Sweep", "Tier", "Checkpoint/profile", "Protocol", "Episodes", "Status"],
        size="scriptsize",
    )
    for sweep_id in sorted(by_sweep):
        group = by_sweep[sweep_id]
        checkpoints = sorted({r.get("checkpoint_id") or r.get("metadata_profile", "") for r in group if r.get("checkpoint_id") or r.get("metadata_profile")})
        statuses = sorted({r.get("status", "") for r in group if r.get("status")})
        plans = sorted({r.get("planning_chunk_size", "") for r in group if r.get("planning_chunk_size")})
        execs = sorted({r.get("execution_chunk_size", "") for r in group if r.get("execution_chunk_size")})
        delays = sorted({r.get("planner_delay_steps", "") for r in group if r.get("planner_delay_steps")})
        episodes = sorted({r.get("n_episodes", "") for r in group if r.get("n_episodes")})
        protocol = f"plan={','.join(plans) or 'not recorded'}; exec={','.join(execs) or 'not recorded'}"
        if delays:
            protocol += f"; delay={','.join(delays)}"
        lines.append(
            " & ".join(
                [
                    url_cell(sweep_id),
                    latex_escape(tiers.get(sweep_id, "inventory only")),
                    url_list_cell(checkpoints),
                    latex_escape(protocol),
                    latex_escape(",".join(episodes) if episodes else "pending/not recorded"),
                    latex_escape(",".join(statuses) if statuses else "not recorded"),
                ]
            )
            + r" \\"
        )
    lines.extend(end_longtable("scriptsize"))
    return lines


def render_core_sweep_table(
    rows: list[dict[str, str]],
    *,
    sweep_id: str,
    label: str,
    caption: str,
    k_field: str = "execution_chunk_size",
    policies: tuple[str, ...] = ("smolvla", "hfrvla"),
    suite: str = "libero_spatial",
) -> list[str]:
    index: dict[tuple[str, int], dict[str, str]] = {}
    ks: set[int] = set()
    for row in rows:
        if row.get("sweep_id") != sweep_id or row.get("suite") != suite or row.get("policy") not in policies:
            continue
        k = as_int(row.get(k_field, ""))
        index[(row["policy"], k)] = row
        ks.add(k)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{@{}rrr@{}}",
        r"\toprule",
        r"$K$ & SmolVLA success & HFRVLA success \\",
        r"\midrule",
    ]
    for k in sorted(ks):
        if ("hfrvla", k) not in index or ("smolvla", k) not in index:
            continue
        h = index[("hfrvla", k)]
        s = index[("smolvla", k)]
        lines.append(rf"{k} & {float(s['pc_success']):.1f}\% & {float(h['pc_success']):.1f}\% \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return lines


def render_async_table(rows: list[dict[str, str]]) -> list[str]:
    index: dict[tuple[str, int], dict[str, str]] = {}
    delays: set[int] = set()
    for row in rows:
        if row.get("sweep_id") != "async_timestep_planner_delay_eval_sweep":
            continue
        delay = as_int(row.get("planner_delay_steps", ""))
        index[(row["policy"], delay)] = row
        delays.add(delay)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Async-timestep planner-delay sweep exact values. 每列為 LIBERO-Spatial 100 episodes，plan=50、execution/replan=16、async request interval=8。}",
        r"\label{tab:thesis-async-delay}",
        r"\begin{tabular}{@{}rrr@{}}",
        r"\toprule",
        r"Delay $d$ & Disable-fast success & HFRVLA success \\",
        r"\midrule",
    ]
    for delay in sorted(delays):
        h = index.get(("hfrvla", delay))
        d = index.get(("hfrvla_disable_fast", delay))
        if not h or not d:
            continue
        lines.append(rf"{delay} & {float(d['pc_success']):.1f}\% & {float(h['pc_success']):.1f}\% \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return lines


def render_alpha_clip_table(rows: list[dict[str, str]], sweep_id: str, label: str, caption: str) -> list[str]:
    selected = [
        row
        for row in rows
        if row.get("sweep_id") == sweep_id and row.get("suite") == "libero_spatial" and row.get("policy") == "hfrvla"
    ]
    selected.sort(key=lambda r: (float(r.get("eval_alpha") or 0), float(r.get("eval_delta_max") or 0)))
    lines = begin_longtable(
        r"@{}rrrrr@{}",
        caption,
        label,
        [r"$\alpha$", r"$\delta_{\max}$", "Plan", "Exec", "Success"],
    )
    for row in selected:
        lines.append(
            " & ".join(
                [
                    latex_escape(text_or_not_recorded(row.get("eval_alpha"))),
                    latex_escape(text_or_not_recorded(row.get("eval_delta_max"))),
                    latex_escape(row.get("planning_chunk_size")),
                    latex_escape(row.get("execution_chunk_size")),
                    success_cell(row),
                ]
            )
            + r" \\"
        )
    lines.extend(end_longtable())
    return lines


def render_lrwd_table(rows: list[dict[str, str]]) -> list[str]:
    selected = [
        row
        for row in rows
        if row.get("sweep_id") == "hfrvla_lrwd_alpha075_clip02_eval" and row.get("policy") == "hfrvla"
    ]
    selected.sort(
        key=lambda r: (
            float(r.get("lr") or 0),
            float(r.get("weight_decay") or 0),
            r.get("suite", ""),
        )
    )
    lines = begin_longtable(
        r"@{}p{0.16\linewidth}p{0.16\linewidth}p{0.17\linewidth}p{0.20\linewidth}p{0.17\linewidth}@{}",
        r"LR/WD diagnostic exact rows for plan=50, exec=8, $\alpha=0.75,\delta_{\max}=0.2$. 這些 rows 用於 checkpoint selection，不作為主結果。",
        "tab:thesis-lrwd-diagnostic",
        ["LR", "Weight decay", "Suite", "Checkpoint/profile", "Success"],
    )
    for row in selected:
        checkpoint = row.get("checkpoint_id") or row.get("metadata_profile") or ""
        lines.append(
            " & ".join(
                [
                    latex_escape(text_or_not_recorded(row.get("lr"))),
                    latex_escape(text_or_not_recorded(row.get("weight_decay"))),
                    latex_escape(text_or_not_recorded(row.get("suite"))),
                    url_cell(checkpoint),
                    success_cell(row),
                ]
            )
            + r" \\"
        )
    lines.extend(end_longtable())
    return lines


def render_full_registry(rows: list[dict[str, str]]) -> list[str]:
    selected = [r for r in rows if r.get("pc_success") or r.get("status") == "pending"]
    selected.sort(
        key=lambda r: (
            r.get("sweep_id", ""),
            r.get("suite", ""),
            r.get("policy", ""),
            as_int(r.get("planning_chunk_size", "")),
            as_int(r.get("execution_chunk_size", "")),
            as_int(r.get("planner_delay_steps", "")),
            float(r.get("eval_alpha") or 0),
            float(r.get("eval_delta_max") or 0),
        )
    )
    lines = begin_longtable(
        r"@{}p{0.18\linewidth}p{0.10\linewidth}p{0.10\linewidth}rrrrp{0.10\linewidth}p{0.08\linewidth}p{0.08\linewidth}p{0.11\linewidth}p{0.07\linewidth}@{}",
        "完整評估註冊表列。此表保留所有有 success 值或 pending 狀態的 rows，用於追溯主文與附錄中的所有實驗數字。",
        "tab:full-eval-registry",
        ["Sweep", "Suite", "Policy", "Plan", "Exec", "Delay", r"$\alpha/\delta$", "LR", "WD", "Success", "Status"],
        size="tiny",
    )
    for row in selected:
        lines.append(
            " & ".join(
                [
                    url_cell(row.get("sweep_id", "")),
                    url_cell(row.get("suite", "")),
                    url_cell(row.get("policy", "")),
                    latex_escape(text_or_not_recorded(row.get("planning_chunk_size"))),
                    latex_escape(text_or_not_recorded(row.get("execution_chunk_size"))),
                    latex_escape(text_or_not_recorded(row.get("planner_delay_steps"))),
                    alpha_delta_cell(row),
                    latex_escape(text_or_not_recorded(row.get("lr"))),
                    latex_escape(text_or_not_recorded(row.get("weight_decay"))),
                    success_cell(row),
                    latex_escape(row.get("status", "")),
                ]
            )
            + r" \\"
        )
    lines.extend(end_longtable("tiny"))
    return lines


def render_markdown_summary(rows: list[dict[str, str]]) -> str:
    by_sweep: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_sweep[row["sweep_id"]].append(row)
    lines = ["# HFRVLA Thesis Experiment Registry Summary", ""]
    lines.append(f"- Master rows: `{len(rows)}`")
    lines.append(f"- Sweep groups: `{len(by_sweep)}`")
    lines.append("")
    for sweep_id in sorted(by_sweep):
        group = by_sweep[sweep_id]
        lines.append(f"## {sweep_id}")
        lines.append(f"- Rows: {len(group)}")
        lines.append(f"- Type: `{group[0].get('sweep_type','')}`")
        lines.append(f"- Policies: `{', '.join(sorted({r.get('policy','') for r in group}))}`")
        lines.append(f"- Suites: `{', '.join(sorted({r.get('suite','') for r in group}))}`")
        lines.append(f"- Statuses: `{', '.join(sorted({r.get('status','') for r in group}))}`")
        ok = [r for r in group if r.get("pc_success")]
        if ok:
            best = max(ok, key=lambda r: float(r["pc_success"]))
            lines.append(
                "- Best success row: "
                f"`{best.get('policy')}` `{best.get('suite')}` plan={best.get('planning_chunk_size')} "
                f"exec={best.get('execution_chunk_size')} alpha={text_or_not_recorded(best.get('eval_alpha'))} "
                f"delta={text_or_not_recorded(best.get('eval_delta_max'))} -> {best.get('n_successes')}/{best.get('n_episodes')} "
                f"({float(best.get('pc_success')):.1f}%)"
            )
        lines.append("")
    return "\n".join(lines)


def render(master: Path, out_dir: Path) -> None:
    rows = read_rows(master)
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory_table_lines: list[str] = []
    inventory_table_lines.extend(render_inventory(rows))
    inventory_table_lines.extend(render_checkpoint_protocol_matrix(rows))
    inventory_table_lines.extend(render_provenance(rows))
    result_table_lines: list[str] = []
    result_table_lines.extend(
        render_core_sweep_table(
            rows,
            sweep_id="fwr_generated_plan50_exec_10x10_spatial",
            label="tab:thesis-generated-plan50",
            caption="Generated checkpoint LIBERO-Spatial 10x10 synchronous plan-50 execution sweep exact values.",
        )
    )
    result_table_lines.extend(
        render_core_sweep_table(
            rows,
            sweep_id="fwr_generated_matched_chunk_10x10_spatial",
            label="tab:thesis-generated-matched",
            caption="Generated checkpoint LIBERO-Spatial 10x10 synchronous matched chunk sweep exact values.",
        )
    )
    result_table_lines.extend(render_async_table(rows))
    result_table_lines.extend(
        render_alpha_clip_table(
            rows,
            "action_step8_alpha_clip_sweep",
            "tab:thesis-alpha-clip-exec8",
            r"Plan=50, exec=8 的 Spatial alpha/clip calibration。每列為 50 episodes。",
        )
    )
    result_table_lines.extend(
        render_alpha_clip_table(
            rows,
            "n50_alpha_clip_spatial_50eps",
            "tab:thesis-alpha-clip-n50",
            r"Plan=exec=replan=50 的 Spatial alpha/clip long-chunk calibration。每列為 50 episodes。",
        )
    )
    result_table_lines.extend(render_lrwd_table(rows))
    appendix_table_lines = render_full_registry(rows)
    (out_dir / "eval_registry_inventory_tables.tex").write_text("\n".join(inventory_table_lines), encoding="utf-8")
    (out_dir / "eval_registry_result_tables.tex").write_text("\n".join(result_table_lines), encoding="utf-8")
    (out_dir / "eval_registry_main_tables.tex").write_text(
        "\n".join(inventory_table_lines + result_table_lines),
        encoding="utf-8",
    )
    (out_dir / "eval_registry_appendix_tables.tex").write_text("\n".join(appendix_table_lines), encoding="utf-8")
    (out_dir / "eval_registry_tables.tex").write_text(
        "\n".join(inventory_table_lines + result_table_lines + appendix_table_lines),
        encoding="utf-8",
    )
    (out_dir / "eval_registry_summary.md").write_text(render_markdown_summary(rows), encoding="utf-8")
    print(f"[hfrvla-thesis] wrote {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render(args.master, args.out_dir)


if __name__ == "__main__":
    main()
