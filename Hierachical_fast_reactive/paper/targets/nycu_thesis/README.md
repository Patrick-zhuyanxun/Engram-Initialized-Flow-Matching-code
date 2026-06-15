# NYCU Thesis Target - HFRVLA

Last updated: 2026-06-15

This folder maintains the NYCU master's thesis version of HFRVLA. Its purpose is
not only to produce a final PDF, but to preserve the research process: why the
problem matters, how the method evolved, which experiments were run, what failed,
and which claims are still unsupported.

The NYCU-template rebuild lives at:

```text
paper/targets/nycu_thesis/latex/main.tex
```

It was rebuilt from the local template archive:

```text
paper/templates/nycu_thesis/NYCU_thesis_template__1_.zip
```

The legacy thesis draft remains available as a migration reference:

```text
paper/thesis/main.tex
```

This target folder is the process-control layer that feeds the NYCU-template
LaTeX project.

## Core Thesis Problem

Large VLA policies such as SmolVLA can predict action chunks, which reduces how
often an expensive planner must run. The cost is that later actions in the chunk
are executed under observations that may already be stale. HFRVLA asks whether a
small fast wrist correction module can correct the next action of a frozen
SmolVLA chunk without fine-tuning the slow planner.

This problem must appear before the method in the thesis introduction. The
thesis should not start from "we propose a model"; it should start from the
closed-loop execution failure caused by stale VLA action chunks.

## Local Sources Of Truth

| Topic | Source |
|---|---|
| Compilable NYCU-template draft | `paper/targets/nycu_thesis/latex/main.tex` |
| Legacy thesis draft, migration reference only | `paper/thesis/main.tex` |
| Shared method contract | `paper/notes/methodology_blueprint.md` |
| Research questions | `paper/notes/research_questions.md` |
| Contribution and wording guardrails | `paper/notes/contribution_statement.md` |
| Literature notes | `paper/notes/literature_review.md` |
| Registered results | `experiments/eval_registry/eval_results_master.csv` |
| Registry summary | `paper/thesis/generated/eval_registry_summary.md` |
| Long paper draft | `paper/src/main.tex`, `paper/src/main_zh.tex`, `paper/targets/long_paper/README.md` |
| Review/debate logs | `paper/deliberations/nycu_thesis/review_rounds/` |

## Maintenance Rules

1. Every quantitative claim must trace to `experiments/eval_registry/eval_results_master.csv` or a named generated table.
2. Literature additions must be searched and verified through a dedicated academic search skill, MCP, DOI, arXiv, CrossRef, publisher page, or official repository. Prefer top conferences, journals, and established robotics/vision venues when available.
3. Main thesis results should report success rate. Do not center margin plots or extra metrics unless they directly explain a failure mode.
4. The method description must be explicit about DINO wrist patches, cached SmolVLA context, chunk index, residual clipping, and the frozen-planner contract.
5. Unfinished experiments belong in a clearly marked future-work or appendix section, not as main-text evidence.
6. When a section feels generic, add one of these anchors: local artifact path, protocol variable, failure case, exact dataset feature, registry row group, or hypothesis being tested.
7. Keep the thesis separate from the long paper at the manuscript level. Reuse shared notes, figures, registry numbers, and verified bibliography entries, but do not force thesis chapters to mirror the long paper section order.

## Files In This Target Folder

| File | Role |
|---|---|
| `research_process.md` | Full thesis workflow from Idea to Appendix, with HFRVLA-specific content. |
| `source_map.md` | Local artifact map for chapters, figures, tables, and claims. |
| `format_and_template_notes.md` | NYCU template and final-format checklist. |
| `latex/` | Complete NYCU-template LaTeX paper rebuilt from `paper/templates/nycu_thesis/NYCU_thesis_template__1_.zip`. |

## Immediate Writing Backlog

| Priority | Task | Output |
|---:|---|---|
| 1 | Rewrite Chapter 1 so it starts from stale action chunks and inference-latency pressure. | Thesis introduction section. |
| 2 | Expand Related Work into a comparison-driven chapter, not a citation list. | Prior-work matrix and gap discussion. |
| 3 | Expand Methodology around DINO wrist patch extraction and visual evidence flow. | Method chapter and DINO figure discussion. |
| 4 | Keep Results success-rate-first and separate synchronous sweeps from async delay. | Results chapter and tables. |
| 5 | Move unsupported claims to limitations/future work. | Discussion and appendix. |
