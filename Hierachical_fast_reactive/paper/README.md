# HFRVLA Paper Workspace

Updated: 2026-06-15

This directory contains multiple paper-facing outputs for the HFRVLA project.
Do not treat every file under `paper/` as the same manuscript target.

## Manuscript Strategy

Maintain the long paper and the master's thesis as separate manuscript targets,
but keep one shared evidence base.

| Layer | Shared or separate? | Files |
|---|---|---|
| Method contract, research questions, contribution wording | Shared | `paper/notes/` |
| Quantitative evidence | Shared | `experiments/eval_registry/sources.csv`, `experiments/eval_registry/eval_results_master.csv` |
| Core figures and bibliography | Shared first, copied only when a target needs format-specific assets | `paper/src/figures/`, `paper/src/references.bib` |
| Long paper narrative | Separate | `paper/src/main.tex`, `paper/src/main_zh.tex`, target dashboard in `paper/targets/long_paper/` |
| NYCU thesis narrative | Separate | `paper/targets/nycu_thesis/latex/main.tex` |

Rationale: the two manuscripts need different pacing and evidence density. The
long paper should stay concise and claim-driven; the thesis should include
background, implementation detail, negative results, and research process. They
should not share one monolithic LaTeX source, because thesis chapters and paper
sections will diverge. They should share notes, registry-backed numbers,
figures, and citation provenance so the scientific claims do not drift.

## Directory Roles

| Path | Role |
|---|---|
| `AGENTS.md` | Scope-local operating rules for future agents editing `paper/`. Read this before changing paper files. |
| `src/` | Buildable bilingual long-paper source. English and Chinese drafts are synchronized here. |
| `targets/` | Target-specific manuscript packages. Each conference, journal, proposal, or degree requirement gets its own subfolder. |
| `targets/long_paper/` | Long-paper target dashboard. Current buildable source remains in `paper/src/`. |
| `targets/nycu_thesis/` | NYCU master's thesis target, including process notes and a template-based LaTeX project. |
| `notes/` | Shared HFRVLA research notes used across targets: method contract, RQs, contribution framing, literature notes, and protocols. |
| `deliberations/` | Skill/subagent outputs: reviewer simulations, writer/reviewer debate logs, claim audits, citation audits, and decision records. |
| `templates/` | Raw external format templates and template-specific notes. Do not draft thesis content here. |
| `thesis/` | Legacy non-template thesis draft and generated registry tables. Keep for migration reference until fully superseded. |
| `build/` | Generated PDFs from the long-paper bilingual build. |
| `proposal.md` | Scratch proposal note; promote into `targets/<target>/` before treating it as a maintained target. |

## Source-Of-Truth Rule

Use this order when resolving disagreements:

1. `experiments/eval_registry/eval_results_master.csv` for numbers.
2. `paper/notes/methodology_blueprint.md` for method framing.
3. `paper/notes/research_questions.md` for active RQs and ablations.
4. `paper/notes/contribution_statement.md` for wording guardrails.
5. Target dashboards under `paper/targets/<target>/README.md`.
6. Buildable manuscript source.

## Maintenance Rule

For a new target, create:

```text
paper/targets/<target_name>/
```

Use that folder to record the target's research question, evidence standard,
format requirements, claim boundaries, figure/table plan, and writing backlog.
Only after the target logic is clear should content be synchronized into the
LaTeX manuscript under `src/`, `thesis/`, or another buildable source folder.

The current long paper target is tracked at:

```text
paper/targets/long_paper/
```

Its current buildable source is:

```text
paper/src/main.tex
paper/src/main_zh.tex
```

The NYCU master's thesis target is maintained at:

```text
paper/targets/nycu_thesis/
```

Its current template-based PDF is built from:

```text
paper/targets/nycu_thesis/latex/main.tex
```

and outputs to:

```text
paper/targets/nycu_thesis/latex/build/main.pdf
```

## Build Commands

```bash
# Long paper, English and Traditional Chinese
cd paper && bash build_bilingual.sh

# NYCU thesis
cd paper/targets/nycu_thesis/latex && bash build.sh
```
