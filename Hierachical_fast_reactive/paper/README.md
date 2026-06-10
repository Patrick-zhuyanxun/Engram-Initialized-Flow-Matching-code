# HFRVLA Paper Workspace

This directory contains multiple paper-facing outputs for the HFRVLA project.
Do not treat every file under `paper/` as the same manuscript target.

## Directory Roles

| Path | Role |
|---|---|
| `AGENTS.md` | Scope-local operating rules for future agents editing `paper/`. Read this before changing paper files. |
| `src/` | Bilingual workshop/arXiv-style paper source. English and Chinese drafts are synchronized here. |
| `targets/` | Target-specific manuscript packages. Each conference, journal, proposal, or degree requirement gets its own subfolder. |
| `targets/nycu_thesis/` | NYCU master's thesis target, including process notes and a template-based LaTeX project. |
| `notes/` | Shared HFRVLA research notes used across targets: method contract, RQs, contribution framing, literature notes, and protocols. |
| `deliberations/` | Skill/subagent outputs: reviewer simulations, writer/reviewer debate logs, claim audits, citation audits, and decision records. |
| `templates/` | Raw external format templates and template-specific notes. Do not draft thesis content here. |
| `thesis/` | Legacy non-template thesis draft and generated registry tables. Keep for migration reference until fully superseded. |
| `build/` | Generated PDFs from the workshop/arXiv paper build. |
| `proposal.md` | Scratch proposal note; promote into `targets/<target>/` before treating it as a maintained target. |

## Maintenance Rule

For a new target, create:

```text
paper/targets/<target_name>/
```

Use that folder to record the target's research question, evidence standard,
format requirements, claim boundaries, figure/table plan, and writing backlog.
Only after the target logic is clear should content be synchronized into the
LaTeX manuscript under `src/`, `thesis/`, or another buildable source folder.

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
