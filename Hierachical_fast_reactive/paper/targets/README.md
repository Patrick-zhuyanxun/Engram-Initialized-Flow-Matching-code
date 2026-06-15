# Paper Targets

Updated: 2026-06-15

`paper/targets/` separates manuscript goals from shared source material. HFRVLA
may produce a workshop paper, conference submission, thesis, defense slides, or
internal proposal; each target needs a different narrative, evidence threshold,
format, and appendix policy.

For the current project, maintain the long paper and thesis separately at the
target level, while sharing `paper/notes/`, the eval registry, verified
bibliography entries, and core figures.

## Current Targets

| Target | Folder | Output source | Purpose |
|---|---|---|---|
| Long paper | `long_paper/` | `paper/src/main.tex`, `paper/src/main_zh.tex` | Concise paper-style manuscript for workshop/arXiv/conference adaptation. |
| NYCU master's thesis | `nycu_thesis/` | `paper/targets/nycu_thesis/latex/main.tex` | Detailed research-process record and degree-oriented thesis plan. |

## Standard Target Folder Contents

Each target should eventually contain:

| File | Purpose |
|---|---|
| `README.md` | Target dashboard, current status, and maintenance rules. |
| `research_process.md` | Full Idea-to-Appendix research workflow. |
| `source_map.md` | Mapping from local artifacts to thesis chapters and claims. |
| `format_and_template_notes.md` | Venue or institution formatting requirements. |
| `source_materials.md` | Optional target-specific source/evidence index when the target has many local artifacts. |
| `latex/`, `src/`, or `manuscript/` | Buildable source for that specific target, if the target has a dedicated format. |
| `build/` | Target-local generated PDFs and intermediate build products, if the target source needs them. |

For conference targets, add deadline, page limit, anonymity, required style
file, and submission-specific claim restrictions.
