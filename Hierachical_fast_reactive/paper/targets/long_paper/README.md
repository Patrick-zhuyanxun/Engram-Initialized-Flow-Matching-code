# Long Paper Target - HFRVLA

Updated: 2026-06-15

This target controls the concise HFRVLA paper manuscript. The current buildable
source still lives in the shared bilingual paper source folder:

```text
paper/src/main.tex
paper/src/main_zh.tex
```

Build both with:

```bash
cd paper && bash build_bilingual.sh
```

## Role

The long paper should be a claim-driven manuscript for workshop, arXiv,
conference, or journal adaptation. It should not carry the full thesis research
process. Keep it focused on:

1. Stale action chunks as the problem.
2. Frozen SmolVLA plus fast wrist correction as the intervention.
3. `a_final = a_base + alpha * clip(delta_a)` as the deployment contract.
4. Registry-backed LIBERO evidence.
5. Clear limitations around real hardware, multi-seed coverage, and long-chunk
   failure modes.

## Relationship To The Thesis

Maintain separately from the NYCU thesis at the manuscript level:

| Shared with thesis | Separate from thesis |
|---|---|
| `paper/notes/` method and RQ notes | Section order and depth |
| Eval registry numbers | Background/tutorial material |
| Core figures from `paper/src/figures/` | Thesis process history and failed branches |
| Verified bibliography entries | Administrative NYCU template metadata |

The thesis may expand failed attempts, implementation details, and full
research-process reasoning. The long paper should only include such material
when it directly supports a claim or limitation.

## Source Of Truth

Use this order before editing claims:

1. `experiments/eval_registry/eval_results_master.csv`
2. `experiments/eval_registry/sources.csv`
3. `paper/notes/methodology_blueprint.md`
4. `paper/notes/research_questions.md`
5. `paper/notes/contribution_statement.md`
6. `paper/src/main.tex` and `paper/src/main_zh.tex`

## Maintenance Rules

- Keep `main.tex` and `main_zh.tex` synchronized for shared claims, structure,
  results, limitations, and citations.
- Keep quantitative claims provisional unless the user confirms the experiment
  set is complete.
- Distinguish synchronous sweeps from async-timestep planner-delay experiments.
- Do not combine Spatial-only and Spatial+Object claims without an explicit
  table/protocol note.
- Do not reintroduce GRU/gate/contact as active method framing unless the
  project direction changes.

## Current Open Decision

The buildable source can remain in `paper/src/` while the venue is undecided. If
the long paper receives a concrete venue with a style file or page limit, create
a target-local source folder under `paper/targets/long_paper/` and keep
`paper/src/` as shared draft source or archive it after migration.
