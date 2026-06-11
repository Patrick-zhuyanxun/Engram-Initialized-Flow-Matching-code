# HFRVLA Paper Narrative Rewrite Design

Date: 2026-06-11

## Purpose

Rewrite the HFRVLA paper-facing manuscripts so they read as research papers
rather than engineering notes. The main text should explain the method and
evidence in human-readable terms. Exact implementation identifiers should be
kept for reproducibility, but moved out of the narrative path.

The user-approved goal is a complete and correct paper, not a string-level
cleanup.

## Source Order

The canonical short narrative is the bilingual workshop/arXiv-style source:

- `paper/src/main.tex`
- `paper/src/main_zh.tex`

The NYCU thesis target is the expanded version of the same narrative:

- `paper/targets/nycu_thesis/latex/Sections/4.Methodology.tex`
- `paper/targets/nycu_thesis/latex/Sections/5.Experiments.tex`
- `paper/targets/nycu_thesis/latex/Sections/appendix.tex`

Shared claims must stay aligned across both targets. The short paper should
establish the clean story first; the thesis should expand it without moving
engineering identifiers back into the main flow.

## Main Writing Rule

Main text should use conceptual paper language:

- frozen LIBERO-adapted SmolVLA planner
- LeRobot-based implementation
- cached slow-planner context
- wrist patch features
- chunk position
- trained HFRVLA checkpoint
- frozen SmolVLA baseline

Main text should avoid implementation identifiers such as:

- Hugging Face repository IDs
- Python package or plugin names
- dataset root paths
- checkpoint basenames
- command-line flags
- dataset column names such as `observation.extra.*`

Those identifiers belong in an appendix or reproducibility section.

## Math Scope

The main text should keep only the variables that define the method:

- `a_{\mathrm{base}}`
- `\hat{\Delta a}`
- `\alpha`
- `\delta_{\max}`
- `a_{\mathrm{final}}`

The core equation remains:

```latex
a_{\mathrm{final}}
= a_{\mathrm{base}}
+ \alpha \cdot \mathrm{clip}(\hat{\Delta a}, -\delta_{\max}, \delta_{\max})
```

Internal cached feature names such as `z_goal`, `z_phase`, and
`k_idx_norm` should be described in prose as slow-planner context or chunk
position unless the appendix is explicitly documenting the implementation
contract.

## Rewrite Approach

Use paragraph-level rewriting, not find-and-replace.

For each paragraph that currently exposes implementation artifacts:

1. Identify the paragraph role: problem framing, method, experiment protocol,
   result claim, limitation, or reproducibility detail.
2. Keep method and result paragraphs in conceptual language.
3. Move reproducibility material to the appendix.
4. Keep quantitative claims unchanged unless verified against the eval registry.
5. Preserve the distinction between synchronous sweeps and the async
   planner-delay stress test.

## Appendix Plan

Add or update an appendix section named:

```text
Implementation and Reproducibility Details
```

This section may include:

- exact frozen planner identifier
- LeRobot implementation package/plugin identifier
- dataset field names and shapes
- local dataset root
- main checkpoint identifiers
- eval registry file and source manifest
- relevant training or evaluation flags

The main text may refer to this section with one short sentence, for example:

```latex
Implementation identifiers, dataset field names, and checkpoint paths are
reported in Appendix~\ref{app:implementation-reproducibility} for
reproducibility.
```

## Target-Specific Design

### `paper/src`

The bilingual short paper should become the canonical narrative:

- Abstract states the stale-chunk problem, frozen planner, fast wrist residual,
  main Spatial evidence, and limitation without repository IDs.
- Introduction names the system conceptually and avoids implementation tags.
- Method explains the frozen planner, cached slow context, DINO wrist features,
  residual head, and clipped merge in research language.
- Experiments protocol uses method and baseline names, not checkpoint basenames.
- Appendix stores implementation and reproducibility identifiers.
- English and Traditional Chinese versions remain synchronized in claims,
  structure, and result numbers.

### NYCU thesis target

The thesis can be more detailed, but the detail must be layered:

- Methodology chapter explains the method first.
- Feature-schema tables should either become conceptual tables in the chapter
  or move to appendix if they expose raw dataset keys.
- Experiments chapter explains protocol variables in human terms and avoids
  raw registry column names in the main flow.
- Appendix carries the full implementation contract.

## Non-Goals

This rewrite does not:

- introduce new citations
- change numerical claims without registry verification
- modify figures
- change model architecture or training code
- promote unfinished ablations into main claims
- mix LIBERO-Spatial-only claims with Spatial+Object claims

## Verification

After implementation, run:

```bash
(cd paper && bash build_bilingual.sh)
(cd paper/targets/nycu_thesis/latex && bash build.sh)
```

If result numbers are touched, also run:

```bash
python3 scripts/build_eval_results_master.py --check
```

## Success Criteria

The rewrite is complete when:

- abstract and introduction no longer expose repository IDs, package names,
  checkpoint basenames, or dataset field names
- method sections read as a method description, not an implementation contract
- experiments sections describe protocols and baselines conceptually
- reproducibility identifiers are preserved in appendix
- bilingual short paper and NYCU thesis remain aligned
- both build commands succeed
