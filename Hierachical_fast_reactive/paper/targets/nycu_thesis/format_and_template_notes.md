# NYCU Thesis Format And Template Notes

Last updated: 2026-06-10

Reference template:

```text
https://www.overleaf.com/latex/templates/nycu-thesis-template/jgcmcnchmbrc
```

The public Overleaf metadata identifies this as an NYCU thesis template and
notes support for English TOC and Chinese table of contents. This repository now
contains a complete NYCU-template rebuild under `paper/targets/nycu_thesis/latex/`,
created from the local archive
`paper/templates/nycu_thesis/NYCU_thesis_template__1_.zip`.

## Current Local Thesis Source

```text
paper/targets/nycu_thesis/latex/main.tex
paper/targets/nycu_thesis/latex/build.sh
paper/targets/nycu_thesis/latex/build/main.pdf
```

## Format Checklist

| Item | Status | Notes |
|---|---|---|
| Title page | Template-based draft | Uses `covers/front_var.tex`; advisor/department metadata still needs confirmation. |
| Chinese title | Draft | Needs final title after RQ stabilizes. |
| English title | Draft | Needs final title after RQ stabilizes. |
| Advisor / department / degree fields | Partial | Placeholder values exist; replace after administrative metadata is known. |
| Chinese abstract | Draft | Rewritten with stale-action problem framing. |
| English abstract | Draft | Mirrors the Chinese abstract. |
| Keywords | Draft | Technical keywords included. |
| Table of contents | Present | Chinese TOC enabled through `\toggletrue{toc-use-cn}`. |
| List of figures | Present | Must ensure figure captions are thesis-quality. |
| List of tables | Present | Tables should be success-rate/protocol focused. |
| Chapter structure | Template-based draft | Seven chapters plus appendix are present. |
| References | Present | Metadata must remain verified. |
| Appendix | Present but evolving | Use for registry tables and unfinished experiments. |

## Proposed Thesis Chapter Skeleton

1. Introduction
   - Problem: expensive VLA inference and stale action chunks.
   - Motivation: bounded local correction instead of full replanning.
   - Research questions and contributions.
2. Related Work
   - VLA and robot foundation policies.
   - Action chunking and real-time execution.
   - Correction heads and fast-slow policies.
   - Wrist vision and dense DINO features.
   - Research gap.
3. Methodology
   - HFRVLA system overview.
   - Frozen SmolVLA planner.
   - Wrist DINO feature extraction.
   - Fast wrist correction module.
   - Training objective and residual merge.
4. Implementation
   - LeRobot policy plugin.
   - HFRVLA dataset and feature cache.
   - Checkpoint packaging.
   - Evaluation registry.
5. Experiments
   - LIBERO-Spatial protocol.
   - Synchronous plan=50 execution sweep.
   - Synchronous matched chunk sweep.
   - Residual alpha/clip calibration.
   - Async-timestep planner-delay stress test.
6. Results
   - Success-rate-first figures and exact-value tables.
   - Failure patterns and calibration behavior.
7. Discussion
   - Interpretation, limitations, and failure modes.
   - Causal visual limitations.
   - Future experiments.
8. Conclusion
   - Conservative summary of supported claims.

Appendices:

- Full experiment registry.
- Unfinished experiments.
- Extra implementation details.
- DINO visualization caveats.

## Formatting Principles For This Thesis

- The thesis can be more detailed than the workshop paper. It should document
  failed designs, protocol changes, and why decisions were made.
- Main-text results should remain success-rate-first.
- Avoid margin-heavy plots unless they answer a thesis question directly.
- Mark unfinished work explicitly. Do not hide it inside vague discussion text.
- Keep exact paths and commands in appendix or implementation sections so the
  research process is reproducible.
