# Round 1 Writer / Structure Review

Agent: `019eac02-38a0-7060-a0c9-879bf5263ac5` (Carson)

Role: thesis writer and structure reviewer, read-only.

## Verdict

The paper draft was useful as an arXiv/workshop article, but it was not yet a
clear master's thesis. The thesis needed a chapter-scale structure, full
experiment provenance, and a design-history narrative instead of only the
figure-first paper claims.

## Main Findings

- Add thesis chapters for introduction, related work, method, implementation,
  dataset/cache construction, training design, evaluation protocol, complete
  results, diagnostics, discussion, limitations, and appendices.
- Separate FWR-v1 and FWR-v2. The current draft blurred the simple wrist
  residual and the chunk-aware full-base-chunk model.
- Fix the training objective wording. The implementation uses both raw residual
  supervision and deployment-aligned final-action loss after alpha scaling and
  clipping, not only a simple residual MSE.
- Treat generated-checkpoint metadata carefully. Do not infer hyperparameters
  that are absent from the registry or checkpoint notes.
- Keep Object/combined older results as diagnostic or historical evidence, not
  as the main Spatial-only claim.
- Pending no-residual-clip rows must not be used as evidence.

## Revision Actions Taken After Round 1

- Added claim tiers in the introduction.
- Added a Traditional Chinese glossary.
- Added FWR-v1/FWR-v2 method section with trainable parameter counts.
- Replaced the simplified residual-MSE loss with the implemented SmoothL1
  delta/final/residual/clip objective.
- Added design evolution from gated/contact/GRU to FWR.
- Added result provenance table generation.
- Rewrote results language to use "numerical margin" for single-seed sweeps.
