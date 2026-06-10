# Round 1 Methodology Review

Agent: `019eac02-6418-77f2-8488-04daa8efd36b` (Confucius)

Role: methodology reviewer, read-only.

## Verdict

Major revision. The draft had useful conference-paper material but did not yet
meet a clear Traditional Chinese master's-thesis standard.

## Major Findings

- Result provenance was mixed across checkpoints. Generated FWR-v2 synchronous
  sweeps, `hfrvla_30k` alpha/clip calibration, and async planner-delay rows
  should be split into distinct result families.
- The method description was under-specified for the implemented method. The
  thesis needed FWR-v1 versus FWR-v2, input dimensions, merge rule, loss terms,
  clipping rule, and exact training/inference paths.
- Statistical reporting was too strong for single-seed simulation. Use effect
  sizes and Wilson intervals where available, and say "numerical margin" unless
  replicated.
- Wrist-correction causality was not isolated. No-wrist/no-DINO, no-latent,
  patch masking, alpha=0/disable-fast, and FWR-v1/FWR-v2 ablations remain open.
- Dataset and reproducibility reporting needed thesis-level detail: dataset
  root, feature schema, cache schema v3, DINO preprocessing, offline training,
  seed, success definition, and registry workflow.
- Baseline framing needed stronger separation among SmolVLA, disable-fast,
  alpha=0/no-clip controls, FWR-v1, FWR-v2, and conceptual A2C2 comparisons.

## Revision Actions Taken After Round 1

- Added a result provenance table to the generated thesis artifacts.
- Added explicit synchronous-versus-async protocol wording.
- Split generated FWR-v2 main results from 30k/A2C2-alias diagnostics.
- Added limitations for causal visual ablations, multi-seed evidence, and
  checkpoint metadata gaps.
- Added a literature-source hierarchy separating peer-reviewed venue papers
  from recent preprints.
