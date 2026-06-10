# Research Log

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-06-04 | bootstrap | Installed Orchestra AI Research Skills package `@orchestra-research/ai-research-skills@1.6.0`, which created 95 skill symlinks under `~/.codex/skills`. Loaded `0-autoresearch-skill/SKILL.md`. The required session-bound continuity loop could not be created because this Codex session exposes no `cron.add` tool or `/loop` command. Initialized autoresearch state from local HFRVLA memory, paper notes, and `experiments/eval_registry/eval_results_master.csv`; no experiments were launched. |
| 2 | 2026-06-04 | outer-loop | Audited current HFRVLA training/inference paths for weak long-horizon gains. Confirmed eval patterns, inspected `select_action()`, fast-wrist losses, fast-cache v3 chunk construction, packaged configs, and targeted tests in the LeRobot venv. Wrote `to_human/hfrvla_training_inference_audit_2026-06-04.md`. Main conclusion: keep frozen SmolVLA + wrist residual framework, but align dataset/loss/runtime around stale generated chunk correction and deployed `alpha * clip(delta)` merge. |

<!-- Entry types:
  bootstrap  - initial scoping, literature search, hypothesis formation
  inner-loop - experiment run and result
  outer-loop - synthesis, reflection, direction decision
  pivot      - change in research direction
  report     - progress presentation generated
  conclude   - decision to finalize and write paper
-->
