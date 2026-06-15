# HFRVLA Docs Index

Updated: 2026-06-15

This is the active entry point for HFRVLA documentation. Use this file first
when deciding which document is current.

## Active Source Of Truth

| Purpose | Current file |
|---|---|
| Method / implementation contract | `paper/notes/implementation_spec.md` |
| Training and evaluation workflow | `docs/training.md` |
| Paper-facing visual briefing | `docs/hfrvla_experiment_briefing.html` |
| Editable slide source | `docs/presentations/hfrvla-training-open-slide/slides/hfrvla-training/index.tsx` |
| LeRobot/plugin/dataset context | `docs/lerobot_hfrvla_context.md` |
| Experiment registry manifest | `experiments/eval_registry/sources.csv` |
| Regenerated experiment master table | `experiments/eval_registry/eval_results_master.csv` |

## Current Method Contract

The active HFRVLA path is frozen `HuggingFaceVLA/smolvla_libero` plus a small
fast wrist correction module:

```text
a_final = a_base + alpha * clip(delta_a)
```

Current active implementation modes are `fast_wrist` and `fast_wrist_chunk`.
These are code identifiers, not a second public method name. The older
GRU/gate/contact auxiliary design and Stage A/B/C gated losses are retired
research history and should not be used for new runs unless explicitly revived.

## Presentation

Open the current presentation locally:

```text
http://127.0.0.1:8179/hfrvla_experiment_briefing.html
```

The HTML is generated. `docs/training_presentation.html` is retained only as a
compatibility redirect. Edit the open-slide source under
`docs/presentations/hfrvla-training-open-slide/`, then rebuild with:

```bash
python3 scripts/generate_training_presentation.py
python3 scripts/generate_training_presentation.py --check
```

## Archive

Retired docs live under `docs/archive/`. They are kept for traceability, not as
active instructions. See `docs/archive/README.md`.
