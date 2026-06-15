# HFRVLA: Hierarchical Fast-Reactive VLA

Updated: 2026-06-15

This repository maintains HFRVLA, a LeRobot-native research project for testing
whether a small fast wrist correction module can improve frozen SmolVLA action
chunks without fine-tuning the slow planner.

Current deployment contract:

```text
a_final = a_base + alpha * clip(delta_a)
```

The active HFRVLA implementation uses the `fast_wrist` and
`fast_wrist_chunk` correction modes. These names are implementation identifiers;
the human-facing method name remains HFRVLA. The older GRU/gate/contact
auxiliary design is retired research history and is kept only for checkpoint
provenance.

## Where To Start

| Need | Current entry point |
|---|---|
| Maintained docs map | `docs/README.md` |
| Training and evaluation workflow | `docs/training.md` |
| Paper workspace map | `paper/README.md` |
| Method contract and paper framing | `paper/notes/methodology_blueprint.md` |
| Implementation contract | `paper/notes/implementation_spec.md` |
| Visual briefing | `docs/hfrvla_experiment_briefing.html` |
| Eval registry manifest | `experiments/eval_registry/sources.csv` |
| Eval registry master table | `experiments/eval_registry/eval_results_master.csv` |

Local presentation URL when the docs server is running:

```text
http://127.0.0.1:8179/hfrvla_experiment_briefing.html
```

## Project Layout

| Path | Purpose |
|---|---|
| `policy/lerobot_policy_hfrvla/` | LeRobot policy plugin. |
| `scripts/` | Recording, training, packaging, eval, registry, and figure scripts. |
| `docs/` | Maintained operational docs and presentation source. Retired docs live in `docs/archive/`. |
| `experiments/eval_registry/` | Tracked manifest and regenerated results table for paper claims. |
| `paper/` | Manuscript workspace for the long paper and NYCU thesis. |
| `checkpoints/` | Local datasets, caches, checkpoints, and package outputs. |
| `outputs/` | Local run outputs and ignored eval artifacts. |

## Active Data And Model Contract

Verified slow planner:

```text
HuggingFaceVLA/smolvla_libero
```

Verified HFRVLA dataset:

```text
checkpoints/HFRVLA_libero_v1_merged_reindexed
```

Expected cached feature shapes:

| Feature | Shape |
|---|---:|
| `observation.state` | `(8,)` |
| `action` | `(7,)` |
| `observation.extra.a_base` | `(7,)` |
| `observation.extra.k_idx_norm` | `(1,)` |
| `observation.extra.z_goal` | `(960,)` |
| `observation.extra.z_phase` | `(480,)` |
| `observation.extra.dino_patches` | `(196, 384)` |

`contact_label` remains in the shared dataset for legacy compatibility, but the
active HFRVLA correction losses do not use a contact auxiliary head.

## Common Commands

Install the policy plugin after source changes:

```bash
cd ~/Robotic_infra/lerobot
uv pip install -e ~/Patrick/VLA_research/Hierachical_fast_reactive/policy/lerobot_policy_hfrvla
cd ~/Patrick/VLA_research/Hierachical_fast_reactive
```

Use local/offline roots for restricted runs:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HFRVLA_TMP_ROOT=$HOME/tmp/hfrvla
export HF_DATASETS_CACHE=$HFRVLA_TMP_ROOT/hf_datasets
export TMPDIR=$HFRVLA_TMP_ROOT/tmp
```

Run registry check before changing paper numbers:

```bash
python3 scripts/build_eval_results_master.py --check
```

Rebuild the paper-facing experiment briefing:

```bash
python3 scripts/generate_training_presentation.py
python3 scripts/generate_training_presentation.py --check
```

Build paper targets:

```bash
# Long paper, English + Traditional Chinese
cd paper && bash build_bilingual.sh

# NYCU thesis
cd paper/targets/nycu_thesis/latex && bash build.sh
```

## Manuscript Maintenance Policy

There are two maintained manuscript targets:

| Target | Source | Role |
|---|---|---|
| Long paper | `paper/src/main.tex`, `paper/src/main_zh.tex` with target notes in `paper/targets/long_paper/` | Concise paper-style manuscript for workshop/arXiv/conference adaptation. |
| NYCU master's thesis | `paper/targets/nycu_thesis/latex/main.tex` | Longer degree-oriented manuscript with process, background, and appendix detail. |

Maintain them separately at the manuscript level, but share one evidence base:
`paper/notes/`, `experiments/eval_registry/`, `paper/src/figures/`, and verified
bibliography entries. See `paper/README.md` for the detailed rule.

## Safety For Future Edits

- Do not reintroduce GRU/gate/contact as the active method unless explicitly
  requested.
- Do not manually edit `experiments/eval_registry/eval_results_master.csv`;
  update `sources.csv` and regenerate it.
- Do not hand-edit generated PDFs or build products.
- Keep quantitative paper claims tied to the eval registry.
