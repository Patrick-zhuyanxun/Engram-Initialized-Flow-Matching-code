# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

This is a multi-package robotics/ML research workspace with the following packages:

- **`lerobot/`** — The main LeRobot framework (upstream HuggingFace repo). Contains 16+ policy implementations, robot/camera/motor drivers, dataset tooling, and training/eval CLI scripts.
- **`lerobot_policy_eifm/`** — **The active research focus.** A LeRobot plugin implementing the EI-FM policy (Engram-Initialized Flow Matching). Florence-2/X-VLA backbone + EE6D action space + engram hash table replacing `randn` noise initialization.
- **`lerobot_robot_tm12/`** / **`lerobot_robot_tm/`** — LeRobot plugins for Techman robot arm integration.
- **`LIBERO/`** — Lifelong Robot Learning benchmark (130 MuJoCo manipulation tasks across 4 suites).
- **`eifm/`** — Engram table build tools: `build_engram_table.py`, `ngram_extractor.py`, `engram_analysis.py`. Engram table at `eifm/engram_table_projected.pt` (14 keys, dim=1024, centered).
- **`checkpoints/eifm-libero/`** — X-VLA base checkpoint adapted for EI-FM. Contains `config.json`, `model.safetensors`, `policy_preprocessor.json`.
- **`outputs/`** — Training run outputs (e.g. `eifm_libero_spatial_v4/`).
- **`papers/`** — LaTeX paper workspace (see below).

Each subdirectory is an independently installable Python package with its own `pyproject.toml`.

## Common Commands

All commands run from within the `lerobot/` directory (which contains the `.venv`).

### Install
```bash
# Install lerobot with development extras
cd lerobot && pip install -e ".[dev]"

# Install EI-FM plugin (re-run after any code changes to lerobot_policy_eifm)
cd lerobot && uv pip install -e ../lerobot_policy_eifm
```

### Training & Evaluation
```bash
# Train EI-FM on LIBERO-Spatial (uses train_eifm.sh)
bash eifm/train_eifm.sh
# Output dir: outputs/eifm_libero_spatial_v<N>/
# Base checkpoint: checkpoints/eifm-libero/
# Engram table: eifm/engram_table_projected.pt

# Evaluate a trained policy
lerobot-eval --policy.path=outputs/eifm_libero_spatial_v4/checkpoints/005000/pretrained_model --env.type=libero

# Generic train command
lerobot-train --policy.type=eifm --dataset.repo_id=...
```

### Tests
```bash
# Run all tests (from lerobot/)
cd lerobot && pytest

# Run a single test file
pytest tests/policies/test_policies.py

# Run with device detection (tests auto-select CPU/GPU)
pytest tests/ -v
```

### Linting
```bash
# From lerobot/
ruff check .
ruff format .
```

### Papers (LaTeX / Tectonic)
```bash
# Compile pp_fm paper
cd papers/pp_fm && tectonic -X build
# Output: papers/pp_fm/src/main.pdf
```

## Architecture: LeRobot Plugin System

LeRobot uses a plugin discovery mechanism: external packages register policy/robot types via `pyproject.toml` entry points under `lerobot.policies` or `lerobot.robots`. After any code change to a plugin, **reinstall it**:
```bash
cd lerobot && uv pip install -e ../lerobot_policy_eifm
```
This is how `lerobot_policy_eifm` integrates without modifying the core `lerobot` package.

## Architecture: EI-FM Policy (`lerobot_policy_eifm`)

**Core idea**: Replace `x1 = torch.randn(...)` in flow-matching with a motor primitive embedding looked up from an O(1) engram hash table keyed by verb-phrase N-grams extracted from the language instruction.

Key files in `src/lerobot_policy_eifm/`:

| File | Purpose |
|------|---------|
| `configuration_eifm.py` | `EIFMConfig` — all hyperparams + EI-FM fields: `engram_path`, `engram_p_engram` (0.8), `engram_key_mode` ("multi"), `train_action_projections` |
| `modeling_eifm.py` | `EIFMPolicy` + `EIFMModel`: Florence-2 encoder → SoftPromptedTransformer action head with flow matching; `_get_engram_noise()`, `_set_current_instruction()`, `_apply_freezing()` |
| `processor_eifm.py` | Processor steps (registered): `libero_action_to_ee6d`, `xvla_rotation_6d_to_axis_angle`, `xvla_imagenet_normalize`, `xvla_add_domain_id`; factory `make_xvla_libero_pre_post_processors()` |
| `action_hub.py` | `EE6DActionSpace` — dim=20, gripper_idx=(9,19); pos MSE (×500) + rot6d MSE (×10) + gripper BCE (×1) |
| `utils.py` | `axis_angle_to_rot6d()`, `rotate6d_to_axis_angle()`, `mat2quat()`, `quat2axisangle()`, `mat_to_rotate6d()` |
| `soft_transformer.py` | `SoftPromptedTransformer` action head (24 layers, 16 heads, 30 domains) |

### Trainable Parameters (1.6M total — frozen everything else)

| Component | Params |
|-----------|--------|
| `soft_prompt_hub` | ~983K |
| `w_engram` (1024 → chunk×dim) | ~615K |

Vision encoder, language encoder, and policy transformer are all frozen. Controlled by `freeze_vision_encoder`, `freeze_language_encoder`, `train_policy_transformer`, `train_soft_prompts` in config.

### Action Format

| Stage | Format | Dims |
|-------|--------|------|
| LIBERO dataset | `[delta_pos(3), delta_rot_aa(3), gripper±1(1)]` | 7D |
| After `LiberoActionToEE6DProcessorStep` | `[pos(3), rot6d(6), g01(1)] × 2` | 20D |
| Model output | 20D ee6d | 20D |
| After `xvla_rotation_6d_to_axis_angle` (eval) | `[pos(3), axis_angle(3), gripper(1)]` | 7D |
| LIBERO env input | 7D | 7D |

### Eval Pipeline

```
model output (20D)
  → postprocessor (checkpoint JSON): unnormalizer + device
  → env_postprocessor (make_xvla_libero_pre_post_processors):
      xvla_rotation_6d_to_axis_angle  (20D → 7D)
  → LIBERO env action (7D)
```

### Engram Table

- Path: `eifm/engram_table_projected.pt`
- 14 keys, dim=1024, centered (global mean subtracted to reduce DomainAwareLinear bias dominance)
- Built from LIBERO demos projected through frozen `DomainAwareLinear` action_encoder (domain_id=3)
- Build script: `eifm/build_engram_table.py`

### Known Bugs & Fixes

**1. bf16 eval crash** (`TypeError: Got unsupported ScalarType BFloat16`):
- Cause: config `dtype="bfloat16"` but NumPy doesn't support bf16
- Fix: `select_action()` and `generate_actions()` return `.float()` before handing off to NumPy

**2. Action format mismatch → 0% success rate, arm flies upward**:
- Cause: `_prepare_action_targets()` was zero-padding 7D LIBERO actions to 20D instead of converting. This gave garbage rot6d targets (`[axis_angle, gripper, 0, 0]` in rot6d slots) and always-zero gripper BCE targets.
- Fix: Added `LiberoActionToEE6DProcessorStep` (`libero_action_to_ee6d`) using `axis_angle_to_rot6d()` (Rodrigues formula) to properly convert 7D → 20D before training. Added step to `make_xvla_libero_pre_post_processors()` and `policy_preprocessor.json`.

## Architecture: LeRobot Policies

All built-in policies live in `lerobot/src/lerobot/policies/<name>/` and follow a standard interface:
- `configuration_<name>.py` — Draccus config class
- `modeling_<name>.py` — `nn.Module` implementing `select_action()` and `forward()`

Policy extras (e.g., `pip install lerobot[smolvla]`) install additional dependencies declared in `pyproject.toml`.

## Papers Workspace (`papers/`)

LaTeX papers compiled with [Tectonic](https://tectonic-typesetting.github.io/).

```
papers/
├── pp_fm/          — Prior-Preserved Flow Matching (PP-FM) [= EI-FM in code]
│   ├── src/        — main.tex + sections/ (LaTeX source)
│   ├── notes/      — methodology_blueprint.md, literature_review.md, research_questions.md, etc.
│   └── Tectonic.toml
├── paper_02/       — TBD
├── paper_03/       — TBD
└── shared/
    ├── bib/        — shared .bib bibliography
    ├── figures/    — shared figures/assets
    └── templates/  — CoRL / IROS / IEEE LaTeX templates
```

**Naming note**: PP-FM (Prior-Preserved Flow Matching) is the paper name; EI-FM (Engram-Initialized Flow Matching) is the codebase name. They refer to the same method.

**Principles**:
- All Table numbers stay `[PLACEHOLDER]` until experiments are complete
- All citations verified via Zotero or WebSearch before writing
- `notes/` is the **ground truth** for current method — `src/sections/` LaTeX follows from it
- **When any concept changes** (new experiments, new findings, method adjustments): update `papers/pp_fm/notes/methodology_blueprint.md` first

### Current Locked Concepts (aligned 2026-04-09)

| Concept | Current Implementation |
|---------|----------------------|
| Engram key | **Composite key** — verb phrases sorted + joined (`pick_up_place`); no multi-key averaging |
| Loss function | **Target-prediction** — model directly predicts clean action; not velocity regression |
| Centering | **Global-mean centering** — subtract mean of all engrams to reduce `DomainAwareLinear` bias |
| Data scope | **LIBERO only** — cross-embodiment engram construction not yet claimed |

## Code Style

- **Python**: 3.12+
- **Line length**: 110
- **Formatter/linter**: `ruff` (rules: E, W, F, I, B, C4, T20, N, UP, SIM)
- MyPy is configured with strict checking on `envs`, `configs`, `optim`, `model`, `cameras`, `motors` modules; relaxed on most policy implementations

## Academic Research Skills

A suite of Claude Code skills for rigorous academic research, paper writing, peer review, and pipeline orchestration. Available via `@academic-research-skills`.

| Skill | Purpose | Key Modes |
|-------|---------|-----------|
| `deep-research` v2.5 | 13-agent research team | full, quick, socratic, review, lit-review, fact-check, systematic-review |
| `academic-paper` v2.5 | 12-agent paper writing | full, plan, revision, citation-check, format-convert, bilingual-abstract, writing-polish, full-auto, revision-coach |
| `academic-paper-reviewer` v1.5 | Multi-perspective paper review (5 reviewers + optional cross-model) | full, re-review, quick, methodology-focus, guided |
| `academic-pipeline` v2.8 | Full pipeline orchestrator | (coordinates all above) |

### Routing Rules

1. **academic-pipeline vs individual skills**: Use `academic-pipeline` for full research → write → review → revise → finalize. For single functions (just research, just write, just review), trigger the corresponding skill directly.
2. **deep-research vs academic-paper**: Complementary — `deep-research` is upstream (investigation + fact-checking), `academic-paper` is downstream (paper writing + bilingual abstracts). Recommended: `deep-research` → `academic-paper`.
3. **deep-research socratic vs full**: `socratic` = guided Socratic dialogue to clarify research question. `full` = direct report production. Suggest `socratic` when the research question is unclear.
4. **academic-paper plan vs full**: `plan` = chapter-by-chapter guided planning. `full` = direct paper production.
5. **academic-paper-reviewer guided vs full**: `guided` = Socratic review engaging the author. `full` = standard multi-perspective review report.

### Full Academic Pipeline

```
deep-research (socratic/full)
  → academic-paper (plan/full)
    → academic-paper-reviewer (full/guided)
      → academic-paper (revision)
        → academic-paper-reviewer (re-review, max 2 loops)
          → academic-paper (format-convert → final output)
          → AI Self-Reflection Report
```

### Key Rules

- All claims must have citations
- Evidence hierarchy respected (meta-analyses > RCTs > cohort > case reports > expert opinion)
- Contradictions disclosed with evidence quality comparison
- AI disclosure in all reports
- Default output language matches user input (Traditional Chinese or English)
