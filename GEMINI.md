# Project Overview

This is a multi-package robotics and machine learning research workspace. The active research focus is **State-Aware Engram-Initialized Flow Matching (State-Aware EI-FM)**, previously known just as EI-FM or PP-FM. It is implemented as a plugin for the LeRobot framework. The project aims to improve VLA (Vision-Language-Action) models by replacing standard Gaussian noise initialization in flow matching with an $O(1)$ retrieved sensorimotor reflex prior using Multi-Head Hashing (MHH).

## Key Directories and Packages

*   **`lerobot/`**: The main LeRobot framework (upstream HuggingFace repo). Contains policy implementations, drivers, and CLI scripts for training/eval.
*   **`lerobot_policy_eifm/`**: The active research plugin implementing the EI-FM policy. It uses a Florence-2/X-VLA backbone and an engram hash table.
*   **`eifm/`**: Tools for building and analyzing the engram table (e.g., `build_engram_table.py`, `ngram_extractor.py`).
*   **`LIBERO/`**: Lifelong Robot Learning benchmark containing 130 MuJoCo manipulation tasks across 4 suites.
*   **`checkpoints/eifm-libero/`**: Base checkpoints adapted for EI-FM.
*   **`outputs/`**: Training run outputs.
*   **`papers/pp_fm/`**: LaTeX workspace for the research paper. Contains the `src/` directory for LaTeX source and the `notes/` directory for the methodology and defense truth sources.

## Architecture: State-Aware EI-FM

**Core idea**: Shift from language-only static engrams to **State-Aware Embodied Reflex Memory**.
1. **State Tokenizer**: Discretize visual, proprioceptive, and language features into sensorimotor tokens.
2. **Multi-Head Hashing (MHH)**: Use distinct hash functions for different token subsets to retrieve continuous action chunk priors in $O(1)$ time, avoiding combinatorial explosion and hash collisions.
3. **Continuous Prior as ODE Initialization**: The retrieved prior acts as a conditional source initialization ($x_0 = \lambda \cdot x_{reflex} + \sigma \cdot \epsilon$), sidestepping PreNorm dilution inside the Transformer.
4. **Confidence Gating**: Track activation count and action variance for each memory slot. If variance is high or count is low, $\lambda \to 0$, gracefully degrading to standard Gaussian noise.

### Training Parameterization
The implementation uses a **target-prediction (data-prediction)** parameterization rather than standard velocity regression. The Transformer directly predicts the clean target action chunk $\hat{A}_1$.

### Current Locked Concepts (aligned 2026-04-09)
*   **Engram key**: Composite key (verb phrases sorted + joined) + State Tokenizer.
*   **Loss function**: Target-prediction.
*   **Centering**: Global-mean centering to reduce `DomainAwareLinear` bias.
*   **Data scope**: Empirical validation is strictly on **LIBERO** to isolate the effect of MHH initialization.

## Development & Paper Workflow

### ⚠️ IMPORTANT: Note Synchronization Rule
When updating paper concepts, claims, baseline positioning, or aligning the paper with the implementation in the EI-FM project, **ALWAYS synchronously update the corresponding note files in `papers/pp_fm/notes/`** (specifically `research_questions.md`, `methodology_blueprint.md`, and `reviewer_defense.md`). Do not just modify the `src/` LaTeX files. Treat these three markdown files as the **primary truth source** for the project's methodology and defense strategy.

### Building and Running

All primary Python commands should be run from within the `lerobot/` directory.

**Installation:**
```bash
cd lerobot
pip install -e ".[dev]"
uv pip install -e ../lerobot_policy_eifm
```
*Note: Re-run the plugin installation command after any code changes to `lerobot_policy_eifm`.*

**Training & Evaluation:**
```bash
cd lerobot
bash ../eifm/train_eifm.sh
lerobot-eval --policy.path=../outputs/eifm_libero_spatial_v4/checkpoints/005000/pretrained_model --env.type=libero
```

**Paper Compilation:**
```bash
cd papers/pp_fm
tectonic -X build
```