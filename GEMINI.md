# Project Overview

This is a multi-package robotics and machine learning research workspace. The active research focus is **State-Aware Prior Initialization for Low-Step Flow Matching Policies**. It is implemented as a plugin for the LeRobot framework. The project aims to improve VLA (Vision-Language-Action) models by replacing standard Gaussian noise initialization in flow matching with a state-aware prior, effectively evaluated in the low-step regime.

## Key Directories and Packages

*   **`lerobot/`**: The main LeRobot framework (upstream HuggingFace repo). Contains policy implementations, drivers, and CLI scripts for training/eval.
*   **`lerobot_policy_eifm/`**: The active research plugin implementing the policy. It uses a Florence-2/X-VLA backbone and an action prior retrieval mechanism.
*   **`eifm/`**: Tools for building and analyzing the memory table.
*   **`LIBERO/`**: Lifelong Robot Learning benchmark containing 130 MuJoCo manipulation tasks across 4 suites.
*   **`outputs/`**: Training run outputs.
*   **`papers/pp_fm/`**: LaTeX workspace for the research paper. Contains the `src/` directory for LaTeX source and the `notes/` directory for the methodology and defense truth sources.

## Architecture: State-Aware Prior Initialization

**Core idea**: Shift from uninformed Gaussian noise or semantic-only priors to **State-Aware Prior Initialization**, targeting the low-step flow matching regime.
1. **State-Aware Prior Construction**: The retrieval key explicitly incorporates Visual context, Proprioception, and Language instruction to mitigate spatial blindness.
2. **Prior Injection**: The retrieved continuous action chunk prior is injected at the ODE source state ($x_0 = \lambda \cdot \mathbf{W}(x_{prior}) + \sigma \cdot \epsilon$), reducing the effective transport path without vulnerable intermediate hidden-state injection.
3. **Practical Realization (Multi-Head Hashing)**: Continuous states are tokenized. Multiple deterministic hash heads route token subsets to fetch offline-consolidated memory slots in $O(1)$ time, replacing costly dense retrieval.
4. **Confidence Gating**: Memory slots track count and action variance. A confidence gate gracefully degrades to vanilla Gaussian noise ($\lambda \to 0$) during out-of-distribution cache misses or high-variance aliasing.

### Training Parameterization
The implementation uses a **target-prediction (data-prediction)** parameterization. The Transformer directly predicts the clean target action chunk $\hat{\mathbf{A}}_1$.

### Current Locked Concepts (aligned 2026-04-09)
*   **Primary Claim**: State-aware prior outperforms language-only priors by resolving spatial ambiguity, and proves critical in low-step (S=1,3,5) inference.
*   **Positioning**: A deployment-friendly alternative to Dense Retrieval (OptimusVLA), NOT the inventor of prior-guided flow matching.
*   **Data scope**: Empirical validation is strictly on **LIBERO** (especially LIBERO-Spatial) with a strict leakage-aware train-eval split.

## Development & Paper Workflow

### ⚠️ IMPORTANT: Note Synchronization Rule
When updating paper concepts, claims, baseline positioning, or aligning the paper with the implementation, **ALWAYS synchronously update the corresponding note files in `papers/pp_fm/notes/`** (specifically `research_questions.md`, `methodology_blueprint.md`, and `reviewer_defense.md`). Do not just modify the `src/` LaTeX files. Treat these three markdown files as the **primary truth source** for the project's methodology and defense strategy.

### Building and Running

All primary Python commands should be run from within the `lerobot/` directory.

**Installation:**
```bash
cd lerobot
pip install -e ".[dev]"
uv pip install -e ../lerobot_policy_eifm
```

**Training & Evaluation:**
```bash
cd lerobot
bash ../eifm/train_eifm.sh
lerobot-eval --policy.path=../outputs/... --env.type=libero
```

**Paper Compilation:**
```bash
cd papers/pp_fm
tectonic -X build
```