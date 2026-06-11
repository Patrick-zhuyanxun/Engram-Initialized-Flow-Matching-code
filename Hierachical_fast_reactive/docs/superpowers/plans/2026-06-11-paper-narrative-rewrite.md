# HFRVLA Paper Narrative Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the HFRVLA paper-facing manuscripts into a detailed, correct, high-quality, readable paper while moving implementation identifiers to reproducibility appendices.

**Architecture:** Treat `paper/src/main.tex` and `paper/src/main_zh.tex` as the canonical short narrative, then synchronize the NYCU thesis as the expanded version. Main text uses conceptual research language; appendix sections preserve exact repo IDs, dataset fields, checkpoint names, and registry provenance.

**Tech Stack:** LaTeX, BibTeX/natbib, shell verification with `rg`, bilingual paper build via `paper/build_bilingual.sh`, NYCU thesis build via `paper/targets/nycu_thesis/latex/build.sh`.

---

### Task 1: Baseline Audit

**Files:**
- Read: `docs/superpowers/specs/2026-06-11-paper-narrative-rewrite-design.md`
- Read: `paper/src/main.tex`
- Read: `paper/src/main_zh.tex`
- Read: `paper/targets/nycu_thesis/latex/Sections/4.Methodology.tex`
- Read: `paper/targets/nycu_thesis/latex/Sections/5.Experiments.tex`
- Read: `paper/targets/nycu_thesis/latex/Sections/appendix.tex`

- [ ] **Step 1: Locate implementation identifiers in maintained manuscripts**

Run:

```bash
rg -n -F \
  -e "HuggingFaceVLA/smolvla_libero" \
  -e "lerobot_policy_hfrvla" \
  -e "observation.extra" \
  -e "checkpoints/" \
  -e "hfrvla_fwr_chunk_generated_seq2_b1024p3_50k" \
  -e "smolvla_libero" \
  -e 'z_{\mathrm{goal}}' \
  -e 'z_{\mathrm{phase}}' \
  -e "z_goal" \
  -e "z_phase" \
  -e "k_idx_norm" \
  -e "policy." \
  -e "vlm_model_name" \
  -e "expert_width_multiplier" \
  -e "num_vlm_layers" \
  -e "load_vlm_weights" \
  paper/src/main.tex \
  paper/src/main_zh.tex \
  paper/targets/nycu_thesis/latex/Sections/4.Methodology.tex \
  paper/targets/nycu_thesis/latex/Sections/5.Experiments.tex \
  paper/targets/nycu_thesis/latex/Sections/appendix.tex
```

Expected: Output identifies the exact paragraphs to rewrite or move.

- [ ] **Step 2: Confirm no numerical edits are needed before rewriting**

Run:

```bash
python3 scripts/build_eval_results_master.py --check
```

Expected: The registry check passes. If it fails, do not change numerical claims; continue only with wording and appendix placement.

### Task 2: Rewrite Canonical English Paper

**Files:**
- Modify: `paper/src/main.tex`

- [ ] **Step 1: Rewrite abstract and introduction in research language**

Edit `paper/src/main.tex` so the abstract and introduction use:

```text
LIBERO-adapted SmolVLA planner
frozen planner
slow planner
LeRobot-based implementation
cached slow-planner context
wrist patch features
trained HFRVLA policy
frozen SmolVLA baseline
```

Do not use repository IDs, package names, checkpoint basenames, dataset field
names, or raw path strings in the abstract or introduction.

- [ ] **Step 2: Rewrite Method section as a conceptual method description**

Edit `paper/src/main.tex` so the Method section has these conceptual units:

```latex
\subsection{Problem Setting}
\subsection{Frozen Planner and Cached Context}
\subsection{Wrist Patch Features}
\subsection{Fast Wrist Residual}
\subsection{Training Objective}
```

Keep only the core method variables in main text:

```latex
\abase, \deltah, \alpha, \dmax, \afinal
```

Describe slow-planner context in prose instead of naming internal cached tensor
fields. Keep the residual merge equation:

```latex
\afinal =
\abase + \alpha \cdot \mathrm{clip}(\deltah, -\dmax, \dmax).
```

- [ ] **Step 3: Rewrite Experiments protocol without checkpoint basenames**

Edit `paper/src/main.tex` so the protocol paragraph says:

```text
All quantitative claims are cross-checked against the eval registry. Main
results use LIBERO-Spatial with seed 42. The two synchronous sweeps use 100
episodes per row. The trained HFRVLA policy is evaluated with alpha=0.5 and
delta_max=0.2 unless otherwise specified, and the comparison baseline is the
frozen SmolVLA planner.
```

Do not include `hfrvla_fwr_chunk_generated_seq2_b1024p3_50k`,
`smolvla_libero`, or registry file paths in the main protocol paragraph.

- [ ] **Step 4: Add English implementation appendix**

Add a section after `\appendix` and before the existing appendix summary:

```latex
\section{Implementation and Reproducibility Details}
\label{app:implementation-reproducibility}

The main text describes the method at the algorithmic level. For
reproducibility, this appendix records the implementation identifiers used in
the local experiments. The frozen slow planner is
\texttt{HuggingFaceVLA/smolvla\_libero}. The LeRobot policy package is
\texttt{lerobot\_policy\_hfrvla}. The verified cached dataset root is
\texttt{checkpoints/HFRVLA\_libero\_v1\_merged\_reindexed}, with 1,693
episodes, 273,465 frames, and 40 LIBERO tasks.

The cached dataset stores top-view RGB, wrist RGB, robot state, expert action,
the selected base action, normalized chunk position, slow-planner task and
phase context, and DINOv3 wrist patch tokens. In the local LeRobotDataset v3
schema these fields are stored under keys including
\texttt{observation.images.image}, \texttt{observation.images.image2},
\texttt{observation.state}, \texttt{action},
\texttt{observation.extra.a\_base},
\texttt{observation.extra.k\_idx\_norm},
\texttt{observation.extra.z\_goal},
\texttt{observation.extra.z\_phase}, and
\texttt{observation.extra.dino\_patches}.

The main generated-checkpoint experiments use the trained HFRVLA artifact
\texttt{hfrvla\_fwr\_chunk\_generated\_seq2\_b1024p3\_50k}; the comparison
baseline is \texttt{smolvla\_libero}. Quantitative claims are tracked through
\texttt{experiments/eval\_registry/eval\_results\_master.csv} and its source
manifest.
```

### Task 3: Rewrite Canonical Traditional Chinese Paper

**Files:**
- Modify: `paper/src/main_zh.tex`

- [ ] **Step 1: Mirror the English narrative in Traditional Chinese**

Edit `paper/src/main_zh.tex` so it matches the English structure and claims.
Use readable paper language:

```text
針對 LIBERO 適配的凍結 SmolVLA 規劃器
凍結慢速規劃器
基於 LeRobot 的實作
快取的慢速規劃器脈絡
腕部 patch features
訓練後的 HFRVLA policy
凍結 SmolVLA baseline
```

Avoid repository IDs, package names, checkpoint basenames, dataset field names,
and raw path strings in the Chinese abstract, introduction, method, and
experiment protocol main text.

- [ ] **Step 2: Add Chinese implementation appendix**

Add the corresponding appendix section after `\appendix` and before the
existing generated-sweep appendix:

```latex
\section{Implementation and Reproducibility Details}
\label{app:implementation-reproducibility-zh}

本文主文以 algorithmic level 描述方法；為了 reproducibility，本附錄記錄
本地實驗使用的 implementation identifiers。Frozen slow planner 為
\texttt{HuggingFaceVLA/smolvla\_libero}；LeRobot policy package 為
\texttt{lerobot\_policy\_hfrvla}。已驗證的 cached dataset root 是
\texttt{checkpoints/HFRVLA\_libero\_v1\_merged\_reindexed}，包含 1,693
episodes、273,465 frames 與 40 個 LIBERO tasks。

Cached dataset 儲存 top-view RGB、wrist RGB、robot state、expert action、
selected base action、normalized chunk position、slow-planner task/phase
context，以及 DINOv3 wrist patch tokens。在本地 LeRobotDataset v3 schema 中，
這些欄位包含 \texttt{observation.images.image}、
\texttt{observation.images.image2}、\texttt{observation.state}、
\texttt{action}、\texttt{observation.extra.a\_base}、
\texttt{observation.extra.k\_idx\_norm}、
\texttt{observation.extra.z\_goal}、
\texttt{observation.extra.z\_phase} 與
\texttt{observation.extra.dino\_patches}。

Main generated-checkpoint experiments 使用 trained HFRVLA artifact
\texttt{hfrvla\_fwr\_chunk\_generated\_seq2\_b1024p3\_50k}；comparison
baseline 為 \texttt{smolvla\_libero}。所有 quantitative claims 由
\texttt{experiments/eval\_registry/eval\_results\_master.csv} 及其 source
manifest 追蹤。
```

### Task 4: Rewrite NYCU Thesis Methodology and Experiments

**Files:**
- Modify: `paper/targets/nycu_thesis/latex/Sections/4.Methodology.tex`
- Modify: `paper/targets/nycu_thesis/latex/Sections/5.Experiments.tex`
- Modify: `paper/targets/nycu_thesis/latex/Sections/appendix.tex`

- [ ] **Step 1: Convert Methodology feature table into conceptual table**

In `4.Methodology.tex`, replace the raw dataset-key table with a conceptual
table whose rows are:

```text
Third-person scene image
Wrist image
Robot proprioceptive state
Expert action
Frozen planner base action
Chunk position
Cached slow-planner context
Frozen DINOv3 wrist patch tokens
```

Do not show `observation.extra.*` keys in the methodology chapter. Mention that
exact dataset keys are listed in the appendix.

- [ ] **Step 2: Rewrite Methodology prose**

Rewrite `4.Methodology.tex` so it reads as a method chapter:

```text
The system combines a frozen LIBERO-adapted SmolVLA planner with a trainable
wrist-camera residual module. The slow planner supplies action chunks and
cached context; the fast module reads current wrist evidence, state, chunk
position, and the selected base action to predict a bounded 7D correction.
```

Keep the training target and merge equations, but avoid package paths,
checkpoint paths, and dataset keys in the chapter body.

- [ ] **Step 3: Rewrite Experiments protocol variables**

In `5.Experiments.tex`, replace code-style protocol names in the table body
with human-readable labels:

```text
Planned chunk length
Executed open-loop length
Replanning interval
```

Keep the mathematical symbol `K` and explain protocols in prose. Move registry
file paths and script names to the thesis appendix.

- [ ] **Step 4: Add thesis implementation appendix**

In `appendix.tex`, add:

```latex
\section{Implementation and reproducibility details}
\label{app:thesis-implementation-reproducibility}

本節集中列出主文刻意不展開的 implementation identifiers。Frozen slow
planner 使用 \texttt{HuggingFaceVLA/smolvla\_libero}。HFRVLA 在 LeRobot
框架下以 \texttt{lerobot\_policy\_hfrvla} package 實作。已驗證的
LeRobotDataset v3 root 為
\texttt{checkpoints/HFRVLA\_libero\_v1\_merged\_reindexed}，包含 1,693
episodes、273,465 frames 與 40 tasks。

本地 dataset schema 包含 \texttt{observation.images.image}、
\texttt{observation.images.image2}、\texttt{observation.state}、
\texttt{action}、\texttt{observation.extra.a\_base}、
\texttt{observation.extra.k\_idx\_norm}、
\texttt{observation.extra.z\_goal}、
\texttt{observation.extra.z\_phase} 與
\texttt{observation.extra.dino\_patches}。主要 generated-checkpoint 實驗使用
\texttt{hfrvla\_fwr\_chunk\_generated\_seq2\_b1024p3\_50k}；baseline 為
\texttt{smolvla\_libero}。所有 result claims 由
\texttt{experiments/eval\_registry/eval\_results\_master.csv} 和
\texttt{experiments/eval\_registry/sources.csv} 追蹤。
```

### Task 5: Identifier Leakage Checks

**Files:**
- Verify: `paper/src/main.tex`
- Verify: `paper/src/main_zh.tex`
- Verify: `paper/targets/nycu_thesis/latex/Sections/4.Methodology.tex`
- Verify: `paper/targets/nycu_thesis/latex/Sections/5.Experiments.tex`
- Verify: `paper/targets/nycu_thesis/latex/Sections/appendix.tex`

- [ ] **Step 1: Check short paper main text before appendix**

Run:

```bash
awk '/\\appendix/{exit} {print}' paper/src/main.tex | rg -n "HuggingFaceVLA|lerobot_policy|observation\\.extra|checkpoints/|hfrvla_fwr_chunk|smolvla_libero|policy\\.|vlm_model_name|expert_width_multiplier|num_vlm_layers|load_vlm_weights"
```

Expected: No matches.

- [ ] **Step 2: Check Chinese short paper main text before appendix**

Run:

```bash
awk '/\\appendix/{exit} {print}' paper/src/main_zh.tex | rg -n "HuggingFaceVLA|lerobot_policy|observation\\.extra|checkpoints/|hfrvla_fwr_chunk|smolvla_libero|policy\\.|vlm_model_name|expert_width_multiplier|num_vlm_layers|load_vlm_weights"
```

Expected: No matches.

- [ ] **Step 3: Check thesis Methodology and Experiments main chapters**

Run:

```bash
rg -n "HuggingFaceVLA|lerobot_policy|observation\\.extra|checkpoints/|hfrvla_fwr_chunk|smolvla_libero|policy\\.|vlm_model_name|expert_width_multiplier|num_vlm_layers|load_vlm_weights" \
  paper/targets/nycu_thesis/latex/Sections/4.Methodology.tex \
  paper/targets/nycu_thesis/latex/Sections/5.Experiments.tex
```

Expected: No matches.

- [ ] **Step 4: Confirm appendix preserves reproducibility identifiers**

Run:

```bash
rg -n -F \
  -e "HuggingFaceVLA/smolvla\\_libero" \
  -e "lerobot\\_policy\\_hfrvla" \
  -e "observation.extra.a\\_base" \
  -e "experiments/eval\\_registry" \
  -e "hfrvla\\_fwr\\_chunk\\_generated\\_seq2\\_b1024p3\\_50k" \
  -e "smolvla\\_libero" \
  paper/src/main.tex \
  paper/src/main_zh.tex \
  paper/targets/nycu_thesis/latex/Sections/appendix.tex
```

Expected: Matches appear only in appendix sections.

### Task 6: Build Verification

**Files:**
- Verify: `paper/build`
- Verify: `paper/targets/nycu_thesis/latex/build/main.pdf`

- [ ] **Step 1: Build bilingual short paper**

Run:

```bash
(cd paper && bash build_bilingual.sh)
```

Expected: English and Traditional Chinese paper builds finish without LaTeX
errors.

- [ ] **Step 2: Build NYCU thesis**

Run:

```bash
(cd paper/targets/nycu_thesis/latex && bash build.sh)
```

Expected: NYCU thesis build finishes without LaTeX errors and produces
`paper/targets/nycu_thesis/latex/build/main.pdf`.

- [ ] **Step 3: Review git diff**

Run:

```bash
git diff -- paper/src/main.tex paper/src/main_zh.tex paper/targets/nycu_thesis/latex/Sections/4.Methodology.tex paper/targets/nycu_thesis/latex/Sections/5.Experiments.tex paper/targets/nycu_thesis/latex/Sections/appendix.tex
```

Expected: Diff shows paragraph-level rewrites, no numerical result changes, and
implementation identifiers moved to appendix sections.

### Task 7: Commit Rewrite

**Files:**
- Modify: `paper/src/main.tex`
- Modify: `paper/src/main_zh.tex`
- Modify: `paper/targets/nycu_thesis/latex/Sections/4.Methodology.tex`
- Modify: `paper/targets/nycu_thesis/latex/Sections/5.Experiments.tex`
- Modify: `paper/targets/nycu_thesis/latex/Sections/appendix.tex`

- [ ] **Step 1: Stage only rewrite files**

Run:

```bash
git add \
  paper/src/main.tex \
  paper/src/main_zh.tex \
  paper/targets/nycu_thesis/latex/Sections/4.Methodology.tex \
  paper/targets/nycu_thesis/latex/Sections/5.Experiments.tex \
  paper/targets/nycu_thesis/latex/Sections/appendix.tex
```

Expected: Only those files are staged.

- [ ] **Step 2: Confirm staged files**

Run:

```bash
git diff --cached --name-only
```

Expected: The staged list contains only the five rewrite files.

- [ ] **Step 3: Commit rewrite**

Run:

```bash
git commit -m "Rewrite HFRVLA paper narrative"
```

Expected: Commit succeeds.
