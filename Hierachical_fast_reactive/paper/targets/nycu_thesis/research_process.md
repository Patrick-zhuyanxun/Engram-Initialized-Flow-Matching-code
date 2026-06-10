# NYCU Thesis Research Process - HFRVLA

Last updated: 2026-06-10

This file is the detailed research-process record for the NYCU master's thesis.
It follows the user's requested order, but each phase is rewritten for HFRVLA so
the thesis does not remain generic.

## 0. Problem To Solve

### Practical Problem

Modern vision-language-action policies can interpret language and visual
observations, but running a large VLA planner at every robot control tick is
expensive. Action chunking is a practical workaround: the slow planner predicts
multiple future actions at once, and the robot executes them over several
control steps. The unresolved problem is that a chunk is generated from an older
observation. By the time the robot executes the later actions, the wrist camera,
gripper pose, object contact, and local geometry may have changed.

The thesis problem is therefore:

> How can a robot keep the efficiency of frozen VLA action chunks while
> recovering some closed-loop reactivity at execution time?

### Research-Specific Problem

For this project, the slow planner is frozen `HuggingFaceVLA/smolvla_libero`.
The research question is not "can we train a better VLA from scratch?" It is:

> Can a small wrist-camera residual module correct the next action of a frozen
> SmolVLA chunk, especially when the execution step is far from the observation
> that generated the chunk?

### Why The Current Draft Felt Too Empty

The thesis must explicitly show the chain:

```text
VLA inference is expensive
-> action chunking reduces planner calls
-> chunk execution creates stale actions
-> stale actions are local, time-varying execution errors
-> wrist camera sees current gripper-relative evidence
-> bounded residual correction may repair the next action without changing the planner
```

If this chain is missing, the method appears arbitrary.

## 1. Research Idea

### Current Idea

HFRVLA wraps a frozen SmolVLA planner with a lightweight fast residual module.
The slow planner provides base action chunks and slow context. The fast path uses
current wrist-camera DINO patch features, robot state, base action, chunk index,
and cached slow-planner context to predict a 7D residual:

```text
a_final = a_base + alpha * clip(delta_a, -delta_max, delta_max)
```

### Important Design Commitments

| Commitment | Reason |
|---|---|
| Freeze SmolVLA | Isolates whether fast correction helps without slow-planner fine-tuning. |
| Use wrist DINO patches as new high-frequency visual evidence | Wrist view is gripper-relative and changes with contact/local geometry. |
| Keep residual clipped | Prevents the fast module from overriding the frozen planner with unbounded corrections. |
| Report success rate as the main metric | Matches the user's thesis requirement and LIBERO evaluation convention. |
| Separate synchronous and async experiments | Avoids mixing different runtime contracts into one claim. |

## 2. Research Background And Evidence

### Background Claims To Support

| Claim | Evidence Needed | Current Local Evidence |
|---|---|---|
| VLA inference motivates action chunking. | VLA/action-chunking literature. | To be verified in literature matrix. |
| Longer chunks can become stale. | Chunking, receding-horizon, and real-time VLA papers. | LIBERO sweeps show success changes with `execution_chunk_size` and matched `K`. |
| Wrist view can provide local manipulation evidence. | Wrist/egocentric manipulation literature and dense feature papers. | HFRVLA uses `observation.images.image2` DINO patches. |
| Residual magnitude must be calibrated. | Residual/control safety reasoning and alpha/clip sweep. | `n50_alpha_clip_spatial_50eps` and Figure 4 calibration. |
| Async planner delay is a different stress test. | Real-time/async robot inference literature. | `async_timestep_planner_delay_eval_sweep`. |

### Current Empirical Evidence

Source of truth:

```text
experiments/eval_registry/eval_results_master.csv
paper/thesis/generated/eval_registry_summary.md
```

Current registry status:

- Master rows: 267.
- Sweep groups: 13.
- Main generated Spatial evidence is split into:
  - `fwr_generated_plan50_exec_10x10_spatial`
  - `fwr_generated_matched_chunk_10x10_spatial`
  - `n50_alpha_clip_spatial_50eps`
  - `async_timestep_planner_delay_eval_sweep`

These should be written as single-seed LIBERO-Spatial evidence unless additional
multi-seed experiments are completed.

## 3. References

### Literature Collection Buckets

The thesis literature review should be maintained by buckets, not by a flat list
of citations:

| Bucket | Purpose | Priority sources |
|---|---|---|
| VLA and robot foundation models | Establish the slow-planner context. | RT-1, RT-2, OpenVLA, Octo, pi0, SmolVLA, Open X-Embodiment. |
| Action chunking and receding-horizon control | Explain why chunks exist and why they become stale. | ACT, Diffusion Policy, FAST, real-time VLA systems. |
| Correction heads and fast-slow policies | Position HFRVLA against closest methods. | A2C2, Reactive Diffusion Policy, Fast-in-Slow, DynamicVLA. |
| Wrist and egocentric manipulation | Justify wrist camera as local evidence. | Wrist-view manipulation, active vision, egocentric robot policies. |
| Dense visual features and DINO | Justify DINO patch tokens and visualization. | DINO, DINOv2, DINOv3, dense ViT descriptors, R3M/DINOBot. |
| Infrastructure and benchmarks | Define evaluation context. | LIBERO, LeRobot, SmolVLA documentation/papers. |

### Verification Rule

Do not add a reference because it "sounds right." For each new reference, record:

- Title.
- Venue or preprint status.
- DOI/arXiv/OpenReview/publisher URL.
- Why it is relevant to a thesis claim.
- Whether it was human-read, skimmed, or only metadata-verified.

## 4. Literature Comparison And Research Gap

### Draft Gap Statement

Existing VLA and robot foundation model work mainly improves the planner itself
through scale, data, or architecture. Real-time/action-chunking work improves
how chunks are generated, scheduled, or streamed. HFRVLA instead studies a
narrower gap: when a frozen VLA planner already produces action chunks, can a
small wrist-conditioned residual module repair the next executed action without
changing the slow planner?

### Comparison Matrix To Maintain

| Prior direction | What it solves | What remains open for this thesis |
|---|---|---|
| General VLA scaling | Better task/generalization behavior. | Does not isolate execution-time stale action correction for a frozen planner. |
| Action chunking | Reduces planner call frequency. | Later actions can be stale under changed local geometry. |
| Receding-horizon execution | Improves closed-loop behavior by replanning. | Requires frequent policy calls; less useful when slow VLA is expensive. |
| A2C2-style correction | Adds correction head to frozen chunks. | Need same-backbone comparison and wrist-centric specialization analysis. |
| Wrist-view policies | Use gripper-relative visual evidence. | Need to combine wrist evidence with frozen VLA chunk context. |
| DINO/dense features | Provide transferable patch-level visual features. | Need to show how patch tokens enter the residual head and whether they causally matter. |

## 5. Research Questions

### Primary RQ

Can a small wrist-camera residual module improve execution of frozen SmolVLA
action chunks without fine-tuning the slow planner?

### Secondary RQs

| ID | Question | Evidence |
|---|---|---|
| RQ1 | Under plan=50 synchronous execution sweeps, where does HFRVLA improve over SmolVLA? | `fwr_generated_plan50_exec_10x10_spatial`. |
| RQ2 | Under matched `planning=execution=replan=K`, where does correction help and where does staleness dominate? | `fwr_generated_matched_chunk_10x10_spatial`. |
| RQ3 | How sensitive is the residual module to `alpha` and `delta_max`? | `n50_alpha_clip_spatial_50eps` and Figure 4. |
| RQ4 | Does the fast path reduce degradation when the planner chunk arrives late? | `async_timestep_planner_delay_eval_sweep`. |
| RQ5 | What remains unsupported without ablations? | No-wrist, no-DINO, no-latent, no-clip, multi-seed, A2C2 same-backbone comparison. |

## 6. Research Framework

### Conceptual Framework

```text
Task language + third-person/wrist observation
          |
          v
Frozen SmolVLA slow planner
          |
          +--> base chunk a_base[0:H]
          +--> slow context z_goal, z_phase
          |
Current control tick
          |
          +--> current wrist DINO patches
          +--> robot state
          +--> chunk index k
          +--> selected base action a_base[k]
          |
          v
Fast wrist residual module
          |
          v
alpha * clip(delta_a)
          |
          v
executed action a_final
```

### Thesis Argument Framework

1. Slow planner provides global task interpretation and nominal action chunk.
2. Chunk execution can become stale as local geometry changes.
3. Wrist DINO patches provide current local visual evidence.
4. A small residual module predicts bounded correction.
5. Success-rate sweeps test whether the correction helps under controlled chunk protocols.
6. Limitations identify what has not yet been causally proven.

## 7. Methodology

### Data And Cache

Use the HFRVLA LeRobotDataset v3 contract, not raw LIBERO alone:

| Feature | Shape | Thesis role |
|---|---:|---|
| `observation.images.image` | `(256,256,3)` | Third-person source image for slow planner context. |
| `observation.images.image2` | `(256,256,3)` | Wrist image used for DINO patch extraction. |
| `observation.state` | `(8,)` | Robot proprioceptive context. |
| `action` | `(7,)` | Expert target action. |
| `observation.extra.a_base` | `(7,)` | Frozen SmolVLA base action. |
| `observation.extra.k_idx_norm` | `(1,)` | Normalized chunk index. |
| `observation.extra.z_goal` | `(960,)` | Slow-planner text/task context. |
| `observation.extra.z_phase` | `(480,)` | Slow-planner action/expert context. |
| `observation.extra.dino_patches` | `(196,384)` | Wrist DINO patch tokens. |

Verified dataset root:

```text
checkpoints/HFRVLA_libero_v1_merged_reindexed
```

### DINO Method Detail To Add

The thesis should explicitly describe:

- Which camera supplies DINO features: wrist camera, `observation.images.image2`.
- Patch grid: 196 tokens, corresponding to a 14 by 14 grid for 224/16-style ViT patches after preprocessing.
- Feature dimension: 384.
- Freezing: DINO is used as a frozen feature extractor.
- Why patches matter: they preserve local spatial evidence rather than collapsing the wrist image into one global vector.
- What the DINO visualization does and does not prove: it checks data flow and attention patterns, but causal visual attribution requires masking ablations.

### Training Objective

Current residual target:

```text
target_delta = action[t] - a_base[t]
```

Current deployment merge:

```text
a_final = a_base + alpha * clip(delta_a, -delta_max, delta_max)
```

Important limitation:

The current target is simultaneous. Long-chunk deployment may need chunk-age or
stale-target training, where `a_base` comes from an older planner observation.

### Evaluation Protocol

All results must state:

- `planning_chunk_size`.
- `execution_chunk_size`.
- `replan_interval_steps`.
- synchronous vs async-timestep inference.
- policy checkpoint.
- alpha and delta clip setting.
- number of episodes.
- suite, currently main text should emphasize LIBERO-Spatial.

## 8. Data Analysis

### Primary Metric

Use success rate:

```text
success_rate = successes / episodes
```

The thesis may include Wilson confidence intervals only when needed for visual
uncertainty, but the main claim should be written in success-rate terms.

### Main Analysis Slices

| Slice | Question |
|---|---|
| plan=50 execution/replan sweep | Does correction help when execution interval varies under a long planned chunk? |
| matched chunk sweep | Does correction help when planning, execution, and replan intervals are all `K`? |
| alpha x delta calibration | How much residual authority is stable? |
| async planner-delay sweep | Does the fast path help when planner output arrives late? |
| generated long-K sweeps | Do results support or contradict long-chunk behavior claims? |

## 9. Research Results

### Current Result Policy

Only registry-backed rows should enter thesis result tables. If a result is
pending, cached, derived, or historical, label it accordingly.

### Main Results To Maintain

| Result block | Source | Thesis status |
|---|---|---|
| Spatial plan=50 execution/replan sweep | `fwr_generated_plan50_exec_10x10_spatial` | Main synchronous evidence. |
| Spatial matched chunk sweep | `fwr_generated_matched_chunk_10x10_spatial` | Main synchronous evidence. |
| Residual calibration | `n50_alpha_clip_spatial_50eps` | Calibration/diagnostic evidence. |
| Async planner-delay stress test | `async_timestep_planner_delay_eval_sweep` | Separate async-timestep evidence. |
| No-clip control | no-resclip sweep groups | Pending or appendix only unless complete. |
| Multi-seed confidence | not complete | Future work. |

### Result-Writing Rule

Each result paragraph should answer three questions:

1. What exact protocol was used?
2. What success-rate pattern appears?
3. What claim is supported, and what claim is not supported?

## 10. Discussion

The discussion should not simply repeat results. It should explain:

- Why correction helps in some chunk regimes but not every-step replanning.
- Why residual clipping is necessary.
- Why wrist evidence is local and cannot solve global scene ambiguity alone.
- Why current single-seed simulation evidence is insufficient for strong final claims.
- Why delayed-target training is a plausible next method step.

## 11. Conclusion

The conclusion should be conservative:

HFRVLA shows that bounded wrist-conditioned residual correction can improve
frozen SmolVLA chunk execution in specific LIBERO-Spatial stale-action regimes,
while leaving multi-seed, causal vision ablations, same-backbone A2C2 comparison,
and real-robot validation for future work.

## 12. Introduction

Chapter 1 should be drafted in this order:

1. VLA policies are useful but expensive to run at high frequency.
2. Action chunking reduces planner calls.
3. Chunking creates stale action execution.
4. Stale action error is often local and time-varying.
5. Wrist camera can observe current gripper-relative geometry.
6. HFRVLA tests a bounded residual correction layer over frozen SmolVLA.
7. Contributions and claim boundaries.

## 13. Abstract

The abstract should mention:

- Frozen SmolVLA slow planner.
- Wrist-camera DINO residual module.
- Bounded merge.
- LIBERO-Spatial success-rate evidence.
- Synchronous sweeps vs async planner-delay sweep.
- Limitations.

Avoid claiming:

- real-robot validation,
- universal VLA improvement,
- solved long-horizon chunk execution,
- wrist-only full policy.

## 14. Keywords

Candidate keywords:

- Vision-Language-Action model
- Action chunking
- Robotic manipulation
- Wrist camera
- Residual correction
- DINO features
- LeRobot
- LIBERO

## 15. Title

Current working title:

> HFRVLA: Wrist-Camera Residual Correction for Frozen Vision-Language-Action Action Chunks

Chinese working title:

> HFRVLA：凍結視覺語言動作模型之動作片段腕部相機殘差修正

Title requirements:

- Mention frozen VLA or action chunks.
- Mention wrist/residual correction.
- Avoid overstating real-time or real-robot claims.

## 16. References Formatting

Current BibTeX source:

```text
paper/src/references.bib
```

NYCU thesis formatting may later require a different bibliography style from
the workshop draft. Keep metadata verified first; style conversion should happen
after citation integrity is stable.

## 17. Formatting And TOC Generation

The Overleaf NYCU thesis template page describes an NYCU thesis template and
notes support for English TOC and Chinese table of contents. The current
buildable NYCU source is maintained under
`paper/targets/nycu_thesis/latex/main.tex`; this process note tracks final-format
needs so template changes, administrative fields, and migrated thesis content
can be handled deliberately.

Formatting items to maintain:

- Title page and approval/signature page requirements.
- Chinese and English abstract order.
- Chinese and English table of contents policy.
- Figure/table list policy.
- Chapter numbering and appendix numbering.
- Reference style.
- PDF compile command and font assumptions.

## 18. Appendix

Appendices should preserve research process without polluting main claims:

| Appendix type | Content |
|---|---|
| Experiment registry | Full sweep tables and status labels. |
| Failed or retired designs | Gate/contact/GRU history, if useful for process record. |
| Pending ablations | No-clip, latent-context, DINO/wrist masking, multi-seed. |
| Implementation details | Dataset schema, cache construction, policy plugin contract. |
| Visualization notes | DINO patch grid and attention visualization caveats. |
