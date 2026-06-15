# NYCU Thesis Source Map

Last updated: 2026-06-10

This file maps thesis chapters and claims to local artifacts. Use it before
editing `paper/targets/nycu_thesis/latex/main.tex`.

## Chapter-To-Source Map

| Thesis area | Primary sources | Notes |
|---|---|---|
| Chapter 1 Introduction | `paper/targets/nycu_thesis/research_process.md`, `paper/notes/contribution_statement.md`, `paper/targets/nycu_thesis/latex/Sections/1.Introduction.tex` | Start with stale action chunk problem before method. |
| Chapter 2 Related Work | `paper/notes/literature_review.md`, `paper/src/references.bib`, `paper/targets/nycu_thesis/latex/Sections/2.Relatedwork.tex` | Expand as comparison and research gap, not a citation list. |
| Chapter 3 Framework and Method | `paper/notes/methodology_blueprint.md`, `paper/notes/implementation_spec.md`, `paper/targets/nycu_thesis/latex/Sections/3.Framework.tex`, `paper/targets/nycu_thesis/latex/Sections/4.Methodology.tex` | Include frozen planner, DINO wrist features, residual merge, and training target. |
| Chapter 4 Implementation | `AGENTS.md`, `docs/training.md`, policy plugin files | Record LeRobot plugin contract and dataset creation process. |
| Chapter 5 Experiments | `experiments/eval_registry/sources.csv`, `scripts/build_eval_results_master.py` | Define registry, protocols, and synchronous vs async separation. |
| Chapter 6 Results | `experiments/eval_registry/eval_results_master.csv`, `paper/targets/nycu_thesis/latex/Sections/6.ResultsDiscussion.tex`, `paper/thesis/generated/*.tex` | Success-rate-first result writing. |
| Chapter 7 Discussion | `paper/notes/reviewer_defense.md`, `paper/targets/nycu_thesis/research_process.md`, `paper/targets/nycu_thesis/latex/Sections/6.ResultsDiscussion.tex` | Limitations, failure modes, and unsupported claims. |
| Appendices | `paper/targets/nycu_thesis/latex/Sections/appendix.tex`, `paper/thesis/generated/`, `paper/notes/*protocol*.md` | Full registry tables, protocol details, pending experiments. |

## Claim-To-Artifact Map

| Claim | Required artifact |
|---|---|
| SmolVLA is frozen in HFRVLA | `policy/lerobot_policy_hfrvla/src/lerobot_policy_hfrvla/modeling_hfrvla.py`, config, training notes. |
| HFRVLA uses wrist DINO patches | dataset contract in `AGENTS.md`, method notes, recorder/cache scripts. |
| Fast path predicts 7D residual | policy fast module and methodology notes. |
| Merge uses `a_base + alpha * clip(delta_a)` | method notes and evaluation config. |
| Main evidence is LIBERO-Spatial | eval registry groups and result tables. |
| Planner-delay experiment is async-timestep only | `paper/notes/async_timestep_planner_delay_protocol.md` and registry group. |
| Residual scale must be calibrated | `n50_alpha_clip_spatial_50eps` registry rows and Figure 4. |
| Multi-seed and real robot are future work | red appendix / future-work list. |

## Figure Source Map

| Figure purpose | Current asset | Thesis requirement |
|---|---|---|
| Problem schematic | `paper/src/figures/fig1_problem_schematic.*` | Chapter 1 figure. Must show a multi-row Gantt chart where each SmolVLA chunk has a planning row and an execution row; chunk B/C planning starts during the tail of the active execution chunk, becomes ready at the switch, and execution chunks remain back-to-back with no idle gap. It must also show stale/open-loop feedback during long execution and define H/e/K/d in human terms. |
| Solution schematic | `paper/src/figures/fig1_solution_schematic.*` | Chapter 3 figure. Must show HFRVLA architecture: frozen low-frequency SmolVLA consuming language, top RGB, wrist RGB, and state; frozen DINOv3 wrist features; trainable fast correction module; and clipped/scaled merge with selected base action. |
| Plan=50 execution sweep | `paper/src/figures/fig2_plan50_execution_sweep.*` | Success rate only. |
| Matched chunk sweep | `paper/src/figures/fig3_matched_chunk_sweep.*` | Success rate only. |
| Residual calibration | `paper/src/figures/fig4_residual_calibration.png` | Use PNG in LaTeX to avoid PDF color artifact. |
| Async planner delay | `paper/src/figures/fig5_async_planner_delay.*` | Blue HFRVLA vs gray disable-fast. |
| DINO patch visualization | `paper/src/figures/fig_dino_patch_grid_preview.png`, `fig_dino_cross_attention_example.png` | Explain data flow, not causal proof. |

## Verification Commands

```bash
python3 scripts/build_eval_results_master.py --check
cd paper/targets/nycu_thesis/latex && bash build.sh
```

Use the bilingual workshop build only when synchronizing `paper/src`:

```bash
cd paper && bash build_bilingual.sh
```
