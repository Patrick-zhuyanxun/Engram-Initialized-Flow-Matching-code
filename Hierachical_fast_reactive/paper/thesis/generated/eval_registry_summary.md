# HFRVLA Thesis Experiment Registry Summary

- Master rows: `267`
- Sweep groups: `13`

## action_step8_alpha_clip_sweep
- Rows: 36
- Type: `alpha_clip_sweep`
- Policies: `hfrvla`
- Suites: `combined, libero_object, libero_spatial`
- Statuses: `derived, imported, ok`
- Best success row: `hfrvla` `libero_object` plan=50 exec=8 alpha=0.5 delta=0.2 -> 47/50 (94.0%)

## action_steps_eval_sweep
- Rows: 30
- Type: `execution_replan_sweep`
- Policies: `hfrvla, smolvla`
- Suites: `combined, libero_object, libero_spatial`
- Statuses: `derived, imported, ok`
- Best success row: `hfrvla` `libero_object` plan=50 exec=8 alpha=0.5 delta=not recorded -> 47/50 (94.0%)

## action_steps_eval_sweep_no_resclip
- Rows: 10
- Type: `execution_replan_sweep`
- Policies: `hfrvla`
- Suites: `libero_object, libero_spatial`
- Statuses: `pending`

## async_timestep_planner_delay_eval_sweep
- Rows: 10
- Type: `planner_delay_sweep`
- Policies: `hfrvla, hfrvla_disable_fast`
- Suites: `libero_spatial`
- Statuses: `ok`
- Best success row: `hfrvla` `libero_spatial` plan=50 exec=16 alpha=0.5 delta=not recorded -> 72/100 (72.0%)

## chunk_size_eval_sweep
- Rows: 36
- Type: `matched_planning_execution_sweep`
- Policies: `hfrvla, smolvla`
- Suites: `combined, libero_object, libero_spatial`
- Statuses: `derived, ok`
- Best success row: `hfrvla` `libero_object` plan=16 exec=16 alpha=0.5 delta=not recorded -> 48/50 (96.0%)

## chunk_size_eval_sweep_no_resclip
- Rows: 12
- Type: `matched_planning_execution_sweep`
- Policies: `hfrvla`
- Suites: `libero_object, libero_spatial`
- Statuses: `pending`

## fwr_action_steps_10x10
- Rows: 21
- Type: `execution_replan_sweep`
- Policies: `hfrvla`
- Suites: `combined, libero_object, libero_spatial`
- Statuses: `derived, ok`
- Best success row: `hfrvla` `libero_object` plan=50 exec=4 alpha=0.5 delta=0.2 -> 92/100 (92.0%)

## fwr_chunk_plan50_exec50_eval
- Rows: 3
- Type: `matched_planning_execution_sweep`
- Policies: `hfrvla`
- Suites: `combined, libero_object, libero_spatial`
- Statuses: `derived, ok`
- Best success row: `hfrvla` `libero_object` plan=50 exec=50 alpha=0.5 delta=0.2 -> 29/50 (58.0%)

## fwr_chunk_size_10x10
- Rows: 21
- Type: `matched_planning_execution_sweep`
- Policies: `hfrvla`
- Suites: `combined, libero_object, libero_spatial`
- Statuses: `derived, ok`
- Best success row: `hfrvla` `libero_object` plan=2 exec=2 alpha=0.5 delta=0.2 -> 90/100 (90.0%)

## fwr_generated_matched_chunk_10x10_spatial
- Rows: 14
- Type: `matched_planning_execution_sweep`
- Policies: `hfrvla, smolvla`
- Suites: `libero_spatial`
- Statuses: `ok`
- Best success row: `smolvla` `libero_spatial` plan=1 exec=1 alpha=not recorded delta=not recorded -> 77/100 (77.0%)

## fwr_generated_plan50_exec_10x10_spatial
- Rows: 14
- Type: `execution_replan_sweep`
- Policies: `hfrvla, smolvla`
- Suites: `libero_spatial`
- Statuses: `ok`
- Best success row: `hfrvla` `libero_spatial` plan=50 exec=4 alpha=0.5 delta=0.2 -> 80/100 (80.0%)

## hfrvla_lrwd_alpha075_clip02_eval
- Rows: 24
- Type: `learning_rate_weight_decay_sweep`
- Policies: `hfrvla`
- Suites: `combined, libero_object, libero_spatial`
- Statuses: `cached, derived`
- Best success row: `hfrvla` `libero_object` plan=50 exec=8 alpha=0.75 delta=0.2 -> 49/50 (98.0%)

## n50_alpha_clip_spatial_50eps
- Rows: 36
- Type: `alpha_clip_sweep`
- Policies: `hfrvla`
- Suites: `libero_spatial`
- Statuses: `cached, ok`
- Best success row: `hfrvla` `libero_spatial` plan=50 exec=50 alpha=0.5 delta=0.2 -> 28/50 (56.0%)
